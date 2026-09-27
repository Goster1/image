"""Synthetic round trip of the independent line pipeline (plumb-line -> VPs -> f, and the joint fit).

Truth: the joint line estimate (JRE main: k1k2, pp free, tied cart frames) with the principal point (= distortion
centre) shifted by (+15, +15) px.  Geometry: the REAL edges.  For every accepted real edge the points are
undistorted with the TRUE lens; VP-group edges get the line through the true vanishing point (K_T d_g) and
their undistorted centroid, rotated by a random member direction error psi ~ N(0, sig_psi); other edges keep
their own best line.  Points are projected onto these lines and distorted with the true lens.  Noise along the
edge normal, matched per edge to the real residuals of the main plumb fit:
    random bow  b_e ~ N(0, sig_bow)  (sig_bow = rms of the real per-edge residual bows)
    correlated noise (Gaussian-smoothed, corr. length ~3 points) + white noise, total = real per-edge noise.
Each trial reruns: plumb k1k2 (+ free centre), VPs, f from cart-X _|_ cart-vertical with pp = distortion
centre (self-consistent), and the joint fit.  Recovery errors are compared with the claimed (bootstrap)
uncertainties of 40_lines_indep_b_vp.py / 40_lines_indep_c_joint.py (read from their cache files if present).
Writes work/cache/40_lines_indep_synth.json, results/40_lines_indep_synth.png
Usage: python3 40_lines_indep_d_synth.py [NTRIALS]
"""
import importlib
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from common import CACHE, RESULTS, K_from, load_json  # noqa: E402
from evaltools import grid, mapping_displacement, region_defs  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(4004)
T0 = time.time()
SHIFT = np.array([15.0, 15.0])

edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
for e in acc:
    e["vgroup"] = "CZ" if e["group"] in ("310Z", "80Z") else e["group"]

# ------------------------------------------------------------------ estimate on the real data (truth basis)
pl = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
es_all = L.EdgeSet(acc)
st_real = es_all.per_edge_stats(pl["r"])
SIG_BOW = float(np.sqrt(np.mean([s["bow"] ** 2 for s in st_real])))
vps = L.vp_estimates(acc, pl["dist"], ["310X", "80X", "CZ", "WV"])
f_init = L.f_orthogonal(vps, [("310X", "CZ"), ("80X", "CZ")], pl["dist"].c)
init = dict(f=f_init, pp=tuple(pl["dist"].c), a=list(pl["dist"].a), vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in vps.items()})
se_cl = np.sqrt(np.diag(pl["cov_cluster"]))[2:]
se_c = np.sqrt(np.diag(pl["cov_classic"]))[2:]
SIG_PT = pl["rms"] * float(np.median(se_cl / se_c))
jm = L.JointModel(acc, init=init, sig_pt=SIG_PT, sig_psi_deg=0.155)
jr = jm.fit(loss="linear")
f_est, pp_est, dist_est, dirs = jm.unpack(jr.x)
print(f"real-data joint estimate: f={f_est:.1f} pp={np.round(pp_est, 1)} a={np.round(dist_est.a, 5)}; sig_bow {SIG_BOW:.3f} px")

# GLS weights for the VP-only f (same estimator as 40_lines_indep_b_vp.py): VP covariances of the real data
rates_real = L.edge_angle_rates(acc, pl["dist"], f_init, pl["dist"].c)
COVS = {g: L.vp_cov_mc(acc, pl["dist"], g, vps[g], rates_real, np.radians(0.155), n=60, rng_=rng) for g in ("310X", "80X", "CZ")}
f_real_gls = L.gls_f(vps, COVS, [("310X", "CZ"), ("80X", "CZ")], pl["dist"].c, f0=f_init)
print(f"real-data VP f (GLS, pp = distortion centre) {f_real_gls:.1f} (unweighted {f_init:.1f})")

F_T = f_est
PP_T = pp_est + SHIFT
DIST_T = L.Dist("poly", dist_est.a.copy(), None, PP_T)
K_T = np.array([[F_T, 0, PP_T[0]], [0, F_T, PP_T[1]], [0, 0, 1.0]])
SIG_PSI = np.radians(0.155)
rates = L.edge_angle_rates(acc, DIST_T, F_T, PP_T)
gdir = {"310X": dirs["310X"], "80X": dirs["80X"], "310Z": dirs["CZ"], "80Z": dirs["CZ"], "WV": dirs["WV"]}
print(f"truth: f={F_T:.1f} pp={np.round(PP_T, 1)} a={np.round(DIST_T.a, 5)} -> OpenCV k={np.round(DIST_T.opencv(F_T), 4)}")


def synth_edges(rng_):
    out = []
    for k, e in enumerate(acc):
        P = e["points"]
        U, _, _ = DIST_T.undistort(P)
        c = U.mean(0)
        g = e["group"]
        if g in gdir:
            v = K_T @ gdir[g]
            dvec = v[:2] - c * v[2]  # direction from the centroid towards the VP (homogeneous safe)
            dvec = dvec / np.linalg.norm(dvec)
            dth = rng_.normal(0, SIG_PSI) / max(rates[e["id"]]["rate"], 1e-9)
            R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
            dvec = R @ dvec
        else:
            _, _, vt = np.linalg.svd(U - c)
            dvec = vt[0]
        n = np.array([-dvec[1], dvec[0]])
        Up = U - ((U - c) @ n)[:, None] * n[None]
        Pi = DIST_T.distort_px(Up)
        # noise along the local normal of the distorted edge
        tang = np.gradient(Pi, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(Pi, axis=0).T))]
        s = 2 * s / s[-1] - 1
        sig = st_real[k]["noise"]
        bow = rng_.normal(0, SIG_BOW) * (1 - s * s)
        corr = gaussian_filter1d(rng_.normal(0, 1, len(Pi)), 3.0)
        corr *= sig / np.sqrt(2) / max(corr.std(), 1e-9)
        white = rng_.normal(0, sig / np.sqrt(2), len(Pi))
        out.append(dict(e, points=Pi + (bow + corr + white)[:, None] * nrm))
    return out


uv, _ = grid(40)
REG = region_defs()


def mapping_err(f, pp, dist):
    """rotation-compensated displacement (px) of an estimate vs the truth, per region (rms)."""
    K1 = K_from(F_T, F_T, *PP_T)
    d1 = DIST_T.opencv(F_T)
    K2 = K_from(f, f, *pp)
    d2 = dist.opencv(f)
    disp = mapping_displacement(K1, d1, K2, d2, uv=uv, compensate=True)
    m = np.hypot(disp[:, 0], disp[:, 1])
    return {k: float(np.sqrt(np.mean(m[fn(uv) ] ** 2))) for k, fn in REG.items()}


rows = []
for t in range(NT):
    se = synth_edges(rng)
    row = {}
    # plumb, centre free and fixed
    p1 = L.fit_plumb(L.EdgeSet(se), L.PlumbModel("poly", 2, centre_free=True), x0=pl["x"])
    p0 = L.fit_plumb(L.EdgeSet(se), L.PlumbModel("poly", 2))
    row["plumb_c"] = list(p1["x"])
    row["plumb_fixed"] = list(p0["x"])
    # VPs + f (pp = distortion centre)
    v1 = L.vp_estimates(se, p1["dist"], ["310X", "80X", "CZ", "WV"])
    row["vp_f_ppDist"] = L.gls_f(v1, COVS, [("310X", "CZ"), ("80X", "CZ")], p1["dist"].c)
    row["vp_f_ppDist_unweighted"] = L.f_orthogonal(v1, [("310X", "CZ"), ("80X", "CZ")], p1["dist"].c)
    v0 = L.vp_estimates(se, p0["dist"], ["310X", "80X", "CZ", "WV"])
    row["vp_f_ppC0"] = L.gls_f(v0, COVS, [("310X", "CZ"), ("80X", "CZ")], L.C0)
    row["map_vp_ppDist"] = mapping_err(row["vp_f_ppDist"], p1["dist"].c, p1["dist"])
    # joint
    ini = dict(f=row["vp_f_ppDist"], pp=tuple(p1["dist"].c), a=list(p1["dist"].a), vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in v1.items()})
    m = L.JointModel(se, init=ini, sig_pt=SIG_PT, sig_psi_deg=0.155)
    r = m.fit(loss="linear")
    f, pp, dist, _ = m.unpack(r.x)
    row["joint"] = [float(f), float(pp[0]), float(pp[1])] + list(dist.a)
    row["map_joint"] = mapping_err(f, pp, dist)
    rows.append(row)
    print(f"trial {t:2d}: plumb_c centre ({p1['x'][0]:.1f},{p1['x'][1]:.1f}) a1 {p1['x'][2]:.4f} | f_vp(ppDist) {row['vp_f_ppDist']:.1f} "
          f"f_vp(ppC0,fixed-centre plumb) {row['vp_f_ppC0']:.1f} | joint f {f:.1f} pp ({pp[0]:.1f},{pp[1]:.1f}) | map joint cart_band "
          f"{row['map_joint']['cart_band']:.2f} corners {row['map_joint']['corners']:.2f} ({time.time() - T0:.0f}s)")

# ------------------------------------------------------------------ summary vs truth and vs claimed uncertainty
P1 = np.array([r["plumb_c"] for r in rows])
J = np.array([r["joint"] for r in rows])
fv = np.array([r["vp_f_ppDist"] for r in rows])
fv0 = np.array([r["vp_f_ppC0"] for r in rows])
truth = dict(f=F_T, cx=PP_T[0], cy=PP_T[1], a1=DIST_T.a[0], a2=DIST_T.a[1])


def st(x, t_):
    x = np.asarray(x, float)
    return dict(mean=float(np.mean(x)), bias=float(np.mean(x) - t_), sd=float(np.std(x, ddof=1)), rmse=float(np.sqrt(np.mean((x - t_) ** 2))))


summ = dict(
    plumb_centre_x=st(P1[:, 0], PP_T[0]), plumb_centre_y=st(P1[:, 1], PP_T[1]), plumb_a1=st(P1[:, 2], DIST_T.a[0]), plumb_a2=st(P1[:, 3], DIST_T.a[1]),
    vp_f_ppDist=st(fv, F_T), vp_f_ppDist_unweighted=st([r["vp_f_ppDist_unweighted"] for r in rows], F_T),
    vp_f_ppC0_with_fixed_centre_plumb=st(fv0, F_T),
    joint_f=st(J[:, 0], F_T), joint_cx=st(J[:, 1], PP_T[0]), joint_cy=st(J[:, 2], PP_T[1]), joint_a1=st(J[:, 3], DIST_T.a[0]), joint_a2=st(J[:, 4], DIST_T.a[1]),
    map_joint={k: float(np.sqrt(np.mean([r["map_joint"][k] ** 2 for r in rows]))) for k in REG},
    map_vp_ppDist={k: float(np.sqrt(np.mean([r["map_vp_ppDist"][k] ** 2 for r in rows]))) for k in REG},
)
claimed = {}
fn_b = f"{CACHE}/40_lines_indep_vp.json"
fn_c = f"{CACHE}/40_lines_indep_joint.json"
fn_a = f"{CACHE}/40_lines_indep_plumb.json"
if os.path.exists(fn_a):
    a = load_json(fn_a)["models"]["k1k2_c"]
    claimed["plumb_k1k2_c_boot_sd"] = dict(zip(a["names"], a.get("boot_sd", [np.nan] * 4)))
if os.path.exists(fn_b):
    b = load_json(fn_b)["bootstrap"]["k1k2_c"]["solutions"]
    s = b.get("S1_cartX_cartZ|ppDist")
    if s:
        claimed["vp_f_ppDist_boot_sd"] = s["sd"][0]
        claimed["vp_f_ppDist_boot_sd_robust"] = s["sd_robust"][0]
if os.path.exists(fn_c):
    c = load_json(fn_c)["variants"]["JRE_k1k2_ppfree"]
    if "boot" in c:
        claimed["joint_boot_sd"] = dict(zip(c["boot"]["columns"], c["boot"]["sd"]))
        claimed["joint_boot_sd_robust"] = dict(zip(c["boot"]["columns"], c["boot"]["sd_robust"]))
    if "mapping_uncertainty" in c:
        claimed["joint_mapping_uncertainty"] = {k: v["rms_median_px"] for k, v in c["mapping_uncertainty"].items() if k != "whole_image"}
print("\nsummary (truth f={:.1f}, pp=({:.1f},{:.1f}))".format(F_T, *PP_T))
for k, v in summ.items():
    print(f"  {k:36s} {v}")
print("claimed:", claimed)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].hist(fv - F_T, bins=15, alpha=0.6, label=f"VP, pp=dist. centre: bias {np.mean(fv) - F_T:+.1f}, sd {np.std(fv, ddof=1):.1f}")
ax[0].hist(J[:, 0] - F_T, bins=15, alpha=0.6, label=f"joint: bias {np.mean(J[:, 0]) - F_T:+.1f}, sd {np.std(J[:, 0], ddof=1):.1f}")
ax[0].hist(fv0 - F_T, bins=15, alpha=0.4, label=f"VP, pp=image centre (wrong by 21 px): bias {np.mean(fv0) - F_T:+.1f}")
if "vp_f_ppDist_boot_sd" in claimed:
    ax[0].axvspan(-claimed["vp_f_ppDist_boot_sd"], claimed["vp_f_ppDist_boot_sd"], color="gray", alpha=0.2, label="claimed 1-sigma (VP bootstrap)")
ax[0].set_xlabel("f error [px]")
ax[0].legend(fontsize=7)
ax[0].set_title(f"synthetic round trip, {NT} trials")
ax[1].plot(P1[:, 0] - PP_T[0], P1[:, 1] - PP_T[1], "o", ms=4, label="plumb k1k2 free centre")
ax[1].plot(J[:, 1] - PP_T[0], J[:, 2] - PP_T[1], "s", ms=4, label="joint pp")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].axvline(0, color="k", lw=0.5)
ax[1].set_xlabel("x error [px]")
ax[1].set_ylabel("y error [px]")
ax[1].legend(fontsize=7)
ax[1].set_title("distortion centre / principal point error")
ax[1].set_aspect("equal")
ks = list(REG)
ax[2].bar(np.arange(len(ks)) - 0.2, [summ["map_joint"][k] for k in ks], 0.4, label="joint: rms mapping error vs truth")
if "joint_mapping_uncertainty" in claimed:
    ax[2].bar(np.arange(len(ks)) + 0.2, [claimed["joint_mapping_uncertainty"].get(k, np.nan) for k in ks], 0.4, label="claimed (real-data bootstrap)")
ax[2].set_xticks(range(len(ks)))
ax[2].set_xticklabels(ks)
ax[2].set_ylabel("px (rotation-compensated)")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_synth.png", dpi=110)
plt.close()

L.save(L.jsonable(dict(truth=dict(f=F_T, pp=PP_T.tolist(), a=DIST_T.a.tolist(), opencv=DIST_T.opencv(F_T).tolist(), shift=SHIFT.tolist()),
                       noise_model=dict(sig_bow_px=SIG_BOW, sig_psi_deg=0.155, per_edge_noise="real per-edge residual noise, half white half correlated (sigma 3 pts)"),
                       n_trials=NT, summary=summ, claimed=claimed, trials=rows)), "40_lines_indep_synth.json")
print(f"done in {time.time() - T0:.0f}s")
