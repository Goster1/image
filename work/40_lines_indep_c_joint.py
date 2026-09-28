"""(3) JOINT line self-calibration (independent formulation): distortion + f + principal point from lines only.

Model: OpenCV-compatible (distortion centre = principal point, fx = fy, zero skew), radial k1,k2 in pixel units.
Directions: a common vertical z for both carts (posts of cart 310 and cart 80) and the X axis of each cart in
the plane _|_ z ('tied' frames = carts upright on one floor); scene verticals with their own free direction.

Cost (vp_mode 're', main):   sum_points (d_i / sig_pt)^2  +  sum_VPedges (psi_e / sqrt(sig_psi^2 + noise_e^2))^2
  d_i   : distance of edge point i to the edge's own (free) line, measured in the DISTORTED image
  psi_e : angle between the edge's interpretation plane (camera centre + undistorted line) and the direction of
          its group -> "VP consistency" with the per-member direction error treated as a random effect
          (sig_psi from the redundancy of the cart-X groups, re-estimated from the joint fit)
  sig_pt: straightness weight; the within-edge correlation (bows, pixel-locking sawtooth) makes the point
          residuals far from independent -> sig_pt = rms * inflation, inflation = cluster-sandwich / classic SE
          ratio of the plumb-line fit (sensitivity: inflation 1 and 2x)
Variant 'pt' : every point of a VP edge is measured against the line through the VP (classic formulation).
Uncertainty: stratified cluster bootstrap (as in 40_lines_indep_b_vp.py): rich groups resampled, small groups
perturbed by the random direction error; everything re-estimated in every replicate.  + focal-length profile.
Writes work/cache/40_lines_indep_joint.json, results/40_lines_indep_joint*.png
Usage: python3 40_lines_indep_c_joint.py [NBOOT]
"""
import importlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import K_from  # noqa: E402
from evaltools import grid, mapping_displacement, mapping_stats  # noqa: E402

from common import RESULTS  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(4003)
T0 = time.time()

edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
for e in acc:
    e["vgroup"] = "CZ" if e["group"] in ("310Z", "80Z") else e["group"]
print("edge files reviewed:", L.file_status(), "| rejected:", [e["id"] for e, a in zip(edges, act) if not a])

# ------------------------------------------------------------------ initialisation from plumb + VPs
pl = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
d0 = pl["dist"]
vps = L.vp_estimates(acc, d0, ["310X", "80X", "CZ", "WV"])
f_init = L.f_orthogonal(vps, [("310X", "CZ"), ("80X", "CZ")], d0.c)
init = dict(f=f_init, pp=tuple(d0.c), a=list(d0.a), vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in vps.items()})
se_cl = np.sqrt(np.diag(pl["cov_cluster"]))[2:]
se_c = np.sqrt(np.diag(pl["cov_classic"]))[2:]
INFL = float(np.median(se_cl / se_c))
SIG_PT = pl["rms"] * INFL
print(f"init: plumb k1k2_c centre {np.round(d0.c, 1)}, f(S1) {f_init:.1f}; straightness inflation {INFL:.2f} -> sig_pt {SIG_PT:.2f} px")

# random direction error: from the cart-X groups (psi of each edge vs its group direction), iterate with the fit
rates = L.edge_angle_rates(acc, d0, f_init, d0.c)


def sig_psi_from_fit(m, x):
    f, pp, dist, dirs = m.unpack(x)
    r = m.residuals(x)
    npt = len(m.es.P)
    psi = r[npt:] * np.sqrt(m.sig_psi ** 2 + m.noise_psi ** 2)
    rich = np.array([m.gr[k] in ("310X", "80X") for k in m.gidx])
    n = rich.sum()
    ex = (np.sum(psi[rich] ** 2) - np.sum(m.noise_psi[rich] ** 2)) / max(n - 2 - 1, 1)  # 2 X directions + share of f
    return float(np.sqrt(max(ex, 1e-12))), psi


VARIANTS = {
    "JRE_k1k2_ppfree": dict(),
    "JRE_k1k2_ppC0": dict(pp_free=False),
    "JRE_div2_ppfree": dict(kind="div", a=[-0.17, -0.03]),
    "JRE_k1k2k3_ppfree": dict(nrad=3, a=list(d0.a) + [0.0]),
    "JRE_k1k2_framesfree": dict(frames="free"),
    "JRE_k1k2_wvtied": dict(wv="tied"),
    "JRE_k1k2_sigpt_noinfl": dict(sig_pt=pl["rms"]),
    "JRE_k1k2_sigpt_2xinfl": dict(sig_pt=2 * SIG_PT),
    "JRE_k1k2_noWV": dict(wv=None),
    "JPT_k1k2_ppfree": dict(vp_mode="pt"),
    "JPT_k1k2_ppC0": dict(vp_mode="pt", pp_free=False),
    "JPT_k1k2_framesfree": dict(vp_mode="pt", frames="free"),
}


def make_model(edge_list, spec, sig_psi_deg, init_=None):
    ini = dict(init if init_ is None else init_)
    if "a" in spec:
        ini["a"] = spec["a"]
    if spec.get("pp_free", True) is False:
        ini["pp"] = tuple(L.C0)
    kw = dict(kind=spec.get("kind", "poly"), nrad=spec.get("nrad", 2), pp_free=spec.get("pp_free", True), frames=spec.get("frames", "tied"),
              wv=spec.get("wv", "free"), vp_mode=spec.get("vp_mode", "re"), sig_pt=spec.get("sig_pt", SIG_PT), sig_psi_deg=sig_psi_deg, init=ini)
    return L.JointModel(edge_list, **kw)


def summarize(m, x):
    f, pp, dist, dirs = m.unpack(x)
    out = dict(f=float(f), pp=[float(pp[0]), float(pp[1])], coeffs=dist.a.tolist(), kind=dist.kind, fold_margin_px=dist.fold_margin())
    if dist.kind == "poly":
        out["opencv_dist"] = dist.opencv(f).tolist()
        out["invariants"] = dist.invariants()
    r = m.residuals(x)
    npt = len(m.es.P)
    if m.vp_mode == "re":
        out["rms_straightness_px"] = float(np.sqrt(np.mean((r[:npt] * m.sig_pt) ** 2)))
        psi = r[npt:] * np.sqrt(m.sig_psi ** 2 + m.noise_psi ** 2)
        out["psi_rms_deg"] = float(np.degrees(np.sqrt(np.mean(psi ** 2))))
        out["psi_deg"] = {m.edges[k]["id"]: float(np.degrees(p)) for k, p in zip(m.gidx, psi)}
    else:
        out["rms_px"] = float(np.sqrt(np.mean(r ** 2)))
    if "CZ" in dirs and "WV" in dirs:
        out["angle_CZ_WV_deg"] = float(np.degrees(np.arccos(min(1.0, abs(dirs["CZ"] @ dirs["WV"])))))
    if "310X" in dirs and "80X" in dirs:
        out["angle_310X_80X_deg"] = float(np.degrees(np.arccos(min(1.0, abs(dirs["310X"] @ dirs["80X"])))))
    # tilt of the camera: angle between the optical axis and the vertical
    z = dirs.get("CZ", dirs.get("310Z"))
    out["camera_tilt_from_vertical_deg"] = float(np.degrees(np.arccos(abs(z[2]))))
    return out


fits = {}
SIG_PSI_DEG = 0.157
for nm, spec in VARIANTS.items():
    t = time.time()
    sp = SIG_PSI_DEG
    m = make_model(acc, spec, sp)
    res = m.fit(loss="linear")
    if m.vp_mode == "re":  # one variance-component update of sig_psi
        s_new, _ = sig_psi_from_fit(m, res.x)
        sp = float(np.degrees(s_new))
        m = make_model(acc, spec, sp)
        res = m.fit(x0=res.x, loss="linear")
    s = summarize(m, res.x)
    s["sig_psi_deg"] = sp
    s["sig_pt_px"] = m.sig_pt
    s["x"] = res.x.tolist()
    s["names"] = m.names
    s["cost"] = float(res.cost)
    # formal covariance (Gauss-Newton, residuals already in sigma units) + cluster sandwich
    J = res.jac
    try:
        C = np.linalg.inv(J.T @ J)
        s["se_formal"] = dict(zip(m.names[: m.nin], np.sqrt(np.diag(C))[: m.nin].tolist()))
    except np.linalg.LinAlgError:
        pass
    fits[nm] = dict(model=m, res=res, summary=s, spec=spec)
    print(f"{nm:24s} ({time.time() - t:4.1f}s) f={s['f']:7.1f} pp=({s['pp'][0]:6.1f},{s['pp'][1]:6.1f}) coeffs={np.round(s['coeffs'], 5)} "
          f"sig_psi={sp:.3f}deg " + (f"psi_rms={s['psi_rms_deg']:.3f} rms_str={s['rms_straightness_px']:.3f}" if "psi_rms_deg" in s else f"rms={s['rms_px']:.3f}")
          + (f" k={np.round(s['opencv_dist'], 4)[[0, 1, 4]]}" if "opencv_dist" in s else "") + f" CZ-WV {s.get('angle_CZ_WV_deg', np.nan):.2f}deg"
          + (f"  se {np.round(list(s['se_formal'].values())[:3], 1)}" if "se_formal" in s else ""))

MAIN = "JRE_k1k2_ppfree"
main_m, main_res = fits[MAIN]["model"], fits[MAIN]["res"]
SIG_PSI_MAIN = fits[MAIN]["summary"]["sig_psi_deg"]

# ------------------------------------------------------------------ focal-length profile (main variant)
print("\nf profile (main variant, other parameters re-optimised)")
fgrid = np.arange(1300, 1701, 25.0)
prof = []
for fv in fgrid:
    m = make_model(acc, VARIANTS[MAIN], SIG_PSI_MAIN)
    x0 = main_res.x.copy()
    x0[0] = fv
    fixed_f = fv

    def fun(z, m=m, fixed_f=fixed_f):
        return m.residuals(np.r_[fixed_f, z])

    from scipy.optimize import least_squares

    r = least_squares(fun, x0[1:], x_scale=m.x_scale()[1:], method="trf", diff_step=1e-6, max_nfev=200, xtol=1e-12, ftol=1e-12)
    pp = r.x[:2]
    prof.append([fv, 2 * r.cost, pp[0], pp[1]])
prof = np.array(prof)
prof[:, 1] -= prof[:, 1].min()
print(np.round(prof, 1))


# ------------------------------------------------------------------ bootstrap
def boot_run(name, nb):
    spec = VARIANTS[name]
    base = fits[name]
    out = []
    for b in range(nb):
        sub = L.stratified_resample(acc, rng)
        sub = L.perturb_directions(sub, rates, np.radians(SIG_PSI_MAIN), rng, groups={"CZ", "WV"})
        try:
            m = make_model(sub, spec, base["summary"]["sig_psi_deg"])
            r = m.fit(x0=base["res"].x, loss="linear", max_nfev=150)
            f, pp, dist, dirs = m.unpack(r.x)
            row = [f, pp[0], pp[1]] + list(dist.a) + [0.0] * (3 - len(dist.a))
            out.append(row)
        except Exception as ex:  # noqa
            print("   replicate failed", ex)
        if b % 25 == 0:
            print(f"   {name} {b}/{nb} ({time.time() - T0:.0f}s)")
    return np.array(out)


NB_OTHER = max(NB // 2, min(60, NB))
boot = {}
for name, nb in [(MAIN, NB), ("JRE_k1k2_ppC0", NB_OTHER), ("JPT_k1k2_ppfree", NB_OTHER), ("JRE_div2_ppfree", NB_OTHER)]:
    boot[name] = boot_run(name, nb)
    B = boot[name]
    q = np.percentile(B, [15.87, 50, 84.13], axis=0)
    fits[name]["summary"]["boot"] = dict(n=len(B), mean=B.mean(0).tolist(), sd=B.std(0, ddof=1).tolist(), median=q[1].tolist(),
                                         sd_robust=((q[2] - q[0]) / 2).tolist(), corr=np.round(np.corrcoef(B.T), 3).tolist(),
                                         columns=["f", "cx", "cy", "c1", "c2", "c3"])
    print(f"bootstrap {name}: median {np.round(q[1], 4)} sd {np.round(B.std(0, ddof=1), 4)} sd_rob {np.round((q[2] - q[0]) / 2, 4)}")


# ------------------------------------------------------------------ mapping uncertainty (evaltools, rotation-compensated)
def opencv_from_row(row, kind):
    f, cx, cy = row[:3]
    if kind == "poly":
        d = L.Dist("poly", [c for c in row[3:5]], None, (cx, cy))
        return K_from(f, f, cx, cy), d.opencv(f)
    d = L.Dist("div", [c for c in row[3:5]], None, (cx, cy))
    K, dd, _, _ = L.fit_opencv_to_model(f, np.array([cx, cy]), d, nk=3)
    return K, dd


uv, _ = grid(40)
for name in [MAIN, "JRE_k1k2_ppC0", "JPT_k1k2_ppfree"]:
    s = fits[name]["summary"]
    kind = fits[name]["model"].kind
    row0 = [s["f"]] + s["pp"] + s["coeffs"]
    K0, dd0 = opencv_from_row(row0, kind)
    disp = []
    for row in boot[name]:
        K1, d1 = opencv_from_row(row, kind)
        disp.append(mapping_displacement(K0, dd0, K1, d1, uv=uv, compensate=True))
    ms, rmsmap = mapping_stats(np.array(disp), uv)
    s["mapping_uncertainty"] = ms
    print(f"mapping uncertainty {name}: " + ", ".join(f"{k} {v['rms_median_px']:.2f} (max {v['rms_max_px']:.2f})" for k, v in ms.items() if k != "whole_image"))
    if name == MAIN:
        rms_main = rmsmap

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(prof[:, 0], prof[:, 1], "b.-")
ax[0].axhline(1, color="gray", ls=":")
ax[0].axhline(4, color="gray", ls=":")
ax[0].set_xlabel("f [px] (fixed)")
ax[0].set_ylabel("delta chi^2 (joint cost, sigma units)")
ax[0].set_title(f"focal profile, {MAIN}")
ax2 = ax[0].twinx()
ax2.plot(prof[:, 0], prof[:, 3], "r--", label="pp_y at the profile point")
ax2.set_ylabel("pp_y [px]", color="r")
B = boot[MAIN]
ax[1].plot(B[:, 2], B[:, 0], ".", ms=3, alpha=0.6, label=MAIN)
for name, col in [("JRE_k1k2_ppC0", "tab:red"), ("JPT_k1k2_ppfree", "tab:green"), ("JRE_div2_ppfree", "tab:purple")]:
    ax[1].plot(boot[name][:, 2], boot[name][:, 0], ".", ms=3, alpha=0.6, color=col, label=name)
ax[1].set_xlabel("pp_y [px]")
ax[1].set_ylabel("f [px]")
ax[1].legend(fontsize=7)
ax[1].set_title("bootstrap replicates (joint line self-calibration)")
ax[1].grid(alpha=0.3)
names = list(VARIANTS)
fv = [fits[n]["summary"]["f"] for n in names]
ax[2].barh(range(len(names)), fv, color="tab:blue", alpha=0.6)
for i, n in enumerate(names):
    s = fits[n]["summary"]
    ax[2].text(1310, i, f"pp=({s['pp'][0]:.0f},{s['pp'][1]:.0f})", va="center", fontsize=7)
ax[2].set_yticks(range(len(names)))
ax[2].set_yticklabels(names, fontsize=7)
ax[2].set_xlim(1300, 1600)
ax[2].set_xlabel("f [px]")
ax[2].set_title("joint variants")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_joint.png", dpi=110)
plt.close()

out = dict(edge_files_reviewed=L.file_status(), rejected=[e["id"] for e, a in zip(edges, act) if not a], n_edges=len(acc),
           init=dict(f=f_init, pp=list(d0.c), a=list(d0.a)), straightness_inflation=INFL, sig_pt_px=SIG_PT, main=MAIN,
           variants={n: fits[n]["summary"] for n in fits}, variant_specs={n: {k: v for k, v in s.items()} for n, s in VARIANTS.items()},
           f_profile=dict(f=prof[:, 0].tolist(), dchi2=prof[:, 1].tolist(), ppx=prof[:, 2].tolist(), ppy=prof[:, 3].tolist()),
           n_boot=NB)
L.save(L.jsonable(out), "40_lines_indep_joint.json")
np.savez(f"{L.CACHE}/40_lines_indep_joint_boot.npz", **boot)
print(f"done in {time.time() - T0:.0f}s")
