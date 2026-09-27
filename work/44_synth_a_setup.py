"""Synthetic study, step A: real reference fits, real residual statistics, the two true lenses and the
calibration of the 'realistic' unknown-geometry errors.

Writes work/cache/synthetic_setup.json (read by 44_synth_b_roundtrip.py / 44_synth_c_sensitivity.py).
"""
import importlib
import os
import time

import numpy as np

from common import CACHE, K_from, load_json, save_json

L = importlib.import_module("44_synth_lib")
rng = np.random.default_rng(4401)
t0 = time.time()

M, E, S = L.load_real()
init = load_json(f"{CACHE}/initial_calib.json")
K_init, d_init = np.array(init["K"]), np.array(init["dist"])
P_init = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
print("edges", len(E), "sides", len(S), "review status", L.edge_review_status())

# ------------------------------------------------------------------ real reference fits
real = {}
for name, spec in L.MARKER_SPECS.items():
    r = L.marker_estimate(spec, [p for p in __import__("markerdata").to_points(M)], K_init, d_init, P_init)
    real[f"markers_{name}"] = dict(names=r["names"], x=r["x"].tolist(), rms=r["rms"],
                                   sig_cov=np.sqrt(np.diag(r["C"])).tolist(), sig_sand=np.sqrt(np.diag(r["Cs"])).tolist())
    print("real markers", name, np.round(r["x"], 4), "rms", round(r["rms"], 3))
K13 = K_from(1300, 1300, *L.CEN)
lcj, rj, Ccj, Csj = L.joint_estimate(E, K13, np.array([-0.28, 0.05, 0, 0, 0]), P_init)
Kj, dj, rotsj, wvj = lcj.unpack(rj.x)
pe_j = lcj.residuals(rj.x, per_edge=True)
real["lines_joint"] = dict(names=lcj.names[: lcj.n_intr], x=rj.x[: lcj.n_intr].tolist(), sig_sand=np.sqrt(np.diag(Csj)).tolist(),
                           rms=float(np.sqrt(np.mean(rj.fun ** 2))))
print("real joint", np.round(rj.x[: lcj.n_intr], 4))
lcp, rp, _, Csp = L.plumb_estimate(E, centre_free=False)
lcpf, rpf, _, Cspf = L.plumb_estimate(E, centre_free=True)
real["plumb_fixed"] = dict(x=rp.x.tolist(), sig_sand=np.sqrt(np.diag(Csp)).tolist(), rms=float(np.sqrt(np.mean(rp.fun ** 2))))
real["plumb_free"] = dict(x=rpf.x.tolist(), sig_sand=np.sqrt(np.diag(Cspf)).tolist(), rms=float(np.sqrt(np.mean(rpf.fun ** 2))))
print("real plumb fixed", np.round(rp.x, 4), "free", np.round(rpf.x, 4))
dpx = np.array([rp.x[0] / L.F0 ** 2, rp.x[1] / L.F0 ** 4])
lcv, rv, _, Csv = L.vp_estimate(E, dpx, K13, P_init)
real["vp"] = dict(x=rv.x[:3].tolist(), sig_sand=np.sqrt(np.diag(Csv)).tolist())
print("real vp", np.round(rv.x[:3], 2))

# ------------------------------------------------------------------ residual statistics of the real edges
stats = L.real_edge_stats(E, S, Kj, dj, rotsj, wvj if wvj is not None else L.world_vertical_dir(E, Kj, dj))
sig = np.array([s["sigma"] for s in stats.values()])
rho = np.array([s["rho"] for s in stats.values()])
bow = np.array([s["bow_rms"] for s in stats.values()])
th = np.array([abs(s["theta_rad"]) for s in stats.values() if s["block"] in ("V", "L")])
print(f"edge white sigma median {np.median(sig):.3f} (IQR {np.percentile(sig,25):.3f}-{np.percentile(sig,75):.3f}), rho median {np.median(rho):.2f}, "
      f"bow rms median {np.median(bow):.3f} max {bow.max():.2f}, |theta| median {np.degrees(np.median(th)):.3f} deg max {np.degrees(th.max()):.2f}")

# ------------------------------------------------------------------ true lenses
TRUTHS = {
    "T1": dict(f=1250.0, cx=955.0, cy=530.0, k1=-0.26, k2=0.06, note="prescribed test lens"),
    # SYNTH_T2="f,cx,cy,k1,k2" overrides T2; the all-methods study (synthetic_setup.json / synthetic_report.*) was run with
    # SYNTH_T2="1400,935,410,-0.32,0.08", the main-estimator study (SYNTH_SETUP=cache/synthetic_setup_main.json) with the default
    "T2": dict(**dict(zip(("f", "cx", "cy", "k1", "k2"), map(float, os.environ.get("SYNTH_T2", "1420,935,505,-0.33,0.09").split(",")))),
               note="near the current edge-based estimates (commit d37a155 pipeline on the reviewed edges: lines joint f 1425, pp (927,503), "
                    "k1 -0.333, k2 0.090; VP f 1419, pp (941,513))"),
}
truth_obj = {}
for tn, p in TRUTHS.items():
    T = L.Truth(tn, p["f"], p["cx"], p["cy"], p["k1"], p["k2"], M, E, P_init)
    truth_obj[tn] = T
    pts = L.sticker_obs(T, M)
    real_uv = np.array([m["corners_px"][j] for m in M for j in range(4) if m["valid"][j]])
    syn_uv = np.array([q["uv"] for q in pts])
    TRUTHS[tn].update(poses={str(c): T.poses[c].tolist() for c in (80, 310)}, wv=T.wv.tolist(), floor={g: v.tolist() for g, v in T.floor.items()},
                      synthetic_vs_real_corner_rms_px=float(np.sqrt(np.mean(np.sum((syn_uv - real_uv) ** 2, 1)))))
    print(tn, T.describe(), "synthetic-vs-real corner rms", round(TRUTHS[tn]["synthetic_vs_real_corner_rms_px"], 2))

# ------------------------------------------------------------------ calibration of the unknown geometry errors (scenario b)
T = truth_obj["T2"]
csig = L.corner_sigmas(M)
real_rms = real["markers_k1k2_ppfree"]["rms"]
target_sticker_part = 2.83  # RMS left after freeing per-cart row heights (geometry_diagnosis.md sec. 3)


def realisations(n, seed):
    """unit-sigma random draws: per-sticker 3D (n x 20 x 3), per cart-row (n x 8), corner noise (n x 20 x 4 x 2)."""
    g = np.random.default_rng(seed)
    rows = sorted({(m["cart"], m["row"]) for m in M})
    return [(g.standard_normal((len(M), 3)), dict(zip(rows, g.standard_normal(len(rows)))), g.standard_normal((len(M), 4, 2)))
            for _ in range(n)]


def offsets_from(z, sig_s, sig_r):
    zs, zr, _ = z
    return {(m["cart"], m["id"]): zs[i] * sig_s + np.array([0, 0, zr[(m["cart"], m["row"])] * sig_r]) for i, m in enumerate(M)}


def med_rms(sig_s, sig_r, Z):
    out = []
    for z in Z:
        off = offsets_from(z, sig_s, sig_r)
        pts = L.sticker_obs(T, M, offsets=off)
        # detection noise (same draws for every sigma)
        k = 0
        for i, m in enumerate(M):
            for j in range(4):
                if m["valid"][j]:
                    pts[k]["uv"] = pts[k]["uv"] + z[2][i, j] * csig[i][j]
                    k += 1
        r = L.marker_estimate(L.MARKER_SPECS["k1k2_ppfree"], pts, T.K, T.d, T.poses)
        out.append(r["rms"])
    return float(np.median(out))


Z = realisations(60, 7)
# sticker part: bisection on sigma_sticker (rows off) for median RMS = target_sticker_part
lo, hi = 0.5, 20.0
for it in range(8):
    mid = 0.5 * (lo + hi)
    if med_rms(mid, 0.0, Z) > target_sticker_part:
        hi = mid
    else:
        lo = mid
s_s = 0.5 * (lo + hi)
print(f"  sigma_sticker {s_s:.2f} mm -> median rms {med_rms(s_s, 0.0, Z):.3f}")
lo, hi = 0.0, 150.0
for it in range(8):
    mid = 0.5 * (lo + hi)
    if med_rms(s_s, mid, Z) > real_rms:
        hi = mid
    else:
        lo = mid
s_r = 0.5 * (lo + hi)
Z2 = realisations(80, 99)
final_rms = med_rms(s_s, s_r, Z2)
print(f"geometry noise: sticker {s_s:.2f} mm (3D isotropic), row {s_r:.2f} mm (per cart x row height); median synthetic rms {final_rms:.3f} vs real {real_rms:.3f}")
# diagnosed what-if (deterministic)
off_c = L.diagnosed_offsets(M)
rc = []
for tn in ("T2",):
    for i in range(10):
        pts = L.sticker_obs(truth_obj[tn], M, offsets=off_c, rng=rng, corner_sigma=csig)
        rc.append(L.marker_estimate(L.MARKER_SPECS["k1k2_ppfree"], pts, truth_obj[tn].K, truth_obj[tn].d, truth_obj[tn].poses)["rms"])
print("diagnosed-geometry what-if rms", np.round(np.median(rc), 3))

save_json(dict(
    review_status=L.edge_review_status(), n_edges=len(E), n_sides=len(S), n_corners=int(sum(m["valid"].sum() for m in M)),
    real=real, truths=TRUTHS,
    reference_lens_for_edge_stats=dict(K=Kj.tolist(), d=dj.tolist(), note="lines joint on the real edges"),
    edge_stats=stats,
    edge_stats_summary=dict(sigma_median=float(np.median(sig)), sigma_p25=float(np.percentile(sig, 25)), sigma_p75=float(np.percentile(sig, 75)),
                            rho_median=float(np.median(rho)), bow_rms_median=float(np.median(bow)), bow_rms_max=float(bow.max()),
                            theta_abs_median_deg=float(np.degrees(np.median(th))), theta_abs_max_deg=float(np.degrees(th.max()))),
    corner_sigma_per_coord=[c.tolist() for c in csig],
    corner_sigma_median=float(np.median(np.concatenate([c[m["valid"]] for c, m in zip(csig, M)]))),
    geometry_noise=dict(sigma_sticker_mm=s_s, sigma_row_mm=s_r, target_sticker_part_px=target_sticker_part, real_rms_px=real_rms,
                        synthetic_median_rms_px=final_rms, diagnosed_whatif_rms_px=float(np.median(rc)),
                        diagnosed_offsets={f"{k[0]}:{k[1]}": v.tolist() for k, v in off_c.items()}),
    runtime_s=time.time() - t0), os.environ.get("SYNTH_SETUP", f"{CACHE}/synthetic_setup.json"))
print("done", round(time.time() - t0, 1), "s")
