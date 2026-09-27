"""Methods from edges only (orchestrator implementation, lineselfcal.LineCal):

  plumbline   : distortion (k1,k2 [,k3]) from straightness; centre fixed at image centre or free
  vanishing   : f, pp from VP orthogonality with the plumb-line distortion fixed
  lines_joint : distortion + f + pp jointly (straight + VP-consistent)

Uncertainty: cluster bootstrap over edges (each replicate re-estimates everything) + sandwich cov.
k scaling: plumb-line alone determines k_i / f^(2i) only; its OpenCV k's are quoted for the f of the
vanishing-point / joint fit (and the invariant pixel-unit values are stored too).
Writes work/cache/method_plumbline*.json, method_vanishing.json, method_lines_joint.json, lines_full.json.
"""
import sys

import numpy as np

from common import CACHE, H, W, K_from, lens_json, load_json, save_json
from evaltools import grid, mapping_displacement, mapping_stats
from linedata import load_edges
from lineselfcal import LineCal, sandwich

rng = np.random.default_rng(2024)
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
F0 = 1300.0
init = load_json(f"{CACHE}/initial_calib.json")
P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
EDGES = load_edges(verified_only=True)
# VP-group policy (documented in REPORT): end-board edges are straight but not aligned with the cart axes
# to VP precision -> straightness only; parallel floor lines form horizontal VP groups perpendicular to the
# vertical (edges of one tape / blue line).
FLOOR = {"scene_tapeV_left": "floor_V", "scene_tapeV_right": "floor_V", "scene_tapeH_top": "floor_H",
         "scene_tapeH_bottom": "floor_H", "scene_blueH_top": "floor_H", "scene_blueL_top": "floor_H", "scene_blueL_bottom": "floor_H"}
for e in EDGES:
    if "board" in e["id"]:
        e["vp_group"] = None
    if e["id"] in FLOOR:
        e["vp_group"] = FLOOR[e["id"]]
print("edges:", len(EDGES), "by region:", {r: sum(e["region"] == r for e in EDGES) for r in sorted({e["region"] for e in EDGES})})


def groups_idx(lc):
    return lc.edge_groups_index()


def fit_plumb(edges, centre_free=False, dist=("k1", "k2"), centre=None, sub=2):
    lc = LineCal(edges, "plumb", dist_free=dist, centre_free=centre_free, f0=F0, centre_fixed=centre, subsample=sub)
    r = lc.solve(lc.x0(), loss="huber", f_scale=0.5)
    return lc, r


def fit_vp(edges, dist_px, pp_free=True, x_init=None, sub=2):
    """dist_px: distortion in pixel units as (k1/f^2, k2/f^4, p1/f, p2/f, k3/f^6) -> converted per f inside."""
    lc = LineCal(edges, "vp", pp_free=pp_free, subsample=sub)

    # distortion fixed in pixel units: re-scale OpenCV coefficients with the current f
    def unpack_fixed(x, _orig=lc.unpack):
        K, d, rots, wv = _orig(x)
        f = K[0, 0]
        d = np.array([dist_px[0] * f ** 2, dist_px[1] * f ** 4, dist_px[2] * f, dist_px[3] * f, dist_px[4] * f ** 6])
        return K, d, rots, wv

    lc.unpack = unpack_fixed
    K0 = K_from(F0, F0, (W - 1) / 2, (H - 1) / 2)
    x0 = lc.x0(K0, np.zeros(5), P0) if x_init is None else x_init
    r = lc.solve(x0, loss="huber", f_scale=0.5)
    return lc, r


def fit_joint(edges, pp_free=True, dist=("k1", "k2"), x_init=None, dist_init=None, sub=2):
    lc = LineCal(edges, "joint", dist_free=dist, pp_free=pp_free, subsample=sub)
    K0 = K_from(F0, F0, (W - 1) / 2, (H - 1) / 2)
    x0 = lc.x0(K0, dist_init if dist_init is not None else np.array([-0.28, 0.05, 0, 0, 0]), P0) if x_init is None else x_init
    r = lc.solve(x0, loss="huber", f_scale=0.5)
    return lc, r


def px_units(d, f):
    return np.array([d[0] / f ** 2, d[1] / f ** 4, d[2] / f, d[3] / f, d[4] / f ** 6])


def summarize(lc, r):
    C = sandwich(r, groups_idx(lc))
    s = np.sqrt(np.abs(np.diag(C)))
    return {n: (float(v), float(e)) for n, v, e in zip(lc.names[: lc.n_intr], r.x[: lc.n_intr], s[: lc.n_intr])}


out = {}
# ---------------- plumb-line variants ----------------
for name, kw in {"plumb_k1k2_centre_fixed": dict(), "plumb_k1k2_centre_free": dict(centre_free=True),
                 "plumb_k1k2k3_centre_free": dict(centre_free=True, dist=("k1", "k2", "k3")),
                 "plumb_k1_centre_fixed": dict(dist=("k1",)),
                 "plumb_k1k2p1p2_centre_fixed": dict(dist=("k1", "k2", "p1", "p2"))}.items():
    lc, r = fit_plumb(EDGES, **kw)
    pe = np.array(lc.residuals(r.x, per_edge=True))
    out[name] = dict(params=summarize(lc, r), rms=float(np.sqrt(np.mean(r.fun ** 2))), median_edge_rms=float(np.median(pe)),
                     worst_edges=sorted([(EDGES[i]["id"], float(pe[i])) for i in range(len(pe))], key=lambda t: -t[1])[:8])
    print(name, {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in out[name]["params"].items()}, "rms", round(out[name]["rms"], 3))

# per-region plumb (centre free) for consistency
for reg in sorted({e["region"] for e in EDGES}):
    sub = [e for e in EDGES if e["region"] == reg]
    if len(sub) < 6:
        continue
    for cf in (False, True):
        lc, r = fit_plumb(sub, centre_free=cf)
        out[f"plumb_region_{reg}_{'free' if cf else 'fixed'}"] = dict(params=summarize(lc, r), n=len(sub))
        print("  region", reg, "centre free" if cf else "centre fixed", {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in summarize(lc, r).items()})

# ---------------- joint (distortion + f + pp from lines) ----------------
lcj, rj = fit_joint(EDGES)
out["lines_joint"] = dict(params=summarize(lcj, rj), rms=float(np.sqrt(np.mean(rj.fun ** 2))))
print("joint", {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in out["lines_joint"]["params"].items()}, "rms", round(out["lines_joint"]["rms"], 3))
lcj0, rj0 = fit_joint(EDGES, pp_free=False)
out["lines_joint_ppfixed"] = dict(params=summarize(lcj0, rj0), rms=float(np.sqrt(np.mean(rj0.fun ** 2))))
print("joint pp fixed", {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in out["lines_joint_ppfixed"]["params"].items()})

# ---------------- VP with plumb distortion (centre fixed at image centre) ----------------
lcp, rp = fit_plumb(EDGES)
dp = np.zeros(5)
dp[0], dp[1] = rp.x[0], rp.x[1]
dpx = px_units(dp, F0)
lcv, rv = fit_vp(EDGES, dpx)
out["vanishing"] = dict(params=summarize(lcv, rv), rms=float(np.sqrt(np.mean(rv.fun ** 2))), dist_px_units=dpx.tolist())
print("vp (plumb dist fixed)", {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in out["vanishing"]["params"].items()})
lcv0, rv0 = fit_vp(EDGES, dpx, pp_free=False)
out["vanishing_ppfixed"] = dict(params=summarize(lcv0, rv0))
print("vp pp fixed", {k: f"{v[0]:.5g}±{v[1]:.2g}" for k, v in out["vanishing_ppfixed"]["params"].items()})

# ---------------- cluster bootstrap over edges (everything re-estimated) ----------------
boot = {"plumb": [], "vp": [], "joint": []}
for b in range(NBOOT):
    idx = rng.integers(0, len(EDGES), len(EDGES))
    sub = [dict(EDGES[i], id=f"{EDGES[i]['id']}#{q}") for q, i in enumerate(idx)]
    try:
        lc, r = fit_plumb(sub, sub=4)
        boot["plumb"].append(r.x[:2])
        d = np.zeros(5)
        d[:2] = r.x[:2]
        lc2, r2 = fit_vp(sub, px_units(d, F0), x_init=rv.x, sub=4)
        boot["vp"].append(r2.x[:3])
        lc3, r3 = fit_joint(sub, x_init=rj.x, sub=4)
        boot["joint"].append(r3.x[: lcj.n_intr])
    except Exception as ex:  # noqa
        print("boot fail", ex)
    if b % 10 == 0:
        print("boot", b)
for k in boot:
    boot[k] = np.array(boot[k])
    if len(boot[k]):
        print("bootstrap", k, "mean", np.round(boot[k].mean(0), 4), "std", np.round(boot[k].std(0), 4))


def lens_from_joint(x, names):
    p = dict(zip(names, x))
    K = K_from(p["f"], p["f"], p.get("cx", (W - 1) / 2), p.get("cy", (H - 1) / 2))
    d = np.array([p.get("k1", 0), p.get("k2", 0), p.get("p1", 0), p.get("p2", 0), p.get("k3", 0)])
    return K, d


uv, _ = grid(40)
# joint lens json
Kj, dj = lens_from_joint(rj.x[: lcj.n_intr], lcj.names[: lcj.n_intr])
disp = np.array([mapping_displacement(Kj, dj, *lens_from_joint(xb, lcj.names[: lcj.n_intr]), uv) for xb in boot["joint"]]) if len(boot["joint"]) else None
mj = mapping_stats(disp, uv)[0] if disp is not None else None
sd = boot["joint"].std(0) if len(boot["joint"]) else np.full(lcj.n_intr, np.nan)
save_json(lens_json(Kj, dj, method="lines_joint (edges only: straightness + VP orthogonality)", model="fx=fy, pp free, k1,k2",
                    rms_reprojection_error_px=out["lines_joint"]["rms"],
                    uncertainty={f"{n}_1sigma": float(s) for n, s in zip(lcj.names[: lcj.n_intr], sd)},
                    mapping_uncertainty_px={k: v["rms_median_px"] for k, v in mj.items() if k != "whole_image"} if mj else None,
                    data_used=f"{len(EDGES)} verified straight edges", geometry_assumptions="no dimensions used; only straightness, parallelism and orthogonality of cart axes",
                    details=out["lines_joint"]), f"{CACHE}/method_lines_joint.json")
# vanishing lens json (f, pp from VP with plumb distortion)
pv = dict(zip(lcv.names, rv.x))
fv = pv["f"]
Kv = K_from(fv, fv, pv["cx"], pv["cy"])
dv = np.array([dpx[0] * fv ** 2, dpx[1] * fv ** 4, 0, 0, 0])
sdv = boot["vp"].std(0) if len(boot["vp"]) else [np.nan] * 3
save_json(lens_json(Kv, dv, method="vanishing_points (f, pp from orthogonal VPs; distortion from plumb-line)", model="fx=fy, pp free; k1,k2 from plumb-line",
                    rms_reprojection_error_px=out["vanishing"]["rms"], uncertainty={"f_1sigma": float(sdv[0]), "cx_1sigma": float(sdv[1]), "cy_1sigma": float(sdv[2])},
                    data_used=f"{len(EDGES)} verified edges (VP groups: cart X/Y/Z of both carts, scene verticals)",
                    geometry_assumptions="no dimensions; cart axes mutually orthogonal", details=out["vanishing"]), f"{CACHE}/method_vanishing.json")
# plumb lens json (distortion; quoted at the VP focal length)
sdp = boot["plumb"].std(0) if len(boot["plumb"]) else [np.nan] * 2
Kp = K_from(fv, fv, (W - 1) / 2, (H - 1) / 2)
dpo = np.array([dpx[0] * fv ** 2, dpx[1] * fv ** 4, 0, 0, 0])
save_json(lens_json(Kp, dpo, method="plumb_line (distortion only, centre = image centre; f taken from the VP method)",
                    model="k1,k2 radial, distortion centre fixed at image centre",
                    rms_reprojection_error_px=out["plumb_k1k2_centre_fixed"]["rms"],
                    uncertainty={"k1_1sigma": float(sdp[0] * (fv / F0) ** 2), "k2_1sigma": float(sdp[1] * (fv / F0) ** 4),
                                 "k1_over_f2_px": float(dpx[0]), "k2_over_f4_px": float(dpx[1])},
                    data_used=f"{len(EDGES)} verified straight edges", geometry_assumptions="none (straightness only)",
                    details={k: v for k, v in out.items() if k.startswith("plumb")}), f"{CACHE}/method_plumbline.json")
save_json(dict(out=out, boot={k: v.tolist() for k, v in boot.items()}, mapping_joint=mj), f"{CACHE}/lines_full.json")
print("done")
