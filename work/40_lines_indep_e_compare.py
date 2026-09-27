"""Comparison of the independent line pipeline (40_lines_indep_*) with the orchestrator implementation
(lineselfcal.LineCal, modes plumb / vp / joint) on EXACTLY the same accepted edge set and VP groups.

Mapping of groups: 310X -> 310_cartX, 310Z -> 310_cartZ, 80X -> 80_cartX, 80Z -> 80_cartZ, WV -> world_vertical;
end boards, floor lines and all other edges -> straightness only (vp_group None).
LineCal uses separate cart frames and point-level VP constraints (comparable to our 'JPT_k1k2_framesfree');
its residuals are 'forward' (distance between the observed point and the distorted foot point), ours are the
first-order orthogonal distances to the distorted line.  Both residual modes of LineCal are run.
Writes work/cache/40_lines_indep_compare.json
"""
import importlib
import time

import cv2
import numpy as np

from common import W, H, K_from
from common import CACHE, load_json
import lineselfcal as LS

L = importlib.import_module("40_lines_indep_lib")
T0 = time.time()
F0 = 1300.0

edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
GMAP = {"310X": "310_cartX", "310Z": "310_cartZ", "80X": "80_cartX", "80Z": "80_cartZ", "WV": "world_vertical"}
ls_edges = []
for e in acc:
    ls_edges.append(dict(id=e["id"], points=e["points"], direction=e["direction"], cart=e["cart"], vp_group=GMAP.get(e["group"])))

out = {}
# ------------------------------------------------------------------ ours
ours = {}
for nm, m in [("k1k2", L.PlumbModel("poly", 2)), ("k1k2_c", L.PlumbModel("poly", 2, centre_free=True))]:
    f = L.fit_plumb(L.EdgeSet(acc), m)
    ours[nm] = dict(x=f["x"].tolist(), centre=f["dist"].c.tolist(), k1_over_f2=f["dist"].a[0] / L.S ** 2, k2_over_f4=f["dist"].a[1] / L.S ** 4,
                    k_at_f1300=f["dist"].opencv(F0).tolist(), rms=f["rms"])
out["ours_plumb"] = ours


# ------------------------------------------------------------------ LineCal plumb
def ls_plumb(centre_free, mode):
    lc = LS.LineCal(ls_edges, "plumb", dist_free=("k1", "k2"), centre_free=centre_free, f0=F0, subsample=1)
    lc.resid_mode = mode
    r = lc.solve(lc.x0(), loss="huber", f_scale=0.5)
    K, d, _, _ = lc.unpack(r.x)
    cov = LS.sandwich(r, lc.edge_groups_index())
    se = np.sqrt(np.abs(np.diag(cov)))
    return dict(names=lc.names, x=r.x.tolist(), se_sandwich=se.tolist(), centre=[K[0, 2], K[1, 2]], k=d.tolist(),
                k1_over_f2=d[0] / F0 ** 2, k2_over_f4=d[1] / F0 ** 4, rms=float(np.sqrt(np.mean(r.fun[:-1] ** 2))))


ls = {}
for cf in (False, True):
    for mode in ("forward", "jacobian"):
        key = f"plumb_{'free' if cf else 'fixed'}_{mode}"
        ls[key] = ls_plumb(cf, mode)
        print(f"LineCal {key}: centre {np.round(ls[key]['centre'], 1)} k@1300 {np.round(ls[key]['k'][:2], 4)} rms {ls[key]['rms']:.3f}")
for nm, o in ours.items():
    print(f"ours    plumb {nm}: centre {np.round(o['centre'], 1)} k@1300 {np.round(o['k_at_f1300'][:2], 4)} rms {o['rms']:.3f}")
out["linecal_plumb"] = ls

# ------------------------------------------------------------------ initial poses from our VPs
pl = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
vps = L.vp_estimates(acc, pl["dist"], ["310X", "80X", "CZ", "WV", "310Z"])
f_init = L.f_orthogonal(vps, [("310X", "CZ"), ("80X", "CZ")], pl["dist"].c)
Kin = np.array([[f_init, 0, pl["dist"].c[0]], [0, f_init, pl["dist"].c[1]], [0, 0, 1.0]])
Ki = np.linalg.inv(Kin)
poses = {}
for c in (310, 80):
    x = L.unit(Ki @ L.from_scaled_h(vps[f"{c}X"]["V"]))
    z = L.unit(Ki @ L.from_scaled_h(vps["CZ"]["V"]))
    if z[2] < 0:
        z = -z
    R = L.rot_from_two(x, z)
    poses[c] = cv2.Rodrigues(R)[0].ravel()


def ls_vp_joint(mode_, pp_free, dist_px=None, resid="forward"):
    lc = LS.LineCal(ls_edges, mode_, dist_free=("k1", "k2"), pp_free=pp_free, f0=F0, subsample=1,
                    centre_fixed=None)
    lc.resid_mode = resid
    if mode_ == "vp":
        orig = lc.unpack

        def unpack_fixed(x, _orig=orig):
            K, d, rots, wv = _orig(x)
            f = K[0, 0]
            d = np.array([dist_px[0] * f ** 2, dist_px[1] * f ** 4, 0.0, 0.0, 0.0])
            return K, d, rots, wv

        lc.unpack = unpack_fixed
    K0 = Kin.copy() if pp_free else K_from(f_init, f_init, (W - 1) / 2, (H - 1) / 2)
    d0 = pl["dist"].opencv(f_init)
    r = lc.solve(lc.x0(K0, d0, poses), loss="huber", f_scale=0.5)
    K, d, rots, wv = lc.unpack(r.x)
    cov = LS.sandwich(r, lc.edge_groups_index())
    se = np.sqrt(np.abs(np.diag(cov)))[: lc.n_intr]
    return dict(names=lc.names[: lc.n_intr], x=r.x[: lc.n_intr].tolist(), se_sandwich=se.tolist(), f=float(K[0, 0]), pp=[float(K[0, 2]), float(K[1, 2])],
                k=d.tolist(), k1_over_f2=float(d[0] / K[0, 0] ** 2), k2_over_f4=float(d[1] / K[0, 0] ** 4), rms=float(np.sqrt(np.mean(r.fun[:-1] ** 2))))


# LineCal vp with our plumb distortion (pixel units) and with its own plumb distortion
res_ls = {}
for tag, src in [("ourplumb_c", (pl["dist"].a[0] / L.S ** 2, pl["dist"].a[1] / L.S ** 4)),
                 ("lsplumb_fixed", (ls["plumb_fixed_forward"]["k1_over_f2"], ls["plumb_fixed_forward"]["k2_over_f4"]))]:
    for ppf in (True, False):
        key = f"vp_{tag}_{'ppfree' if ppf else 'ppC0'}"
        try:
            res_ls[key] = ls_vp_joint("vp", ppf, dist_px=src)
            print(f"LineCal {key}: f {res_ls[key]['f']:.1f} pp {np.round(res_ls[key]['pp'], 1)}  se {np.round(res_ls[key]['se_sandwich'], 1)}")
        except Exception as ex:  # noqa
            print("LineCal", key, "failed", ex)
for ppf in (True, False):
    for resid in ("forward", "jacobian"):
        key = f"joint_{'ppfree' if ppf else 'ppC0'}_{resid}"
        try:
            res_ls[key] = ls_vp_joint("joint", ppf, resid=resid)
            print(f"LineCal {key}: f {res_ls[key]['f']:.1f} pp {np.round(res_ls[key]['pp'], 1)} k {np.round(res_ls[key]['k'][:2], 4)} rms {res_ls[key]['rms']:.3f}")
        except Exception as ex:  # noqa
            print("LineCal", key, "failed", ex)
out["linecal_vp_joint"] = res_ls

# ------------------------------------------------------------------ ours (from the cache files of b and c)
try:
    b = load_json(f"{CACHE}/40_lines_indep_vp.json")
    out["ours_vp"] = {k: b["plumb_variants"]["k1k2_c"]["solutions"].get(k) for k in ["S1_cartX_cartZ|ppDist", "S1_cartX_cartZ|ppC0", "S1w_cartX_WV|ppDist"]}
    out["ours_vp"]["k1k2_centre_fixed:S1|ppC0"] = b["plumb_variants"]["k1k2"]["solutions"].get("S1_cartX_cartZ|ppC0")
except Exception as ex:  # noqa
    print("no vp cache", ex)
try:
    c = load_json(f"{CACHE}/40_lines_indep_joint.json")
    out["ours_joint"] = {k: {q: v.get(q) for q in ("f", "pp", "opencv_dist", "invariants")} for k, v in c["variants"].items()}
except Exception as ex:  # noqa
    print("no joint cache", ex)
L.save(L.jsonable(out), "40_lines_indep_compare.json")
print(f"done in {time.time() - T0:.0f}s")
