"""Cross-validation of the main (combined) estimator: leave one sticker ROW / one CART's stickers / one EDGE
REGION out, refit, predict the held-out data.

Prediction of held-out stickers: lens from the reduced fit; the cart pose is re-estimated from the held-out
stickers alone ("transfer" test, 6 pose dof) when the whole cart is left out, otherwise the pose of the reduced
fit is used. Held-out edges: straightness + VP residual of those edges under the reduced lens (VP directions of
the reduced fit; groups that disappear entirely are evaluated for straightness only).
Usage: python3 51_combined_cv.py <fold-name>|all ; writes work/cache/combined_cv_<fold>.json
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

os.environ.setdefault("COMBINED_MAIN", "nonrobust")
sys.argv = [sys.argv[0], "fit"] + sys.argv[1:] if False else sys.argv  # noqa

from combined import Combined  # noqa: E402
from common import CACHE, H, W, K_from, load_json, project, save_json  # noqa: E402
from linedata import load_edges  # noqa: E402
from lineselfcal import LineCal  # noqa: E402
from markerdata import load_markers, marker_side_edges, to_points  # noqa: E402

FOLD = sys.argv[1] if len(sys.argv) > 1 else "all"
init = load_json(f"{CACHE}/initial_calib.json")
P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
M = load_markers()
PTS = to_points(M)
SIDES = marker_side_edges(M, only_partial=True)
LINES = load_edges(verified_only=True)
FLOOR = {"scene_tapeV_left": "floor_V", "scene_tapeV_right": "floor_V", "scene_tapeH_top": "floor_H", "scene_tapeH_bottom": "floor_H",
         "scene_blueH_top": "floor_H", "scene_blueL_top": "floor_H", "scene_blueL_bottom": "floor_H"}
for e in LINES:
    if "board" in e["id"]:
        e["vp_group"] = None
        if e.get("model_line"):
            e["model_line"] = dict(e["model_line"], exact=False)
    if e["id"] in FLOOR:
        e["vp_group"] = FLOOR[e["id"]]
SPEC = dict(f="single", pp="free", dist=["k1", "k2"])
FIT = load_json(f"{CACHE}/combined_fit.json")
x_main = np.array(FIT["x_main"])

FOLDS = {
    "row_top": dict(stickers=lambda m: m["row"] == "top"),
    "row_A": dict(stickers=lambda m: m["row"] == "A"),
    "row_B": dict(stickers=lambda m: m["row"] == "B"),
    "row_C": dict(stickers=lambda m: m["row"] == "C"),
    "cart_80": dict(stickers=lambda m: m["cart"] == 80, whole_cart=80),
    "cart_310": dict(stickers=lambda m: m["cart"] == 310, whole_cart=310),
    "edges_cart310": dict(edges=lambda e: e["region"] == "cart310"),
    "edges_cart80": dict(edges=lambda e: e["region"] == "cart80"),
    "edges_scene": dict(edges=lambda e: e["region"] == "scene"),
}


def run_fold(name):
    fd = FOLDS[name]
    sel_m = fd.get("stickers", lambda m: False)
    sel_e = fd.get("edges", lambda e: False)
    out_mi = {mi for mi, m in enumerate(M) if sel_m(m)}
    tr_pts = [p for p in PTS if p["mi"] not in out_mi]
    te_pts = [p for p in PTS if p["mi"] in out_mi]
    tr_sides = [s for s in SIDES if s.get("sticker_mi") not in out_mi]
    tr_lines = [e for e in LINES if not sel_e(e)]
    te_lines = [e for e in LINES if sel_e(e)]
    cb = Combined(tr_pts, tr_lines + tr_sides, SPEC, sig=dict(FIT["sig_main"]))
    # start from the main solution (same parameter layout when both carts keep stickers; else rebuild)
    K0 = K_from(x_main[0], x_main[0], x_main[1], x_main[2])
    d0 = np.array([x_main[3], x_main[4], 0, 0, 0])
    x0 = cb.x0(K0, d0, P0)
    if len(x0) == len(x_main):
        x0 = x_main.copy()
    r = cb.solve(x0)
    K, d, poses, wv = cb.unpack(r.x)
    res = dict(fold=name, intrinsics=dict(zip(cb.inames, r.x[: cb.ni].tolist())), n_train_pts=len(tr_pts), n_train_edges=len(tr_lines))
    # held-out stickers
    if te_pts:
        err = []
        wc = fd.get("whole_cart")
        if wc is not None:
            X = np.array([p["X"] for p in te_pts])
            uv = np.array([p["uv"] for p in te_pts])
            rp = least_squares(lambda pz: (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), P0[wc])
            e = rp.fun.reshape(-1, 2)
            res["heldout_stickers_pose_refit_rms_px"] = float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
        else:
            for p in te_pts:
                pz = poses[cb.cidx[p["cart"]]]
                err.append(project(p["X"][None], K, d, rvec=pz[:3], tvec=pz[3:])[0] - p["uv"])
            e = np.array(err)
            res["heldout_stickers_pred_rms_px"] = float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
    # held-out edges: straightness under the reduced lens
    if te_lines:
        lc = LineCal(te_lines, "plumb", dist_free=(), f0=K[0, 0], centre_fixed=[K[0, 2], K[1, 2]], dist_fixed=d[:5], subsample=1)
        rr = lc.residuals(np.zeros(0))[:-1]
        res["heldout_edges_straightness_rms_px"] = float(np.sqrt(np.mean(rr ** 2)))
    # in-fit reference
    b = cb.blocks(r.x)
    res["train_block_rms"] = {k: float(np.sqrt(np.mean(v ** 2))) if len(v) else None for k, v in b.items()}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "train_block_rms"}, flush=True)
    save_json(res, f"{CACHE}/combined_cv_{name}.json")
    return res


names = list(FOLDS) if FOLD == "all" else FOLD.split(",")
for n in names:
    run_fold(n)
