"""How discriminating are the floor checks of 71_grid_checks.py? Refit the main combined estimator with f FIXED at
several values (everything else - pp, k1, k2, poses, block weights of the main fit - refitted, like the f profile),
and recompute the width of the white floor tape and the angle between its two directions on the floor of cart 310.
Writes work/cache/grid_fsens.json."""
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import CACHE, RESULTS, load_json, save_json, undistort_points

sys.argv = ["50_combined.py", "none"]
ns = {}
exec(compile(open("50_combined.py").read().split('if MODE == "fit":')[0], "50c", "exec"), ns)
FIT = load_json(f"{CACHE}/combined_fit.json")
Combined = ns["Combined"]
cb = Combined(ns["PTS"], ns["EDGES"], ns["SPECS"]["k1k2"], sig=dict(FIT["sig_main"]))
x0 = np.array(FIT["x_main"])
E = {e["id"]: e for b in load_json(f"{RESULTS}/edges.json")["regions"] for e in b["edges"]}
FLOOR_Z = -1800.0


def floor_pts(uv, K, d, rv, tv):
    R = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))[0]
    C = -R.T @ tv
    n = undistort_points(np.asarray(uv, float), K, d)
    r = np.column_stack([n, np.ones(len(n))]) @ R
    return (C[None] + ((FLOOR_Z - C[2]) / r[:, 2])[:, None] * r)[:, :2]


def line(Q):
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c)
    return c, vt[0]


out = []
for fv in (1400.0, 1420.0, 1440.0, 1455.0, float(x0[0]), 1485.0, 1500.0, 1520.0, 1540.0):
    rr = least_squares(lambda y: cb.residuals(np.r_[fv, y]), x0[1:], method="trf", x_scale="jac", max_nfev=300)
    x = np.r_[fv, rr.x]
    K, d, poses, _ = cb.unpack(x)
    p310 = poses[cb.cidx[310]]
    L = {k: line(floor_pts(E[k]["points"], K, d[:5], p310[:3], p310[3:])) for k in ("scene_tapeV_left", "scene_tapeV_right", "scene_tapeH_top", "scene_tapeH_bottom")}
    (ca, da), (cbb, _) = L["scene_tapeV_left"], L["scene_tapeV_right"]
    nrm = np.array([-da[1], da[0]])
    Qb = floor_pts(E["scene_tapeV_right"]["points"], K, d[:5], p310[:3], p310[3:])
    wV = float(np.median(np.abs((Qb - ca) @ nrm)))
    (ch, dh), _ = L["scene_tapeH_top"], L["scene_tapeH_bottom"]
    Qh = floor_pts(E["scene_tapeH_bottom"]["points"], K, d[:5], p310[:3], p310[3:])
    wH = float(np.median(np.abs((Qh - ch) @ np.array([-dh[1], dh[0]]))))
    ang = float(np.degrees(np.arccos(min(1.0, abs(da @ dh)))))
    R = cv2.Rodrigues(p310[:3].reshape(3, 1))[0]
    h = float((-R.T @ p310[3:])[2] - FLOOR_Z)
    b = cb.blocks(x)
    rec = dict(f=fv, cx=float(K[0, 2]), cy=float(K[1, 2]), k1=float(d[0]), k2=float(d[1]), width_V_mm=wV, width_H_mm=wH, angle_deg=ang,
               camera_height_mm=h, block_rms={k: float(np.sqrt(np.mean(v ** 2))) for k, v in b.items() if len(v)})
    out.append(rec)
    print(f"f {fv:7.1f}  pp ({K[0, 2]:6.1f},{K[1, 2]:6.1f})  tape width V {wV:5.1f} H {wH:5.1f} mm  angle {ang:6.2f} deg  cam h {h:6.0f} mm", flush=True)
save_json(dict(description=__doc__, rows=out), f"{CACHE}/grid_fsens.json")
