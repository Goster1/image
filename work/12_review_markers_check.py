"""Review of stage 11 (markers): independent checks computed ONLY from results/detections.json
(plus the spec), i.e. a test that the file allows repeating the computation.

1. schema / consistency checks of results/detections.json (corners_3d_mm vs spec + rot_k, validity,
   per-frame values vs corners_px, frames, jitter model)
2. joint fit (f single, pp free, k1, k2; one pose per cart) with a seed that does NOT use any cache
   file (f = 1300, pp = image centre, poses from solvePnP) -> RMS must match the reported 5.34 px
3. rot_k check by leave-one-sticker-out prediction: the fit without the sticker predicts its four
   corners under each of the 4 in-plane rotations; the chosen rot_k must be the nearest one
4. raw still coordinates: per-frame corners mapped back into each still (inverse jitter model)

Output: work/cache/review_markers/check.json (printed summary)
"""
import os

import cv2
import numpy as np

from calib import Model, Problem
from common import CACHE, H, RESULTS, W, K_from, load_json, marker_corners_3d, save_json

OUT = f"{CACHE}/review_markers"
os.makedirs(OUT, exist_ok=True)
MODEL = dict(f="single", pp="free", dist=["k1", "k2"])


def points_from_det(det, rot=None, skip=None):
    pts = []
    for m in det["markers"]:
        key = (m["cart"], m["id"])
        if skip is not None and key == skip:
            continue
        rk = m["rot_k"] if rot is None else rot.get(key, m["rot_k"])
        X = marker_corners_3d(m["id"], m["cart"], rk)
        for j in range(4):
            if m["corner_valid"][j]:
                pts.append(dict(cart=m["cart"], id=m["id"], corner=j, X=X[j], uv=np.array(m["corners_px"][j], float)))
    return pts


def seed(pts):
    K = K_from(1300.0, 1300.0, (W - 1) / 2, (H - 1) / 2)
    x = [1300.0, (W - 1) / 2, (H - 1) / 2, 0.0, 0.0]
    for c in (80, 310):
        P = [p for p in pts if p["cart"] == c]
        X = np.array([p["X"] for p in P])
        uv = np.array([p["uv"] for p in P])
        ok, rv, tv = cv2.solvePnP(X, uv, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        x += list(rv.ravel()) + list(tv.ravel())
    return np.array(x)


def fit(pts):
    pr = Problem(Model(MODEL), pts, carts=[80, 310])
    r = pr.solve(seed(pts))
    res = pr.predict(r.x) - pr.uv
    return pr, r, res


def main():
    det = load_json(f"{RESULTS}/detections.json")
    out = {}
    # ---------------- 1. schema / consistency
    need = ["id", "cart", "role", "rot_k", "center_mm", "corners_px", "corners_3d_mm", "corner_valid", "per_frame_corners_px"]
    issues = []
    for m in det["markers"]:
        for k in need:
            if k not in m:
                issues.append(f"{m.get('cart')}:{m.get('id')} missing {k}")
        X = marker_corners_3d(m["id"], m["cart"], m["rot_k"])
        if np.abs(X - np.array(m["corners_3d_mm"])).max() > 1e-9:
            issues.append(f"{m['cart']}:{m['id']} corners_3d_mm inconsistent with spec+rot_k")
        pf = np.array([[p if p is not None else [np.nan, np.nan] for p in fr] for fr in m["per_frame_corners_px"]], float)
        if pf.shape[0] != len(det["frames"]):
            issues.append(f"{m['cart']}:{m['id']} per-frame count {pf.shape[0]}")
        for j in range(4):
            if m["corner_valid"][j]:
                mu = np.nanmean(pf[:, j], 0)
                if np.linalg.norm(mu - np.array(m["corners_px"][j])) > 1e-6:
                    issues.append(f"{m['cart']}:{m['id']}:{j} corners_px != mean of per-frame")
            else:
                if np.isfinite(pf[:, j]).any():
                    issues.append(f"{m['cart']}:{m['id']}:{j} invalid corner has per-frame values")
    out["schema_issues"] = issues
    print("schema issues:", issues if issues else "none")

    # ---------------- 2. independent joint fit
    pts = points_from_det(det)
    pr, r, res = fit(pts)
    K, dist, poses = pr.split(r.x)
    e = np.linalg.norm(res, axis=1)
    out["fit_all"] = dict(n=len(pts), rms=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()),
                          f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]), k1=float(dist[0]), k2=float(dist[1]))
    print("independent fit from detections.json:", {k: round(v, 4) for k, v in out["fit_all"].items()})

    # ---------------- 3. rot_k by leave-one-sticker-out prediction
    rot_rows = []
    for m in det["markers"]:
        key = (m["cart"], m["id"])
        nv = int(np.sum(m["corner_valid"]))
        pts_o = points_from_det(det, skip=key)
        pro, ro, _ = fit(pts_o)
        Ko, do, po = pro.split(ro.x)
        from common import project
        uv = np.array(m["corners_px"], float)
        ok = np.array(m["corner_valid"])
        ci = [80, 310].index(m["cart"])
        errs = []
        for rk in range(4):
            X = marker_corners_3d(m["id"], m["cart"], rk)
            p = project(X, Ko, do, rvec=po[ci, :3], tvec=po[ci, 3:])
            errs.append(float(np.sqrt(np.mean(np.sum((p[ok] - uv[ok]) ** 2, 1)))))
        best = int(np.argmin(errs))
        srt = sorted(errs)
        rot_rows.append(dict(marker=f"{m['cart']}:{m['id']}", n_valid=nv, rot_k=m["rot_k"], loo_best=best,
                             loo_rms_per_rot=errs, gap=srt[1] - srt[0], agree=best == m["rot_k"]))
        print(f"{m['cart']:4d}:{m['id']:<4d} nvalid={nv} rot_k={m['rot_k']} LOO best={best} rms/rot="
              + " ".join(f"{v:6.1f}" for v in errs) + ("" if best == m["rot_k"] else "   <-- DISAGREE"))
    out["rot_k_loo"] = rot_rows

    # ---------------- 4. raw still coordinates (inverse jitter)
    J = np.array(det["frame_jitter"]["params_per_frame_ax0_ax1_ay0_ay1"], float)

    def mean_to_frame(P, f):
        ax0, ax1, ay0, ay1 = J[f]
        Q = P.copy()
        for _ in range(20):
            yy = Q[:, 1] - H / 2
            fm = np.column_stack([Q[:, 0] - (ax0 + ax1 * yy), Q[:, 1] - (ay0 + ay1 * yy)])
            Q = Q + (P - fm)
        return Q

    mx = 0.0
    for m in det["markers"]:
        pf = np.array([[p if p is not None else [np.nan, np.nan] for p in fr] for fr in m["per_frame_corners_px"]], float)
        for f in range(len(pf)):
            q = mean_to_frame(np.nan_to_num(pf[f]), f)
            mx = max(mx, float(np.nanmax(np.abs(q - np.nan_to_num(pf[f])))))
    out["max_jitter_correction_px"] = mx
    print("max |raw - jitter-corrected| over all corners/frames: %.3f px" % mx)
    save_json(out, f"{OUT}/check.json")


if __name__ == "__main__":
    main()
