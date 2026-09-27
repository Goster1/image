"""Local, nearly lens-independent check of the two 'exact' board edges of cart 310: a plane
homography of the board top (Z=0) from the 8 corners of the two markers lying on that board predicts
where the lines X=0 / X=1600 (and the inner edges) fall; compared with the traced edges.  Done on raw
pixels and after a rough reference undistortion (f=1290, k1=-0.29, k2=0.08, centre = image centre),
because the homography cannot absorb lens distortion.  Indicative only."""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import CACHE, load_json, marker_corners_3d, K_from, undistort_to_pixels, save_json

REF = (K_from(1290, 1290, 959.5, 539.5), np.array([-0.29, 0.08, 0, 0, 0]))


def run(undist):
    ic = load_json(f"{CACHE}/initial_calib.json")
    keys = [tuple(k) for k in ic["keys"]]
    C = np.array(ic["corners_mean"])
    D = load_json(f"{CACHE}/edges_cart310.json")
    E = {e["id"]: np.array(e["points"]) for e in D["edges"]}
    f = (lambda P: undistort_to_pixels(P, *REF)) if undist else (lambda P: np.asarray(P, float))
    out = {}
    for mids, tests in [([0, 322], [("cart310_board0_outer", 0.0), ("cart310_board0_inner_top", 165.0)]),
                        ([1, 2], [("cart310_board1600_outer", 1600.0), ("cart310_board1600_inner", 1435.0)])]:
        obj, img = [], []
        for m in mids:
            i = keys.index((310, m))
            X = marker_corners_3d(m, 310, ic["rot"][f"310_{m}"])
            obj += list(X[:, :2])
            img += list(C[i])
        obj = np.array(obj, float)
        img = f(np.array(img, float))
        Hm, _ = cv2.findHomography(obj, img, 0)
        rms = float(np.sqrt(np.mean(np.sum((cv2.perspectiveTransform(obj[None], Hm)[0] - img) ** 2, 1))))
        for eid, xnom in tests:
            P = f(E[eid])
            t = np.linspace(-50, 500, 400)
            best = None
            for v in np.arange(xnom - 60, xnom + 60, 0.25):
                Q = cv2.perspectiveTransform(np.column_stack([np.full_like(t, v), t])[None], Hm)[0]
                dd, _ = cKDTree(Q).query(P)
                m = float(np.median(dd))
                if best is None or m < best[1]:
                    best = (float(v), m)
            Q = cv2.perspectiveTransform(np.column_stack([np.full_like(t, xnom), t])[None], Hm)[0]
            dd, _ = cKDTree(Q).query(P)
            out[eid] = dict(homography_rms_px=rms, nominal_X=xnom, median_dist_to_nominal_px=float(np.median(dd)),
                            best_X_mm=best[0], residual_px=best[1])
            print(f"{'undist' if undist else 'raw   '} {eid:26s} H-rms {rms:.2f}px  |dist| to X={xnom:.0f}: "
                  f"{np.median(dd):5.2f}px  best X={best[0]:7.1f} mm (res {best[1]:.2f}px)")
    return out


if __name__ == "__main__":
    res = {"raw": run(False), "ref_undistorted": run(True)}
    save_json(res, f"{CACHE}/edgecrops_cart310/_board_homography_check.json")
