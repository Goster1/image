"""Render the initial-calibration cart-310 cuboid model on the mean colour image (guide for edge work).

Output: work/cache/edgecrops_cart310/_model_overlay.png (+ zoomed parts)
"""
import cv2
import numpy as np

from common import CACHE, SHELF_Z, CART_W, CART_D, FLOOR_Z, load_json, project

CART = 310
OUT = f"{CACHE}/edgecrops_cart310"


def load_pose(cart=CART):
    ic = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(ic["K"], float)
    dist = np.array(ic["dist"], float)
    p = np.array(ic["poses"][str(cart)], float)
    return K, dist, p[:3], p[3:]


def proj_line(P0, P1, K, dist, rv, tv, n=60):
    t = np.linspace(0, 1, n)[:, None]
    X = P0[None] * (1 - t) + P1[None] * t
    return project(X, K, dist, rv, tv)


def model_segments():
    segs = []
    zs = {"top": 0.0, **{k: v for k, v in SHELF_Z.items()}, "floor": FLOOR_Z}
    for name, z in zs.items():
        c = [(0, 0), (CART_W, 0), (CART_W, CART_D), (0, CART_D)]
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            segs.append((f"{name}", np.array([a[0], a[1], z]), np.array([b[0], b[1], z])))
    for (x, y) in [(0, 0), (CART_W, 0), (CART_W, CART_D), (0, CART_D)]:
        segs.append(("post", np.array([x, y, 0.0]), np.array([x, y, FLOOR_Z])))
    return segs


def main():
    K, dist, rv, tv = load_pose()
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    cols = {"top": (0, 0, 255), "A": (0, 165, 255), "B": (0, 255, 255), "C": (0, 255, 0), "D": (255, 255, 0),
            "E": (255, 0, 0), "floor": (255, 0, 255), "post": (255, 255, 255)}
    for name, a, b in model_segments():
        uv = proj_line(a, b, K, dist, rv, tv)
        cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], False, cols[name], 1, cv2.LINE_AA, shift=2)
    # corner labels
    for (x, y, lab) in [(0, 0, "X0Y0"), (CART_W, 0, "X1600Y0"), (CART_W, CART_D, "X1600Y450"), (0, CART_D, "X0Y450")]:
        uv = project(np.array([[x, y, 0.0]]), K, dist, rv, tv)[0]
        cv2.putText(img, lab, (int(uv[0]) + 4, int(uv[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(f"{OUT}/_model_overlay.png", img)
    crop = img[0:1080, 100:950]
    cv2.imwrite(f"{OUT}/_model_overlay_crop.png", crop)


if __name__ == "__main__":
    main()
