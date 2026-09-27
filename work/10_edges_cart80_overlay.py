"""Cart 80: project the drawing cuboid (top plate, shelf surfaces, floor, corner verticals) with the
initial calibration onto the colour mean image, as a guide for identifying the physical edges.
Output: work/cache/edgecrops_cart80/overlay_model.png (+ zoomed halves)."""
import cv2
import numpy as np

from common import CACHE, load_json, project, SHELF_Z, CART_W, CART_D, FLOOR_Z

OUT = f"{CACHE}/edgecrops_cart80"


def calib80():
    ic = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(ic["K"], float)
    dist = np.array(ic["dist"], float)
    p = np.array(ic["poses"]["80"], float)
    return K, dist, p[:3], p[3:]


def proj_line(K, dist, rv, tv, A, B, n=60):
    t = np.linspace(0, 1, n)[:, None]
    P = np.asarray(A, float)[None] * (1 - t) + np.asarray(B, float)[None] * t
    return project(P, K, dist, rv, tv)


def main():
    K, dist, rv, tv = calib80()
    im = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    levels = [("top", 0.0)] + [(k, v) for k, v in SHELF_Z.items()] + [("floor", FLOOR_Z)]
    cols = {"top": (0, 0, 255), "A": (0, 128, 255), "B": (0, 255, 255), "C": (0, 255, 0), "D": (255, 255, 0),
            "E": (255, 0, 255), "floor": (255, 0, 0)}
    for name, z in levels:
        corners = [(0, 0, z), (CART_W, 0, z), (CART_W, CART_D, z), (0, CART_D, z), (0, 0, z)]
        for a, b in zip(corners[:-1], corners[1:]):
            uv = proj_line(K, dist, rv, tv, a, b)
            cv2.polylines(im, [np.round(uv * 4).astype(np.int32)], False, cols[name], 1, cv2.LINE_AA, shift=2)
        uv = project(np.array([[0, 0, z]]), K, dist, rv, tv)[0]
        cv2.putText(im, name, (int(uv[0]) - 30, int(uv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cols[name], 1)
    for x in (0, CART_W):
        for y in (0, CART_D):
            uv = proj_line(K, dist, rv, tv, (x, y, 0), (x, y, FLOOR_Z))
            cv2.polylines(im, [np.round(uv * 4).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA, shift=2)
    # axis labels
    for lab, P in [("X0Y0", (0, 0, 0)), ("X1600Y0", (CART_W, 0, 0)), ("X0Y450", (0, CART_D, 0)),
                   ("X1600Y450", (CART_W, CART_D, 0))]:
        uv = project(np.array([P]), K, dist, rv, tv)[0]
        cv2.putText(im, lab, (int(uv[0]) + 3, int(uv[1]) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(f"{OUT}/overlay_model_full.png", im)
    crop = im[0:1000, 1080:1760]
    cv2.imwrite(f"{OUT}/overlay_model.png", crop)
    for i, (y0, y1) in enumerate([(0, 360), (320, 680), (640, 1000)]):
        c = cv2.resize(im[y0:y1, 1080:1760], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{OUT}/overlay_model_z{i}.png", c)


if __name__ == "__main__":
    main()
