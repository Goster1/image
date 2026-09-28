"""Stage 1: plain ArUco detection (several detector settings, all 7 stills).

Keeps only ids that can exist on the two carts (0..8, 92, 322); assigns shared ids to a cart
by image half (left cart = 310 / unique 322, right cart = 80 / unique 92; verified later by the
pose fit). Writes work/cache/markers_initial.json with per-frame corners of every detection.
"""
import cv2
import numpy as np

from common import CACHE, still_paths, load_gray, save_json

VALID = set(range(9)) | {92, 322}
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)


def detector(**kw):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    for k, v in kw.items():
        setattr(p, k, v)
    return cv2.aruco.ArucoDetector(D, p)


CONFIGS = {
    "default": (detector(), 1.0, False),
    "clahe": (detector(), 1.0, True),
    "clahe_persp": (detector(minMarkerPerimeterRate=0.01, polygonalApproxAccuracyRate=0.1,
                             perspectiveRemovePixelPerCell=8, perspectiveRemoveIgnoredMarginPerCell=0.2,
                             maxErroneousBitsInBorderRate=0.5, errorCorrectionRate=0.9), 1.0, True),
    "up2_persp": (detector(minMarkerPerimeterRate=0.01, polygonalApproxAccuracyRate=0.1,
                           perspectiveRemovePixelPerCell=8, perspectiveRemoveIgnoredMarginPerCell=0.2,
                           maxErroneousBitsInBorderRate=0.5, errorCorrectionRate=0.9), 2.0, False),
}


def cart_of(mid, centre):
    if mid == 322:
        return 310
    if mid == 92:
        return 80
    return 310 if centre[0] < 960 else 80


def main():
    out = []
    clahe = cv2.createCLAHE(3.0, (8, 8))
    for fi, path in enumerate(still_paths()):
        g = load_gray(path)
        for name, (det, sc, use_clahe) in CONFIGS.items():
            img = clahe.apply(g) if use_clahe else g
            if sc != 1.0:
                img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = det.detectMarkers(img)
            if ids is None:
                continue
            for mid, c in zip(ids.ravel(), corners):
                mid = int(mid)
                if mid not in VALID:
                    continue
                c = c[0].astype(float)
                if sc != 1.0:
                    c = (c + 0.5) / sc - 0.5
                out.append(dict(frame=fi, file=path.split("/")[-1], config=name, id=mid,
                                cart=cart_of(mid, c.mean(0)), corners=c.tolist()))
    save_json(out, f"{CACHE}/markers_initial.json")
    keys = sorted({(d["cart"], d["id"]) for d in out})
    for k in keys:
        ds = [d for d in out if (d["cart"], d["id"]) == k]
        cfgs = sorted({d["config"] for d in ds})
        c = np.array([d["corners"] for d in ds if d["config"] == cfgs[0]])
        print(k, len(ds), cfgs, np.round(c.mean((0, 1)), 1), "frame-std px", np.round(c.std(0).mean(), 3))


if __name__ == "__main__":
    main()
