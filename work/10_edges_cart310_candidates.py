"""Automatic edge candidates for region cart310 (LSD on the mean image), classified by the
vanishing direction of the initial calibration (cart X / Y / Z).  Writes a labelled overlay
for visual inspection and work/cache/edgecrops_cart310/_lsd_candidates.json.
"""
import cv2
import numpy as np

from common import CACHE, load_json, undistort_points, save_json, rodrigues

OUT = f"{CACHE}/edgecrops_cart310"
REGION = np.array([[150, 0], [900, 0], [900, 1010], [150, 1010]], float)


def calib():
    ic = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(ic["K"], float)
    dist = np.array(ic["dist"], float)
    p = np.array(ic["poses"]["310"], float)
    return K, dist, rodrigues(p[:3]), p[3:]


def vp_dirs(R):
    """normalised-image vanishing points (homogeneous) of cart X, Y, Z directions"""
    return {k: R @ v for k, v in zip("XYZ", np.eye(3))}


def classify(seg, K, dist, R, tol_deg=4.0):
    a, b = seg[:2], seg[2:]
    n = undistort_points(np.array([a, b]), K, dist)
    pa, pb = np.r_[n[0], 1.0], np.r_[n[1], 1.0]
    m = 0.5 * (pa + pb)
    d = pb - pa
    d = d[:2] / np.linalg.norm(d[:2])
    best, bang = None, 99
    angs = {}
    for k, v in vp_dirs(R).items():
        # direction from midpoint towards the vanishing point (in normalised plane)
        if abs(v[2]) > 1e-9:
            w = v[:2] / v[2] - m[:2]
        else:
            w = v[:2]
        w = w / np.linalg.norm(w)
        ang = np.degrees(np.arccos(min(1, abs(d @ w))))
        angs[k] = ang
        if ang < bang:
            best, bang = k, ang
    return (best if bang < tol_deg else "?"), angs


def main():
    img = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float32)
    g = np.clip(img, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(g)[0].reshape(-1, 4)
    K, dist, R, t = calib()
    keep = []
    for s in lines:
        L = np.hypot(s[2] - s[0], s[3] - s[1])
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        if L < 20 or not (150 <= mx <= 900 and my <= 1010):
            continue
        c, angs = classify(s, K, dist, R)
        keep.append(dict(seg=s.tolist(), len=float(L), cls=c, angs=angs))
    print(len(keep), "segments; by class:", {k: sum(1 for q in keep if q["cls"] == k) for k in "XYZ?"})
    col = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    colors = {"X": (0, 0, 255), "Y": (0, 200, 0), "Z": (255, 0, 0), "?": (0, 255, 255)}
    order = np.argsort([-q["len"] for q in keep])
    keep = [keep[i] for i in order]
    for i, q in enumerate(keep):
        q["idx"] = i
        s = np.array(q["seg"])
        cv2.line(col, (int(round(s[0] * 4)), int(round(s[1] * 4))), (int(round(s[2] * 4)), int(round(s[3] * 4))),
                 colors[q["cls"]], 1, cv2.LINE_AA, shift=2)
    save_json(keep, f"{OUT}/_lsd_candidates.json")
    cv2.imwrite(f"{OUT}/_lsd_candidates.png", col)
    # zoomed tiles with index labels
    for ti, (x0, y0) in enumerate([(150, 0), (450, 0), (150, 330), (450, 330), (300, 660), (550, 660)]):
        tile = col[y0:y0 + 350, x0:x0 + 350].copy()
        tile = cv2.resize(tile, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        for q in keep:
            if q["len"] < 30:
                continue
            s = np.array(q["seg"])
            m = (s[:2] + s[2:]) / 2
            if x0 <= m[0] < x0 + 350 and y0 <= m[1] < y0 + 350:
                p = ((m - [x0, y0]) * 2.6).astype(int)
                cv2.putText(tile, str(q["idx"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(tile, str(q["idx"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[q["cls"]], 1)
        cv2.imwrite(f"{OUT}/_lsd_tile{ti}.png", tile)


if __name__ == "__main__":
    main()
