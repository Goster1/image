"""Review of stage 11 (markers): zoomed crops of every sticker with the stored corners, the stored
side lines and a fresh sub-pixel re-trace of the four sides on the mean image (for visual checks).

Output: work/cache/review_markers/crop_<cart>_<id>.png and montages review_markers/montage_*.png
Legend: thin yellow = completed quad (corners_px), green line = valid stored side line, red line =
invalid side, cyan dots = good re-traced edge samples, magenta = rejected samples, green circle =
valid corner, red X = invalid (completed) corner, blue dot = ArUco-rule corner, "0" = ArUco corner 0.
"""
import importlib
import os

import cv2
import numpy as np

from common import CACHE, load_json

L = importlib.import_module("11_markers_lib")
OUT = f"{CACHE}/review_markers"
os.makedirs(OUT, exist_ok=True)
S = 6  # zoom


def to_z(P, x0, y0):
    P = np.asarray(P, float).reshape(-1, 2)
    return np.column_stack([(P[:, 0] - x0 + 0.5) * S - 0.5, (P[:, 1] - y0 + 0.5) * S - 0.5])


def ipt(p):
    return (int(round(p[0] * 16)), int(round(p[1] * 16)))


def render(m, color, mean, retrace=True, half=None):
    q = np.array(m["corners_px"], float)
    c = q.mean(0)
    side = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).mean()
    hh = int(round(0.95 * side)) if half is None else half
    x0, y0 = int(c[0]) - hh, int(c[1]) - hh
    crop = color[max(y0, 0):y0 + 2 * hh, max(x0, 0):x0 + 2 * hh]
    pad = np.zeros((2 * hh, 2 * hh, 3), np.uint8)
    pad[max(0, -y0):max(0, -y0) + crop.shape[0], max(0, -x0):max(0, -x0) + crop.shape[1]] = crop
    Z = cv2.resize(pad, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    sh = 4
    # completed quad
    zq = to_z(q, x0, y0)
    cv2.polylines(Z, [np.round(zq * 16).astype(np.int32)], True, (0, 220, 255), 1, cv2.LINE_AA, shift=sh)
    # stored side lines
    for i, sl in enumerate(m["side_lines"]):
        if sl["point"] is None:
            continue
        p = np.array(sl["point"], float)
        d = np.array(sl["direction"], float)
        a = p - d * side * 0.75
        b = p + d * side * 0.75
        za, zb = to_z([a, b], x0, y0)
        col = (0, 200, 0) if m["side_valid"][i] else (0, 0, 255)
        cv2.line(Z, ipt(za), ipt(zb), col, 1, cv2.LINE_AA, shift=sh)
        zm = to_z(p, x0, y0)[0]
        cv2.putText(Z, f"s{i}", (int(zm[0]) + 4, int(zm[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if retrace:
        e = L.edge_quad(mean, q, iters=2)
        for s in e["sides"]:
            tr = s["trace"]
            g = s["quality"]["good"]
            zp = to_z(tr["points"], x0, y0)
            for k, p in enumerate(zp):
                cv2.circle(Z, ipt(p), 2 * 16, (255, 255, 0) if g[k] else (255, 0, 255), -1, cv2.LINE_AA, shift=sh)
    # ArUco-rule corners
    ac = m.get("aruco_corners_px")
    if ac is not None:
        for p in to_z(np.array(ac, float), x0, y0):
            cv2.circle(Z, ipt(p), 3 * 16, (255, 80, 0), -1, cv2.LINE_AA, shift=sh)
    for i, p in enumerate(zq):
        if m["corner_valid"][i]:
            cv2.circle(Z, ipt(p), 7 * 16, (0, 255, 0), 1, cv2.LINE_AA, shift=sh)
        else:
            cv2.drawMarker(Z, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(Z, str(i), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(Z, (0, 0), (Z.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(Z, f"{m['cart']}:{m['id']} {m['role']} rot_k={m['rot_k']} valid={''.join(str(int(v)) for v in m['corner_valid'])}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return Z


def main():
    M = load_json(f"{CACHE}/markers_final.json")
    color = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    mean = L.mean_gray()
    tiles = []
    for m in M:
        Z = render(m, color, mean)
        cv2.imwrite(f"{OUT}/crop_{m['cart']}_{m['id']}.png", Z)
        # also a plain (no overlay) crop for comparison
        tiles.append((f"{m['cart']}_{m['id']}", Z))
    # montages of 4
    for k in range(0, len(tiles), 4):
        grp = tiles[k:k + 4]
        side = 520
        mont = np.zeros((2 * side, 2 * side, 3), np.uint8)
        for j, (n, Z) in enumerate(grp):
            Zs = cv2.resize(Z, (side, side), interpolation=cv2.INTER_AREA) if Z.shape[0] != side else Z
            r, cc = divmod(j, 2)
            mont[r * side:(r + 1) * side, cc * side:(cc + 1) * side] = Zs
        cv2.imwrite(f"{OUT}/montage_{k // 4}.png", mont)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
