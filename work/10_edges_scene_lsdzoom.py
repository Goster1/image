"""Scene region: zoomed crop with the LSD segments (outside the carts) drawn and numbered.

usage: python3 10_edges_scene_lsdzoom.py x0 y0 x1 y1 scale name [minlen]
writes work/cache/edgecrops_scene/lsd_<name>.png"""
import sys

import cv2
import numpy as np

import edgesscene_lib as L

x0, y0, x1, y1 = [int(v) for v in sys.argv[1:5]]
sc = float(sys.argv[5])
name = sys.argv[6]
mn = float(sys.argv[7]) if len(sys.argv) > 7 else 20
S = np.load(f"{L.CACHE}/edges_scene_lsd.npy")
img = L.color_image()[y0:y1, x0:x1]
z = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
rng = np.random.default_rng(1)
for i, s in enumerate(S):
    if s[4] < mn:
        continue
    mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    if not (x0 <= mx <= x1 and y0 <= my <= y1):
        continue
    c = tuple(int(v) for v in rng.integers(80, 256, 3))
    a = (int((s[0] - x0) * sc), int((s[1] - y0) * sc))
    b = (int((s[2] - x0) * sc), int((s[3] - y0) * sc))
    cv2.line(z, a, b, c, 1, cv2.LINE_AA)
    cv2.putText(z, str(i), ((a[0] + b[0]) // 2 + 3, (a[1] + b[1]) // 2), cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1)
for gx in range((x0 // 50 + 1) * 50, x1, 50):
    cv2.putText(z, str(gx), (int((gx - x0) * sc), 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
    cv2.line(z, (int((gx - x0) * sc), 14), (int((gx - x0) * sc), 22), (0, 255, 255), 1)
for gy in range((y0 // 50 + 1) * 50, y1, 50):
    cv2.putText(z, str(gy), (2, int((gy - y0) * sc) - 2), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
    cv2.line(z, (0, int((gy - y0) * sc)), (10, int((gy - y0) * sc)), (0, 255, 255), 1)
cv2.imwrite(f"{L.CROPDIR}/lsd_{name}.png", z)
print(f"{L.CROPDIR}/lsd_{name}.png", z.shape)
