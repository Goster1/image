"""Scene region: zoomed crops with a pixel grid (for hand-picking guide polylines).

usage: python3 10_edges_scene_zoom.py x0 y0 x1 y1 [scale] [name]
writes work/cache/edgecrops_scene/zoom_<name>.png (grid lines every 20 px, labels every 100 px)."""
import sys

import cv2
import numpy as np

import edgesscene_lib as L

x0, y0, x1, y1 = [int(v) for v in sys.argv[1:5]]
sc = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
name = sys.argv[6] if len(sys.argv) > 6 else f"{x0}_{y0}_{x1}_{y1}"
img = L.color_image()[y0:y1, x0:x1]
z = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
for gx in range((x0 // 20 + 1) * 20, x1, 20):
    X = int((gx - x0) * sc)
    c = (0, 255, 255) if gx % 100 == 0 else (0, 140, 140)
    cv2.line(z, (X, 0), (X, z.shape[0]), c, 1)
    if gx % 100 == 0:
        cv2.putText(z, str(gx), (X + 2, 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
for gy in range((y0 // 20 + 1) * 20, y1, 20):
    Y = int((gy - y0) * sc)
    c = (0, 255, 255) if gy % 100 == 0 else (0, 140, 140)
    cv2.line(z, (0, Y), (z.shape[1], Y), c, 1)
    if gy % 100 == 0:
        cv2.putText(z, str(gy), (2, Y - 2), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
cv2.imwrite(f"{L.CROPDIR}/zoom_{name}.png", z)
print(f"{L.CROPDIR}/zoom_{name}.png", z.shape)
