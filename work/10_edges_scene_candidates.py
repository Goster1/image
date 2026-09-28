"""Scene region: automatic line-segment candidates outside the two carts.

* projects the cart cuboids (initial calibration) -> masks of the carts (dilated)
* LSD on the mean image (cv2.createLineSegmentDetector), segments >= 15 px outside the cart masks
* overview images (numbered segments, cart wireframes) into work/cache/edgecrops_scene/overview_*.png
* segments saved to work/cache/edges_scene_lsd.npy  [x0,y0,x1,y1,len,in_cart]
"""
import os

import cv2
import numpy as np

import edgesscene_lib as L

os.makedirs(L.CROPDIR, exist_ok=True)

img8 = np.clip(L.IMG, 0, 255).astype(np.uint8)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
segs = lsd.detect(img8)[0].reshape(-1, 4)
lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
masks = L.cart_masks(dilate=10)
anymask = masks[80] | masks[310]


def frac_in(seg, m, n=9):
    t = np.linspace(0, 1, n)
    x = np.clip(np.round(seg[0] + t * (seg[2] - seg[0])).astype(int), 0, L.W - 1)
    y = np.clip(np.round(seg[1] + t * (seg[3] - seg[1])).astype(int), 0, L.H - 1)
    return m[y, x].mean()


inc = np.array([frac_in(s, anymask) for s in segs])
sel = (lens >= 15)
out = np.column_stack([segs, lens, inc])[sel]
np.save(f"{L.CACHE}/edges_scene_lsd.npy", out)
print("segments >=15 px:", sel.sum(), " outside carts:", (out[:, 5] < 0.5).sum())

col = L.color_image()
over = col.copy()
# cart wireframes
for cart, c in ((80, (0, 255, 0)), (310, (255, 255, 0))):
    for name, P in L.cart_wireframe(cart):
        cv2.polylines(over, [np.round(P).astype(np.int32)], False, c, 1, cv2.LINE_AA)
cnt = cv2.findContours(anymask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(over, cnt, -1, (0, 128, 255), 2)
for i, s in enumerate(out):
    if s[5] >= 0.5:
        continue
    c = (0, 0, 255) if s[4] >= 40 else (255, 0, 255)
    cv2.line(over, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), c, 1, cv2.LINE_AA)
    if s[4] >= 25:
        cv2.putText(over, str(i), (int((s[0] + s[2]) / 2) + 2, int((s[1] + s[3]) / 2)), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (0, 255, 255), 1)
cv2.imwrite(f"{L.CROPDIR}/overview_lsd.png", over)

# zoomed overview tiles
tiles = {"topleft": (0, 0, 480, 660), "botleft": (0, 620, 640, 1080), "centre_top": (640, 0, 1240, 460),
         "centre_bot": (780, 420, 1240, 1080), "right_top": (1560, 0, 1920, 540), "right_bot": (1480, 500, 1920, 1080)}
for name, (x0, y0, x1, y1) in tiles.items():
    t = over[y0:y1, x0:x1]
    sc = min(3.0, 1500 / max(x1 - x0, y1 - y0))
    cv2.imwrite(f"{L.CROPDIR}/overview_{name}.png", cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(f"{L.CROPDIR}/plain_{name}.png",
                cv2.resize(col[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC))
