"""Stage 3: jitter-compensated mean of the 7 stills (float32 .npy + 8-bit png, gray and colour).

Per-frame jitter (from 02_initial_calib.py): features of frame f are displaced by
dx = ax0 + ax1*(y-540), dy = ay0 + ay1*(y-540) relative to the 7-frame mean. Each frame is
resampled at (x+dx, y+dy) (bicubic) and the frames are averaged. The per-pixel std over the
aligned frames is also saved (noise map used for edge weighting).
"""
import cv2
import numpy as np

from common import CACHE, H, W, load_json, still_paths

d = load_json(f"{CACHE}/initial_calib.json")
fp = np.array(d["frame_jitter"])
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
acc_g, acc_c, stack = 0, 0, []
for f, p in enumerate(still_paths()):
    im = cv2.imread(p).astype(np.float32)
    ax0, ax1, ay0, ay1 = fp[f]
    mx = (xx + (ax0 + ax1 * (yy - H / 2))).astype(np.float32)
    my = (yy + (ay0 + ay1 * (yy - H / 2))).astype(np.float32)
    al = cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    g = 0.114 * al[..., 0] + 0.587 * al[..., 1] + 0.299 * al[..., 2]
    stack.append(g)
    acc_c = acc_c + al
stack = np.array(stack)
mean_g = stack.mean(0)
np.save(f"{CACHE}/mean_aligned_gray.npy", mean_g.astype(np.float32))
np.save(f"{CACHE}/std_aligned_gray.npy", stack.std(0).astype(np.float32))
cv2.imwrite(f"{CACHE}/mean_aligned_gray.png", np.clip(np.round(mean_g), 0, 255).astype(np.uint8))
cv2.imwrite(f"{CACHE}/mean_aligned_color.png", np.clip(np.round(acc_c / len(stack)), 0, 255).astype(np.uint8))
print("per-pixel temporal std (median):", float(np.median(stack.std(0))))
