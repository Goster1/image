"""Scene region: review montages of the per-edge verification crops (work/cache/edgecrops_scene/<id>.png).
usage: python3 10_edges_scene_montage.py out_name id_substring [id_substring ...]"""
import sys

import cv2
import numpy as np

import edgesscene_lib as L
from edgesscene_defs import E

out = sys.argv[1]
sel = sys.argv[2:]
ids = [e["id"] for e in E if any(s in e["id"] for s in sel)]
ims = [cv2.imread(f"{L.CROPDIR}/{i}.png") for i in ids]
ims = [i for i in ims if i is not None]
w = max(i.shape[1] for i in ims)
parts = []
for i in ims:
    parts += [np.hstack([i, np.zeros((i.shape[0], w - i.shape[1], 3), np.uint8)]), np.full((6, w, 3), (0, 255, 0), np.uint8)]
cv2.imwrite(f"{L.CROPDIR}/montage_{out}.png", np.vstack(parts))
print(ids)
