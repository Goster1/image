"""Adds to results/detections.json what the main fit uses besides the corners: the visible sides of partly hidden
stickers (traced sub-pixel points in the 7-frame mean image + the exact 3D line on the sticker from the drawing),
and the per-sticker side validity/line fits. With this file the main (combined) fit can be repeated from results/
(corners + sides) and results/edges.json (structural edges)."""
import numpy as np

from common import CACHE, RESULTS, load_json, save_json
from markerdata import load_markers, marker_side_edges

D = load_json(f"{RESULTS}/detections.json")
MF = load_json(f"{CACHE}/markers_final.json")
by = {(m["cart"], m["id"]): m for m in MF}
for m in D["markers"]:
    src = by[(m["cart"], m["id"])]
    m["side_valid"] = src.get("side_valid")
    m["side_lines"] = src.get("side_lines")
M = load_markers()
S = marker_side_edges(M, only_partial=True)
D["sticker_sides_used"] = [dict(id=e["id"], cart=e["cart"], sticker_id=M[e["sticker_mi"]]["id"], side_index=int(e["id"][-1]),
                                corners_joined=[int(e["id"][-1]), (int(e["id"][-1]) + 1) % 4],
                                model_line_mm=e["model_line"], points_px=np.asarray(e["points"]).round(3).tolist(),
                                image="7-frame mean (jitter compensated, same frame as corners_px)") for e in S]
D["sticker_sides_note"] = ("Side j of a sticker joins corners j and j+1 (OpenCV corner order). A side is used only when at least "
                           "one of its end corners is hidden (not in the corner observations) and the side itself is visible. "
                           "Residual = distance of the traced points from the projected exact 3D line (drawing geometry).")
save_json(D, f"{RESULTS}/detections.json")
print("sides used", len(S))
