"""Merge the reviewed region edge files into results/edges.json and render results/edges_all.png.

results/edges.json = {"description", "image", "point_convention", "regions": [region blocks as produced by
10_edges_*.py + 12_review_edges_*.py + 13_edges_dedupe.py]}. Every edge keeps its id, physical description,
class, cart, straight_3d, direction, model_line, sub-pixel points, verification flags and review notes.
Edges with verified == "no" are kept for traceability but are not used by any method.
"""
import cv2
import numpy as np

from common import CACHE, RESULTS, load_json, save_json

regions = []
for r in ("cart310", "cart80", "scene"):
    d = load_json(f"{CACHE}/edges_{r}.json")
    d["region"] = r
    regions.append(d)
out = {
    "description": "Structural straight edges found and sub-pixel traced in the jitter-compensated mean of the 7 stills; "
                   "each edge visually verified on zoomed crops (crop paths per edge) and independently reviewed. "
                   "Used by the plumb-line, vanishing-point, cuboid and combined methods (see REPORT.md).",
    "image": "mean of data/stills/*.jpg after per-frame jitter compensation (work/03_mean_image.py); pixel coords, "
             "OpenCV convention (centre of the top-left pixel = (0,0))",
    "usage": "straight_3d==true and verified!='no' -> used for straightness; direction/cart -> vanishing-point groups "
             "(end-board edges excluded from VP groups, see REPORT.md); model_line.exact==true -> exact cuboid line",
    "regions": regions,
}
save_json(out, f"{RESULTS}/edges.json")

col = {"cartX": (0, 0, 255), "cartY": (0, 200, 0), "cartZ": (255, 0, 0), "world_vertical": (255, 0, 255),
       "floor_plane": (0, 215, 255), "horizontal_other": (255, 255, 0), "unknown": (180, 180, 180)}


def render(blocks, fn, labels=False):
    """Edges on the mean image; colour = direction class, grey = not used; labels = short edge id + '*' when the
    edge is used for straightness only (end boards; not in a vanishing-point group)."""
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    n_used = 0
    for blk in blocks:
        for e in blk["edges"]:
            P = np.array(e["points"], float)
            if len(P) < 2:
                continue
            used = e.get("straight_3d", False) and e.get("verified", "yes") != "no"
            n_used += used
            c = col.get(e["direction"], (180, 180, 180)) if used else (90, 90, 90)
            cv2.polylines(img, [np.round(P * 4).astype(np.int32)], False, c, 2 if used else 1, cv2.LINE_AA, shift=2)
            if labels:
                m = P[len(P) // 2]
                t = e["id"].replace("cart310_", "").replace("cart80_", "").replace("scene_", "") + ("*" if used and "board" in e["id"] else "")
                t = t if used else t + " (not used)"
                cv2.putText(img, t, (int(m[0]) + 4, int(m[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, t, (int(m[0]) + 4, int(m[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    y = 30
    for k, c in col.items():
        cv2.putText(img, k, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
        y += 28
    cv2.putText(img, "grey = not used (bent / duplicate / rejected)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
    if labels:
        cv2.putText(img, "* = end board: straightness only (no vanishing point)", (20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.imwrite(fn, img)
    return n_used


n_used = render(regions, f"{RESULTS}/edges_all.png")
for blk in regions:  # per-region views with labels (final status after review and de-duplication)
    render([blk], f"{RESULTS}/edges_{blk['region']}.png", labels=True)
print("regions", len(regions), "edges used", n_used)
