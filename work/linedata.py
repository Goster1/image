"""Loading traced edges (work/cache/edges_*.json or results/edges.json)."""
from __future__ import annotations

import glob
import os

import numpy as np

from common import CACHE, RESULTS, load_json


def load_edges(min_len=40.0, only_straight=True, verified_only=True, source=None):
    """Return list of dicts: id, region, cart, direction, model_line, points (N x 2), length."""
    files = []
    region_files = [f"{CACHE}/edges_{r}.json" for r in ("cart310", "cart80", "scene") if os.path.exists(f"{CACHE}/edges_{r}.json")]
    if source is not None:
        files = [source]
    elif region_files:  # the reviewed region files are the source of truth
        files = region_files
    else:  # reproduce from the published merge
        files = [f"{RESULTS}/edges.json"]
    out = []
    for fn in files:
        d = load_json(fn)
        blocks = d["regions"] if "regions" in d else [d]
        for blk in blocks:
            for e in blk["edges"]:
                if only_straight and not e.get("straight_3d", False):
                    continue
                if verified_only and e.get("verified", "yes") == "no":
                    continue
                P = np.array(e["points"], float)
                if len(P) < 5:
                    continue
                L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
                if L < min_len:
                    continue
                out.append(dict(id=e["id"], region=blk.get("region", e.get("region")), cart=e.get("cart"),
                                direction=e.get("direction", "unknown"), model_line=e.get("model_line"),
                                what=e.get("what", ""), points=P, length=L, cls=e.get("class")))
    return out
