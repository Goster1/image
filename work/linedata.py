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


def edge_stratum(e):
    """Stratum for the bootstrap: VP group if any, else 'straight'."""
    g = e.get("vp_group", "__auto__")
    if g == "__auto__":
        d = e.get("direction")
        if d in ("cartX", "cartY", "cartZ") and e.get("cart") in (80, 310):
            g = f"{e['cart']}_{d}"
        elif d == "world_vertical":
            g = "world_vertical"
        else:
            g = None
    return g or "straight"


def stratified_resample(edges, rng):
    """Resample edges with replacement WITHIN each stratum (keeps every VP group populated)."""
    import collections

    groups = collections.defaultdict(list)
    for e in edges:
        groups[edge_stratum(e)].append(e)
    out = []
    q = 0
    for g, lst in groups.items():
        idx = rng.integers(0, len(lst), len(lst))
        for i in idx:
            out.append(dict(lst[i], id=f"{lst[i]['id']}#{q}"))
            q += 1
    return out
