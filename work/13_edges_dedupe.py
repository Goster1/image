"""Orchestrator fix after the edge reviews: remove cross-region duplicates (keep one copy).

cart310_floor_tape_blue_upper/lower are the same image edges as scene_blueL_top/bottom (scene review);
the scene copies are kept (used in the floor VP group), the cart-310 copies are set verified='no'.
Idempotent; run after 12_review_edges_*.py.
"""
from common import CACHE, load_json, save_json

fn = f"{CACHE}/edges_cart310.json"
d = load_json(fn)
dup = {"cart310_floor_tape_blue_upper": "scene_blueL_top", "cart310_floor_tape_blue_lower": "scene_blueL_bottom"}
for e in d["edges"]:
    if e["id"] in dup:
        e["verified"] = "no"
        e["duplicate_of"] = dup[e["id"]]
        e.setdefault("verification_note", "")
        if "duplicate" not in e["verification_note"]:
            e["verification_note"] += f" | set to 'no' by 13_edges_dedupe.py: duplicate of {dup[e['id']]} (scene region)"
save_json(d, fn)
print("done")
