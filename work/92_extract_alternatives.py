"""Extract labelled alternatives stored inside method files into their own method_*.json (for the comparison
and the systematic part of the uncertainty budget)."""
from common import CACHE, load_json, save_json

j = load_json(f"{CACHE}/method_lines_joint.json")
alt = dict(j["details"]["review"]["alternative_floorline_merged"])
alt["method"] = "lines_joint_indep_mergedfloor (edges only; the two pieces of the blue floor line treated as one straight line)"
save_json(alt, f"{CACHE}/method_lines_joint_indep_mergedfloor.json")
print("ok", alt["camera_matrix"][0][0], alt["camera_matrix"][0][2], alt["camera_matrix"][1][2], alt["dist_coeffs"])
