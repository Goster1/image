"""Post-processing of results/lens_result.json (explanatory fields) + results/lens_alternatives.json.

Run after 95_final.py (and 96_plots.py for the sticker residuals)."""
import numpy as np
from scipy.optimize import least_squares

from common import CACHE, RESULTS, fold_margin, lens_json, load_json, project, save_json
from markerdata import load_markers, to_points

L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"])
d = np.array(L["dist_coeffs"])
init = load_json(f"{CACHE}/initial_calib.json")
M = load_markers()
res = []
for c in (80, 310):
    pts = to_points(M, sel=lambda m: m["cart"] == c)
    X = np.array([p["X"] for p in pts])
    uv = np.array([p["uv"] for p in pts])
    r = least_squares(lambda pz: (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), np.array(init["poses"][str(c)]))
    res.append(r.fun.reshape(-1, 2))
res = np.vstack(res)
rms_st = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
fit = load_json(f"{CACHE}/combined_fit.json")
br = fit["summary"]["k1k2"]["block_rms"]
L["rms_reprojection_error_px"] = rms_st
L["rms_details"] = {
    "sticker_corners_px": rms_st,
    "sticker_corners_max_px": float(np.max(np.hypot(res[:, 0], res[:, 1]))),
    "sticker_corners_definition": "62 valid corners, drawing geometry, one least-squares pose per cart with the main intrinsics fixed; "
                                  "dominated by the sticker-geometry mismatch (REPORT.md ch. 5), not by noise (~0.1 px)",
    "sticker_corners_in_joint_fit_px_per_point": float(br["M"][0] * np.sqrt(2)),
    "edge_straightness_rms_px": float(br["S"][0]),
    "edge_vanishing_point_rms_px": float(br["V"][0]),
    "sticker_side_lines_rms_px": float(br["L"][0]),
    "for_reference_sticker_only_fit_rms_px": 5.34,
}
L["model"] = ("OpenCV pinhole + Brown-Conrady, fx = fy, principal point free, radial k1, k2; p1 = p2 = k3 = 0 fixed "
              "(not determined by the data: k3 only consistent, p1/p2 and fx != fy not determined - REPORT.md ch. 7)")
L["data_used"] = ("all 7 stills (jitter-compensated 7-frame mean); 20 ArUco stickers (ids 0-8 + 92 on cart 80, 0-8 + 322 on "
                  "cart 310), 62 valid corners + 13 traced sides of partly hidden stickers; 61 verified straight structural "
                  "edges (cart 310: 18, cart 80: 17, scene incl. floor lines: 26) as straightness + vanishing-point groups "
                  "(cart X/Y/Z of both carts, scene verticals, 2 floor-line groups); detections in results/detections.json, "
                  "edges in results/edges.json")
L["geometry_assumptions"] = ("Rozmery vozíka, výšky políc a polohy nálepiek sú použité presne podľa spec/cart-marker-layout.json "
                             "(nie sú fitované). Dáta s nimi nie sú konzistentné (top-nálepky voči policiam ~+60..+80 mm, "
                             "sklony koncových dosiek 4-13 deg, RMS 5.3 px aj pre všeobecnú kameru) - opísané ako pozorovanie "
                             "v REPORT.md kap. 5 (kde, koľko px). Hrany nepoužívajú žiadne rozmery, len priamosť, "
                             "rovnobežnosť a kolmosť osí vozíka.")
L["determination"] = {
    "determined": ["radial distortion profile (k1, k2 jointly; in pixel units k1/f^2 = %.3e, k2/f^4 = %.3e)" % (d[0] / K[0, 0] ** 2, d[1] / K[0, 0] ** 4),
                   "cx (+-12 px)", "f to ~1.7 % (+-25 px), only with stickers + edges together (edges alone: fragile)"],
    "consistent_only": ["cy (426-513 px depending on assumptions about the floor lines; +-35 px)", "k3 ~ -0.02 +- 0.02"],
    "not_determinable": ["p1, p2 (tangential)", "fx != fy (pixel aspect)", "absolute sticker geometry (top vs shelves offset)"],
}
save_json(L, f"{RESULTS}/lens_result.json")

# ---------- lens_alternatives.json: models the data support about equally well
ALT = [
    ("method_combined_alternatives.json", "combined_k1k2k3", "k3 free (only consistent with the data)"),
    ("method_combined_alternatives.json", "combined_k1k2_robust", "same data, stickers with cluster-robust weights (top stickers ~0 weight)"),
    ("method_lines_joint.json", None, "edges only, no dimensions (independent implementation, common vertical of both carts)"),
    ("method_lines_joint_indep_mergedfloor.json", None, "edges only, blue floor line treated as one straight line"),
    ("method_alt_brown_k1k2.json", None, "model study: same model, stickers + edges, non-robust sandwich/jackknife uncertainties"),
    ("method_alt_division_l1l2.json", None, "division model l1,l2 converted to OpenCV Brown"),
]
alts = []
for fn, nm, note in ALT:
    j = load_json(f"{CACHE}/{fn}")
    if isinstance(j, list):
        j = [x for x in j if x.get("method") == nm][0]
    Ka = np.array(j["camera_matrix"])
    da = np.array(j["dist_coeffs"])
    a = lens_json(Ka, da, method=j.get("method"), model=j.get("model"), rms_reprojection_error_px=j.get("rms_reprojection_error_px"),
                  uncertainty=j.get("uncertainty"), mapping_uncertainty_px=j.get("mapping_uncertainty_px"),
                  data_used=j.get("data_used"), geometry_assumptions=j.get("geometry_assumptions"),
                  note=note, fold_margin=fold_margin(Ka, da))
    alts.append(a)
save_json(alts, f"{RESULTS}/lens_alternatives.json")
print("rms stickers", rms_st, "alternatives", len(alts))
