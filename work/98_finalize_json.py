"""Post-processing of results/lens_result.json (explanatory fields, residual breakdown) + results/lens_alternatives.json.

Run after 95_final.py."""
import numpy as np
from scipy.optimize import least_squares

from common import CACHE, RESULTS, load_json, marker_corners_3d, project, save_json
from markerdata import load_markers, to_points
from resultnorm import normalise

L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"])
d = np.array(L["dist_coeffs"])
f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
U = L["uncertainty"]
init = load_json(f"{CACHE}/initial_calib.json")
M = load_markers()
DET = load_json(f"{RESULTS}/detections.json")["markers"]


def pose_fit(X, uv, p0):
    r = least_squares(lambda pz: (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), p0)
    return r.fun.reshape(-1, 2), r.x


# ---- residuals of the 62 sticker corners: one LS pose per cart (7-frame mean) ----
res, lab = [], []
for c in (80, 310):
    pts = to_points(M, sel=lambda m, c=c: m["cart"] == c)
    e, _ = pose_fit(np.array([p["X"] for p in pts]), np.array([p["uv"] for p in pts]), np.array(init["poses"][str(c)]))
    res.append(e)
    lab += [(c, M[p["mi"]]["id"], M[p["mi"]]["row"]) for p in pts]
res = np.vstack(res)
en = np.hypot(res[:, 0], res[:, 1])


def rm(sel):
    return dict(rms_px=float(np.sqrt(np.mean(en[sel] ** 2))), max_px=float(en[sel].max()), n_corners=int(sel.sum()))


lab_a = np.array([f"{c}:{i}" for c, i, _ in lab])
cart_a = np.array([c for c, _, _ in lab])
row_a = np.array([r for _, _, r in lab])
per_sticker = {k: rm(lab_a == k) for k in dict.fromkeys(lab_a)}
per_cart = {str(c): rm(cart_a == c) for c in (80, 310)}
per_row = {r: rm(row_a == r) for r in ("top", "A", "B", "C")}
# ---- per photo: corners of each still (still coordinates, no jitter compensation), pose refit per photo and cart ----
per_photo = {}
nfr = len(DET[0]["per_frame_corners_still_px"])
for fr in range(nfr):
    ee = []
    for c in (80, 310):
        X, uv = [], []
        for m in DET:
            if m["cart"] != c:
                continue
            X3 = np.array(m["corners_3d_mm"])
            for j in range(4):
                q = m["per_frame_corners_still_px"][fr][j]
                if m["corner_valid"][j] and q is not None and np.all(np.isfinite(np.array(q, float))):
                    X.append(X3[j])
                    uv.append(q)
        e, _ = pose_fit(np.array(X), np.array(uv), np.array(init["poses"][str(c)]))
        ee.append(np.hypot(e[:, 0], e[:, 1]))
    ee = np.concatenate(ee)
    per_photo[f"frame{fr}"] = dict(rms_px=float(np.sqrt(np.mean(ee ** 2))), max_px=float(ee.max()), n_corners=int(len(ee)))

fit = load_json(f"{CACHE}/combined_fit.json")
br = fit["summary"]["k1k2"]["block_rms"]
L["rms_reprojection_error_px"] = float(np.sqrt(np.mean(en ** 2)))
L["rms_details"] = {
    "definition": "62 valid sticker corners, drawing geometry, one least-squares pose per cart with the main intrinsics fixed; "
                  "dominated by the sticker-geometry mismatch (REPORT.md ch. 5), not by detection noise (~0.1-0.2 px)",
    "sticker_corners_px": float(np.sqrt(np.mean(en ** 2))),
    "sticker_corners_max_px": float(en.max()),
    "per_cart": per_cart,
    "per_row": per_row,
    "per_photo": per_photo,
    "per_photo_note": "each still separately (its own corner positions, pose refitted per still and cart)",
    "per_sticker": per_sticker,
    "in_joint_fit_block_rms_px": {"sticker_corners_per_coordinate": br["M"][0], "sticker_sides": br["L"][0],
                                  "vanishing_point_groups": br["V"][0], "edge_straightness": br["S"][0]},
    "for_reference_sticker_only_fits_rms_px": "5.0-5.9 (markers_only_* in lens_by_method.json)",
}
L["model"] = ("OpenCV pinhole + Brown-Conrady, fx = fy, principal point free, radial k1, k2; p1 = p2 = k3 = 0 fixed "
              "(k3 only consistent with the data, p1/p2 and fx != fy not determined - REPORT.md ch. 7)")
L["data_used"] = ("all 7 stills (jitter-compensated 7-frame mean); 20 ArUco stickers (ids 0-8 + 92 on cart 80, 0-8 + 322 on "
                  "cart 310): 62 valid corners + 13 traced sides of partly hidden stickers (results/detections.json); "
                  "61 verified straight structural edges (cart 310: 18, cart 80: 17, scene incl. floor lines: 26; "
                  "results/edges.json) as straightness + vanishing-point groups: cart X (shelf lips) and cart Z (posts) of "
                  "both carts, scene verticals, 2 floor-line groups. No cart-Y (depth) group: the only depth-direction "
                  "edges are end-board edges, which are not parallel to the cart axes and are used for straightness only")
L["geometry_assumptions"] = ("Rozmery vozíka, výšky políc a polohy nálepiek sú použité presne podľa spec/cart-marker-layout.json "
                             "(nie sú fitované). Dáta s nimi nie sú úplne konzistentné: top-nálepky vychádzajú voči nálepkám "
                             "na policiach ~+60..+80 mm vyššie, koncové dosky naklonené 4-13 deg, RMS 4.9-5.1 px aj pre "
                             "všeobecnú projektívnu kameru 3x4 (metrický fit 5.3 px) - opísané ako pozorovanie v REPORT.md "
                             "kap. 5 (kde, koľko px). Hrany nepoužívajú žiadne rozmery, len priamosť, rovnobežnosť "
                             "a kolmosť osí vozíka.")
S = fit["summary"]
fs = S.get("k1k2_floor_straight_only", {}).get("x")
L["determination"] = {
    "determined": [
        "radial distortion profile (k1, k2 jointly, correlation %.2f; in pixel units k1/f^2 = %.3e px^-2, k2/f^4 = %.3e px^-4)"
        % (L["uncertainty_details"]["correlation_bootstrap"][3][4], d[0] / f ** 2, d[1] / f ** 4),
        "cx = %.0f +- %.0f px" % (cx, U["cx_px_1sigma"]),
        "f = %.0f +- %.0f px (%.1f %%): mainly from the vanishing points of the edges (+ sticker sides at the 80/X0 end); "
        "edges alone give f only fragile" % (f, U["fx_px_1sigma"], 100 * U["fx_px_1sigma"] / f),
    ],
    "consistent_only": [
        "cy = %.0f +- %.0f px: carried by the vanishing-point groups (without them cy %.0f); floor lines as straightness only: "
        "cy %.0f; edge-only fits give 426-509 depending on the floor-line / vertical assumptions"
        % (cy, U["cy_px_1sigma"], dict(zip(S["k1k2_influence_no_vp"]["names"], S["k1k2_influence_no_vp"]["x"]))["cy"], fs[2] if fs else float("nan")),
        "k3 ~ -0.03 +- 0.025 (improvement not significant)",
    ],
    "not_determinable": ["p1, p2 (tangential)", "fx != fy (pixel aspect; edges alone fx 1383 +- 91 vs fy 1426 +- 34; combined fit "
                         "fx/fy - 1 = %+.1f %%)" % (100 * (S["k1k2_fxfy"]["x"][0] / S["k1k2_fxfy"]["x"][1] - 1)),
                         "cause of the sticker-geometry mismatch (top plate vs shelves)"],
}
save_json(L, f"{RESULTS}/lens_result.json")

# ---------- lens_alternatives.json: models / modelling choices the data support about equally well
MAIN_UNC = {k: v for k, v in U.items() if k in ("fx_px_1sigma", "fy_px_1sigma", "cx_px_1sigma", "cy_px_1sigma", "k1_1sigma", "k2_1sigma",
                                                "p1_1sigma", "p2_1sigma", "k3_1sigma")}
SAME = ("not evaluated separately: same data and estimator family as the main result, the main-result uncertainty is quoted "
        "(this alternative is part of its systematic budget)")
ALT = [
    ("method_combined_alternatives.json", "combined_k1k2k3", "k3 free (only consistent with the data)", "alt_brown_k1k2k3"),
    ("method_alt_division_l1l2.json", None, "division model l1, l2 (converted to OpenCV Brown)", None),
    ("method_combined_alternatives.json", "combined_k1k2_rowblocks", "separate variance components for top-plate and shelf sticker corners", "main"),
    ("method_combined_alternatives.json", "combined_k1k2_robust", "stickers with cluster-robust weights (top stickers ~0 weight)", "main"),
    ("method_combined_alternatives.json", "combined_k1k2_common_vertical", "posts of both carts parallel to the scene vertical", "main"),
    ("method_lines_joint.json", None, "edges only, no dimensions (common vertical of both carts)", None),
    ("method_lines_joint_indep_mergedfloor.json", None, "edges only, blue floor line treated as one straight line", None),
]
alts = []
for fn, nm, note, unc_from in ALT:
    j = load_json(f"{CACHE}/{fn}")
    if isinstance(j, list):
        j = [x for x in j if x.get("method") == nm][0]
    j = dict(j)
    if unc_from == "main":
        j["uncertainty"] = dict(MAIN_UNC, note=SAME)
        j["mapping_uncertainty_px"] = dict(L["mapping_uncertainty_px"])
    elif unc_from == "alt_brown_k1k2k3":  # same model, stickers + edges: cluster (jackknife/sandwich) uncertainties
        a = load_json(f"{CACHE}/method_alt_brown_k1k2k3.json")
        j["uncertainty"] = dict(a["uncertainty"], note="cluster jackknife/sandwich of the same model in the model study (43_altmodels)")
        j["mapping_uncertainty_px"] = a["mapping_uncertainty_px"]
    if not j.get("data_used") and j.get("method", "").startswith("combined"):
        j["data_used"] = L["data_used"]
        j["geometry_assumptions"] = "drawing dimensions and sticker positions exactly as specified (not fitted)"
    a = normalise(j, nm or fn[7:-5])
    a["note"] = note
    alts.append(a)
save_json(alts, f"{RESULTS}/lens_alternatives.json")
print("rms stickers", L["rms_reprojection_error_px"], "per cart", {k: round(v["rms_px"], 2) for k, v in per_cart.items()},
      "per photo", [round(v["rms_px"], 2) for v in per_photo.values()], "alternatives", len(alts))
