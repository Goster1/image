"""Normalisation of per-method result objects to the lens_result.json shape (standard uncertainty keys, uniform
residual definition, mapping block with the three regions, data/geometry fields)."""
import numpy as np
from scipy.optimize import least_squares

from common import CACHE, fold_margin, load_json, project

# standard key -> candidate keys in the method files, in order of preference (cluster/bootstrap/incl. systematics
# first, formal covariance last). native_k* of non-Brown models are NOT mapped (different coefficients).
STD_KEYS = {  # formal (Gauss-Newton) covariances are NOT used as standard keys: they underestimate the error 9-15x
    "fx_px_1sigma": ["fx_px_1sigma", "f_1sigma_incl_geometry", "f_1sigma_bootstrap", "f_1sigma", "fx_1sigma", "fx_px_1sigma_bootstrap_stickers",
                     "f_1sigma_robust", "native_f_1sigma"],
    "fy_px_1sigma": ["fy_px_1sigma", "f_1sigma_incl_geometry", "f_1sigma_bootstrap", "f_1sigma", "fy_1sigma", "fx_px_1sigma_bootstrap_stickers",
                     "f_1sigma_robust", "native_f_1sigma"],
    "cx_px_1sigma": ["cx_px_1sigma", "cx_1sigma", "native_cx_1sigma"],
    "cy_px_1sigma": ["cy_px_1sigma", "cy_1sigma", "cy_1sigma_robust", "native_cy_1sigma"],
    "k1_1sigma": ["k1_1sigma_bootstrap", "k1_1sigma"],
    "k2_1sigma": ["k2_1sigma_bootstrap", "k2_1sigma"],
    "p1_1sigma": ["p1_1sigma"],
    "p2_1sigma": ["p2_1sigma"],
    "k3_1sigma": ["k3_1sigma"],
}
EDGE_ONLY_WORDS = ("lines_joint", "plumb", "vanishing")

_STICK = None


def sticker_rms_ls(K, d):
    """RMS / max of the 62 valid sticker corners with the drawing geometry, one least-squares pose per cart with the
    given intrinsics fixed (the definition of rms_reprojection_error_px in lens_result.json)."""
    global _STICK
    if _STICK is None:
        from markerdata import load_markers, to_points
        M = load_markers()
        init = load_json(f"{CACHE}/initial_calib.json")
        _STICK = []
        for c in (80, 310):
            pts = to_points(M, sel=lambda m, c=c: m["cart"] == c)
            _STICK.append((np.array([p["X"] for p in pts]), np.array([p["uv"] for p in pts]), np.array(init["poses"][str(c)])))
    res = []
    for X, uv, p0 in _STICK:
        r = least_squares(lambda pz: (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), p0)
        res.append(r.fun.reshape(-1, 2))
    res = np.vstack(res)
    e = np.hypot(res[:, 0], res[:, 1])
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


def std_uncertainty(u, method=""):
    u = dict(u or {})
    out, src = {}, {}
    edge_only = any(w in method for w in EDGE_ONLY_WORDS)
    for k, cands in STD_KEYS.items():
        out[k] = None
        for c in cands:
            v = u.get(c)
            if isinstance(v, (int, float)) and np.isfinite(v):
                out[k] = float(v)
                src[k] = c
                break
    if method.startswith("plumb"):  # plumb-line does not estimate f (taken from elsewhere)
        out["fx_px_1sigma"] = out["fy_px_1sigma"] = None
        src.pop("fx_px_1sigma", None)
        src.pop("fy_px_1sigma", None)
    out["standard_keys_from"] = src
    out["method_native"] = {k: v for k, v in u.items() if k not in out}
    return out, edge_only


def normalise(obj, method_id, K=None, d=None):
    """Return a copy of a method object in the lens_result.json shape (+ method_id, fold flag, uniform RMS)."""
    o = dict(obj)
    K = np.array(o["camera_matrix"], float) if K is None else K
    d = np.array(o["dist_coeffs"], float) if d is None else d
    o["method_id"] = method_id
    unc, edge_only = std_uncertainty(o.get("uncertainty"), o.get("method", ""))
    o["uncertainty"] = unc
    rms, mx = sticker_rms_ls(K, d)
    if o.get("rms_reprojection_error_px") is not None and abs(o["rms_reprojection_error_px"] - rms) > 1e-6:
        o["rms_method_internal"] = o["rms_reprojection_error_px"]
    o["rms_reprojection_error_px"] = rms
    o["rms_definition"] = ("62 valid sticker corners, drawing geometry, one least-squares pose per cart with these intrinsics "
                           "fixed (same definition as lens_result.json); max %.2f px" % mx)
    mu = o.get("mapping_uncertainty_px")
    if not isinstance(mu, dict) or any(not isinstance(mu.get(k), (int, float)) for k in ("centre", "cart_band", "corners")):
        o["mapping_uncertainty_px"] = {k: (mu.get(k) if isinstance(mu, dict) and isinstance(mu.get(k), (int, float)) else None)
                                       for k in ("centre", "cart_band", "corners")}
        o.setdefault("mapping_uncertainty_note", "not evaluated (or not meaningful) for this method; see REPORT.md")
    else:
        o["mapping_uncertainty_px"] = {k: float(mu[k]) for k in ("centre", "cart_band", "corners")}
        extra = {k: v for k, v in mu.items() if k not in ("centre", "cart_band", "corners")}
        if extra:
            o["mapping_uncertainty_details"] = extra
    if not o.get("data_used"):
        o["data_used"] = ("61 verified structural edges (results/edges.json), no dimensions" if edge_only else
                          "see REPORT.md, method table (ch. 6); detections in results/detections.json")
    if not o.get("geometry_assumptions"):
        o["geometry_assumptions"] = ("no dimensions used (straightness, parallelism, perpendicularity of the cart axes only)" if edge_only else
                                     "drawing dimensions and sticker positions exactly as specified (not fitted)")
    fm = fold_margin(K, d)
    mono = monotonic_to_corners(K, d)
    o["radial_map_folds_inside_image"] = bool(fm <= 0 or mono < 0.02)
    o["fold_margin"] = float(fm) if fm < 1.0 else None  # large values only mean 'monotonic far beyond the image'
    o["min_radial_slope_to_corners"] = mono
    notes = []
    if mono < 0.05:
        notes.append("radial map (nearly) flat or folding inside the image (min slope %.3f)" % mono)
    if abs(K[0, 2] - 959.5) > 150 or K[1, 2] < 300:
        notes.append("principal point far from all other estimates")
    if notes:
        o["plausibility_note"] = "local minimum / not a credible lens: " + "; ".join(notes)
    elif 0 < fm < 0.05:
        o["fit_constraint_note"] = "the fit is held by the fold barrier (its unconstrained optimum would fold inside the image)"
    return o


def monotonic_to_corners(K, d, n=40000):
    """Minimum relative slope d r_d / d r of the radial distortion map between the centre and the undistorted radius of
    the farthest image corner (the corner's pixel radius is a DISTORTED radius). -1 if the map never reaches it
    (folds before the corner); values near 0 mean a nearly flat, ill-conditioned mapping inside the image."""
    cx, cy, f = K[0, 2], K[1, 2], K[0, 0]
    rc = max(np.hypot(x - cx, y - cy) for x in (0, 1919) for y in (0, 1079)) / f
    r = np.linspace(0, 4.0, n)
    r2 = r * r
    rd = r * (1 + d[0] * r2 + d[1] * r2 ** 2 + (d[4] if len(d) > 4 else 0) * r2 ** 3)
    sl = np.diff(rd) / np.diff(r)
    run = np.maximum.accumulate(rd)
    hit = np.nonzero(run >= rc)[0]
    if not len(hit):
        return -1.0
    j = hit[0]
    return float(np.min(sl[: max(j, 1)]))
