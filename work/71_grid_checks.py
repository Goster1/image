"""Numerical companion of 70_render_grid.py (main lens + poses of the main combined fit):

  1. drawing model lines vs traced edges: for every structural edge whose 3D line is fully fixed by the drawing
     (model_line with two fixed coordinates), the signed image distance of the traced points from the projected
     model line [px] (median, and at both ends) and the equivalent 3D offset [mm] perpendicular to the viewing ray
  2. floor markings back-projected onto the floor plane of cart 310 (Z = -1800 mm): straightness [mm], width of the
     white tape along its length (left/right and top/bottom edge pairs), angle between the white tape's two directions
     (independent of the fit: the two floor groups have separate direction parameters), angle to the blue line
  3. both carts on one floor: height of cart 80's footprint corners in the cart-310 frame, angle between the Z axes
Writes work/cache/grid_checks.json and prints a summary.
"""
import cv2
import numpy as np

from common import CACHE, RESULTS, load_json, project, save_json, undistort_points

L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"], float)
D = np.array(L["dist_coeffs"], float)
FIT = load_json(f"{CACHE}/combined_fit.json")
X = np.array(FIT["x_main"])
NI = len(FIT["inames"])
POSES = {80: X[NI:NI + 6], 310: X[NI + 6:NI + 12]}
RT = {c: (cv2.Rodrigues(p[:3].reshape(3, 1))[0], p[3:]) for c, p in POSES.items()}
EJ = load_json(f"{RESULTS}/edges.json")
EDGES = {e["id"]: e for b in EJ["regions"] for e in b["edges"]}
FLOOR_Z = -1800.0


def cart_of(eid):
    return 80 if eid.startswith("cart80") else (310 if eid.startswith("cart310") else None)


def signed_dist_to_curve(P, C):
    """signed distance of points P to the polyline C (sign by the curve normal)."""
    out = []
    for p in P:
        d = np.linalg.norm(C - p, axis=1)
        i = int(np.argmin(d))
        j = min(max(i, 1), len(C) - 1)
        t = C[j] - C[j - 1]
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        out.append(float((p - C[i]) @ n))
    return np.array(out)


# ---------------------------------------------------------------- 1. model lines
model = []
for eid, e in EDGES.items():
    ml = e.get("model_line")
    c = cart_of(eid)
    if not ml or c is None or len(ml.get("fixed", {})) < 2 or not (e.get("straight_3d") and e.get("verified", "yes") != "no"):
        continue
    free = ml["free"]
    a, b = ml.get("range_mm", [0, 1600])
    t = np.linspace(a - 0.3 * (b - a), b + 0.3 * (b - a), 800)
    P3 = np.zeros((len(t), 3))
    for k, v in ml["fixed"].items():
        P3[:, "XYZ".index(k)] = v
    P3[:, "XYZ".index(free)] = t
    Cp = project(P3, K, D, rvec=POSES[c][:3], tvec=POSES[c][3:])
    P = np.asarray(e["points"], float)
    sd = signed_dist_to_curve(P, Cp)
    # mm per px at the edge: depth / f
    R, tv = RT[c]
    depth = float(np.median((P3 @ R.T + tv)[:, 2]))
    model.append(dict(id=eid, cart=c, fixed=ml["fixed"], free=free, exact=bool(ml.get("exact")), n=len(P),
                      median_px=float(np.median(sd)), end_a_px=float(np.median(sd[:5])), end_b_px=float(np.median(sd[-5:])),
                      rms_px=float(np.sqrt(np.mean(sd ** 2))), mm_per_px=depth / K[0, 0], median_mm=float(np.median(sd)) * depth / K[0, 0],
                      note=ml.get("note", "")[:120]))

# ---------------------------------------------------------------- 2. floor markings on the floor plane of cart 310
R, tv = RT[310]
Cc = -R.T @ tv  # camera centre in the cart-310 frame


def to_floor(uv, cart=310, z=FLOOR_Z):
    R_, t_ = RT[cart]
    C_ = -R_.T @ t_
    n = undistort_points(np.asarray(uv, float), K, D)
    rays = np.column_stack([n, np.ones(len(n))]) @ R_  # camera -> cart frame (R^T applied as row @ R)
    s = (z - C_[2]) / rays[:, 2]
    return C_[None] + s[:, None] * rays


def line2d(Q):
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c)
    d = vt[0]
    r = (Q - c) @ np.array([-d[1], d[0]])
    return c, d, r


floor = {}
fl = {}
for eid in ("scene_tapeV_left", "scene_tapeV_right", "scene_tapeH_top", "scene_tapeH_bottom", "scene_blueH_top", "scene_blueL_top", "scene_blueL_bottom"):
    if eid not in EDGES:
        continue
    Q = to_floor(EDGES[eid]["points"])[:, :2]
    c, d, r = line2d(Q)
    fl[eid] = (Q, c, d)
    floor[eid] = dict(n=len(Q), length_mm=float(np.ptp((Q - c) @ d)), straightness_rms_mm=float(np.sqrt(np.mean(r ** 2))),
                      straightness_max_mm=float(np.max(np.abs(r))), direction_deg=float(np.degrees(np.arctan2(d[1], d[0]))))


def width_profile(a, b, nbin=6):
    """distance of the points of edge b from the line of edge a, binned along the line of a [mm]."""
    Qa, ca, da = fl[a]
    Qb = fl[b][0]
    nrm = np.array([-da[1], da[0]])
    s = (Qb - ca) @ da
    w = np.abs((Qb - ca) @ nrm)
    edges_ = np.linspace(s.min(), s.max(), nbin + 1)
    prof = [float(np.median(w[(s >= edges_[i]) & (s <= edges_[i + 1])])) for i in range(nbin) if np.any((s >= edges_[i]) & (s <= edges_[i + 1]))]
    return dict(median_mm=float(np.median(w)), profile_mm=prof, span_mm=float(np.ptp(s)))


def ang(a, b):
    da, db = fl[a][2], fl[b][2]
    return float(np.degrees(np.arccos(min(1.0, abs(da @ db)))))


floor_checks = dict(
    per_edge=floor,
    width_white_V=width_profile("scene_tapeV_left", "scene_tapeV_right") if "scene_tapeV_right" in fl else None,
    width_white_H=width_profile("scene_tapeH_top", "scene_tapeH_bottom") if "scene_tapeH_bottom" in fl else None,
    width_blue_L=width_profile("scene_blueL_top", "scene_blueL_bottom") if "scene_blueL_bottom" in fl else None,
    angle_whiteV_whiteH_deg=ang("scene_tapeV_left", "scene_tapeH_top"),
    angle_whiteV_left_right_deg=ang("scene_tapeV_left", "scene_tapeV_right"),
    angle_whiteH_blueH_deg=ang("scene_tapeH_top", "scene_blueH_top") if "scene_blueH_top" in fl else None,
    angle_whiteH_blueL_deg=ang("scene_tapeH_top", "scene_blueL_top") if "scene_blueL_top" in fl else None,
    note="angles between floor groups: the white tape V group and the H group (white tape + blue lines) have separate direction "
         "parameters in the fit, so their mutual angle is NOT imposed; parallelism inside a group IS imposed by the fit",
)

# ---------------------------------------------------------------- 3. both carts on one floor
R80, t80 = RT[80]
foot80 = np.array([[x, y, FLOOR_Z] for x in (0, 1600) for y in (0, 450)], float)
cam = foot80 @ R80.T + t80
in310 = (cam - tv) @ R  # camera -> cart 310: X = R^T (Xc - t)
z_axes = float(np.degrees(np.arccos(np.clip(R80[:, 2] @ R[:, 2], -1, 1))))
two = dict(cart80_footprint_z_in_cart310_mm=in310[:, 2].tolist(), angle_between_cart_Z_axes_deg=z_axes,
           camera_height_above_floor_mm={"310": float(Cc[2] - FLOOR_Z), "80": float((-R80.T @ t80)[2] - FLOOR_Z)})

save_json(dict(model_lines=model, floor=floor_checks, two_carts=two,
               lens=dict(K=K.tolist(), dist=D.tolist())), f"{CACHE}/grid_checks.json")
print("model lines (signed px, + = normal side; mm perpendicular):")
for m in sorted(model, key=lambda q: q["id"]):
    print(f"  {m['id']:32s} exact={m['exact']!s:5s} median {m['median_px']:+6.1f} px (ends {m['end_a_px']:+6.1f} / {m['end_b_px']:+6.1f}) ~{m['median_mm']:+6.0f} mm")
print("floor:", {k: (round(v["straightness_rms_mm"], 1), round(v["length_mm"])) for k, v in floor.items()})
for k in ("width_white_V", "width_white_H", "width_blue_L"):
    v = floor_checks[k]
    if v:
        print(f"  {k}: median {v['median_mm']:.1f} mm, profile {np.round(v['profile_mm'], 1).tolist()} over {v['span_mm']:.0f} mm")
print("  angle white V vs white H %.2f deg; V left/right %.2f; white H vs blue H %s; vs blue L %s" % (
    floor_checks["angle_whiteV_whiteH_deg"], floor_checks["angle_whiteV_left_right_deg"], floor_checks["angle_whiteH_blueH_deg"], floor_checks["angle_whiteH_blueL_deg"]))
print("two carts:", {k: (np.round(v, 1).tolist() if isinstance(v, list) else v) for k, v in two.items()})
