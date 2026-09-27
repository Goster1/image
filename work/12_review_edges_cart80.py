"""Independent review of the cart-80 edge set (work/cache/edges_cart80.json, written by
10_edges_cart80_trace.py).  Applies the review decisions to the JSON (same schema; every changed
entry gets a "review" record, the file gets a top-level "review" block) and re-renders
results/edges_cart80.png.

Run order: 10_edges_cart80_trace.py -> 12_review_edges_cart80.py.  The trace output is copied once to
cache/edges_cart80_prereview.json and the review is always applied to that copy, so the script is
idempotent.

Findings that drive the changes (numbers are recomputed and printed by this script; crops of every
changed edge in cache/review_cart80/*_reviewed_strip.png, reviewer crops rv_*.png / z_*.png / dup_*.png):
 1. y_board1600_outer was 'exact' (X=1600, Z=0).  Plane homographies from the valid edge-based corners of
    markers_final.json put the traced edge at X = 1604 mm (both stickers 1+2) and 1603.3 / 1605.3 mm
    (each sticker alone), i.e. +1.9..+3.0 px outside the drawing line (the trace report quoted 1602-1603).
    Rule used here: an edge stays 'exact' only if every homography (both stickers, each sticker alone)
    puts it within 2 mm of the drawing line -> exact=false.  x_board0_front (-0.9..-1.8 mm) and
    x_board1600_front (-0.2..+0.7 mm) stay exact.
 2. y_board0_inner is neither straight nor parallel to cart Y: residual bow ~2 px after any of four
    rough undistortions (the outer edge of the same board: 0.0-0.3 px), 1.7-2.0 deg off the cart-Y
    vanishing direction of the two outer board edges, and X = 175.9 / 185.5 mm under the two stickers
    of the board -> direction cartY -> unknown, straight_3d -> false.
 3. y_board1600_inner: smooth +-1 px wave along the low-contrast white/tan boundary (the outer edge of
    the same board is straight to +-0.2 px) -> straight_3d -> false (still documented).
 4. Point-level errors removed: (a) fixed trims with a visual reason: x_E_front / x_E_front_in first
    segment y<50 (corner of the shelf at the X=0 end frame, +0.7..1.0 px), x_C_front y>595 (lip
    against the white crate, strength 11-21 vs median 47), x_back_low y>770 (strength 7-15, curls
    -0.9 px), x_B_front_top y>300 (isolated segment after the crate occlusion, slope inside the
    segment, +1.2..-0.7 px), u_div2_low x>1545 (0.8 px step where the flange crosses the back members),
    z_post1600_sil_low y>920 (end points at a different angle next to the board),
    y_board1600_outer 6 px either side of the occluding stick; (b) the '_in' pair traces lose 12 px
    after every interruption > 20 px (mean -0.2..-0.25 px dip there); (c) a robust smooth fit
    (degree 3, 4 for edges > 400 px) removes points > max(0.5 px, 3 MAD) off (label-holder corners,
    jumps onto the holder at A_in y~555, E_in y~490, post slots) and fragments of < 5 points.
 5. x_B_front_top and x_B_front_bot are two pieces of the same shelf-B lip interrupted by a crate
    (a single quadratic fits the union with piece mean residuals <= 0.03 px and rms 0.19 px vs 0.15/0.17 px
    for the pieces, under all four reference lenses) -> merged into one edge cart80_x_B_front (~546 px
    instead of 2 x ~230 px), which constrains the distortion much better than two short pieces.
 6. Duplicates / non-structure: s_frame2 = scene_rpost_rod (0.7 px), s_frame3 = scene_rpost_rodR (0.8 px,
    and s_frame3 jumps onto other structures below y~630), s_rail_right = scene_wall_railB (1.7 px, the
    scene copy is the rod centre line and straighter) -> verified 'no' here (kept in the file with
    duplicate_of).  s_frame is the boundary of the dark shadow/gap 2.5 px left of the rod edge, not a
    physical edge -> verified 'no'.  x_back_rail_out is ALSO in edges_scene.json as scene_wall_railA
    (same image edge, median 0.8 px, labelled 'wall conduit' there) - it is kept here (it runs from the
    X=0 board corner to the X=1600 board), but the scene copy must not be used together with it.
 7. Pairs (same physical member, correlated shape): x_E_front_in -> pair_of x_E_front (verified
    yes -> partly), x_A_front_in -> pair_of x_A_front_out, x_back_rail_in -> pair_of x_back_rail_out.
 8. z_post1600_sil_low is parallel to z_post1600_sil (both point to the nadir) but offset by 4-5 px, so
    the two are different outline lines of the post; at most one can be the X=1600/Y=0 corner line.
 9. Diagnostics stored (not used to change labels): cart-X pencil test.  The five shelf lips meet in one
    vanishing point to <= 0.1 deg; the three back members form their own pencil (<= 0.07 deg) but deviate
    from the lip VP by 0.1-0.7 deg depending strongly on the lens model used for undistortion (sign
    changes between plausible k1/k2) -> this is lens information for the joint fit, not a label error.
    The two board fronts deviate from the lip VP by 2.4-3.5 deg (2-3 px at their ends) with every lens
    tried -> observation for the top-plate / shelf geometry question in CONTEXT.
"""
from __future__ import annotations

import copy
import os

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from common import CACHE, RESULTS, K_from, load_json, save_json, undistort_to_pixels
import edgelib

REGION = "cart80"
R = REGION + "_"
SRC = f"{CACHE}/edges_{REGION}.json"
PRE = f"{CACHE}/edges_{REGION}_prereview.json"
OUT = f"{CACHE}/review_{REGION}"
SCRIPT = "work/12_review_edges_cart80.py"
SCENE = f"{CACHE}/edges_scene.json"

# rough reference lenses, only for diagnostics (fixed numbers so the script is reproducible)
REFS = {"ref_f1290": (K_from(1290, 1290, 959.5, 539.5), np.array([-0.29, 0.08, 0, 0, 0])),
        "ref_f1394": (K_from(1394, 1394, 951.0, 416.0), np.array([-0.31, 0.097, 0, 0, 0])),
        "ref_plumb": (K_from(1394, 1394, 959.5, 539.5), np.array([-0.357, 0.128, 0, 0, 0])),
        "ref_joint": (K_from(1397, 1397, 913.2, 386.5), np.array([-0.307, 0.074, 0, 0, 0]))}
DIRCOL = {"cartX": (0, 0, 255), "cartY": (0, 200, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
          "floor_plane": (0, 255, 255), "horizontal_other": (0, 160, 255), "unknown": (200, 200, 200)}
EXACT_TOL_MM = 2.0


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def load_prereview():
    D = load_json(SRC)
    if "review" not in D:  # fresh output of the trace script
        save_json(D, PRE)
    return load_json(PRE)


def undist(P, ref):
    return undistort_to_pixels(np.asarray(P, float), *ref)


def frame(P):
    c, d, _, _ = edgelib.line_fit(P)
    d = d * np.sign(d[0]) if abs(d[0]) > abs(d[1]) else d * np.sign(d[1])
    return c, d, np.array([-d[1], d[0]])


def bow(P):
    c, d, n = frame(P)
    t, r = (P - c) @ d, (P - c) @ n
    q = np.polyfit(t, r, 2)
    return float(q[0] * (np.ptp(t) / 2) ** 2)


def arclen(P):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def order_along(e):
    P = np.array(e["points"], float)
    c, d, _ = frame(P)
    o = np.argsort((P - c) @ d)
    if (P[-1] - P[0]) @ d < 0:  # keep the original running direction
        o = o[::-1]
    e["points"] = [e["points"][i] for i in o]
    e["strength"] = [e["strength"][i] for i in o]


def note(e, text):
    e.setdefault("review", {"changes": []})["changes"].append(text)


def apply_mask(e, keep, what):
    n0 = len(e["points"])
    keep = np.asarray(keep, bool)
    e["points"] = [p for p, k in zip(e["points"], keep) if k]
    e["strength"] = [s for s, k in zip(e["strength"], keep) if k]
    nrem = n0 - int(keep.sum())
    if nrem:
        note(e, f"{what}: removed {nrem} of {n0} points")
    return nrem


def robust_mask(P, thr=0.5, iters=8):
    c, d, n = frame(P)
    t, r = (P - c) @ d, (P - c) @ n
    deg = 4 if np.ptp(t) > 400 else 3
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        q = np.polyfit(t[keep], r[keep], deg)
        res = r - np.polyval(q, t)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(thr, 3 * mad)
        if (new == keep).all():
            break
        keep = new
    return keep


def fragment_mask(P, gap=8.0, min_pts=5):
    s = arclen(P)
    keep = np.ones(len(P), bool)
    brk = np.nonzero(np.diff(s) > gap)[0]
    for seg in np.split(np.arange(len(P)), brk + 1):
        if len(seg) < min_pts:
            keep[seg] = False
    return keep


def after_gap_mask(P, after=12.0, min_gap=20.0):
    s = arclen(P)
    keep = np.ones(len(P), bool)
    for g in np.nonzero(np.diff(s) > min_gap)[0]:
        keep &= ~((s >= s[g + 1]) & (s < s[g + 1] + after))
    return keep


def refresh(e):
    P = np.array(e["points"], float)
    b, rms = edgelib.sagitta(P)
    e["chord_bow_px"] = round(b, 3)
    e["chord_dev_rms_px"] = round(rms, 3)


# ------------------------------------------------------------------------------------------
# independent checks
# ------------------------------------------------------------------------------------------
BOARD_TESTS = {(0, 92): [("y_board0_outer", 0, 0.0), ("y_board0_inner", 0, 170.0), ("x_board0_front", 1, 0.0)],
               (1, 2): [("y_board1600_outer", 0, 1600.0), ("y_board1600_inner", 0, 1444.0),
                        ("y_board1600_inner_top", 0, 1444.0), ("x_board1600_front", 1, 0.0)]}


def board_homography_checks(edges, ref_name="ref_f1394"):
    """coordinate (X for cart-Y edges, Y for cart-X edges) of every board edge in the plane of the
    stickers on that board: homography from the valid edge-based corners of markers_final.json, from
    both stickers and from each sticker alone (then only edge points within 80 mm of that sticker)."""
    M = load_json(f"{CACHE}/markers_final.json")
    get = lambda mid: [m for m in M if m["cart"] == 80 and m["id"] == mid][0]
    ref = REFS[ref_name]
    out = {}
    for mids, lst in BOARD_TESTS.items():
        for sub in [mids, (mids[0],), (mids[1],)]:
            obj, img, ctr = [], [], []
            for mid in sub:
                m = get(mid)
                for c3, c2, v in zip(m["corners_3d_mm"], m["corners_px"], m["corner_valid"]):
                    if v:
                        obj.append(c3[:2])
                        img.append(c2)
                ctr.append(np.mean(np.array(m["corners_3d_mm"])[:, :2], axis=0))
            obj = np.array(obj, float)
            img = undist(img, ref)
            H, _ = cv2.findHomography(obj, img, 0)
            hr = float(np.sqrt(np.mean(np.sum((cv2.perspectiveTransform(obj[None], H)[0] - img) ** 2, 1))))
            for k, ax, nom in lst:
                if R + k not in edges:
                    continue
                P = undist(edges[R + k]["points"], ref)
                Q = cv2.perspectiveTransform(P[None], np.linalg.inv(H))[0]
                if len(sub) == 1:
                    sel = np.abs(Q[:, 1 - ax] - ctr[0][1 - ax]) < 80
                    # single-sticker homography only for edges close to that sticker (<= 60 mm from its code
                    # square across the edge); farther edges are extrapolations over several sticker widths
                    cc = np.array(get(sub[0])["corners_3d_mm"])[:, ax]
                    if sel.sum() < 5 or min(abs(cc.min() - nom), abs(cc.max() - nom)) > 60:
                        continue
                    Q = Q[sel]
                v = Q[:, ax]
                p0 = np.zeros(2)
                p0[ax], p0[1 - ax] = nom, float(np.median(Q[:, 1 - ax]))
                p1 = p0.copy()
                p1[ax] += 1.0
                a = cv2.perspectiveTransform(np.array([[p0, p1]]), H)[0]
                ppm = float(np.hypot(*(a[1] - a[0])))
                slope = float(np.polyfit(Q[:, 1 - ax], v, 1)[0]) if np.ptp(Q[:, 1 - ax]) > 30 else None
                out.setdefault(k, {})["+".join(map(str, sub))] = dict(
                    coord="XY"[ax], nominal_mm=nom, value_mm=round(float(np.median(v)), 1),
                    offset_mm=round(float(np.median(v)) - nom, 1), offset_px=round((float(np.median(v)) - nom) * ppm, 2),
                    slope_mm_per_100mm=None if slope is None else round(100 * slope, 2),
                    n_points=int(len(v)), n_corners=int(len(obj)), homography_rms_px=round(hr, 2))
    return out


def _line(P):
    c, d, n = frame(P)
    return np.array([n[0], n[1], -n @ c]), c, d


def pencil(L, keys):
    A = np.array([_line(L[k])[0] * np.sqrt(np.ptp(L[k], axis=0).max()) for k in keys])
    v = np.linalg.svd(A)[2][-1]
    return v


def dev_to_vp(P, v):
    _, c, d = _line(P)
    w = v[:2] / v[2] - c if abs(v[2]) > 1e-12 else v[:2]
    w = w / np.linalg.norm(w)
    ang = float(np.degrees(np.arcsin(np.clip(d[0] * w[1] - d[1] * w[0], -1, 1))))
    half = float(np.ptp((P - c) @ d)) / 2
    return round(ang, 2), round(float(np.sin(np.radians(ang)) * half), 2)


def vp_checks(edges):
    """cart-X: pencil of the shelf lips (A_out, B, C, D, E) vs every cart-X edge and vs the back-member
    pencil; cart-Y: VP of the two outer board edges vs the other cart-Y candidates; cart-Z: nadir."""
    out = {}
    for rn, ref in REFS.items():
        L = {k.replace(R, ""): undist(e["points"], ref) for k, e in edges.items()}
        lips = [k for k in ["x_A_front_out", "x_B_front", "x_C_front", "x_D_front", "x_E_front"] if k in L]
        back = [k for k in ["x_back_rail_out", "x_back_rail_in", "x_back_low"] if k in L]
        vx, vb = pencil(L, lips), pencil(L, back)
        vy = pencil(L, ["y_board0_outer", "y_board1600_outer"])
        res = {"vp_x_lips": (vx[:2] / vx[2]).round(0).tolist(), "vp_x_back": (vb[:2] / vb[2]).round(0).tolist(),
               "vp_y_outer_boards": (vy[:2] / vy[2]).round(0).tolist() if abs(vy[2]) > 1e-9 else "at infinity",
               "cartX_vs_lip_vp": {}, "cartX_vs_back_vp": {}, "cartY_vs_outer_board_vp": {}}
        for k, P in L.items():
            if k.startswith("x_"):
                res["cartX_vs_lip_vp"][k] = dict(zip(["angle_deg", "end_offset_px"], dev_to_vp(P, vx)))
                res["cartX_vs_back_vp"][k] = dict(zip(["angle_deg", "end_offset_px"], dev_to_vp(P, vb)))
            elif k.startswith("y_") or k.startswith("u_"):
                res["cartY_vs_outer_board_vp"][k] = dict(zip(["angle_deg", "end_offset_px"], dev_to_vp(P, vy)))
        zs = [k for k in L if k.startswith("z_")]
        if len(zs) >= 2:
            vz = pencil(L, zs)
            res["vp_z_posts"] = (vz[:2] / vz[2]).round(0).tolist()
        out[rn] = res
    return out


def bows_table(D):
    out = {}
    for e in D["edges"]:
        P = np.array(e["points"], float)
        k = e["id"].replace(R, "")
        out[k] = {"raw": round(bow(P), 2)}
        for rn, ref in REFS.items():
            out[k][rn] = round(bow(undist(P, ref)), 2)
    return out


def duplicate_check(edges):
    """points of every cart-80 edge within 3 px of an edge of the scene / cart-310 sets."""
    other = {}
    for fn in [SCENE, f"{CACHE}/edges_cart310.json"]:
        if os.path.exists(fn):
            for e in load_json(fn)["edges"]:
                other[e["id"]] = np.array(e["points"], float)
    out = {}
    for k, e in edges.items():
        P = np.array(e["points"], float)
        for k2, Q in other.items():
            d, _ = cKDTree(Q).query(P)
            m = d < 3
            if m.sum() > 5:
                out.setdefault(k.replace(R, ""), {})[k2] = dict(n_within_3px=int(m.sum()), n_points=int(len(P)),
                                                              median_dist_px=round(float(np.median(d[m])), 2))
    return out


def merge_test(eA, eB):
    """collinearity of two pieces after each rough undistortion: offset of B from A's line and rms of a
    single quadratic through the union vs the pieces."""
    out = {}
    for rn, ref in REFS.items():
        A, B = undist(eA["points"], ref), undist(eB["points"], ref)
        U = np.vstack([A, B])
        c, d, n = frame(U)
        t, r = (U - c) @ d, (U - c) @ n
        q = np.polyfit(t, r, 2)
        res = r - np.polyval(q, t)
        ra, rb = res[:len(A)], res[len(A):]
        cA, dA, nA = frame(A)
        off = (B - cA) @ nA
        tA, rA = (A - cA) @ dA, (A - cA) @ nA
        qA = np.polyfit(tA, rA, 2)
        cB, dB, nB = frame(B)
        tB, rB = (B - cB) @ dB, (B - cB) @ nB
        qB = np.polyfit(tB, rB, 2)
        out[rn] = dict(union_bow_px=round(bow(U), 2), union_line_rms_px=round(edgelib.line_fit(U)[2], 3),
                       mean_offset_B_from_line_A_px=round(float(off.mean()), 2),
                       union_quad_rms_px=round(float(np.sqrt(np.mean(res ** 2))), 3),
                       union_mean_res_A_B_px=[round(float(ra.mean()), 3), round(float(rb.mean()), 3)],
                       pieces_quad_rms_px=[round(float(np.std(rA - np.polyval(qA, tA))), 3),
                                           round(float(np.std(rB - np.polyval(qB, tB))), 3)])
    return out


# ------------------------------------------------------------------------------------------
# the review
# ------------------------------------------------------------------------------------------
def review(D):
    edges = {e["id"]: e for e in D["edges"]}
    for e in D["edges"]:
        order_along(e)
    summary = []
    hom0 = board_homography_checks(edges)
    dups = duplicate_check(edges)

    # ---- 1. fixed trims (visual reasons, see docstring) ----
    def trim(k, cond, why):
        e = edges[R + k]
        P = np.array(e["points"], float)
        apply_mask(e, ~cond(P), f"fixed trim ({why})")

    trim("x_E_front", lambda P: P[:, 1] < 50, "y<50: segment before the first holder at the corner of the X=0 end frame, "
                                             "+0.7..1.0 px off the smooth curve, crop z_Estart.png")
    trim("x_E_front_in", lambda P: P[:, 1] < 50, "y<50: same corner segment as x_E_front")
    trim("x_C_front", lambda P: P[:, 1] > 595, "y>595: lip against the white crate, strength 11-21 vs median 47, "
                                              "-0.4 px, crop z_Cend.png")
    trim("x_back_low", lambda P: P[:, 1] > 770, "y>770: strength 7-15 (median 35), curls -0.9 px at the X=1600 end")
    trim("x_B_front_top", lambda P: P[:, 1] > 300, "y>300: isolated segment after the crate occlusion, tilted inside the "
                                                  "segment (+1.2..-0.7 px)")
    trim("u_div2_low", lambda P: P[:, 0] > 1545, "x>1545: 0.8 px step where the flange line crosses the back members")
    trim("z_post1600_sil_low", lambda P: P[:, 1] > 920, "y>920: last 7 points next to the X=1600 board run at a different "
                                                         "angle (-0.65 -> +0.53 px over 12 px)")
    e = edges[R + "y_board1600_outer"]
    P = np.array(e["points"], float)
    s = arclen(P)
    keep = np.ones(len(P), bool)
    for g in np.nonzero(np.diff(s) > 8)[0]:
        keep &= ~((s > s[g] - 6) & (s < s[g + 1] + 6))
    apply_mask(e, keep, "6 px either side of the occluding stick (points pulled by up to -0.3 px)")
    summary.append("fixed trims: x_E_front/_in y<50, x_C_front y>595, x_back_low y>770, x_B_front_top y>300, "
                   "u_div2_low x>1545, z_post1600_sil_low y>920, y_board1600_outer +-6 px at the occluder")

    # ---- 2. '_in' pair traces: 12 px after interruptions ----
    for k in ["x_A_front_in", "x_E_front_in"]:
        e = edges[R + k]
        apply_mask(e, after_gap_mask(np.array(e["points"], float)), "12 px after every interruption > 20 px "
                                                                   "(mean dip -0.2..-0.25 px there)")
    # ---- 3. robust cleaning of all remaining cart edges ----
    nrob = {}
    for k, e in edges.items():
        if e.get("class") == "scene_structure":
            continue
        P = np.array(e["points"], float)
        n1 = apply_mask(e, robust_mask(P), "robust smooth fit (deg 3/4), |res| > max(0.5 px, 3 MAD)")
        P = np.array(e["points"], float)
        n2 = apply_mask(e, fragment_mask(P), "fragments of < 5 points between gaps > 8 px")
        if n1 + n2:
            nrob[k.replace(R, "")] = n1 + n2
    summary.append(f"robust cleaning removed points on {len(nrob)} edges: {nrob}")

    # ---- 4. merge the two B-lip pieces ----
    eA, eB = edges[R + "x_B_front_top"], edges[R + "x_B_front_bot"]
    mt = merge_test(eA, eB)
    # same line <=> one smooth (quadratic) curve fits the union: no mean offset of either piece and the union
    # rms not larger than 1.3 x the worse piece (the offset from piece A's straight chord alone is not a test:
    # it contains the lens bow of the 546 px union)
    ok = all(max(abs(x) for x in v["union_mean_res_A_B_px"]) < 0.1
             and v["union_quad_rms_px"] < 1.3 * max(v["pieces_quad_rms_px"]) for v in mt.values())
    if ok:
        m = copy.deepcopy(eA)
        m["id"] = R + "x_B_front"
        m["what"] = ("shelf B front lip (Y~0) along the cart: outline of the front lip; the two traced pieces "
                     "(y 100-300 and y 430-646) are interrupted by a plastic crate and merged here")
        m["points"] = eA["points"] + eB["points"]
        m["strength"] = eA["strength"] + eB["strength"]
        m["model_line"]["note"] = eA["model_line"]["note"].split(" [initial")[0] + (
            " [the pieces gave Z = -555 / -528 mm at Y=0 with the ~5 px initial calibration; they are collinear "
            "after undistortion, see review.merge_test]")
        m["verified"] = "yes"
        m["verification_note"] = (f"top piece: {eA['verification_note']} || bottom piece: {eB['verification_note']}")
        m["crop"] = f"work/cache/review_{REGION}/x_B_front_reviewed_strip.png"
        m["crop_pieces"] = [eA["crop"], eB["crop"]]
        m["review"] = {"changes": (eA.get("review", {}).get("changes", []) + eB.get("review", {}).get("changes", []) +
                                   ["merged from cart80_x_B_front_top + cart80_x_B_front_bot (same lip: one quadratic "
                                    "fits the union with piece mean residuals <= 0.03 px and rms ~0.19 px vs 0.15/0.17 px "
                                    "for the pieces)"]),
                       "merge_test": mt}
        order_along(m)
        idx = D["edges"].index(eA)
        D["edges"] = [x for x in D["edges"] if x["id"] not in (eA["id"], eB["id"])]
        D["edges"].insert(idx, m)
        edges = {e["id"]: e for e in D["edges"]}
        summary.append("x_B_front_top + x_B_front_bot merged into x_B_front (one lip; one smooth curve fits both pieces, "
                       f"piece mean residuals <= 0.03 px; merge test {mt['ref_plumb']})")
    else:
        summary.append(f"B pieces NOT merged (merge test failed): {mt}")

    # ---- 5. exactness of the board lines ----
    def hom_txt(k):
        h = hom0[k]
        return "; ".join(f"stickers {s}: {v['coord']} = {v['value_mm']} mm ({v['offset_px']:+.1f} px)" for s, v in h.items())

    for k in ["y_board1600_outer", "x_board0_front", "x_board1600_front"]:
        e = edges[R + k]
        offs = [abs(v["offset_mm"]) for v in hom0[k].values()]
        e["model_line"]["measured_in_sticker_plane"] = hom0[k]
        if max(offs) > EXACT_TOL_MM:
            e["model_line"]["exact"] = False
            e["model_line"]["note"] = (
                "nominal outer end edge of the top plate at X=1600 (drawing X=1600, Z=0) - NOT used as exact: plane "
                "homographies from the valid edge-based corners of the board's stickers (markers_final.json, rough "
                f"undistortion ref_f1394) give {hom_txt(k)}; i.e. the visible board edge lies 3-5 mm outside X=1600 "
                "(the trace note quoted 1602-1603 mm). Rule: exact only if all homographies agree within "
                f"{EXACT_TOL_MM:.0f} mm.")
            note(e, "model_line.exact true -> false (3.3-5.3 mm / +1.9..+3.0 px outside the drawing line per the "
                    "board's own stickers)")
            summary.append(f"{k}: exact -> false ({hom_txt(k)})")
        else:
            e["model_line"]["note"] += f" [review: kept exact; {hom_txt(k)}]"
            note(e, "model_line: independent sticker-plane check added, kept exact")
    for k in ["y_board0_outer", "y_board1600_inner", "y_board1600_inner_top"]:
        edges[R + k]["model_line"]["measured_in_sticker_plane"] = hom0[k]
        edges[R + k]["model_line"]["note"] += f" [review: {hom_txt(k)}]"

    # ---- 6. y_board0_inner: not straight, not cart-Y ----
    e = edges[R + "y_board0_inner"]
    e["direction"] = "unknown"
    e["straight_3d"] = False
    e["model_line"]["measured_in_sticker_plane"] = hom0["y_board0_inner"]
    e["model_line"]["note"] += (f" [review: {hom_txt('y_board0_inner')} - the edge runs obliquely across the board "
                                "(~1.7-2 deg off cart Y)]")
    e["verification_note"] += (" [review: straight_3d -> false and direction cartY -> unknown: residual bow ~2 px after "
                               "any of four rough undistortions (outer edge of the same board 0.0-0.3 px), 1.7-2.0 deg "
                               "off the cart-Y vanishing direction of the two outer board edges, and X = 176 / 186 mm "
                               "under the two stickers of this board; the white sheet is visibly curled at this end]")
    note(e, "straight_3d true -> false, direction cartY -> unknown")
    summary.append("y_board0_inner: straight_3d -> false, direction cartY -> unknown (2 px residual bow, ~2 deg oblique)")

    # ---- 7. y_board1600_inner: wavy ----
    e = edges[R + "y_board1600_inner"]
    e["straight_3d"] = False
    e["verified"] = "partly"
    e["verification_note"] += (" [review: straight_3d -> false, verified -> partly: the low-contrast white/tan boundary "
                               "shows a smooth +-1 px wave along the edge (residual to a quadratic, raw and undistorted), "
                               "the outer edge of the same board is straight to +-0.2 px; not usable for plumb-line / VP]")
    note(e, "straight_3d true -> false, verified yes -> partly (+-1 px wave)")
    summary.append("y_board1600_inner: straight_3d -> false (+-1 px wave on a low-contrast boundary)")

    # ---- 8. pairs ----
    for k, p, dist in [("x_E_front_in", "x_E_front", "2.9 px"), ("x_A_front_in", "x_A_front_out", "6-7.5 px"),
                       ("x_back_rail_in", "x_back_rail_out", "16-17 px")]:
        e = edges[R + k]
        e["pair_of"] = R + p
        if e["verified"] == "yes":
            e["verified"] = "partly"
            note(e, "verified yes -> partly")
        e["verification_note"] += (f" [review: other boundary of the same member as {p} ({dist} away, weaker); "
                                   "its shape is not independent of the partner - do not use both as independent lines]")
        note(e, f"pair_of = {R + p}")
    summary.append("pairs: x_E_front_in (-> partly), x_A_front_in, x_back_rail_in get pair_of")

    # ---- 9. duplicates / non-structure on the scene side ----
    for k, dup in [("s_frame2", "scene_rpost_rod"), ("s_frame3", "scene_rpost_rodR"), ("s_rail_right", "scene_wall_railB")]:
        e = edges[R + k]
        e["verified"] = "no"
        e["duplicate_of"] = dup
        d = dups.get(k, {}).get(dup, {})
        e["verification_note"] += (f" [review: verified -> no: the same image edge is in edges_scene.json as {dup} "
                                   f"({d.get('n_within_3px')} of {d.get('n_points')} points within 3 px, median "
                                   f"{d.get('median_dist_px')} px); scene lines belong to the scene set"
                                   + ("; this copy also jumps onto other structures below y~630 (+1..3.5 px)" if k == "s_frame3" else "")
                                   + ("; this copy includes the curved rod end (bow 1.6-2.1 px after undistortion vs "
                                      "0.4-1.3 px on its neighbours)" if k == "s_rail_right" else "") + "]")
        note(e, f"verified -> no (duplicate of {dup})")
    e = edges[R + "s_frame"]
    e["verified"] = "no"
    e["verification_note"] += (" [review: verified -> no: this is the boundary of the dark shadow/gap strip ~2.5 px left of "
                               "the rod's left edge (scene_rpost_rod / s_frame2), not a physical edge; kink of +0.7/-0.9 px "
                               "at y~440-460; crop dup_frame_a.png]")
    note(e, "verified -> no (shadow/gap boundary next to the rod)")
    e = edges[R + "x_back_rail_out"]
    d = dups.get("x_back_rail_out", {}).get("scene_wall_railA", {})
    e["duplicate_in_other_region"] = "scene_wall_railA"
    e["verification_note"] += (f" [review: the same image edge is also in edges_scene.json as scene_wall_railA "
                               f"({d.get('n_within_3px')} points within 3 px, median {d.get('median_dist_px')} px), labelled "
                               "there as a wall conduit. Kept here as the cart's back member: the rod emerges from under the "
                               "X=0 board corner and runs to the X=1600 board (crops dup_rails_*.png); the scene copy must "
                               "not be used together with this one]")
    note(e, "duplicate_in_other_region = scene_wall_railA (resolve in the scene set)")
    summary.append("s_frame2/s_frame3/s_rail_right: verified -> no (duplicates of scene edges); s_frame: verified -> no "
                   "(shadow boundary); x_back_rail_out duplicated by scene_wall_railA (flagged, kept here)")

    # ---- 10. lip straightness caveat (documented, labels kept) ----
    for k in ["x_E_front", "x_E_front_in"]:
        e = edges[R + k]
        e["verification_note"] += (
            " [review: after each of four reference undistortions the chord bow of the E lip exceeds that of "
            "x_D_front by +0.54..+0.60 px (both boundaries of the lip agree), although E lies closer to the image "
            "centre than D and a radial lens residual would bow it less; the E lip is probably bent by ~0.5 px "
            "(~1 mm). Kept straight_3d=true (below the ~1.3 px level used for the divider flanges), but it is "
            "the least straight lip: use a low weight or check the plumb-line result without it]")
        note(e, "straightness caveat added (+0.55 px excess bow relative to x_D_front)")

    # ---- 11. posts ----
    e = edges[R + "z_post1600_sil_low"]
    e["model_line"] = None
    e["verification_note"] += (" [review: model_line (X=1600, Y=0, not exact) removed: this post outline below the push "
                               "handle is parallel to z_post1600_sil (both point to the nadir) but offset by 4-5 px from "
                               "it, so the two are different outline lines of the post and at most one can be the "
                               "X=1600/Y=0 corner line; direction cartZ kept]")
    note(e, "model_line removed (4-5 px parallel offset from z_post1600_sil, not the same outline line)")

    for e in D["edges"]:
        if "review" in e:
            refresh(e)
            e["review"]["crop"] = f"work/cache/review_{REGION}/{e['id'].replace(R, '')}_reviewed_strip.png"
    return summary, hom0, dups


# ------------------------------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------------------------------
def render_overview(D):
    col = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    for e in D["edges"]:
        P = np.array(e["points"])
        c = DIRCOL.get(e["direction"], (200, 200, 200))
        if e.get("verified") == "no":
            c = (90, 90, 90)
        gaps = np.nonzero(np.hypot(*np.diff(P, axis=0).T) > 8)[0]
        s0 = 0
        for g in list(gaps) + [len(P) - 1]:
            seg = P[s0:g + 1]
            if len(seg) > 1:
                cv2.polylines(col, [np.round(seg * 8).astype(np.int32)], False, c, 1, cv2.LINE_AA, shift=3)
            s0 = g + 1
        m = P[len(P) // 2]
        lab = e["id"].replace(R, "") + ("" if e.get("verified") == "yes" else " (p)" if e.get("verified") == "partly" else " (no)")
        if e.get("model_line") and e["model_line"].get("exact"):
            lab += " [exact]"
        if not e.get("straight_3d", True):
            lab += " {not straight}"
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1, cv2.LINE_AA)
    y = 300  # legend in the empty aisle between the carts
    for k, c in list(DIRCOL.items()) + [("verified no (duplicate / not structure)", (90, 90, 90))]:
        cv2.putText(col, k, (880, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(col, k, (880, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        y += 17
    cv2.putText(col, "(p) partly, [exact] drawing-exact line",
                (880, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{RESULTS}/edges_{REGION}.png", col)


def render_review_crops(D, ids, before):
    """straightened strip (3x along, 6x across, +-10 px) with the kept points (red) and the points removed
    by the review (cyan)."""
    os.makedirs(OUT, exist_ok=True)
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    for e in D["edges"]:
        if e["id"] not in ids:
            continue
        P = np.array(e["points"], float)
        src = before.get(e["id"]) if e["id"] in before else None
        if src is None and e["id"] == R + "x_B_front":
            src = np.vstack([before[R + "x_B_front_top"], before[R + "x_B_front_bot"]])
        c, d, n = frame(P)
        tp, rp = (P - c) @ d, (P - c) @ n
        q = np.polyfit(tp, rp, 2)
        hs, vs, hw = 3, 6, 10.0
        lo, hi = tp.min(), tp.max()
        if src is not None:
            ts0 = (src - c) @ d
            lo, hi = min(lo, ts0.min()), max(hi, ts0.max())
        ts = np.arange(lo - 15, hi + 15, 1 / hs)
        os_ = np.arange(-hw, hw, 1 / vs)
        base = np.polyval(q, ts)
        X = c[0] + d[0] * ts[None] + n[0] * (base[None] + os_[:, None])
        Y = c[1] + d[1] * ts[None] + n[1] * (base[None] + os_[:, None])
        S = np.stack([map_coordinates(img[:, :, k].astype(float), [Y.ravel(), X.ravel()], order=1).reshape(X.shape)
                      for k in range(3)], -1)
        S = np.clip(S, 0, 255).astype(np.uint8)

        def dots(Q, colr):
            for tt, rr in zip((Q - c) @ d, (Q - c) @ n):
                u = (tt - ts[0]) * hs
                v = (rr - np.polyval(q, tt) + hw) * vs
                if 0 <= v < S.shape[0]:
                    cv2.circle(S, (int(round(u * 4)), int(round(v * 4))), 5, colr, -1, shift=2)

        if src is not None:
            kept = cKDTree(P).query(src)[0] < 1e-6
            dots(src[~kept], (255, 255, 0))
        dots(P, (0, 0, 255))
        W = 900
        rows = []
        for k0 in range(0, S.shape[1], W):
            r = S[:, k0:k0 + W]
            if r.shape[1] < 20:
                continue
            rows.append(np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=40))
        out = np.vstack(rows)
        out = np.pad(out, ((0, 18), (0, 0), (0, 0)), constant_values=255)
        cv2.putText(out, e["id"].replace(R, "") + "  red = kept, cyan = removed by review", (4, out.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUT}/{e['id'].replace(R, '')}_reviewed_strip.png", out)


def straightness_table(D, title):
    print(f"\n{title}: rms about a straight line after rough undistortion [px] ({' / '.join(REFS)})")
    for e in D["edges"]:
        P = np.array(e["points"])
        v = [edgelib.line_fit(undist(P, ref))[2] for ref in REFS.values()]
        print(f"  {e['id'].replace(R, ''):24s} n={len(P):4d} {e['verified']:6s} {e['direction']:16s} "
              f"straight={str(e.get('straight_3d')):5s} " + " / ".join(f"{x:.3f}" for x in v))


def main():
    D0 = load_prereview()
    D = copy.deepcopy(D0)
    before = {e["id"]: np.array(e["points"], float) for e in D0["edges"]}
    summary, hom0, dups = review(D)
    edges = {e["id"]: e for e in D["edges"]}
    vps = vp_checks({k: e for k, e in edges.items() if e.get("verified") != "no"})
    bows = bows_table(D)
    changed = [e["id"] for e in D["edges"] if "review" in e]
    obs = [
        "cart-X pencil: the five shelf lips (A_out, B, C, D, E) meet in one vanishing point to <= 0.15 deg; the three "
        "back members form their own pencil (<= 0.07 deg) whose angle to the lip VP is +0.0..+0.6 deg and changes "
        "sign between plausible lenses -> lens information for the joint fit, not a label error",
        "the two board fronts (exact lines x_board0_front / x_board1600_front) deviate from the lip VP by about "
        "-2.6 / +3.0 deg (2-3 px at their ends) with every lens tried, in opposite senses at the two ends; in the "
        "plane of their own stickers both are parallel to the sticker X axis (0.2-1 mm per 100 mm) -> the top-plate "
        "boards (with their stickers) are not parallel to the shelf lips in the image: observation for the "
        "top-plate / shelf geometry question, do not put the board fronts into the cart-X VP group together "
        "with the lips without a separate check",
        "relative lip bows after undistortion: C < D < E by ~0.3 and ~0.55 px (monotonic with shelf height, opposite "
        "to the radial-lens expectation) -> shelf lips are straight only to ~0.3-0.6 px",
        "the Z values quoted in the lip model_line notes (Z=-167..-1459 at Y=0) come from the initial calibration "
        "(RMS ~5 px, f~1290); they are rough and must not be read as a measured shelf-height mismatch",
        "not reviewed here: cart-Z coverage stays weak (z_post1600_sil ~100 px, z_post1600_sil_low ~75 px, 4-5 px "
        "apart and different outline lines)"]
    D["review"] = {"script": SCRIPT, "observations": obs, "input": "work/cache/edges_cart80_prereview.json (output of 10_edges_cart80_trace.py)",
                   "summary": summary, "changed_edges": changed,
                   "board_homography_check": hom0, "vp_check": vps, "chord_bows_px": bows,
                   "duplicates_with_other_regions": dups,
                   "crops": "work/cache/review_cart80/*_reviewed_strip.png (red kept, cyan removed); reviewer crops "
                            "rv_*.png, z_*.png, dup_*.png, ctx_*.png"}
    save_json(D, SRC)
    render_overview(D)
    render_review_crops(D, changed, before)
    for s in summary:
        print("*", s)
    for k, v in hom0.items():
        print(f"  board {k:22s} " + " | ".join(f"{s}: {c['coord']}={c['value_mm']} ({c['offset_px']:+.2f}px, "
                                                  f"slope {c['slope_mm_per_100mm']})" for s, c in v.items()))
    for rn, v in vps.items():
        print(f"  VP {rn}: lips {v['vp_x_lips']} back {v['vp_x_back']} outer-boards-Y {v['vp_y_outer_boards']}")
        print("      cartX vs lip VP: " + ", ".join(f"{k} {x['angle_deg']:+.2f}/{x['end_offset_px']:+.1f}"
                                                  for k, x in v["cartX_vs_lip_vp"].items()))
        print("      cartY vs board VP: " + ", ".join(f"{k} {x['angle_deg']:+.2f}/{x['end_offset_px']:+.1f}"
                                                    for k, x in v["cartY_vs_outer_board_vp"].items()))
    straightness_table(D0, "before review")
    straightness_table(D, "after review")
    print("\nchord bows (raw / " + " / ".join(REFS) + ")")
    for k, b in bows.items():
        print(f"  {k:24s} " + " ".join(f"{x:+6.2f}" for x in b.values()))


if __name__ == "__main__":
    main()
