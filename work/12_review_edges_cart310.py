"""Independent review of the cart-310 edge set (work/cache/edges_cart310.json, written by
10_edges_cart310_trace.py).  Applies the review decisions to the JSON (same schema; every changed
entry gets a "review" record) and re-renders results/edges_cart310.png.

Run order: 10_edges_cart310_trace.py -> 12_review_edges_cart310.py.  The trace output is copied
once to cache/edges_cart310_prereview.json and the review is always applied to that copy, so the
script is idempotent (running it twice gives the same result).

Findings that drive the changes (numbers printed by this script, crops in cache/review_cart310/):
 1. board1600_outer was 'exact' (X=1600, Z=0) although a plane homography from the valid edge-based
    corners of markers 1 and 2 (markers_final.json) puts the traced edge at X = 1618-1620 mm
    (12-16 px beyond the X=1600 line, raw and after two different rough undistortions).
    An 'exact' line is used by the combined estimator as a hard geometric constraint, so a line that
    is 19 mm off would inject a 12-16 px systematic error -> exact=false (the board overhangs /
    its visible edge is not the X=1600 line).  Same policy as cart80_y_board0_outer (-20 mm).
 2. Shelf-front lips: right after each label-holder interruption the lip dips by 0.3-0.7 px over
    ~30 px (residuals vs. a straight line after a rough undistortion, all five fronts), i.e. the
    12 px trimmed by the trace were not enough.  A fixed geometric rule (not residual driven) now
    removes a further 24 px after and 12 px before every interruption > 20 px.
 3. The '_rise' traces of B and C are the weak side (strength ~20 vs 30-43) of the same ~2.5 px dark
    outline line whose strong side is the '_fall' trace; the weak-side location depends on the
    profile window (0.35 px systematic shift over y 261-409 on C_rise) and the rise/fall chord bows
    differ by 0.5 px -> kept but downgraded to 'partly' and marked as pair of the fall trace
    (do not use both of a pair as independent lines).
 4. Post traces: isolated ~1 px jumps (screw, orange handle behind the post, knob at the castor end,
    weak far end beyond the crossing rod) removed by a robust cubic fit (|res| > 0.5 px) plus
    explicit end trims.
 5. y_lip4: the holder end bends by up to 0.7 px (visible on the crop) and the far end is very weak
    (strength ~5) -> both ends trimmed.
 6. Notes updated with independent checks (board homography X/Y of the board edges, angle of the
    X=0 board front to the data-driven shelf-front vanishing point).
"""
from __future__ import annotations

import copy
import os

import cv2
import numpy as np

from common import CACHE, RESULTS, K_from, load_json, save_json, undistort_to_pixels
import edges310lib as E

REGION = "cart310"
SRC = f"{CACHE}/edges_{REGION}.json"
PRE = f"{CACHE}/edges_{REGION}_prereview.json"
OUT = f"{CACHE}/review_{REGION}"
SCRIPT = "work/12_review_edges_cart310.py"

# rough reference lenses, only used for diagnostics (never to select points)
REFS = {"ref_f1290": (K_from(1290, 1290, 959.5, 539.5), np.array([-0.29, 0.08, 0, 0, 0])),
        "ref_f1394": (K_from(1394, 1394, 951.0, 416.0), np.array([-0.31, 0.097, 0, 0, 0]))}

HOLDER_AFTER, HOLDER_BEFORE, HOLDER_MIN_GAP = 24.0, 12.0, 20.0


# ------------------------------------------------------------------------------------------
def load_prereview():
    D = load_json(SRC)
    if "review" not in D:  # fresh output of the trace script
        save_json(D, PRE)
    return load_json(PRE)


def arclen(P):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def holder_trim_mask(P, after=HOLDER_AFTER, before=HOLDER_BEFORE, min_gap=HOLDER_MIN_GAP):
    s = arclen(P)
    keep = np.ones(len(P), bool)
    for g in np.nonzero(np.diff(s) > min_gap)[0]:
        keep &= ~((s >= s[g + 1]) & (s < s[g + 1] + after))
        keep &= ~((s <= s[g]) & (s > s[g] - before))
    # drop left-over fragments (fewer than 6 points between gaps > 8 px)
    idx = np.nonzero(keep)[0]
    if len(idx):
        brk = np.nonzero(np.diff(s[idx]) > 8.0)[0]
        for seg in np.split(idx, brk + 1):
            if len(seg) < 6:
                keep[seg] = False
    return keep


def robust_mask(P, thr=0.5, deg=3, iters=6):
    c, d, _, _ = E.edgelib.line_fit(P)
    n = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ n
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


def apply_mask(e, keep, what):
    n0 = len(e["points"])
    e["points"] = [p for p, k in zip(e["points"], keep) if k]
    e["strength"] = [s for s, k in zip(e["strength"], keep) if k]
    e.setdefault("review", {"changes": []})["changes"].append(f"{what}: removed {n0 - int(np.sum(keep))} of {n0} points")


def refresh(e):
    P = np.array(e["points"], float)
    rms, bow, L = E.chord_stats(P)
    e["chord_dev_rms_px"] = round(rms, 3)
    e["chord_bow_px"] = round(bow, 3)
    e["length_px"] = round(L, 1)
    e["n_points"] = int(len(P))
    e["angle_to_initial_vp_deg"] = {k: round(v, 2) for k, v in E.direction_angles(P).items()}


def note(e, text):
    e.setdefault("review", {"changes": []})["changes"].append(text)


# ------------------------------------------------------------------------------------------
# independent checks
# ------------------------------------------------------------------------------------------
def board_homography_checks(edges):
    """X (or Y) of the board edges in the plane of the two stickers on that board (valid edge-based
    corners of markers_final.json), raw and after two rough undistortions."""
    M = load_json(f"{CACHE}/markers_final.json")
    get = lambda mid: [m for m in M if m["cart"] == 310 and m["id"] == mid][0]
    tests = {(0, 322): [("board0_outer", "X", 0.0), ("board0_inner_top", "X", 165.0), ("x_board0_front", "Y", 0.0)],
             (1, 2): [("board1600_outer", "X", 1600.0), ("board1600_inner", "X", 1435.0),
                      ("x_board1600_front", "Y", 0.0)]}
    out = {}
    for mids, lst in tests.items():
        for rn in ["raw"] + list(REFS):
            f = (lambda P: np.asarray(P, float)) if rn == "raw" else (lambda P, r=REFS[rn]: undistort_to_pixels(np.asarray(P, float), *r))
            obj, img = [], []
            for mid in mids:
                m = get(mid)
                for c3, c2, v in zip(m["corners_3d_mm"], m["corners_px"], m["corner_valid"]):
                    if v:
                        obj.append(c3[:2])
                        img.append(c2)
            obj = np.array(obj, float)
            img = f(np.array(img, float))
            H, _ = cv2.findHomography(obj, img, 0)
            hr = float(np.sqrt(np.mean(np.sum((cv2.perspectiveTransform(obj[None], H)[0] - img) ** 2, 1))))
            for k, coord, nom in lst:
                P = f(np.array(edges[f"{REGION}_{k}"]["points"]))
                Q = cv2.perspectiveTransform(P[None], np.linalg.inv(H))[0]
                v = Q[:, 0] if coord == "X" else Q[:, 1]
                cc = Q.mean(0)
                dv = np.array([1.0, 0.0]) if coord == "X" else np.array([0.0, 1.0])
                a = cv2.perspectiveTransform(np.array([[cc, cc + 10 * dv]]), H)[0]
                ppm = float(np.hypot(*(a[1] - a[0])) / 10)
                out.setdefault(k, {})[rn] = dict(markers=list(mids), n_corners=len(obj), homography_rms_px=round(hr, 2),
                                                 coord=coord, value_mm=round(float(np.median(v)), 1),
                                                 spread_p10_p90_mm=round(float(np.percentile(v, 90) - np.percentile(v, 10)), 1),
                                                 nominal_mm=nom, offset_px=round((float(np.median(v)) - nom) * ppm, 1))
    return out


def _line(P):
    c, d, _, _ = E.edgelib.line_fit(P)
    n = np.array([-d[1], d[0]])
    return np.array([n[0], n[1], -n @ c])


def vp_checks(edges):
    """Data-driven vanishing points (rough undistortion): cart-X from the five strong shelf-front
    ('_fall') traces, cart-Y from the four board edges; angle / end offset of every edge."""
    out = {}
    for rn, ref in REFS.items():
        f = lambda P: undistort_to_pixels(np.asarray(P, float), *ref)
        L = {k.replace(f"{REGION}_", ""): f(np.array(e["points"])) for k, e in edges.items()}

        def vp(keys):
            A = np.array([_line(L[k]) * np.sqrt(np.ptp(L[k], axis=0).max()) for k in keys])
            return np.linalg.svd(A)[2][-1]

        def dev(k, v):
            P = L[k]
            c, d, _, _ = E.edgelib.line_fit(P)
            w = v[:2] / v[2] - c if abs(v[2]) > 1e-12 else v[:2]
            w = w / np.linalg.norm(w)
            ang = float(np.degrees(np.arcsin(min(1.0, abs(d[0] * w[1] - d[1] * w[0])))))
            half = float(np.ptp((P - c) @ d)) / 2
            return round(ang, 2), round(np.sin(np.radians(ang)) * half, 2)

        vx = vp([f"x_{s}_front_fall" for s in "ABCDE"])
        vy = vp(["board0_outer", "board1600_outer", "board1600_inner", "board0_inner_top"])
        res = {"vp_x": (vx[:2] / vx[2]).round(0).tolist(), "vp_y": (vy[:2] / vy[2]).round(0).tolist(), "edges": {}}
        for k, e in edges.items():
            kk = k.replace(f"{REGION}_", "")
            if e["direction"] == "cartX" or kk.startswith("x_"):
                res["edges"][kk] = dict(zip(["ang_to_vpx_deg", "end_offset_px"], dev(kk, vx)))
            elif e["direction"] == "cartY" or kk.startswith("y_") or kk.startswith("board"):
                res["edges"][kk] = dict(zip(["ang_to_vpy_deg", "end_offset_px"], dev(kk, vy)))
        out[rn] = res
    return out


# ------------------------------------------------------------------------------------------
def review(D):
    edges = {e["id"]: e for e in D["edges"]}
    R = f"{REGION}_"
    hom = board_homography_checks(edges)
    summary = []

    # 1. board1600_outer: not at X=1600 per its own board's stickers -> not exact
    e = edges[R + "board1600_outer"]
    h = hom["board1600_outer"]
    e["model_line"]["exact"] = False
    e["model_line"]["note"] = (
        "nominal outer end edge of the top plate (drawing X=1600, Z=0) - NOT used as exact: a plane homography "
        "from the valid edge-based corners of markers 1 and 2 (markers_final.json) puts the traced edge at "
        f"X = {h['raw']['value_mm']} mm raw / {h['ref_f1290']['value_mm']}-{h['ref_f1394']['value_mm']} mm after two rough "
        f"undistortions ({h['raw']['offset_px']:+.0f}..{h['ref_f1290']['offset_px']:+.0f} px beyond the X=1600 line); the "
        "visible board edge overhangs ~19 mm (compare cart80_y_board0_outer, -20 mm, also not exact). "
        "The same homography puts x_board1600_front at Y = -0.7..-1.1 mm, i.e. the board and its stickers agree in Y.")
    note(e, "model_line.exact true -> false (edge 12-16 px / ~19 mm off the drawing line X=1600 per the board's own stickers)")
    summary.append("board1600_outer: exact -> false (X=1618-1620 mm by the homography of markers 1/2)")

    e = edges[R + "board0_outer"]
    h = hom["board0_outer"]
    e["model_line"]["note"] += (f" [review: re-checked with the edge-based corners of markers_final.json: X = {h['raw']['value_mm']} mm raw, "
                                f"{h['ref_f1290']['value_mm']} / {h['ref_f1394']['value_mm']} mm after rough undistortions "
                                f"({h['raw']['offset_px']:+.1f} px), homography rms {h['raw']['homography_rms_px']} px -> kept exact]")
    note(edges[R + "board0_outer"], "model_line.note: independent homography re-check added (kept exact)")
    for k, nom in [("board0_inner_top", 165), ("board1600_inner", 1435)]:
        h = hom[k]
        edges[R + k]["model_line"]["note"] += (
            f" [review: homography of the board's stickers puts this edge at X = {h['raw']['value_mm']} mm "
            f"({h['ref_f1394']['value_mm']} mm undistorted), not {nom}; not exact anyway]")
        note(edges[R + k], f"model_line.note: measured X from the board homography added (nominal {nom} is only assumed)")

    # 2. holder-adjacent dips on the shelf-front lips (fixed geometric trimming)
    for k in ["x_A_front_fall", "x_B_front_fall", "x_B_front_rise", "x_C_front_fall", "x_C_front_rise",
              "x_D_front_fall", "x_E_front_fall", "x_B_inner_fall"]:
        e = edges[R + k]
        keep = holder_trim_mask(np.array(e["points"]))
        apply_mask(e, keep, f"holder trim (further {HOLDER_AFTER:.0f} px after / {HOLDER_BEFORE:.0f} px before each "
                            f"interruption > {HOLDER_MIN_GAP:.0f} px; lip dips 0.3-0.7 px up to ~30 px after the holders)")
        e["verification_note"] += (f" [review: a further {HOLDER_AFTER:.0f} px after / {HOLDER_BEFORE:.0f} px before each "
                                   "holder interruption removed: residuals after rough undistortion showed 0.3-0.7 px dips "
                                   "extending ~30 px beyond the 12 px trim]")
    summary.append("shelf fronts A-E (+B/C rise, B inner): extra 24/12 px trimmed at every holder interruption")

    # 3. rise traces = weak side of the same dark outline line as the fall traces
    for sh in "BC":
        e = edges[R + f"x_{sh}_front_rise"]
        e["verified"] = "partly"
        e["pair_of"] = R + f"x_{sh}_front_fall"
        e["verification_note"] += (
            f" [review: weak side (strength ~20 vs {'30' if sh == 'B' else '43'} of the fall side) of the same ~2.5 px dark "
            f"outline line as x_{sh}_front_fall; its sub-pixel position depends on the profile window (C: 0.35 px systematic "
            "shift over y 261-409 between 0.25 and 0.125 px profile sampling) and the rise/fall chord bows differ by ~0.5 px. "
            "Downgraded to 'partly'; do NOT use it together with its fall partner as an independent line]")
        note(e, "verified yes -> partly; pair_of added")
    summary.append("x_B_front_rise, x_C_front_rise: verified -> partly, pair_of = the fall trace (same physical dark line)")

    # 4. posts: jumps / end trims
    e = edges[R + "z_post_front_left_right"]
    P = np.array(e["points"])
    keep = robust_mask(P) & ~((P[:, 1] > 660) & (P[:, 1] < 672))
    apply_mask(e, keep, "robust cubic fit |res|>0.5 px (1 px jumps at the orange handle behind the post) + screw y 660-672")
    e["verification_note"] += " [review: isolated ~1 px jumps (handle behind the post, reflective screw) removed]"

    e = edges[R + "z_post_front_right"]
    P = np.array(e["points"])
    keep = (P[:, 0] < 660) & (P[:, 0] > 550)
    apply_mask(e, keep, "end trims: x>660 (points slide onto the dark knob at the castor end, -1.6 px) and x<550 "
                        "(beyond the crossing rod, strength ~7, -0.6 px)")
    P = np.array(e["points"])
    keep = robust_mask(P)
    apply_mask(e, keep, "robust cubic fit |res|>0.5 px")
    e["verification_note"] += (
        " [review: knob end and weak far end trimmed. Identity check: the intersections of this line with the five "
        "shelf-front lines (A-D) have the cross ratio 1.272-1.281 vs 1.276 expected for Z=-195,-595,-995,-1300 on one "
        "vertical line -> it is a vertical (cart-Z) line in the plane Y~0; with E included 1.42-1.46 vs 1.40]")
    summary.append("z posts: ~1 px jumps and bad ends removed")

    # 5. y_lip4: bent holder end and weak far end
    e = edges[R + "y_lip4"]
    P = np.array(e["points"])
    keep = (P[:, 0] < 683) & (P[:, 0] > 570)
    apply_mask(e, keep, "trimmed x>683 (flange bends up to 0.7 px towards the holder, visible on the crop) and x<570 "
                        "(strength ~5)")
    e["verification_note"] += " [review: bent holder end and very weak far end removed; remaining part still wavy ~0.3 px]"

    # 6. x_board0_front: record why it is not cart-X
    e = edges[R + "x_board0_front"]
    h = hom["x_board0_front"]
    e["model_line"]["note"] += (
        f" [review: in the plane of markers 0/322 this edge lies at Y = {h['raw']['value_mm']} mm (spread "
        f"{h['raw']['spread_p10_p90_mm']} mm), i.e. parallel to the sticker X axis, but it deviates ~5.9 deg (4.6 px at "
        "its ends) from the cart-X vanishing point of the five shelf fronts; the shelf-X direction appears ~5.7 deg rotated "
        "in the frame of this board -> the X=0 end board (with its stickers) is not parallel to the shelves in the image "
        "(tilted/raised board?) - observation, kept 'unknown']")
    note(e, "model_line.note: board-frame Y and angle to the data-driven cart-X vanishing point added")

    # 7. divider flanges: demonstrably not straight lines at the sub-pixel level
    bows = local_bow_comparison(D)
    for k in ["y_lip1", "y_lip2", "y_lip4", "y_lip4b"]:
        e = edges[R + k]
        e["straight_3d"] = False
        b = bows[k]
        e["verification_note"] += (
            f" [review: straight_3d -> false. Chord bow {b['raw']:+.2f} px raw, {b['ref_f1290']:+.2f} / {b['ref_f1394']:+.2f} px "
            "after two rough undistortions. Lens distortion is a function of image position and orientation only; "
            + ("board1600_inner (same orientation, ~90 px away, same length) goes from "
               f"{bows['board1600_inner']['raw']:+.2f} to {bows['board1600_inner']['ref_f1290']:+.2f} / "
               f"{bows['board1600_inner']['ref_f1394']:+.2f} px, so the ~1.3-1.5 px left on this flange is not lens "
               "distortion: the flange (or its shading boundary) is curved"
               if k in ("y_lip1", "y_lip2") else
               "y_lip4 and y_lip4b are parallel neighbours ~60 px apart with OPPOSITE bows (~-0.8 / +0.8 px) that the "
               "undistortion hardly changes -> the flange lines are curved at the 0.8 px level")
            + "; not usable for plumb-line]")
        note(e, "straight_3d true -> false (non-lens curvature, see verification_note)")
    summary.append("y_lip1, y_lip2, y_lip4, y_lip4b: straight_3d -> false (curvature not explainable by any lens: "
                   "1.3-1.5 px residual bow next to the straight board1600_inner; lip4/lip4b opposite bows)")

    # 8. floor tape: both edges of the same tape bow differently after undistortion
    bu, bl = bows["floor_tape_blue_upper"], bows["floor_tape_blue_lower"]
    for k in ["floor_tape_blue_upper", "floor_tape_blue_lower"]:
        e = edges[R + k]
        e["verified"] = "partly"
        e["verification_note"] += (
            f" [review: the two edges of the same ~14 px wide tape have chord bows {bu['ref_f1394']:+.2f} and "
            f"{bl['ref_f1394']:+.2f} px after a rough undistortion ({bu['raw']:+.2f} / {bl['raw']:+.2f} raw), and the tape "
            "width varies non-monotonically by ~0.4 px -> the hand-laid tape is straight only to ~0.5 px; "
            "downgraded to 'partly' (low weight / sensitivity check only)]")
        note(e, "verified yes -> partly (tape not straight to ~0.5 px)")
    summary.append("floor tape edges: verified -> partly (edges of the same tape bow differently by ~0.5 px)")

    for e in D["edges"]:
        if "review" in e:
            refresh(e)
            e["review"]["crop"] = f"work/cache/review_{REGION}/{e['id'].replace(REGION + '_', '')}_reviewed_strip.png"
    return summary, hom


def _bow(P):
    c, d, _, _ = E.edgelib.line_fit(P)
    if d[0] < 0:
        d = -d
    n = np.array([-d[1], d[0]])
    t, r = (P - c) @ d, (P - c) @ n
    q = np.polyfit(t, r, 2)
    return float(q[0] * (np.ptp(t) / 2) ** 2)


def local_bow_comparison(D):
    """signed chord bow (normal rotated +90 deg from a left-to-right direction) raw and after the rough
    undistortions, for comparing neighbouring, similarly oriented edges."""
    out = {}
    for e in D["edges"]:
        P = np.array(e["points"], float)
        k = e["id"].replace(f"{REGION}_", "")
        out[k] = {"raw": round(_bow(P), 2)}
        for rn, ref in REFS.items():
            out[k][rn] = round(_bow(undistort_to_pixels(P, *ref)), 2)
    return out


def render_overview(D):
    col = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    colors = {"cartX": (0, 0, 255), "cartY": (0, 170, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
              "floor_plane": (0, 200, 255), "horizontal_other": (255, 255, 0), "unknown": (128, 128, 128)}
    for e in D["edges"]:
        P = np.array(e["points"])
        c = colors[e["direction"]]
        gaps = np.nonzero(np.hypot(*np.diff(P, axis=0).T) > 8)[0]
        s0 = 0
        for g in list(gaps) + [len(P) - 1]:
            seg = P[s0:g + 1]
            if len(seg) > 1:
                cv2.polylines(col, [np.round(seg * 8).astype(np.int32)], False, c, 1, cv2.LINE_AA, shift=3)
            s0 = g + 1
        m = P[len(P) // 2]
        lab = e["id"].replace(f"{REGION}_", "") + ("" if e.get("verified") == "yes" else " (p)")
        if e.get("model_line") and e["model_line"].get("exact"):
            lab += " [exact]"
        if not e.get("straight_3d", True):
            lab += " {not straight}"
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1, cv2.LINE_AA)
    y = 20
    for k, c in colors.items():
        cv2.putText(col, k, (1700, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        y += 18
    cv2.putText(col, "(p) = verified partly, [exact] = drawing-exact model line", (1500, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{RESULTS}/edges_{REGION}.png", col)


def render_review_crops(D, ids):
    """straightened strip (6x across, 3x along the edge) with the stored points, for changed edges."""
    os.makedirs(OUT, exist_ok=True)
    col = cv2.imread(f"{CACHE}/mean_aligned_color.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
    from scipy.ndimage import map_coordinates
    for e in D["edges"]:
        if e["id"] not in ids:
            continue
        P = np.array(e["points"])
        c, d, _, _ = E.edgelib.line_fit(P)
        if d @ (P[-1] - P[0]) < 0:
            d = -d
        n = np.array([-d[1], d[0]])
        tp, rp = (P - c) @ d, (P - c) @ n
        q = np.polyfit(tp, rp, 2)
        hs, vs, hw = 3, 6, 10.0
        ts = np.arange(tp.min() - 15, tp.max() + 15, 1 / hs)
        os_ = np.arange(-hw, hw, 1 / vs)
        base = np.polyval(q, ts)
        X = c[0] + d[0] * ts[None] + n[0] * (base[None] + os_[:, None])
        Y = c[1] + d[1] * ts[None] + n[1] * (base[None] + os_[:, None])
        S = map_coordinates(col, [Y.ravel(), X.ravel()], order=1).reshape(X.shape)
        S = cv2.cvtColor(np.clip(S, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for tt, rr in zip(tp, rp):
            u = (tt - ts[0]) * hs
            v = (rr - np.polyval(q, tt) + hw) * vs
            cv2.circle(S, (int(round(u * 4)), int(round(v * 4))), 5, (0, 0, 255), -1, shift=2)
        W = 600
        rows = []
        for k0 in range(0, S.shape[1], W):
            r = S[:, k0:k0 + W]
            if r.shape[1] < 20:
                continue
            rows.append(np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=40))
        cv2.imwrite(f"{OUT}/{e['id'].replace(REGION + '_', '')}_reviewed_strip.png", np.vstack(rows))


def straightness_table(D, title):
    print(f"\n{title}: per-edge rms about a straight line after rough undistortion [px] ({' / '.join(REFS)})")
    for e in D["edges"]:
        P = np.array(e["points"])
        v = []
        for ref in REFS.values():
            _, _, rms, _ = E.edgelib.line_fit(undistort_to_pixels(P, *ref))
            v.append(rms)
        print(f"  {e['id'].replace(REGION + '_', ''):26s} n={len(P):4d} " + " / ".join(f"{x:.3f}" for x in v))


def main():
    D0 = load_prereview()
    D = copy.deepcopy(D0)
    summary, hom = review(D)
    vps = vp_checks({e["id"]: e for e in D["edges"]})
    changed = [e["id"] for e in D["edges"] if "review" in e]
    D["review"] = {"script": SCRIPT, "input": "work/cache/edges_cart310_prereview.json (output of 10_edges_cart310_trace.py)",
                   "summary": summary, "changed_edges": changed,
                   "board_homography_check": hom, "data_driven_vp_check": vps,
                   "crops": "work/cache/review_cart310/*_reviewed_strip.png"}
    save_json(D, SRC)
    render_overview(D)
    render_review_crops(D, changed)
    for s in summary:
        print("*", s)
    for k, v in hom.items():
        print(f"  board check {k:18s} " + "  ".join(f"{rn}: {c['coord']}={c['value_mm']} ({c['offset_px']:+.1f}px, H {c['homography_rms_px']})"
                                                   for rn, c in v.items()))
    for rn, v in vps.items():
        print(f"  VP check {rn}: vp_x {v['vp_x']} vp_y {v['vp_y']}")
        for k, x in v["edges"].items():
            print(f"      {k:26s} {x}")
    straightness_table(D0, "before review")
    straightness_table(D, "after review")


if __name__ == "__main__":
    main()
