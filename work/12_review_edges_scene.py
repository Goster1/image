"""Independent review of the scene edge set (work/cache/edges_scene.json, written by 10_edges_scene_final.py).
Applies the review decisions to the JSON (same schema; every changed entry gets a "review" record),
re-renders results/edges_scene.png and writes reviewed strips to work/cache/review_scene/.

Run order: 10_edges_scene_final.py -> 12_review_edges_scene.py.  The final-script output is copied once to
cache/edges_scene_prereview.json and the review is always applied to that copy (idempotent).

Findings that drive the changes (crops in work/cache/review_scene/, numbers printed by this script):
 1. DUPLICATES across regions.  scene_wall_railA is the same image edge as cart80_x_back_rail_out (signed
    normal offset median -0.01 px, 148/150 points within 3 px).  It is not a wall conduit but the back top
    rail of cart 80 (emerges under the X=0 board, same pencil as the other back members, 0.7 deg from the
    cart-80 X vanishing direction vs 5 deg for the real wall pipes).  The cart-80 review kept its copy and
    left the resolution to the scene set -> here verified 'no', duplicate_of = cart80_x_back_rail_out.
    scene_blueL_top / scene_blueL_bottom are point-identical to cart310_floor_tape_blue_upper / _lower
    (both copies in use -> double weight).  Floor lines belong to the scene set and the scene ids are the
    ones used for the floor VP groups (30_lines_methods.py), so the scene copies are kept and flagged with
    duplicate_in_other_region; the cart-310 copies must be dropped (not editable from this region).
 2. scene_beam_lower: 'world_vertical' although it is the only accepted edge of its object (upper edge
    rejected, left continuation excluded) - the stage's own rule needs several edges of one object; it is
    1.5-2.6 deg off the cart-Z nadir and 0.6-0.9 deg off the common vanishing point of the other scene
    verticals (3 reference lenses) -> direction 'unknown' (straightness only).
 3. scene_blue_left_upper: last points (next to the paper notice) hook by +0.6..+1.0 px; the rest bows
    ~0.5 px after undistortion with every reference lens and its direction differs by ~0.9 deg from the
    lower part of the same column edge -> hook trimmed, verified yes -> partly.
 4. scene_jointB_V_right: chipped joint edge, first piece on a dark blob, systematic +-0.5 px wave over
    80 px after undistortion (partner edge 0.2 px) -> verified 'no'.
 5. Local defects confirmed on x11 zoom windows (win_trims.png) -> fixed coordinate trims:
    tapeH_bottom: debris against the edge (x 1025-1052, +0.9 px) and a dark notch (x 1080-1092);
    tapeV_left: kink/halo change next to the T-junction / handle shadow (y < 746);
    tapeH_top: first points at the T-junction (x < 880); blueH_top: first points (x < 1071);
    frame_br: notch of a dark object (y 944-990); pole_top: band rim at the right end (x > 1754);
    leg_right: last point; joint450: left end next to cart 310 (x < 876, +0.8 px step).
 6. Generic cleaning (all edges, not residual-driven w.r.t. any lens): points within 3 px of the image
    border removed; robust smooth fit in arc length (cubic, quartic for > 300 px),
    |res| > max(0.5 px, 3 * 1.4826 * MAD) removed, two passes.
 7. pair_of set for the second boundary of the same member (tape / pipe / leg / joint / bollard / column):
    their shapes are not independent (do not treat them as independent lines in a bootstrap).
Checks that did NOT lead to changes (see "review" block of the JSON):
 * scene_rack_tube_a (609 px, far-left border): hold-out test - a plumb-line fit (k1,k2, free centre) on
   all OTHER edges predicts its shape to 0.38 px rms (bow -0.6 px over 609 px, point noise 0.34 px),
   whereas the centre-fixed fit leaves -3.4 px.  A physically bent rail would have to coincide with an
   independent prediction -> kept ('partly'), it mainly discriminates centre-fixed vs centre-free models.
 * world verticals: blue column (3 edges) and bollard (2 silhouettes, converging at ~760 px from the
   bollard = the nadir distance) are consistent with one vanishing point to <= 0.8 px at their ends; this
   common scene VP lies ~30-35 px (~1.4 deg) from the nadir of the four cart-Z edges (which themselves
   scatter by 4-8 px): observation for the VP / verticality question.
"""
from __future__ import annotations

import copy
import os

import cv2
import numpy as np

import edgelib
import reviewscene_lib as R
from common import CACHE, RESULTS, W, H, load_json, save_json, undistort_points
from linedata import load_edges
from lineselfcal import LineCal

REGION = "scene"
SRC = f"{CACHE}/edges_{REGION}.json"
PRE = f"{CACHE}/edges_{REGION}_prereview.json"
OUT = f"{CACHE}/review_{REGION}"
SCRIPT = "work/12_review_edges_scene.py"
os.makedirs(OUT, exist_ok=True)

DIRCOL = {"cartX": (0, 0, 255), "cartY": (0, 200, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
          "floor_plane": (0, 255, 255), "horizontal_other": (0, 160, 255), "unknown": (200, 200, 200)}

# ---------------------------------------------------------------- decisions (see docstring)
# fixed trims: id -> (description, function(P) -> boolean mask of points to REMOVE)
TRIMS = {
    "scene_tapeH_bottom": ("debris against the edge x 1025-1052 (+0.9 px) and dark notch x 1080-1092",
                           lambda P: ((P[:, 0] > 1025) & (P[:, 0] < 1052)) | ((P[:, 0] > 1080) & (P[:, 0] < 1092))),
    "scene_tapeV_left": ("kink / halo change next to the T-junction and the handle shadow (y < 746)",
                         lambda P: P[:, 1] < 746),
    "scene_tapeH_top": ("first points at the T-junction (x < 880, +0.6 px hook)", lambda P: P[:, 0] < 880),
    "scene_blueH_top": ("first points (x < 1071, -0.45 px hook)", lambda P: P[:, 0] < 1071),
    "scene_frame_br": ("notch of a dark object touching the edge (y 944-990)",
                       lambda P: (P[:, 1] > 944) & (P[:, 1] < 990)),
    "scene_pole_top": ("band rim at the right end of the yellow band (x > 1754, +0.45 px)", lambda P: P[:, 0] > 1754),
    "scene_leg_right": ("last point (-0.4 px, end of the member)", lambda P: P[:, 1] > 300.5),
    "scene_joint450": ("left end next to cart 310 (x < 876): +0.8 px step where the joint enters the dark area",
                       lambda P: P[:, 0] < 876),
    "scene_blue_left_upper": ("hook next to the paper notice (y > 821, +0.6..+1.0 px)", lambda P: P[:, 1] > 821),
}

SET_NO = {
    "scene_wall_railA": "duplicate of cart80_x_back_rail_out (same image edge, signed offset -0.01 px): this is the "
                        "back top rail of cart 80, not a wall conduit; the cart-80 copy is the one to use",
    "scene_jointB_V_right": "chipped joint edge: first piece sits on a dark blob, systematic +-0.5 px wave over 80 px "
                            "after undistortion with every reference lens (partner edge 0.2 px) -> not usable",
}
DOWNGRADE = {
    "scene_blue_left_upper": "bows ~0.5 px after undistortion with every reference lens (lens-independent) and its "
                             "direction differs by ~0.9 deg from the lower part of the same column edge -> "
                             "straightness / direction only to ~0.5 px",
}
RELABEL = {
    "scene_beam_lower": ("unknown", "world_vertical -> unknown: only accepted edge of its object (upper edge rejected, "
                                    "left continuation excluded), 1.5-2.6 deg off the cart-Z nadir and 0.6-0.9 deg off "
                                    "the common VP of the other scene verticals -> straightness only"),
}
PAIRS = {"scene_tapeV_right": "scene_tapeV_left", "scene_tapeH_bottom": "scene_tapeH_top",
         "scene_blueL_bottom": "scene_blueL_top", "scene_rpost_rodR": "scene_rpost_rod",
         "scene_leg_right": "scene_leg_left", "scene_jointB_V_right": "scene_jointB_V_left",
         "scene_pole_bottom": "scene_pole_top", "scene_blue_left_lower": "scene_blue_left_upper",
         "scene_blue_right": "scene_blue_left_upper"}
DUP_OTHER = {"scene_blueL_top": "cart310_floor_tape_blue_upper", "scene_blueL_bottom": "cart310_floor_tape_blue_lower"}
BORDER = 3.0


# ---------------------------------------------------------------- helpers
def load_prereview():
    D = load_json(SRC)
    if "review" not in D:
        save_json(D, PRE)
    return load_json(PRE)


def robust_mask(P, passes=2):
    keep = np.ones(len(P), bool)
    for _ in range(passes):
        Q = P[keep]
        if len(Q) < 12:
            break
        s = R.arclen(Q)
        deg = 4 if s[-1] > 300 else 3
        px = np.polyfit(s, Q[:, 0], deg)
        py = np.polyfit(s, Q[:, 1], deg)
        C = np.column_stack([np.polyval(px, s), np.polyval(py, s)])
        _, nrm = R.normals_along(s, px, py)
        r = np.sum((Q - C) * nrm, axis=1)
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        bad = np.abs(r) > max(0.5, 3 * mad)
        if not bad.any():
            break
        idx = np.nonzero(keep)[0]
        keep[idx[bad]] = False
    return keep


def refresh_stats(e, P, S):
    e["points"] = [[round(float(x), 3), round(float(y), 3)] for x, y in P]
    e["strength"] = [round(float(v), 2) for v in S]
    _, _, rms, _ = edgelib.line_fit(P)
    bow, _ = edgelib.sagitta(P)
    e["chord_dev_rms_px"] = round(float(rms), 3)
    e["chord_bow_px"] = round(float(bow), 3)


def reviewed_strip(e, P0, keep, fn):
    """Straightened strip along the ORIGINAL points: red = kept, cyan = removed."""
    img, s, o, offs = R.strip(P0, hw=6.0, zoom=8, step=0.5)
    img = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    zx = 2 / 0.5
    for si, oi, k in zip(s, o, keep):
        c = (int(round(si * zx)), int(round((oi + 6.0) * 8)))
        cv2.circle(img, c, 2, (0, 0, 255) if k else (255, 255, 0), -1)
    pad = np.zeros((22, img.shape[1], 3), np.uint8)
    cv2.putText(pad, f"{e['id']}: red kept, cyan removed ({(~keep).sum()} of {len(keep)})", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(fn, np.vstack([pad, img]))


def folded(K, d):
    """True if r_d(r_u) = r_u (1 + k1 r^2 + k2 r^4 + k3 r^6) has its maximum below the distorted radius of the
    farthest image corner (the model then has no inverse there; the Jacobian-scaled line residual goes to 0
    near the fold, which a fit can exploit)."""
    r = np.linspace(0, 3, 30001)
    rd = r * (1 + d[0] * r ** 2 + d[1] * r ** 4 + d[4] * r ** 6)
    dec = np.nonzero(np.diff(rd) <= 0)[0]
    rmax = rd[dec[0]] if len(dec) else np.inf
    cor = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)
    rc = np.max(np.hypot((cor[:, 0] - K[0, 2]) / K[0, 0], (cor[:, 1] - K[1, 2]) / K[1, 1]))
    return rmax < rc


def line_und(P, K, d):
    n = undistort_points(P, K, d)
    U = np.column_stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])
    c = U.mean(0)
    _, _, Vt = np.linalg.svd(U - c)
    return U, c, Vt[0]


def vp_ls(lines):
    A = np.array([[-dd[1], dd[0]] for _, c, dd in lines])
    b = np.array([np.array([-dd[1], dd[0]]) @ c for _, c, dd in lines])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def ang_to(c, dd, v):
    to = v - c
    a = np.degrees(np.arctan2(dd[0] * to[1] - dd[1] * to[0], dd @ to))
    return (a + 90) % 180 - 90


# ---------------------------------------------------------------- checks (diagnostics only)
def vertical_check(edges):
    carts = []
    for r in ("cart80", "cart310"):
        carts += [e for e in load_json(f"{CACHE}/edges_{r}.json")["edges"] if e["direction"] == "cartZ" and e["verified"] != "no"]
    out = {}
    for name, (K, d) in R.REFS.items():
        if name == "combined":
            continue
        LZ = [line_und(np.array(e["points"]), K, d) for e in carts]
        vz = vp_ls(LZ)
        wv = [e for e in edges if e["direction"] == "world_vertical" and e["verified"] != "no"]
        LW = [line_und(np.array(e["points"]), K, d) for e in wv]
        vw = vp_ls(LW)
        per = {}
        for e, (U, c, dd) in zip(wv, LW):
            L = float(np.ptp((U - c) @ dd))
            per[e["id"]] = {"deg_to_cartZ_vp": round(float(ang_to(c, dd, vz)), 2),
                            "deg_to_scene_vp": round(float(ang_to(c, dd, vw)), 2),
                            "end_offset_scene_vp_px": round(float(np.radians(ang_to(c, dd, vw)) * L / 2), 2)}
        out[name] = {"nadir_cartZ_px": np.round(vz, 1).tolist(), "vp_scene_verticals_px": np.round(vw, 1).tolist(),
                     "cartZ_line_dist_px": [round(float(np.array([-dd[1], dd[0]]) @ (vz - c)), 1) for _, c, dd in LZ],
                     "per_edge": per}
    return out


def holdout_check(edges):
    """Plumb-line (k1,k2) fit on all other verified edges, prediction of each scene edge (distorted px)."""
    base = [e for e in load_edges(verified_only=True) if e["region"] != "scene"]
    mine = []
    for e in edges:
        if e["verified"] == "no" or not e.get("straight_3d", False):
            continue
        P = np.array(e["points"], float)
        mine.append(dict(id=e["id"], region="scene", cart=None, direction=e["direction"], model_line=None,
                         what=e["what"], points=P, length=float(np.sum(np.hypot(*np.diff(P, axis=0).T))),
                         cls=e["class"]))
    # the cart-310 copies of the blue floor line are duplicates of scene edges -> excluded here
    base = [e for e in base if e["id"] not in DUP_OTHER.values()]
    res = {}
    for cf in (False, True):
        tag = "centre_free" if cf else "centre_fixed"
        for tgt in ("scene_rack_tube_a", "scene_frame_br", "scene_rpost_rod", "scene_tapeV_left",
                    "scene_blueL_top", "scene_trolley_rodA", "scene_roller_br1"):
            others = base + [m for m in mine if m["id"] != tgt]
            lc = LineCal(others, "plumb", dist_free=("k1", "k2"), centre_free=cf, f0=1394.0)
            best, nfold = None, 0
            starts = ((959.5, 539.5), (930.0, 430.0), (930.0, 480.0)) if cf else ((959.5, 539.5),)
            for c0 in starts:  # several starts: the centre-free fit has local minima
                if cf:
                    lc.centre_fixed = np.array(c0)
                r = lc.solve(lc.x0(), loss="huber", f_scale=0.5)
                Kr, dr = lc.unpack(r.x)[:2]
                if folded(Kr, dr):  # radial function not monotonic inside the image: reject
                    nfold += 1
                    continue
                if best is None or r.cost < best.cost:
                    best = r
            if best is None:
                res.setdefault(tgt, {})[tag] = {"all_starts_folded": True}
                continue
            K, d = lc.unpack(best.x)[:2]
            P = [m for m in mine if m["id"] == tgt][0]["points"]
            rr = R.straight_resid_distorted(P, K, d)
            s = R.arclen(P)
            q = np.polyfit(s - s.mean(), rr, 2)
            res.setdefault(tgt, {})[tag] = {
                "n_starts_rejected_folded": nfold,
                "centre_px": [round(float(K[0, 2]), 1), round(float(K[1, 2]), 1)], "k1": round(float(d[0]), 4),
                "k2": round(float(d[1]), 4), "pred_rms_px": round(float(np.sqrt(np.mean(rr ** 2))), 2),
                "pred_bow_px": round(float(q[0] * (np.ptp(s) / 2) ** 2), 2)}
    return res


def floor_line_check(edges):
    E = {e["id"]: e for e in edges}
    out = {}
    for name, (K, d) in R.REFS.items():
        if name == "combined":
            continue
        o = {}
        for i in ("scene_blueL_top", "scene_blueL_bottom", "scene_tapeH_top", "scene_tapeH_bottom", "scene_blueH_top"):
            _, c, dd = line_und(np.array(E[i]["points"]), K, d)
            dd = dd if dd[0] > 0 else -dd
            o[i] = round(float(np.degrees(np.arctan2(dd[1], dd[0]))), 2)
        for a, b in (("scene_blueL_top", "scene_blueH_top"), ("scene_blueL_bottom", "scene_tapeH_top")):
            UA, cA, dA = line_und(np.array(E[a]["points"]), K, d)
            UB, _, _ = line_und(np.array(E[b]["points"]), K, d)
            n = np.array([-dA[1], dA[0]])
            o[f"{b}_offset_from_{a}_line_px"] = round(float(np.mean((UB - cA) @ n)), 1)
        out[name] = o
    return out


# ---------------------------------------------------------------- main
def main():
    D = load_prereview()
    edges = D["edges"]
    changed, summary = [], []
    for e in edges:
        P0 = np.array(e["points"], float)
        S0 = np.array(e["strength"], float)
        ch = []
        keep = np.ones(len(P0), bool)
        border = (P0[:, 0] < BORDER) | (P0[:, 0] > W - 1 - BORDER) | (P0[:, 1] < BORDER) | (P0[:, 1] > H - 1 - BORDER)
        if border.any():
            keep &= ~border
            ch.append(f"points within {BORDER:.0f} px of the image border: removed {border.sum()}")
        if e["id"] in TRIMS:
            desc, fn = TRIMS[e["id"]]
            m = fn(P0) & keep
            keep &= ~m
            ch.append(f"fixed trim ({desc}): removed {m.sum()} of {len(P0)} points")
        idx = np.nonzero(keep)[0]
        rk = robust_mask(P0[idx])
        if (~rk).any():
            keep[idx[~rk]] = False
            ch.append(f"robust smooth fit (deg 3/4), |res| > max(0.5 px, 3 MAD): removed {(~rk).sum()} points")
        if e["id"] in SET_NO:
            e["verified"] = "no"
            e["verification_note"] += f" [review: verified -> no: {SET_NO[e['id']]}]"
            ch.append("verified -> no")
        if e["id"] == "scene_wall_railA":
            e["duplicate_of"] = "cart80_x_back_rail_out"
            e["class"], e["cart"], e["direction"] = "cart_structure", 80, "cartX"
            e["what"] += " [review: physically the back top rail of cart 80 (cart X direction), duplicate of cart80_x_back_rail_out]"
            ch.append("class/cart/direction -> cart_structure / 80 / cartX (duplicate, not used)")
        if e["id"] in DOWNGRADE and e["verified"] == "yes":
            e["verified"] = "partly"
            e["verification_note"] += f" [review: verified yes -> partly: {DOWNGRADE[e['id']]}]"
            ch.append("verified yes -> partly")
        if e["id"] in RELABEL:
            new, why = RELABEL[e["id"]]
            e["direction"] = new
            e["verification_note"] += f" [review: {why}]"
            ch.append(f"direction -> {new}")
        if e["id"] in PAIRS:
            e["pair_of"] = PAIRS[e["id"]]
            ch.append(f"pair_of = {PAIRS[e['id']]}")
        if e["id"] in DUP_OTHER:
            e["duplicate_in_other_region"] = DUP_OTHER[e["id"]]
            e["verification_note"] += (f" [review: point-identical copy exists as {DUP_OTHER[e['id']]} in the cart-310 set; "
                                       "keep this scene copy (used in the floor VP groups) and drop the cart-310 copy]")
            ch.append(f"duplicate_in_other_region = {DUP_OTHER[e['id']]} (drop that copy)")
        if not keep.all():
            refresh_stats(e, P0[keep], S0[keep])
            crop = f"{OUT}/{e['id'].replace('scene_', '')}_reviewed_strip.png"
            reviewed_strip(e, P0, keep, crop)
        else:
            crop = None
        if ch:
            e["review"] = {"changes": ch}
            if crop:
                e["review"]["crop"] = "work/cache/" + os.path.relpath(crop, CACHE)
            changed.append(e["id"])
            summary.append(f"{e['id']}: " + "; ".join(ch))

    vchk = vertical_check(edges)
    fchk = floor_line_check(edges)
    hchk = holdout_check(edges)
    D["review"] = {
        "script": SCRIPT,
        "input": "work/cache/edges_scene_prereview.json (output of 10_edges_scene_final.py)",
        "summary": summary,
        "changed_edges": changed,
        "duplicates_with_other_regions": {
            "scene_wall_railA": "== cart80_x_back_rail_out (resolved here: scene copy verified no)",
            "scene_blueL_top": "== cart310_floor_tape_blue_upper (NOT resolved here: drop the cart-310 copy)",
            "scene_blueL_bottom": "== cart310_floor_tape_blue_lower (NOT resolved here: drop the cart-310 copy)",
            "scene_wall_railB / scene_rpost_rod / scene_rpost_rodR": "cart-80 copies already verified no"},
        "observations": [
            "rack tube (far-left border): hold-out plumb-line prediction from all other edges is consistent with a "
            "straight member when the distortion centre is free (see holdout_check); it discriminates centre-fixed vs "
            "centre-free models (centre fixed leaves a ~-3 px bow)",
            "the centre-free plumb-line fit (lineselfcal, k1,k2) has a spurious minimum: centre ~(850,519), k1 ~-0.245, "
            "k2 ~0, whose radial function FOLDS inside the image (max r_d 0.77 < bottom-right corner r_d 0.87): near a "
            "fold the undistortion Jacobian diverges and the Jacobian-scaled residual r_u/|J^T n| goes to 0, so the "
            "corner edges (rollers, frame_br) stop costing anything. Several edge subsets converge there from the "
            "image-centre start. Any line / combined fit must reject or penalise lenses that are not invertible up "
            "to the image corners (this review rejects folded solutions and uses 3 starts)",
            "lens files that are folded inside the image (no inverse at the far corner): method_combined (fold at "
            "r_d 0.773 < corner 0.891), combined_k1k2k3 (0.655 < 0.901), combined_k1k2_ppfixed, combined_k1_only, "
            "markers_only (0.712 < 0.847); not folded: lines_joint, plumbline, vanishing, twocart, markershape",
            "scene world verticals (blue column 3 edges + bollard 2 silhouettes) meet in one VP to <= 0.8 px at their "
            "ends, but that VP is ~30-35 px (~1.4 deg) from the nadir of the four cart-Z edges (which scatter by 4-8 px "
            "themselves) - see vertical_check",
            "floor lines: the left blue line (x 172-434) and the blue line / white tape between the carts (x 872-1148) "
            "are parallel (0.4 deg) and nearly collinear (offset ~1-2 px across the 630 px gap under cart 310) with the "
            "centre-fixed plumb lens, but 1.6-1.8 deg apart and 12-15 px off-line with the lines_joint lens "
            "(pp (913,387)) - the floor_H VP group is therefore a strong, assumption-laden constraint (one painted line "
            "across the hall) - see floor_line_check",
            "near-vertical / near-horizontal edges show a periodic sawtooth of +-0.2..0.3 px with a period of ~2 px "
            "divided by the edge slope (e.g. ~43 px on the white tape, slope 0.046): colour-filter-array / pixel "
            "locking of the sub-pixel edge; zero-mean over long edges, but it is correlated noise, not white noise",
            "consequence seen here: with the method_combined lens the undistortion of the roller / frame_br points "
            "in the bottom-right corner diverges (straightness residuals 20-180 px in the reviewer renders)",
        ],
        "vertical_check": vchk,
        "floor_line_check": fchk,
        "holdout_check": hchk,
        "crops": "work/cache/review_scene/*_reviewed_strip.png (red kept, cyan removed); reviewer renders "
                 "work/cache/review_scene/<id>.png (context, straightened strip, residuals), win_*.png, ctx_*.png",
    }
    save_json(D, SRC)

    # overlay (accepted edges only = verified != no)
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    used = [e for e in edges if e["verified"] != "no"]
    for k, e in enumerate(used):
        col = DIRCOL[e["direction"]]
        P = np.array(e["points"], float)
        s = R.arclen(P)
        cut = np.nonzero(np.diff(s) > 10.0)[0]
        for F in np.split(P, cut + 1):
            if len(F) >= 2:
                cv2.polylines(img, [np.round(F * 8).astype(np.int32)], False, col, 1, cv2.LINE_AA, shift=3)
        c = P[int((0.3, 0.7)[k % 2] * (len(P) - 1))]
        lab = e["id"].replace("scene_", "") + ("" if e["verified"] == "yes" else "*")
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
        x0 = int(min(max(c[0] + 6, 2), W - 2 - tw))
        y0 = int(min(max(c[1] + 4, th + 3), H - 3))
        cv2.rectangle(img, (x0 - 1, y0 - th - 2), (x0 + tw + 1, y0 + 2), (0, 0, 0), -1)
        cv2.putText(img, lab, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 0.8, col, 1, cv2.LINE_AA)
    y = 20
    for k, col in DIRCOL.items():
        if k in {e["direction"] for e in used}:
            (tw, th), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (758, y - th - 3), (762 + tw, y + 3), (0, 0, 0), -1)
            cv2.putText(img, k, (760, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y += 18
    note = "scene edges after review (* = verified partly)"
    (tw, th), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (758, y - th - 3), (762 + tw, y + 3), (0, 0, 0), -1)
    cv2.putText(img, note, (760, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{RESULTS}/edges_scene.png", img)

    # printout
    print("changed:", len(changed))
    for s in summary:
        print("  ", s)
    by = {}
    for e in used:
        P = np.array(e["points"])
        L = float(np.ptp((P - P.mean(0)) @ edgelib.line_fit(P)[1]))
        b = by.setdefault(e["direction"], [0, 0.0])
        b[0] += 1
        b[1] += L
    print("used edges:", len(used), {k: (n, round(l)) for k, (n, l) in by.items()},
          "verified:", {v: sum(e["verified"] == v for e in used) for v in ("yes", "partly")})
    print("vertical check:")
    for k, v in vchk.items():
        print("  ", k, "nadir cartZ", v["nadir_cartZ_px"], "scene VP", v["vp_scene_verticals_px"], v["per_edge"])
    print("floor check:", fchk)
    print("hold-out:")
    for k, v in hchk.items():
        print("  ", k, v)


if __name__ == "__main__":
    main()
