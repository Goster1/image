"""Review fixes for stage 11 (markers). Keeps the schema of work/cache/markers_final.json and
results/detections.json; only adds fields / corrects labels. corners_px are NOT changed.

Findings handled here
1. occlusion[*].side in markers_final.json is indexed in the READ order of stage 11a, while
   side_valid / side_lines use the ArUco order -> add occlusion[*].side_aruco (+ side_order note) and a
   per-marker occluded_sides_aruco list.
2. id-confirmation verdicts of 310:5 and 310:8 said "unique in the whole dictionary" although on the
   fully visible bits 8 (310:5) and 2 (310:8) dictionary codes fit; they are unique only when the
   partially visible cells are used. 310:7: 9 codes fit the 9 fully visible bits, 2 the 13 bits incl.
   partial cells. Verdicts are rewritten with both numbers.
3. results/detections.json holds per-frame corners only in the jitter-corrected MEAN frame; the corners
   in each still's own pixel coordinates are added (per_frame_corners_still_px, exact inverse of the
   stored jitter model) so every photo's measurement is explicit.
4. rot_k: independent leave-one-sticker-out check (fit without the sticker, predict its corners under
   the 4 rotations) added as rot_k_check (all agree, see 12_review_markers_check.py).
5. NEW systematic: edge-location bias of the gradient-maximum criterion, measured from the code
   interiors (12_review_markers_bias.py, validated on synthetic codes in 12_review_markers_biassynth.py).
   Black/white edges are found ~0.5-1.3 px towards the BLACK side (black looks eroded), the same for
   the outer sides of the black square -> the stored corners lie INSIDE the true 90 mm square by |b|
   along both side normals. Per sticker b is fitted from the interior transitions with a model that
   absorbs errors of the corner homography (delta = b + s*(e_ax + g_ax*(k-3)) per axis), so it also works
   for partly occluded stickers (visible cells only). corners_px_bias_corrected = corners moved outward
   by |b| along both adjacent side normals (valid corners only). corners_px stays the raw measurement.
6. Effect of the bias on the joint fit is evaluated (work/cache/review_markers/review.json).

Run order (after 11_markers_c_fit.py, which rewrites the two files without the review fields):
    python3 12_review_markers_check.py ; python3 12_review_markers.py ; python3 12_review_markers_plot.py
(visual crops: 12_review_markers_render.py; bias per still: 12_review_markers_bias.py; synthetic
validation of the bias estimator: 12_review_markers_biassynth.py)
"""
import importlib
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from calib import Model, Problem
from common import CACHE, H, RESULTS, W, K_from, load_json, marker_center, marker_corners_3d, project, save_json

L = importlib.import_module("11_markers_lib")
CHK = importlib.import_module("12_review_markers_check")
OUT = f"{CACHE}/review_markers"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------------------------------
# interior transitions with visibility
# ------------------------------------------------------------------------------------------

def transitions(img, quad, grid, vis6, step=0.01, sigma_px=0.8):
    """Black/white transitions inside the code (cell boundaries k = 1..5) along lines of constant v
    (axis 0) and constant u (axis 1), only between visible cells. quad in ArUco order.
    Returns array rows (axis, k, s, delta_cells, cell_px)."""
    Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32),
                                     np.asarray(quad, np.float32)).astype(float)
    cell = np.linalg.norm(np.roll(quad, -1, 0) - quad, axis=1).mean() / 6.0
    t = np.arange(-0.6, 6.6 + 1e-9, step)
    rows = []
    for axis in (0, 1):
        g = grid if axis == 0 else grid.T
        vv = vis6 if axis == 0 else vis6.T
        for lc in range(1, 5):
            for fv in (0.3, 0.4, 0.5, 0.6, 0.7):
                w = lc + fv
                uv = np.column_stack([t, np.full_like(t, w)]) if axis == 0 else np.column_stack([np.full_like(t, w), t])
                q = np.column_stack([uv, np.ones(len(uv))]) @ Hm.T
                P = q[:, :2] / q[:, 2:3]
                prof = map_coordinates(img, [P[:, 1], P[:, 0]], order=3, mode="nearest")
                ds = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))
                der = gaussian_filter1d(prof, sigma_px / ds, order=1)
                for k in range(1, 6):
                    a, b = g[lc, k - 1], g[lc, k]
                    if a == b or not (vv[lc, k - 1] and vv[lc, k]):
                        continue
                    s = 1 if b == 1 else -1
                    sel = np.where(np.abs(t - k) < 0.4)[0]
                    dd = s * der[sel]
                    j = int(np.argmax(dd))
                    if j == 0 or j == len(sel) - 1:
                        continue
                    y0, y1, y2 = dd[j - 1], dd[j], dd[j + 1]
                    den = y0 - 2 * y1 + y2
                    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                    u = t[sel[j]] + off * step
                    rows.append((axis, k, s, s * (u - k), cell))
    return np.array(rows, float)


def fit_bias(R):
    """delta = b + s*(e_ax + g_ax*(k-3)) ; returns b (cells), se, rms, plus the 1- and 2-parameter
    variants (valid only when the quad corners are measured, i.e. all 4 corners valid)."""
    ax, k, s, d = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    A = np.column_stack([np.ones_like(k), s * (ax == 0), s * (ax == 0) * (k - 3), s * (ax == 1), s * (ax == 1) * (k - 3)])
    keep = np.linalg.norm(A, axis=0) > 0
    A = A[:, keep]
    x, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ x
    dof = max(len(d) - A.shape[1], 1)
    C = np.linalg.pinv(A.T @ A) * np.sum(r ** 2) / dof
    out = dict(b5=float(x[0]), b5_se=float(np.sqrt(C[0, 0])), rms5=float(np.sqrt(np.mean(r ** 2))), n=int(len(d)),
               n_s_plus=int((s > 0).sum()), n_s_minus=int((s < 0).sum()))
    a1 = 1 + s * (1 - k / 3)
    out["b1"] = float(np.sum(a1 * d) / np.sum(a1 * a1))
    A2 = np.column_stack([np.ones_like(k), s * (1 - k / 3)])
    x2, *_ = np.linalg.lstsq(A2, d, rcond=None)
    out["b2_inner"], out["b2_outer"] = float(x2[0]), float(x2[1])
    return out


def visibility6(g_entry, quad_valid_sides, mid):
    """6x6 visibility (canonical orientation) from stage 11a (inner bits visible, read order) and the
    ArUco-order side validity (border cells)."""
    r = g_entry["rotation_read"]
    vis_read = np.array(g_entry["bits_visible_read"], bool)
    bits_read = np.array(g_entry["bits_read"], int)
    vis_in = np.rot90(vis_read, -r)
    bits_can = np.rot90(bits_read, -r)
    code = L.code_bits(mid)
    assert np.all(bits_can[vis_in] == code[vis_in]), "orientation mapping read->canonical inconsistent"
    v6 = np.zeros((6, 6), bool)
    v6[1:5, 1:5] = vis_in
    sv = quad_valid_sides  # ArUco sides: 0 v=0 (top row), 1 u=6 (right col), 2 v=6 (bottom row), 3 u=0 (left col)
    for i in range(1, 5):
        v6[0, i] = sv[0] and vis_in[0, i - 1]
        v6[5, i] = sv[2] and vis_in[3, i - 1]
        v6[i, 5] = sv[1] and vis_in[i - 1, 3]
        v6[i, 0] = sv[3] and vis_in[i - 1, 0]
    return v6


def corner_shift(m, i, d):
    """Displacement of corner i when both adjacent side lines move outward by d px."""
    q = np.array(m["corners_px"], float)
    c = q.mean(0)
    ns = []
    for si in ((i - 1) % 4, i):
        sl = m["side_lines"][si]
        dd = np.array(sl["direction"], float)
        n = np.array([-dd[1], dd[0]])
        mid = 0.5 * (q[si] + q[(si + 1) % 4])
        if np.dot(mid - c, n) < 0:
            n = -n
        ns.append(n)
    return np.linalg.solve(np.array(ns), np.array([d, d]))


# ------------------------------------------------------------------------------------------
# fits
# ------------------------------------------------------------------------------------------

def fit_points(pts):
    pr = Problem(Model(dict(f="single", pp="free", dist=["k1", "k2"])), pts, carts=[80, 310])
    r = pr.solve(CHK.seed(pts))
    res = pr.predict(r.x) - pr.uv
    K, dist, _ = pr.split(r.x)
    e = np.linalg.norm(res, axis=1)
    lev = {}
    for p, ee in zip(pts, e):
        z = marker_center(p["id"], p["cart"])[2]
        lev.setdefault({0: "top", -195: "A", -595: "B", -995: "C"}[int(z)], []).append(ee)
    return dict(n=len(pts), rms=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()), f=float(K[0, 0]), cx=float(K[0, 2]),
                cy=float(K[1, 2]), k1=float(dist[0]), k2=float(dist[1]),
                per_level={k: float(np.sqrt(np.mean(np.square(v)))) for k, v in lev.items()})


def pts_from(M, key_uv, subset=None):
    pts = []
    for m in M:
        if subset is not None and not subset(m):
            continue
        X = marker_corners_3d(m["id"], m["cart"], m["rot_k"])
        for j in range(4):
            if m["corner_valid"][j]:
                pts.append(dict(cart=m["cart"], id=m["id"], corner=j, X=X[j], uv=np.array(m[key_uv][j], float)))
    return pts


# ------------------------------------------------------------------------------------------

def main():
    M = load_json(f"{CACHE}/markers_final.json")
    G = {(g["cart"], g["id"]): g for g in load_json(f"{CACHE}/markers_guided.json")}
    det = load_json(f"{RESULTS}/detections.json")
    chk = load_json(f"{OUT}/check.json") if os.path.exists(f"{OUT}/check.json") else None
    mean = L.mean_gray()
    review = dict(bias={}, fits={}, relabels=[], verdicts={})

    # ---------------- edge bias per sticker
    for m in M:
        key = (m["cart"], m["id"])
        v6 = visibility6(G[key], m["side_valid"], m["id"])
        q = np.array(m["corners_px"], float)
        R = transitions(mean, q, L.code_grid(m["id"]), v6)
        fb = fit_bias(R)
        cell = float(R[0, 4])
        fb.update(cell_px=cell, b_px=fb["b5"] * cell, b_se_px=fb["b5_se"] * cell, all_valid=bool(all(m["corner_valid"])),
                  b1_px=fb["b1"] * cell, b2_inner_px=fb["b2_inner"] * cell, b2_outer_px=fb["b2_outer"] * cell)
        review["bias"][f"{key[0]}:{key[1]}"] = fb
        print(f"{key[0]:4d}:{key[1]:<4d} n={fb['n']:3d} (+{fb['n_s_plus']}/-{fb['n_s_minus']}) cell={cell:5.2f}  "
              f"b={fb['b_px']:+.3f}+-{fb['b_se_px']:.3f} px (rms {fb['rms5'] * cell:.3f})"
              + (f"   [all corners valid: b1={fb['b1_px']:+.3f} b_inner={fb['b2_inner_px']:+.3f} b_outer={fb['b2_outer_px']:+.3f}]"
                 if fb["all_valid"] else "   [partly occluded]"))
    full = np.array([v["b_px"] for v in review["bias"].values() if v["all_valid"]])
    part = np.array([v["b_px"] for v in review["bias"].values() if not v["all_valid"]])
    bo = np.array([v["b2_outer_px"] for v in review["bias"].values() if v["all_valid"]])
    bi = np.array([v["b2_inner_px"] for v in review["bias"].values() if v["all_valid"]])
    review["bias_summary"] = dict(
        full_mean_px=float(full.mean()), full_std_px=float(full.std(ddof=1)), partial_mean_px=float(part.mean()),
        partial_std_px=float(part.std(ddof=1)), inner_mean_px=float(bi.mean()), outer_mean_px=float(bo.mean()),
        outer_minus_inner_mean_px=float((bo - bi).mean()), outer_minus_inner_sem_px=float((bo - bi).std(ddof=1) / np.sqrt(len(bo))))
    print("bias summary:", {k: round(v, 3) for k, v in review["bias_summary"].items()})

    # ---------------- corrected corners
    for m in M:
        key = f"{m['cart']}:{m['id']}"
        b = review["bias"][key]["b_px"]
        corr = []
        for i in range(4):
            if m["corner_valid"][i]:
                corr.append((np.array(m["corners_px"][i], float) + corner_shift(m, i, -b)).tolist())
            else:
                corr.append(None)
        m["corners_px_bias_corrected"] = corr
        m["edge_bias_px"] = dict(value=b, se=review["bias"][key]["b_se_px"], n_transitions=review["bias"][key]["n"],
                                 cell_px=review["bias"][key]["cell_px"])

    # ---------------- fits: raw vs bias-corrected corners
    subsets = dict(all=None, top=lambda m: m["role"].startswith("TOP"), shelves=lambda m: not m["role"].startswith("TOP"))
    for name, sub in subsets.items():
        for key_uv in ("corners_px", "corners_px_bias_corrected"):
            r = fit_points(pts_from(M, key_uv, sub))
            review["fits"][f"{name}:{key_uv}"] = r
            print(f"fit {name:8s} {key_uv:26s} n={r['n']:2d} rms={r['rms']:.3f} max={r['max']:.2f} f={r['f']:.1f} "
                  f"cx={r['cx']:.1f} cy={r['cy']:.1f} k1={r['k1']:+.4f} k2={r['k2']:+.4f} levels="
                  + " ".join(f"{k}:{v:.2f}" for k, v in r["per_level"].items()))

    # ---------------- labels / verdicts
    for m in M:
        key = (m["cart"], m["id"])
        r = G[key]["rotation_read"]
        occ_aruco = []
        for o in m["occlusion"]:
            o["side_aruco"] = int((o["side"] + r) % 4)
            o["side_order_note"] = "'side' = index in the read order of stage 11a; 'side_aruco' = ArUco order (side i joins corners i and i+1)"
            occ_aruco.append(o["side_aruco"])
            review["relabels"].append(f"{key[0]}:{key[1]} occlusion side {o['side']} (read) -> {o['side_aruco']} (ArUco)")
        # consistency: occluded ArUco sides must be invalid sides
        for s in occ_aruco:
            if m["side_valid"][s]:
                review["relabels"].append(f"WARNING {key[0]}:{key[1]} occluded side {s} is marked valid")
        m["occluded_sides_aruco"] = sorted(set(occ_aruco))
        ic = m["id_confirmation"]
        n_full, n0_full = ic["n_visible_bits"], ic["n_dictionary_codes_at_distance0"]
        wp = ic["with_partially_visible_cells"]
        n_ext, n0_ext = wp["n_bits"], wp["n_dictionary_codes_at_distance0"]
        cart_ids = "other ids used on the carts differ by >= %d fully visible bits" % ic["min_hamming_other_cart_ids"]
        if n0_full == 1:
            v = f"confirmed: unique in the whole dictionary on the {n_full} fully visible bits; {cart_ids}"
        elif n0_ext == 1:
            v = (f"confirmed within the cart ids ({cart_ids}); on the {n_full} fully visible bits {n0_full - 1} other dictionary "
                 f"code(s) also fit, unique in the whole dictionary only when the partially visible cells are used ({n_ext} bits)")
        else:
            v = (f"confirmed within the cart ids ({cart_ids}); {n0_full - 1} other dictionary code(s) fit the {n_full} fully visible "
                 f"bits and {n0_ext - 1} the {n_ext} bits incl. partially visible cells (none of them is used on the carts)")
        if v != ic.get("verdict"):
            review["verdicts"][f"{key[0]}:{key[1]}"] = dict(old=ic.get("verdict"), new=v)
        ic["verdict"] = v
        tt = ic["template_test"]
        ic["template_test"]["note"] = ("tie: another dictionary code matches the visible part exactly as well"
                                       if abs(tt["ncc_best_other"] - tt["ncc_expected"]) < 1e-9 else
                                       "expected id correlates best")
    for k, v in review["verdicts"].items():
        print("verdict", k, ":", v["new"])

    # ---------------- rot_k check
    loo = {r["marker"]: r for r in chk["rot_k_loo"]} if chk else {}
    for m in M:
        r = loo.get(f"{m['cart']}:{m['id']}")
        if r is not None:
            m["rot_k_check"] = dict(how="leave-one-sticker-out fit (f, pp, k1, k2, poses) predicts the sticker's valid corners under the 4 rotations",
                                    loo_rms_per_rot_px=r["loo_rms_per_rot"], loo_best=r["loo_best"], agrees=bool(r["agree"]))
    save_json(M, f"{CACHE}/markers_final.json")

    # ---------------- detections.json
    J = np.array(det["frame_jitter"]["params_per_frame_ax0_ax1_ay0_ay1"], float)
    byk = {(m["cart"], m["id"]): m for m in M}
    for d in det["markers"]:
        m = byk[(d["cart"], d["id"])]
        still = []
        for f, fr in enumerate(d["per_frame_corners_px"]):
            row = []
            for p in fr:
                if p is None:
                    row.append(None)
                else:
                    row.append(L.mean_to_frame(np.array(p, float)[None], f)[0].tolist())
            still.append(row)
        d["per_frame_corners_still_px"] = still
        d["corners_px_bias_corrected"] = m["corners_px_bias_corrected"]
        d["edge_bias_px"] = m["edge_bias_px"]
        d["occluded_sides_aruco"] = m["occluded_sides_aruco"]
        d["id_confirmation"] = m["id_confirmation"]
        if "rot_k_check" in m:
            d["rot_k_check"] = dict(loo_rms_per_rot_px=m["rot_k_check"]["loo_rms_per_rot_px"], agrees=m["rot_k_check"]["agrees"])
        d["rot_k_fit_rms_per_rot_px"] = m["rot_k_fit"]["marker_rms_per_rot"]
    bs = review["bias_summary"]
    det["per_frame_note"] = ("per_frame_corners_px: per-still measurement mapped into the common mean frame (jitter model); "
                             "per_frame_corners_still_px: the same measurement in the still's own pixel coordinates (index = frames[]). "
                             "corners_std_px is the frame-to-frame scatter of one still (std of the 7-still mean is smaller by sqrt(7)).")
    det["edge_bias"] = dict(
        description=("Systematic edge-location bias of the corner measurement, measured from the black/white transitions inside every "
                     "code (known to lie on the 15 mm cell grid): the gradient-maximum edge lies b px towards the WHITE side "
                     "(b < 0: towards black, the black square looks eroded). Validated on synthetic codes (zero bias for a linear "
                     "tone curve, correct recovery of a gamma-induced bias). The outer sides of the black square show the same "
                     "bias as the interior edges within the errors (mean outer - inner = %+.2f +- %.2f px), so corners_px lie "
                     "inside the true 90 mm square by ~|b| along both side normals (~1.4 |b| along the diagonal). "
                     "corners_px_bias_corrected = valid corners moved outward by |b| along both adjacent side normals "
                     "(per-sticker b). corners_px is left as the raw measurement." % (bs["outer_minus_inner_mean_px"], bs["outer_minus_inner_sem_px"])),
        mean_b_px_fully_visible=bs["full_mean_px"], std_b_px_fully_visible=bs["full_std_px"],
        mean_b_px_partly_occluded=bs["partial_mean_px"],
        systematic_uncertainty_px=0.15,
        effect_on_joint_fit={k: dict(rms=v["rms"], f=v["f"], cx=v["cx"], cy=v["cy"], k1=v["k1"], k2=v["k2"]) for k, v in review["fits"].items()})
    save_json(det, f"{RESULTS}/detections.json")
    save_json(review, f"{OUT}/review.json")
    print("wrote markers_final.json, results/detections.json, review.json")


if __name__ == "__main__":
    main()
