"""Region cart310 (left cart No. 310): curated straight structural edges, traced sub-pixel on the
jitter-compensated mean image (edgelib.trace_edge), cleaned, classified and written to
work/cache/edges_cart310.json, with per-edge verification crops in work/cache/edgecrops_cart310/
and the overview results/edges_cart310.png.

The guides (approximate anchor points) were obtained from LSD chains / row-tracking in a rotated
view (10_edges_cart310_chains.py, _cluster.py, _rottrack.py) and by visual inspection of zoomed
crops; every edge was checked on its crop.  Guides are only guides: the sub-pixel positions come
from the gradient extremum along the normal (parabolic refinement, iterated, smooth cubic path).

Polarity convention (edgelib): +1 = image gets brighter along the left-hand normal of the guide
direction (image coordinates, y down).
"""
import cv2
import numpy as np

from common import CACHE, RESULTS, save_json
import edges310lib as E
import edgelib

REGION = "cart310"
OUT = f"{CACHE}/edgecrops_{REGION}"

from edges310_defs import EDGES  # noqa: E402  (curated guide table)


def guide_curve(anchors, deg):
    A = np.asarray(anchors, float)
    if len(A) == 2:
        deg = 1
    cvv = E.curve_from_points(A, deg=min(deg, len(A) - 1))
    return cvv


def process(ed):
    A = np.asarray(ed["guide"], float)
    deg = ed.get("guide_deg", 2)
    cvv = guide_curve(A, deg)
    t0, t1 = cvv[4] - ed.get("extend", 0.0), cvv[5] + ed.get("extend", 0.0)
    G = E.eval_curve(cvv, np.arange(t0, t1 + 1e-9, 2.0))
    # orientation: guide runs from the first to the last anchor
    if (G[-1] - G[0]) @ (A[-1] - A[0]) < 0:
        G = G[::-1]
    if ed.get("kind", "edge") == "line_centre":
        # thin dark/bright line: locate the centre of the line (own tracer, same scheme as trace_edge)
        out = E.trace_line_centre(G, dark=ed.get("dark", True), halfwidth=ed.get("hw", 2.5),
                                  sigma=ed.get("sigma", 0.8), fit_deg=ed.get("fit_deg", 3),
                                  min_contrast=ed.get("min_strength", 3.0))
        P0, S0 = out["points"], out["strength"]
        if len(P0) < 10:
            return None
        keep = S0 > ed.get("min_rel", 0.4) * np.median(S0)
        for _ in range(4):
            c, d, _, _ = edgelib.line_fit(P0[keep])
            nrm = np.array([-d[1], d[0]])
            t = (P0 - c) @ d
            r = (P0 - c) @ nrm
            q = np.polyfit(t[keep], r[keep], 3)
            res = r - np.polyval(q, t)
            new = keep & (np.abs(res) < ed.get("max_res", 0.5))
            new = (S0 > ed.get("min_rel", 0.4) * np.median(S0)) & (np.abs(res) < ed.get("max_res", 0.5))
            if (new == keep).all():
                break
            keep = new
        P0, S0 = P0[keep], S0[keep]
        c, d, _, _ = edgelib.line_fit(P0)
        if d @ (G[-1] - G[0]) < 0:
            d = -d
        o = np.argsort((P0 - c) @ d)
        tr = dict(points=P0[o], strength=S0[o])
    else:
        tr = E.trace_clean(G, polarity=ed["pol"], halfwidth=ed.get("hw", 2.0), fit_deg=ed.get("fit_deg", 3),
                           min_rel=ed.get("min_rel", 0.4), max_res=ed.get("max_res", 0.5),
                           min_strength=ed.get("min_strength", 4.0), sigma=ed.get("sigma", 1.0))
    if tr is None:
        return None
    P, S = tr["points"], tr["strength"]
    # optional exclusion windows (occluders identified on the crops): list of [x0,y0,x1,y1]
    keep = np.ones(len(P), bool)
    for (x0, y0, x1, y1) in ed.get("exclude", []):
        keep &= ~((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
    P, S = P[keep], S[keep]
    # drop isolated fragments (< min_frag points between gaps > 8 px)
    parts = E.split_gaps(dict(points=P, strength=S), max_gap=8.0, min_pts=ed.get("min_frag", 6))
    if not parts:
        return None
    # optionally trim fragment ends next to interruptions (label holders clipped on the lip locally
    # disturb the edge profile)
    tg = ed.get("trim_gap", 0.0)
    if tg > 0 and len(parts) > 1:
        tp = []
        for k, p in enumerate(parts):
            Q, Sq = p["points"], p["strength"]
            s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
            m = np.ones(len(Q), bool)
            if k > 0:
                m &= s_ >= tg
            if k < len(parts) - 1:
                m &= s_ <= s_[-1] - tg
            if m.sum() >= ed.get("min_frag", 6):
                tp.append(dict(points=Q[m], strength=Sq[m]))
        parts = tp
        if not parts:
            return None
    P = np.vstack([p["points"] for p in parts])
    S = np.concatenate([p["strength"] for p in parts])
    if len(P) < 10:
        return None
    return dict(points=P, strength=S, guide=G, cv=cvv)


def verify_images(ed, res):
    P = res["points"]
    eid = ed["id"]
    # 1) straightened strip along the traced curve (x1.5 along, x5 across)
    cvv = E.curve_from_points(P, deg=3)
    strip = E.strip_image(cvv, cvv[4] - 15, cvv[5] + 15, hw=9, sperp=5, salong=1.5, pts=P, rowlen=600)
    # 2) local zoom windows (x6) at 5 places along the edge
    E.local_crops(P, f"{OUT}/_tmp_local.png", label=eid, n=5, half=16, scale=6)
    loc = cv2.imread(f"{OUT}/_tmp_local.png")
    # 3) context crop (whole edge, points drawn)
    E.render_crop(P, f"{OUT}/_tmp_ctx.png", label=eid, scale=3, pad=25, maxside=600)
    ctx = cv2.imread(f"{OUT}/_tmp_ctx.png")
    W = max(strip.shape[1], loc.shape[1], ctx.shape[1])

    def padw(im):
        return np.hstack([im, np.zeros((im.shape[0], W - im.shape[1], 3), np.uint8)]) if im.shape[1] < W else im

    sep = np.full((6, W, 3), 255, np.uint8)
    mont = np.vstack([padw(ctx), sep, padw(loc), sep, padw(strip)])
    cv2.imwrite(f"{OUT}/{eid}.png", mont)
    return f"work/cache/edgecrops_{REGION}/{eid}.png"


def main(ids=None):
    edges = []
    for ed in EDGES:
        if ids and ed["id"] not in ids:
            continue
        res = process(ed)
        if res is None:
            print("FAILED", ed["id"])
            continue
        P, S = res["points"], res["strength"]
        rms, bow, L = E.chord_stats(P)
        ang = E.direction_angles(P)
        crop = verify_images(ed, res)
        rec = {
            "id": ed["id"],
            "what": ed["what"],
            "class": ed.get("class_", ed.get("class", "cart_structure")),
            "cart": (None if ed.get("class_") == "floor_marking" else ed.get("cart", 310)),
            "straight_3d": ed.get("straight_3d", True),
            "direction": ed["direction"],
            "model_line": ed.get("model_line"),
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in P],
            "strength": [round(float(s), 2) for s in S],
            "chord_dev_rms_px": round(rms, 3),
            "chord_bow_px": round(bow, 3),
            "verified": ed.get("verified", "yes"),
            "verification_note": ed.get("note", ""),
            "crop": crop,
            "length_px": round(L, 1),
            "n_points": int(len(P)),
            "polarity": ed["pol"],
            "angle_to_initial_vp_deg": {k: round(v, 2) for k, v in ang.items()},
        }
        edges.append(rec)
        print(f"{ed['id']:28s} {ed['direction']:15s} L={L:6.1f} n={len(P):4d} rms={rms:.2f} bow={bow:+.2f} "
              f"s_med={np.median(S):5.1f} angXYZ=({ang['X']:.1f},{ang['Y']:.1f},{ang['Z']:.1f})")
    return edges


def write_all(edges):
    save_json({"region": REGION,
               "image": "work/cache/mean_aligned_gray.npy (jitter-compensated mean of the 7 stills)",
               "edges": edges}, f"{CACHE}/edges_{REGION}.json")
    col = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    colors = {"cartX": (0, 0, 255), "cartY": (0, 170, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
              "floor_plane": (0, 200, 255), "horizontal_other": (255, 255, 0), "unknown": (128, 128, 128)}
    for e in edges:
        P = np.array(e["points"])
        c = colors[e["direction"]]
        # draw as polyline through the points, but not across gaps
        gaps = np.nonzero(np.hypot(*np.diff(P, axis=0).T) > 8)[0]
        s0 = 0
        for g in list(gaps) + [len(P) - 1]:
            seg = P[s0:g + 1]
            if len(seg) > 1:
                cv2.polylines(col, [np.round(seg * 8).astype(np.int32)], False, c, 1, cv2.LINE_AA, shift=3)
            s0 = g + 1
        m = P[len(P) // 2]
        lab = e["id"].replace(f"{REGION}_", "")
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(col, lab, (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1, cv2.LINE_AA)
    y = 20
    for k, c in colors.items():
        cv2.putText(col, k, (1700, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        y += 18
    cv2.imwrite(f"{RESULTS}/edges_{REGION}.png", col)


if __name__ == "__main__":
    import sys
    ids = sys.argv[1:] or None
    ed = main(ids)
    if ids is None:
        write_all(ed)
    import os
    for f in ("_tmp_local.png", "_tmp_ctx.png"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
