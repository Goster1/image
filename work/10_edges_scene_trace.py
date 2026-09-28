"""Scene region: sub-pixel tracing of the curated edges (edgesscene_defs.py), cleaning, statistics,
vanishing-direction check against the initial calibration, zoomed verification crops.

Per edge: guide polyline -> edgelib.trace_edge (gradient extremum along the normal + parabola, iterated,
cubic path in arc length; polarity from the known bright side) or, for thin dark lines (floor joints),
the extremum of the smoothed profile (edgesscene_lib.trace_line_centre) -> cleaning (weak points
< min_rel x median strength, points off a local smooth curve by > max(0.3 px, 4 robust sigma),
excluded boxes, 4 px trimmed at every interruption, fragments < 12 px dropped).

Outputs: work/cache/edges_scene_raw.json (all traces + diagnostics), work/cache/edgecrops_scene/<id>.png
(zoomed crop with the points + 4 high-zoom windows), work/cache/edgecrops_scene/montage_*.png."""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

import edgelib
import edgesscene_lib as L
from common import save_json, undistort_points
from edgesscene_defs import E


def vp_angles(P):
    """angle (deg) between the (undistorted, initial calibration) edge line and the image direction of
    each cart axis at the edge centre (classification lens: edgesscene_lib.class_calib)."""
    K, dist, poses, _ = L.class_calib()
    n = undistort_points(P, K, dist)
    c, d, rms, mx = edgelib.line_fit(n)
    out = {}
    for cart, (rv, tv, _) in poses.items():
        R = cv2.Rodrigues(rv.reshape(3, 1))[0]
        for a, nm in enumerate("XYZ"):
            D = R[:, a]
            u = D[:2] - c * D[2]
            u = u / np.linalg.norm(u)
            ang = np.degrees(np.arccos(min(1.0, abs(float(u @ d)))))
            out[f"{cart}{nm}"] = round(ang, 2)
    # straightness after undistortion (initial calibration; informative)
    npx = n * K[0, 0]
    _, _, rms_u, mx_u = edgelib.line_fit(npx)
    return out, rms_u


_COL = None


def color_filter(tr, side, cond, off=3.0):
    """keep only points whose colour `off` px towards `side` satisfies `cond` (e.g. yellow paint)."""
    global _COL
    if _COL is None:
        _COL = L.color_image().astype(np.float64)
    v = np.array({"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[side], float)
    Q = tr["points"] + off * v
    b, g, r = [map_coordinates(_COL[:, :, k], [Q[:, 1], Q[:, 0]], order=1) for k in range(3)]
    if cond == "yellow":
        ok = ((r + g) / 2 - b > 50) & (r > 110)
    else:
        raise ValueError(cond)
    return {k: (val[ok] if isinstance(val, np.ndarray) and len(val) == len(ok) else val) for k, val in tr.items()}


def run(defs, render=True):
    rows = []
    for e in defs:
        pieces = e["guide"] if isinstance(e["guide"][0][0], (list, tuple)) else [e["guide"]]
        trs = []
        for g in pieces:
            G = np.asarray(g, float)
            G = L.smooth_curve(G, deg=min(3, len(G) - 1), step=3.0)
            fd = min(e["fit_deg"], 2 if len(pieces) > 1 else e["fit_deg"])
            if e["kind"] == "edge":
                pol = L.polarity_for(g, e["bright"])
                t1 = edgelib.trace_edge(G, halfwidth=e["hw"], step=2.0, polarity=pol, sigma=e["sigma"], iters=4,
                                        min_strength=e["min_strength"], fit_deg=fd)
            else:
                t1 = L.trace_line_centre(G, dark=(e["kind"] == "dark_line"), halfwidth=e["hw"], sigma=e["sigma"],
                                         fit_deg=fd, min_strength=e["min_strength"])
            if len(trs):
                t1["s"] = t1["s"] + trs[-1]["s"].max() + 50.0 if len(trs[-1]["s"]) else t1["s"]
            trs.append(t1)
        tr = {k: np.concatenate([t[k] for t in trs]) for k in ("points", "strength", "normal", "s")}
        G = np.vstack([L.smooth_curve(np.asarray(g, float), deg=min(3, len(g) - 1), step=3.0) for g in pieces])
        if e.get("keep_color"):
            tr = color_filter(tr, *e["keep_color"])
        tr = L.clean(tr, min_rel=e["min_rel"], cut_boxes=e["cut"])
        P, S = tr["points"], tr["strength"]
        r = dict(e)
        r["points"] = P
        r["strength"] = S
        r["s"] = tr["s"]
        if len(P) < 10:
            r["n"] = len(P)
            print(f"{e['id']:28s} FAILED ({len(P)} pts)")
            rows.append(r)
            continue
        st = L.chord_stats(P)
        noise = L.resid_noise(P)
        ang, rms_u = vp_angles(P)
        r.update(n=len(P), length=st["length"], chord_rms=st["rms"], chord_bow=st["bow"], noise=noise,
                 med_strength=float(np.median(S)), vp_angles=ang, undist_rms=rms_u)
        best = sorted(ang.items(), key=lambda kv: kv[1])[:3]
        print(f"{e['id']:28s} n={len(P):4d} L={st['length']:6.1f} str={np.median(S):6.1f} noise={noise:.3f} "
              f"chord_rms={st['rms']:.2f} bow={st['bow']:+.2f} undist_rms={rms_u:.2f}  vp: {best}")
        if render:
            ctx = L.context_view(P, maxside=460, guide=G)
            strip = L.strip_view(P, hw=10, sperp=6, salong=1.5, rowlen=900, label=e["id"])
            win = L.local_windows(P, None, label="", n=4, half=16, scale=7)
            top = L.compose([win]) if ctx.shape[1] + win.shape[1] > 1400 else None
            if top is None:
                h = max(ctx.shape[0], win.shape[0])
                padh = lambda im: np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])
                head = np.hstack([padh(ctx), np.zeros((h, 6, 3), np.uint8), padh(win)])
            else:
                head = L.compose([ctx, win])
            cv2.imwrite(f"{L.CROPDIR}/{e['id']}.png", L.compose([head, strip]))
        rows.append(r)
    return rows


if __name__ == "__main__":
    import sys

    sel = sys.argv[1:]
    defs = [e for e in E if not sel or any(s in e["id"] for s in sel)]
    rows = run(defs)
    if not sel:
        save_json(rows, f"{L.CACHE}/edges_scene_raw.json")
