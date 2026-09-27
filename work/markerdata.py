"""Loading marker corner observations (final detections if available, else the initial ones)."""
from __future__ import annotations

import os

import numpy as np

from common import CACHE, marker_center, marker_corners_3d, marker_role, load_json


def surface_of(mid, cart):
    z = marker_center(mid, cart)[2]
    return {0.0: "top", -195.0: "A", -595.0: "B", -995.0: "C"}.get(float(z), str(z))


def load_markers(prefer_final=True, which="final"):
    """Return list of marker dicts: cart, id, role, row, rot_k, corners_px (4x2), corners_3d (4x3),
    valid (4 bools), std (4)."""
    fn = f"{CACHE}/markers_final.json"
    out = []
    if prefer_final and os.path.exists(fn):
        for m in load_json(fn):
            c = np.array(m["corners_px"], float)
            if which == "aruco" and m.get("aruco_corners_px") is not None:
                c = np.array(m["aruco_corners_px"], float)
            if which == "edge" and m.get("edge_corners_px") is not None:
                c = np.array([p if p is not None else [np.nan, np.nan] for p in m["edge_corners_px"]], float)
            if which == "biascorr" and m.get("corners_px_bias_corrected") is not None:
                c = np.array([p if p is not None else [np.nan, np.nan] for p in m["corners_px_bias_corrected"]], float)
            valid = np.array(m.get("corner_valid", [True] * 4), bool)
            if np.any(np.isnan(c)):
                valid &= ~np.isnan(c).any(1)
            out.append(dict(cart=int(m["cart"]), id=int(m["id"]), role=marker_role(int(m["id"]), int(m["cart"])),
                            row=surface_of(int(m["id"]), int(m["cart"])), rot_k=int(m["rot_k"]), corners_px=c,
                            corners_3d=marker_corners_3d(int(m["id"]), int(m["cart"]), int(m["rot_k"])),
                            valid=valid, std=np.array([v if v is not None else 0.3 for v in m.get("corners_std_px", [0.2] * 4)], float),
                            side_valid=list(m.get("side_valid", [True] * 4)), src="final"))
        return out
    d = load_json(f"{CACHE}/initial_calib.json")
    rot = {tuple(map(int, k.split("_"))): v for k, v in d["rot"].items()}
    mean = np.array(d["corners_mean"])
    sd = np.array(d["corners_frame_std"])
    for i, k in enumerate([tuple(k) for k in d["keys"]]):
        out.append(dict(cart=k[0], id=k[1], role=marker_role(k[1], k[0]), row=surface_of(k[1], k[0]), rot_k=rot[k],
                        corners_px=mean[i], corners_3d=marker_corners_3d(k[1], k[0], rot[k]), valid=np.ones(4, bool),
                        std=np.hypot(sd[i, :, 0], sd[i, :, 1]), src="initial"))
    return out


def to_points(markers, sel=None, use_centres=False):
    pts = []
    for mi, m in enumerate(markers):
        if sel is not None and not sel(m):
            continue
        if use_centres:
            if m["valid"].all():
                pts.append(dict(cart=m["cart"], id=m["id"], mi=mi, corner=-1, X=marker_center(m["id"], m["cart"]),
                                uv=m["corners_px"].mean(0)))
            continue
        for j in range(4):
            if m["valid"][j]:
                pts.append(dict(cart=m["cart"], id=m["id"], mi=mi, corner=j, X=m["corners_3d"][j], uv=m["corners_px"][j]))
    return pts


def marker_side_edges(markers, only_partial=True, frac=0.18, step=1.5):
    """Sub-pixel traced sides of the black code square as exact 3D model lines (cart frame).

    Side j joins corners j and j+1 (OpenCV order). The side is used when both of its end corners are
    NOT already used as corner observations (only_partial=True) but the side itself is visible; the side
    is traced between the (possibly extrapolated) corner positions, excluding `frac` at each end.
    Returns edges in the linedata format with model_line = axis-aligned 3D line on the sticker plane.
    """
    from edgelib import trace_edge, line_fit

    out = []
    for mi, m in enumerate(markers):
        c = m["corners_px"]
        if np.any(np.isnan(c)):
            continue
        X = m["corners_3d"]
        valid = m["valid"]
        side_ok = m.get("side_valid", [True] * 4)
        for j in range(4):
            a, b = j, (j + 1) % 4
            if only_partial and valid[a] and valid[b]:
                continue
            if not side_ok[j]:
                continue
            P0, P1 = c[a], c[b]
            L = np.hypot(*(P1 - P0))
            t = np.linspace(frac, 1 - frac, max(int(L * (1 - 2 * frac) / step), 5))
            poly = P0[None] + (P1 - P0)[None] * t[:, None]
            tr = trace_edge(poly, halfwidth=3.0, step=step, iters=2, min_strength=3.0)
            if len(tr["points"]) < 6:
                continue
            _, _, rms, mx = line_fit(tr["points"])
            if rms > 0.35:
                continue
            A, B = X[a], X[b]
            ax = int(np.argmax(np.abs(B - A)))
            fixed = {"XYZ"[i]: float(A[i]) for i in range(3) if i != ax}
            out.append(dict(id=f"marker_{m['cart']}_{m['id']}_side{j}", region=f"cart{m['cart']}", cart=m["cart"],
                            direction="cart" + "XYZ"[ax], what=f"side {j} of sticker {m['id']} (cart {m['cart']})",
                            model_line={"fixed": fixed, "free": "XYZ"[ax], "range_mm": [float(min(A[ax], B[ax])), float(max(A[ax], B[ax]))], "exact": True},
                            points=tr["points"], length=float(L * (1 - 2 * frac)), cls="marker_side", sticker_mi=mi,
                            vp_group=None))
    return out
