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
                c = np.array(m["edge_corners_px"], float)
            valid = np.array(m.get("corner_valid", [True] * 4), bool)
            if np.any(np.isnan(c)):
                valid &= ~np.isnan(c).any(1)
            out.append(dict(cart=int(m["cart"]), id=int(m["id"]), role=marker_role(int(m["id"]), int(m["cart"])),
                            row=surface_of(int(m["id"]), int(m["cart"])), rot_k=int(m["rot_k"]), corners_px=c,
                            corners_3d=marker_corners_3d(int(m["id"]), int(m["cart"]), int(m["rot_k"])),
                            valid=valid, std=np.array(m.get("corners_std_px", [0.2] * 4), float), src="final"))
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
