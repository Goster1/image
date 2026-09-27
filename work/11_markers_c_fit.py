"""Stage 11c: detection quality statistics, in-plane rotation (rot_k) of every sticker, joint
calibration with all valid corners, and the output files.

Inputs : work/cache/markers_guided.json (id confirmation), work/cache/markers_corners.json (corners)
Outputs: work/cache/markers_final.json, work/cache/markers_summary.json, results/detections.json,
         results/markers_overlay.png, results/markers_crops.png, results/markers_rect_views.png

Joint fit: calib.Problem (shared intrinsics + one 6-DoF pose per cart), drawing geometry exactly
as specified (sticker centre from the layout, 90 mm black square, face up, rot_k*90 deg in-plane).
Model for the checks here: f single (fx = fy), principal point free, k1, k2 (p1 = p2 = k3 = 0).
"""
import importlib
import os

import cv2
import numpy as np

from calib import Model, Problem, covariance
from common import (CACHE, CODE_MM, H, RESULTS, W, load_json, marker_center, marker_corners_3d, save_json,
                    still_paths)

L = importlib.import_module("11_markers_lib")
MODEL = dict(f="single", pp="free", dist=["k1", "k2"])


# ------------------------------------------------------------------------------------------
# detection statistics
# ------------------------------------------------------------------------------------------

def radial_unit(quad, i):
    c = np.asarray(quad, float).mean(0)
    v = np.asarray(quad, float)[i] - c
    return v / np.linalg.norm(v)


def detection_stats(M):
    rows = []
    for m in M:
        A = np.array(m["aruco_frames"], float)  # 7x4x2
        B = np.array(m["edge_frames"], float)
        am, bm = A.mean(0), np.nanmean(B, 0) if np.isfinite(B).any() else np.full((4, 2), np.nan)
        quad = np.array(m["corners_px"], float)
        for i in range(4):
            both = m["aruco_valid"][i] and m["edge_valid"][i]
            if not both:
                continue
            d = bm[i] - am[i]
            u = radial_unit(quad, i)
            # per frame differences
            dfr = B[:, i] - A[:, i]
            rows.append(dict(cart=m["cart"], id=m["id"], corner=i, dx=d[0], dy=d[1], dist=float(np.linalg.norm(d)),
                             radial=float(d @ u), a_std=float(np.sqrt(np.mean(np.sum((A[:, i] - am[i]) ** 2, 1)))),
                             b_std=float(np.sqrt(np.mean(np.sum((B[:, i] - bm[i]) ** 2, 1)))),
                             diff_frame_std=float(np.sqrt(np.mean(np.sum((dfr - dfr.mean(0)) ** 2, 1)))),
                             mean_vs_framemean_a=float(np.linalg.norm(np.array(m["aruco_mean_image"])[i] - am[i])),
                             mean_vs_framemean_b=float(np.linalg.norm(np.array(m["edge_mean_image"], float)[i] - bm[i])),
                             curved_minus_straight=float(np.linalg.norm(np.nanmean(np.array(m["edge_curved_frames"], float), 0)[i] - bm[i]))))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    D = np.column_stack([R["dx"], R["dy"]])
    s = dict(n_corners_both_valid=len(rows),
             a_minus_b_note="b (edge) minus a (ArUco-rule cornerSubPix), frame-mean corners, px",
             mean_offset_px=[float(-D[:, 0].mean()), float(-D[:, 1].mean())],
             edge_minus_aruco_mean_px=[float(D[:, 0].mean()), float(D[:, 1].mean())],
             rms_offset_px=float(np.sqrt(np.mean(R["dist"] ** 2))), median_offset_px=float(np.median(R["dist"])),
             max_offset_px=float(R["dist"].max()),
             radial_edge_minus_aruco_mean_px=float(R["radial"].mean()),
             radial_edge_minus_aruco_sem_px=float(R["radial"].std(ddof=1) / np.sqrt(len(rows))),
             radial_edge_minus_aruco_std_px=float(R["radial"].std(ddof=1)),
             frame_std_aruco_median_px=float(np.median(R["a_std"])), frame_std_edge_median_px=float(np.median(R["b_std"])),
             frame_std_aruco_rms_px=float(np.sqrt(np.mean(R["a_std"] ** 2))), frame_std_edge_rms_px=float(np.sqrt(np.mean(R["b_std"] ** 2))),
             mean_image_vs_frame_mean_aruco_rms_px=float(np.sqrt(np.mean(R["mean_vs_framemean_a"] ** 2))),
             mean_image_vs_frame_mean_edge_rms_px=float(np.sqrt(np.mean(R["mean_vs_framemean_b"] ** 2))),
             curved_vs_straight_side_lines_rms_px=float(np.sqrt(np.mean(R["curved_minus_straight"] ** 2))),
             curved_vs_straight_side_lines_max_px=float(R["curved_minus_straight"].max()),
             per_corner=rows)
    return s


# ------------------------------------------------------------------------------------------
# joint fit
# ------------------------------------------------------------------------------------------

def seed_x(model):
    sd = L.seed()
    return np.r_[model.pack(sd["K"], sd["dist"]), sd["poses"][80], sd["poses"][310]]


def build_points(M, rot, which="final", only=None):
    pts = []
    for m in M:
        key = (m["cart"], m["id"])
        if only is not None and key not in only:
            continue
        X = marker_corners_3d(m["id"], m["cart"], rot[key])
        if which == "final":
            uv, ok = np.array(m["corners_px"], float), np.array(m["corner_valid"])
        elif which == "aruco":
            uv, ok = np.array(m["aruco_frames"], float).mean(0), np.array(m["aruco_valid"])
        elif which == "edge":
            uv, ok = np.nanmean(np.array(m["edge_frames"], float), 0), np.array(m["edge_valid"])
        elif which == "aruco_mean_image":
            uv, ok = np.array(m["aruco_mean_image"], float), np.array(m["aruco_valid"])
        elif which == "edge_mean_image":
            uv, ok = np.array(m["edge_mean_image"], float), np.array(m["edge_valid"])
        for j in range(4):
            if ok[j] and np.isfinite(uv[j]).all():
                pts.append(dict(cart=m["cart"], id=m["id"], corner=j, X=X[j], uv=uv[j]))
    return pts


def fit(pts, x0=None, loss="linear"):
    model = Model(MODEL)
    pr = Problem(model, pts, carts=[80, 310])
    x0 = seed_x(model) if x0 is None else x0
    r = pr.solve(x0, loss=loss)
    res = pr.predict(r.x) - pr.uv
    return pr, r, res


def per_group_rms(pts, res, key):
    out = {}
    e = np.linalg.norm(res, axis=1)
    for g in sorted({key(p) for p in pts}, key=str):
        m = np.array([key(p) == g for p in pts])
        out[str(g)] = dict(n=int(m.sum()), rms=float(np.sqrt(np.mean(e[m] ** 2))), max=float(e[m].max()))
    return out


def determine_rotations(M, which="final"):
    rot = {(m["cart"], m["id"]): (0 if m["role"].startswith("TOP") else 2) for m in M}
    info = {}
    for it in range(3):
        changed = False
        pts = build_points(M, rot, which)
        pr, r, res = fit(pts)
        x_ref = r.x
        for m in M:
            key = (m["cart"], m["id"])
            costs = []
            mrms = []
            for rk in range(4):
                rr = dict(rot)
                rr[key] = rk
                p2 = build_points(M, rr, which)
                pr2, r2, res2 = fit(p2, x0=x_ref)
                sel = np.array([(p["cart"], p["id"]) == key for p in p2])
                costs.append(float(np.sum(res2 ** 2)))
                mrms.append(float(np.sqrt(np.mean(np.sum(res2[sel] ** 2, 1)))) if sel.any() else np.nan)
            best = int(np.argmin(costs))
            srt = np.sort(costs)
            info[key] = dict(rot_k=best, total_rms_per_rot=[float(np.sqrt(c / (2 * len(pts)))) for c in costs],
                             marker_rms_per_rot=mrms, marker_rms_best=mrms[best],
                             marker_rms_second=float(np.sort(mrms)[1]),
                             delta_chi2_second_minus_best=float(srt[1] - srt[0]))
            if best != rot[key]:
                rot[key] = best
                changed = True
        if not changed:
            break
    return rot, info


def calib_summary(pts, pr, r, res, label):
    C, s2, dof = covariance(r)
    names = pr.model.names
    sig = np.sqrt(np.diag(C))[: pr.model.n]
    e = np.linalg.norm(res, axis=1)
    return dict(label=label, model="f single, pp free, k1, k2", n_points=len(pts),
                params={nm: float(v) for nm, v in zip(names, r.x[: pr.model.n])},
                sigma_scaled={nm: float(v) for nm, v in zip(names, sig)},
                rms_px=float(np.sqrt(np.mean(e ** 2))), rms_per_coord_px=float(np.sqrt(np.mean(res ** 2))), max_px=float(e.max()),
                per_cart=per_group_rms(pts, res, lambda p: p["cart"]),
                per_marker=per_group_rms(pts, res, lambda p: f"{p['cart']}:{p['id']}"),
                per_level=per_group_rms(pts, res, lambda p: "top" if marker_center(p["id"], p["cart"])[2] == 0 else f"z={marker_center(p['id'], p['cart'])[2]:.0f}"),
                residuals=[dict(cart=p["cart"], id=p["id"], corner=p["corner"], du=float(d[0]), dv=float(d[1])) for p, d in zip(pts, res)])


# ------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------

def draw_overlay(M, path):
    img = L.mean_color()
    for m in M:
        q = np.array(m["corners_px"], float)
        ok = m["corner_valid"]
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            col = (0, 220, 0) if m["side_valid"][i] else (0, 0, 255)
            cv2.line(img, tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int)), col, 1, cv2.LINE_AA, shift=2)
        for i in range(4):
            cv2.circle(img, tuple(np.round(q[i] * 4).astype(int)), 12, (0, 255, 0) if ok[i] else (0, 0, 255), 2, cv2.LINE_AA, shift=2)
        cv2.circle(img, tuple(np.round(q[0] * 4).astype(int)), 24, (255, 255, 0), 1, cv2.LINE_AA, shift=2)
        c = q.mean(0)
        txt = f"{m['cart']}:{m['id']}"
        org = (int(c[0]) - 22, int(q[:, 1].max()) + 16)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if m["found_by"].startswith("guided") else (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), -1)
    cv2.putText(img, "green = valid corner / measured side, red = invalid corner / hidden side, cyan ring = ArUco corner 0,"
                " yellow label = found by guided detection, white = plain detector", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def draw_crops(M, path, tile=300):
    img = L.mean_color()
    tiles = []
    for m in M:
        q = np.array(m["corners_px"], float)
        c = q.mean(0)
        side = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).max()
        s = tile / (1.6 * side)
        half = tile / (2 * s)
        x0, y0 = int(np.floor(c[0] - half)), int(np.floor(c[1] - half))
        x0c, y0c = max(x0, 0), max(y0, 0)
        crop = np.full((int(2 * half) + 1, int(2 * half) + 1, 3), 80, np.uint8)
        sub = img[y0c:y0 + crop.shape[0], x0c:x0 + crop.shape[1]]
        crop[y0c - y0:y0c - y0 + sub.shape[0], x0c - x0:x0c - x0 + sub.shape[1]] = sub
        cr = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if cr.shape[0] < tile or cr.shape[1] < tile:
            cr = cv2.copyMakeBorder(cr, 0, max(0, tile - cr.shape[0]), 0, max(0, tile - cr.shape[1]), cv2.BORDER_CONSTANT, value=(80, 80, 80))
        T = lambda p: (int(round((p[0] - x0 + 0.5) * s - 0.5)), int(round((p[1] - y0 + 0.5) * s - 0.5)))
        # side lines (mean image fit) and traced corners
        for ln in m["side_lines"]:
            if ln["point"] is None or not ln["valid"]:
                continue
            p, d = np.array(ln["point"]), np.array(ln["direction"])
            cv2.line(cr, T(p - d * side * 0.8), T(p + d * side * 0.8), (0, 200, 0), 1, cv2.LINE_AA)
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            if not m["side_valid"][i]:
                cv2.line(cr, T(a), T(b), (0, 0, 255), 1, cv2.LINE_AA)
        A = np.array(m["aruco_frames"], float).mean(0)
        for i in range(4):
            if m["aruco_valid"][i]:
                cv2.drawMarker(cr, T(A[i]), (255, 0, 255), cv2.MARKER_CROSS, 9, 1)
            cv2.circle(cr, T(q[i]), 5, (0, 255, 0) if m["corner_valid"][i] else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(cr, str(i), (T(q[i])[0] + 6, T(q[i])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cr = cr[:tile, :tile]
        lab = f"{m['cart']}:{m['id']} {m['role']} rot_k={m['rot_k']} {'guided' if m['found_by'].startswith('guided') else 'plain'}"
        cv2.rectangle(cr, (0, 0), (tile, 18), (0, 0, 0), -1)
        cv2.putText(cr, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cr)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    out = np.vstack(rows)
    cv2.putText(out, "green circle = corner used (valid), red = invalid; green lines = fitted valid sides, red = hidden side; magenta + = ArUco-rule subpix corner; numbers = ArUco corner index",
                (5, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def draw_rect_views(M, path):
    tiles = []
    for m in M:
        p = f"{CACHE}/markers11/rect_{m['cart']}_{m['id']}.png"
        im = cv2.imread(p)
        im = cv2.resize(im, (240, 240))
        cv2.putText(im, f"{m['cart']}:{m['id']}", (4, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        tiles.append(im)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    out = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
    cv2.putText(out, "rectified views: expected code bits (green = read equal, orange = read on partly visible cell, x = hidden)",
                (5, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


# ------------------------------------------------------------------------------------------

def main():
    G = {(g["cart"], g["id"]): g for g in load_json(f"{CACHE}/markers_guided.json")}
    M = load_json(f"{CACHE}/markers_corners.json")
    order = {(c, i): k for k, (c, i) in enumerate([(80, j) for j in list(range(9)) + [92]] + [(310, j) for j in list(range(9)) + [322]])}
    M.sort(key=lambda m: order[(m["cart"], m["id"])])

    stats = detection_stats(M)
    print(f"(a) vs (b): n={stats['n_corners_both_valid']} rms {stats['rms_offset_px']:.3f} px, median {stats['median_offset_px']:.3f},"
          f" max {stats['max_offset_px']:.3f}; mean (b-a) {np.round(stats['edge_minus_aruco_mean_px'], 3)};"
          f" radial (b-a) {stats['radial_edge_minus_aruco_mean_px']:.3f} +- {stats['radial_edge_minus_aruco_sem_px']:.3f} px")
    print(f"frame std: aruco median {stats['frame_std_aruco_median_px']:.3f} rms {stats['frame_std_aruco_rms_px']:.3f};"
          f" edge median {stats['frame_std_edge_median_px']:.3f} rms {stats['frame_std_edge_rms_px']:.3f}")
    print(f"mean image vs frame mean: aruco {stats['mean_image_vs_frame_mean_aruco_rms_px']:.3f}, edge {stats['mean_image_vs_frame_mean_edge_rms_px']:.3f};"
          f" curved vs straight side lines rms {stats['curved_vs_straight_side_lines_rms_px']:.3f} max {stats['curved_vs_straight_side_lines_max_px']:.3f}")

    # ---- rot_k
    rot, rinfo = determine_rotations(M, "final")
    for k, v in rinfo.items():
        print(f"rot {k}: rot_k={v['rot_k']} marker rms best {v['marker_rms_best']:.2f} second {v['marker_rms_second']:.2f}"
              f" dchi2 {v['delta_chi2_second_minus_best']:.0f}")

    # ---- joint fits
    fits = {}
    plain = {k for k, g in G.items() if g["method"].startswith("plain")}
    for which, only, label in [("final", None, "all valid corners, final (edge where valid, else ArUco-rule)"),
                               ("edge", None, "edge-based corners only"),
                               ("aruco", None, "ArUco-rule subpix corners only (corner neighbourhood visible)"),
                               ("aruco", plain, "ArUco-rule corners, only the 12 plainly detected stickers"),
                               ("edge", plain, "edge corners, only the 12 plainly detected stickers"),
                               ("edge_mean_image", None, "edge corners measured on the mean image"),
                               ("aruco_mean_image", None, "ArUco-rule corners measured on the mean image")]:
        pts = build_points(M, rot, which, only)
        pr, r, res = fit(pts)
        s = calib_summary(pts, pr, r, res, label)
        fits[f"{which}{'' if only is None else '_plain12'}"] = s
        print(f"{label:70s} n={s['n_points']:3d} rms={s['rms_px']:.3f} max={s['max_px']:.2f} "
              + " ".join(f"{k}={v:.4g}" for k, v in s["params"].items())
              + " | carts " + " ".join(f"{k}:{v['rms']:.2f}" for k, v in s["per_cart"].items()))
    fin = fits["final"]
    worst = sorted(fin["per_marker"].items(), key=lambda kv: -kv[1]["rms"])
    print("per marker rms (final):", ", ".join(f"{k} {v['rms']:.2f}" for k, v in worst))
    print("per level rms (final):", {k: round(v["rms"], 2) for k, v in fin["per_level"].items()})

    # ---- final records
    frames = [os.path.basename(p) for p in still_paths()]
    final = []
    for m in M:
        key = (m["cart"], m["id"])
        g = G[key]
        dec = g["id_confirmation"]["decoding"]
        ext = dec["extended"]
        tt = g["id_confirmation"]["template_test"]
        others_cart = [i for i in list(range(9)) + [92, 322] if i != m["id"]]
        conf = dict(how="own 6x6 grid decoding in the rectified view (visible cells only) + masked dictionary template correlation",
                    n_visible_bits=dec["n_valid_bits"], hamming_expected=dec["hamming_expected"],
                    best_other_code=dict(id=dec["best_other_id"], hamming=dec["best_other_hamming"]),
                    n_dictionary_codes_at_distance0=dec["n_ids_at_distance0"],
                    with_partially_visible_cells=dict(n_bits=ext["n_bits"], hamming_expected=ext["hamming_expected"],
                                                      n_dictionary_codes_at_distance0=ext["n_ids_at_distance0"],
                                                      best_other_code=dict(id=ext["best_other_id"], hamming=ext["best_other_hamming"])),
                    template_test=dict(ncc_expected=tt["ncc_expected"], best_other_id=tt["best_other_id"],
                                       ncc_best_other=tt["ncc_best_other"], n_pixels=tt["n_pixels"]),
                    weakest_bit_margin=dec["weakest_bit_margin"], rotation_read=g["rotation_read"])
        # unique among the ids that exist on the carts?
        D = L.hamming_table(np.array(g["bits_read"]), np.array(g["bits_visible_read"]))
        conf["min_hamming_other_cart_ids"] = int(D[others_cart].min())
        amb = ext["n_ids_at_distance0"] > 1
        conf["verdict"] = ("confirmed: unique in the whole dictionary" if not amb else
                           f"confirmed within the ids used on the carts (other cart ids differ by >= {conf['min_hamming_other_cart_ids']} visible bits);"
                           f" {ext['n_ids_at_distance0'] - 1} unrelated dictionary code(s) also fit the visible bits")
        X3 = marker_corners_3d(m["id"], m["cart"], rot[key])
        pf = np.array(m["per_frame"], float)
        rec = dict(cart=m["cart"], id=m["id"], role=m["role"], rot_k=rot[key],
                   rot_k_fit=dict(marker_rms_per_rot=rinfo[key]["marker_rms_per_rot"],
                                  marker_rms_gap_second_minus_best=rinfo[key]["marker_rms_second"] - rinfo[key]["marker_rms_best"],
                                  delta_chi2_second_minus_best=rinfo[key]["delta_chi2_second_minus_best"]),
                   found_by=m["found_by"],
                   corners_px=m["corners_px"], corners_std_px=[None if not np.isfinite(v) else v for v in m["corners_std_px"]],
                   corner_valid=m["corner_valid"], corners_3d_mm=X3, method_per_corner=m["method_per_corner"],
                   id_confirmation=conf,
                   aruco_corners_px=np.array(m["aruco_frames"], float).mean(0), aruco_corner_valid=m["aruco_valid"],
                   edge_corners_px=[None if not np.isfinite(p).all() else p for p in np.nanmean(np.array(m["edge_frames"], float), 0)]
                   if np.isfinite(np.array(m["edge_frames"], float)).any() else None,
                   edge_corner_valid=m["edge_valid"],
                   per_frame=[[None if not np.isfinite(p).all() else p for p in fr] for fr in pf],
                   aruco_per_frame=m["aruco_frames"], edge_per_frame=[[None if not np.isfinite(np.array(p, float)).all() else p for p in fr] for fr in m["edge_frames"]],
                   side_lines=m["side_lines"], side_valid=m["side_valid"],
                   occlusion=g["occlusion"], subpix_window_px=m["subpix_window"])
        final.append(rec)
    save_json(final, f"{CACHE}/markers_final.json")
    summary = dict(detection_stats={k: v for k, v in stats.items() if k != "per_corner"}, rotations={f"{k[0]}:{k[1]}": v for k, v in rinfo.items()},
                   fits={k: {kk: vv for kk, vv in v.items() if kk != "residuals"} for k, v in fits.items()},
                   residuals_final=fits["final"]["residuals"], per_corner_a_vs_b=stats["per_corner"])
    save_json(summary, f"{CACHE}/markers_summary.json")

    # ---- results/detections.json (clean)
    det = dict(
        description=("ArUco sticker corners (DICT_4X4_1000, 90 mm black code square incl. its 1-cell black border) on the two carts in"
                     " the 7 stills. corners_px are in the common MEAN frame (every still was mapped to it with the per-frame jitter"
                     " model below); they are the mean over the 7 stills of per-still measurements. Per corner the edge-based"
                     " measurement is used when both adjacent sides of the black square are visible and straight (sub-pixel side"
                     " traces, line fit, intersection), else the ArUco-rule cornerSubPix corner when the corner neighbourhood is"
                     " visible; corners hidden by an occluder are listed with corner_valid = false and must not be used."),
        image_size=[W, H], frames=frames,
        frame_jitter=dict(model="a point (x, y) of still f maps to the mean frame as (x - (ax0 + ax1*(y-540)), y - (ay0 + ay1*(y-540)))",
                          params_per_frame_ax0_ax1_ay0_ay1=L.seed()["jitter"]),
        pixel_convention="OpenCV: x right, y down, (0,0) = centre of the top-left pixel",
        corner_order="OpenCV ArUco detection order: corner 0 = top-left corner of the code in its canonical (dictionary) orientation, then clockwise as seen on the printed face (1 = top-right, 2 = bottom-right, 3 = bottom-left)",
        corner_to_physical=("Cart frame (spec/cart-marker-layout.json): origin top front-left corner, X along the 1600 mm front, Y into the cart"
                            " (0..450), Z up (0 = top plate, shelves negative). Sticker face up on its surface. With canon = [(-1,+1), (+1,+1),"
                            " (+1,-1), (-1,-1)] (marker x right, y up on the face) and a = rot_k * 90 deg, corner j is at"
                            " centre_mm + [cos a * 45*cx_j - sin a * 45*cy_j,  sin a * 45*cx_j + cos a * 45*cy_j,  0]  (45 = half of the 90 mm"
                            " black square), i.e. corners_3d_mm. rot_k (in-plane rotation, not given by the spec) is chosen per sticker by the"
                            " best reprojection in a joint fit (all 4 candidates tested, gap reported)."),
        corner_validity="corner_valid: both adjacent sides measured (or, for ArUco-rule corners, the corner neighbourhood visible);"
                        " method: 'edge' or 'aruco_subpix'; invalid corners are hidden by an occluder (value = completion, not a measurement)",
        quality=dict(frame_to_frame_std_px=dict(edge_median=stats["frame_std_edge_median_px"], aruco_median=stats["frame_std_aruco_median_px"]),
                     edge_minus_aruco=dict(rms_px=stats["rms_offset_px"], mean_px=stats["edge_minus_aruco_mean_px"],
                                           radial_mean_px=stats["radial_edge_minus_aruco_mean_px"], radial_sem_px=stats["radial_edge_minus_aruco_sem_px"]),
                     joint_fit_f_pp_k1k2_rms_px=fin["rms_px"], joint_fit_note="see work/cache/markers_summary.json"),
        markers=[dict(id=r["id"], cart=r["cart"], role=r["role"], rot_k=r["rot_k"],
                      surface="top plate" if marker_center(r["id"], r["cart"])[2] == 0 else f"shelf {r['role'].split('-')[0]} (Z = {marker_center(r['id'], r['cart'])[2]:.0f} mm)",
                      center_mm=marker_center(r["id"], r["cart"]), corners_px=r["corners_px"], corners_3d_mm=r["corners_3d_mm"],
                      corners_std_px=r["corners_std_px"], corner_valid=r["corner_valid"], method=r["method_per_corner"],
                      detection="plain ArUco detector" if r["found_by"].startswith("plain") else "guided (prediction + rectified code search)",
                      id_confirmation=r["id_confirmation"], per_frame_corners_px=r["per_frame"]) for r in final])
    save_json(det, f"{RESULTS}/detections.json")

    for r, m in zip(final, M):
        m["rot_k"] = r["rot_k"]
    draw_overlay(M, f"{RESULTS}/markers_overlay.png")
    draw_crops(M, f"{RESULTS}/markers_crops.png")
    draw_rect_views(M, f"{RESULTS}/markers_rect_views.png")


if __name__ == "__main__":
    main()
