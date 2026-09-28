"""Stage 11b: corner localisation of all 20 stickers from image content, per frame and on the mean.

Input: work/cache/markers_guided.json (confirmed quads in ArUco corner order, mean frame).

For every sticker and every image (7 stills + the jitter-compensated mean):
  (a) ArUco rule sub-pixel corners: cv2.cornerSubPix with exactly the window / criteria that the
      ArUco detector uses for CORNER_REFINE_SUBPIX (window = 0.3 x module size, capped at 5;
      30 iterations, eps 0.1), started from the confirmed quad. Valid only where the corner
      neighbourhood is visible on both adjacent sides.
      For the stickers that the plain detector finds in a frame, the detector's own output is kept
      as well (`detector`) - it agrees with (a) to ~0.01 px (reported).
  (b) edge-based corners: the four sides of the black square are traced sub-pixel along their
      normal (gradient maximum, black->white outward, samples every 1 px, 18 % excluded at both
      ends), a straight line is fitted per side (total least squares after robust rejection) and
      adjacent lines are intersected. A side is valid if it passes the checks of 11_markers_lib
      (strength, polarity, straightness, full or long partial run) on the MEAN image and was not
      rejected by the shape check of stage 11a (occlusion is static, so the side validity decided on
      the low-noise mean applies to all frames); a corner is valid if both its sides are valid.
      Variant (b') fits lines that are straight after removing a typical lens distortion
      (k1 = -0.29, k2 = 0.08, f = 1300 px, centre = image centre; the curvature over one side is
      < 0.1 px) - reported as a sensitivity check only.
  Stills are measured in their own coordinates and mapped to the common mean frame with the
  per-frame jitter model (x - (ax0 + ax1 (y-540)), y - (ay0 + ay1 (y-540))).

Final corner = mean over the 7 frames of the jitter-corrected corners (method b where valid on the
mean image and measured in >= 4 stills, else a where valid in all stills); corners_std_px = sample
std over the frames used (per-frame scatter; the std of the mean is smaller by sqrt(n)). Also reported: (a) vs (b) offsets (mean,
rms), the signed radial (dilation / erosion) offset along the diagonal, frame-to-frame std of
each method, mean-image vs frame-mean agreement.

Output: work/cache/markers_corners.json
"""
import importlib

import cv2
import numpy as np

from common import CACHE, H, W, K_from, load_json, save_json

L = importlib.import_module("11_markers_lib")
G = importlib.import_module("11_markers_a_guided")

K_CURV = K_from(1300.0, 1300.0, (W - 1) / 2, (H - 1) / 2)
D_CURV = np.array([-0.29, 0.08, 0, 0, 0])


def detector_output(img_u8):
    """Plain ArUco detection (default parameters + SUBPIX) -> {id: [quads]}."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(L.DICT, p)
    out = {}
    for im in (img_u8, cv2.createCLAHE(3.0, (8, 8)).apply(img_u8)):
        cs, ids, _ = det.detectMarkers(im)
        if ids is None:
            continue
        for mid, c in zip(ids.ravel(), cs):
            out.setdefault(int(mid), []).append(c[0].astype(float))
    return out


def measure(img, quad_a, side_valid_a, K=None, dist=None):
    """(a) and (b) corners of one sticker in one image (quad_a: ArUco order, image coordinates)."""
    e = L.edge_quad(img, quad_a, K=K, dist=dist, iters=2)
    # impose the side validity decided on the mean image
    for i in range(4):
        if not side_valid_a[i]:
            e["sides"][i]["valid"] = False
    cvalid = np.array([side_valid_a[(i - 1) % 4] and side_valid_a[i] and e["sides"][(i - 1) % 4]["valid"] and e["sides"][i]["valid"]
                       for i in range(4)])
    # recompute (b) corners strictly from the two measured lines
    qb = np.full((4, 2), np.nan)
    for i in range(4):
        s1, s2 = e["sides"][(i - 1) % 4], e["sides"][i]
        if cvalid[i] and s1["line"][0] is not None and s2["line"][0] is not None:
            p = L.intersect(s1["line"][0], s1["line"][1], s2["line"][0], s2["line"][1])
            qb[i] = L.to_pixels(p, K, dist) if K is not None else p
    start = np.where(np.isfinite(qb), qb, quad_a)
    qa, win = L.subpix(img, start)
    vis = np.array([e["sides"][(i - 1) % 4]["quality"]["near_b"] >= 0.6 and e["sides"][i]["quality"]["near_a"] >= 0.6
                    and side_valid_a[(i - 1) % 4] and side_valid_a[i] for i in range(4)])
    lines = []
    for s in e["sides"]:
        c, d = s["line_px"]
        lines.append(dict(point=None if c is None else c, direction=None if d is None else d, valid=bool(s["valid"]),
                          status=s["quality"]["status"], rms_px=s["rms"] if np.isfinite(s["rms"]) else None,
                          sagitta_px=s["sagitta"] if np.isfinite(s["sagitta"]) else None,
                          n_samples=int(s["quality"]["good"].sum()), frac_good=s["quality"]["frac_good"]))
    return dict(a=qa, a_valid=vis, b=qb, b_valid=cvalid, win=win, lines=lines)


def main():
    guided = load_json(f"{CACHE}/markers_guided.json")
    frames = L.frames()
    mean = L.mean_gray()
    images = [(f, name, g) for f, (name, g) in enumerate(frames)] + [(None, "mean", mean)]
    det_out = {name: detector_output(L.to_u8(g)) for f, name, g in images}
    out = []
    for m in guided:
        if not m.get("found"):
            continue
        r = m["rotation_read"]
        qa0 = np.array(m["quad_aruco"])
        # side validity in ArUco order: ArUco side j = read side (j - r) mod 4
        sv_read = m["side_valid_read"]
        sv = [bool(sv_read[(j - r) % 4]) for j in range(4)]
        per = {}
        for f, name, g in images:
            q_img = qa0 if f is None else L.mean_to_frame(qa0, f)
            res = measure(g, q_img, sv)
            resc = measure(g, q_img, sv, K=K_CURV, dist=D_CURV)
            # back to the mean frame
            to_mean = (lambda P: P) if f is None else (lambda P, f=f: L.frame_to_mean(P, f))
            a = to_mean(res["a"])
            b = to_mean(np.nan_to_num(res["b"], nan=0.0))
            b[~np.isfinite(res["b"]).all(1)] = np.nan
            bc = to_mean(np.nan_to_num(resc["b"], nan=0.0))
            bc[~np.isfinite(resc["b"]).all(1)] = np.nan
            # plain detector output for this sticker in this image (match by id + position)
            detq = None
            for cand in det_out[name].get(m["id"], []):
                cm = to_mean(cand)
                if np.linalg.norm(cm.mean(0) - qa0.mean(0)) < 10:
                    # bring to the same corner order (the detector order is the ArUco order already)
                    detq = cm
            lines = res["lines"]
            if f is not None:
                for ln in lines:
                    if ln["point"] is not None:
                        ln["point"] = to_mean(np.array(ln["point"])[None])[0]
            per[name] = dict(a=a, a_valid=res["a_valid"], b=b, b_valid=res["b_valid"], b_curved=bc, detector=detq,
                             win=res["win"], lines=lines)
        # ---- combine over frames
        fr_names = [n for f, n, g in images if f is not None]
        A = np.array([per[n]["a"] for n in fr_names])  # 7x4x2
        B = np.array([per[n]["b"] for n in fr_names])
        BC = np.array([per[n]["b_curved"] for n in fr_names])
        a_valid = np.all([per[n]["a_valid"] for n in fr_names], axis=0) & per["mean"]["a_valid"]
        # edge corner usable if valid on the mean image and measured in >= 4 of the 7 stills (a partly
        # hidden side can drop below the sample-count threshold in a noisy still)
        n_b = np.isfinite(B).all(-1).sum(0)
        b_valid = per["mean"]["b_valid"] & (n_b >= 4)
        final = np.full((4, 2), np.nan)
        fstd = np.full(4, np.nan)
        method = []
        per_frame = np.full((len(fr_names), 4, 2), np.nan)
        for i in range(4):
            if b_valid[i]:
                P = B[:, i]
                method.append("edge")
            elif a_valid[i]:
                P = A[:, i]
                method.append("aruco_subpix")
            else:
                method.append("none")
                continue
            per_frame[:, i] = P
            ok = np.isfinite(P).all(1)
            final[i] = P[ok].mean(0)
            fstd[i] = float(np.sqrt(np.sum(np.sum((P[ok] - final[i]) ** 2, 1)) / max(ok.sum() - 1, 1)))
        valid = np.array([mm != "none" for mm in method])
        # fallback geometry for invalid corners (for drawing / decoding only): mean-image completed quad
        corners_all = np.where(np.isfinite(final), final, qa0)
        det_frames = [per[n]["detector"] for n in fr_names]
        entry = dict(cart=m["cart"], id=m["id"], role=m["role"], found_by=m["method"], rotation_read=r,
                     side_valid=sv, corners_px=corners_all, corner_valid=valid, corners_std_px=fstd,
                     method_per_corner=method, per_frame=per_frame, n_frames_edge=n_b,
                     aruco_frames=A, aruco_valid=a_valid, edge_frames=B, edge_valid=b_valid, edge_curved_frames=BC,
                     aruco_mean_image=per["mean"]["a"], edge_mean_image=per["mean"]["b"],
                     aruco_frame_mean=A.mean(0), edge_frame_mean=np.nanmean(B, 0) if np.isfinite(B).any() else None,
                     aruco_frame_std=np.sqrt(np.mean(np.sum((A - A.mean(0)) ** 2, -1), 0)),
                     edge_frame_std=np.sqrt(np.nanmean(np.sum((B - np.nanmean(B, 0)) ** 2, -1), 0)) if np.isfinite(B).any() else None,
                     detector_frames=[None if q is None else q for q in det_frames],
                     detector_mean_image=per["mean"]["detector"],
                     subpix_window=per["mean"]["win"], side_lines=per["mean"]["lines"],
                     completed_quad_mean=qa0)
        out.append(entry)
        print(f"{m['cart']:4d} {m['id']:4d} valid={valid.astype(int)} method={[x[0] for x in method]} "
              f"std={np.round(fstd, 3)} a-b={np.round(np.linalg.norm(A.mean(0) - np.nanmean(B, 0), axis=1), 2) if np.isfinite(B).any() else None}")
    save_json(out, f"{CACHE}/markers_corners.json")


if __name__ == "__main__":
    main()
