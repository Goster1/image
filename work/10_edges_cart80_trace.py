"""Cart 80: final sub-pixel tracing of the curated edges (edges80_defs.py), cleaning, statistics,
zoomed verification crops and the JSON / overview outputs.

Per edge: smooth guide (deg-3 fit of the curated polyline, optionally extended at the ends) ->
edges80lib.trace_edge (= edgelib.trace_edge algorithm: gradient extremum along the normal, parabolic
sub-pixel peak, iterated, polynomial-guided path) -> cleaning: weak points (strength < 0.5 x median),
contrast changes (label holders, cards: local contrast outside 0.5..1.8 x median), points off the smooth
curve (|res| > max(0.25 px, 4 robust sigma)), +-6 px trimmed at every interruption, fragments < 12 px
dropped, keep/exclude windows from the table.
Outputs: work/cache/edges_cart80.json, work/cache/edgecrops_cart80/<id>.png, results/edges_cart80.png,
work/cache/edges80_summary.json (extra diagnostics: VP-direction angles, residual noise)."""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE, RESULTS, save_json
import edgelib
import edges80lib as L
from edges80_defs import EDGES

REGION = "cart80"
CROPDIR = f"{CACHE}/edgecrops_{REGION}"
# shelf-sticker cards (markers 7/5/3 at the X=0 end, 8/6/4 at the X=1600 end): straight card borders run
# along / across the lip lines there and are not cart structure -> removed from every lip / post trace
CARD_BOXES = [[1243, 14, 1406, 99], [1174, 648, 1242, 712], [1208, 726, 1276, 788], [1248, 822, 1318, 886]]
DIRCOL = {"cartX": (0, 0, 255), "cartY": (0, 200, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
          "floor_plane": (0, 255, 255), "horizontal_other": (0, 160, 255), "unknown": (200, 200, 200)}


def chord_frame(P):
    c, d, _, _ = edgelib.line_fit(P)
    if abs(d[1]) > abs(d[0]):
        d = d if d[1] > 0 else -d
    else:
        d = d if d[0] > 0 else -d
    return c, d, np.array([-d[1], d[0]])


def smooth_guide(G, ext=(0, 0), step=4.0):
    G = np.asarray(G, float)
    c, d, n = chord_frame(G)
    t = (G - c) @ d
    r = (G - c) @ n
    deg = min(3, len(G) - 1)
    q = np.polyfit(t, r, deg)
    q2 = np.polyfit(t, r, min(2, len(G) - 1))
    tt = np.arange(t.min() - ext[0], t.max() + ext[1] + 1e-9, step)
    inside = (tt >= t.min()) & (tt <= t.max())
    rr = np.where(inside, np.polyval(q, tt), np.polyval(q2, tt))  # quadratic extrapolation outside
    out = c[None] + tt[:, None] * d[None] + rr[:, None] * n[None]
    # keep the original orientation
    if np.dot(out[-1] - out[0], G[-1] - G[0]) < 0:
        out = out[::-1]
    return out


def local_contrast(P, N, pol, off=2.5):
    a = map_coordinates(L.IMG, [P[:, 1] + off * N[:, 1], P[:, 0] + off * N[:, 0]], order=1)
    b = map_coordinates(L.IMG, [P[:, 1] - off * N[:, 1], P[:, 0] - off * N[:, 0]], order=1)
    return (a - b) * pol


def running_median(t, v, keep, half=50.0):
    out = np.empty(len(t))
    for i in range(len(t)):
        m = keep & (np.abs(t - t[i]) <= half)
        out[i] = np.median(v[m]) if m.sum() >= 3 else np.median(v[keep])
    return out


def local_residual(t, r, keep, half=40.0):
    """residual of every point w.r.t. a quadratic fitted to the kept neighbours within +-half px
    (arc length along the chord), excluding the point itself -> robust to the global (distortion)
    curvature of long edges."""
    res = np.full(len(t), np.inf)
    for i in range(len(t)):
        m = keep & (np.abs(t - t[i]) <= half)
        m[i] = False
        if m.sum() < 6:
            m = keep & (np.abs(t - t[i]) <= 2 * half)
            m[i] = False
            if m.sum() < 6:
                continue
        tt = t[m] - t[i]
        deg = 2 if np.ptp(tt) > 20 else 1
        q = np.polyfit(tt, r[m], deg)
        res[i] = r[i] - np.polyval(q, 0.0)
    return res


def trace_def(e):
    G = smooth_guide(e["guide"], ext=e.get("ext", [0, 0]))
    hw = e.get("hw", 3.0)
    # polarity from the guide itself
    pol = e.get("pol")
    if pol is None:
        o = L.trace_edge(G, halfwidth=hw, step=2.0, polarity=0, sigma=1.0, iters=2, min_strength=0.0, fit_deg=3)
        Q, nrm = o["points"], o["normal"]
        pol = 1 if np.median(local_contrast(Q, nrm, 1)) > 0 else -1
    o = L.trace_edge(G, halfwidth=hw, step=2.0, polarity=pol, sigma=1.0, iters=3, min_strength=-1e9, fit_deg=3)
    P, S, N = o["points"], o["strength"], o["normal"]
    C = local_contrast(P, N, pol)
    n = len(P)
    ok_basic = np.ones(n, bool)
    if "keep" in e:
        for ax, (lo, hi) in e["keep"].items():
            k = 0 if ax == "x" else 1
            ok_basic &= (P[:, k] >= lo) & (P[:, k] <= hi)
    boxes = list(e.get("excl", []))
    if e.get("card_excl", e["direction"] in ("cartX", "unknown") and not e["id"].startswith("x_board")):
        boxes += CARD_BOXES
    for (x0, y0, x1, y1) in boxes:
        ok_basic &= ~((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
    ok_basic &= (P[:, 0] > 2) & (P[:, 0] < 1917) & (P[:, 1] > 2) & (P[:, 1] < 1077)
    c, d, nn = chord_frame(P[ok_basic] if ok_basic.sum() > 10 else P)
    t = (P - c) @ d
    r = (P - c) @ nn
    keep = ok_basic & (S > 0.2 * np.median(S[ok_basic]))
    cr = e.get("contrast_range")  # optional (off by default: silhouettes change contrast legitimately)
    for _ in range(8):
        medS = np.median(S[keep])
        locS = running_median(t, S, keep, half=50.0)
        res = local_residual(t, r, keep, half=e.get("local_half", 40.0))
        sig = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = ok_basic & (np.abs(res) < max(e.get("res_tol", 0.3), 4 * sig)) & \
            (S > e.get("min_rel", 0.5) * locS) & (S > 0.2 * medS) & (S > e.get("min_abs", 6.0))
        if cr is not None:
            medC = np.median(C[keep])
            new &= (C > cr[0] * medC) & (C < cr[1] * medC)
        if np.array_equal(new, keep) or new.sum() < 8:
            keep = new if new.sum() >= 8 else keep
            break
        keep = new
    # trim around interruptions (runs of >= 2 rejected points) by 6 px of arc length
    s = o["s"]
    bad = ok_basic & ~keep
    trim = e.get("trim", 6.0)
    runs = []
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            if j - i + 1 >= 2:
                runs.append((s[i], s[j]))
            i = j + 1
        else:
            i += 1
    for s0, s1 in runs:
        keep &= ~((s >= s0 - trim) & (s <= s1 + trim))
    # fragments
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return None
    frags = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    for f in frags:
        if len(f) < e.get("min_frag", 6):
            keep[f] = False
    idx = np.where(keep)[0]
    if len(idx) < 8:
        return None
    Pk, Sk = P[keep], S[keep]
    # final smooth-curve residual (noise) in the chord frame
    c2, d2, n2 = chord_frame(Pk)
    tk = (Pk - c2) @ d2
    rk = (Pk - c2) @ n2
    lres = local_residual(tk, rk, np.ones(len(tk), bool), half=40.0)
    noise = float(np.sqrt(np.mean(lres[np.isfinite(lres)] ** 2)))
    return dict(points=Pk, strength=Sk, rejected=P[ok_basic & ~keep], pol=pol, noise=noise,
                n_frag=len(np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)))


def implied_position(e, P):
    """where the initial calibration (~5 px) puts the 3D line of this edge (information only)."""
    ml = e.get("model_line")
    if not ml:
        return ""
    fx = ml["fixed"]
    try:
        if e["direction"] == "cartX" and "Y" in fx:
            m = []
            for z in (-2000.0, 500.0):
                m.append(np.mean(L.backproject_plane(P, 2, z)[:, 1]))
            z = -2000.0 + (fx["Y"] - m[0]) * 2500.0 / (m[1] - m[0])
            return f" [initial calibration (~5 px): this image line lies at Z = {z:.0f} mm if Y = {fx['Y']}]"
        if e["direction"] == "cartY" and "Z" in fx:
            x = np.mean(L.backproject_plane(P, 2, fx["Z"])[:, 0])
            return f" [initial calibration (~5 px): X = {x:.0f} mm if Z = {fx['Z']}]"
        if e["direction"] == "cartZ" and "Y" in fx:
            x = np.mean(L.backproject_plane(P, 1, fx["Y"])[:, 0])
            return f" [initial calibration (~5 px): X = {x:.0f} mm if Y = {fx['Y']}]"
    except Exception:
        return ""
    return ""


def main():
    out_edges = []
    summary = []
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    over = img.copy()
    for e in EDGES:
        eid = f"{REGION}_{e['id']}"
        res = trace_def(e)
        if res is None:
            print("FAILED", eid)
            continue
        P = res["points"]
        bow, rms = edgelib.sagitta(P)
        _, _, _, mx = edgelib.line_fit(P)
        length = float(np.hypot(*(P[-1] - P[0])))
        ang = L.classify_dir(P[0], P[-1])
        crop = L.edge_strip(P, label=eid.replace(REGION + "_", ""), win=64, scale=4, others=[res["rejected"]])
        cpath = f"{CROPDIR}/{eid}.png"
        cv2.imwrite(cpath, crop)
        out_edges.append({
            "id": eid, "what": e["what"], "class": e.get("cls", "cart_structure"), "cart": e.get("cart", 80),
            "straight_3d": e.get("straight_3d", True), "direction": e["direction"],
            "model_line": (dict(e["model_line"], note=e["model_line"]["note"] + implied_position(e, P))
                           if e.get("model_line") else None),
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in P],
            "strength": [round(float(v), 2) for v in res["strength"]],
            "chord_dev_rms_px": round(rms, 3), "chord_bow_px": round(bow, 3),
            "verified": e.get("verified", "no"), "verification_note": e.get("note", ""),
            "crop": f"work/cache/edgecrops_{REGION}/{eid}.png"})
        summary.append(dict(id=eid, n=len(P), chord_length_px=round(length, 1), noise_px=round(res["noise"], 3),
                            n_fragments=res["n_frag"], polarity=res["pol"],
                            angle_to_vp_deg={k: round(v, 2) for k, v in ang.items()},
                            median_strength=round(float(np.median(res["strength"])), 1),
                            chord_max_dev_px=round(mx, 2)))
        print(f"{eid:34s} n={len(P):4d} L={length:6.1f} bow={bow:6.2f} noise={res['noise']:.3f} "
              f"frag={res['n_frag']:2d} ang X/Y/Z={ang['cartX']:5.1f}/{ang['cartY']:5.1f}/{ang['cartZ']:5.1f}")
        col = DIRCOL[e["direction"]]
        cv2.polylines(over, [np.round(P * 8).astype(np.int32)], False, col, 1, cv2.LINE_AA, shift=3)
        m = P[len(P) // 2]
        cv2.putText(over, e["id"], (int(m[0]) + 3, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1,
                    cv2.LINE_AA)
    y = 20
    for k, col in DIRCOL.items():
        if any(ed["direction"] == k for ed in out_edges):
            cv2.putText(over, k, (1760, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += 18
    cv2.imwrite(f"{RESULTS}/edges_{REGION}.png", over)
    cv2.imwrite(f"{CROPDIR}/overview_zoom.png", cv2.resize(over[0:1000, 1100:1760], None, fx=1.5, fy=1.5,
                                                            interpolation=cv2.INTER_CUBIC))
    save_json({"region": REGION,
               "image": "work/cache/mean_aligned_gray.npy (jitter-compensated mean of the 7 stills)",
               "edges": out_edges}, f"{CACHE}/edges_{REGION}.json")
    save_json(summary, f"{CACHE}/edges80_summary.json")
    print(len(out_edges), "edges written")


if __name__ == "__main__":
    main()
