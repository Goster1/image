"""Stage 11a: guided detection and id confirmation of all 20 expected stickers (mean image).

For every expected sticker (ids 0..8 + the unique id on each cart):
  * plainly detected stickers (work/cache/markers_initial.json, 12 of 20): their mean (jitter
    aligned) ArUco quad is the start;
  * the others: the quad predicted by the seed calibration (work/cache/initial_calib.json) gives the
    position; the SHAPE is taken from the nearest already localised sticker of the same cart (its
    measured quad scaled by the predicted size ratio, moved to the predicted centre) because the seed
    model mis-predicts sticker sizes by up to 13 %. The neighbourhood (+-6 cells) is rectified
    through that quad (fronto-parallel square view) and searched with the rendered template of the
    expected code (6x6 cells + 10 mm white margin) in all 4 in-plane rotations, 9 scales and 5
    small angles (normalised cross-correlation). The best match gives the quad (read order).
  * every quad is then refined on image edges (sub-pixel trace of the four black->white sides);
    sides that are hidden by an occluder (strength collapse / wrong polarity / not straight) are
    flagged and completed from the template shape; the occluded band is measured and the cells
    under it are excluded;
  * ID CONFIRMATION (image content only):
      (1) bit decoding: the visible cells of the 6x6 grid are thresholded (mid-level between the
          black border and the white margin) and compared with all 1000 codes x 4 rotations;
          cells cut by the occluded band are read from their visible part when >= 30 % of the cell
          is visible and the reading is clear; reported: number of visible bits, Hamming distance
          to the expected id, number of dictionary codes at distance 0, best other code, minimum
          distance to the other ids that exist on the carts;
      (2) masked dictionary template test: the visible part of the rectified sticker (visible cells
          + white margin except along hidden sides) is correlated with the rendering of every
          code x 4 rotations; the expected code must not be beaten by any other code.
    A sticker is accepted only if (1) gives Hamming 0 for the expected id over >= 8 fully visible
    bits (and no contradiction on the partly visible ones) and (2) holds.

Output: work/cache/markers_guided.json, rectified views work/cache/markers11/rect_<cart>_<id>.png
"""
import importlib
import os

import cv2
import numpy as np

from common import CACHE, cart_marker_ids, load_json, marker_corners_3d, marker_role, project, save_json

L = importlib.import_module("11_markers_lib")
OUTDIR = os.path.join(CACHE, "markers11")
os.makedirs(OUTDIR, exist_ok=True)

C_RECT = 10.0  # px per cell in the rectified search patch
M_RECT = 6.0  # margin (cells) around the predicted square


def predicted_quad(cart, mid):
    sd = L.seed()
    pose = sd["poses"][cart]
    return project(marker_corners_3d(mid, cart, 0), sd["K"], sd["dist"], rvec=pose[:3], tvec=pose[3:])


def plain_quads():
    """Mean ArUco quads (jitter aligned) of the plainly detected stickers."""
    d = load_json(f"{CACHE}/initial_calib.json")
    return {tuple(k): np.array(c) for k, c in zip(d["keys"], d["corners_mean"])}


_RENDER_CACHE = {}


def _render_weights(C, scale, ang, margin, ss):
    """Per template pixel: fraction of its ss x ss supersamples falling in each of the 36 cells
    (index 36 = white margin / outside); cached per geometry."""
    key = (C, round(scale, 4), round(ang, 4), margin, ss)
    if key not in _RENDER_CACHE:
        ext = (3 + margin) * C * scale
        T = int(np.ceil(2 * ext)) | 1
        o = (np.arange(ss) + 0.5) / ss - 0.5
        yy, xx = np.mgrid[0:T, 0:T].astype(float) - (T - 1) / 2
        X = xx[..., None] + np.tile(o, ss)[None, None, :]
        Y = yy[..., None] + np.repeat(o, ss)[None, None, :]
        a = np.deg2rad(ang)
        u = (np.cos(a) * X + np.sin(a) * Y) / (C * scale) + 3
        v = (-np.sin(a) * X + np.cos(a) * Y) / (C * scale) + 3
        inside = (u >= 0) & (u < 6) & (v >= 0) & (v < 6)
        cell = np.where(inside, np.clip(v.astype(int), 0, 5) * 6 + np.clip(u.astype(int), 0, 5), 36)
        Wt = np.zeros((T * T, 37))
        cf = cell.reshape(T * T, -1)
        for k in range(cf.shape[1]):
            np.add.at(Wt, (np.arange(T * T), cf[:, k]), 1.0 / cf.shape[1])
        inm = ((u >= -margin) & (u < 6 + margin) & (v >= -margin) & (v < 6 + margin)).all(-1)
        _RENDER_CACHE[key] = (T, Wt, (inm.astype(np.uint8) * 255))
    return _RENDER_CACHE[key]


def render_code(mid, rot, C, scale=1.0, ang=0.0, margin=0.667, ss=4):
    """Rendered sticker template (black square with the code of `mid` rotated by np.rot90(.., rot),
    white margin) centred in an odd-sized window; mask = square + margin. Anti-aliased by
    ss x ss supersampling. Returns (template float32 0..1, mask uint8)."""
    T, Wt, mask = _render_weights(C, scale, ang, margin, ss)
    vals = np.r_[np.rot90(L.code_grid(mid), rot).ravel().astype(float), 1.0]
    return (Wt @ vals).reshape(T, T).astype(np.float32), mask


def code_search(patch, mid, C, N, radius=5.0, scales=np.arange(0.88, 1.121, 0.03), angles=(-4, -2, 0, 2, 4)):
    """Search the rectified patch for the expected code (4 rotations, scales, angles).
    Returns best dict(score, cx, cy, scale, ang, rot) and the per-rotation best scores."""
    best = None
    per_rot = np.full(4, -1.0)
    for sc in scales:
        for ang in angles:
            for rot in range(4):
                t, m = render_code(mid, rot, C, sc, ang)
                r = cv2.matchTemplate(patch.astype(np.float32), t, cv2.TM_CCOEFF_NORMED, mask=m)
                r[~np.isfinite(r)] = -1
                T = t.shape[0]
                yy, xx = np.mgrid[0:r.shape[0], 0:r.shape[1]]
                far = np.hypot(xx + (T - 1) / 2 - N / 2, yy + (T - 1) / 2 - N / 2) > radius * C
                r[far] = -1
                k = np.unravel_index(np.argmax(r), r.shape)
                sc_ = float(r[k])
                per_rot[rot] = max(per_rot[rot], sc_)
                if best is None or sc_ > best["score"]:
                    best = dict(score=sc_, cx=k[1] + (T - 1) / 2, cy=k[0] + (T - 1) / 2, scale=float(sc), ang=float(ang),
                                rot=rot, T=T)
    return best, per_rot


def quad_from_search(best, C):
    """Square (rect coords, read order = rect order) of a code-search result."""
    s = 6 * C * best["scale"] / 2
    a = np.deg2rad(best["ang"])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * s
    return sq @ R.T + [best["cx"], best["cy"]]


# ------------------------------------------------------------------------------------------
# occlusion analysis in marker cell coordinates
# ------------------------------------------------------------------------------------------
# side i of a quad in read order joins corner i and i+1: 0 = top (v=0), 1 = right (u=6),
# 2 = bottom (v=6), 3 = left (u=0)

def _to_cell_H(quad):
    return L.homog4(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], float), np.asarray(quad, float))


def sample_uv(img, quad, uv):
    from scipy.ndimage import map_coordinates

    P = L.apply_H(_to_cell_H(quad), np.asarray(uv, float).reshape(-1, 2))
    return map_coordinates(img, [P[:, 1], P[:, 0]], order=1, mode="nearest")


def occluded_extent(img, quad, side, black, white, n=121):
    """Depth (cells, measured from the hidden side inward) up to which the two border columns
    perpendicular to `side` are NOT black, i.e. covered by a bright occluder. Returns (d_a, d_b)
    for the two perpendicular border columns (at 0.5 and 5.5 cells)."""
    t = np.linspace(0.0, 6.0, n)  # distance from the side
    out = []
    for w in (0.5, 5.5):
        if side == 0:
            uv = np.column_stack([np.full(n, w), t])
        elif side == 1:
            uv = np.column_stack([6 - t, np.full(n, w)])
        elif side == 2:
            uv = np.column_stack([np.full(n, w), 6 - t])
        else:
            uv = np.column_stack([t, np.full(n, w)])
        v = (sample_uv(img, quad, uv) - black) / max(white - black, 1e-6)
        dark = v < 0.5
        # walk from the far end (t=6) towards the side; the black border column must be dark
        k = n - 1
        while k > 0 and dark[k - 1]:
            k -= 1
        out.append(float(t[k]) if not dark[0] else 0.0)
    return out


def margin_dark_fraction(img, quad, side, black, white, n=40):
    """Fraction of the white margin (0.35 cell outside `side`) that is not bright (dark occluder)."""
    s = np.linspace(0.8, 5.2, n)
    o = -0.35
    if side == 0:
        uv = np.column_stack([s, np.full(n, o)])
    elif side == 1:
        uv = np.column_stack([np.full(n, 6 - o), s])
    elif side == 2:
        uv = np.column_stack([s, np.full(n, 6 - o)])
    else:
        uv = np.column_stack([np.full(n, o), s])
    v = (sample_uv(img, quad, uv) - black) / max(white - black, 1e-6)
    return float(np.mean(v < 0.6))


def occlusion_mask(img, quad, side_valid, black, white, e=None, safety=0.25, min_depth=1.08):
    """Inner-bit visibility mask (6x6 cells, True = usable) + description of the occluded bands.

    For every hidden (invalid) side the occluded band depth (cells from that side, at both ends) is
    the largest of: (1) where the black border columns perpendicular to it stop being dark (bright
    occluder), (2) where the good edge samples of the perpendicular sides stop (any occluder),
    (3) 1.08 cells = the untraced 18 % end zone of the perpendicular sides (an occluder shorter than
    that cannot be measured, so the border row and the adjacent inner row are dropped). A cell is
    excluded when its central 60 % reaches into the band + `safety` cells."""
    bands = []
    for sdx in range(4):
        if side_valid[sdx]:
            continue
        da, db = occluded_extent(img, quad, sdx, black, white)
        ta, tb = occluded_depths_from_traces(e, sdx) if e is not None else (0.0, 0.0)
        if sdx in (2, 3):  # trace depths are (start corner, end corner); border scan is (w=0.5, w=5.5)
            ta, tb = tb, ta
        mdark = margin_dark_fraction(img, quad, sdx, black, white)
        # occluded_extent measures at u/v = 0.5 (start of the side) and 5.5 (end); the trace depth
        # list is (side before = start end, side after = far end)
        depth_a = max(da, ta, min_depth)
        depth_b = max(db, tb, min_depth)
        kind = "bright occluder" if max(da, db) > 0.3 else ("dark occluder / shadow over the margin" if mdark > 0.3 else "edge not usable")
        bands.append(dict(side=sdx, depth=[depth_a, depth_b], border_scan=[da, db], trace_scan=[ta, tb],
                          margin_dark=mdark, kind=kind))
    # visibility of sample points inside every cell
    ok = np.ones((6, 6), bool)
    partial = {}
    t80 = np.linspace(0.1, 0.9, 8)
    for r in range(6):
        for c in range(6):
            U, V = np.meshgrid(c + t80, r + t80)
            vis = np.ones(U.shape, bool)
            for b in bands:
                sdx = b["side"]
                depth_a, depth_b = b["depth"]
                dist, pos = [(V, U), (6 - U, V), (6 - V, U), (U, V)][sdx]
                lim = depth_a + (depth_b - depth_a) * (pos - 0.5) / 5.0 + safety
                vis &= dist >= lim
            core = (U >= c + 0.2) & (U <= c + 0.8) & (V >= r + 0.2) & (V <= r + 0.8)
            if not vis[core].all():
                ok[r, c] = False
                if vis.mean() >= 0.3:
                    partial[(r, c)] = np.column_stack([U[vis], V[vis]])
    return ok, partial, bands


def decode_partial(img, quad, partial, dec):
    """Read the partially visible inner cells from their visible sample points only."""
    bits = {}
    for (r, c), uv in partial.items():
        if not (1 <= r <= 4 and 1 <= c <= 4):
            continue
        val = sample_uv(img, quad, uv).mean()
        m = (val - dec["thr"]) / max(dec["white"] - dec["black"], 1e-6)
        bits[(r - 1, c - 1)] = (int(val > dec["thr"]), float(m), len(uv) / 64.0)
    return bits


def extended_report(dec, pbits, mid):
    """Hamming comparison using fully visible + partially visible bits."""
    b = dec["bits"].copy()
    v = dec["valid"].copy()
    for (r, c), (bit, m, frac) in pbits.items():
        if abs(m) >= 0.2:  # clear black / white reading only
            b[r, c] = bit
            v[r, c] = True
    D = L.hamming_table(b, v)
    Dm = D.min(1)
    others = Dm.copy()
    others[mid] = 99
    return dict(n_bits=int(v.sum()), n_partial_used=int(v.sum() - dec["valid"].sum()), hamming_expected=int(D[mid].min()),
                rotation=int(np.argmin(D[mid])), n_ids_at_distance0=int((Dm == 0).sum()),
                best_other_id=int(np.argmin(others)), best_other_hamming=int(others.min()),
                partial_cells=[dict(cell=[int(r), int(c)], bit=int(bit), margin=float(m), visible_fraction=float(fr))
                               for (r, c), (bit, m, fr) in pbits.items()])


def _similarity_to(T, Q):
    """Least-squares similarity (a, b, tx, ty) mapping T -> Q (p' = [[a,-b],[b,a]] (p - cT) + cT + t)."""
    cT = T.mean(0)
    X = T - cT
    A = []
    y = []
    for (px, py), (qx, qy) in zip(X, Q - cT):
        A.append([px, -py, 1, 0])
        y.append(qx)
        A.append([py, px, 0, 1])
        y.append(qy)
    return np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]


def _apply_sim(par, T):
    a, b, tx, ty = par
    cT = T.mean(0)
    X = T - cT
    return np.column_stack([a * X[:, 0] - b * X[:, 1], b * X[:, 0] + a * X[:, 1]]) + cT + [tx, ty]


def complete_quad(e, template, quad_now, prior_rel=0.05):
    """Corners of the black square from the measured side lines; sides that are hidden (invalid)
    are taken from the template shape (a measured neighbour quad), placed by a similarity fitted to
    the measured lines (+ weak scale prior). Returns (quad, info)."""
    from scipy.optimize import least_squares

    valid = [s["valid"] for s in e["sides"]]
    lines = [s["line_px"] for s in e["sides"]]
    T = np.asarray(template, float)
    par0 = _similarity_to(T, np.asarray(quad_now, float))
    s0 = np.hypot(par0[0], par0[1])
    side_px = np.linalg.norm(np.roll(T, -1, 0) - T, axis=1).mean() * s0
    info = dict(n_valid_sides=int(sum(valid)))
    if not all(valid):
        def res(par):
            Q = _apply_sim(par, T)
            r = []
            for i in range(4):
                if not valid[i]:
                    continue
                c, d = lines[i]
                nn = np.array([-d[1], d[0]])
                for t in (0.0, 0.5, 1.0):
                    p = Q[i] + t * (Q[(i + 1) % 4] - Q[i])
                    r.append(np.dot(p - c, nn))
            r.append((np.hypot(par[0], par[1]) / s0 - 1) / prior_rel * 0.5)
            return np.array(r)

        rr = least_squares(res, par0)
        model = _apply_sim(rr.x, T)
        info.update(scale_vs_prior=float(np.hypot(rr.x[0], rr.x[1]) / s0), fit_rms=float(np.sqrt(np.mean(rr.fun[:-1] ** 2))) if len(rr.fun) > 1 else None)
    else:
        model = np.asarray(quad_now, float)
    L_ = []
    for i in range(4):
        if valid[i]:
            L_.append(lines[i])
        else:
            a, b = model[i], model[(i + 1) % 4]
            L_.append((0.5 * (a + b), (b - a) / np.linalg.norm(b - a)))
    q = np.zeros((4, 2))
    for i in range(4):
        (c1, d1), (c2, d2) = L_[(i - 1) % 4], L_[i]
        q[i] = L.intersect(c1, d1, c2, d2)
    info["side_px"] = float(side_px)
    return q, info


def occluded_depths_from_traces(e, side):
    """Depth of the occluded band next to hidden side `side` (cells), from where the good samples of
    the two perpendicular sides stop (their traces cover 18..82 % of the side length)."""
    out = []
    for sd, towards_b in (((side - 1) % 4, True), ((side + 1) % 4, False)):
        q = e["sides"][sd]["quality"]
        g = q["good"]
        n = len(g)
        if n == 0 or not e["sides"][sd]["valid"]:
            out.append(0.0)  # perpendicular side itself unusable -> no information from it
            continue
        tr = e["sides"][sd]["trace"]
        Ls = tr["s"][-1] / (1 - 0.18)  # full side length (px)
        # side sd runs from corner sd to corner sd+1; the hidden side is at its end (b) if towards_b
        idx = np.where(g)[0]
        if len(idx) == 0:
            out.append(6.0)
            continue
        if towards_b:
            last = tr["s"][idx.max()]
            depth = (Ls - last) / Ls * 6
        else:
            first = tr["s"][idx.min()]
            depth = first / Ls * 6
        out.append(float(depth))
    return out


def _aspect(q):
    l = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
    return (l[0] + l[2]) / (l[1] + l[3])


def _reject(e, rej):
    for i in rej:
        e["sides"][i]["valid"] = False
        e["sides"][i]["quality"]["status"] = "rejected (shape)"
        e["sides"][i]["quality"]["valid"] = False
    for i in range(4):
        e["corner_valid"][i] = e["sides"][(i - 1) % 4]["valid"] and e["sides"][i]["valid"]
    return e


def analyse_quad(img, quad, mid, template=None, aspect_tol=0.05):
    """Edge refinement + hidden-side completion + occlusion + decoding of one quad (read order).

    Shape check: an occluder edge parallel to a hidden side can pass all per-side tests (it is a
    straight dark->bright edge); it then makes the quad too narrow. If the aspect ratio of the
    measured quad differs from the template's (neighbour shape) by more than `aspect_tol`, the side
    responsible (of the pair that bounds the too-short direction: a 'partial' one first, else the
    one with fewer good samples) is rejected and completed from the template."""
    template = np.asarray(quad, float) if template is None else np.asarray(template, float)
    rej = set()
    q = np.asarray(quad, float)
    e = None
    for it in range(6):
        e = _reject(L.edge_quad(img, q, iters=2 if it == 0 else 1), rej)
        q, cinfo = complete_quad(e, template, q)
        dev = _aspect(q) / _aspect(template) - 1
        if abs(dev) <= aspect_tol or len(rej) >= 2:
            if it >= 1:
                break
            continue
        pair = [1, 3] if dev < 0 else [0, 2]
        cand = [i for i in pair if e["sides"][i]["valid"] and i not in rej]
        if not cand:
            break
        cand.sort(key=lambda i: (e["sides"][i]["quality"]["status"] != "partial", e["sides"][i]["quality"]["frac_good"]))
        rej.add(cand[0])
    e2 = _reject(L.edge_quad(img, q, iters=1), rej)
    q, cinfo = complete_quad(e2, template, q)
    cinfo["aspect_vs_template"] = float(_aspect(q) / _aspect(template))
    cinfo["rejected_sides"] = sorted(rej)
    side_valid = [s["valid"] for s in e2["sides"]]
    black, white = e2["black"], e2["white"]
    ok, partial, bands = occlusion_mask(img, q, side_valid, black, white, e2)
    dec = L.decode(img, q, cell_ok=ok)
    rep = L.id_report(dec, mid)
    pbits = decode_partial(img, q, partial, dec)
    rep["extended"] = extended_report(dec, pbits, mid)
    tt = masked_template_test(img, q, ok, side_valid, mid)
    return dict(quad=q, edge=e2, side_valid=side_valid, side_status=[s["quality"]["status"] for s in e2["sides"]],
                cell_ok=ok, bands=bands, dec=dec, rep=rep, completion=cinfo, template_test=tt, partial_bits=pbits)


_ALLGRID = None


def masked_template_test(img, quad, cell_ok, side_valid, mid, C=10.0, M=1.0, edge_excl=0.12):
    """Dictionary template test on the VISIBLE part: the neighbourhood is rectified through the final
    quad; pixels of visible cells (border cells included) and of the white margin (except along
    hidden sides) are compared with the rendering of every code x 4 rotations by normalised
    correlation. Pixels within `edge_excl` cells of a cell boundary are skipped (blur).
    Returns rank of the expected id and the margin to the best other code."""
    global _ALLGRID
    if _ALLGRID is None:
        g = []
        for i in range(len(L.DICT.bytesList)):
            for r in range(4):
                g.append(np.r_[np.rot90(L.code_grid(i), r).ravel(), 1])
        _ALLGRID = np.array(g, np.float32)  # (4000, 37)
    Hm, N = L.rect_H(quad, C, M)
    patch = L.warp(img, Hm, N)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    u = (xx - M * C) / C
    v = (yy - M * C) / C
    inside = (u >= 0) & (u < 6) & (v >= 0) & (v < 6)
    fu, fv = u - np.floor(u), v - np.floor(v)
    far_from_bound = (fu > edge_excl) & (fu < 1 - edge_excl) & (fv > edge_excl) & (fv < 1 - edge_excl)
    ci = np.where(inside, np.clip(v.astype(int), 0, 5) * 6 + np.clip(u.astype(int), 0, 5), 36)
    vis_cell = np.zeros_like(inside)
    vis_cell[inside] = np.asarray(cell_ok, bool).ravel()[ci[inside]]
    mg = 0.667
    margin = (~inside) & (u > -mg + edge_excl) & (u < 6 + mg - edge_excl) & (v > -mg + edge_excl) & (v < 6 + mg - edge_excl) \
        & ((u < -edge_excl) | (u > 6 + edge_excl) | (v < -edge_excl) | (v > 6 + edge_excl))
    hidden_strip = np.zeros_like(margin)
    for sdx, ok_ in enumerate(side_valid):
        if ok_:
            continue
        hidden_strip |= [(v < 0), (u > 6), (v > 6), (u < 0)][sdx]
    mask = (inside & vis_cell & far_from_bound) | (margin & ~hidden_strip)
    x = patch[mask].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    T = _ALLGRID[:, ci[mask]].astype(np.float64)
    T = T - T.mean(1, keepdims=True)
    sd = T.std(1)
    ncc = (T @ x) / (len(x) * np.where(sd > 1e-9, sd, np.inf))
    per_id = ncc.reshape(-1, 4)
    best_rot = per_id.argmax(1)
    pk = per_id.max(1)
    order = np.argsort(-pk)
    other = [i for i in order if i != mid]
    return dict(rank_expected=int(np.where(order == mid)[0][0]) + 1, ncc_expected=float(pk[mid]),
                rotation_expected=int(best_rot[mid]), best_other_id=int(other[0]), ncc_best_other=float(pk[other[0]]),
                margin=float(pk[mid] - pk[other[0]]), n_pixels=int(mask.sum()))


def draw_rect_view(img, quad, res, mid, path, C=20, M=1.5):
    Hm, N = L.rect_H(quad, C, M)
    rp = L.warp(img, Hm, N)
    v = cv2.cvtColor(L.to_u8(rp), cv2.COLOR_GRAY2BGR)
    for i in range(7):
        a = int(round(M * C + i * C))
        cv2.line(v, (a, int(M * C)), (a, int(M * C + 6 * C)), (0, 200, 255), 1)
        cv2.line(v, (int(M * C), a), (int(M * C + 6 * C), a), (0, 200, 255), 1)
    rep = res["rep"]
    exp = np.rot90(L.code_bits(mid), rep["rotation_read"])
    v = cv2.resize(v, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for r in range(4):
        for c in range(4):
            cx, cy = int(2 * (M * C + (c + 1.5) * C)), int(2 * (M * C + (r + 1.5) * C))
            if not res["cell_ok"][r + 1, c + 1]:
                pb = res.get("partial_bits", {}).get((r, c))
                if pb is not None and abs(pb[1]) >= 0.2:
                    cv2.putText(v, str(exp[r, c]), (cx - 5, cy + 6), 0, 0.45, (0, 140, 255) if pb[0] == exp[r, c] else (0, 0, 255), 1)
                else:
                    cv2.putText(v, "x", (cx - 5, cy + 6), 0, 0.6, (255, 0, 255), 2)
                continue
            b = res["dec"]["bits"][r, c]
            cv2.putText(v, str(exp[r, c]), (cx - 5, cy + 6), 0, 0.6, (0, 200, 0) if b == exp[r, c] else (0, 0, 255), 2)
    cv2.putText(v, f"id {mid}: H={rep['hamming_expected']} n={rep['n_valid_bits']}", (3, 16), 0, 0.5, (255, 0, 0), 1)
    cv2.imwrite(path, v)


def main():
    img = L.mean_gray()
    plain = plain_quads()
    done = {}
    report = []
    # processing order: plain ones first, then the missing ones, each time the one closest to an
    # already finished sticker of its cart (so that a good neighbour shape is available)
    order = [(c, m) for c in (80, 310) for m in cart_marker_ids(c) if (c, m) in plain]
    rest = [(c, m) for c in (80, 310) for m in cart_marker_ids(c) if (c, m) not in plain]
    while rest:
        best = None
        for k in rest:
            pc = predicted_quad(*k).mean(0)
            dmin = min(np.linalg.norm(pc - predicted_quad(*j).mean(0)) for j in order if j[0] == k[0])
            if best is None or dmin < best[0]:
                best = (dmin, k)
        order.append(best[1])
        rest.remove(best[1])

    for cart, mid in order:
        p = predicted_quad(cart, mid)
        entry = dict(cart=cart, id=mid, role=marker_role(mid, cart), predicted_quad=p)
        if (cart, mid) in plain:
            q0 = plain[(cart, mid)]
            ref = None
            method = "plain ArUco detection (mean of jitter-aligned frames), re-verified"
        else:
            refs = [j for j in done if j[0] == cart and all(done[j]["side_valid"])]
            refs = refs or [j for j in done if j[0] == cart]
            ref = min(refs, key=lambda j: np.linalg.norm(done[j]["quad"].mean(0) - p.mean(0)))
            rq = done[ref]["quad"]
            pr = predicted_quad(*ref)
            ratio = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).mean() / np.linalg.norm(np.roll(pr, -1, 0) - pr, axis=1).mean()
            q0 = (rq - rq.mean(0)) * ratio + p.mean(0)
            method = "guided: predicted position + neighbour shape, code search in rectified patch"
        Hm, N = L.rect_H(q0, C_RECT, M_RECT)
        patch = L.warp(img, Hm, N)
        best, per_rot = code_search(patch, mid, C_RECT, N)
        q_rect = quad_from_search(best, C_RECT)
        q_img = L.apply_H(Hm, q_rect)
        tmpl = L.apply_H(Hm, quad_from_search(dict(best, cx=N / 2, cy=N / 2), C_RECT))
        res = analyse_quad(img, q_img, mid, template=tmpl if ref is not None else q_img)
        rep = res["rep"]
        dtest = res["template_test"]
        ext = rep["extended"]
        # confirmation: expected id at Hamming 0 on >= 8 fully visible bits, no contradiction on the
        # partially visible bits, and no other dictionary code explains the visible part better
        found = bool(rep["hamming_expected"] == 0 and rep["n_valid_bits"] >= 8 and ext["hamming_expected"] == 0
                     and dtest["ncc_expected"] >= dtest["ncc_best_other"] - 1e-9)
        r_ = rep["rotation_read"]
        qa = L.aruco_order(res["quad"], r_)
        search = dict(score=best["score"], offset_cells=[(best["cx"] - N / 2) / C_RECT, (best["cy"] - N / 2) / C_RECT],
                      scale=best["scale"], angle_deg=best["ang"], rotation=best["rot"], per_rotation_peak=per_rot)
        entry.update(method=method, reference=None if ref is None else f"{ref[0]}:{ref[1]}", found=found,
                     quad_aruco=qa, quad_read=res["quad"], rotation_read=r_, side_valid_read=res["side_valid"],
                     side_status_read=res["side_status"], occlusion=res["bands"], cells_read=res["dec"]["cells"],
                     bits_read=res["dec"]["bits"], bits_visible_read=res["cell_ok"][1:-1, 1:-1],
                     id_confirmation=dict(decoding=rep, template_test=dtest, search=search),
                     black=res["dec"]["black"], white=res["dec"]["white"], completion=res["completion"])
        if found:
            done[(cart, mid)] = dict(quad=qa, side_valid=res["side_valid"])
        report.append(entry)
        draw_rect_view(img, res["quad"], res, mid, f"{OUTDIR}/rect_{cart}_{mid}.png")
        print(f"{cart:4d} {mid:4d} {'plain ' if ref is None else 'guided'} ref={entry['reference']} found={found}"
              f" sides={''.join(s[0] for s in res['side_status'])} H_exp={rep['hamming_expected']} nbits={rep['n_valid_bits']}"
              f" best={rep['best_id']} 2nd={rep['best_other_id']}(H={rep['best_other_hamming']}) n0={rep['n_ids_at_distance0']}"
              f" | ext n={ext['n_bits']} H={ext['hamming_expected']} n0={ext['n_ids_at_distance0']} 2nd={ext['best_other_id']}(H={ext['best_other_hamming']})"
              f" | tmpl rank={dtest['rank_expected']} ncc={dtest['ncc_expected']:.3f} other={dtest['best_other_id']}"
              f"({dtest['ncc_best_other']:.3f}) asp={res['completion']['aspect_vs_template']:.3f} rej={res['completion']['rejected_sides']} | srch off={np.round(search['offset_cells'], 1)} sc={best['scale']:.2f}"
              f" rot={best['rot']} rot_read={r_} bands={[(b['side'], np.round(b['depth'], 2).tolist()) for b in res['bands']]}")
    save_json(report, f"{CACHE}/markers_guided.json")


if __name__ == "__main__":
    main()
