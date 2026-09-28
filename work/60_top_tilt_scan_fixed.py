"""What-if analysis: the TOP stickers (top plates / end boards) slightly tilted.

This is an ADDITIONAL analysis requested after the main result. The main result (results/lens_result.json) keeps the
drawing geometry (every sticker flat and horizontal), as the task demands; nothing here changes it.

The main combined estimator (50_combined.py: stickers + sides of partly hidden stickers + straight edges, block weights
by variance components) is re-run with the 3D corner points of the 8 top stickers rotated by small angles; the shelf
stickers (rows A, B, C) and all edges are unchanged. For every configuration we record the combined lens, the converged
block sigmas, sticker residuals with LS per-cart poses, sticker-only fits and mapping differences.

Cart frame (spec/cart-marker-layout.json): origin at the top front-left corner, X along the 1600 mm front, Y into the
cart (0 front .. 450 back), Z up, Z = 0 at the top plate. Top stickers: ids 0 (82.5, 60), 1 (1517.5, 60),
2 (1517.5, 390), 92 / 322 (82.5, 390). Plate "X0" = ids 0 + 92/322 (outer side = towards X = 0), plate "X1600" =
ids 1 + 2 (outer side = towards X = 1600).

Tilt families (angles in degrees):
  Y_centre : rotation about an axis parallel to cart Y through the sticker centre; + = OUTER side of the plate up.
  X_centre : rotation about an axis parallel to cart X through the sticker centre; + = BACK side (larger Y) up.
  Y_hinge  : like Y_centre, but about a hinge line parallel to Y, 55 mm from the sticker centre towards the cart middle
             (inner edge of the 110 mm sticker incl. white border); the centre rises by 55 sin(angle) mm.
  per-plate: 4 plate tilts (Y_centre convention; optionally + 4 X_centre tilts) fitted to the sticker corners.

REVIEW FIX (this file = corrected copy of 60_top_tilt_scan.py; all phases of the original are unchanged):
  * new phase "refdef": the per-plate tilts depend on how the cart reference (pose) is defined. The original quoted
    1-sigma = max(jackknife, bootstrap) (+) lens propagation and left this dependence out (only mentioned it). The
    tilts are now also fitted with (B) LS pose + free plate heights, (C) shape-only (free top-sticker positions, pose
    from the shelf stickers + top-sticker shapes), (D) shape-only relative to the cart rotation of the main combined fit,
    (E) as D with the edges-only lens and its rotation, (J) the joint fit; half the range over A-E + J is added in
    quadrature as a systematic ("sigma_total_incl_definition"). Lens variants (pp at the image centre; the plumb lens
    of the earlier shape diagnostic at f 1300/1400/1500) are listed for information.
  * the hinge sensitivity: Y_hinge also evaluated with the hinge at 45 mm (edge of the black code square) at +10/+20 deg.
  * report text: per-plate +- now include the definition systematic; the sign statement at 310/X1600 is qualified.

usage (from work/, OMP_NUM_THREADS=1):
  python3 60_top_tilt_scan_fixed.py check    -> sign-convention check + 0 deg reproduction of the main fit
  python3 60_top_tilt_scan_fixed.py scan     -> angle scans of the three families
  python3 60_top_tilt_scan_fixed.py plates   -> per-plate fitted tilts (+ alternation, + joint) (needs nothing from scan)
  python3 60_top_tilt_scan_fixed.py common   -> combined fits at the refined argmin angles of the scan
  python3 60_top_tilt_scan_fixed.py refdef   -> [fix] reference-definition systematic of the per-plate tilts + hinge 45 mm
  python3 60_top_tilt_scan_fixed.py report   -> work/cache/top_tilt_scan.md + results/top_tilt_scan.png
  python3 60_top_tilt_scan_fixed.py all      -> scan, plates, common, refdef, report
Output: work/cache/top_tilt_scan.json (sections filled by the phases), work/cache/top_tilt_scan.md,
results/top_tilt_scan.png.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import glob
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ARGV = list(sys.argv)
PHASE = ARGV[1] if len(ARGV) > 1 else "all"
NPROC = 2

# ---------------------------------------------------------------- main estimator (module header of 50_combined.py)
sys.argv = ["50_combined.py", "none"]
NS = {}
exec(compile(open("50_combined.py").read().split('if MODE == "fit":')[0], "50_combined_header", "exec"), NS)
sys.argv = ARGV

from calib import Model, Problem  # noqa: E402
from combined import Combined  # noqa: E402
from common import CACHE, CART_W, RESULTS, H, W, K_from, fold_margin, load_json, marker_center, project, save_json  # noqa: E402
from evaltools import grid, mapping_displacement, region_defs  # noqa: E402
from resultnorm import monotonic_to_corners  # noqa: E402

M = NS["M"]
PTS = NS["PTS"]
EDGES = NS["EDGES"]
SPEC = NS["SPECS"]["k1k2"]
P0 = NS["P0"]
RUN = NS["run"]
OUT = f"{CACHE}/top_tilt_scan.json"
PNG = f"{RESULTS}/top_tilt_scan.png"
MD = f"{CACHE}/top_tilt_scan.md"

HINGE_MM = 55.0
PLATES = ["80/X0", "80/X1600", "310/X0", "310/X1600"]
ANGLES = {
    "Y_centre": [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 10, -10, 12, -12, 15, -15],
    "X_centre": [2, -2, 4, -4, 6, -6, 8, -8, 10, -10],
    "Y_hinge": [2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 12, 15, 20, 30],
}
# beyond the requested range (added because the Y_hinge metrics were still falling at +10 deg)
EXTENDED = {"Y_hinge": [12, 15, 20, 30]}
FAMILY = {"Y_centre": ("Y", "centre"), "X_centre": ("X", "centre"), "Y_hinge": ("Y", "hinge")}

# reference lenses
_L = load_json(f"{RESULTS}/lens_result.json")
K_MAIN = np.array(_L["camera_matrix"], float)
D_MAIN = np.array(_L["dist_coeffs"], float)
_FIT = load_json(f"{CACHE}/combined_fit.json")["summary"]
_E = _FIT["k1k2_edges_only"]["x"]  # edges-only combined fit (blocks V + S): f, cx, cy, k1, k2
K_EDGE = K_from(_E[0], _E[0], _E[1], _E[2])
D_EDGE = np.array([_E[3], _E[4], 0.0, 0.0, 0.0])
_INIT = load_json(f"{CACHE}/initial_calib.json")
K_INIT, D_INIT = np.array(_INIT["K"]), np.array(_INIT["dist"])
UVG, _ = grid(30)
REG = {k: fn(UVG) for k, fn in region_defs().items()}


def plate_of(m):
    if m["row"] != "top":
        return None
    c = marker_center(m["id"], m["cart"])
    return f"{m['cart']}/X0" if c[0] < CART_W / 2 else f"{m['cart']}/X1600"


def outward(plate):
    """+1 if the outer side of the plate is towards larger X (X1600 end), -1 for the X0 end."""
    return -1.0 if plate.endswith("/X0") else 1.0


def tilt_matrix(ay, bx, o):
    """Rotation applied to offsets from the pivot (cart frame).
    ay [deg]: about cart Y, + = offsets towards the outer side (sign o along X) go UP: (1,0,0)*o -> (o cos a, 0, sin a).
    bx [deg]: about cart X, + = offsets towards the back (larger Y) go UP: (0,1,0) -> (0, cos b, sin b).
    Applied as R = Rx(bx) @ Ry(ay) (Y tilt first)."""
    a, b = np.deg2rad(ay), np.deg2rad(bx)
    Ry = np.array([[np.cos(a), 0.0, -o * np.sin(a)], [0.0, 1.0, 0.0], [o * np.sin(a), 0.0, np.cos(a)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def pivot_of(mid, cart, plate, pivot):
    c = marker_center(mid, cart).astype(float)
    if pivot == "hinge":
        c[0] -= outward(plate) * HINGE_MM  # towards the cart middle
    return c


def tilt_corners(X, mid, cart, plate, ay, bx, pivot):
    piv = pivot_of(mid, cart, plate, pivot)
    R = tilt_matrix(ay, bx, outward(plate))
    return piv + (np.asarray(X, float).reshape(-1, 3) - piv) @ R.T


def family_tilts(family, angle):
    ax, pivot = FAMILY[family]
    return {p: (float(angle) if ax == "Y" else 0.0, float(angle) if ax == "X" else 0.0, pivot) for p in PLATES}


def tilted_pts(plate_tilts):
    """Copy of PTS with the 3D corners of the top stickers tilted. plate_tilts: {plate: (ay, bx, pivot)}."""
    out = []
    for p in PTS:
        m = M[p["mi"]]
        pl = plate_of(m)
        if pl is None or pl not in plate_tilts:
            out.append(dict(p))
            continue
        ay, bx, pivot = plate_tilts[pl]
        X = tilt_corners(p["X"], m["id"], m["cart"], pl, ay, bx, pivot)[0]
        out.append(dict(p, X=X))
    return out


# ---------------------------------------------------------------- per-point bookkeeping (fixed order = PTS order)
LAB_CART = np.array([p["cart"] for p in PTS])
LAB_ROW = np.array([M[p["mi"]]["row"] for p in PTS])
LAB_PLATE = np.array([plate_of(M[p["mi"]]) or "" for p in PTS])
LAB_STK = np.array([f"{p['cart']}:{M[p['mi']]['id']}" for p in PTS])
LAB_MI = np.array([p["mi"] for p in PTS])
UV = np.array([p["uv"] for p in PTS], float)
XBASE = np.array([p["X"] for p in PTS], float)
PIDX = np.array([PLATES.index(pl) if pl else -1 for pl in LAB_PLATE])
CEN = np.array([marker_center(M[p["mi"]]["id"], p["cart"]) for p in PTS], float)
OSGN = np.array([outward(PLATES[i]) if i >= 0 else 0.0 for i in PIDX])


def rstat(e, sel):
    if not np.any(sel):
        return None
    return dict(rms=float(np.sqrt(np.mean(e[sel] ** 2))), max=float(e[sel].max()), n=int(sel.sum()))


def ls_poses(K, d, X, poses0=None):
    """One least-squares 6-DoF pose per cart with the lens fixed; returns residual vectors (N x 2) and poses."""
    res = np.zeros((len(X), 2))
    poses = {}
    for c in (80, 310):
        s = LAB_CART == c
        p0 = P0[c] if poses0 is None else poses0[c]
        r = least_squares(lambda pz: (project(X[s], K, d, rvec=pz[:3], tvec=pz[3:]) - UV[s]).ravel(), p0)
        res[s] = r.fun.reshape(-1, 2)
        poses[c] = r.x
    return res, poses


def sticker_stats(K, d, X, detail=True):
    res, poses = ls_poses(K, d, X)
    e = np.hypot(res[:, 0], res[:, 1])
    out = dict(overall=rstat(e, np.ones(len(e), bool)), top=rstat(e, LAB_ROW == "top"), shelf=rstat(e, LAB_ROW != "top"))
    for r in ("A", "B", "C"):
        out[f"row_{r}"] = rstat(e, LAB_ROW == r)
    out["plates"] = {pl: rstat(e, LAB_PLATE == pl) for pl in PLATES}
    if detail:
        out["stickers"] = {k: rstat(e, LAB_STK == k) for k in dict.fromkeys(LAB_STK)}
        out["poses"] = {str(c): v.tolist() for c, v in poses.items()}
    return out


def mapdiff(K0, d0, K1, d1):
    """Median / max |displacement| of lens 1 relative to lens 0 per region (rotation-compensated)."""
    disp = mapping_displacement(K0, d0, K1, np.asarray(d1, float)[:5], UVG, compensate=True)
    n = np.hypot(disp[:, 0], disp[:, 1])
    return {k: dict(median=float(np.nanmedian(n[m])), max=float(np.nanmax(n[m]))) for k, m in REG.items()}


def lens_dict(K, d):
    d = np.asarray(d, float)
    fm = fold_margin(K, d[:5])
    return dict(f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]), k1=float(d[0]), k2=float(d[1]),
                fold_margin=float(min(fm, 9.99)), min_radial_slope_to_corners=float(monotonic_to_corners(K, d[:5])),
                folds_inside_image=bool(fm <= 0 or monotonic_to_corners(K, d[:5]) < 0.02))


def sticker_only(X, pp, barrier=False):
    """Sticker-only bundle adjustment (drawing / tilted geometry), f single, k1 k2, pp fixed at the image centre or free.
    Multi-start (initial calibration lens with f x 1.0 / 0.85 / 1.15 and the main lens); lowest cost wins."""
    pts = [dict(cart=int(c), X=x, uv=u) for c, x, u in zip(LAB_CART, X, UV)]
    m = Model(dict(f="single", pp=pp, dist=["k1", "k2"]))
    pr = Problem(m, pts, carts=[80, 310])
    pr.fold_barrier = barrier
    starts = []
    for fs in (1.0, 0.85, 1.15):
        x = np.r_[m.pack(K_INIT, D_INIT), P0[80], P0[310]]
        x[0] *= fs
        starts.append(x)
    _, pm = ls_poses(K_MAIN, D_MAIN, X)
    Km = K_MAIN.copy()
    if pp == "fixed":
        Km[0, 2], Km[1, 2] = m.pp0
    starts.append(np.r_[m.pack(Km, D_MAIN), pm[80], pm[310]])
    best = None
    for x0 in starts:
        try:
            r = pr.solve(x0)
        except Exception:  # noqa
            continue
        if best is None or r.cost < best.cost:
            best = r
    K, dist, _ = pr.split(best.x)
    res = pr.predict(best.x) - pr.uv
    e = np.hypot(res[:, 0], res[:, 1])
    out = lens_dict(K, dist[:5])
    out.update(rms=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()), top_rms=rstat(e, LAB_ROW == "top")["rms"],
               shelf_rms=rstat(e, LAB_ROW != "top")["rms"], barrier=barrier, pp=pp)
    out["at_fold_barrier"] = bool(barrier and out["fold_margin"] < 0.035)
    out["mapping_vs_main"] = mapdiff(K_MAIN, D_MAIN, K, dist[:5])
    out["mapping_vs_edges_only"] = mapdiff(K_EDGE, D_EDGE, K, dist[:5])
    return out


def evaluate(cfg):
    """Full record of one configuration. cfg: dict(name, family, angle, plate_tilts)."""
    t0 = time.time()
    pts = tilted_pts(cfg["plate_tilts"])
    X = np.array([p["X"] for p in pts], float)
    cb, r = RUN(SPEC, pts, EDGES)
    K, d, poses, wv = cb.unpack(r.x)
    d5 = np.asarray(d, float)[:5]
    comb = lens_dict(K, d5)
    comb.update(sig={k: float(v) for k, v in cb.sig.items()}, block_rms={k: v for k, v in cb.block_rms(r.x).items()},
                vc_iters=int(getattr(cb, "vc_iters", -1)), cost=float(r.cost), nfev=int(r.nfev),
                poses={str(c): poses[i].tolist() for i, c in enumerate(cb.carts)})
    rec = dict(name=cfg["name"], family=cfg.get("family"), angle=cfg.get("angle"),
               plate_tilts={k: list(v) for k, v in cfg["plate_tilts"].items()}, combined=comb)
    rec["stickers_combined_lens"] = sticker_stats(K, d5, X)
    rec["stickers_main_lens"] = sticker_stats(K_MAIN, D_MAIN, X, detail=False)
    rec["stickers_edges_only_lens"] = sticker_stats(K_EDGE, D_EDGE, X, detail=False)
    rec["sticker_only"] = dict(pp_fixed=sticker_only(X, "fixed"), pp_free=sticker_only(X, "free"),
                               pp_free_fold_barrier=sticker_only(X, "free", barrier=True))
    rec["mapping_vs_main"] = mapdiff(K_MAIN, D_MAIN, K, d5)
    rec["mapping_vs_edges_only"] = mapdiff(K_EDGE, D_EDGE, K, d5)
    rec["seconds"] = time.time() - t0
    s = rec["stickers_combined_lens"]
    so = rec["sticker_only"]
    print(f"[{cfg['name']:>22s}] f {comb['f']:7.1f} cx {comb['cx']:6.1f} cy {comb['cy']:6.1f} k1 {comb['k1']:+.4f} k2 {comb['k2']:+.4f} "
          f"sigM {comb['sig']['M']:5.2f} | stk rms {s['overall']['rms']:5.2f} top {s['top']['rms']:5.2f} shelf {s['shelf']['rms']:5.2f} | "
          f"only ppfix f {so['pp_fixed']['f']:6.0f} rms {so['pp_fixed']['rms']:4.2f}; ppfree f {so['pp_free']['f']:6.0f} rms {so['pp_free']['rms']:4.2f} "
          f"fold {so['pp_free']['fold_margin']:+.2f} | dmap band {rec['mapping_vs_main']['cart_band']['median']:5.2f} ({rec['seconds']:.0f}s)", flush=True)
    return rec


# ---------------------------------------------------------------- per-plate tilt fit (lens fixed, LS poses per cart)
def tilted_all(t, with_x):
    X = XBASE.copy()
    for p in range(4):
        s = PIDX == p
        R = tilt_matrix(t[p], t[4 + p] if with_x else 0.0, outward(PLATES[p]))
        X[s] = CEN[s] + (X[s] - CEN[s]) @ R.T
    return X


def fit_tilts(K, d, with_x, idx=None, q0=None):
    idx = np.arange(len(PTS)) if idx is None else np.asarray(idx)
    nt = 8 if with_x else 4
    cart = LAB_CART[idx]
    uv = UV[idx]

    def res(q):
        X = tilted_all(q[:nt], with_x)[idx]
        out = np.zeros((len(idx), 2))
        for ci, c in enumerate((80, 310)):
            s = cart == c
            pz = q[nt + 6 * ci: nt + 6 * ci + 6]
            out[s] = project(X[s], K, d, rvec=pz[:3], tvec=pz[3:]) - uv[s]
        return out.ravel()

    if q0 is None:
        _, pz = ls_poses(K, d, XBASE)
        q0 = np.r_[np.zeros(nt), pz[80], pz[310]]
    r = least_squares(res, q0, method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12)
    return r, nt


def tilt_names(with_x):
    return [f"ay_{p}" for p in PLATES] + ([f"bx_{p}" for p in PLATES] if with_x else [])


def plate_tilt_analysis(K, d, with_x, nboot=300, lens_prop=True, seed=4242):
    """Tilts with the lens (K, d) fixed; uncertainty: formal, delete-one-sticker jackknife, stratified cluster
    bootstrap (top stickers resampled within their plate, shelf stickers within their cart), lens propagation
    (refit with each bootstrap replicate lens of the main combined estimator)."""
    rng = np.random.default_rng(seed)
    r, nt = fit_tilts(K, d, with_x)
    t = r.x[:nt]
    e = r.fun.reshape(-1, 2)
    en = np.hypot(e[:, 0], e[:, 1])
    dof = max(len(r.fun) - len(r.x), 1)
    C = np.linalg.pinv(r.jac.T @ r.jac) * (2 * r.cost / dof)
    s_formal = np.sqrt(np.diag(C))[:nt]
    mis = sorted(set(LAB_MI.tolist()))
    jk = []
    for mi in mis:
        idx = np.nonzero(LAB_MI != mi)[0]
        rj, _ = fit_tilts(K, d, with_x, idx=idx, q0=r.x.copy())
        jk.append(rj.x[:nt])
    jk = np.array(jk)
    n = len(jk)
    s_jk = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(0)) ** 2, 0))
    # stratified cluster bootstrap
    strata = {}
    for mi in mis:
        m = M[mi]
        key = plate_of(m) or f"shelf{m['cart']}"
        strata.setdefault(key, []).append(mi)
    bt = []
    for b in range(nboot):
        idx = []
        for key, lst in strata.items():
            for mi in rng.choice(lst, len(lst), replace=True):
                idx += np.nonzero(LAB_MI == mi)[0].tolist()
        rb, _ = fit_tilts(K, d, with_x, idx=np.array(idx), q0=r.x.copy())
        bt.append(rb.x[:nt])
    bt = np.array(bt)
    s_bt = 0.5 * (np.percentile(bt, 84, 0) - np.percentile(bt, 16, 0))
    out = dict(names=tilt_names(with_x), tilt_deg=t.tolist(), sigma_formal=s_formal.tolist(), sigma_jackknife_sticker=s_jk.tolist(),
               sigma_bootstrap_stratified=s_bt.tolist(), bootstrap_n=len(bt), rms_px=float(np.sqrt(np.mean(en ** 2))),
               top_rms_px=float(np.sqrt(np.mean(en[LAB_ROW == 'top'] ** 2))), shelf_rms_px=float(np.sqrt(np.mean(en[LAB_ROW != 'top'] ** 2))),
               plates_rms_px={pl: float(np.sqrt(np.mean(en[LAB_PLATE == pl] ** 2))) for pl in PLATES},
               jackknife_values={f"{M[mi]['cart']}:{M[mi]['id']}": v.tolist() for mi, v in zip(mis, jk)},
               lens=lens_dict(K, d))
    if lens_prop:
        boots = []
        for fn in sorted(glob.glob(f"{CACHE}/combined_boot_*.json")):
            boots += load_json(fn)["boot"]
        lt = []
        for xb in boots:
            Kb = K_from(xb[0], xb[0], xb[1], xb[2])
            db = np.array([xb[3], xb[4], 0, 0, 0])
            rl, _ = fit_tilts(Kb, db, with_x, q0=r.x.copy())
            lt.append(rl.x[:nt])
        lt = np.array(lt)
        s_l = 0.5 * (np.percentile(lt, 84, 0) - np.percentile(lt, 16, 0))
        # slope of each tilt vs f over the lens replicates (how the tilt follows the focal length)
        fb = np.array([xb[0] for xb in boots])
        slope = [float(np.polyfit(fb, lt[:, j], 1)[0] * 100) for j in range(nt)]
        out.update(sigma_lens_propagated=s_l.tolist(), lens_replicates_n=len(lt), dtilt_per_100px_f=slope,
                   sigma_total=np.sqrt(np.maximum(s_jk, s_bt) ** 2 + s_l ** 2).tolist(),
                   sigma_total_rule="max(jackknife, stratified bootstrap) (+) lens propagation")
    return out


# ---------------------------------------------------------------- [fix] reference-definition systematic of the tilts
TOPM = sorted({p["mi"] for p in PTS if plate_of(M[p["mi"]])})
SIDX = np.array([TOPM.index(p["mi"]) if p["mi"] in TOPM else -1 for p in PTS])
ISTOP = PIDX >= 0


def tilted_general(t, with_x, dz=None, dpos=None):
    """tilted_all + optional height offset per plate (dz, 4) and free position offset per top sticker (dpos, 8 x 3)."""
    X = tilted_all(t, with_x)
    if dz is not None:
        X[ISTOP, 2] += np.asarray(dz)[PIDX[ISTOP]]
    if dpos is not None:
        X[ISTOP] += np.asarray(dpos)[SIDX[ISTOP]]
    return X


def ls_pose_sel(K, d, X, sel):
    P = {}
    for c in (80, 310):
        s = (LAB_CART == c) & sel
        P[c] = least_squares(lambda pz: (project(X[s], K, d, rvec=pz[:3], tvec=pz[3:]) - UV[s]).ravel(), P0[c]).x
    return P


def fit_tilts_def(defn, K, d, with_x, rot=None):
    """Plate tilts (lens fixed) under other definitions of the cart reference:
    'B' LS pose per cart + a free height offset per plate (diagnostic only: stops the top-vs-shelf height mismatch
        from bending the pose; heights are NOT free in any lens fit);
    'C' shape-only: every top sticker gets a free 3D position (its drawing position is not used), the pose is the LS
        pose from the shelf stickers (drawing) + the top-sticker shapes;
    'D' shape-only relative to a GIVEN cart rotation `rot` (edge-constrained rotation of a combined fit): only the
        32 top-sticker corners, free sticker positions (translation of the cart irrelevant)."""
    nt = 8 if with_x else 4
    Pall = ls_pose_sel(K, d, XBASE, np.ones(len(PTS), bool))
    if defn == "B":
        q0 = np.r_[np.zeros(nt), np.zeros(4), Pall[80], Pall[310]]
        sel = np.ones(len(PTS), bool)

        def unpack(q):
            return q[:nt], q[nt:nt + 4], None, {80: q[nt + 4:nt + 10], 310: q[nt + 10:nt + 16]}
    elif defn == "C":
        Ps = ls_pose_sel(K, d, XBASE, ~ISTOP)
        q0 = np.r_[np.zeros(nt), np.zeros(24), Ps[80], Ps[310]]
        sel = np.ones(len(PTS), bool)

        def unpack(q):
            return q[:nt], None, q[nt:nt + 24].reshape(8, 3), {80: q[nt + 24:nt + 30], 310: q[nt + 30:nt + 36]}
    else:
        poses = {c: np.r_[np.asarray(rot[c])[:3], Pall[c][3:]] for c in (80, 310)}
        q0 = np.r_[np.zeros(nt), np.zeros(24)]
        sel = ISTOP

        def unpack(q):
            return q[:nt], None, q[nt:nt + 24].reshape(8, 3), poses

    def res(q):
        t, dz, dp, poses_ = unpack(q)
        X = tilted_general(t, with_x, dz, dp)
        out = np.zeros((len(PTS), 2))
        for c in (80, 310):
            s = LAB_CART == c
            out[s] = project(X[s], K, d, rvec=poses_[c][:3], tvec=poses_[c][3:]) - UV[s]
        return out[sel].ravel()

    r = least_squares(res, q0, method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=20000)
    t, dz, _, _ = unpack(r.x)
    e = r.fun.reshape(-1, 2)
    return dict(tilt_deg=t.tolist(), rms_px=float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), dz_mm=None if dz is None else dz.tolist(),
                n_corners=int(sel.sum()))


def plate_cfg(t, with_x, name):
    nt = 8 if with_x else 4
    return dict(name=name, family="per_plate_YX" if with_x else "per_plate_Y", angle=None,
                plate_tilts={p: (float(t[i]), float(t[4 + i]) if with_x else 0.0, "centre") for i, p in enumerate(PLATES)})


def plate_chain(with_x, max_iter=4, tol_f=1.0):
    """Tilts fitted with the main lens fixed -> combined fit; then alternate (tilts refitted with the new combined lens)
    until |delta f| < tol_f."""
    tag = "YX" if with_x else "Y"
    K, d = K_MAIN, D_MAIN
    steps = []
    fprev = None
    for it in range(max_iter):
        ta = plate_tilt_analysis(K, d, with_x, nboot=300 if it == 0 else 100, lens_prop=(it == 0))
        rec = evaluate(plate_cfg(ta["tilt_deg"], with_x, f"per_plate_{tag}_it{it}"))
        steps.append(dict(iteration=it, tilt_fit=ta, combined_record=rec))
        c = rec["combined"]
        K = K_from(c["f"], c["f"], c["cx"], c["cy"])
        d = np.array([c["k1"], c["k2"], 0, 0, 0])
        print(f"   chain {tag} it {it}: tilts {np.round(ta['tilt_deg'], 2)} -> f {c['f']:.1f}", flush=True)
        if fprev is not None and abs(c["f"] - fprev) < tol_f:
            break
        fprev = c["f"]
    return dict(kind=tag, steps=steps)


class CombinedTilt(Combined):
    """Combined estimator with the per-plate tilts of the top stickers as FREE parameters (appended to x)."""

    def __init__(self, pts, edges, spec, with_x=False, **kw):
        super().__init__(pts, edges, spec, **kw)
        self.with_x = with_x
        self.nt = 8 if with_x else 4
        self.nbase = len(self.names)
        self.X0m = self.mX.copy()
        self.pi = np.array([PLATES.index(plate_of(M[p["mi"]])) if plate_of(M[p["mi"]]) else -1 for p in self.mp])
        self.cen = np.array([marker_center(M[p["mi"]]["id"], p["cart"]) for p in self.mp], float)

    def set_tilts(self, t):
        X = self.X0m.copy()
        for p in range(4):
            s = self.pi == p
            R = tilt_matrix(t[p], t[4 + p] if self.with_x else 0.0, outward(PLATES[p]))
            X[s] = self.cen[s] + (X[s] - self.cen[s]) @ R.T
        self.mX = X

    def blocks(self, x):
        if len(x) == self.nbase:
            self.set_tilts(np.zeros(self.nt))
            return super().blocks(x)
        self.set_tilts(x[self.nbase:])
        return super().blocks(x[: self.nbase])


def joint_fit(with_x):
    tag = "YX" if with_x else "Y"
    t0 = time.time()
    cb = CombinedTilt(PTS, EDGES, SPEC, with_x=with_x)
    K0 = K_from(1420.0, 1420.0, (W - 1) / 2, (H - 1) / 2)
    x0 = np.r_[cb.x0(K0, np.array([-0.33, 0.09, 0, 0, 0]), P0), np.zeros(cb.nt)]
    r = cb.solve(x0)
    t = r.x[cb.nbase:]
    K, d, _, _ = cb.unpack(r.x)
    J = r.jac
    dof = max(len(r.fun) - len(r.x), 1)
    Cv = np.linalg.pinv(J.T @ J) * (2 * r.cost / dof)
    st = np.sqrt(np.diag(Cv))[cb.nbase:]
    # full record with these tilts fixed (same evaluation as every other configuration)
    rec = evaluate(plate_cfg(t, with_x, f"joint_{tag}"))
    out = dict(kind=tag, tilt_names=tilt_names(with_x), tilt_deg=t.tolist(), tilt_sigma_formal=st.tolist(),
               joint_lens=lens_dict(K, np.asarray(d)[:5]), joint_sig={k: float(v) for k, v in cb.sig.items()},
               joint_block_rms=cb.block_rms(r.x), vc_iters=int(getattr(cb, "vc_iters", -1)), seconds=time.time() - t0,
               record_with_fixed_tilts=rec)
    print(f"   joint {tag}: tilts {np.round(t, 2)} f {K[0, 0]:.1f} sigM {cb.sig['M']:.2f}", flush=True)
    return out


# ---------------------------------------------------------------- IO helpers
def load_out():
    return load_json(OUT) if os.path.exists(OUT) else {}


def store(key, val):
    o = load_out()
    o[key] = val
    o["definitions"] = DEFINITIONS
    o["reference"] = REFERENCE
    save_json(o, OUT)


DEFINITIONS = {
    "cart_frame": "spec/cart-marker-layout.json: origin top front-left corner, X along the 1600 mm front, Y into the cart (0 front, 450 back), Z up, Z = 0 at the top plate",
    "top_stickers": "ids 0 (82.5,60,0), 1 (1517.5,60,0), 2 (1517.5,390,0), 92/322 (82.5,390,0); plate X0 = ids 0 + 92/322, plate X1600 = ids 1 + 2; shelf stickers unchanged",
    "Y_centre": "rotation about an axis parallel to cart Y through the sticker centre; angle > 0: the OUTER side of the plate goes UP (towards X=0 at the X0 plate, towards X=1600 at the X1600 plate); centres unchanged",
    "X_centre": "rotation about an axis parallel to cart X through the sticker centre; angle > 0: the BACK side (larger Y) goes UP; centres unchanged",
    "Y_hinge": f"as Y_centre, but about a hinge line parallel to Y at {HINGE_MM} mm from the sticker centre towards the cart middle (edge of the 110 mm sticker incl. white border); angle > 0: outer side up, the centre rises by {HINGE_MM} sin(angle) mm",
    "per_plate": "one angle per plate (80/X0, 80/X1600, 310/X0, 310/X1600) in the Y_centre convention (ay_*), optionally + one X_centre angle per plate (bx_*); both stickers of a plate share the angles; composite rotation R = Rx(bx) Ry(ay)",
    "per_plate_uncertainty": "[review fix] 1-sigma of a plate tilt = [max(delete-one-sticker jackknife, stratified cluster bootstrap) (+) lens propagation over the 100 bootstrap lenses] (+) half the range over 6 definitions of the cart reference (A LS pose all corners, B + free plate heights, C/D/E shape-only with free top-sticker positions, J joint fit); key refdef.<kind>.sigma_total_incl_definition",
    "tilt_matrix": "offset (dx,dy,dz) from the pivot -> R (dx,dy,dz); Ry: (o,0,0) -> (o cos a, 0, sin a) with o = -1 (X0 plate) / +1 (X1600 plate); Rx: (0,1,0) -> (0, cos b, sin b)",
    "combined": "50_combined.py main estimator (k1k2, fx=fy, pp free), same start and variance-component block weighting; only the top-sticker 3D corners differ",
    "sig": "converged block sigmas of the combined fit [px]: M sticker corners (per coordinate), L traced sides of partly hidden stickers, V vanishing-point groups, S straightness",
    "sticker_rms": "radial corner residuals [px] of all 62 valid corners with the given lens FIXED and one least-squares 6-DoF pose per cart (same definition as lens_result.json rms_details); 'top' = 32 corners of the 8 top stickers, 'shelf' = 30 corners of rows A-C",
    "sticker_only": "bundle adjustment of the 62 corners alone (f single, k1, k2; pp fixed at the image centre (959.5, 539.5) or free; one pose per cart); no fold barrier unless 'fold_barrier'; fold_margin < 0 or min_radial_slope < 0.02 = radial map folds inside the image (not a usable lens)",
    "mapping": "rotation-compensated displacement of the lens relative to the reference (main result or edges-only lens) on a 30 px grid, median / max |d| per region (evaltools.region_defs: centre, cart_band, corners)",
}
REFERENCE = {
    "main": dict(lens_dict(K_MAIN, D_MAIN), sigma=dict(f=46.6, cx=18.8, cy=29.1, k1=0.0209, k2=0.0222),
                 mapping_1sigma=dict(centre=4.03, cart_band=15.28, corners=21.56), sticker_rms=7.886),
    "edges_only": dict(lens_dict(K_EDGE, D_EDGE), source="work/cache/combined_fit.json summary k1k2_edges_only (V + S blocks only)"),
}


# ---------------------------------------------------------------- phases
def phase_check():
    print("=== sign-convention check (Y_centre, +10 deg) ===")
    lines = []
    for mi, m in enumerate(M):
        if m["row"] != "top" or m["cart"] != 80 or m["id"] not in (0, 1):
            continue
        pl = plate_of(m)
        c = marker_center(m["id"], m["cart"])
        Xt = tilt_corners(m["corners_3d"], m["id"], m["cart"], pl, 10.0, 0.0, "centre")
        for j in range(4):
            X0 = m["corners_3d"][j]
            side = "outer" if (X0[0] - c[0]) * outward(pl) > 0 else "inner"
            s = (f"sticker {m['cart']}:{m['id']} plate {pl} corner {j}: X {X0[0]:7.1f} -> {Xt[j, 0]:7.2f}, Y {X0[1]:5.1f}, "
                 f"Z {X0[2]:.1f} -> {Xt[j, 2]:+6.2f} ({side} side)")
            print(s)
            lines.append(s)
            assert (Xt[j, 2] > 0) == (side == "outer"), "sign convention violated"
        assert np.allclose(Xt.mean(0), c), "centre moved"
    # X_centre: back up
    m = next(mm for mm in M if mm["row"] == "top" and mm["cart"] == 80 and mm["id"] == 0)
    Xt = tilt_corners(m["corners_3d"], 0, 80, "80/X0", 0.0, 10.0, "centre")
    for j in range(4):
        up = Xt[j, 2] > 0
        s = f"X_centre +10: sticker 80:0 corner {j}: Y {m['corners_3d'][j, 1]:6.1f} (centre 60) -> Z {Xt[j, 2]:+6.2f}"
        print(s)
        lines.append(s)
        assert up == (m["corners_3d"][j, 1] > 60.0)
    # Y_hinge: centre rises by 55 sin(a)
    for sid, pl in ((0, "80/X0"), (1, "80/X1600")):
        mm = next(q for q in M if q["row"] == "top" and q["cart"] == 80 and q["id"] == sid)
        Xt = tilt_corners(mm["corners_3d"], sid, 80, pl, 10.0, 0.0, "hinge")
        s = (f"Y_hinge +10: sticker 80:{sid} ({pl}) centre Z {Xt.mean(0)[2]:+.3f} mm (expected {HINGE_MM * np.sin(np.deg2rad(10)):+.3f}), "
             f"centre X {Xt.mean(0)[0]:.2f} (drawing {marker_center(sid, 80)[0]})")
        print(s)
        lines.append(s)
        assert abs(Xt.mean(0)[2] - HINGE_MM * np.sin(np.deg2rad(10))) < 1e-9
    # image check: at +10 deg, where do the outer corners move in the image (main lens, LS poses of the flat geometry)?
    _, pz = ls_poses(K_MAIN, D_MAIN, XBASE)
    Xt = tilted_all(np.full(4, 10.0), False)
    for c in (80, 310):
        s = LAB_CART == c
        u0 = project(XBASE[s], K_MAIN, D_MAIN, rvec=pz[c][:3], tvec=pz[c][3:])
        u1 = project(Xt[s], K_MAIN, D_MAIN, rvec=pz[c][:3], tvec=pz[c][3:])
        dd = np.hypot(*(u1 - u0).T)
        msg = f"cart {c}: +10 deg Y_centre moves the top-sticker corners in the image by {dd[LAB_ROW[s] == 'top'].mean():.2f} px on average (max {dd.max():.2f})"
        print(msg)
        lines.append(msg)
    print("sign convention OK")
    print("=== 0 deg reproduction of the main fit ===")
    rec = evaluate(dict(name="Y_centre_+0", family="Y_centre", angle=0.0, plate_tilts=family_tilts("Y_centre", 0.0)))
    f0 = rec["combined"]["f"]
    ok = abs(f0 - K_MAIN[0, 0]) < 0.5
    print(f"0 deg: f = {f0:.3f} (main {K_MAIN[0, 0]:.3f}) -> {'OK' if ok else 'MISMATCH'}; sticker rms (combined lens) "
          f"{rec['stickers_combined_lens']['overall']['rms']:.3f} (main 7.886)")
    assert ok
    store("verification", dict(sign_check=lines, zero_deg_f=f0, main_f=float(K_MAIN[0, 0]), zero_deg_ok=bool(ok),
                               zero_deg_cx=rec["combined"]["cx"], zero_deg_cy=rec["combined"]["cy"],
                               zero_deg_sticker_rms=rec["stickers_combined_lens"]["overall"]["rms"]))
    o = load_out()
    scan = o.get("scan", {})
    scan.setdefault("Y_centre", {})["0"] = rec
    store("scan", scan)


def pool():
    return mp.get_context("fork").Pool(NPROC)


def phase_scan():
    o = load_out()
    scan = o.get("scan", {})
    cfgs = []
    for fam, angs in ANGLES.items():
        for a in angs:
            if str(a) in scan.get(fam, {}):
                continue
            cfgs.append(dict(name=f"{fam}_{a:+d}", family=fam, angle=float(a), plate_tilts=family_tilts(fam, a)))
    if "0" not in scan.get("Y_centre", {}):
        cfgs.insert(0, dict(name="Y_centre_+0", family="Y_centre", angle=0.0, plate_tilts=family_tilts("Y_centre", 0.0)))
    print(f"scan: {len(cfgs)} configurations with {NPROC} workers", flush=True)
    with pool() as pl:
        for rec in pl.imap_unordered(evaluate, cfgs):
            o = load_out()
            scan = o.get("scan", {})
            scan.setdefault(rec["family"], {})[str(int(rec["angle"]))] = rec
            store("scan", scan)


def phase_plates():
    with pool() as pl:
        chains = pl.map(plate_chain, [False, True])
    store("per_plate", {c["kind"]: c for c in chains})
    with pool() as pl:
        joints = pl.map(joint_fit, [False, True])
    store("joint", {j["kind"]: j for j in joints})


def metric_values(recs, key):
    f = {
        "M_sigma": lambda r: r["combined"]["sig"]["M"],
        "sticker_rms_combined_lens": lambda r: r["stickers_combined_lens"]["overall"]["rms"],
        "top_rms_combined_lens": lambda r: r["stickers_combined_lens"]["top"]["rms"],
        "sticker_rms_main_lens": lambda r: r["stickers_main_lens"]["overall"]["rms"],
        "sticker_rms_edges_only_lens": lambda r: r["stickers_edges_only_lens"]["overall"]["rms"],
        "sticker_only_ppfixed_rms": lambda r: r["sticker_only"]["pp_fixed"]["rms"],
        "sticker_only_ppfree_rms": lambda r: r["sticker_only"]["pp_free"]["rms"],
        "sticker_only_ppfree_barrier_rms": lambda r: r["sticker_only"]["pp_free_fold_barrier"]["rms"],
    }[key]
    return np.array([f(r) for r in recs])


METRICS = ["M_sigma", "sticker_rms_combined_lens", "top_rms_combined_lens", "sticker_rms_main_lens", "sticker_rms_edges_only_lens",
           "sticker_only_ppfixed_rms", "sticker_only_ppfree_rms", "sticker_only_ppfree_barrier_rms"]


def argmins(scan):
    out = {}
    for fam in ANGLES:
        a, recs = fam_rows(scan, fam)  # includes the shared 0 deg record
        if len(recs) < 3:
            continue
        angs = np.array(a)
        out[fam] = {}
        for key in METRICS:
            v = metric_values(recs, key)
            i = int(np.argmin(v))
            ref = None
            edge = i in (0, len(angs) - 1)
            if not edge:
                c = np.polyfit(angs[i - 1:i + 2], v[i - 1:i + 2], 2)
                ref = float(np.clip(-c[1] / (2 * c[0]), angs[i - 1], angs[i + 1])) if c[0] > 0 else float(angs[i])
            out[fam][key] = dict(grid_argmin=float(angs[i]), value=float(v[i]), refined_argmin=ref, at_range_edge=bool(edge),
                                 value_at_0=float(v[np.argmin(np.abs(angs))]))
    return out


def phase_common():
    o = load_out()
    am = argmins(o["scan"])
    cfgs = []
    for fam, dd in am.items():
        cand = []
        for key in ("M_sigma", "sticker_rms_combined_lens", "top_rms_combined_lens"):
            a = dd[key]["refined_argmin"]
            if a is None or dd[key]["at_range_edge"]:
                continue
            a = round(a, 2)
            if any(abs(a - b) < 0.25 for b, _ in cand) or any(abs(a - float(x)) < 0.25 for x in o["scan"][fam]):
                continue
            cand.append((a, key))
        for a, key in cand:
            cfgs.append(dict(name=f"{fam}_common_{a:+.2f}", family=fam, angle=a, plate_tilts=family_tilts(fam, a), chosen_by=key))
    print("common-angle fits:", [(c["name"], c["chosen_by"]) for c in cfgs], flush=True)
    recs = []
    if cfgs:
        with pool() as pl:
            recs = pl.map(evaluate, cfgs)
    for c, r in zip(cfgs, recs):
        r["chosen_by"] = c["chosen_by"]
    store("argmins", am)
    store("common_angle", recs)


def phase_refdef():
    """[fix] Reference-definition systematic of the per-plate tilts (needs 'plates') + hinge at 45 mm."""
    global HINGE_MM
    o = load_out()
    fa = load_json(f"{CACHE}/combined_fit.json")
    cbm = Combined(PTS, EDGES, SPEC)
    Km, dm, pm, _ = cbm.unpack(np.array(fa["x_main"]))
    assert np.allclose(Km, K_MAIN) and np.allclose(np.asarray(dm)[:5], D_MAIN), "combined_fit.json x_main != lens_result.json"
    rot_main = {c: pm[i] for i, c in enumerate(cbm.carts)}
    cbe, re_ = RUN(SPEC, PTS, EDGES, use=("V", "S"))
    Ke, de, pe, _ = cbe.unpack(re_.x)
    de = np.asarray(de, float)[:5]
    rot_edge = {c: pe[i] for i, c in enumerate(cbe.carts)}
    lens_var = {"pp at the image centre (f main)": (K_from(K_MAIN[0, 0], K_MAIN[0, 0], (W - 1) / 2, (H - 1) / 2), D_MAIN)}
    for f in (1300.0, 1400.0, 1500.0):  # plumb lens of geometry_diagnosis sect. 5 (k in pixel units, pp at the centre)
        s = f / 1300.0
        lens_var[f"shape-diagnostic lens f {f:.0f} (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre)"] = (
            K_from(f, f, (W - 1) / 2, (H - 1) / 2), np.array([-0.2920 * s ** 2, 0.0716 * s ** 4, 0.0, 0.0, 0.0]))
    out = dict(edges_only_lens=dict(f=float(Ke[0, 0]), cx=float(Ke[0, 2]), cy=float(Ke[1, 2]), k1=float(de[0]), k2=float(de[1])))
    for kind, with_x in (("Y", False), ("YX", True)):
        A = o["per_plate"][kind]["steps"][0]["tilt_fit"]
        defs = {
            "A: LS pose per cart from all 62 corners (reported value)": dict(tilt_deg=A["tilt_deg"], rms_px=A["rms_px"], n_corners=len(PTS)),
            "B: A + free height offset per plate (diagnostic)": fit_tilts_def("B", K_MAIN, D_MAIN, with_x),
            "C: shape-only, free top-sticker positions, pose from shelf stickers + top shapes": fit_tilts_def("C", K_MAIN, D_MAIN, with_x),
            "D: shape-only relative to the cart rotation of the main combined fit": fit_tilts_def("D", K_MAIN, D_MAIN, with_x, rot=rot_main),
            "E: as D with the edges-only lens and its cart rotation": fit_tilts_def("D", Ke, de, with_x, rot=rot_edge),
            "J: joint fit inside the combined estimator": dict(tilt_deg=o["joint"][kind]["tilt_deg"], rms_px=None, n_corners=len(PTS)),
        }
        V = np.array([v["tilt_deg"] for v in defs.values()])
        half = 0.5 * (V.max(0) - V.min(0))
        st = np.array(A["sigma_total"])
        lv = {}
        for nm, (K, d) in lens_var.items():
            ra, nt = fit_tilts(K, d, with_x)
            lv[nm] = {"A": ra.x[:nt].tolist(), "C": fit_tilts_def("C", K, d, with_x)["tilt_deg"]}
        out[kind] = dict(names=A["names"], definitions=defs, definition_min=V.min(0).tolist(), definition_max=V.max(0).tolist(),
                         sigma_definition_halfrange=half.tolist(), sigma_total_without_definition=st.tolist(),
                         sigma_total_incl_definition=np.sqrt(st ** 2 + half ** 2).tolist(),
                         rule="sigma_total_incl_definition = [max(jackknife, bootstrap) (+) lens propagation] (+) half range over definitions A-E, J",
                         lens_variants_information=lv)
        print(f"refdef {kind}: " + ", ".join(f"{n} {v:+.1f} [{a:+.1f}..{b:+.1f}] +- {s:.1f}" for n, v, a, b, s in
                                              zip(A["names"], A["tilt_deg"], V.min(0), V.max(0), out[kind]["sigma_total_incl_definition"])), flush=True)
    store("refdef", out)
    # hinge at the edge of the black code square (45 mm) instead of the sticker edge (55 mm)
    HINGE_MM = 45.0
    cfgs = [dict(name=f"Y_hinge45_{a:+d}", family="Y_hinge45", angle=float(a), plate_tilts=family_tilts("Y_hinge", a)) for a in (10, 20)]
    with pool() as pl:
        recs = pl.map(evaluate, cfgs)
    HINGE_MM = 55.0
    store("hinge45", recs)


# ---------------------------------------------------------------- report
def fam_rows(scan, fam):
    d = scan.get(fam, {})
    if fam != "Y_centre" and "0" not in d and "0" in scan.get("Y_centre", {}):
        d = dict(d, **{"0": scan["Y_centre"]["0"]})
    angs = sorted(d, key=float)
    return [float(a) for a in angs], [d[a] for a in angs]


def phase_report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = load_out()
    scan = o["scan"]
    o["conclusions"] = auto_conclusions(o)
    save_json(o, OUT)
    colors = {"Y_centre": "#2a78d6", "X_centre": "#eb6834", "Y_hinge": "#1baf7a"}
    ink, ink2, grid_c, red = "#0b0b0b", "#52514e", "#e4e3df", "#e34948"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "text.color": ink})
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), dpi=110)
    ref = REFERENCE

    def famplot(ax, fn, style="-o", lw=2.0, ms=4, suffix="", label=True):
        for fam in ANGLES:
            a, recs = fam_rows(scan, fam)
            if recs:
                ax.plot(a, [fn(r) for r in recs], style, color=colors[fam], lw=lw, ms=ms, label=(fam + suffix) if label else None)

    ax = axs[0, 0]
    ax.axhspan(ref["main"]["f"] - 46.6, ref["main"]["f"] + 46.6, color=ink2, alpha=0.08, lw=0)
    ax.axhline(ref["edges_only"]["f"], color=red, lw=1.2, ls=":")
    famplot(ax, lambda r: r["combined"]["f"])
    ax.set_ylim(1410, 1525)
    ax.text(-15, ref["main"]["f"] + 46.6, "main 1468.6, band = 1-sigma 46.6 px", va="bottom", fontsize=8, color=ink2)
    ax.text(-15, ref["edges_only"]["f"], f"edges-only fit {ref['edges_only']['f']:.1f}", va="bottom", fontsize=8, color=red)
    ax.set_title("f [px], combined fit (main estimator)", loc="left")
    ax = axs[0, 1]
    famplot(ax, lambda r: r["combined"]["cy"])
    ax.axhspan(ref["main"]["cy"] - 29.1, ref["main"]["cy"] + 29.1, color=ink2, alpha=0.08, lw=0)
    ax.text(-15, ref["main"]["cy"] + 29.1, "main 503.2, band = 1-sigma 29.1 px", va="bottom", fontsize=8, color=ink2)
    ax.set_ylim(465, 545)
    ax.set_title("cy [px], combined fit", loc="left")
    ax = axs[0, 2]
    famplot(ax, lambda r: r["sticker_only"]["pp_fixed"]["f"], suffix=": pp fixed")
    famplot(ax, lambda r: r["sticker_only"]["pp_free"]["f"], style="--s", lw=1.4, ms=3.5, suffix=": pp free")
    ax.axhline(ref["edges_only"]["f"], color=red, lw=1.2, ls=":")
    ax.axhline(ref["main"]["f"], color=ink2, lw=1.0, ls=":")
    ax.text(-15, ref["edges_only"]["f"], "edges-only 1436.4", va="top", fontsize=8, color=red)
    ax.text(-15, ref["main"]["f"], "main 1468.6", va="bottom", fontsize=8, color=ink2)
    ax.set_title("f [px], sticker-only fits (all folded lenses)", loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="center right")
    ax = axs[1, 0]
    famplot(ax, lambda r: r["stickers_combined_lens"]["top"]["rms"], suffix=": top row (32)")
    famplot(ax, lambda r: r["stickers_combined_lens"]["overall"]["rms"], style="--s", lw=1.4, ms=3.5, suffix=": all 62 corners")
    ax.set_title("sticker RMS [px], combined lens fixed, LS pose per cart", loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax = axs[1, 1]
    famplot(ax, lambda r: r["combined"]["sig"]["M"])
    ax.set_title("M-block sigma [px] (sticker corners, per coordinate)", loc="left")
    ax = axs[1, 2]
    famplot(ax, lambda r: r["mapping_vs_main"]["cart_band"]["median"])
    ax.set_ylim(0, 1.0)
    ax.text(-15, 0.95, "main mapping 1-sigma in the cart band = 15.3 px (off scale)", va="top", fontsize=8, color=ink2)
    ax.set_title("mapping change vs main, cart band median [px]", loc="left")
    for ax in axs.ravel():
        ax.set_xlabel("tilt angle [deg]   (+ = outer side up: Y_centre, Y_hinge;  + = back side up: X_centre)", color=ink2, fontsize=8)
        ax.grid(color=grid_c, lw=0.8)
        ax.axvline(0, color=ink2, lw=0.6)
        ax.set_xlim(-16, 31)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for e in EXTENDED.get("Y_hinge", []):
        pass
    axs[0, 0].axvspan(10.5, 31, color=colors["Y_hinge"], alpha=0.05, lw=0)
    axs[0, 0].text(30.5, 1412, "Y_hinge beyond the\nrequested range", ha="right", va="bottom", fontsize=7.5, color=ink2)
    # per-plate fitted tilts (final alternation step), drawn at the mean Y tilt of the 4 plates
    pp = o.get("per_plate", {})
    for kind, mk in (("Y", "D"), ("YX", "P")):
        if kind in pp:
            last = pp[kind]["steps"][-1]
            t = float(np.mean(last["tilt_fit"]["tilt_deg"][:4]))
            r = last["combined_record"]
            lab = f"per-plate fitted {kind} tilts (x = mean ay)"
            axs[0, 0].plot([t], [r["combined"]["f"]], mk, color=ink, ms=7, label=lab)
            axs[1, 1].plot([t], [r["combined"]["sig"]["M"]], mk, color=ink, ms=7, label=lab)
            axs[1, 0].plot([t], [r["stickers_combined_lens"]["top"]["rms"]], mk, color=ink, ms=7)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
    axs[1, 1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("What-if: top stickers tilted (all other geometry = drawing; main result unchanged).  "
                 "Combined estimator and sticker-only fits vs tilt angle", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(PNG)
    plt.close(fig)
    write_md(o)
    print("report written")


def auto_conclusions(o):
    """Factual statements computed from the numbers (no hand-typed values)."""
    scan = o["scan"]
    C = []
    f0 = scan["Y_centre"]["0"]["combined"]
    for fam in ANGLES:
        a, recs = fam_rows(scan, fam)
        req = [(x, r) for x, r in zip(a, recs) if x not in EXTENDED.get(fam, [])]
        df = [r["combined"]["f"] - f0["f"] for _, r in req]
        dcy = [r["combined"]["cy"] - f0["cy"] for _, r in req]
        dk1 = [r["combined"]["k1"] - f0["k1"] for _, r in req]
        dm = [r["mapping_vs_main"]["cart_band"]["median"] for _, r in req]
        dmc = [r["mapping_vs_main"]["corners"]["median"] for _, r in req]
        sm = [r["combined"]["sig"]["M"] for _, r in req]
        so = [r["sticker_only"]["pp_free"] for _, r in req]
        sf = [r["sticker_only"]["pp_fixed"] for _, r in req]
        C.append(f"{fam} ({req[0][0]:+.0f}..{req[-1][0]:+.0f} deg): combined f changes by {min(df):+.2f}..{max(df):+.2f} px, cy by "
                 f"{min(dcy):+.2f}..{max(dcy):+.2f} px, k1 by {min(dk1):+.5f}..{max(dk1):+.5f}; mapping change vs the main result <= "
                 f"{max(dm):.2f} px (cart band median) / {max(dmc):.2f} px (corners); M-block sigma {min(sm):.2f}..{max(sm):.2f} px "
                 f"(0 deg: {f0['sig']['M']:.2f}). Sticker-only fits: pp fixed f {min(s['f'] for s in sf):.0f}..{max(s['f'] for s in sf):.0f}, "
                 f"pp free f {min(s['f'] for s in so):.0f}..{max(s['f'] for s in so):.0f}, RMS {min(s['rms'] for s in so):.2f}..{max(s['rms'] for s in so):.2f} px, "
                 f"folded in {sum(s['folds_inside_image'] for s in so)}/{len(so)} (fold margin {min(s['fold_margin'] for s in so):+.2f}..{max(s['fold_margin'] for s in so):+.2f}).")
        ext = [(x, r) for x, r in zip(a, recs) if x in EXTENDED.get(fam, [])]
        if ext:
            C.append(f"{fam} beyond the requested range: " + "; ".join(
                f"{x:+.0f} deg: f {r['combined']['f']:.1f}, sig M {r['combined']['sig']['M']:.2f}, RMS all {r['stickers_combined_lens']['overall']['rms']:.2f}, "
                f"sticker-only pp free f {r['sticker_only']['pp_free']['f']:.0f} ({'folded' if r['sticker_only']['pp_free']['folds_inside_image'] else 'not folded'})"
                for x, r in ext) + f" (centre rise {HINGE_MM:.0f} sin(angle) mm = " + ", ".join(f"{HINGE_MM * np.sin(np.deg2rad(x)):.1f}" for x, _ in ext) + " mm).")
    infl = _FIT.get("k1k2_influence_no_corners")
    if infl:
        C.append(f"Why the combined lens hardly moves: with the converged block weights the 62 sticker corners carry little weight "
                 f"(sigma M = {f0['sig']['M']:.1f} px vs V {f0['sig']['V']:.2f} px on {f0['block_rms']['V'][1]} edge points and S {f0['sig']['S']:.2f} px on "
                 f"{f0['block_rms']['S'][1]}); removing ALL corners at fixed block weights moves f only from {K_MAIN[0, 0]:.1f} to {infl['x'][0]:.1f} px "
                 f"(combined_fit.json, k1k2_influence_no_corners). A tilt changes the top corners by a few px only, and the M-block sigma stays "
                 f"far above the detection noise (0.1-0.2 px), so the stickers' weight never rises.")
    am = o.get("argmins", {})
    for fam, dd in am.items():
        v = dd["M_sigma"]
        w = dd["sticker_rms_combined_lens"]
        t = dd["top_rms_combined_lens"]
        e = dd["sticker_rms_edges_only_lens"]
        C.append(f"{fam} best common angle: M sigma argmin {v['grid_argmin']:+.0f} deg (refined {_f(v['refined_argmin'], 2)}{', range edge' if v['at_range_edge'] else ''}) "
                 f"{v['value']:.2f} vs {v['value_at_0']:.2f} px at 0; overall sticker RMS argmin {w['grid_argmin']:+.0f} (refined {_f(w['refined_argmin'], 2)}) "
                 f"{w['value']:.2f} vs {w['value_at_0']:.2f}; top-row RMS argmin {t['grid_argmin']:+.0f} (refined {_f(t['refined_argmin'], 2)}) "
                 f"{t['value']:.2f} vs {t['value_at_0']:.2f}; with the edges-only lens fixed: argmin {e['grid_argmin']:+.0f}, {e['value']:.2f} vs {e['value_at_0']:.2f} px.")
    pp = o.get("per_plate", {})
    rd = o.get("refdef", {})
    for kind, ch in pp.items():
        t0 = ch["steps"][0]["tilt_fit"]
        last = ch["steps"][-1]
        r = last["combined_record"]
        st = t0.get("sigma_total", t0["sigma_jackknife_sticker"])
        rule = "1-sigma = max(jackknife, bootstrap) (+) lens propagation"
        rng_txt = ""
        if kind in rd:
            st = rd[kind]["sigma_total_incl_definition"]
            rule = ("1-sigma = [max(jackknife, bootstrap) (+) lens propagation] (+) half range over 6 definitions of the cart reference; "
                    "without the definition term: " + ", ".join(f"{s:.1f}" for s in rd[kind]["sigma_total_without_definition"]))
            rng_txt = ("; range over the definitions A-E, J: " + ", ".join(f"{n} {a:+.1f}..{b:+.1f}" for n, a, b in
                                                                      zip(t0["names"], rd[kind]["definition_min"], rd[kind]["definition_max"])))
        C.append(f"Per-plate {kind} tilts (main lens fixed, definition A): " + ", ".join(f"{n} {v:+.1f} +- {s:.1f}" for n, v, s in zip(t0["names"], t0["tilt_deg"], st)) +
                 f" deg ({rule}){rng_txt}; sticker RMS {t0['rms_px']:.2f} px (top {t0['top_rms_px']:.2f}) vs "
                 f"{scan['Y_centre']['0']['stickers_main_lens']['overall']['rms']:.2f} flat. Combined fit with the tilts (after {len(ch['steps'])} step(s)): "
                 f"f {r['combined']['f']:.1f}, cx {r['combined']['cx']:.1f}, cy {r['combined']['cy']:.1f}, k1 {r['combined']['k1']:.4f}, "
                 f"k2 {r['combined']['k2']:.4f}, sig M {r['combined']['sig']['M']:.2f}; mapping vs main {r['mapping_vs_main']['cart_band']['median']:.2f} px (cart band).")
    jt = o.get("joint", {})
    for kind, j in jt.items():
        dd = ""
        if kind in pp:
            t0 = pp[kind]["steps"][0]["tilt_fit"]["tilt_deg"]
            dd = " Difference joint - LS-pose fit: " + ", ".join(f"{n} {v - w:+.1f}" for n, v, w in zip(j["tilt_names"], j["tilt_deg"], t0)) + \
                 " deg (how the cart pose is defined: edge-constrained vs sticker-only LS)."
        C.append(f"Joint fit ({kind} tilts free inside the combined estimator): " + ", ".join(f"{n} {v:+.1f}" for n, v in zip(j["tilt_names"], j["tilt_deg"])) +
                 f" deg; f {j['joint_lens']['f']:.1f}, cx {j['joint_lens']['cx']:.1f}, cy {j['joint_lens']['cy']:.1f}, k1 {j['joint_lens']['k1']:.4f}, "
                 f"k2 {j['joint_lens']['k2']:.4f}, sig M {j['joint_sig']['M']:.2f}." + dd)
    if "Y" in rd:
        r = rd["Y"]
        dn = list(r["definitions"])
        i4 = 3  # 310/X1600
        vals = [r["definitions"][k]["tilt_deg"][i4] for k in dn]
        lv = r["lens_variants_information"]
        lvv = [x for v in lv.values() for x in (v["A"][i4], v["C"][i4])]
        oth = [r["definitions"][k]["tilt_deg"][i] for k in dn for i in range(3)] + [x for v in lv.values() for i in range(3) for x in (v["A"][i], v["C"][i])]
        C.append("[review fix] The per-plate tilts depend on how the cart reference is defined (A LS pose from all corners, B + free plate "
                 "heights, C/D/E shape-only with free top-sticker positions, J joint): half the range over these definitions ("
                 + ", ".join(f"{n} {h:.1f}" for n, h in zip(r["names"], r["sigma_definition_halfrange"])) +
                 " deg) is of the same size as the statistical 1-sigma and is now included in the quoted +-. The shape-only definition D "
                 "(cart rotation of the main combined fit, only the top-sticker shapes) gives " +
                 ", ".join(f"{n} {v:+.1f}" for n, v in zip(r["names"], r["definitions"][dn[3]]["tilt_deg"])) + " deg. "
                 f"At 310/X1600 all definitions with the main lens give a small NEGATIVE tilt ({min(vals):+.1f}..{max(vals):+.1f} deg, outer side down) "
                 f"and so do the lens variants incl. the plumb lens of the earlier shape diagnostic ({min(lvv):+.1f}..{max(lvv):+.1f} deg); the earlier "
                 "shape diagnostic (geometry_diagnosis.md sect. 5: +3.9 / +4.8 deg at f 1400) used its own cart reference (nadir + sticker-shape "
                 "rotation, free in-plane yaw per sticker). The sign at 310/X1600 is therefore reference-dependent and only weakly determined; "
                 f"the outer-side-up tilts of the other three plates are robust (all definitions and lens variants >= {min(oth):+.1f} deg).")
    h45 = o.get("hinge45")
    if h45:
        C.append("[review fix] Hinge at 45 mm (edge of the black code square) instead of 55 mm: " + "; ".join(
            f"{x['angle']:+.0f} deg: f {x['combined']['f']:.2f}, cy {x['combined']['cy']:.2f}, k1 {x['combined']['k1']:.4f}, sig M {x['combined']['sig']['M']:.2f}, "
            f"mapping vs main {x['mapping_vs_main']['cart_band']['median']:.2f} / {x['mapping_vs_main']['corners']['median']:.2f} px (band / corners)"
            for x in h45) + " - the same conclusion as with 55 mm.")
    # consistency with the edges
    allrec = [r for fam in ANGLES for r in fam_rows(scan, fam)[1] if not (fam != "Y_centre" and r["name"] == "Y_centre_+0")]  # [fix] flat record once
    sb = [r["sticker_only"]["pp_free_fold_barrier"] for r in allrec]
    C.append(f"Sticker-only pp free WITH the fold barrier is multimodal: over the scan it lands at f {min(x['f'] for x in sb):.0f}..{max(x['f'] for x in sb):.0f}, "
             f"cy {min(x['cy'] for x in sb):.0f}..{max(x['cy'] for x in sb):.0f}, k1 {min(x['k1'] for x in sb):.3f}..{max(x['k1'] for x in sb):.3f} "
             f"(RMS {min(x['rms'] for x in sb):.2f}..{max(x['rms'] for x in sb):.2f}), i.e. the stickers alone do not define a valid lens for any tilt.")
    allrec += [s["combined_record"] for ch in pp.values() for s in ch["steps"]]
    allrec += list(o.get("common_angle", []) or [])
    so = [r["sticker_only"]["pp_free"] for r in allrec]
    sfx = [r["sticker_only"]["pp_fixed"] for r in allrec]
    se = [r["stickers_edges_only_lens"]["overall"]["rms"] for r in allrec]
    C.append(f"Consistency with the edges: over all {len(allrec)} tilt configurations the sticker-only fits give f = "
             f"{min(s['f'] for s in sfx):.0f}..{max(s['f'] for s in sfx):.0f} (pp fixed) / {min(s['f'] for s in so):.0f}..{max(s['f'] for s in so):.0f} (pp free) "
             f"vs {REFERENCE['edges_only']['f']:.1f} from the edges alone, the pp-free sticker lens folds inside the image in {sum(s['folds_inside_image'] for s in so)}/{len(so)} cases, "
             f"and the sticker RMS with the edges-only lens fixed stays at {min(se):.2f}..{max(se):.2f} px (flat: "
             f"{scan['Y_centre']['0']['stickers_edges_only_lens']['overall']['rms']:.2f}). No tested tilt makes the stickers consistent with the edges; "
             f"the dominant sticker mismatch (top stickers ~60-80 mm high relative to the shelf stickers, REPORT.md ch. 5) is a height offset that a tilt about the sticker "
             f"centre cannot produce and the hinge tilt produces only as {HINGE_MM:.0f} sin(angle) mm.")
    return C


def _f(v, n=1):
    return "-" if v is None else f"{v:.{n}f}"


def write_md(o):
    scan = o["scan"]
    L = []
    L.append("# What-if: top stickers tilted (angle scan)\n")
    L.append("Additional analysis on request; the main result (results/lens_result.json, drawing geometry = all stickers flat) "
             "is NOT changed. Script: `work/60_top_tilt_scan_fixed.py` (reviewed and corrected copy of `work/60_top_tilt_scan.py`; the scan, plates, "
             "common and check phases are identical, the review added the phases/sections marked [review fix]); all numbers: `work/cache/top_tilt_scan.json`; plot: "
             "`results/top_tilt_scan.png`.\n")
    L.append("## Definitions and sign conventions\n")
    for k, v in DEFINITIONS.items():
        L.append(f"* **{k}**: {v}")
    ver = o.get("verification", {})
    if ver:
        L.append("\n## Verification\n")
        L.append("Sign convention (Y_centre +10 deg; corners with Z > 0 are on the outer side, centres unchanged):\n")
        L.append("```")
        L += ver["sign_check"]
        L.append("```")
        L.append(f"0 deg reproduces the main fit: f = {ver['zero_deg_f']:.3f} px (main {ver['main_f']:.3f}), cx {ver['zero_deg_cx']:.2f}, "
                 f"cy {ver['zero_deg_cy']:.2f}, sticker RMS {ver['zero_deg_sticker_rms']:.3f} px (main 7.886).\n")
    rm, re_ = REFERENCE["main"], REFERENCE["edges_only"]
    L.append(f"References: main f {rm['f']:.1f} +- 46.6, cx {rm['cx']:.1f}, cy {rm['cy']:.1f}, k1 {rm['k1']:.4f}, k2 {rm['k2']:.4f}; "
             f"edges-only (V + S blocks) f {re_['f']:.1f}, cx {re_['cx']:.1f}, cy {re_['cy']:.1f}, k1 {re_['k1']:.4f}, k2 {re_['k2']:.4f}. "
             f"Main mapping 1-sigma: centre 4.0 / cart band 15.3 / corners 21.6 px.\n")
    for fam in ANGLES:
        a, recs = fam_rows(scan, fam)
        if not recs:
            continue
        L.append(f"\n## {fam}\n")
        if EXTENDED.get(fam):
            L.append(f"(ext.) = beyond the requested range, added because the metrics were still falling at the range edge.\n")
        L.append("Combined fit (main estimator) with the tilted top stickers; sticker RMS = radial, combined lens fixed, LS pose per cart; "
                 "dmap = rotation-compensated median |displacement| vs the main result (centre / cart band / corners).\n")
        L.append("| angle [deg] | f | cx | cy | k1 | k2 | sig M | sig L | sig V | sig S | RMS all | RMS top | RMS shelf | max all | dmap vs main c/b/k [px] | dmap vs edges-only band | fold margin |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ang, r in zip(a, recs):
            c = r["combined"]
            s = r["stickers_combined_lens"]
            mm = r["mapping_vs_main"]
            L.append(f"| {ang:+.0f}{' (ext.)' if ang in EXTENDED.get(fam, []) else ''} | {c['f']:.1f} | {c['cx']:.1f} | {c['cy']:.1f} | {c['k1']:.4f} | {c['k2']:.4f} | {c['sig']['M']:.2f} | "
                     f"{c['sig']['L']:.2f} | {c['sig']['V']:.3f} | {c['sig']['S']:.3f} | {s['overall']['rms']:.2f} | {s['top']['rms']:.2f} | "
                     f"{s['shelf']['rms']:.2f} | {s['overall']['max']:.1f} | {mm['centre']['median']:.1f} / {mm['cart_band']['median']:.1f} / "
                     f"{mm['corners']['median']:.1f} | {r['mapping_vs_edges_only']['cart_band']['median']:.1f} | {c['fold_margin']:.2f} |")
        L.append("\nPer plate (top-sticker RMS, combined lens) and sticker RMS with the main / edges-only lens fixed; sticker-only fits "
                 "(no fold barrier; pp fixed = image centre):\n")
        L.append("| angle | 80/X0 | 80/X1600 | 310/X0 | 310/X1600 | RMS main lens | RMS edges lens | only pp fixed: f / k1 / k2 / RMS | only pp free: f / cx / cy / RMS / fold | only pp free + barrier: f / cy / RMS |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for ang, r in zip(a, recs):
            s = r["stickers_combined_lens"]["plates"]
            sf, sp, sb = r["sticker_only"]["pp_fixed"], r["sticker_only"]["pp_free"], r["sticker_only"]["pp_free_fold_barrier"]
            L.append(f"| {ang:+.0f}{' (ext.)' if ang in EXTENDED.get(fam, []) else ''} | " + " | ".join(f"{s[p]['rms']:.2f}" for p in PLATES) +
                     f" | {r['stickers_main_lens']['overall']['rms']:.2f} | {r['stickers_edges_only_lens']['overall']['rms']:.2f} | "
                     f"{sf['f']:.0f} / {sf['k1']:.3f} / {sf['k2']:.3f} / {sf['rms']:.2f} | "
                     f"{sp['f']:.0f} / {sp['cx']:.0f} / {sp['cy']:.0f} / {sp['rms']:.2f} / {'FOLDED' if sp['folds_inside_image'] else 'ok'} ({sp['fold_margin']:+.2f}) | "
                     f"{sb['f']:.0f} / {sb['cy']:.0f} / {sb['rms']:.2f}{' (at barrier)' if sb['at_fold_barrier'] else ''} |")
    am = o.get("argmins")
    if am:
        L.append("\n## Best-fitting common angle per family (argmin over the scan; refined = parabola through the 3 grid points around the minimum)\n")
        L.append("| family | metric | grid argmin [deg] | refined [deg] | value there | value at 0 deg |")
        L.append("|---|---|---|---|---|---|")
        for fam, dd in am.items():
            for key, v in dd.items():
                if key == "sticker_only_ppfree_barrier_rms":
                    continue  # multimodal (pp wanders between local minima), see the family tables
                L.append(f"| {fam} | {key} | {v['grid_argmin']:+.0f}{' (range edge)' if v['at_range_edge'] else ''} | {_f(v['refined_argmin'], 2)} | "
                         f"{v['value']:.3f} | {v['value_at_0']:.3f} |")
    ca = o.get("common_angle")
    if ca:
        L.append("\nCombined fits at the refined argmin angles:\n")
        L.append("| configuration | chosen by | f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k | dmap vs edges-only band |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in ca:
            c = r["combined"]
            s = r["stickers_combined_lens"]
            mm = r["mapping_vs_main"]
            L.append(f"| {r['name']} | {r['chosen_by']} | {c['f']:.1f} | {c['cx']:.1f} | {c['cy']:.1f} | {c['k1']:.4f} | {c['k2']:.4f} | "
                     f"{c['sig']['M']:.2f} | {s['overall']['rms']:.2f} | {s['top']['rms']:.2f} | {mm['centre']['median']:.1f} / "
                     f"{mm['cart_band']['median']:.1f} / {mm['corners']['median']:.1f} | {r['mapping_vs_edges_only']['cart_band']['median']:.1f} |")
    pp = o.get("per_plate")
    if pp:
        L.append("\n## Per-plate fitted tilts\n")
        L.append("Tilts fitted to the 62 sticker corners with the lens FIXED (iteration 0: main result) and one LS pose per cart; then the "
                 "combined fit with these tilts; then alternation (tilts refitted with the new combined lens) until |delta f| < 1 px. "
                 "ay = Y_centre convention (+ = outer side up), bx = X_centre convention (+ = back side up).\n")
        L.append("Uncertainty columns: 'formal' = Gauss-Newton covariance scaled by the global residual variance (5.5 px per coordinate, "
                 "dominated by the top-vs-shelf height mismatch, not by the sticker shapes, which are clean squares to ~0.2 px) - an upper bound; "
                 "jackknife = delete one of the 20 stickers; bootstrap = stratified cluster bootstrap (top stickers resampled within their plate, "
                 "shelf stickers within their cart); lens-propagated = refit with the 100 bootstrap lenses of the main estimator; "
                 "total = max(jackknife, bootstrap) (+) lens; definition half-range = half the range of the tilt over 6 definitions of the "
                 "cart reference (section 'Reference-definition systematic' below); total incl. definition = total (+) definition half-range "
                 "= the 1-sigma to quote [review fix: the original quoted 'total' without the definition term].\n")
        L.append("For comparison (work/cache/geometry_diagnosis.md, sect. 5, per-sticker free pose, f = 1400, plumb distortion; converted to this "
                 "sign convention): 80/X0 +10.5 / +13.4, 80/X1600 +7.2 / +6.2, 310/X0 +11.2 / +13.3, 310/X1600 +3.9 / +4.8 deg (+- 1.4-2.4); the "
                 "metric board-tilt fits of sect. 3 also gave the opposite sign at 310/X1600. The sect. 5 values use a different lens and cart "
                 "reference; see 'Reference-definition systematic' below.\n")
        for kind, ch in pp.items():
            st0 = ch["steps"][0]["tilt_fit"]
            L.append(f"\n### {kind} (iteration 0, main lens fixed)\n")
            rdk = o.get("refdef", {}).get(kind)
            hdr = ("| tilt | value [deg] | formal | jackknife (sticker) | bootstrap (stratified) | lens-propagated | total | definition half-range "
                   "| **total incl. definition** | range over definitions | d tilt / 100 px f |")
            L.append(hdr)
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for j, nm in enumerate(st0["names"]):
                dh = f"{rdk['sigma_definition_halfrange'][j]:.2f}" if rdk else "-"
                ti = f"**{rdk['sigma_total_incl_definition'][j]:.2f}**" if rdk else "-"
                rg = f"{rdk['definition_min'][j]:+.1f}..{rdk['definition_max'][j]:+.1f}" if rdk else "-"
                L.append(f"| {nm} | {st0['tilt_deg'][j]:+.2f} | {st0['sigma_formal'][j]:.2f} | {st0['sigma_jackknife_sticker'][j]:.2f} | "
                         f"{st0['sigma_bootstrap_stratified'][j]:.2f} | {st0.get('sigma_lens_propagated', [None] * 8)[j] or 0:.2f} | "
                         f"{st0.get('sigma_total', [None] * 8)[j] or 0:.2f} | {dh} | {ti} | {rg} | {st0.get('dtilt_per_100px_f', [0] * 8)[j]:+.2f} |")
            L.append(f"\nSticker RMS with the main lens: {st0['rms_px']:.2f} px (top {st0['top_rms_px']:.2f}, shelf {st0['shelf_rms_px']:.2f}; "
                     "plates " + ", ".join(f"{p} {v:.2f}" for p, v in st0["plates_rms_px"].items()) + ") vs 7.89 px flat.\n")
            L.append("| iteration | lens used for the tilts (f) | tilts [deg] | combined f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k | dmap vs edges band |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for stp in ch["steps"]:
                tf = stp["tilt_fit"]
                r = stp["combined_record"]
                c = r["combined"]
                s = r["stickers_combined_lens"]
                mm = r["mapping_vs_main"]
                L.append(f"| {stp['iteration']} | {tf['lens']['f']:.1f} | " + ", ".join(f"{v:+.1f}" for v in tf["tilt_deg"]) +
                         f" | {c['f']:.1f} | {c['cx']:.1f} | {c['cy']:.1f} | {c['k1']:.4f} | {c['k2']:.4f} | {c['sig']['M']:.2f} | "
                         f"{s['overall']['rms']:.2f} | {s['top']['rms']:.2f} | {mm['centre']['median']:.1f} / {mm['cart_band']['median']:.1f} / "
                         f"{mm['corners']['median']:.1f} | {r['mapping_vs_edges_only']['cart_band']['median']:.1f} |")
            last = ch["steps"][-1]["combined_record"]
            so = last["sticker_only"]
            L.append(f"\nSticker-only fits with the final tilts: pp fixed f {so['pp_fixed']['f']:.0f}, k1 {so['pp_fixed']['k1']:.3f}, "
                     f"k2 {so['pp_fixed']['k2']:.3f}, RMS {so['pp_fixed']['rms']:.2f}; pp free f {so['pp_free']['f']:.0f}, "
                     f"pp ({so['pp_free']['cx']:.0f}, {so['pp_free']['cy']:.0f}), RMS {so['pp_free']['rms']:.2f}, "
                     f"{'FOLDED' if so['pp_free']['folds_inside_image'] else 'not folded'} (margin {so['pp_free']['fold_margin']:+.2f}).\n")
    jt = o.get("joint")
    if jt:
        L.append("\n## Joint fit: plate tilts as free parameters inside the combined estimator\n")
        L.append("| kind | tilts [deg] (formal sigma) | f | cx | cy | k1 | k2 | sig M | RMS all (LS poses) | RMS top | dmap vs main band | dmap vs edges band |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for kind, j in jt.items():
            r = j["record_with_fixed_tilts"]
            c = j["joint_lens"]
            s = r["stickers_combined_lens"]
            L.append(f"| {kind} | " + ", ".join(f"{n.split('_', 1)[0]} {n.split('_', 1)[1]} {v:+.1f} ({e:.1f})" for n, v, e in
                                                 zip(j["tilt_names"], j["tilt_deg"], j["tilt_sigma_formal"])) +
                     f" | {c['f']:.1f} | {c['cx']:.1f} | {c['cy']:.1f} | {c['k1']:.4f} | {c['k2']:.4f} | {j['joint_sig']['M']:.2f} | "
                     f"{s['overall']['rms']:.2f} | {s['top']['rms']:.2f} | {r['mapping_vs_main']['cart_band']['median']:.1f} | "
                     f"{r['mapping_vs_edges_only']['cart_band']['median']:.1f} |")
    rd = o.get("refdef")
    if rd:
        L.append("\n## Reference-definition systematic of the per-plate tilts [review fix]\n")
        L.append("A tilt is only defined relative to the cart frame, i.e. relative to an estimated cart pose. With the main lens fixed the "
                 "tilts were refitted under six definitions of that reference: A = LS pose per cart from all 62 corners (the values above); "
                 "B = A + a free height offset per plate (diagnostic only: stops the top-vs-shelf height mismatch from bending the pose; "
                 "the heights are NOT free in any lens fit); C = shape-only (every top sticker gets a free 3D position, so only its shape "
                 "carries the tilt; pose = LS from the shelf stickers + the top-sticker shapes); D = shape-only relative to the cart rotation "
                 "of the main combined fit (edge-constrained; only the 32 top-sticker corners); E = as D with the edges-only lens "
                 f"(f {rd['edges_only_lens']['f']:.1f}) and its cart rotation; J = joint fit inside the combined estimator. "
                 "Half the range over A-E, J is added in quadrature to the statistical total.\n")
        for kind in ("Y", "YX"):
            r = rd.get(kind)
            if not r:
                continue
            L.append(f"\n### {kind}\n")
            L.append("| definition | " + " | ".join(r["names"]) + " | RMS [px] (corners used) |")
            L.append("|---|" + "---|" * len(r["names"]) + "---|")
            for dn, dv in r["definitions"].items():
                rr = "-" if dv.get("rms_px") is None else f"{dv['rms_px']:.2f} ({dv['n_corners']})"
                extra = "" if not dv.get("dz_mm") else " (dz " + ", ".join(f"{z:+.0f}" for z in dv["dz_mm"]) + " mm)"
                L.append(f"| {dn}{extra} | " + " | ".join(f"{v:+.2f}" for v in dv["tilt_deg"]) + f" | {rr} |")
            L.append("| half range | " + " | ".join(f"{v:.2f}" for v in r["sigma_definition_halfrange"]) + " | |")
            L.append("| 1-sigma without / **incl.** definition | " + " | ".join(
                f"{a:.2f} / **{b:.2f}**" for a, b in zip(r["sigma_total_without_definition"], r["sigma_total_incl_definition"])) + " | |")
            L.append("\nLens variants (information only, not in the +-; A / C definitions):\n")
            L.append("| lens | " + " | ".join(r["names"]) + " |")
            L.append("|---|" + "---|" * len(r["names"]))
            for ln, lvv in r["lens_variants_information"].items():
                L.append(f"| {ln} | " + " | ".join(f"{a:+.1f} / {c:+.1f}" for a, c in zip(lvv["A"], lvv["C"])) + " |")
    h45 = o.get("hinge45")
    if h45:
        L.append("\n## Hinge position sensitivity [review fix]\n")
        L.append("Y_hinge with the hinge 45 mm from the sticker centre (edge of the black code square) instead of 55 mm (sticker edge); "
                 "centre rise 45 sin(angle) mm.\n")
        L.append("| angle | f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for x in h45:
            c = x["combined"]
            s = x["stickers_combined_lens"]
            mm = x["mapping_vs_main"]
            L.append(f"| {x['angle']:+.0f} | {c['f']:.1f} | {c['cx']:.1f} | {c['cy']:.1f} | {c['k1']:.4f} | {c['k2']:.4f} | {c['sig']['M']:.2f} | "
                     f"{s['overall']['rms']:.2f} | {s['top']['rms']:.2f} | {mm['centre']['median']:.2f} / {mm['cart_band']['median']:.2f} / {mm['corners']['median']:.2f} |")
    if o.get("conclusions"):
        L.append("\n## Conclusions\n")
        L += [f"* {c}" for c in o["conclusions"]]
    with open(MD, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    if PHASE == "check":
        phase_check()
    elif PHASE == "scan":
        phase_scan()
    elif PHASE == "plates":
        phase_plates()
    elif PHASE == "common":
        phase_common()
    elif PHASE == "refdef":
        phase_refdef()
    elif PHASE == "report":
        phase_report()
    elif PHASE == "all":
        phase_scan()
        phase_plates()
        phase_common()
        phase_refdef()
        phase_report()
    print(f"phase {PHASE} done in {time.time() - t0:.0f} s")
