"""(61) MEASURED orientation of every sticker from its own image shape (what-if analysis, branch B).

Question: are the top stickers tilted with respect to the cart, about which axis, by how much, with what uncertainty?
This is an additional diagnostic; it does NOT change the main result (results/lens_result.json is only read).

Method
------
* Lens: the main result (results/lens_result.json = x_main of work/cache/combined_fit.json).
* Every sticker is a flat 90 mm square (black code square; its position and height are NOT used).
  - 4 valid corners: corners undistorted with the lens, planar pose by cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE)
    in normalised coordinates -> both IPPE solutions refined by LM in the DISTORTED image (px); the solution whose
    normal is closer to the cart Z axis is taken (the other is reported: RMS and angle).
  - partly hidden shelf stickers (2 or 1 valid corners): pose by least squares from the valid corners + the traced
    visible sides (markerdata.marker_side_edges); with only 4-5 independent constraints (< 6 pose DoF) a weak prior on
    the sticker centre (shelf-sticker cart pose, drawing position, 30 mm per axis) is added -> flagged, not independent.
* Cart rotation (camera <- cart), none of them uses the top stickers:
  (a) 'shelf': per-cart least-squares pose from the SHELF sticker corners only (rows A, B, C; drawing geometry, lens fixed)
  (b) 'vp'   : rotation fitted to the vanishing-point residuals of the cart-X edges (shelf lips, rails) and the cart-Z
               edges (posts) of that cart (combined.Combined V block, lens fixed) - uses no sticker at all.
  (c) 'main' : the pose of the main combined fit (includes all stickers -> for reference only, not independent).
* Angles in the cart frame (X along the 1600 mm front, Y into the cart, Z up), from the sticker normal n (cart frame):
  dZ/dX = -n_x/n_z, dZ/dY = -n_y/n_z.
  pitch_outer = tilt about the cart Y axis, POSITIVE = OUTER END UP, outer = towards X=0 for the X0 end (stickers at
                X=82.5) and towards X=1600 for the X1600 end (X=1517.5): pitch_outer = -atan(dZ/dX) at X0, +atan(dZ/dX) at X1600
  roll        = tilt about the cart X axis, POSITIVE = BACK SIDE (Y=450) UP: roll = atan(dZ/dY)
  total       = angle between the sticker normal and the cart Z axis
  yaw         = in-plane rotation of the sticker about cart Z relative to its nominal rot_k*90 deg (right-handed, +Z)
* Uncertainty (Monte Carlo; robust 1-sigma = half of the 16-84 % range):
  noise  : corner noise N(0, corners_std_px) per coordinate (floor 0.10 px) + traced sides (0.15 px offset / end)
  lens   : the 100 cluster-bootstrap replicates of the main fit (combined_full.json 'boot': f, cx, cy, k1, k2), with
           both cart rotations and all sticker poses recomputed for each replicate
  rotsrc : cart-rotation statistics: (a) shelf stickers resampled within the cart, (b) VP edges resampled within
           their (cart, direction) group
  joint  : all of the above drawn together (headline statistical sigma)
  systematic: half the difference (a)-(b) and the difference raw vs edge-bias-corrected corners (in quadrature)
  frames : the 7 stills separately (empirical frame-to-frame scatter, check of the noise model)
* f dependence: f = 1420 / 1468.6 / 1520 with (i) cx, cy, k1, k2 and the poses refitted by the combined estimator at
  fixed f (block sigmas of the main fit), (ii) distortion fixed in pixel units (k1 f^2, k2 f^4 and pp constant);
  plus a scan f = 1300..1650 in mode (ii).
Outputs: work/cache/top_sticker_orientation.json, work/cache/top_sticker_orientation.md,
         results/top_sticker_orientation.png
usage: python3 61_top_sticker_orientation.py [NJOINT]     (full run, ~45 min with 2 processes)
       python3 61_top_sticker_orientation.py replot       (md / png / conclusion from work/cache/top_sticker_orientation.json)
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from common import CACHE, HALF, RESULTS, K_from, load_json, marker_center, project, rodrigues, save_json, undistort_points  # noqa: E402
from markerdata import load_markers  # noqa: E402

REPLOT = len(sys.argv) > 1 and sys.argv[1] == "replot"  # regenerate md / png / conclusion from the saved JSON (no Monte Carlo)
NJOINT = int(sys.argv[1]) if len(sys.argv) > 1 and not REPLOT else 400
NNOISE, NROT = int(os.environ.get("T61_NNOISE", 300)), int(os.environ.get("T61_NROT", 200))
NLENS = int(os.environ.get("T61_NLENS", 100))
NPROC = 2
T0 = time.time()

# ------------------------------------------------------------------ main-fit module header (data, edges, estimator)
_argv = sys.argv
sys.argv = ["50_combined.py", "none"]
NS = {"__name__": "c50"}
exec(compile(open(os.path.join(HERE, "50_combined.py")).read().split('if MODE == "fit":')[0], "50_combined_header", "exec"), NS)
sys.argv = _argv
Combined = NS["Combined"]
SPEC = NS["SPECS"]["k1k2"]
M = NS["M"]
SIDES = NS["SIDES"]
FIT = load_json(f"{CACHE}/combined_fit.json")
XM = np.array(FIT["x_main"])
CBM = Combined(NS["PTS"], NS["EDGES"], SPEC, sig=FIT["sig_main"])
KM, DM, POSES_M, _ = CBM.unpack(XM)
DM = DM[:5]
LR = load_json(f"{RESULTS}/lens_result.json")
assert abs(LR["camera_matrix"][0][0] - KM[0, 0]) < 1e-6 and abs(LR["dist_coeffs"][0] - DM[0]) < 1e-9
BOOT = np.array(load_json(f"{CACHE}/combined_full.json")["boot"])  # names f, cx, cy, k1, k2
assert FIT["summary"]["k1k2"]["names"] == ["f", "cx", "cy", "k1", "k2"]
CARTS = (80, 310)
CI = {80: 0, 310: 1}
MB = load_markers(which="biascorr")  # edge-bias-corrected corners (same order as M)
FRAMES = [m["per_frame"] for m in load_json(f"{CACHE}/markers_final.json")]  # 7 stills, jitter-compensated

# VP edges: cart X (shelf lips, rails) and cart Z (posts) of both carts; end-board edges excluded (vp_group None)
VE = [e for e in NS["LINES"] if e.get("cart") in CARTS and e.get("direction") in ("cartX", "cartZ") and e.get("vp_group", "__") is not None]
CBV = Combined([], VE, SPEC, use=("V",))
assert not CBV.has_wv and not CBV.floor and len(CBV.E) == len(VE)

# extra lenses from lens_by_method.json for the lens-choice check (edge-only, sticker-only drawing-geometry, shape, image-centre pp)
ALT_EXTRA = ("lines_joint@lines_joint_lc", "plumb_line@plumbline_lc", "vanishing_points@vanishing_lc", "markers_only",
             "markers_only_k1_ppfix", "sticker_shape", "two_cart_consistency", "combined_k1k2_ppfixed", "combined_k1k2p1p2", "alt_model_brown_k1k2")
OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * HALF  # IPPE_SQUARE order == common._CANON


def Rz(k):
    a = np.deg2rad(90.0 * k)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ sticker table
ST = []
for mi, m in enumerate(M):
    c = marker_center(m["id"], m["cart"])
    sides = [(int(e["id"].split("side")[-1]), np.array(e["points"], float)) for e in SIDES if e["sticker_mi"] == mi]
    nc = int(m["valid"].sum())
    ncons = 2 * nc + sum(2 - int(m["valid"][j]) - int(m["valid"][(j + 1) % 4]) for j, _ in sides)
    kind = "full" if nc == 4 else ("partial" if ncons >= 6 else "partial_prior")
    ST.append(dict(mi=mi, cart=m["cart"], id=m["id"], row=m["row"], rot_k=m["rot_k"], end="X0" if c[0] < 800 else "X1600",
                   centre=c, valid=m["valid"].copy(), uv=m["corners_px"].copy(), std=np.maximum(m["std"], 0.10), sides=sides,
                   nc=nc, ncons=ncons, kind=kind, label=f"{m['cart']}:{m['id']}"))
NST = len(ST)
TOPI = [i for i, s in enumerate(ST) if s["row"] == "top"]
PLATES = {}
for i in TOPI:
    PLATES.setdefault(f"{ST[i]['cart']}/{ST[i]['end']}", []).append(i)
PLATES = {k: sorted(v, key=lambda i: ST[i]["centre"][1]) for k, v in sorted(PLATES.items())}  # (Y=60, Y=390)
SHELF_FULL = [i for i, s in enumerate(ST) if s["row"] != "top" and s["kind"] == "full"]
SRC = ("shelf", "vp")
QN = ["pitch_outer", "roll", "total", "yaw", "dZdX"]


# ------------------------------------------------------------------ geometry helpers
def tilt_angles(Rcm, rot_k, end):
    """Rcm: sticker (marker) axes in the cart frame. Returns [pitch_outer, roll, total, yaw, atan(dZ/dX)] in deg."""
    n = Rcm[:, 2]
    px = np.degrees(np.arctan(-n[0] / n[2]))
    roll = np.degrees(np.arctan(-n[1] / n[2]))
    tot = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    D = Rcm @ Rz(rot_k).T
    yaw = np.degrees(np.arctan2(D[1, 0] - D[0, 1], D[0, 0] + D[1, 1]))
    return np.array([-px if end == "X0" else px, roll, tot, yaw, px])


def shelf_pose(K, d, uvs, pick=None, p0=None):
    """LS pose per cart from the shelf-sticker corners (drawing geometry). pick: dict cart -> list of sticker indices
    (cluster bootstrap, duplicates allowed). Returns {cart: (R, t, rms)}."""
    out = {}
    for c in CARTS:
        idx = [i for i, s in enumerate(ST) if s["cart"] == c and s["row"] != "top"] if pick is None else pick[c]
        X, uv = [], []
        for i in idx:
            m = M[ST[i]["mi"]]
            for j in range(4):
                if ST[i]["valid"][j]:
                    X.append(m["corners_3d"][j])
                    uv.append(uvs[i][j])
        X, uv = np.array(X), np.array(uv)
        x0 = POSES_M[CI[c]] if p0 is None else p0[c]
        r = least_squares(lambda pz: (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), x0, method="lm", xtol=1e-12, ftol=1e-12)
        out[c] = (rodrigues(r.x[:3]), r.x[3:].copy(), float(np.sqrt(np.mean(np.sum(r.fun.reshape(-1, 2) ** 2, 1)))), r.x.copy())
    return out


def vp_rot(K, d, cbv=None, r0=None):
    """Cart rotations from the VP residuals of the cart-X / cart-Z edges (lens fixed). Returns {cart: (R, rms)}."""
    d = np.r_[np.asarray(d, float).ravel(), np.zeros(5)][:5]
    if cbv is None:
        cbv = CBV if not np.any(d[2:5] != 0) else Combined([], VE, dict(SPEC, dist0=list(d)), use=("V",))  # p1, p2, k3 held fixed
    xi = [K[0, 0], K[0, 2], K[1, 2], d[0], d[1]]
    r0 = np.r_[POSES_M[0, :3], POSES_M[1, :3]] if r0 is None else r0

    def fun(r):
        return cbv.blocks(np.r_[xi, r[:3], 0, 0, 2000.0, r[3:], 0, 0, 2000.0])["V"]

    rr = least_squares(fun, r0, x_scale="jac", method="trf", xtol=1e-12, ftol=1e-12)
    rms = float(np.sqrt(np.mean(rr.fun ** 2)))
    return {80: (rodrigues(rr.x[:3]), rms), 310: (rodrigues(rr.x[3:]), rms)}, rr.x


def ippe_full(uv, K, d):
    """Both IPPE solutions of a 4-corner sticker, LM-refined in the distorted image. Returns list of (x6, rms)."""
    n = undistort_points(uv, K, d)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(OBJ, n.reshape(-1, 1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    sols = []
    for rv, tv in zip(rvecs, tvecs):
        r = least_squares(lambda pz: (project(OBJ, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), np.r_[rv.ravel(), tv.ravel()],
                          method="lm", xtol=1e-12, ftol=1e-12)
        sols.append((r.x.copy(), float(np.sqrt(np.mean(np.sum(r.fun.reshape(-1, 2) ** 2, 1))))))
    return sols


def partial_pose(s, uv, sides, K, d, R0, t0, prior=None, starts=None):
    """LS pose of a partly hidden sticker from its valid corners + traced sides (+ optional centre prior, mm)."""
    v = s["valid"]
    Us = []
    for j, P in sides:  # side points -> undistorted pixel coordinates
        q = undistort_points(P, K, d)
        Us.append((j, np.column_stack([K[0, 0] * q[:, 0] + K[0, 2], K[1, 1] * q[:, 1] + K[1, 2]])))

    def res(p):
        R = rodrigues(p[:3])
        t = p[3:]
        out = [(project(OBJ[v], K, d, R=R, tvec=t) - uv[v]).ravel()]
        Xc = OBJ @ R.T + t
        for j, U in Us:
            a, b = K @ Xc[j], K @ Xc[(j + 1) % 4]
            l = np.cross(a, b)
            out.append((U @ l[:2] + l[2]) / np.hypot(l[0], l[1]) / np.sqrt(len(U) / 2.0))
        if prior is not None:
            out.append((t - prior) / 30.0 * 0.15)
        return np.concatenate(out)

    best, alts = None, []
    starts = starts or [(0, 0), (12, 0), (-12, 0), (0, 12), (0, -12)]
    for ax, ay in starts:
        Rs = R0 @ cv2.Rodrigues(np.deg2rad(np.array([ax, ay, 0.0])))[0]
        x0 = np.r_[cv2.Rodrigues(Rs)[0].ravel(), t0]
        try:
            r = least_squares(res, x0, method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=400)
        except Exception:  # noqa
            continue
        alts.append((r.x.copy(), float(r.cost)))
        if best is None or r.cost < best[1]:
            best = (r.x.copy(), float(r.cost))
    nc = rodrigues(best[0][:3])[:, 2]
    distinct = [a for a in alts if np.degrees(np.arccos(np.clip(rodrigues(a[0][:3])[:, 2] @ nc, -1, 1))) > 3.0]
    alt = min(distinct, key=lambda a: a[1]) if distinct else None
    return best, alt


# ------------------------------------------------------------------ one evaluation
def evaluate(K, d, uvs=None, sides=None, pick=None, cbv=None, extra_R=None, detail=False):
    """All sticker poses + angles for one lens / data draw. uvs: per-sticker 4x2 corner arrays; sides: per-sticker list."""
    uvs = [s["uv"] for s in ST] if uvs is None else uvs
    sides = [s["sides"] for s in ST] if sides is None else sides
    sp = shelf_pose(K, d, uvs, pick=pick)
    vr, _ = vp_rot(K, d, cbv=cbv)
    Rsrc = {"shelf": {c: sp[c][0] for c in CARTS}, "vp": {c: vr[c][0] for c in CARTS}}
    if extra_R:
        Rsrc.update(extra_R)
    ang = {k: np.full((NST, 5), np.nan) for k in Rsrc}
    ncam = np.full((NST, 3), np.nan)
    tcam = np.full((NST, 3), np.nan)
    rms = np.full(NST, np.nan)
    info = []
    for i, s in enumerate(ST):
        c = s["cart"]
        zc = 0.5 * (Rsrc["shelf"][c][:, 2] + Rsrc["vp"][c][:, 2])
        fl = None
        if s["kind"] == "full":
            sols = ippe_full(uvs[i], K, d)
            sols.sort(key=lambda q: -rodrigues(q[0][:3])[:, 2] @ zc)
            x, rr = sols[0]
            if len(sols) > 1:
                fl = dict(rms_px=sols[1][1], angle_to_cartZ_deg=float(np.degrees(np.arccos(np.clip(rodrigues(sols[1][0][:3])[:, 2] @ zc, -1, 1)))))
        else:
            R0 = Rsrc["shelf"][c] @ Rz(s["rot_k"])
            t0 = sp[c][0] @ s["centre"] + sp[c][1]
            prior = t0 if s["kind"] == "partial_prior" else None
            best, alt = partial_pose(s, uvs[i], sides[i], K, d, R0, t0, prior=prior)
            x = best[0]
            R = rodrigues(x[:3])
            v = s["valid"]
            rr = float(np.sqrt(np.mean(np.sum((project(OBJ[v], K, d, R=R, tvec=x[3:]) - uvs[i][v]) ** 2, 1))))
            if alt is not None:
                fl = dict(cost_ratio=alt[1] / max(best[1], 1e-12), angle_to_cartZ_deg=float(np.degrees(np.arccos(np.clip(rodrigues(alt[0][:3])[:, 2] @ zc, -1, 1)))))
        R = rodrigues(x[:3])
        ncam[i] = R[:, 2]
        tcam[i] = x[3:]
        rms[i] = rr
        for k, Rc in Rsrc.items():
            ang[k][i] = tilt_angles(Rc[c].T @ R, s["rot_k"], s["end"])
        if detail:
            info.append(dict(other_solution=fl))
    out = dict(ang=ang, ncam=ncam, tcam=tcam, rms=rms, R={k: {c: v[c] for c in CARTS} for k, v in Rsrc.items()},
               t_shelf={c: sp[c][1] for c in CARTS}, shelf_rms={c: sp[c][2] for c in CARTS}, vp_rms=vr[80][1])
    if detail:
        out["info"] = info
    return out


# ------------------------------------------------------------------ Monte Carlo draws
def noisy_data(rng, base=None):
    base = [s["uv"] for s in ST] if base is None else base
    uvs = [base[i] + rng.normal(size=(4, 2)) * s["std"][:, None] for i, s in enumerate(ST)]
    sides = []
    for s in ST:
        lst = []
        for j, P in s["sides"]:
            dvec = P[-1] - P[0]
            L = np.hypot(*dvec)
            nrm = np.array([-dvec[1], dvec[0]]) / L
            u = ((P - P.mean(0)) @ dvec) / L  # position along the side
            off = rng.normal(0, 0.15) + rng.normal(0, 0.15) * u / (L / 2)
            lst.append((j, P + off[:, None] * nrm[None, :]))
        sides.append(lst)
    return uvs, sides


def shelf_pick(rng):
    pick = {}
    for c in CARTS:
        idx = [i for i, s in enumerate(ST) if s["cart"] == c and s["row"] != "top"]
        while True:
            p = list(rng.choice(idx, len(idx), replace=True))
            ends = {ST[i]["end"] for i in p}
            rows = {ST[i]["row"] for i in p}
            if len(ends) == 2 and len(rows) >= 2 and sum(ST[i]["nc"] for i in p) >= 6:
                break
        pick[c] = p
    return pick


def edge_resample(rng):
    groups = {}
    for e in VE:
        groups.setdefault((e["cart"], e["direction"]), []).append(e)
    out = []
    for g, lst in groups.items():
        for q in rng.integers(0, len(lst), len(lst)):
            out.append(dict(lst[q], id=f"{lst[q]['id']}#{len(out)}"))
    return Combined([], out, SPEC, use=("V",))


def lens_of_boot(b):
    return K_from(b[0], b[0], b[1], b[2]), np.array([b[3], b[4], 0, 0, 0])


def pack(o):
    return dict(shelf=o["ang"]["shelf"], vp=o["ang"]["vp"], ncam=o["ncam"], tcam=o["tcam"], rms=o["rms"],
                R_shelf=np.array([o["R"]["shelf"][c] for c in CARTS]), R_vp=np.array([o["R"]["vp"][c] for c in CARTS]),
                t_shelf=np.array([o["t_shelf"][c] for c in CARTS]))


def task(args):
    kind, k, seed = args
    rng = np.random.default_rng(seed)
    K, d = KM, DM
    uvs = sides = pick = cbv = None
    if kind in ("lens", "joint"):
        K, d = lens_of_boot(BOOT[k % len(BOOT)])
    if kind in ("noise", "joint"):
        uvs, sides = noisy_data(rng)
    if kind in ("rotsrc", "joint"):
        pick = shelf_pick(rng)
        cbv = edge_resample(rng)
    if kind == "frame":
        uvs = []
        for s in ST:
            q = s["uv"].copy()
            for j in range(4):
                v = FRAMES[s["mi"]][k][j]
                if s["valid"][j] and v is not None and np.all(np.isfinite(np.array(v, float))):
                    q[j] = v
            uvs.append(q)
    try:
        return kind, pack(evaluate(K, d, uvs=uvs, sides=sides, pick=pick, cbv=cbv))
    except Exception as ex:  # noqa
        return kind, dict(error=repr(ex))


def synth_task(args):
    """Synthetic round trip (sign conventions + bias): corners generated with the main lens and main-fit poses, shelf stickers
    flat (drawing), top stickers rotated about their centres by a known (pitch_outer, roll); optional corner noise."""
    po, ro, seed = args
    rng = np.random.default_rng(max(seed, 0))
    uvs = []
    for s in ST:
        C3 = M[s["mi"]]["corners_3d"].copy()
        if s["row"] == "top":
            T = tilt_rot(-po if s["end"] == "X0" else po, ro)
            C3 = s["centre"] + (C3 - s["centre"]) @ T.T
        uv = project(C3, KM, DM, rvec=POSES_M[CI[s["cart"]], :3], tvec=POSES_M[CI[s["cart"]], 3:])
        if seed >= 0:
            uv = uv + rng.normal(size=(4, 2)) * s["std"][:, None]
        uvs.append(uv)
    o = evaluate(KM, DM, uvs=uvs)
    return po, ro, seed, 0.5 * (o["ang"]["shelf"] + o["ang"]["vp"]), o["ang"]["shelf"]


def synthetic_check(nrep=int(os.environ.get("T61_NSYN", 40))):
    truths = [(0.0, 0.0), (5.0, 2.0), (14.0, -1.5), (-8.0, 4.0)]
    jobs = [(po, ro, -1) for po, ro in truths] + [(po, ro, 7000 + q) for po, ro in truths for q in range(nrep)]
    with get_context("fork").Pool(NPROC) as pool:
        res = pool.map(synth_task, jobs, chunksize=4)
    out = []
    for po, ro in truths:
        nf = [r for r in res if r[0] == po and r[1] == ro and r[2] < 0][0]
        ns = np.array([r[3] for r in res if r[0] == po and r[1] == ro and r[2] >= 0])
        top = [i for i in TOPI]
        shf = SHELF_FULL
        out.append(dict(true_pitch_outer=po, true_roll=ro, n_noisy_reps=int(len(ns)),
                        noise_free_top_pitch_outer=[float(v) for v in nf[3][top, 0]], noise_free_top_roll=[float(v) for v in nf[3][top, 1]],
                        noise_free_top_pitch_outer_shelf_source=[float(v) for v in nf[4][top, 0]],
                        noise_free_shelf_pitch_roll=[[float(nf[3][i, 0]), float(nf[3][i, 1])] for i in shf],
                        noisy_top_mean_error_pitch_roll=[float(np.mean(ns[:, top, 0]) - po), float(np.mean(ns[:, top, 1]) - ro)],
                        noisy_top_rms_error_pitch_roll=[float(np.sqrt(np.mean((ns[:, top, 0] - po) ** 2))), float(np.sqrt(np.mean((ns[:, top, 1] - ro) ** 2)))],
                        noisy_shelf_rms_pitch_roll=[float(np.sqrt(np.mean(ns[:, shf, 0] ** 2))), float(np.sqrt(np.mean(ns[:, shf, 1] ** 2)))],
                        top_labels=[ST[i]["label"] for i in top]))
        print(f"synthetic truth pitch {po:+.1f} roll {ro:+.1f}: noise-free top pitch {np.round(nf[3][top, 0], 2)} roll {np.round(nf[3][top, 1], 2)}; "
              f"noisy mean err {out[-1]['noisy_top_mean_error_pitch_roll']} rms {out[-1]['noisy_top_rms_error_pitch_roll']}", flush=True)
    return out


def refit_at_f(fv):
    """Combined estimator at fixed f (block sigmas of the main fit): cx, cy, k1, k2, poses refitted."""
    r = least_squares(lambda y: CBM.residuals(np.r_[fv, y]), XM[1:], method="trf", x_scale="jac", max_nfev=300, xtol=1e-10, ftol=1e-10)
    x = np.r_[fv, r.x]
    K, d, poses, _ = CBM.unpack(x)
    return dict(f=fv, x=x, K=K, d=d[:5], poses=poses, block_rms=CBM.block_rms(x))


def pixel_lens(fv):
    f0 = KM[0, 0]
    return K_from(fv, fv, KM[0, 2], KM[1, 2]), np.array([DM[0] * (fv / f0) ** 2, DM[1] * (fv / f0) ** 4, 0, 0, 0])


def rstd(a, axis=0):
    return 0.5 * (np.nanpercentile(a, 84, axis=axis) - np.nanpercentile(a, 16, axis=axis))


def main():
    print(f"stickers {NST}: " + ", ".join(f"{s['label']}({s['row']},{s['kind']},{s['ncons']})" for s in ST), flush=True)
    # ---------------- nominal
    extra = {"main": {c: rodrigues(POSES_M[CI[c], :3]) for c in CARTS}}
    nom = evaluate(KM, DM, extra_R=extra, detail=True)
    nomb = evaluate(KM, DM, uvs=[mb["corners_px"] for mb in MB], extra_R=extra)
    for c in CARTS:
        for a, b in (("shelf", "vp"), ("main", "vp"), ("shelf", "main")):
            D = nom["R"][a][c].T @ nom["R"][b][c]
            print(f"cart {c}: rotation {a} vs {b}: {np.degrees(np.linalg.norm(cv2.Rodrigues(D)[0])):.3f} deg "
                  f"(about cart X/Y/Z {np.round(np.degrees(cv2.Rodrigues(D)[0].ravel()), 3)})")
    print("shelf-pose RMS px", nom["shelf_rms"], "VP RMS px", nom["vp_rms"], flush=True)
    for i, s in enumerate(ST):
        a, b = nom["ang"]["shelf"][i], nom["ang"]["vp"][i]
        print(f"{s['label']:8s} {s['row']:3s} {s['end']:5s} {s['kind']:13s} pitch_outer {a[0]:7.2f}/{b[0]:7.2f} roll {a[1]:6.2f}/{b[1]:6.2f} "
              f"total {a[2]:6.2f}/{b[2]:6.2f} yaw {a[3]:6.2f}/{b[3]:6.2f} rms {nom['rms'][i]:.3f} other {nom['info'][i]['other_solution']}", flush=True)

    # ---------------- Monte Carlo (2 worker processes)
    jobs = [("noise", k, 1000 + k) for k in range(NNOISE)] + [("lens", k, 2000 + k) for k in range(min(NLENS, len(BOOT)))] + \
           [("rotsrc", k, 3000 + k) for k in range(NROT)] + [("joint", k, 4000 + k) for k in range(NJOINT)] + \
           [("frame", k, 0) for k in range(len(FRAMES[0]))]
    res = {}
    with get_context("fork").Pool(NPROC) as pool:
        for q, (kind, o) in enumerate(pool.imap(task, jobs, chunksize=4)):
            if "error" in o:
                print("MC error", kind, o["error"], flush=True)
                continue
            res.setdefault(kind, []).append(o)
            if q % 100 == 0:
                print(f"  MC {q}/{len(jobs)} {time.time() - T0:.0f} s", flush=True)
    MC = {k: {f: np.array([o[f] for o in v]) for f in v[0]} for k, v in res.items()}
    print({k: len(v["shelf"]) for k, v in MC.items()}, flush=True)

    # ---------------- f dependence
    fvals = [1420.0, float(KM[0, 0]), 1520.0]
    with get_context("fork").Pool(NPROC) as pool:
        refits = pool.map(refit_at_f, [fvals[0], fvals[2]])
    refits = [refits[0], dict(f=fvals[1], x=XM, K=KM, d=DM, poses=POSES_M, block_rms=CBM.block_rms(XM)), refits[1]]
    fdep = {"refit": [], "pixel": []}
    for rf in refits:
        o = evaluate(rf["K"], rf["d"], extra_R={"main": {c: rodrigues(rf["poses"][CI[c], :3]) for c in CARTS}})
        fdep["refit"].append(dict(f=rf["f"], cx=rf["K"][0, 2], cy=rf["K"][1, 2], k1=rf["d"][0], k2=rf["d"][1], block_rms=rf["block_rms"],
                                  ang={k: v for k, v in o["ang"].items()}))
        print(f"refit f {rf['f']:.1f}: pp ({rf['K'][0, 2]:.1f},{rf['K'][1, 2]:.1f}) k1 {rf['d'][0]:.4f} k2 {rf['d'][1]:.4f}", flush=True)
    for fv in sorted(set(list(np.arange(1300.0, 1651.0, float(os.environ.get("T61_FSTEP", 25.0)))) + fvals)):
        K, d = pixel_lens(fv)
        o = evaluate(K, d)
        fdep["pixel"].append(dict(f=float(fv), ang={k: v for k, v in o["ang"].items()}))
    print(f"f dependence done {time.time() - T0:.0f} s", flush=True)
    # ---------------- alternative lens models (systematic check of the lens choice; nominal data, no MC)
    alts = []
    for fn, pick in ((f"{RESULTS}/lens_alternatives.json", None), (f"{RESULTS}/lens_by_method.json", ALT_EXTRA)):
        for o in load_json(fn):
            mid = o.get("method_id", o.get("method", "")[:40])
            if pick is not None and mid not in pick:
                continue
            K = np.array(o["camera_matrix"], float)
            d = np.array(o["dist_coeffs"], float)
            try:
                a = evaluate(K, d)
                alts.append(dict(method_id=mid, f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]), dist=d.tolist(),
                                 folds=bool(o.get("radial_map_folds_inside_image", False)), ang={k: v for k, v in a["ang"].items()}))
                print(f"alt lens {mid:34s} f {K[0, 0]:.0f}", flush=True)
            except Exception as ex:  # noqa
                print("alt lens failed", mid, ex)
    synth = synthetic_check()
    return nom, nomb, MC, fdep, alts, synth


# ------------------------------------------------------------------ summarising
def plate_combo(arr, w):
    """arr: (..., NST, q) -> dict plate -> (..., q) weighted mean of the plate's two stickers."""
    return {p: (arr[..., idx, :] * w[idx][:, None]).sum(-2) / w[idx].sum() for p, idx in PLATES.items()}


def summarise(nom, nomb, MC, fdep, alts):
    comb = lambda a: 0.5 * (a["shelf"] + a["vp"])  # noqa: E731  headline: mean of the two independent rotation sources
    nomA = {k: nom["ang"][k] for k in ("shelf", "vp", "main")}
    nomA["comb"] = comb(nom["ang"])
    nombA = comb(nomb["ang"])
    sig = {}
    for kind in ("noise", "lens", "rotsrc", "joint", "frame"):
        mc = MC[kind]
        sig[kind] = {src: rstd(mc[src]) for src in SRC}
        sig[kind]["comb"] = rstd(comb(mc))
        if kind == "frame":  # 7 samples: plain std, and the std of the 7-frame mean
            sig[kind] = {src: np.nanstd(v if src != "comb" else comb(mc), axis=0, ddof=1) for src, v in
                         (("shelf", mc["shelf"]), ("vp", mc["vp"]), ("comb", None))}
    sys_ab = 0.5 * np.abs(nomA["shelf"] - nomA["vp"])
    sys_bias = np.abs(nombA - nomA["comb"])
    tot = np.sqrt(sig["joint"]["comb"] ** 2 + sys_ab ** 2 + sys_bias ** 2)
    # plate combination: inverse-variance weights from the corner-noise sigma of pitch_outer
    w = 1.0 / np.maximum(sig["noise"]["comb"][:, 0], 1e-3) ** 2
    plates = {}
    pn = plate_combo(nomA["comb"], w)
    pnb = plate_combo(nombA, w)
    pj = plate_combo(comb(MC["joint"]), w)
    pnoise = plate_combo(comb(MC["noise"]), w)
    plens = plate_combo(comb(MC["lens"]), w)
    prot = plate_combo(comb(MC["rotsrc"]), w)
    ps, pv = plate_combo(nomA["shelf"], w), plate_combo(nomA["vp"], w)
    for p, idx in PLATES.items():
        sj = rstd(pj[p])
        sab = 0.5 * np.abs(ps[p] - pv[p])
        sb = np.abs(pnb[p] - pn[p])
        dd = comb(MC["noise"])[:, idx[0], :] - comb(MC["noise"])[:, idx[1], :]
        ddl = comb(MC["lens"])[:, idx[0], :] - comb(MC["lens"])[:, idx[1], :]
        ddj = comb(MC["joint"])[:, idx[0], :] - comb(MC["joint"])[:, idx[1], :]
        diff = nomA["comb"][idx[0]] - nomA["comb"][idx[1]]
        plates[p] = dict(stickers=[ST[i]["label"] for i in idx], weights=[float(w[i]) for i in idx],
                         value=dict(zip(QN, pn[p])), value_shelf=dict(zip(QN, ps[p])), value_vp=dict(zip(QN, pv[p])),
                         value_biascorr=dict(zip(QN, pnb[p])),
                         sigma_noise=dict(zip(QN, rstd(pnoise[p]))), sigma_lens=dict(zip(QN, rstd(plens[p]))),
                         sigma_rotsrc=dict(zip(QN, rstd(prot[p]))), sigma_joint=dict(zip(QN, sj)),
                         sys_half_diff_shelf_vp=dict(zip(QN, sab)), sys_biascorr=dict(zip(QN, sb)),
                         sigma_total=dict(zip(QN, np.sqrt(sj ** 2 + sab ** 2 + sb ** 2))),
                         sticker_difference=dict(zip(QN, diff)), sticker_difference_sigma_noise=dict(zip(QN, rstd(dd))),
                         sticker_difference_sigma_lens=dict(zip(QN, rstd(ddl))), sticker_difference_sigma_joint=dict(zip(QN, rstd(ddj))),
                         sticker_difference_note="first minus second sticker (sorted by Y: Y=60 front sticker minus Y=390 back sticker)")
    # ---------------- per sticker table
    stick = []
    for i, s in enumerate(ST):
        e = dict(label=s["label"], cart=s["cart"], id=s["id"], row=s["row"], end=s["end"], kind=s["kind"], n_valid_corners=s["nc"],
                 n_constraints=s["ncons"], value=dict(zip(QN, nomA["comb"][i])), value_shelf=dict(zip(QN, nomA["shelf"][i])),
                 value_vp=dict(zip(QN, nomA["vp"][i])), value_main_pose=dict(zip(QN, nomA["main"][i])),
                 value_biascorr=dict(zip(QN, nombA[i])),
                 sigma_noise=dict(zip(QN, sig["noise"]["comb"][i])), sigma_lens=dict(zip(QN, sig["lens"]["comb"][i])),
                 sigma_rotsrc=dict(zip(QN, sig["rotsrc"]["comb"][i])), sigma_rotsrc_shelf=dict(zip(QN, sig["rotsrc"]["shelf"][i])),
                 sigma_rotsrc_vp=dict(zip(QN, sig["rotsrc"]["vp"][i])), sigma_joint=dict(zip(QN, sig["joint"]["comb"][i])),
                 sys_half_diff_shelf_vp=dict(zip(QN, sys_ab[i])), sys_biascorr=dict(zip(QN, sys_bias[i])),
                 sigma_total=dict(zip(QN, tot[i])), frames_std=dict(zip(QN, sig["frame"]["comb"][i])),
                 fit_rms_px=float(nom["rms"][i]), other_solution=nom["info"][i]["other_solution"],
                 determined=bool(s["kind"] == "full"), centre_y=float(s["centre"][1]),
                 normal_cam=nom["ncam"][i].tolist(), centre_cam_mm=nom["tcam"][i].tolist())
        stick.append(e)
    # ---------------- sanity checks
    checks = {}
    # (1) shelf normals mutually parallel (camera frame; independent of the cart rotation)
    par = []
    for a in SHELF_FULL:
        for b in SHELF_FULL:
            if a < b and ST[a]["cart"] == ST[b]["cart"]:
                ang = lambda n: np.degrees(np.arccos(np.clip(np.sum(n[..., a, :] * n[..., b, :], -1), -1, 1)))  # noqa: E731
                par.append(dict(pair=f"{ST[a]['label']}-{ST[b]['label']}", angle_deg=float(ang(nom["ncam"])),
                                sigma_joint_deg=float(rstd(ang(MC["joint"]["ncam"]))), sigma_noise_deg=float(rstd(ang(MC["noise"]["ncam"])))))
    checks["shelf_normals_pairwise"] = par
    # (2) top vs shelf sticker at the same image location / end (differences of pitch_outer and roll; rotation cancels)
    pairs = []
    for p, idx in PLATES.items():
        c, end = int(p.split("/")[0]), p.split("/")[1]
        for j in SHELF_FULL:
            if ST[j]["cart"] == c and ST[j]["end"] == end:
                for i in idx:
                    dn = nomA["comb"][i] - nomA["comb"][j]
                    dj = comb(MC["joint"])[:, i, :] - comb(MC["joint"])[:, j, :]
                    dl = comb(MC["lens"])[:, i, :] - comb(MC["lens"])[:, j, :]
                    pairs.append(dict(top=ST[i]["label"], shelf=ST[j]["label"], d_pitch_outer=float(dn[0]), d_roll=float(dn[1]),
                                      sigma_joint=[float(v) for v in rstd(dj)[:2]], sigma_lens_only=[float(v) for v in rstd(dl)[:2]]))
    checks["top_minus_shelf_same_end"] = pairs
    # (3) IPPE flip
    checks["ippe_other_solution"] = {ST[i]["label"]: dict(chosen_rms_px=float(nom["rms"][i]), **(nom["info"][i]["other_solution"] or {}))
                                     for i in range(NST) if ST[i]["kind"] == "full"}
    # (4) scale: chord between the two stickers of a plate (PnP with the 90 mm code) vs the drawing 330 mm, chord direction
    chords = {}
    for p, idx in PLATES.items():
        c = int(p.split("/")[0])
        def chord(t, R):  # noqa: E306
            v = t[..., idx[1], :] - t[..., idx[0], :]
            L = np.linalg.norm(v, axis=-1)
            u = np.einsum("...ji,...j->...i", R, v) / L[..., None]  # cart frame (R: camera <- cart)
            return L, np.degrees(np.arctan2(u[..., 2], u[..., 1])), np.degrees(np.arctan2(u[..., 0], u[..., 1]))
        L0, r0, y0 = chord(nom["tcam"], 0.5 * (nom["R"]["shelf"][c] + nom["R"]["vp"][c]))
        Lb, rb, yb = chord(nomb["tcam"], 0.5 * (nomb["R"]["shelf"][c] + nomb["R"]["vp"][c]))
        Rj = 0.5 * (MC["joint"]["R_shelf"][:, CI[c]] + MC["joint"]["R_vp"][:, CI[c]])
        Lj, rj, yj = chord(MC["joint"]["tcam"], Rj)
        chords[p] = dict(length_mm=float(L0), length_mm_biascorr=float(Lb), length_sigma_joint=float(rstd(Lj)), drawing_mm=330.0,
                         implied_code_mm=float(90 * 330 / L0), implied_code_mm_biascorr=float(90 * 330 / Lb),
                         chord_roll_deg=float(r0), chord_roll_deg_biascorr=float(rb), chord_roll_sigma_joint=float(rstd(rj)),
                         chord_yaw_deg=float(y0), chord_yaw_sigma_joint=float(rstd(yj)))
    checks["plate_chord"] = chords
    # (5) implied sticker centre in the shelf-sticker cart frame (PnP depth with the 90 mm code) vs the drawing
    pos = {}
    for i, s in enumerate(ST):
        c = s["cart"]
        R, t = nom["R"]["shelf"][c], nom["t_shelf"][c]
        X = R.T @ (nom["tcam"][i] - t)
        Xb = nomb["R"]["shelf"][c].T @ (nomb["tcam"][i] - nomb["t_shelf"][c])
        pj = np.einsum("nji,nj->ni", MC["joint"]["R_shelf"][:, CI[c]], MC["joint"]["tcam"][:, i] - MC["joint"]["t_shelf"][:, CI[c]])
        dpred = (R @ s["centre"] + t)[2]
        pos[s["label"]] = dict(d_mm=(X - s["centre"]).tolist(), d_mm_biascorr=(Xb - s["centre"]).tolist(), sigma_joint_mm=rstd(pj).tolist(),
                               implied_code_mm=float(90 * dpred / nom["tcam"][i][2]))
    checks["implied_centre_in_shelf_frame"] = pos
    checks["rotation_sources_nominal"] = {str(c): {f"{a}_vs_{b}_deg_about_XYZ": np.degrees(cv2.Rodrigues(nom["R"][a][c].T @ nom["R"][b][c])[0].ravel()).tolist()
                                                   for a, b in (("shelf", "vp"), ("main", "vp"))} for c in CARTS}
    checks["shelf_pose_rms_px"] = {str(c): nom["shelf_rms"][c] for c in CARTS}
    checks["vp_rms_px"] = nom["vp_rms"]
    # ---------------- f dependence
    fd = {"refit": [], "pixel": []}
    for mode in ("refit", "pixel"):
        for e in fdep[mode]:
            a = comb(e["ang"])
            row = dict(f=e["f"], stickers={ST[i]["label"]: dict(pitch_outer=float(a[i, 0]), roll=float(a[i, 1]), total=float(a[i, 2])) for i in range(NST)},
                       plates={p: dict(zip(QN, v)) for p, v in plate_combo(a, w).items()})
            if mode == "refit":
                row.update(cx=float(e["cx"]), cy=float(e["cy"]), k1=float(e["k1"]), k2=float(e["k2"]),
                           block_rms={k: v[0] for k, v in e["block_rms"].items()},
                           plates_main_pose={p: dict(zip(QN, v)) for p, v in plate_combo(e["ang"]["main"], w).items()})
            fd[mode].append(row)
    slopes = {}
    for p in PLATES:
        fv = np.array([r["f"] for r in fd["pixel"]])
        pv = np.array([r["plates"][p]["pitch_outer"] for r in fd["pixel"]])
        rv = np.array([r["plates"][p]["roll"] for r in fd["pixel"]])
        sel = (fv >= 1400) & (fv <= 1550)
        slopes[p] = dict(pitch_outer_deg_per_100px=float(100 * np.polyfit(fv[sel], pv[sel], 1)[0]), roll_deg_per_100px=float(100 * np.polyfit(fv[sel], rv[sel], 1)[0]))
    fd["slope_pixel_mode_1400_1550"] = slopes
    # ---------------- alternative lenses
    alt_out = []
    for a in alts:
        av = comb(a["ang"])
        alt_out.append(dict(method_id=a["method_id"], f=a["f"], cx=a["cx"], cy=a["cy"], dist=a["dist"], folds_inside_image=a["folds"],
                            plates={p: dict(zip(QN, v)) for p, v in plate_combo(av, w).items()},
                            shelf_control={ST[i]["label"]: dict(pitch_outer=float(av[i, 0]), roll=float(av[i, 1])) for i in SHELF_FULL}))
    # ---------------- what-if: sticker residuals (main lens, one LS pose per cart) with the MEASURED tilts applied to the top stickers
    checks["whatif_residuals_with_measured_tilt"] = whatif_rms(nomA["comb"], plates)
    return dict(stickers=stick, plates=plates, checks=checks, f_dependence=fd, alt_lenses=alt_out, weights_note="plate = inverse-variance (corner-noise sigma of pitch_outer) weighted mean of its two stickers"), nomA


def tilt_rot(pitch_raw_deg, roll_deg, yaw_deg=0.0):
    """Rotation (cart frame) taking a flat sticker to the measured orientation: normal n ~ (-dZ/dX, -dZ/dY, 1), then yaw about n."""
    n = np.array([-np.tan(np.deg2rad(pitch_raw_deg)), -np.tan(np.deg2rad(roll_deg)), 1.0])
    n /= np.linalg.norm(n)
    ax = np.cross([0, 0, 1.0], n)
    sa = np.linalg.norm(ax)
    R = np.eye(3) if sa < 1e-12 else cv2.Rodrigues(ax / sa * np.arcsin(min(sa, 1.0)))[0]
    return cv2.Rodrigues(n * np.deg2rad(yaw_deg))[0] @ R


def whatif_rms(A, plates):
    """RMS of the 62 sticker corners (main lens, one LS pose per cart, drawing positions) when the top stickers are rotated about
    their own centres by the measured orientation; variants: flat (= lens_result definition), per-sticker tilt (pitch+roll),
    per-sticker tilt + yaw, plate-mean tilt, plate-mean pitch only. Also the corner displacement caused by the tilt (main pose)."""
    variants = {"flat_drawing": None, "per_sticker_pitch_roll": "pr", "per_sticker_pitch_roll_yaw": "pry", "plate_mean_pitch_roll": "plate_pr",
                "plate_mean_pitch_only": "plate_p", "plate_pitch_hinged_at_inner_edge": "hinge_in", "plate_pitch_hinged_at_outer_edge": "hinge_out"}
    pmean = {}
    for p, v in plates.items():
        for lab in v["stickers"]:
            pmean[lab] = v["value"]
    out = {}
    for name, mode in variants.items():
        res, rows, disp = [], [], []
        for c in CARTS:
            X, uv, rw = [], [], []
            for i, s in enumerate(ST):
                if s["cart"] != c:
                    continue
                m = M[s["mi"]]
                C3 = m["corners_3d"].copy()
                if mode is not None and s["row"] == "top":
                    cc = s["centre"].copy()
                    if mode.startswith("plate") or mode.startswith("hinge"):
                        q = pmean[s["label"]]
                        T = tilt_rot(q["dZdX"], q["roll"] if mode == "plate_pr" else 0.0)
                        if mode.startswith("hinge"):  # rigid plate rotated about a Y-parallel edge line (Z=0); plate width 2 x 82.5 mm
                            inner = 165.0 if s["end"] == "X0" else 1435.0
                            outer = 0.0 if s["end"] == "X0" else 1600.0
                            cc[0] = inner if mode == "hinge_in" else outer
                    else:
                        T = tilt_rot(A[i, 4], A[i, 1], A[i, 3] if mode == "pry" else 0.0)
                    C3 = cc + (C3 - cc) @ T.T
                    Rm, tm = rodrigues(POSES_M[CI[c], :3]), POSES_M[CI[c], 3:]
                    disp.append(np.hypot(*(project(C3, KM, DM, R=Rm, tvec=tm) - project(m["corners_3d"], KM, DM, R=Rm, tvec=tm)).T))
                for j in range(4):
                    if s["valid"][j]:
                        X.append(C3[j])
                        uv.append(s["uv"][j])
                        rw.append(s["row"])
            X, uv = np.array(X), np.array(uv)
            r = least_squares(lambda pz: (project(X, KM, DM, rvec=pz[:3], tvec=pz[3:]) - uv).ravel(), POSES_M[CI[c]], method="lm")
            res.append(np.hypot(*r.fun.reshape(-1, 2).T))
            rows += rw
        e = np.concatenate(res)
        rows = np.array(rows)
        out[name] = dict(rms_px=float(np.sqrt(np.mean(e ** 2))), max_px=float(e.max()),
                         per_row={rw: float(np.sqrt(np.mean(e[rows == rw] ** 2))) for rw in ("top", "A", "B", "C")},
                         per_cart={str(c): float(np.sqrt(np.mean(r_ ** 2))) for c, r_ in zip(CARTS, res)})
        if disp:
            dd = np.concatenate(disp)
            out[name]["top_corner_shift_px_rms_max"] = [float(np.sqrt(np.mean(dd ** 2))), float(dd.max())]
    return out


def fmt(v, s, n=1):
    return f"{v:+.{n}f} ± {s:.{n}f}"


def write_md(S, nomA, MC):
    L = []
    L.append("# Measured orientation of the stickers (top vs shelf) - what-if analysis, branch B\n")
    L.append("Script `work/61_top_sticker_orientation.py`; numbers in `work/cache/top_sticker_orientation.json`; plot "
             "`results/top_sticker_orientation.png`. The main result is NOT changed by this analysis.\n")
    L.append("**Method.** Main lens (f 1468.6, pp (928.2, 503.2), k1 -0.3516, k2 0.0994). Every sticker = flat 90 mm square; its pose from its own "
             "4 corners (IPPE_SQUARE on undistorted corners, both solutions LM-refined in the distorted image; the one closer to the cart Z axis kept). "
             "Normal expressed in the cart frame with two cart rotations that do not use the top stickers: (a) LS pose from the shelf-sticker "
             "corners only, (b) vanishing points of the cart-X edges (shelf lips, rails) and cart-Z edges (posts) of that cart. Headline = mean of (a) and (b).\n")
    L.append("**Sign conventions.** `pitch_outer` = tilt about the cart Y axis, **positive = outer end of the plate up** (outer = towards X=0 for the X0 "
             "end, towards X=1600 for the X1600 end); `roll` = tilt about the cart X axis, **positive = back side (Y=450) up**; `total` = angle between the "
             "sticker normal and the cart Z axis; `yaw` = in-plane rotation about +Z vs the nominal rot_k*90 deg. Degrees.\n")
    L.append("**1-sigma.** `total` = joint Monte Carlo (corner noise + 100 lens bootstrap replicates + shelf-sticker / VP-edge resampling, "
             "robust half 16-84 % range) (+) half the difference of the two rotation sources (+) raw-vs-bias-corrected corner difference.\n")
    L.append("## Per sticker (4-corner stickers)\n")
    L.append("| sticker | row | end | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | (a) shelf: pitch / roll | (b) VP: pitch / roll | "
             "sigma noise / lens / rot.src (pitch) | fit RMS px |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in S["stickers"]:
        if e["kind"] != "full":
            continue
        v, t = e["value"], e["sigma_total"]
        L.append(f"| {e['label']} | {e['row']} | {e['end']} | {fmt(v['pitch_outer'], t['pitch_outer'])} | {fmt(v['roll'], t['roll'])} | "
                 f"{v['total']:.1f} ± {t['total']:.1f} | {fmt(v['yaw'], t['yaw'])} | {e['value_shelf']['pitch_outer']:+.2f} / {e['value_shelf']['roll']:+.2f} | "
                 f"{e['value_vp']['pitch_outer']:+.2f} / {e['value_vp']['roll']:+.2f} | {e['sigma_noise']['pitch_outer']:.2f} / "
                 f"{e['sigma_lens']['pitch_outer']:.2f} / {e['sigma_rotsrc']['pitch_outer']:.2f} | {e['fit_rms_px']:.3f} |")
    L.append("\n## Per end plate (two top stickers combined)\n")
    L.append("| plate | stickers (front Y=60, back Y=390) | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | front minus back sticker: pitch / roll (joint sigma) |")
    L.append("|---|---|---|---|---|---|---|")
    for p, e in S["plates"].items():
        v, t = e["value"], e["sigma_total"]
        L.append(f"| {p} | {', '.join(e['stickers'])} | {fmt(v['pitch_outer'], t['pitch_outer'])} | {fmt(v['roll'], t['roll'])} | "
                 f"{v['total']:.1f} ± {t['total']:.1f} | {fmt(v['yaw'], t['yaw'])} | {e['sticker_difference']['pitch_outer']:+.1f} / "
                 f"{e['sticker_difference']['roll']:+.1f} ({e['sticker_difference_sigma_joint']['pitch_outer']:.1f} / "
                 f"{e['sticker_difference_sigma_joint']['roll']:.1f}) |")
    L.append("\n## Control: shelf stickers\n")
    L.append("| sticker | row | end | kind (constraints) | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | usable |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in S["stickers"]:
        if e["row"] == "top":
            continue
        v, t = e["value"], e["sigma_total"]
        L.append(f"| {e['label']} | {e['row']} | {e['end']} | {e['kind']} ({e['n_constraints']}) | {fmt(v['pitch_outer'], t['pitch_outer'])} | "
                 f"{fmt(v['roll'], t['roll'])} | {v['total']:.1f} ± {t['total']:.1f} | {fmt(v['yaw'], t['yaw'])} | {'yes' if e['determined'] else 'no (not determined)'} |")
    L.append("\n'partial' = 2 corners + traced sides (6 constraints, no prior); 'partial_prior' = fewer than 6 independent constraints, "
             "centre tied to the shelf-sticker cart pose (30 mm) -> not independent. The visible parts of the partly hidden stickers "
             "(one or two corners + 25-35 px of side) do NOT determine the sticker normal: their solutions scatter by tens of degrees "
             "(Monte Carlo sigma above) and several have near-equal alternative minima -> only the four 4-corner shelf stickers "
             "(80:4, 80:5, 80:6, 310:6) are used as the control.\n")
    ck = S["checks"]
    L.append("## Sanity checks\n")
    L.append("* Shelf-sticker normals mutually parallel (camera frame, 4-corner stickers, same cart): " +
             "; ".join(f"{q['pair']} {q['angle_deg']:.2f} deg (joint sigma {q['sigma_joint_deg']:.2f})" for q in ck["shelf_normals_pairwise"]))
    L.append("* Top minus 4-corner shelf sticker at the same cart end (cart rotation cancels; lens-only sigma in brackets): " +
             "; ".join(f"{q['top']}-{q['shelf']} pitch {q['d_pitch_outer']:+.1f} ± {q['sigma_joint'][0]:.1f} ({q['sigma_lens_only'][0]:.1f}), "
                       f"roll {q['d_roll']:+.1f} ± {q['sigma_joint'][1]:.1f}" for q in ck["top_minus_shelf_same_end"]))
    L.append("* IPPE ambiguity (4-corner stickers): chosen RMS / other solution RMS [px] and its angle to cart Z: " +
             "; ".join(f"{k} {v['chosen_rms_px']:.3f}/{v.get('rms_px', float('nan')):.3f} ({v.get('angle_to_cartZ_deg', float('nan')):.0f} deg)"
                       for k, v in ck["ippe_other_solution"].items()))
    L.append("* Scale (PnP with the 90 mm code): distance between the two sticker centres of a plate vs the drawing 330 mm: " +
             "; ".join(f"{p} {v['length_mm']:.1f} mm (bias-corr. {v['length_mm_biascorr']:.1f}; implied code {v['implied_code_mm']:.1f} / "
                       f"{v['implied_code_mm_biascorr']:.1f} mm), chord roll {v['chord_roll_deg']:+.1f} ± {v['chord_roll_sigma_joint']:.1f} deg"
                       for p, v in ck["plate_chord"].items()))
    L.append("* Implied sticker centre (PnP depth) in the shelf-sticker cart frame minus the drawing [mm] (dX, dY, dZ; bias-corrected dZ): " +
             "; ".join(f"{k} ({v['d_mm'][0]:+.0f}, {v['d_mm'][1]:+.0f}, {v['d_mm'][2]:+.0f}; {v['d_mm_biascorr'][2]:+.0f})"
                       for k, v in ck["implied_centre_in_shelf_frame"].items() if k.split(":")[1] in ("0", "1", "2", "92", "322", "4", "5", "6")))
    if "synthetic_round_trip" in ck:
        L.append("* Synthetic round trip (main lens + main poses, top stickers tilted by a known angle, corner noise as in the MC, %d reps): " % ck["synthetic_round_trip"][0]["n_noisy_reps"] +
                 "; ".join(f"truth pitch {q['true_pitch_outer']:+.1f} / roll {q['true_roll']:+.1f} -> noise-free top pitch "
                           f"{min(q['noise_free_top_pitch_outer']):+.2f}..{max(q['noise_free_top_pitch_outer']):+.2f}, roll "
                           f"{min(q['noise_free_top_roll']):+.2f}..{max(q['noise_free_top_roll']):+.2f}; noisy mean error "
                           f"{q['noisy_top_mean_error_pitch_roll'][0]:+.2f} / {q['noisy_top_mean_error_pitch_roll'][1]:+.2f}, RMS error "
                           f"{q['noisy_top_rms_error_pitch_roll'][0]:.2f} / {q['noisy_top_rms_error_pitch_roll'][1]:.2f} deg" for q in ck["synthetic_round_trip"]) +
                 " (signs and magnitudes recovered; residual offsets of the VP source come from its ~0.05 deg difference to the main-fit pose).")
    L.append(f"* Cart rotation sources: {ck['rotation_sources_nominal']}; shelf-pose RMS {ck['shelf_pose_rms_px']} px, VP RMS {ck['vp_rms_px']:.3f} px.")
    L.append("* Frame-to-frame (7 stills) std of pitch_outer vs the corner-noise MC sigma: " +
             ", ".join(f"{e['label']} {e['frames_std']['pitch_outer']:.2f}/{e['sigma_noise']['pitch_outer']:.2f}" for e in S["stickers"] if e["kind"] == "full"))
    L.append("  (the corner-noise MC uses the single-still corner std per coordinate, i.e. it is conservative: the 7-still mean scatters "
             "~sqrt(7) less than one still; detection systematics are covered by the raw-vs-bias-corrected term)")
    L.append("* 310/X0 has no 4-corner shelf sticker at its end (310:3, 310:5, 310:7 are partly hidden), so its tilt rests on the cart rotation "
             "(shelf pose / VPs) only; 80/X0 is additionally checked against 80:5 (same image region).")
    L.append("\n## Dependence on f\n")
    L.append("Plate pitch_outer / roll [deg]; (i) cx, cy, k1, k2 and the poses refitted by the combined estimator at fixed f; (ii) distortion fixed in pixel units.\n")
    hdr = "| mode | f | " + " | ".join(S["plates"]) + " |"
    L.append(hdr)
    L.append("|---|---|" + "---|" * len(S["plates"]))
    for mode in ("refit", "pixel"):
        for r in S["f_dependence"][mode]:
            if mode == "pixel" and r["f"] not in (1420.0, 1520.0) and abs(r["f"] - 1468.63) > 0.1 and r["f"] not in (1300.0, 1400.0, 1600.0):
                continue
            extra = f" (pp {r['cx']:.0f},{r['cy']:.0f}; k1 {r['k1']:.3f})" if mode == "refit" else ""
            L.append(f"| {mode}{extra} | {r['f']:.0f} | " + " | ".join(f"{r['plates'][p]['pitch_outer']:+.1f} / {r['plates'][p]['roll']:+.1f}" for p in S["plates"]) + " |")
    L.append("\nSlope (pixel mode, 1400-1550): " + "; ".join(f"{p} pitch {v['pitch_outer_deg_per_100px']:+.2f}, roll {v['roll_deg_per_100px']:+.2f} deg / 100 px"
                                                          for p, v in S["f_dependence"]["slope_pixel_mode_1400_1550"].items()))
    L.append("\n## Other lens models (nominal data; plate pitch_outer / roll [deg]; control = mean |pitch|, |roll| of the 4-corner shelf stickers)\n")
    L.append("| lens | f | pp | k1, k2 | " + " | ".join(S["plates"]) + " | control |")
    L.append("|---|---|---|---|" + "---|" * (len(S["plates"]) + 1))
    main_row = {p: v["value"] for p, v in S["plates"].items()}
    L.append(f"| main (lens_result.json) | {S['meta_lens_f']:.0f} | (928, 503) | -0.352, 0.099 | " +
             " | ".join(f"{main_row[p]['pitch_outer']:+.1f} / {main_row[p]['roll']:+.1f}" for p in S["plates"]) + " | - |")
    for a in S["alt_lenses"]:
        ctl = np.array([[abs(v["pitch_outer"]), abs(v["roll"])] for v in a["shelf_control"].values()]).mean(0)
        L.append(f"| {a['method_id']}{' (folds)' if a['folds_inside_image'] else ''} | {a['f']:.0f} | ({a['cx']:.0f}, {a['cy']:.0f}) | "
                 f"{a['dist'][0]:.3f}, {a['dist'][1]:.3f} | " + " | ".join(f"{a['plates'][p]['pitch_outer']:+.1f} / {a['plates'][p]['roll']:+.1f}" for p in S["plates"]) +
                 f" | {ctl[0]:.1f} / {ctl[1]:.1f} |")
    wi = S["checks"]["whatif_residuals_with_measured_tilt"]
    L.append("\n## What-if: sticker residuals when the top stickers carry the measured tilt\n")
    L.append("Main lens fixed, one least-squares pose per cart, drawing positions; top stickers rotated about their own centres by the "
             "measured orientation (this is NOT a refit of the lens).\n")
    L.append("| variant | RMS all [px] | max | top | A | B | C | top-corner shift by the tilt (RMS / max px) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in wi.items():
        sh_ = v.get("top_corner_shift_px_rms_max")
        L.append(f"| {k} | {v['rms_px']:.2f} | {v['max_px']:.2f} | {v['per_row']['top']:.2f} | {v['per_row']['A']:.2f} | {v['per_row']['B']:.2f} | "
                 f"{v['per_row']['C']:.2f} | {'-' if sh_ is None else f'{sh_[0]:.2f} / {sh_[1]:.2f}'} |")
    L.append("\n## Conclusion\n")
    L.append(S["conclusion"])
    return "\n".join(L) + "\n"


def plot(S):
    C80, C310 = "#2a78d6", "#eb6834"
    INK, INK2, GRID, BG, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f1f0ec"
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9})
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=110, gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0]))
    fig.patch.set_facecolor(BG)
    use = [i for i, e in enumerate(S["stickers"]) if e["determined"]]
    order = sorted(use, key=lambda i: (S["stickers"][i]["row"] != "top", S["stickers"][i]["cart"], S["stickers"][i]["end"], S["stickers"][i]["centre_y"]))
    ntop = sum(1 for i in order if S["stickers"][i]["row"] == "top")
    labels = [f"{S['stickers'][i]['label']}  {S['stickers'][i]['row']} {S['stickers'][i]['end']}" for i in order]
    titles = {"pitch_outer": "pitch about cart Y [deg]  (+ = outer end up)", "roll": "roll about cart X [deg]  (+ = back side up)"}
    for k, q in enumerate(("pitch_outer", "roll")):
        ax = axs[k]
        ax.set_facecolor(BG)
        ax.axvspan(-0.5, ntop - 0.5, color=BAND, zorder=0)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        lo, hi = 0.0, 0.0
        for pos, i in enumerate(order):
            e = S["stickers"][i]
            col = C80 if e["cart"] == 80 else C310
            mk = "o" if e["row"] == "top" else "s"
            v, sg = e["value"][q], e["sigma_total"][q]
            ax.errorbar(pos, v, yerr=sg, fmt=mk, color=col, mec="white", mew=1.0, ms=8, lw=1.6, capsize=3, zorder=3)
            lo, hi = min(lo, v - sg), max(hi, v + sg)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 2.2 * pad)
        ax.text((ntop - 1) / 2, 0.975, "top stickers (end plates)", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2)
        ax.text(ntop + (len(order) - ntop - 1) / 2, 0.975, "shelf stickers\n(control)", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(titles[q], color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C80, marker="o", ls="", ms=8, label="cart 80"), Line2D([], [], color=C310, marker="o", ls="", ms=8, label="cart 310"),
         Line2D([], [], color=INK2, marker="o", ls="", ms=7, label="top sticker"), Line2D([], [], color=INK2, marker="s", ls="", ms=7, label="shelf sticker")]
    axs[1].legend(handles=h, loc="lower right", fontsize=8, frameon=False, ncol=2)
    ax = axs[2]
    ax.set_facecolor(BG)
    pm = S["f_dependence"]["pixel"]
    rf = S["f_dependence"]["refit"]
    fv = [r["f"] for r in pm]
    styles = {"80/X0": (C80, "-"), "80/X1600": (C80, "--"), "310/X0": (C310, "-"), "310/X1600": (C310, "--")}
    f0, sf = S["meta_lens_f"], S["meta_lens_sf"]
    ax.axvspan(f0 - sf, f0 + sf, color=BAND, zorder=0)
    ax.axvline(f0, color=INK2, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    for p, (col, ls) in styles.items():
        ax.plot(fv, [r["plates"][p]["pitch_outer"] for r in pm], ls, color=col, lw=2, zorder=2)
        ax.plot([r["f"] for r in rf], [r["plates"][p]["pitch_outer"] for r in rf], "o", color=col, ms=8, mec="white", mew=1.0, zorder=3)
    ends = sorted(((pm[-1]["plates"][p]["pitch_outer"], p) for p in styles))
    ypos = [ends[0][0]]
    for v, _ in ends[1:]:
        ypos.append(max(v, ypos[-1] + 0.9))
    for (v, p), y in zip(ends, ypos):
        ax.text(fv[-1] + 6, y, p, color=INK, fontsize=8.5, va="center")
    ax.set_xlim(fv[0] - 10, fv[-1] + 75)
    ax.set_xlabel("f [px]   (band: main f ± 1 sigma)", color=INK)
    ax.set_ylabel("plate pitch about cart Y [deg]  (+ = outer end up)", color=INK)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h2 = [Line2D([], [], color=INK2, ls="-", lw=2, label="distortion fixed in pixel units (pp fixed)"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=7, label="pp, k1, k2, poses refitted at fixed f")]
    ax.legend(handles=h2, loc="center left", fontsize=8, frameon=False)
    fig.suptitle("Measured sticker orientation in the cart frame (main lens; pose of each sticker from its own 4 corners); "
                 "error bars = total 1-sigma", color=INK)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/top_sticker_orientation.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def conclusion(S):
    pl = S["plates"]
    sh = [e for e in S["stickers"] if e["row"] != "top" and e["determined"]]
    txt = []
    rows = []
    for p, v in pl.items():
        z = v["value"]["pitch_outer"] / v["sigma_total"]["pitch_outer"]
        zr = v["value"]["roll"] / v["sigma_total"]["roll"]
        fr = [r["plates"][p]["pitch_outer"] for r in S["f_dependence"]["refit"]]
        verdict = "tilted (measurable)" if abs(z) >= 3 else ("marginal" if abs(z) >= 2 else "not measurably tilted")
        rows.append(f"- **{p}**: pitch {fmt(v['value']['pitch_outer'], v['sigma_total']['pitch_outer'])} deg ({z:+.1f} sigma, "
                    f"f 1420..1520: {min(fr):+.1f}..{max(fr):+.1f}), roll {fmt(v['value']['roll'], v['sigma_total']['roll'])} deg ({zr:+.1f} sigma), "
                    f"total {v['value']['total']:.1f} deg -> {verdict}")
    sig_p = [p for p, v in pl.items() if abs(v["value"]["pitch_outer"]) >= 3 * v["sigma_total"]["pitch_outer"]]
    non_p = [p for p in pl if p not in sig_p]
    sig_r = [p for p, v in pl.items() if abs(v["value"]["roll"]) >= 3 * v["sigma_total"]["roll"]]
    txt.append("**Answer:** " + ("yes - " if sig_p else "no - ") + "the top stickers are measurably tilted about the cart Y axis (dZ/dX, along the "
               "cart length), outer end up, at " + ", ".join(f"{p} {fmt(pl[p]['value']['pitch_outer'], pl[p]['sigma_total']['pitch_outer'])} deg" for p in sig_p) +
               ("; not significant at " + ", ".join(f"{p} {fmt(pl[p]['value']['pitch_outer'], pl[p]['sigma_total']['pitch_outer'])} deg" for p in non_p) if non_p else "") +
               ". About the cart X axis " + ("no plate is significantly tilted (plate roll " +
                                             ", ".join(f"{fmt(v['value']['roll'], v['sigma_total']['roll'])}" for v in pl.values()) + " deg)" if not sig_r else
                                             "significant roll at " + ", ".join(sig_r)) + ".")
    txt.append("Per end plate (pitch_outer: + = outer end up; roll: + = back side up):\n" + "\n".join(rows))
    shp = np.array([e["value"]["pitch_outer"] for e in sh])
    shr = np.array([e["value"]["roll"] for e in sh])
    txt.append("Control (4-corner shelf stickers 80:4, 80:5, 80:6, 310:6): pitch_outer " +
               ", ".join(f"{e['label']} {fmt(e['value']['pitch_outer'], e['sigma_total']['pitch_outer'])}" for e in sh) +
               "; roll " + ", ".join(f"{e['label']} {fmt(e['value']['roll'], e['sigma_total']['roll'])}" for e in sh) +
               f" deg (mean pitch {shp.mean():+.1f}, mean roll {shr.mean():+.1f}); max |value|/sigma = "
               f"{max(abs(e['value'][q]) / e['sigma_total'][q] for e in sh for q in ('pitch_outer', 'roll')):.1f} -> the method returns "
               "horizontal for the shelf stickers within about 1-1.5 sigma.")
    diffs = S["checks"]["top_minus_shelf_same_end"]
    txt.append("Lens-independent form (top minus 4-corner shelf sticker at the same cart end, cart rotation cancels): " +
               "; ".join(f"{q['top']}-{q['shelf']} {q['d_pitch_outer']:+.1f} ± {q['sigma_joint'][0]:.1f} deg (lens part {q['sigma_lens_only'][0]:.1f})" for q in diffs) + ".")
    dr = [v["sticker_difference"]["roll"] for v in pl.values()]
    dsr = [v["sticker_difference_sigma_joint"]["roll"] for v in pl.values()]
    txt.append("Within each plate the front (Y=60) and back (Y=390) sticker differ in roll by " +
               ", ".join(f"{p} {v['sticker_difference']['roll']:+.1f} ± {v['sticker_difference_sigma_joint']['roll']:.1f}" for p, v in pl.items()) +
               " deg (always front more 'back side up' than back) - consistent with plates slightly bowed up across Y (ridge between the "
               "stickers) of ~%.0f-%.0f deg per sticker; the plate means of the roll are small." % (min(dr) / 2, max(dr) / 2) if all(d > 0 for d in dr) else
               "Within-plate roll differences: " + ", ".join(f"{d:+.1f} ± {s_:.1f}" for d, s_ in zip(dr, dsr)))
    al = S.get("alt_lenses", [])
    if al:
        def rng_(lst, p):
            v = [a["plates"][p]["pitch_outer"] for a in lst]
            return f"{min(v):+.1f}..{max(v):+.1f}"
        cred = [a for a in al if not a["method_id"].startswith("markers_only")]
        txt.append(f"Other lens models ({len(al)}; nominal data): plate pitch_outer range " + "; ".join(f"{p} {rng_(al, p)}" for p in pl) +
                   f". Without the two sticker-only drawing-geometry lenses (markers_only*, f 1326-1358, which absorb the geometry mismatch): " +
                   "; ".join(f"{p} {rng_(cred, p)}" for p in pl) + ".")
    wi = S["checks"].get("whatif_residuals_with_measured_tilt")
    if wi:
        f0, t0 = wi["flat_drawing"], wi["per_sticker_pitch_roll"]
        hi, ho = wi["plate_pitch_hinged_at_inner_edge"], wi["plate_pitch_hinged_at_outer_edge"]
        txt.append(f"What-if (main lens, one LS pose per cart): tilting the top stickers about their own centres by the measured angles lowers the "
                   f"62-corner RMS only from {f0['rms_px']:.2f} to {t0['rms_px']:.2f} px (top row {f0['per_row']['top']:.2f} -> {t0['per_row']['top']:.2f}); "
                   f"the tilt moves the top corners by {t0['top_corner_shift_px_rms_max'][0]:.1f} px RMS (max {t0['top_corner_shift_px_rms_max'][1]:.1f}). "
                   f"A rigid plate tilted about its INNER edge (sticker centres lifted by 82.5 mm x tan(pitch), ~20 mm at the X0 ends) gives "
                   f"{hi['rms_px']:.2f} px (rows top/A/B/C {hi['per_row']['top']:.2f}/{hi['per_row']['A']:.2f}/{hi['per_row']['B']:.2f}/{hi['per_row']['C']:.2f}), "
                   f"about the OUTER edge {ho['rms_px']:.2f} px. The measured tilt is real but explains only a small part of the sticker misfit; "
                   "together with a lifted outer end it points the same way as the known 'top stickers too high' mismatch (REPORT ch. 5), "
                   "which needs ~60-80 mm, not ~20 mm.")
    return "\n\n".join(txt)


if __name__ == "__main__" and REPLOT:
    S = load_json(f"{CACHE}/top_sticker_orientation.json")
    S["conclusion"] = conclusion(S)
    save_json(S, f"{CACHE}/top_sticker_orientation.json")
    open(f"{CACHE}/top_sticker_orientation.md", "w").write(write_md(S, None, None))
    plot(S)
    print(S["conclusion"])
elif __name__ == "__main__":
    nom, nomb, MC, fdep, alts, synth = main()
    S, nomA = summarise(nom, nomb, MC, fdep, alts)
    S["checks"]["synthetic_round_trip"] = synth
    S["meta_lens_f"], S["meta_lens_sf"] = float(KM[0, 0]), float(LR["uncertainty"]["fx_px_1sigma"])
    S["conclusion"] = conclusion(S)
    S["meta"] = dict(lens=dict(f=float(KM[0, 0]), cx=float(KM[0, 2]), cy=float(KM[1, 2]), k1=float(DM[0]), k2=float(DM[1])),
                     n_mc={k: int(len(v["shelf"])) for k, v in MC.items()}, sign_conventions=dict(
                         pitch_outer="tilt about the cart Y axis; positive = outer end up (outer = towards X=0 at the X0 end, towards X=1600 at the X1600 end)",
                         roll="tilt about the cart X axis; positive = back side (Y=450) up", total="angle between sticker normal and cart Z",
                         yaw="in-plane rotation about +Z relative to rot_k*90 deg", dZdX="atan(dZ/dX) in the cart frame (not end-dependent)"),
                     cart_frame="X along the 1600 mm front, Y into the cart (0 front .. 450 back), Z up, Z=0 top plate",
                     rotation_sources=dict(shelf="LS pose of the shelf-sticker corners (drawing geometry, main lens)",
                                           vp="VP fit of cart-X (lips, rails) and cart-Z (posts) edges of the cart (no stickers)",
                                           main="pose of the main combined fit (includes the top stickers; reference only)",
                                           headline="mean of shelf and vp"),
                     sigma="robust 1-sigma = half of the 16-84 % range of the Monte Carlo; total = joint MC (+) |shelf-vp|/2 (+) |raw - bias-corrected corners|",
                     runtime_s=time.time() - T0)
    save_json(S, f"{CACHE}/top_sticker_orientation.json")
    open(f"{CACHE}/top_sticker_orientation.md", "w").write(write_md(S, nomA, MC))
    plot(S)
    print(S["conclusion"])
    print(f"done {time.time() - T0:.0f} s")
