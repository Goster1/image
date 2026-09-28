"""Synthetic study, step C: sensitivity of the orchestrator estimators to small systematic detection errors.

Datasets: noise-free synthetic data of the true lenses T1, T2 (exact real geometry / visibility) and the REAL data.
Every perturbation is applied to the observations, all estimators are re-run with the same starts, and the
change of the parameters and of the image mapping (rotation compensated, evaltools; per region) relative to
the unperturbed estimate is recorded (and scaled to 0.1 px of perturbation where the perturbation has a size).

Corner perturbations (sticker corners + traced sides of partly hidden stickers):
  dilate / erode the black square by 0.3 px (every side moved along its normal), all corners +0.2 px in x / y,
  rolling-shutter-like y scale 5e-4 about y = 540, ArUco instead of edge corners, bias-corrected corners
  (markers_final corners_px_bias_corrected), each of the 7 single frames instead of the 7-frame mean.
Edge perturbations (structural edges): +-0.3 px along the normal towards the dark side (edge-detector bias; polarity
  measured on the mean image), 0.3 px away from the image centre, rolling-shutter y scale on everything,
  subsampling 1 / 4 instead of 2 (real data + noisy synthetic replicates only - noise-free lines are unaffected).
Writes work/cache/synthetic_sensitivity.json
"""
import importlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE, K_from, load_json, project, save_json

L = importlib.import_module("44_synth_lib")
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SETUP = load_json(os.environ.get("SYNTH_SETUP", f"{CACHE}/synthetic_setup.json"))
M, E, S = L.load_real()
NE = len(E)
MF = load_json(f"{CACHE}/markers_final.json")
MFD = {(int(m["cart"]), int(m["id"])): m for m in MF}
init = load_json(f"{CACHE}/initial_calib.json")
K_INIT, D_INIT = np.array(init["K"]), np.array(init["dist"])
P_INIT = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
IMG = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(float)
F_REAL = 1400.0  # focal length used to express the plumb-line distortion of the real data as a mapping


class _T:
    def __init__(self, name, t):
        self.name = name
        self.K = K_from(t["f"], t["f"], t["cx"], t["cy"])
        self.d = np.array([t["k1"], t["k2"], 0, 0, 0.0])
        self.poses = {int(c): np.array(v) for c, v in t["poses"].items()}
        self.wv = np.array(t["wv"])
        self.floor = {g: np.array(v) for g, v in t["floor"].items()}
        self.params = {k: t[k] for k in ("f", "cx", "cy", "k1", "k2")}


TR = {k: _T(k, v) for k, v in SETUP["truths"].items()}


def arr(v):
    return np.array([[np.nan, np.nan] if q is None else q for q in v], float)


# ------------------------------------------------------------------ polarity of the real edges
def dark_sign(P):
    n = L.curve_normals(P)
    a = map_coordinates(IMG, [(P + 2.0 * n)[:, 1], (P + 2.0 * n)[:, 0]], order=1)
    b = map_coordinates(IMG, [(P - 2.0 * n)[:, 1], (P - 2.0 * n)[:, 0]], order=1)
    return 1.0 if np.median(a - b) < 0 else -1.0  # +1: dark side along +normal


POL = {e["id"]: dark_sign(e["points"]) for e in E}


# ------------------------------------------------------------------ datasets
def dataset(kind):
    """corners: list of 4x2 (all four corners; invalid = completion), valid masks from markers_final,
    sides: sticker sides (L lines), edges: structural edges."""
    if kind == "real":
        corners = [m["corners_px"].copy() for m in M]
        return dict(kind=kind, corners=corners, sides=[dict(s) for s in S], edges=[dict(e) for e in E])
    T = TR[kind]
    corners = [project(m["corners_3d"], T.K, T.d, rvec=T.poses[m["cart"]][:3], tvec=T.poses[m["cart"]][3:]) for m in M]
    ide = L.ideal_edges(T, E, S)
    return dict(kind=kind, corners=corners, sides=ide[NE:], edges=ide[:NE])


def points_of(ds):
    pts = []
    for mi, m in enumerate(M):
        for j in range(4):
            if m["valid"][j]:
                pts.append(dict(cart=m["cart"], id=m["id"], mi=mi, corner=j, X=m["corners_3d"][j], uv=ds["corners"][mi][j]))
    return pts


def quad_dilate(Q, delta):
    c = Q.mean(0)
    lines = []
    for j in range(4):
        a, b = Q[j], Q[(j + 1) % 4]
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        if ((a + b) / 2 - c) @ n < 0:
            n = -n
        lines.append((n, n @ a + delta))
    out = np.zeros_like(Q)
    for j in range(4):
        n1, c1 = lines[(j - 1) % 4]
        n2, c2 = lines[j]
        out[j] = np.linalg.solve(np.array([n1, n2]), np.array([c1, c2]))
    return out


def owner_centre(ds, e):
    c, i = L.side_owner(e)
    mi = [k for k, m in enumerate(M) if m["cart"] == c and m["id"] == i][0]
    return ds["corners"][mi].mean(0)


def perturb(ds, name, val):
    d2 = dict(ds, corners=[q.copy() for q in ds["corners"]], sides=[dict(s) for s in ds["sides"]], edges=[dict(e) for e in ds["edges"]])
    if name == "dilate":
        d2["corners"] = [quad_dilate(q, val) for q in ds["corners"]]
        for s in d2["sides"]:
            P = s["points"]
            n = L.curve_normals(P)
            sg = np.sign(((P - owner_centre(ds, s)) * n).sum(1))
            s["points"] = P + val * sg[:, None] * n
    elif name in ("shift_x", "shift_y"):
        v = np.array([val, 0.0]) if name == "shift_x" else np.array([0.0, val])
        d2["corners"] = [q + v for q in ds["corners"]]
        for s in d2["sides"]:
            s["points"] = s["points"] + v
    elif name in ("yscale_corners", "yscale_all"):
        def ys(P):
            P = P.copy()
            P[:, 1] += val * (P[:, 1] - 540.0)
            return P
        d2["corners"] = [ys(q) for q in ds["corners"]]
        for s in d2["sides"]:
            s["points"] = ys(s["points"])
        if name == "yscale_all":
            for e in d2["edges"]:
                e["points"] = ys(e["points"])
    elif name in ("aruco", "bias_corrected") or name.startswith("frame"):
        for mi, m in enumerate(M):
            mf = MFD[(m["cart"], m["id"])]
            base = arr(mf["corners_px"])
            if name == "aruco":
                alt = arr(mf["aruco_corners_px"]) if mf.get("aruco_corners_px") is not None else base
                ok = np.array(mf.get("aruco_corner_valid") or [False] * 4, bool)
            elif name == "bias_corrected":
                alt = arr(mf["corners_px_bias_corrected"])
                ok = np.ones(4, bool)
            else:
                f = int(name[5:])
                alt = arr(mf["per_frame"][f])
                ok = np.ones(4, bool)
            dlt = alt - base
            ok &= np.isfinite(dlt).all(1)
            dlt[~ok] = 0.0
            d2["corners"][mi] = ds["corners"][mi] + dlt
    elif name == "edge_dark":
        for e in d2["edges"]:
            P = e["points"]
            e["points"] = P + val * POL[e["id"]] * L.curve_normals(P)
    elif name == "edge_outward":
        for e in d2["edges"]:
            P = e["points"]
            n = L.curve_normals(P)
            sg = np.sign(((P - L.CEN) * n).sum(1))
            e["points"] = P + val * sg[:, None] * n
    else:
        raise ValueError(name)
    return d2


# ------------------------------------------------------------------ estimators
def starts(kind):
    if kind == "real":
        return dict(mk=(K_INIT, D_INIT, P_INIT), ln=(K_from(1300, 1300, *L.CEN), np.array([-0.28, 0.05, 0, 0, 0]), P_INIT),
                    cb=(K_from(1420, 1420, *L.CEN), np.array([-0.33, 0.09, 0, 0, 0]), P_INIT), f_ref=F_REAL)
    T = TR[kind]
    return dict(mk=(T.K, T.d, T.poses), ln=(T.K, T.d, T.poses), cb=(T.K, T.d, T.poses), f_ref=T.params["f"])


def lens(names, x, f_ref, kind, dpx=None):
    p = dict(zip(names, x))
    if kind == "plumb":
        return K_from(f_ref, f_ref, p.get("cx", L.CEN[0]), p.get("cy", L.CEN[1])), np.array([p["k1"] * (f_ref / L.F0) ** 2, p["k2"] * (f_ref / L.F0) ** 4, 0, 0, 0])
    f = p["f"]
    if kind == "vp":
        d = np.array([dpx[0] * f ** 2, dpx[1] * f ** 4, 0, 0, 0])
    else:
        d = np.array([p.get("k1", 0.0), p.get("k2", 0.0), 0, 0, 0])
    return K_from(f, f, p.get("cx", L.CEN[0]), p.get("cy", L.CEN[1])), d


def estimate(ds, which=("markers", "lines", "combined"), subsample=2):
    st = starts(ds["kind"])
    out = {}
    pts = points_of(ds)
    if "markers" in which:
        for name, spec in L.MARKER_SPECS.items():
            r = L.marker_estimate(spec, pts, *st["mk"])
            K, d = lens(r["names"], r["x"], None, "std")
            out["markers_" + name] = dict(names=r["names"], x=r["x"], K=K, d=d, rms=r["rms"])
    if "lines" in which:
        ed = ds["edges"]
        lcp, rp, _, _ = L.plumb_estimate(ed, subsample=subsample)
        out["plumb_fixed"] = dict(names=["k1", "k2"], x=rp.x[:2], **dict(zip(("K", "d"), lens(["k1", "k2"], rp.x[:2], st["f_ref"], "plumb"))))
        lcf, rf, _, _ = L.plumb_estimate(ed, centre_free=True, subsample=subsample)
        out["plumb_free"] = dict(names=["cx", "cy", "k1", "k2"], x=rf.x[:4], **dict(zip(("K", "d"), lens(["cx", "cy", "k1", "k2"], rf.x[:4], st["f_ref"], "plumb"))))
        dpx = np.array([rp.x[0] / L.F0 ** 2, rp.x[1] / L.F0 ** 4])
        lcv, rv, _, _ = L.vp_estimate(ed, dpx, st["ln"][0], st["ln"][2], subsample=subsample)
        out["vp"] = dict(names=["f", "cx", "cy"], x=rv.x[:3], **dict(zip(("K", "d"), lens(["f", "cx", "cy"], rv.x[:3], None, "vp", dpx))))
        lcj, rj, _, _ = L.joint_estimate(ed, *st["ln"], subsample=subsample)
        nm = lcj.names[: lcj.n_intr]
        out["lines_joint"] = dict(names=nm, x=rj.x[: lcj.n_intr], **dict(zip(("K", "d"), lens(nm, rj.x[: lcj.n_intr], None, "std"))))
    if "combined" in which:
        cb, rc, _, K, d = L.combined_estimate(pts, ds["edges"] + ds["sides"], *st["cb"], subsample=subsample)
        out["combined_k1k2"] = dict(names=cb.inames, x=rc.x[: cb.ni], K=K, d=d, block_rms=cb.block_rms(rc.x))
    return out


def compare(base, pert, scale):
    """scale: perturbation size / 0.1 px (None -> no normalisation)."""
    res = {}
    for k, b in base.items():
        if k not in pert:
            continue
        p = pert[k]
        dx = p["x"] - b["x"]
        D = L.disp_field(b["K"], b["d"], p["K"], p["d"])
        mag = np.linalg.norm(D, axis=1)
        reg = L.region_summary(mag)
        res[k] = dict(names=b["names"], dparams=dx.tolist(), map={r: v["median"] for r, v in reg.items()},
                      map_max={r: v["max"] for r, v in reg.items()})
        if scale:
            res[k]["dparams_per_0p1px"] = (dx / scale).tolist()
            res[k]["map_per_0p1px"] = {r: (v["median"] / scale if v["median"] is not None else None) for r, v in reg.items()}
    return res


PERTS = [
    ("dilate", +0.3, ("markers", "combined")), ("dilate", -0.3, ("markers", "combined")),
    ("shift_x", 0.2, ("markers", "combined")), ("shift_y", 0.2, ("markers", "combined")),
    ("yscale_corners", 5e-4, ("markers", "combined")), ("yscale_all", 5e-4, ("markers", "lines", "combined")),
    ("aruco", None, ("markers", "combined")), ("bias_corrected", None, ("markers", "combined")),
] + [(f"frame{f}", None, ("markers", "combined")) for f in range(7)] + [
    ("edge_dark", +0.3, ("lines", "combined")), ("edge_dark", -0.3, ("lines", "combined")), ("edge_outward", +0.3, ("lines", "combined")),
]


def pert_size(name, val):
    if val is None:
        return None
    if name.startswith("yscale"):
        return abs(val) * 540.0 / 0.1  # max shift (image top / bottom) in units of 0.1 px
    return abs(val) / 0.1


_BASE = {}


def run_one(task):
    kind, name, val, which = task
    t0 = time.time()
    ds = dataset(kind)
    if (kind, which) not in _BASE:
        _BASE[(kind, which)] = estimate(ds, which)
    base = _BASE[(kind, which)]
    pds = perturb(ds, name, val)
    pert = estimate(pds, which)
    # size of the corner perturbation actually applied (valid corners)
    sz = None
    if which != ("lines", "combined"):
        dd = np.concatenate([(pds["corners"][i] - ds["corners"][i])[m["valid"]] for i, m in enumerate(M)])
        sz = dict(rms_px=float(np.sqrt(np.mean(np.sum(dd ** 2, 1)))), mean_px=np.mean(dd, 0).tolist())
    return kind, f"{name}{'' if val is None else f'_{val:+g}'}", dict(result=compare(base, pert, pert_size(name, val)), perturbation_size=sz,
                                                                      base={k: dict(names=v["names"], x=v["x"].tolist()) for k, v in base.items()}), time.time() - t0


def run_subsample(task):
    kind, sub, rep = task
    t0 = time.time()
    if kind == "real":
        ds = dataset("real")
    else:  # noisy synthetic replicate of T2 (scenario b noise: realistic edges)
        T = TR["T2"]
        g = np.random.default_rng(5000 + rep)
        off = L.geometry_offsets(M, g, SETUP["geometry_noise"]["sigma_sticker_mm"], SETUP["geometry_noise"]["sigma_row_mm"])
        ide = L.ideal_edges(T, E, S, side_offsets=off)
        ed = L.noisy_edges(ide, SETUP["edge_stats"], g, bows=True, directions=True)
        pts = L.sticker_obs(T, M, offsets=off, rng=g, corner_sigma=[np.array(c) for c in SETUP["corner_sigma_per_coord"]])
        corners = [m["corners_px"].copy() for m in M]
        k = 0
        for mi, m in enumerate(M):
            for j in range(4):
                if m["valid"][j]:
                    corners[mi][j] = pts[k]["uv"]
                    k += 1
        ds = dict(kind="T2", corners=corners, sides=ed[NE:], edges=ed[:NE])
    base = estimate(ds, ("lines", "combined"), subsample=2)
    pert = estimate(ds, ("lines", "combined"), subsample=sub)
    return kind, f"subsample_{sub}", rep, compare(base, pert, None), time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(kind, n, v, w) for kind in ("T2", "real") for (n, v, w) in PERTS]
    out = {"T2": {}, "real": {}}
    with Pool(NW) as pool:
        for i, (kind, key, res, dt) in enumerate(pool.imap_unordered(run_one, tasks)):
            out[kind][key] = res
            print(f"{i + 1}/{len(tasks)} {kind} {key} {dt:.0f}s", flush=True)
        sub_tasks = [("real", s, 0) for s in (1, 4)] + [("T2b", s, r) for s in (4,) for r in range(8)]
        subs = {}
        for i, (kind, key, rep, res, dt) in enumerate(pool.imap_unordered(run_subsample, sub_tasks)):
            subs.setdefault(kind, {}).setdefault(key, {})[rep] = res
            print(f"sub {kind} {key} rep {rep} {dt:.0f}s", flush=True)
    out["subsample"] = subs
    out["polarity_dark_sign"] = POL
    out["notes"] = dict(
        per_0p1px="parameter / mapping change divided by (perturbation size / 0.1 px); y-scale: size = max shift 5e-4*540 = 0.27 px",
        mapping="rotation-compensated displacement between the unperturbed and the perturbed estimate (evaltools.mapping_displacement), "
                "median / max of |displacement| over the region's grid points (grid 40 px); plumb-line estimates are expressed as a "
                f"mapping at f = f_true (synthetic) or f = {F_REAL:.0f} (real)",
        starts="same start for the unperturbed and the perturbed run (real: the orchestrator's starts; synthetic: the truth)",
        review_status=SETUP["review_status"])
    save_json(out, f"{CACHE}/synthetic_sensitivity.json")
    print("done", round(time.time() - t0), "s")
