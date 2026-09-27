"""Sub-task 43 (lens model choice), step 1: fit many lens models to the SAME data and cross-validate.

Data sets (all 7-frame-mean measurements, drawing geometry exact where used):
  M  : stickers only  - 62 valid sticker corners (20 stickers), reprojection error
  E  : edges only     - verified straight edges: straightness (S) + vanishing-point groups (V: cart X/Y/Z
                        axes of each cart without the end-board edges, scene verticals with a free direction)
  ME : both jointly   - M + V + S (same edge roles as E)
Block weights: variance components of the reference model (Brown k1,k2, pp free) per data set, then FIXED
for all models (same objective for every model -> comparable chi^2).

Models: Brown (k1 / k1k2 / k1k2k3 / +p1p2 / fx!=fy / pp fixed), OpenCV rational (k4 / k4..k6),
Kannala-Brandt (k1k2 / k1..k4), division (l1 / l1l2).

Validation:
  * residuals per block, AIC / BIC in several honest variants (see 43_altmodels_report.py)
  * leave-one-sticker-out (M, ME): refit without the sticker, predict its corners
  * leave-one-edge-region-out (E, ME): source regions (cart310 / cart80 / scene) and 3x3 image tiles;
    predict the held-out edges' straightness (own line) and VP consistency (group direction from training)
  * cross-data prediction: lens from E -> sticker poses refit -> sticker RMS; lens from M -> edges
Writes work/cache/altmodels_fits.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, save_json  # noqa: E402

L = importlib.import_module("43_altmodels_lib")
NPROC = int(os.environ.get("NPROC", "3"))
QUICK = "--quick" in sys.argv
DUPLICATES = ["cart310_floor_tape_blue_upper", "cart310_floor_tape_blue_lower"]  # == scene_blueL_top / _bottom

MODELS = {
    "B_k1": dict(kind="brown", coef=["k1"]),
    "B_k1k2": dict(kind="brown", coef=["k1", "k2"]),
    "B_k1k2k3": dict(kind="brown", coef=["k1", "k2", "k3"]),
    "B_k1k2p1p2": dict(kind="brown", coef=["k1", "k2", "p1", "p2"]),
    "B_k1k2_fxfy": dict(kind="brown", coef=["k1", "k2"], f="fxfy"),
    "B_k1k2k3p1p2_fxfy": dict(kind="brown", coef=["k1", "k2", "p1", "p2", "k3"], f="fxfy"),
    "B_k1_ppfix": dict(kind="brown", coef=["k1"], pp="fixed"),
    "B_k1k2_ppfix": dict(kind="brown", coef=["k1", "k2"], pp="fixed"),
    "B_k1k2k3_ppfix": dict(kind="brown", coef=["k1", "k2", "k3"], pp="fixed"),
    "R_k1k2k4": dict(kind="brown", coef=["k1", "k2", "k4"]),
    "R_k1k2k3k4k5k6": dict(kind="brown", coef=["k1", "k2", "k3", "k4", "k5", "k6"]),
    "KB_k1k2": dict(kind="kb", coef=["k1", "k2"]),
    "KB_k1k2k3k4": dict(kind="kb", coef=["k1", "k2", "k3", "k4"]),
    "D_l1": dict(kind="div", coef=["l1"]),
    "D_l1l2": dict(kind="div", coef=["l1", "l2"]),
    "D_l1_ppfix": dict(kind="div", coef=["l1"], pp="fixed"),
}
REF = "B_k1k2"
DATASETS = ["M", "E", "ME"]
SIG0 = dict(M=4.0, L=1.0, V=0.4, S=0.25)


def make_lens(name):
    s = MODELS[name]
    return L.Lens(s["kind"], s["coef"], f=s.get("f", "single"), pp=s.get("pp", "free"), name=name)


D = None


def get_data():
    global D
    if D is None:
        D = L.load_all(include_sides=False, drop_edge_ids=DUPLICATES)
    return D


def P_generic(P):
    """Dict of intrinsics -> plain floats (json)."""
    return {k: float(v) for k, v in P.items()}


def fit_one(name, ds, sig, P_start, poses, wv=None, drop_sticker=None, drop_edges=(), starts=None):
    """Fit model `name` on data set `ds` (optionally without a sticker / some edges)."""
    Dd = get_data()
    lens = make_lens(name)
    data = L.build(Dd, ds, drop_sticker=drop_sticker, drop_edges=drop_edges)
    est = L.Est(lens, data, sig)
    best = None
    for fs in (starts or [1.0]):
        Pi = dict(P_start)
        Pi["fx"] *= fs
        Pi["fy"] *= fs
        try:
            r = est.solve(est.x0(Pi, poses, wv))
        except Exception as ex:  # noqa
            print("fit fail", name, ds, ex)
            continue
        if best is None or r.cost < best.cost:
            best = r
    return lens, est, best


def summarize_fit(lens, est, r, with_cov=True):
    P = lens.P(r.x[: lens.ni])
    bs = est.block_stats(r.x)
    b = est.blocks(r.x)
    out = dict(x=r.x.tolist(), pnames=est.pnames, intr_names=lens.names, P=P_generic(P), chi2=float(2 * r.cost),
               n_resid=int(len(r.fun)), k=int(len(r.x)), k_intr=int(lens.ni), block_stats=bs,
               rss={k: float(np.sum(b[k] ** 2)) for k in ("M", "V", "S") if len(b[k])},
               nblk={k: int(len(b[k])) for k in ("M", "V", "S") if len(b[k])}, monotone=lens.monotone_ok(P),
               poses={str(c): v.tolist() for c, v in est.poses_of(r.x).items()},
               wv=(est.wv_of(r.x).tolist() if est.has_wv else None), status=int(r.status), nfev=int(r.nfev))
    if with_cov:
        cl = est.clusters()
        Cs = L.sandwich_cov(r, cl)
        Cn = L.naive_cov(r)
        ni = lens.ni
        out["cov_sandwich_intr"] = Cs[:ni, :ni].tolist()
        out["cov_naive_intr"] = Cn[:ni, :ni].tolist()
        out["sd_sandwich"] = dict(zip(lens.names, np.sqrt(np.abs(np.diag(Cs[:ni, :ni]))).tolist()))
        out["sd_naive"] = dict(zip(lens.names, np.sqrt(np.abs(np.diag(Cn[:ni, :ni]))).tolist()))
        # design effect: sandwich / naive variance of the intrinsics (overdispersion incl. cluster correlation)
        out["deff_intr"] = (np.diag(Cs[:ni, :ni]) / np.maximum(np.diag(Cn[:ni, :ni]), 1e-300)).tolist()
        out["n_clusters"] = int(len(np.unique(cl)))
        out["cov_sandwich_full"] = Cs.tolist()
    return out


# ------------------------------------------------------------------------------------------
# prediction helpers (lens fixed)
# ------------------------------------------------------------------------------------------

def predict_sticker(lens, P, poses, sticker):
    """Corners of one sticker predicted with the training lens + training cart pose."""
    Dd = get_data()
    pts = [p for p in Dd["pts"] if (p["cart"], p["id"]) == sticker]
    X = np.array([p["X"] for p in pts])
    uv = np.array([p["uv"] for p in pts])
    pose = poses[sticker[0]]
    from common import rodrigues

    pred = lens.project(X @ rodrigues(pose[:3]).T + pose[3:], P)
    e = np.hypot(*(pred - uv).T)
    return e.tolist()


def predict_edges(lens, P, rots, wv, edge_ids, sig):
    """Held-out edges: straightness (own line) and VP residual (training direction) in distorted px."""
    Dd = get_data()
    edges = [e for e in Dd["edges"] if e["id"] in set(edge_ids)]
    out = {}
    # straightness: every held-out edge as S
    dataS = dict(M=[], L=[], V=[], S=[dict(id=e["id"], P=e["points"], cart=e.get("cart"), info=None, cl=e["id"]) for e in edges])
    est = L.Est(lens, dataS, sig)
    x = est.x0(P)
    b = est.blocks(x)["S"]
    for k, e in enumerate(est.E):
        s = slice(est.start[k], est.start[k] + est.cnt[k])
        out.setdefault(e["id"], {})["S_ms"] = float(np.mean(b[s] ** 2))
        out[e["id"]]["n"] = int(est.cnt[k])
    # VP: held-out V edges whose direction is known from training
    vs = []
    for e in edges:
        g = L.vp_group(e)
        if g is None:
            continue
        if g[0] == "wv" and wv is None:
            continue
        if g[0] != "wv" and g[0] not in rots:
            continue
        vs.append(dict(id=e["id"], P=e["points"], cart=e.get("cart"), info=g, cl=e["id"]))
    if vs:
        est = L.Est(lens, dict(M=[], L=[], V=vs, S=[]), sig)
        poses = {c: np.r_[rots[c], np.zeros(3)] for c in rots}
        for c in est.carts:
            poses.setdefault(c, np.zeros(6))
        x = est.x0(P, poses, wv)
        b = est.blocks(x)["V"]
        for k, e in enumerate(est.E):
            s = slice(est.start[k], est.start[k] + est.cnt[k])
            out[e["id"]]["V_ms"] = float(np.mean(b[s] ** 2))
    return out


def fit_poses_only(lens, P, ds, sig, poses, wv=None):
    """Lens fixed, refit poses (cross-data prediction)."""
    Dd = get_data()
    data = L.build(Dd, ds)
    est = L.Est(lens, data, sig)
    x0 = est.x0(P, poses, wv)
    ni = lens.ni
    th = x0[:ni]
    r = least_squares(lambda y: est.residuals(np.r_[th, y]), x0[ni:], method="trf", x_scale="jac", max_nfev=300)
    return est, np.r_[th, r.x]


# ------------------------------------------------------------------------------------------
# task runners (for the pool)
# ------------------------------------------------------------------------------------------

def task_full(args):
    name, ds, sig, P_start, poses, wv = args
    t = time.time()
    starts = [1.0, 0.9, 1.1] if ds == "M" else [1.0]
    lens, est, r = fit_one(name, ds, sig, P_start, {int(k): np.array(v) for k, v in poses.items()}, wv, starts=starts)
    s = summarize_fit(lens, est, r)
    s["time"] = time.time() - t
    return name, ds, s


def task_cv(args):
    kind, name, ds, fold, sig, full = args
    lens = make_lens(name)
    P_start = lens.P(np.array(full["x"][: lens.ni]))
    poses = {int(k): np.array(v) for k, v in full["poses"].items()}
    wv = np.array(full["wv"]) if full["wv"] is not None else None
    if kind == "loso":
        lens, est, r = fit_one(name, ds, sig, P_start, poses, wv, drop_sticker=tuple(fold))
        P = lens.P(r.x[: lens.ni])
        e = predict_sticker(lens, P, est.poses_of(r.x), tuple(fold))
        return kind, name, ds, str(tuple(fold)), dict(err=e, P=P_generic(P))
    # region-out
    fid, ids = fold
    lens, est, r = fit_one(name, ds, sig, P_start, poses, wv, drop_edges=ids)
    P = lens.P(r.x[: lens.ni])
    rots = {c: est.poses_of(r.x)[c][:3] for c in est.carts}
    pred = predict_edges(lens, P, rots, est.wv_of(r.x), ids, sig)
    return kind, name, ds, fid, dict(pred=pred, P=P_generic(P))


def main():
    t0 = time.time()
    Dd = get_data()
    print(f"stickers {len(Dd['markers'])}, corners {len(Dd['pts'])}, edges {len(Dd['edges'])} (duplicates dropped: {DUPLICATES})")
    out = dict(models=MODELS, ref=REF, duplicates_dropped=DUPLICATES, sig={}, fits={}, cv={}, cross={})
    # ---------------- reference fits with variance components ----------------
    ref_state = {}
    for ds in DATASETS:
        lens = make_lens(REF)
        data = L.build(Dd, ds)
        best = None
        for f0 in ([1250.0, 1400.0, 1550.0] if ds == "M" else [1400.0]):
            est = L.Est(lens, data, dict(SIG0))
            P = lens.P([f0, L.C0[0], L.C0[1], -0.3, 0.07])
            r = est.varcomp(est.x0(P))
            if best is None or r.cost < best[1].cost:
                best = (est, r)
        est, r = best
        sig = {k: v for k, v in est.sig.items()}
        out["sig"][ds] = sig
        ref_state[ds] = (lens.P(r.x[: lens.ni]), est.poses_of(r.x), est.wv_of(r.x))
        print(f"[ref {ds}] sig {({k: round(v, 3) for k, v in sig.items()})}  intr {np.round(r.x[: lens.ni], 4)}  blocks {est.block_stats(r.x)}")
    # ---------------- all models x data sets ----------------
    tasks = []
    for ds in DATASETS:
        Pref, poses, wv = ref_state[ds]
        lref = make_lens(REF)
        for name in MODELS:
            lens = make_lens(name)
            Pinit = {k: Pref[k] for k in ("fx", "fy", "cx", "cy")}
            Pinit.update({k: (Pref[k] if lens.kind == "brown" and k in ("k1", "k2") else 0.0) for k in lens.coef})
            Pi, _ = L.convert(lref, Pref, lens, Pinit)
            tasks.append((name, ds, out["sig"][ds], Pi, {str(c): v.tolist() for c, v in poses.items()}, (wv.tolist() if wv is not None else None)))
    with Pool(NPROC) as pool:
        for name, ds, s in pool.imap_unordered(task_full, tasks):
            out["fits"].setdefault(ds, {})[name] = s
            bs = s["block_stats"]
            print(f"  {ds:2s} {name:18s} chi2 {s['chi2']:9.1f} k_intr {s['k_intr']} " +
                  " ".join(f"{k}={v:.4g}" for k, v in s["P"].items() if v != 0.0 or k in ("fx", "cx", "cy")) +
                  " | " + " ".join(f"{b}:{v['rms']:.3f}" for b, v in bs.items()) + f"  mono={s['monotone']}  {s['time']:.1f}s", flush=True)
    print(f"full fits done {time.time() - t0:.0f}s", flush=True)
    # ---------------- cross-data predictions ----------------
    for name in MODELS:
        lens = make_lens(name)
        # lens from E -> stickers (poses refit)
        fE = out["fits"]["E"][name]
        PE = lens.P(np.array(fE["x"][: lens.ni]))
        est, x = fit_poses_only(lens, PE, "M", out["sig"]["M"], ref_state["M"][1])
        bs = est.block_stats(x)
        # lens from M -> edges (rotations refit)
        fM = out["fits"]["M"][name]
        PM = lens.P(np.array(fM["x"][: lens.ni]))
        est2, x2 = fit_poses_only(lens, PM, "E", out["sig"]["E"], ref_state["E"][1], ref_state["E"][2])
        bs2 = est2.block_stats(x2)
        out["cross"][name] = dict(E_to_M=bs, M_to_E=bs2)
        print(f"  cross {name:18s} lens(E)->stickers RMS {bs['M']['rms_radial']:.2f} px | lens(M)->edges S {bs2['S']['rms']:.3f} V {bs2['V']['rms']:.3f}", flush=True)
    # ---------------- cross-validation ----------------
    stickers = sorted({(p["cart"], p["id"]) for p in Dd["pts"]})
    src_regions = sorted({e["region"] for e in Dd["edges"]})
    folds_region = [(f"src_{r}", [e["id"] for e in Dd["edges"] if e["region"] == r]) for r in src_regions]
    tiles = sorted({L.tile_of(e) for e in Dd["edges"]})
    folds_tile = [(f"tile_{t}", [e["id"] for e in Dd["edges"] if L.tile_of(e) == t]) for t in tiles]
    cv_models = list(MODELS) if not QUICK else [REF, "B_k1", "KB_k1k2", "D_l1"]
    ctasks = []
    for name in cv_models:
        for ds in ("M", "ME"):
            for st in stickers:
                ctasks.append(("loso", name, ds, list(st), out["sig"][ds], out["fits"][ds][name]))
        for ds in ("E", "ME"):
            for fold in folds_region + folds_tile:
                ctasks.append(("region", name, ds, fold, out["sig"][ds], out["fits"][ds][name]))
    print(f"CV tasks: {len(ctasks)}", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for kind, name, ds, fid, res in pool.imap_unordered(task_cv, ctasks, chunksize=4):
            out["cv"].setdefault(kind, {}).setdefault(ds, {}).setdefault(name, {})[fid] = res
            done += 1
            if done % 100 == 0:
                print(f"  cv {done}/{len(ctasks)}  {time.time() - t0:.0f}s", flush=True)
    out["folds"] = dict(stickers=[list(s) for s in stickers], region=folds_region, tile=folds_tile)
    for ds in out["fits"]:
        for name in out["fits"][ds]:
            out["fits"][ds][name].pop("cov_sandwich_full", None)
    save_json(out, f"{CACHE}/altmodels_fits.json")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
