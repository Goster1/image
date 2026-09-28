"""Cuboid cart model, part 1: joint fit of sticker corners + exact 3D lines (drawing dimensions exact).

Data (all from the 7-frame jitter-compensated mean image):
  M  62 valid sticker corners (markers_final.json)
  S  traced sides of partly hidden stickers (exact by the sticker geometry; markerdata.marker_side_edges)
  E  traced cart edges with model_line.exact == True (edges_cart310/80.json), each checked on a crop
     (results/cuboid_exact_crops.png):
        cart310_board0_outer     (X = 0, Z = 0; free Y)   top-plate end edge at the X=0 board
        cart80_x_board0_front    (Y = 0, Z = 0; free X)   top-plate front edge at the X=0 board
        cart80_x_board1600_front (Y = 0, Z = 0; free X)   top-plate front edge at the X=1600 board
Unknowns: intrinsics (model grid) + one 6-DoF pose per cart. No geometric parameter is free.
Residuals: corner reprojection error; for lines the signed distance (distorted image) of each traced
point to the projected 3D line.  Weighting: variance components per block; MAIN = cluster weighting
(each traced line counts as n_eff = clip(chord / 25 px, 2, 6) observations), variant = one observation
per traced point (subsampled x2).
Model choice: leave-one-sticker-out prediction error (LOSO) + leave-one-exact-edge-out.
Uncertainty: cluster bootstrap over stickers (corners + sides together) and exact edges; profile of f;
mapping uncertainty (rotation compensated) from the bootstrap samples via evaltools.
Outputs: work/cache/method_cuboid.json/.md, work/cache/cuboid_fit_full.json,
         results/cuboid_exact_crops.png, results/cuboid_f_profile.png, results/cuboid_residuals.png
Usage: python3 41_cuboid_fit.py [NBOOT]
"""
import importlib
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, RESULTS, H, W, K_from, lens_json, load_json, save_json  # noqa: E402
from evaltools import grid, mapping_displacement, mapping_stats  # noqa: E402
from markerdata import load_markers, to_points  # noqa: E402

L = importlib.import_module("41_cuboid_lib")
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
QUICK = os.environ.get("CUBOID_QUICK", "0") == "1"  # debugging: fewer LOSO models / profile points
NPROC = 3
rng = np.random.default_rng(4141)
t_start = time.time()

REVIEW = L.edges_review_state()
M, PTS, SIDES, EXACT, P0 = L.base_data()
print(f"edge files reviewed: {REVIEW}")
print(f"stickers {len(M)}, corners {len(PTS)}, sticker sides {len(SIDES)}, exact structure edges {len(EXACT)}: {[e['id'] for e in EXACT]}")
LINES = SIDES + EXACT
# cluster id of every observation: sticker index for corners and sides, own id for structure edges
MI_OF_SIDE = {}
for s in SIDES:
    _, c, mid, _ = s["id"].split("_")
    MI_OF_SIDE[s["id"]] = [i for i, m in enumerate(M) if m["cart"] == int(c) and m["id"] == int(mid)][0]

SPECS = {
    "k1_ppfix": dict(f="single", pp="fixed", dist=["k1"]),
    "k1k2_ppfix": dict(f="single", pp="fixed", dist=["k1", "k2"]),
    "k1_ppfree": dict(f="single", pp="free", dist=["k1"]),
    "k1k2_ppfree": dict(f="single", pp="free", dist=["k1", "k2"]),
    "k1k2k3_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "k3"]),
    "k1k2_ppfree_fxfy": dict(f="fxfy", pp="free", dist=["k1", "k2"]),
    "k1k2p1p2_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "p1", "p2"]),
}


def fit(spec, pts, lines, wm="cluster", x0=None, sig=None, starts=(1300.0, 1450.0), reweight=True, fold_barrier=True):
    """Best of several starts. Review fixes: (1) fold barrier on by default (a folded lens is not a valid
    model inside the image); (2) the starts are compared by the profiled likelihood cf.nll (the reweighted
    cost is ~n_eff/2 for every converged solution and cannot rank them)."""
    best = None
    if x0 is not None:
        starts = (None,)
    for f0 in starts:
        cf = L.CuboidFit(pts, lines, spec, weight_mode=wm, sig=sig, fold_barrier=fold_barrier)
        if f0 is None:
            xs = x0.copy()
        else:
            xs = cf.x0(K_from(f0, f0, (W - 1) / 2, (H - 1) / 2), np.array([-0.25, 0.05]), P0)
        r = cf.solve(xs, reweight=reweight)
        v = cf.nll(r.x) + (1e6 * max(0.0, cf.FOLD_MIN - 0.005 - cf.fold_margin(r.x)) if fold_barrier else 0.0)
        if best is None or v < best[2]:
            best = (cf, r, v)
    return best[0], best[1]


def predict_corners(cf, x, te_pts):
    cam = cf.camera(x)
    out = []
    for p in te_pts:
        out.append(np.linalg.norm(cam.proj(np.asarray(p["X"])[None], p["cart"])[0] - p["uv"]))
    return np.array(out)


def edge_pred_rms(cf, x, ln):
    cam = cf.camera(x)
    r, _ = L.line_offsets(cam, ln["cart"], ln["A"], ln["D"], ln["points"])
    return float(np.sqrt(np.mean(r ** 2))), float(np.mean(r))


# ---------------------------------------------------------------- worker helpers (fork)
def _loso_job(args):
    name, wm, mi, xref, sig = args
    spec = SPECS[name]
    tr = [p for p in PTS if p["mi"] != mi]
    te = [p for p in PTS if p["mi"] == mi]
    lines = [ln for ln in LINES if MI_OF_SIDE.get(ln["id"], -1) != mi]
    cf, r = fit(spec, tr, lines, wm=wm, x0=np.array(xref), sig=dict(sig))
    return name, wm, mi, predict_corners(cf, r.x, te).tolist()


def _leo_job(args):
    name, wm, k, xref, sig = args
    spec = SPECS[name]
    lines = [ln for ln in LINES if ln["id"] != EXACT[k]["id"]]
    cf, r = fit(spec, PTS, lines, wm=wm, x0=np.array(xref), sig=dict(sig))
    return name, wm, k, edge_pred_rms(cf, r.x, EXACT[k])


def _boot_job(args):
    b, name, wm, xref, sig, pick_m, pick_e = args
    spec = SPECS[name]
    bp, bl = [], []
    for q, mi in enumerate(pick_m):
        bp += [dict(p, mi=1000 * q + mi) for p in PTS if p["mi"] == mi]
        bl += [dict(ln, id=f"{ln['id']}#{q}") for ln in SIDES if MI_OF_SIDE[ln["id"]] == mi]
    for q, k in enumerate(pick_e):
        bl.append(dict(EXACT[k], id=f"{EXACT[k]['id']}#{q}"))
    if len({p["cart"] for p in bp}) < 2:
        return b, None
    try:
        cf, r = fit(spec, bp, bl, wm=wm, x0=np.array(xref), sig=dict(sig))
        return b, r.x.tolist()
    except Exception as ex:  # noqa
        print("boot fail", b, ex)
        return b, None


def main():
    # ------------------------------------------------------------------ model grid
    summary = {}
    fits = {}
    for wm in ("cluster", "point"):
        for name, spec in SPECS.items():
            cf, r = fit(spec, PTS, LINES, wm=wm)
            C = L.cov_from(r, cf)
            s = np.sqrt(np.diag(C))[: cf.ni]
            br = cf.block_rms(r.x)
            fits[(name, wm)] = (cf, r)
            fm = cf.fold_margin(r.x)
            summary[f"{name}|{wm}"] = dict(names=cf.inames, x=r.x[: cf.ni].tolist(), sigma_cov=s.tolist(), block_rms=br,
                                           sig=dict(cf.sig), poses={c: r.x[cf.ni + 6 * i: cf.ni + 6 * i + 6].tolist() for i, c in enumerate(cf.carts)},
                                           per_line=cf.per_line_rms(r.x), fold_margin=fm, on_fold_barrier=bool(fm < cf.FOLD_MIN + 0.005))
            print(f"{wm:7s} {name:17s} " + " ".join(f"{n}={v:.5g}" for n, v in zip(cf.inames, r.x[: cf.ni]))
                  + f" | M {br['M']['rms_point']:.2f} px/pt, S {br['S']['rms']:.2f}, E {br['E']['rms']:.2f} | fold margin {fm:.3f}")
    # the same grid WITHOUT the fold barrier (as originally run) - for transparency only: these lenses fold
    nobar = {}
    for name, spec in SPECS.items():
        cfn, rn = fit(spec, PTS, LINES, wm="cluster", fold_barrier=False)
        nobar[name] = dict(names=cfn.inames, x=rn.x[: cfn.ni].tolist(), block_rms=cfn.block_rms(rn.x), fold_margin=cfn.fold_margin(rn.x))
        print(f"nobarrier {name:17s} " + " ".join(f"{n}={v:.5g}" for n, v in zip(cfn.inames, rn.x[: cfn.ni])) + f" | fold margin {nobar[name]['fold_margin']:.3f}")
    # ------------------------------------------------------------------ LOSO / leave-one-edge-out
    jobs = []
    loso_keys = [k for k in fits if k[1] == "cluster" and (not QUICK or k[0] in ("k1k2_ppfix", "k1_ppfix"))]
    for (name, wm) in loso_keys:
        cf, r = fits[(name, wm)]
        for mi in sorted({p["mi"] for p in PTS}):
            jobs.append((name, wm, mi, r.x.tolist(), dict(cf.sig)))
    ejobs = [(name, wm, k, fits[(name, wm)][1].x.tolist(), dict(fits[(name, wm)][0].sig)) for (name, wm) in loso_keys for k in range(len(EXACT))]
    with mp.get_context("fork").Pool(NPROC) as pool:
        loso = pool.map(_loso_job, jobs, chunksize=4)
        leo = pool.map(_leo_job, ejobs, chunksize=2)
    for (name, wm) in loso_keys:
        errs = np.concatenate([np.array(e) for n, w, mi, e in loso if n == name and w == wm])
        rows = {}
        for n, w, mi, e in loso:
            if n == name and w == wm:
                rows.setdefault(M[mi]["row"], []).extend(e)
        s = summary[f"{name}|{wm}"]
        s["loso_rms_px"] = float(np.sqrt(np.mean(errs ** 2)))
        s["loso_rms_by_row_px"] = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in rows.items()}
        s["leave_edge_out"] = {EXACT[k]["id"]: dict(rms_px=v[0], mean_px=v[1]) for n, w, k, v in leo if n == name and w == wm}
        print(f"{wm:7s} {name:17s} LOSO {s['loso_rms_px']:.2f} px  rows " + str({k: round(v, 2) for k, v in s['loso_rms_by_row_px'].items()})
              + "  edge-out " + str({k.split('_', 1)[1]: round(v['rms_px'], 2) for k, v in s["leave_edge_out"].items()}))
    # ------------------------------------------------------------------ choose the main model (cluster weighting)
    # Review fix: with the fold barrier the free-pp / fx!=fy variants reach a lower LOSO, but their pp is NOT determined:
    # it jumps by ~100 px between data subsets / local minima (pp_stability below), and fx != fy by ~2 % only absorbs the
    # geometry mismatch (square pixels expected).  Parameters the data do not determine are fixed (CLAUDE.md), so the main
    # model is chosen by LOSO among the pp-fixed, fx = fy models only; the free-pp LOSO values are reported.
    pp_stab = {}
    for nm_ in ("k1k2_ppfree", "k1k2_ppfree_fxfy"):
        rows_ = {"all": dict(zip(summary[f"{nm_}|cluster"]["names"], summary[f"{nm_}|cluster"]["x"]))}
        for tag, lines, pts in (("corners_only", [], PTS), ("corners+sides", SIDES, PTS), ("corners+exact_edges", EXACT, PTS)):
            cfv, rv = fit(SPECS[nm_], pts, lines, wm="cluster")
            rows_[tag] = dict(zip(cfv.inames, rv.x[: cfv.ni].tolist()))
        pp_stab[nm_] = rows_
        print("pp stability", nm_, {k: (round(v["cx"]), round(v["cy"])) for k, v in rows_.items()})
    cand = {k: v for k, v in summary.items() if k.endswith("|cluster") and "loso_rms_px" in v and "cx" not in v["names"] and "fx" not in v["names"]}
    best_key = min(cand, key=lambda k: cand[k]["loso_rms_px"] + 0.02 * len(cand[k]["x"]))
    best_name = best_key.split("|")[0]
    cf, r = fits[(best_name, "cluster")]
    print("main model:", best_key)
    # ------------------------------------------------------------------ bias-corrected corners variant
    Mb = load_markers()
    for m, raw in zip(Mb, load_json(f"{CACHE}/markers_final.json")):
        if raw.get("corners_px_bias_corrected") is not None:
            c = np.array([q if q is not None else [np.nan, np.nan] for q in raw["corners_px_bias_corrected"]], float)
            ok = ~np.isnan(c).any(1)
            m["corners_px"] = np.where(ok[:, None], c, m["corners_px"])
    pts_b = to_points(Mb)
    cfb, rb = fit(SPECS[best_name], pts_b, LINES, wm="cluster", x0=r.x.copy(), sig=dict(cf.sig))
    summary["bias_corrected_corners"] = dict(names=cfb.inames, x=rb.x[: cfb.ni].tolist(), block_rms=cfb.block_rms(rb.x), fold_margin=cfb.fold_margin(rb.x))
    # data-subset variants of the main model
    for tag, lines, pts in (("corners_only", [], PTS), ("corners+sides", SIDES, PTS), ("corners+exact_edges", EXACT, PTS)):
        cfv, rv = fit(SPECS[best_name], pts, lines, wm="cluster")
        summary[f"subset_{tag}"] = dict(names=cfv.inames, x=rv.x[: cfv.ni].tolist(), block_rms=cfv.block_rms(rv.x), fold_margin=cfv.fold_margin(rv.x))
        print(f"subset {tag:22s}", np.round(rv.x[: cfv.ni], 4), {k: round(v.get('rms_point', v.get('rms')), 2) for k, v in cfv.block_rms(rv.x).items()})
    # ------------------------------------------------------------------ profile of f (main model)
    prof = []
    sig_fix = dict(cf.sig)
    for fv in np.linspace(r.x[0] - 250, r.x[0] + 250, 5 if QUICK else 21):
        cfp = L.CuboidFit(PTS, LINES, SPECS[best_name], weight_mode="cluster", sig=sig_fix)
        cfp.update_feet(r.x)
        y = r.x[1:].copy()
        for it in range(3):
            rr = least_squares(lambda yy: cfp.residuals(np.r_[fv, yy]), y, method="trf", x_scale="jac", max_nfev=200)
            y = rr.x
            cfp.update_feet(np.r_[fv, y])
        br = cfp.block_rms(np.r_[fv, y])
        prof.append([fv, 2 * rr.cost, br["M"]["rms_point"], br["S"]["rms"], br["E"]["rms"]] + list(y[: cfp.ni - 1]) + [cfp.fold_margin(np.r_[fv, y])])
        print(f"  profile f={fv:.0f}: chi2 {2 * rr.cost:.1f}  M {br['M']['rms_point']:.2f} S {br['S']['rms']:.2f} E {br['E']['rms']:.2f}")
    prof = np.array(prof)
    # ------------------------------------------------------------------ cluster bootstrap
    mis = sorted({p["mi"] for p in PTS})
    bjobs = []
    for b in range(NBOOT):
        pick_m = rng.choice(mis, len(mis), replace=True).tolist()
        pick_e = rng.integers(0, len(EXACT), len(EXACT)).tolist()
        bjobs.append((b, best_name, "cluster", r.x.tolist(), dict(cf.sig), pick_m, pick_e))
    with mp.get_context("fork").Pool(NPROC) as pool:
        bres = pool.map(_boot_job, bjobs, chunksize=2)
    boot = np.array([x for b, x in sorted(bres) if x is not None])
    print(f"bootstrap: {len(boot)} of {NBOOT} ok")
    bi = boot[:, : cf.ni]
    boot_fold = np.array([cf.fold_margin(xb) for xb in boot])
    print(f"bootstrap fold margins: min {boot_fold.min():.3f}, on barrier (<{cf.FOLD_MIN + 0.005}) {np.mean(boot_fold < cf.FOLD_MIN + 0.005):.2f}")
    K, dist = cf.camera(r.x).K, cf.camera(r.x).dist
    uv, _ = grid(40)
    disp = []
    for xb in boot:
        cb = cf.camera(xb)
        disp.append(mapping_displacement(K, dist[:5], cb.K, cb.dist[:5], uv))
    disp = np.array(disp)
    mstats, mrms = mapping_stats(disp, uv)
    disp_raw = np.array([mapping_displacement(K, dist[:5], cf.camera(xb).K, cf.camera(xb).dist[:5], uv, compensate=False) for xb in boot])
    mstats_raw, _ = mapping_stats(disp_raw, uv)
    bstd = bi.std(0, ddof=1)
    q16, q84 = np.percentile(bi, 16, axis=0), np.percentile(bi, 84, axis=0)
    print("bootstrap std:", dict(zip(cf.inames, np.round(bstd, 4))))
    print("mapping 1 sigma (rot-comp):", {k: round(v["rms_median_px"], 2) for k, v in mstats.items()})
    # f spread over models / weightings (systematic)
    f_all = {k: v["x"][0] for k, v in summary.items() if "|" in k}
    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
    ax[0].plot(prof[:, 0], prof[:, 1] - prof[:, 1].min(), "k-o", ms=3)
    ax[0].axhline(1, color="r", lw=0.8, label="delta chi2 = 1")
    ax[0].set_xlabel("f [px] (fixed; pp, distortion, poses refit)")
    ax[0].set_ylabel("delta chi^2 (block weights of the main fit)")
    ax[0].set_title("profile of f - cuboid joint fit")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(prof[:, 0], prof[:, 2], "r-o", ms=3, label="sticker corners (px per point)")
    ax[1].plot(prof[:, 0], prof[:, 3], "g-o", ms=3, label="sticker sides (px)")
    ax[1].plot(prof[:, 0], prof[:, 4], "b-o", ms=3, label="exact structure edges (px)")
    ax[1].set_xlabel("f [px]")
    ax[1].set_ylabel("block RMS [px]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/cuboid_f_profile.png")
    plt.close(fig)
    # residual plot
    img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
    pred = cf.marker_pred(r.x)
    res = pred - cf.muv
    fig, ax = plt.subplots(figsize=(17.5, 10), dpi=110)
    ax.imshow((img * 0.55).astype(np.uint8))
    ax.plot(cf.muv[:, 0], cf.muv[:, 1], "g+", ms=6)
    ax.quiver(cf.muv[:, 0], cf.muv[:, 1], res[:, 0] * 10, res[:, 1] * 10, angles="xy", scale_units="xy", scale=1, color="yellow", width=0.0015)
    cam = cf.camera(r.x)
    CL = L.cart_lines()
    for ln in LINES:
        rr_, nv = L.line_offsets(cam, ln["cart"], ln["A"], ln["D"], ln["points"])
        P = ln["points"]
        col = "cyan" if ln["kind"] == "exact_edge" else "orange"
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1)
        Q = P - 10 * rr_[:, None] * nv
        ax.plot(Q[:, 0], Q[:, 1], ":", color=col, lw=1)
    for nm in ("top_front", "top_back", "top_x0", "top_x1600"):
        for c in (80, 310):
            uvp, _ = L.project_segment(cam, c, CL[nm]["A"], CL[nm]["D"], CL[nm]["t0"], CL[nm]["t1"])
            ax.plot(uvp[:, 0], uvp[:, 1], "r-", lw=0.6)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    br = cf.block_rms(r.x)
    ax.set_title(f"cuboid joint fit ({best_name}, cluster weights): corner residuals x10 (yellow), traced sticker sides (orange) / exact edges (cyan) "
                 f"with residual x10 (dotted); red = projected top-plate outline.  corners {br['M']['rms_point']:.2f} px/pt, sides {br['S']['rms']:.2f} px, edges {br['E']['rms']:.2f} px",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/cuboid_residuals.png")
    plt.close(fig)
    # exact-edge crops: traced points (green), prediction markers-only (red), joint (cyan)
    cam_m = L.markers_only_camera()
    raw_img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    fig, axs = plt.subplots(1, len(EXACT), figsize=(16, 7), dpi=110, gridspec_kw=dict(width_ratios=[3, 1, 1][: len(EXACT)]))
    for ax, e in zip(np.atleast_1d(axs), EXACT):
        P = e["points"]
        x0, y0 = np.maximum(P.min(0) - 25, 0).astype(int)
        x1, y1 = np.minimum(P.max(0) + 25, [W - 1, H - 1]).astype(int)
        ax.imshow(cv2.cvtColor(raw_img[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2RGB), extent=(x0 - 0.5, x1 + 0.5, y1 + 0.5, y0 - 0.5), interpolation="bicubic")
        ax.plot(P[:, 0], P[:, 1], ".", color="lime", ms=2, label="traced edge")
        for cc, colr, lab in ((cam_m, "red", "markers-only fit"), (cam, "cyan", "cuboid joint fit")):
            uvp, _ = L.project_segment(cc, e["cart"], e["A"], e["D"], -300, 1900, n=3000)
            ax.plot(uvp[:, 0], uvp[:, 1], "-", color=colr, lw=0.8, label=lab)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(e["id"], fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("exact structure edges (model_line.exact): traced points and predicted lines (drawing geometry)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/cuboid_exact_crops.png")
    plt.close(fig)
    # implied position of the exact edges with the joint camera (check of 'exact')
    exact_check = {}
    AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
    for e in EXACT:
        ax_name = "X" if e["D"][1] == 1 else "Y"
        for cname, cc in (("joint", cam), ("markers_only", cam_m)):
            sft, rms_, sens = L.implied_shift(cc, e["cart"], e["A"], e["D"], e["points"], AX[ax_name])
            exact_check.setdefault(e["id"], {})[cname] = dict(axis=ax_name, implied_shift_mm=sft, rms_after_px=rms_, px_per_mm=sens,
                                                               mean_offset_px=float(np.mean(L.line_offsets(cc, e["cart"], e["A"], e["D"], e["points"])[0])))
    # ------------------------------------------------------------------ outputs
    names = cf.inames
    unc = {f"{n}_1sigma": float(v) for n, v in zip(names, bstd)}
    unc.update({f"{n}_1sigma_cov": float(v) for n, v in zip(names, summary[best_key]["sigma_cov"])})
    unc["f_1sigma_systematic_model_weighting"] = float(np.std([v for k, v in f_all.items()]))
    unc["method"] = ("1-sigma = std of a cluster bootstrap (resampling stickers - corners and sides together - and exact edges, "
                     f"n={len(boot)}, every replicate refit with the fold barrier); *_cov = Gauss-Newton covariance scaled by the residual variance "
                     "per effective observation (cluster n_eff; fold-barrier row dropped; ignores the geometry misfit correlation); "
                     "f_1sigma_systematic_model_weighting = std of f over all models x weightings. NONE of these contains the bias of the "
                     "drawing geometry (diagnostic 41_cuboid_diag.py: freeing the top-plate height moves f by ~+100 px; written below as "
                     "f_geometry_shift_px_diag / f_1sigma_incl_geometry by 41_cuboid_diag.py)")
    brm = summary[best_key]["block_rms"]
    cam_best = cf.camera(r.x)
    out = lens_json(cam_best.K, cam_best.dist[:5], method="cuboid (sticker corners + sticker sides + exact top-plate edges, drawing dimensions exact)",
                    model=f"{best_name}: " + {"k1_ppfix": "fx=fy, pp fixed at image centre, k1 only", "k1k2_ppfix": "fx=fy, pp fixed at image centre, k1+k2",
                                              "k1_ppfree": "fx=fy, pp free, k1", "k1k2_ppfree": "fx=fy, pp free, k1+k2"}.get(best_name, best_name),
                    rms_reprojection_error_px=brm["M"]["rms_point"], uncertainty=unc,
                    mapping_uncertainty_px={k: v["rms_median_px"] for k, v in mstats.items() if k != "whole_image"},
                    data_used=(f"{len(PTS)} valid sticker corners of {len(M)} stickers, {len(SIDES)} traced sides of partly hidden stickers, "
                               f"{len(EXACT)} exact structure edges ({', '.join(e['id'] for e in EXACT)}); mean of 7 jitter-compensated stills"),
                    geometry_assumptions="drawing dimensions and sticker positions exactly as in spec/cart-marker-layout.json (nothing fitted); exact edges on the drawing lines",
                    details=dict(main_key=best_key, block_rms=brm, fold_margin=cf.fold_margin(r.x), fold_barrier_min=cf.FOLD_MIN,
                                 on_fold_barrier=bool(cf.fold_margin(r.x) < cf.FOLD_MIN + 0.005),
                                 bootstrap_fold_margin=dict(min=float(boot_fold.min()), median=float(np.median(boot_fold)),
                                                            frac_on_barrier=float(np.mean(boot_fold < cf.FOLD_MIN + 0.005))),
                                 model_choice=("LOSO (+0.02 px per intrinsic) among the pp-fixed, fx=fy models; free-pp / fx!=fy models excluded because "
                                               "their pp is not determined (pp_stability: jumps between data subsets / local minima)"),
                                 loso_all_models={k: v.get("loso_rms_px") for k, v in summary.items() if k.endswith("|cluster")},
                                 pp_stability=pp_stab,
                                 without_fold_barrier=dict(note="same grid without the barrier (the original run): every k1/k1k2 lens folds inside the image "
                                                                "(fold margin < 0), i.e. it is not invertible there and is not a valid main result", grid=nobar), block_sigmas=dict(cf.sig), weighting="cluster: every traced line = n_eff = clip(chord/25px, 2, 6) observations",
                                 rms_note="rms_reprojection_error_px = RMS per corner point (radial); block S/E = RMS of the signed point-to-line distance [px]",
                                 bootstrap=dict(n=len(boot), std=dict(zip(names, bstd.tolist())), q16=dict(zip(names, q16.tolist())), q84=dict(zip(names, q84.tolist()))),
                                 mapping=mstats, mapping_raw=mstats_raw, loso_rms_px=summary[best_key]["loso_rms_px"],
                                 loso_rms_by_row_px=summary[best_key]["loso_rms_by_row_px"], leave_edge_out=summary[best_key]["leave_edge_out"],
                                 f_over_models_and_weightings=f_all, exact_edge_check=exact_check, edge_files_reviewed=REVIEW,
                                 grid={k: dict(x=v["x"], names=v["names"], loso=v.get("loso_rms_px"), blocks=v["block_rms"]) for k, v in summary.items()}))
    save_json(out, f"{CACHE}/method_cuboid.json")
    alts = []
    for k, v in summary.items():
        if "|" not in k or k == best_key:
            continue
        cfa, ra = fits[tuple(k.split("|"))]
        ca = cfa.camera(ra.x)
        alts.append(lens_json(ca.K, ca.dist[:5], method=f"cuboid_{k.replace('|', '_')}", model=k, rms_reprojection_error_px=v["block_rms"]["M"]["rms_point"],
                              uncertainty={f"{n}_1sigma_cov": s for n, s in zip(v["names"], v["sigma_cov"])},
                              details=dict(block_rms=v["block_rms"], loso_rms_px=v.get("loso_rms_px"))))
    save_json(alts, f"{CACHE}/method_cuboid_alternatives.json")
    save_json(dict(summary=summary, main_key=best_key, x_main=r.x.tolist(), names=cf.names, sig=dict(cf.sig), profile=prof.tolist(),
                   boot=boot.tolist(), mapping=mstats, mapping_raw=mstats_raw, exact_check=exact_check), f"{CACHE}/cuboid_fit_full.json")
    # markdown summary
    s = summary[best_key]
    lines_md = ["# Method: cuboid cart model (stickers + exact edges, drawing dimensions exact)", "",
                f"Script `work/41_cuboid_fit.py` (helpers `41_cuboid_lib.py`). Data: {len(PTS)} sticker corners, {len(SIDES)} sticker sides, "
                f"{len(EXACT)} exact structure edges ({', '.join(e['id'] for e in EXACT)}). Edge files reviewed: {REVIEW}.", "",
                f"Main model (min. leave-one-sticker-out error among the pp-fixed, fx=fy models, cluster weighting): **{best_name}**. "
                "Free-pp variants reach a lower LOSO (" + ", ".join(f"{k.split('|')[0]} {v['loso_rms_px']:.2f} px" for k, v in summary.items()
                                                                      if k.endswith("|cluster") and "cx" in v["names"] and "loso_rms_px" in v)
                + ") but their pp is not determined: " + "; ".join(f"{nm_}: " + ", ".join(f"{t} ({v['cx']:.0f}, {v['cy']:.0f})" for t, v in rw.items())
                                                                   for nm_, rw in pp_stab.items()) + ".", "",
                "| param | value | bootstrap 1-sigma | cov 1-sigma |", "|---|---|---|---|"]
    for n, v, e1, e2 in zip(names, s["x"], bstd, s["sigma_cov"]):
        lines_md.append(f"| {n} | {v:.5g} | {e1:.3g} | {e2:.3g} |")
    lines_md += ["", f"Block RMS: corners {brm['M']['rms_point']:.2f} px per point (max {brm['M']['max_point']:.1f}), sticker sides {brm['S']['rms']:.2f} px, "
                 f"exact edges {brm['E']['rms']:.2f} px. LOSO {s['loso_rms_px']:.2f} px.",
                 f"Mapping uncertainty (rot.-comp., bootstrap): " + ", ".join(f"{k} {v['rms_median_px']:.2f} px" for k, v in mstats.items()), "",
                 f"Fold barrier (fold margin >= {cf.FOLD_MIN}) ON in every fit; main-model fold margin {cf.fold_margin(r.x):.3f}"
                + (" - the solution SITS ON THE BARRIER: the data (drawing geometry) push the distortion towards a lens that folds inside the image; "
                   "k1, k2 are then fixed by the validity constraint, not by the data." if cf.fold_margin(r.x) < cf.FOLD_MIN + 0.005 else "")
                + f" Bootstrap: {np.mean(boot_fold < cf.FOLD_MIN + 0.005) * 100:.0f} % of the replicates on the barrier.", "",
                "| model / weighting | f | pp | dist | corners px/pt | sides | edges | LOSO | fold margin |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        if "|" not in k:
            continue
        x = dict(zip(v["names"], v["x"]))
        f_ = x.get("f", x.get("fx"))
        pp = f"({x['cx']:.0f}, {x['cy']:.0f})" if "cx" in x else "centre"
        dd = ", ".join(f"{n}={x[n]:.3f}" for n in ("k1", "k2", "k3", "p1", "p2") if n in x)
        b = v["block_rms"]
        lines_md.append(f"| {k} | {f_:.0f} | {pp} | {dd} | {b['M']['rms_point']:.2f} | {b['S']['rms']:.2f} | {b['E']['rms']:.2f} | {v.get('loso_rms_px', float('nan')):.2f} | {v['fold_margin']:.3f} |")
    lines_md += ["", "Without the fold barrier (original run; all k1 / k1k2 lenses fold inside the image - not valid):", "",
                 "| model (cluster) | f | pp | dist | corners px/pt | sides | edges | fold margin |", "|---|---|---|---|---|---|---|---|"]
    for k, v in nobar.items():
        x = dict(zip(v["names"], v["x"]))
        pp = f"({x['cx']:.0f}, {x['cy']:.0f})" if "cx" in x else "centre"
        dd = ", ".join(f"{n}={x[n]:.3f}" for n in ("k1", "k2", "k3", "p1", "p2") if n in x)
        b = v["block_rms"]
        lines_md.append(f"| {k} | {x.get('f', x.get('fx')):.0f} | {pp} | {dd} | {b['M']['rms_point']:.2f} | {b['S']['rms']:.2f} | {b['E']['rms']:.2f} | {v['fold_margin']:.3f} |")
    lines_md += ["", "Exact-edge check (implied shift of the traced edge from its drawing line with the fitted camera, mm):"]
    for eid, v in exact_check.items():
        lines_md.append(f"* {eid}: joint {v['joint']['implied_shift_mm']:+.1f} mm along {v['joint']['axis']} ({v['joint']['mean_offset_px']:+.2f} px); "
                        f"markers-only {v['markers_only']['implied_shift_mm']:+.1f} mm ({v['markers_only']['mean_offset_px']:+.2f} px)")
    open(f"{CACHE}/method_cuboid.md", "w").write("\n".join(lines_md) + "\n")
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
