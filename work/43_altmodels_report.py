"""Sub-task 43 (lens model choice), step 2: compare the fitted models (from 43_altmodels_fit.py).

  * information criteria with several noise hypotheses (own sigma, fixed block sigma, overdispersion-
    corrected QAIC/QBIC with the cluster design effect, detection noise only)
  * cross-validation summaries (leave-one-sticker-out, leave-one-edge-region-out) with paired SE vs reference
  * MAPPING differences between models (rotation-compensated and raw) per region
  * parameter-sample mapping uncertainty per model (cluster-robust sandwich covariance)
  * conversion of non-Brown models to the closest OpenCV Brown [k1,k2,p1,p2,k3] (evaltools.brown_from_mapping)
Writes work/cache/altmodels_summary.json/.md, work/cache/method_alt_<model>.json/.md,
results/altmodels_mapping_diff.png, results/altmodels_cv.png, results/altmodels_radial.png.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, H, RESULTS, W, K_from, lens_json, load_json, save_json  # noqa: E402
from evaltools import brown_from_mapping, mapping_stats, region_defs  # noqa: E402

L = importlib.import_module("43_altmodels_lib")
F = importlib.import_module("43_altmodels_fit")
rng = np.random.default_rng(4343)
FITS = load_json(os.environ.get("ALT_FITS", f"{CACHE}/altmodels_fits.json"))
MODELS = FITS["models"]
REF = FITS["ref"]
NSAMP = int(os.environ.get("ALT_NSAMP", "150"))
DET_SIGMA = 0.1  # px, detection noise of a 7-frame mean corner / edge point (illustration only)

NICE = {
    "B_k1": "Brown k1", "B_k1k2": "Brown k1,k2", "B_k1k2k3": "Brown k1,k2,k3", "B_k1k2p1p2": "Brown k1,k2,p1,p2",
    "B_k1k2_fxfy": "Brown k1,k2, fx!=fy", "B_k1k2k3p1p2_fxfy": "Brown k1,k2,k3,p1,p2, fx!=fy",
    "B_k1_ppfix": "Brown k1, pp fixed", "B_k1k2_ppfix": "Brown k1,k2, pp fixed", "B_k1k2k3_ppfix": "Brown k1,k2,k3, pp fixed",
    "R_k1k2k4": "rational k1,k2 / k4", "R_k1k2k3k4k5k6": "rational k1..k6", "KB_k1k2": "Kannala-Brandt k1,k2",
    "KB_k1k2k3k4": "Kannala-Brandt k1..k4", "D_l1": "division l1", "D_l1l2": "division l1,l2", "D_l1_ppfix": "division l1, pp fixed",
}


def lens_of(name):
    return F.make_lens(name)


def P_of(ds, name):
    f = FITS["fits"][ds][name]
    return lens_of(name).P(np.array(f["x"][: f["k_intr"]]))


# =====================================================================================
# 1. information criteria
# =====================================================================================

def info_criteria(ds):
    fits = FITS["fits"][ds]
    sig = FITS["sig"][ds]
    ref = fits[REF]
    chat = float(np.median(ref["deff_intr"]))  # overdispersion incl. within-cluster correlation (reference model)
    rows = {}
    for name, f in fits.items():
        n = sum(f["nblk"].values())
        k = f["k"]
        chi2 = sum(f["rss"][b] / sig[b] ** 2 for b in f["rss"])  # without the barrier term
        aic_own = sum(f["nblk"][b] * np.log(f["rss"][b] / f["nblk"][b]) for b in f["rss"]) + 2 * k
        bic_own = aic_own - 2 * k + k * np.log(n)
        rows[name] = dict(n=n, k=k, k_intr=f["k_intr"], chi2=chi2, AIC_fixed=chi2 + 2 * k, AIC_own=aic_own, BIC_own=bic_own,
                          QAIC=chi2 / chat + 2 * k, QBIC=chi2 / chat + k * np.log(n / chat),
                          AIC_det=sum(f["rss"].values()) / DET_SIGMA ** 2 + 2 * k,
                          fold_margin=f["fold_margin"], barrier_active=f["barrier_active"])
    for crit in ("AIC_fixed", "AIC_own", "BIC_own", "QAIC", "QBIC", "AIC_det"):
        best = min(r[crit] for r in rows.values())
        for r in rows.values():
            r["d" + crit] = r[crit] - best
    return rows, chat


# =====================================================================================
# 2. cross-validation summaries
# =====================================================================================

def cv_loso(ds):
    cv = FITS["cv"].get("loso", {}).get(ds, {})
    out = {}
    if REF not in cv:
        return out
    stickers = sorted(cv[REF].keys())
    ref_ms = np.array([np.mean(np.square(cv[REF][s]["err"])) for s in stickers])
    for name, folds in cv.items():
        if sorted(folds.keys()) != stickers:
            continue
        ms = np.array([np.mean(np.square(folds[s]["err"])) for s in stickers])
        allerr = np.concatenate([folds[s]["err"] for s in stickers])
        d = ms - ref_ms
        out[name] = dict(rms_px=float(np.sqrt(np.mean(allerr ** 2))), median_px=float(np.median(allerr)), max_px=float(allerr.max()),
                         per_sticker_rms=dict(zip(stickers, np.sqrt(ms).tolist())),
                         dMS_vs_ref=float(d.mean()), dMS_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                         f_spread=float(np.std([folds[s]["P"]["fx"] for s in stickers])))
    return out


def cv_region(ds, kind):
    cv = FITS["cv"].get("region", {}).get(ds, {})
    out = {}
    if REF not in cv:
        return out
    folds_ref = {k: v for k, v in cv[REF].items() if k.startswith(kind)}

    def collect(folds):
        S, V = {}, {}
        for fid, res in folds.items():
            for eid, p in res["pred"].items():
                S[eid] = (p["S_ms"], p["n"])
                if "V_ms" in p:
                    V[eid] = (p["V_ms"], p["n"])
        return S, V

    Sr, Vr = collect(folds_ref)
    for name, folds in cv.items():
        fo = {k: v for k, v in folds.items() if k.startswith(kind)}
        if sorted(fo) != sorted(folds_ref):
            continue
        S, V = collect(fo)
        eS = sorted(set(S) & set(Sr))
        eV = sorted(set(V) & set(Vr))
        wS = np.array([S[e][1] for e in eS], float)
        msS = np.array([S[e][0] for e in eS])
        dS = msS - np.array([Sr[e][0] for e in eS])
        o = dict(S_rms_px=float(np.sqrt(np.sum(msS * wS) / wS.sum())), S_dMS_vs_ref=float(dS.mean()),
                 S_dMS_se=float(dS.std(ddof=1) / np.sqrt(len(dS))), n_edges_S=len(eS))
        if eV:
            wV = np.array([V[e][1] for e in eV], float)
            msV = np.array([V[e][0] for e in eV])
            dV = msV - np.array([Vr[e][0] for e in eV])
            o.update(V_rms_px=float(np.sqrt(np.sum(msV * wV) / wV.sum())), V_dMS_vs_ref=float(dV.mean()),
                     V_dMS_se=float(dV.std(ddof=1) / np.sqrt(len(dV))), n_edges_V=len(eV))
        o["f_spread"] = float(np.std([fo[k]["P"]["fx"] for k in fo]))
        o["f_range"] = [float(min(fo[k]["P"]["fx"] for k in fo)), float(max(fo[k]["P"]["fx"] for k in fo))]
        out[name] = o
    return out


# =====================================================================================
# 3. mapping comparisons
# =====================================================================================
UV, SHAPE = L.pixel_grid(40)
REG = region_defs()


def mapping_vs(ds_a, a, ds_b, b, comp=True):
    la, lb = lens_of(a), lens_of(b)
    d = L.mapping_disp(la, P_of(ds_a, a), lb, P_of(ds_b, b), UV, compensate=comp)
    return d


def sample_mapping_unc(ds, name, n=NSAMP):
    f = FITS["fits"][ds][name]
    lens = lens_of(name)
    th = np.array(f["x"][: f["k_intr"]])
    C = np.array(f["cov_sandwich_intr"])
    P = lens.P(th)
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, 0, None)
    A = V * np.sqrt(w)
    disp = []
    nfold = 0
    for _ in range(n):
        ths = th + A @ rng.standard_normal(len(th))
        Ps = lens.P(ths)
        if lens.fold_margin(Ps) < 0:
            nfold += 1
            continue
        disp.append(L.mapping_disp(lens, P, lens, Ps, UV, compensate=True))
    if len(disp) < max(5, n // 5):
        return None, None, nfold
    st, rms = mapping_stats(np.array(disp), UV)
    return st, rms, nfold


# =====================================================================================
# 4. conversion to OpenCV Brown
# =====================================================================================

def to_brown(ds, name, dist_names=("k1", "k2", "p1", "p2", "k3")):
    lens = lens_of(name)
    P = P_of(ds, name)
    uvf, _ = L.pixel_grid(20)
    rays = lens.rays(uvf, P)
    K, d, rms = brown_from_mapping(uvf, rays, init_f=P["fx"], dist_names=dist_names, pp_free=True)
    lb = L.Lens("brown", ["k1", "k2", "p1", "p2", "k3"], name="conv")
    Pb = lb.P([K[0, 0], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3], d[4]])
    uvt, _ = L.pixel_grid(10)
    disp = lb.project(lens.rays(uvt, P), Pb) - uvt  # direct (same rays -> pixel difference)
    st = L.region_stats(disp, uvt)
    return K, d, rms, st, lb.fold_margin(Pb)


# =====================================================================================
def main():
    summ = dict(models={k: NICE.get(k, k) for k in MODELS}, ref=REF, sig=FITS["sig"], ic={}, chat={}, cv={}, mapping={},
                mapping_param_unc={}, conversion={}, cross=FITS["cross"])
    # ---- information criteria
    for ds in ("M", "E", "ME"):
        rows, chat = info_criteria(ds)
        summ["ic"][ds] = rows
        summ["chat"][ds] = chat
    # ---- CV
    for ds in ("M", "ME"):
        summ["cv"][f"loso_{ds}"] = cv_loso(ds)
    for ds in ("E", "ME"):
        for kind in ("src", "tile"):
            summ["cv"][f"region_{kind}_{ds}"] = cv_region(ds, kind)
    # ---- mapping differences vs the reference model on the same data set
    for ds in ("M", "E", "ME"):
        summ["mapping"][ds] = {}
        for name in MODELS:
            if name == REF:
                continue
            dc = mapping_vs(ds, REF, ds, name, True)
            dr = mapping_vs(ds, REF, ds, name, False)
            summ["mapping"][ds][name] = dict(rotcomp=L.region_stats(dc, UV), raw=L.region_stats(dr, UV))
    # cross-data (same model) E vs ME vs M
    summ["mapping"]["cross_data"] = {}
    for name in (REF, "KB_k1k2", "B_k1k2k3", "D_l1l2"):
        for a, b in (("E", "ME"), ("M", "ME"), ("M", "E")):
            dc = mapping_vs(a, name, b, name, True)
            summ["mapping"]["cross_data"][f"{name}:{a}->{b}"] = L.region_stats(dc, UV)
    # ---- parameter-sample mapping uncertainty (sandwich covariance)
    for ds in ("E", "ME"):
        summ["mapping_param_unc"][ds] = {}
        for name in MODELS:
            st, _, nf = sample_mapping_unc(ds, name)
            summ["mapping_param_unc"][ds][name] = dict(stats=st, n_folded_samples=nf)
    # ---- conversions to Brown
    for ds in ("E", "ME"):
        summ["conversion"][ds] = {}
        for name in MODELS:
            if MODELS[name]["kind"] == "brown" and not name.startswith("R_"):
                continue
            K, d, rms, st, fm = to_brown(ds, name)
            K3, d3, rms3, st3, fm3 = to_brown(ds, name, ("k1", "k2", "k3"))
            K2, d2, rms2, st2, fm2 = to_brown(ds, name, ("k1", "k2"))
            summ["conversion"][ds][name] = dict(
                full=dict(K=K.tolist(), dist=d.tolist(), fit_rms_px=rms, err=st, fold_margin=fm),
                k1k2k3=dict(K=K3.tolist(), dist=d3.tolist(), fit_rms_px=rms3, err=st3, fold_margin=fm3),
                k1k2=dict(K=K2.tolist(), dist=d2.tolist(), fit_rms_px=rms2, err=st2, fold_margin=fm2))
    save_json(summ, f"{CACHE}/altmodels_summary_raw.json")
    return summ


# =====================================================================================
# 5. outputs: plots, method files, summary
# =====================================================================================
PLOT_MODELS = ["B_k1", "B_k1k2k3", "B_k1k2p1p2", "B_k1k2_fxfy", "B_k1k2_ppfix", "B_k1k2k3p1p2_fxfy",
               "R_k1k2k4", "R_k1k2k3k4k5k6", "KB_k1k2", "KB_k1k2k3k4", "D_l1", "D_l1l2"]


def data_coverage():
    D = F.get_data()
    E = np.vstack([e["points"][::3] for e in D["edges"]])
    M = np.array([p["uv"] for p in D["pts"]])
    return E, M


def draw_regions(ax):
    from matplotlib.patches import Circle, Polygon, Rectangle

    ax.add_patch(Circle(((W - 1) / 2, (H - 1) / 2), 150, fill=False, ec="w", lw=0.8, ls="--"))
    for poly in ([(165, 100), (500, -5), (880, 560), (830, 1000), (560, 1010)],
                 [(1170, 0), (1720, 30), (1660, 380), (1540, 980), (1270, 990), (1130, 600)]):
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec="c", lw=0.8, ls="--"))
    for x0, y0 in ((0, 0), (W - 200, 0), (0, H - 150), (W - 200, H - 150)):
        ax.add_patch(Rectangle((x0, y0), 200, 150, fill=False, ec="y", lw=0.8, ls="--"))


def plot_mapping(summ, ds="ME"):
    Ecov, Mcov = data_coverage()
    n = len(PLOT_MODELS)
    fig, axs = plt.subplots(3, 4, figsize=(21, 11.2), dpi=110)
    fig.subplots_adjust(left=0.005, right=0.925, top=0.9, bottom=0.01, wspace=0.03, hspace=0.33)
    levels = [0.25, 0.5, 1, 2, 5, 10]
    xs = UV[:, 0].reshape(SHAPE)
    ys = UV[:, 1].reshape(SHAPE)
    im = None
    for ax, name in zip(axs.ravel(), PLOT_MODELS):
        d = mapping_vs(ds, REF, ds, name, True)
        m = np.hypot(d[:, 0], d[:, 1]).reshape(SHAPE)
        im = ax.imshow(np.log10(np.clip(m, 0.05, 50)), extent=(-20, W + 20, H + 20, -20), cmap="magma", vmin=np.log10(0.05), vmax=np.log10(20))
        cs = ax.contour(xs, ys, m, levels=levels, colors="w", linewidths=0.6)
        ax.clabel(cs, fmt="%g", fontsize=7)
        ax.plot(Ecov[:, 0], Ecov[:, 1], ",", color="lime", alpha=0.6)
        ax.plot(Mcov[:, 0], Mcov[:, 1], "+", color="deepskyblue", ms=3)
        draw_regions(ax)
        st = summ["mapping"][ds][name]["rotcomp"]
        raw = summ["mapping"][ds][name]["raw"]
        ax.set_title(f"{NICE[name]}\nrot-comp median: centre {st['centre']['median']:.2f}, band {st['cart_band']['median']:.2f}, "
                     f"corners {st['corners']['median']:.2f} px\nraw median: {raw['centre']['median']:.1f} / {raw['cart_band']['median']:.1f} / {raw['corners']['median']:.1f} px",
                     fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    cb = fig.colorbar(im, cax=fig.add_axes([0.935, 0.2, 0.012, 0.6]))
    cb.set_ticks(np.log10([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
    cb.set_ticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    cb.set_label("|mapping difference| vs Brown k1,k2 (pp free) [px], rotation-compensated")
    fig.suptitle(f"Lens-model choice: mapping of each model minus the reference Brown k1,k2 model, same data ({'stickers + edges' if ds == 'ME' else ds}); "
                 "green = edge points, blue + = sticker corners; dashed: centre / cart band / corner regions", fontsize=10)
    fig.savefig(f"{RESULTS}/altmodels_mapping_diff.png")
    plt.close(fig)


def plot_radial(summ, ds="ME"):
    Ecov, Mcov = data_coverage()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6), dpi=110)
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    rmax = np.hypot(W / 2, H / 2)
    rd_px = np.linspace(1, rmax + 20, 300)
    Pref = P_of(ds, REF)
    rI = np.hypot(UV[:, 0] - Pref["cx"], UV[:, 1] - Pref["cy"])
    bins = np.linspace(0, rmax + 40, 24)
    bc = 0.5 * (bins[1:] + bins[:-1])
    for j, name in enumerate([REF] + PLOT_MODELS):
        lens = lens_of(name)
        P = P_of(ds, name)
        xd = rd_px / P["fx"]
        xu, _ = lens.inv(xd, np.zeros_like(xd), dict(P, p1=0.0, p2=0.0) if lens.kind == "brown" else P)
        kw = dict(lw=2.4, color="k") if name == REF else dict(lw=1.1, color=cols[j % 20])
        ax[0].plot(rd_px, (xd - xu) * P["fx"], label=NICE[name], **kw)
        if name == REF:
            continue
        d = mapping_vs(ds, REF, ds, name, True)
        m = np.hypot(d[:, 0], d[:, 1])
        med = [np.median(m[(rI >= a) & (rI < b)]) if np.any((rI >= a) & (rI < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])]
        ax[1].semilogy(bc, med, "-o", ms=2.5, label=NICE[name], **kw)
    ax[0].set_xlabel("distorted radius from the model's principal point [px] (horizontal direction)")
    ax[0].set_ylabel("radial distortion r_d - r_u [px]")
    ax[0].axvline(rmax, color="grey", ls=":")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"radial distortion curves ({ds} data)", fontsize=10)
    ax[1].set_xlabel("image radius from the reference principal point [px]")
    ax[1].set_ylabel("median |mapping difference| vs Brown k1,k2 [px] (rot.-comp.)")
    ax[1].axvline(rmax, color="grey", ls=":")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].set_ylim(0.05, 100)
    ax[1].set_title("mapping difference to the reference vs radius", fontsize=10)
    rE = np.hypot(Ecov[:, 0] - Pref["cx"], Ecov[:, 1] - Pref["cy"])
    rM = np.hypot(Mcov[:, 0] - Pref["cx"], Mcov[:, 1] - Pref["cy"])
    uvf, shp = L.pixel_grid(10)
    rIf = np.hypot(uvf[:, 0] - Pref["cx"], uvf[:, 1] - Pref["cy"])
    b2 = np.linspace(0, rmax + 40, 45)
    ax[2].hist(rIf, bins=b2, density=True, alpha=0.3, color="grey", label="image area")
    ax[2].hist(rE, bins=b2, density=True, histtype="step", lw=1.5, color="g", label="edge points")
    ax[2].hist(rM, bins=b2, density=True, histtype="step", lw=1.5, color="b", label="sticker corners")
    ax[2].set_xlabel("radius from the principal point of the reference fit [px]")
    ax[2].set_ylabel("density")
    ax[2].legend(fontsize=8)
    ax[2].set_title("data support vs image area (radius)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/altmodels_radial.png")
    plt.close(fig)


def plot_cv(summ):
    names = list(MODELS)
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), dpi=110)
    y = np.arange(len(names))
    # (a) information criteria
    for j, (ds, c) in enumerate((("E", "g"), ("ME", "m"), ("M", "b"))):
        v = [summ["ic"][ds][n]["dQAIC"] if n in summ["ic"][ds] else np.nan for n in names]
        ax[0, 0].barh(y + (j - 1) * 0.27, np.clip(v, 0, 200), height=0.27, color=c, label=f"{ds}: dQAIC (c-hat {summ['chat'][ds]:.1f})")
    ax[0, 0].set_yticks(y)
    ax[0, 0].set_yticklabels([NICE[n] + (" *" if summ["ic"]["ME"][n]["barrier_active"] else "") for n in names], fontsize=8)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("dQAIC (clipped at 200); * = fold barrier active in ME")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, axis="x")
    # (b) leave-one-sticker-out
    for j, (ds, c) in enumerate((("M", "b"), ("ME", "m"))):
        cv = summ["cv"].get(f"loso_{ds}", {})
        v = [cv[n]["rms_px"] if n in cv else np.nan for n in names]
        ax[0, 1].barh(y + (j - 0.5) * 0.35, v, height=0.35, color=c, label=f"{ds}")
    ax[0, 1].set_yticks(y)
    ax[0, 1].set_yticklabels([NICE[n] for n in names], fontsize=8)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlabel("leave-one-sticker-out: RMS of predicted held-out corners [px]")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3, axis="x")
    # (c) region CV straightness, (d) VP
    for k, key in enumerate(("S_rms_px", "V_rms_px")):
        a = ax[1, k]
        for j, (tag, c) in enumerate((("region_tile_E", "g"), ("region_src_E", "lime"), ("region_tile_ME", "m"), ("region_src_ME", "violet"))):
            cv = summ["cv"].get(tag, {})
            v = [cv[n].get(key, np.nan) if n in cv else np.nan for n in names]
            a.barh(y + (j - 1.5) * 0.2, v, height=0.2, color=c, label=tag.replace("region_", ""))
        a.set_yticks(y)
        a.set_yticklabels([NICE[n] for n in names], fontsize=8)
        a.invert_yaxis()
        a.set_xlabel(("held-out edges: straightness RMS [px]" if key == "S_rms_px" else "held-out edges: VP-consistency RMS [px]") +
                     " (leave-one-region-out)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3, axis="x")
        vals = [cv[n].get(key, np.nan) for tag in ("region_tile_E", "region_tile_ME") for cv in [summ["cv"].get(tag, {})] for n in names if n in cv]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            a.set_xlim(min(vals) * 0.9, min(max(vals), 3 * min(vals)) * 1.05)
    fig.suptitle("Lens-model choice: information criteria and cross-validation (E = edges only, M = stickers only, ME = joint)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/altmodels_cv.png")
    plt.close(fig)


METHOD_NAMES = {"B_k1k2": "brown_k1k2", "B_k1": "brown_k1", "B_k1k2k3": "brown_k1k2k3", "B_k1k2p1p2": "brown_k1k2p1p2",
                "B_k1k2_fxfy": "brown_k1k2_fxfy", "B_k1k2_ppfix": "brown_k1k2_ppfixed", "R_k1k2k4": "rational_k4",
                "R_k1k2k3k4k5k6": "rational_k1k6", "KB_k1k2": "kb_k1k2", "KB_k1k2k3k4": "kb_k1k4", "D_l1": "division_l1",
                "D_l1l2": "division_l1l2"}


def region_med(st):
    return {k: round(v["median"], 3) for k, v in st.items()}


def adequacy(summ):
    """Models 'adequate to the data': valid lens (not folded, fold barrier inactive) on E and ME, and
    cross-validated no worse than the reference beyond 2 paired standard errors in every CV test."""
    out = {}
    for name in MODELS:
        why = []
        for ds in ("E", "ME"):
            f = FITS["fits"][ds][name]
            if f["barrier_active"] or f["fold_margin"] < 0:
                why.append(f"{ds}: fold barrier active (unconstrained optimum folded inside the image)")
        for tag, keys in (("region_tile_E", ("S", "V")), ("region_src_E", ("S", "V")), ("region_tile_ME", ("S", "V")), ("loso_ME", (None,))):
            cv = summ["cv"].get(tag, {})
            if name not in cv:
                continue
            for k in keys:
                if k is None:
                    d, se = cv[name]["dMS_vs_ref"], cv[name]["dMS_se"]
                else:
                    if f"{k}_dMS_vs_ref" not in cv[name]:
                        continue
                    d, se = cv[name][f"{k}_dMS_vs_ref"], cv[name][f"{k}_dMS_se"]
                if d > 2 * se and d > 0:
                    why.append(f"{tag} {k or 'corners'}: worse than reference by {d:.3g} px^2 (+-{se:.2g})")
        out[name] = dict(adequate=len(why) == 0, reasons=why)
    return out


def model_choice_unc(summ, rec, models, ds):
    """Per-region RMS over the adequate models of the rotation-compensated mapping difference to `rec`."""
    disp = np.array([mapping_vs(ds, rec, ds, m, True) for m in models if m != rec])
    if not len(disp):
        return None
    st, _ = mapping_stats(disp, UV)
    mx = np.max(np.hypot(disp[..., 0], disp[..., 1]), axis=0)
    for name, fn in REG.items():
        st[name]["max_over_models_median_px"] = float(np.median(mx[fn(UV)]))
    return st


def lens_record(summ, name, ds):
    """OpenCV Brown K, dist (converted for non-Brown / rational models) + conversion error."""
    lens = lens_of(name)
    P = P_of(ds, name)
    if lens.kind == "brown" and not name.startswith("R_"):
        return lens.K(P), lens.brown_dist(P)[:5], None
    c = summ["conversion"][ds][name]["full"]
    return np.array(c["K"]), np.array(c["dist"]), c


def write_method(summ, name, ds, rec, adq, mc_unc):
    f = FITS["fits"][ds][name]
    lens = lens_of(name)
    P = P_of(ds, name)
    K, d, conv = lens_record(summ, name, ds)
    pu = summ["mapping_param_unc"][ds][name]["stats"]
    unc = {f"{n}_1sigma": float(v) for n, v in f["sd_sandwich"].items()}
    if conv is not None:
        unc = {("native_" + k): v for k, v in unc.items()}
        unc["note"] = "1-sigma of the NATIVE parameters (see details.native_params); the OpenCV coefficients are a conversion"
    ic = {x: {k: round(summ["ic"][x][name][k], 1) for k in ("dAIC_own", "dQAIC", "dQBIC", "dAIC_det")} for x in ("M", "E", "ME")}
    cvs = {}
    for tag, cv in summ["cv"].items():
        if name in cv:
            cvs[tag] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cv[name].items() if k != "per_sticker_rms"}
    other = {}
    for x in ("M", "E"):
        g = FITS["fits"][x][name]
        other[x] = dict(params={k: round(v, 6) for k, v in g["P"].items()}, block_stats=g["block_stats"], fold_margin=g["fold_margin"],
                        barrier_active=g["barrier_active"], sd_sandwich=g["sd_sandwich"])
    out = lens_json(K, d,
                    method=f"alt_model_{METHOD_NAMES[name]} (lens-model comparison; stickers + edges jointly, same data and weights for every model)",
                    model=NICE[name] + (", pp free" if "ppfix" not in name else "") +
                    ("" if conv is None else " -> converted to the closest OpenCV Brown [k1,k2,p1,p2,k3] over the full image"),
                    rms_reprojection_error_px=f["block_stats"]["M"]["rms_radial"],
                    uncertainty=unc,
                    mapping_uncertainty_px=({k: round(pu[k]["rms_median_px"], 3) for k in ("centre", "cart_band", "corners")} if pu else None),
                    data_used="62 valid sticker corners of 20 stickers (7-frame means) + 61 verified straight edges (straightness; VP groups: cart X/Z axes "
                              "without end-board edges, scene verticals, 2 floor-line groups); block sigmas fixed from the Brown k1,k2 variance components",
                    geometry_assumptions="drawing dimensions and sticker positions exact (not fitted); edges use no dimensions",
                    details=dict(
                        native_params={k: float(v) for k, v in P.items()}, native_model=lens.spec(),
                        conversion_to_opencv=(None if conv is None else dict(fit_rms_px=conv["fit_rms_px"], error_px_by_region=region_med(conv["err"]),
                                                                            error_max_px={k: round(v["max"], 3) for k, v in conv["err"].items()},
                                                                            fold_margin=conv["fold_margin"])),
                        rational_8_coeffs=(lens.brown_dist(P).tolist() if name.startswith("R_") else None),
                        block_stats_joint=f["block_stats"], block_sigmas=FITS["sig"][ds], fold_margin=f["fold_margin"], barrier_active=f["barrier_active"],
                        uncertainty_method="cluster-robust sandwich covariance (clusters = stickers and edges) of the joint fit; mapping uncertainty = "
                                           f"{NSAMP} parameter samples, rotation-compensated displacement vs the fit, RMS over samples, median over region",
                        info_criteria=ic, cross_validation=cvs, other_data_sets=other,
                        mapping_vs_reference_rotcomp_median_px=(region_med(summ["mapping"][ds][name]["rotcomp"]) if name != REF else None),
                        mapping_vs_reference_raw_median_px=(region_med(summ["mapping"][ds][name]["raw"]) if name != REF else None),
                        model_choice_mapping_unc_px=mc_unc, adequate=adq[name], recommended_model=rec))
    tag = METHOD_NAMES[name]
    save_json(out, f"{CACHE}/method_alt_{tag}.json")
    md = [f"# method_alt_{tag}: {NICE[name]} (joint stickers + edges)", "",
          f"* K: f = {K[0, 0]:.1f} / {K[1, 1]:.1f} px, pp = ({K[0, 2]:.1f}, {K[1, 2]:.1f}); dist [k1,k2,p1,p2,k3] = {np.round(d, 5).tolist()}",
          f"* native: " + ", ".join(f"{k} = {v:.5g}" + (f" +- {f['sd_sandwich'][k if k in f['sd_sandwich'] else 'f']:.2g}" if k in f["sd_sandwich"] or k == "fx" else "")
                                  for k, v in P.items() if v != 0 or k in ("fx", "cx", "cy")),
          f"* blocks: " + ", ".join(f"{b} {v['rms']:.3f} px" for b, v in f["block_stats"].items()) + f"; sticker corners radial RMS {f['block_stats']['M']['rms_radial']:.2f} px",
          f"* dQAIC (M/E/ME): {ic['M']['dQAIC']} / {ic['E']['dQAIC']} / {ic['ME']['dQAIC']}; fold margin {f['fold_margin']:.3f}" + (" (BARRIER ACTIVE)" if f["barrier_active"] else ""),
          f"* adequate: {adq[name]['adequate']}" + ("" if adq[name]["adequate"] else " - " + "; ".join(adq[name]["reasons"])),
          (f"* mapping 1-sigma (parameter samples, rot.-comp.): " + ", ".join(f"{k} {v['rms_median_px']:.2f} px" for k, v in pu.items() if k != "whole_image")) if pu else "* mapping 1-sigma: n/a",
          ]
    if name != REF:
        md.append("* mapping vs Brown k1,k2 (same data, rot.-comp. median): " + ", ".join(f"{k} {v} px" for k, v in region_med(summ["mapping"][ds][name]["rotcomp"]).items()))
    if conv is not None:
        md.append("* conversion to OpenCV Brown (k1,k2,p1,p2,k3): error median " + ", ".join(f"{k} {v} px" for k, v in region_med(conv["err"]).items()))
    with open(f"{CACHE}/method_alt_{tag}.md", "w") as fh:
        fh.write("\n".join(md) + "\n")


def fmt_params(P, sd=None):
    out = []
    for k, v in P.items():
        if v == 0 and k not in ("fx", "cx", "cy"):
            continue
        if k == "fy" and P["fy"] == P["fx"]:
            continue
        e = ""
        if sd:
            kk = "f" if (k == "fx" and "f" in sd) else k
            if kk in sd:
                e = f"+-{sd[kk]:.2g}"
        out.append(f"{'f' if (k == 'fx' and P['fy'] == P['fx']) else k}={v:.4g}{e}")
    return ", ".join(out)


def write_summary(summ, rec, adq, mcu, conclusions):
    names = list(MODELS)
    md = ["# Lens-model choice (sub-task 43): which model is adequate to the data", "",
          "Scripts: `work/43_altmodels_lib.py` (lens models + estimator), `43_altmodels_fit.py` (fits + CV), "
          "`43_altmodels_report.py` (criteria, mappings, conversions, plots, method files), `43_altmodels_diaggeom.py` (diagnostics).",
          "Numbers: `work/cache/altmodels_fits.json`, `altmodels_summary.json`, `method_alt_*.json`; plots `results/altmodels_mapping_diff.png`, "
          "`results/altmodels_radial.png`, `results/altmodels_cv.png`.", "",
          "Data: M = 62 sticker corners (drawing geometry exact), E = 61 verified straight edges (straightness S + VP groups V: cart X/Z axes "
          "without end-board edges, free scene vertical, floor_V / floor_H perpendicular to it), ME = both. Block sigmas fixed per data set from the "
          f"Brown k1,k2 variance components: M {FITS['sig']['M']['M']:.2f} px; E: V {FITS['sig']['E']['V']:.3f}, S {FITS['sig']['E']['S']:.3f} px; "
          f"ME: M {FITS['sig']['ME']['M']:.2f}, V {FITS['sig']['ME']['V']:.3f}, S {FITS['sig']['ME']['S']:.3f} px. "
          "Every fit carries a fold barrier (radial map must stay invertible to the image corners, margin >= 0.03).", ""]
    md += ["## Conclusions", ""] + conclusions + [""]
    # fits
    md += ["## 1. Fits (same data, same weights for every model)", "",
           "| model | data | parameters (+- cluster-robust 1 sigma) | block RMS [px] | sticker radial RMS / max [px] | fold margin |", "|---|---|---|---|---|---|"]
    for ds in ("E", "ME", "M"):
        for n in names:
            f = FITS["fits"][ds][n]
            bs = f["block_stats"]
            md.append(f"| {NICE[n]} | {ds} | {fmt_params(f['P'], f['sd_sandwich'])} | " + ", ".join(f"{b} {v['rms']:.3f}" for b, v in bs.items()) +
                      f" | {(str(round(bs['M']['rms_radial'], 2)) + ' / ' + str(round(bs['M']['max_radial'], 1))) if 'M' in bs else '-'} | "
                      f"{f['fold_margin']:.3f}{' **barrier**' if f['barrier_active'] else ''} |")
    md += ["", "## 2. Information criteria (delta to the best model of the data set)", "",
           "AIC_own = sum_b n_b ln(RSS_b/n_b) + 2k (each model its own block variances); QAIC = chi2/c + 2k and QBIC = chi2/c + k ln(n/c) with "
           "chi2 at the fixed block sigmas and c = overdispersion / design effect (median ratio sandwich/naive variance of the intrinsics of the "
           f"reference fit: M {summ['chat']['M']:.1f}, E {summ['chat']['E']:.1f}, ME {summ['chat']['ME']:.1f}) - residuals of one sticker / one edge are "
           f"strongly correlated. AIC_det uses the detection noise only ({DET_SIGMA} px) - shown to demonstrate that it always picks the most flexible model.", "",
           "| model | k_intr | E: dAIC_own / dQAIC / dQBIC / dAIC_det | ME: dAIC_own / dQAIC / dQBIC | M: dAIC_own / dQAIC / dQBIC |", "|---|---|---|---|---|"]
    for n in names:
        e, j, m = summ["ic"]["E"][n], summ["ic"]["ME"][n], summ["ic"]["M"][n]
        md.append(f"| {NICE[n]} | {e['k_intr']} | {e['dAIC_own']:.0f} / {e['dQAIC']:.1f} / {e['dQBIC']:.1f} / {e['dAIC_det']:.0f} | "
                  f"{j['dAIC_own']:.0f} / {j['dQAIC']:.1f} / {j['dQBIC']:.1f} | {m['dAIC_own']:.1f} / {m['dQAIC']:.1f} / {m['dQBIC']:.1f} |")
    md += ["", "## 3. Cross-validation", "",
           "Leave-one-sticker-out (LOSO): refit without the sticker, predict its corners (pose of its cart from the other stickers). "
           "Leave-one-edge-region-out: source regions (cart310 / cart80 / scene) and 3x3 image tiles; held-out edges are predicted for "
           "straightness (S, own line) and for VP consistency (V, direction from the training fit). dMS = mean over held-out clusters of the "
           "difference in mean-square error vs Brown k1,k2 (paired), +- its standard error.", "",
           "| model | LOSO M: RMS [px] | LOSO ME: RMS [px], dMS +- se | E tiles: S / V RMS [px] | E tiles: dMS S, V (+-se) [px^2] | E src: S / V | ME tiles: S / V | adequate |",
           "|---|---|---|---|---|---|---|---|"]
    cv = summ["cv"]

    def g(tag, n, k, fmt="{:.3f}"):
        try:
            return fmt.format(cv[tag][n][k])
        except KeyError:
            return "-"
    for n in names:
        md.append(f"| {NICE[n]} | {g('loso_M', n, 'rms_px', '{:.2f}')} | {g('loso_ME', n, 'rms_px', '{:.2f}')}, {g('loso_ME', n, 'dMS_vs_ref', '{:+.2f}')} +- {g('loso_ME', n, 'dMS_se', '{:.2f}')} | "
                  f"{g('region_tile_E', n, 'S_rms_px')} / {g('region_tile_E', n, 'V_rms_px')} | "
                  f"{g('region_tile_E', n, 'S_dMS_vs_ref', '{:+.4f}')} +- {g('region_tile_E', n, 'S_dMS_se', '{:.4f}')}, {g('region_tile_E', n, 'V_dMS_vs_ref', '{:+.4f}')} +- {g('region_tile_E', n, 'V_dMS_se', '{:.4f}')} | "
                  f"{g('region_src_E', n, 'S_rms_px')} / {g('region_src_E', n, 'V_rms_px')} | {g('region_tile_ME', n, 'S_rms_px')} / {g('region_tile_ME', n, 'V_rms_px')} | "
                  f"{'yes' if adq[n]['adequate'] else 'no'} |")
    md += ["", "Cross-data prediction (lens of one data set, poses/rotations refit on the other):", "",
           "| model | lens(E) -> stickers: radial RMS [px] | lens(M) -> edges: S / V RMS [px] |", "|---|---|---|"]
    for n in names:
        c = summ["cross"][n]
        md.append(f"| {NICE[n]} | {c['E_to_M']['M']['rms_radial']:.2f} | {c['M_to_E']['S']['rms']:.3f} / {c['M_to_E']['V']['rms']:.3f} |")
    md += ["", "## 4. Mapping differences between models (not parameters)", "",
           "Displacement of each model's mapping relative to Brown k1,k2 fitted to the same data (rays of the reference, projected by the "
           "model; rotation-compensated = best common camera rotation removed, raw = none). Median |d| over the region (centre r < 150 px, "
           "cart band = the two cart footprints, corners = 200 x 150 px corner boxes).", "",
           "| model | ME rot-comp: centre / band / corners | ME raw: centre / band / corners | E rot-comp: centre / band / corners | M rot-comp: centre / band / corners |",
           "|---|---|---|---|---|"]
    for n in names:
        if n == REF:
            continue
        r = [summ["mapping"][ds][n][kind] for ds, kind in (("ME", "rotcomp"), ("ME", "raw"), ("E", "rotcomp"), ("M", "rotcomp"))]
        md.append("| " + NICE[n] + " | " + " | ".join(f"{x['centre']['median']:.2f} / {x['cart_band']['median']:.2f} / {x['corners']['median']:.2f}" for x in r) + " |")
    md += ["", "Same model, different data (rot-comp median centre / band / corners [px]):", ""]
    for k, v in summ["mapping"]["cross_data"].items():
        md.append(f"* {k}: {v['centre']['median']:.2f} / {v['cart_band']['median']:.2f} / {v['corners']['median']:.2f}")
    md += ["", "Parameter-sample mapping uncertainty (cluster-robust sandwich covariance, rotation-compensated, 1 sigma, median over region) [px]:", "",
           "| model | E: centre / band / corners | ME: centre / band / corners |", "|---|---|---|"]
    for n in names:
        row = []
        for ds in ("E", "ME"):
            st = summ["mapping_param_unc"][ds][n]["stats"]
            row.append("-" if st is None else f"{st['centre']['rms_median_px']:.2f} / {st['cart_band']['rms_median_px']:.2f} / {st['corners']['rms_median_px']:.2f}")
        md.append(f"| {NICE[n]} | {row[0]} | {row[1]} |")
    if mcu:
        md += ["", f"**Model-choice part of the mapping uncertainty** (RMS over the adequate models {', '.join(NICE[m] for m in mcu['models'])} of the rot.-comp. "
               f"difference to the recommended {NICE[rec]}, median over region):", ""]
        for ds in ("E", "ME"):
            st = mcu[ds]
            if st:
                md.append(f"* {ds}: centre {st['centre']['rms_median_px']:.2f}, cart band {st['cart_band']['rms_median_px']:.2f}, corners {st['corners']['rms_median_px']:.2f} px "
                          f"(max over models: {st['centre']['max_over_models_median_px']:.2f} / {st['cart_band']['max_over_models_median_px']:.2f} / {st['corners']['max_over_models_median_px']:.2f} px)")
    md += ["", "## 5. Conversion of non-Brown models to OpenCV Brown", "",
           "evaltools.brown_from_mapping on a 20 px grid over the full image (f, pp and the coefficients refitted); error = |pixel difference| "
           "of the same rays, median (max) over the region.", "",
           "| model (data) | target | K f / pp | dist [k1,k2,p1,p2,k3] | error centre / band / corners: median (max) [px] |", "|---|---|---|---|---|"]
    for ds in ("ME", "E"):
        for n, c in summ["conversion"][ds].items():
            for t in ("full", "k1k2k3", "k1k2"):
                x = c[t]
                K = np.array(x["K"])
                md.append(f"| {NICE[n]} ({ds}) | {t} | {K[0, 0]:.1f} / ({K[0, 2]:.1f}, {K[1, 2]:.1f}) | {np.round(x['dist'], 4).tolist()} | " +
                          " / ".join(f"{x['err'][r]['median']:.3f} ({x['err'][r]['max']:.2f})" for r in ("centre", "cart_band", "corners")) + " |")
    with open(f"{CACHE}/altmodels_summary.md", "w") as fh:
        fh.write("\n".join(md) + "\n")


REC = "B_k1k2"


def conclusions_text(summ, adq, mcu):
    return ["(filled in after the results were inspected)"]


if __name__ == "__main__":
    S = main()
    plot_mapping(S, "ME")
    plot_radial(S, "ME")
    plot_cv(S)
    ADQ = adequacy(S)
    adequate = [n for n in MODELS if ADQ[n]["adequate"]]
    MCU = dict(models=adequate, recommended=REC)
    for ds in ("E", "ME"):
        MCU[ds] = model_choice_unc(S, REC, adequate, ds)
    S["adequacy"] = ADQ
    S["recommended"] = REC
    S["model_choice_mapping_unc"] = MCU
    for n in METHOD_NAMES:
        write_method(S, n, "ME", REC, ADQ, {ds: ({k: round(v["rms_median_px"], 3) for k, v in MCU[ds].items() if k != "whole_image"} if MCU[ds] else None)
                                           for ds in ("E", "ME")})
    CON = conclusions_text(S, ADQ, MCU)
    S["conclusions"] = CON
    save_json(S, f"{CACHE}/altmodels_summary.json")
    write_summary(S, REC, ADQ, MCU, CON)
    print("adequate:", adequate)
