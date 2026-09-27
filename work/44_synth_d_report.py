"""Synthetic study, step D: statistics, honesty of the claimed uncertainties, plots and the report.

Reads work/cache/synthetic_setup.json, synthetic_roundtrip_raw.pkl, synthetic_sensitivity.json.
Writes work/cache/synthetic_report.json / .md and results/synthetic_*.png.
"""
import importlib
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, RESULTS, load_json, save_json

L = importlib.import_module("44_synth_lib")
SETUP = load_json(f"{CACHE}/synthetic_setup.json")
import glob

RAW = None
for _fn in sorted(glob.glob(f"{CACHE}/synthetic_roundtrip_raw*.pkl")):  # main file + resumed parts
    _d = pickle.load(open(_fn, "rb"))
    if RAW is None:
        RAW = _d
    else:
        for _s, _reps in _d["out"].items():
            RAW["out"].setdefault(_s, {}).update(_reps)
import os

SENS = load_json(f"{CACHE}/synthetic_sensitivity.json") if os.path.exists(f"{CACHE}/synthetic_sensitivity.json") else None
OUT = RAW["out"]
SCEN = [s for s in ("T1_a", "T2_a", "T1_b", "T2_b", "T2_c") if OUT.get(s)]
EST = ["markers_k1_ppfix", "markers_k1k2_ppfix", "markers_k1_ppfree", "markers_k1k2_ppfree", "plumb_fixed", "plumb_free", "vp",
       "lines_joint", "combined_k1k2"]
SHORT = {"markers_k1_ppfix": "M k1 ppfix", "markers_k1k2_ppfix": "M k1k2 ppfix", "markers_k1_ppfree": "M k1 ppfree",
         "markers_k1k2_ppfree": "M k1k2 ppfree", "plumb_fixed": "plumb c-fixed", "plumb_free": "plumb c-free", "vp": "VP",
         "lines_joint": "lines joint", "combined_k1k2": "combined"}
REG = ["centre", "cart_band", "corners"]
CLAIM_KIND = {"markers": ("C", "Cs", "boot"), "plumb_fixed": ("Cs", "boot"), "plumb_free": ("Cs",), "vp": ("Cs", "boot"),
              "lines_joint": ("Cs", "boot"), "combined_k1k2": ("C", "boot")}
CLAIM_NAME = {"C": "covariance (scaled by residual variance)", "Cs": "cluster sandwich (per sticker / per edge)", "boot": "cluster bootstrap"}


def claims_of(est):
    return CLAIM_KIND["markers"] if est.startswith("markers") else CLAIM_KIND[est]


def rsd(v):
    """robust std (1.4826 MAD)"""
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")


def fl(v):
    return None if v is None or not np.isfinite(v) else float(v)


def analyse(scen, est):
    reps = [OUT[scen][r][est] for r in sorted(OUT[scen]) if est in OUT[scen][r]]
    if not reps:
        return None
    names = reps[0]["names"]
    X = np.array([r["x"] for r in reps])
    T = reps[0]["truth"]
    err = X - T
    out = dict(n=len(reps), names=names, truth=T.tolist(), mean=X.mean(0).tolist(), bias=err.mean(0).tolist(),
               bias_se=(err.std(0, ddof=1) / np.sqrt(len(reps))).tolist(), scatter=err.std(0, ddof=1).tolist(),
               scatter_robust=[rsd(err[:, j]) for j in range(err.shape[1])], rmse=np.sqrt(np.mean(err ** 2, 0)).tolist(),
               claimed={}, z_std={}, coverage68={})
    for ck in claims_of(est):
        key = "sig_" + ck
        sg = [r[key] for r in reps if key in r and r[key] is not None and len(r[key]) == len(names)]
        if not sg:
            continue
        sg = np.array(sg)
        out["claimed"][ck] = dict(median=np.median(sg, 0).tolist(), n=len(sg))
        if ck != "boot":
            z = err / np.where(sg > 0, sg, np.nan)
            out["z_std"][ck] = np.sqrt(np.nanmean(z ** 2, 0)).tolist()  # RMS of z (includes bias)
            out["coverage68"][ck] = np.nanmean(np.abs(z) < 1, 0).tolist()
        else:
            out["claimed"][ck]["median_robust"] = np.median([[rsd(r["boot_x"][:, j]) for j in range(len(names))] for r in reps
                                                             if "boot_x" in r and len(r["boot_x"])], 0).tolist()
    # mapping
    D = np.array([r["disp"] for r in reps], float)  # R x N x 2
    mean_f = np.nanmean(D, 0)
    tot = np.sqrt(np.nanmean(np.sum(D ** 2, 2), 0))
    bias = np.linalg.norm(mean_f, axis=1)
    scat = np.sqrt(np.nanmean(np.sum((D - mean_f[None]) ** 2, 2), 0))
    mp = dict(total=L.region_summary(tot), bias=L.region_summary(bias), scatter=L.region_summary(scat), claimed={})
    for ck in claims_of(est):
        key = "claim_map_" + ck
        cm = [r[key] for r in reps if r.get(key) is not None]
        if cm:
            mp["claimed"][ck] = L.region_summary(np.nanmedian(np.array(cm, float), 0))
            mp["claimed"][ck]["n_rep"] = len(cm)
    out["mapping"] = mp
    out["_fields"] = dict(mean=mean_f, tot=tot)
    if est.startswith("markers"):
        xo = np.array([r["x_orchestrator_start"] for r in reps])
        dd = np.abs(xo[:, 0] - X[:, 0])
        ro = np.array([r["rms_orchestrator_start"] for r in reps])
        rr = np.array([r["rms"] for r in reps])
        out["start_check"] = dict(n_differs_f_gt_0p5px=int((dd > 0.5).sum()), n_orchestrator_start_worse=int(((dd > 0.5) & (ro > rr + 1e-6)).sum()),
                                  median_rms=float(np.median(rr)), rms_p10_p90=[float(np.percentile(rr, 10)), float(np.percentile(rr, 90))])
    if est == "combined_k1k2":
        br = {k: np.median([r["block_rms"][k][0] for r in reps if r["block_rms"][k][0] is not None]) for k in "MLVS"}
        out["block_rms_median"] = {k: fl(v) for k, v in br.items()}
    _rv = [r["rms"] for r in reps if r.get("rms") is not None]
    if _rv:
        out["rms_median"] = float(np.median(_rv))
    fails = [r for r in reps if r.get("failed_start_rms") is not None or r.get("failed_start") is not None or r.get("failed_start_S_rms") is not None]
    out["n_failed_from_perturbed_start"] = len(fails)
    if fails:
        out["failed_examples"] = [str(r.get("failed_start_rms", r.get("failed_start", r.get("failed_start_S_rms")))) for r in fails][:5]
    return out


A = {s: {e: analyse(s, e) for e in EST} for s in SCEN}


def honesty_summary():
    """compact: per scenario / estimator: f (or k1) actual RMSE vs each claim, mapping cart band / corners actual vs claims."""
    out = {}
    for s in SCEN:
        for e in EST:
            a = A[s].get(e)
            if a is None:
                continue
            j = a["names"].index("f") if "f" in a["names"] else a["names"].index("k1")
            row = dict(param=a["names"][j], truth=a["truth"][j], bias=a["bias"][j], scatter=a["scatter"][j], rmse=a["rmse"][j],
                       claimed={k: v["median"][j] for k, v in a["claimed"].items()},
                       map_actual={rg: a["mapping"]["total"][rg]["median"] for rg in REG},
                       map_bias={rg: a["mapping"]["bias"][rg]["median"] for rg in REG},
                       map_claimed={ck: {rg: c[rg]["median"] for rg in REG} for ck, c in a["mapping"]["claimed"].items()})
            row["ratio_rmse_over_claim"] = {k: (row["rmse"] / v if v else None) for k, v in row["claimed"].items()}
            row["ratio_map_cart_band_over_claim"] = {k: (row["map_actual"]["cart_band"] / v["cart_band"] if v["cart_band"] else None)
                                                     for k, v in row["map_claimed"].items()}
            out.setdefault(s, {})[e] = row
    return out


HON = honesty_summary()


def htab():
    lines = ["| scenario | estimator | param | bias | scatter | RMSE | claimed cov / sandwich / boot | RMSE / claim | mapping cart band: actual (bias) | claimed cov / sandwich / boot | corners: actual | claimed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in HON.items():
        for e, r in d.items():
            nd = 4 if r["param"].startswith("k") else 1
            cl = " / ".join(f"{r['claimed'][k]:.{nd}f}" if k in r["claimed"] else "-" for k in ("C", "Cs", "boot"))
            rt = " / ".join(f"{r['ratio_rmse_over_claim'][k]:.1f}" if r["ratio_rmse_over_claim"].get(k) else "-" for k in ("C", "Cs", "boot"))
            mc = " / ".join(f"{r['map_claimed'][k]['cart_band']:.2f}" if k in r["map_claimed"] and r["map_claimed"][k]["cart_band"] is not None else "-" for k in ("C", "Cs", "boot"))
            mco = " / ".join(f"{r['map_claimed'][k]['corners']:.2f}" if k in r["map_claimed"] and r["map_claimed"][k]["corners"] is not None else "-" for k in ("C", "Cs", "boot"))
            lines.append(f"| {s} | {SHORT[e]} | {r['param']} | {r['bias']:+.{nd}f} | {r['scatter']:.{nd}f} | {r['rmse']:.{nd}f} | {cl} | {rt} | "
                         f"{r['map_actual']['cart_band']:.2f} ({r['map_bias']['cart_band']:.2f}) | {mc} | {r['map_actual']['corners']:.2f} | {mco} |")
    return "\n".join(lines)

# ------------------------------------------------------------------ tables (markdown)
def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float) and abs(v) < 0.1 and v != 0:
        return f"{v:.4f}"
    return f"{v:.{nd}f}"


def ptab(scen):
    lines = ["| estimator | param | truth | bias +- se | scatter (robust) | claimed 1-sigma: cov / sandwich / bootstrap (median over replicates) "
             "| RMS z (cov / sandwich) | coverage of +-1 sigma (cov / sandwich) |", "|---|---|---|---|---|---|---|---|"]
    for e in EST:
        a = A[scen].get(e)
        if a is None:
            continue
        for j, nm in enumerate(a["names"]):
            cl = []
            for ck in ("C", "Cs", "boot"):
                c = a["claimed"].get(ck)
                cl.append(fmt(c["median"][j], 2) if c else "-")
            zs = " / ".join(fmt(a["z_std"][ck][j], 2) if ck in a["z_std"] else "-" for ck in ("C", "Cs"))
            cv = " / ".join(f"{a['coverage68'][ck][j]:.2f}" if ck in a["coverage68"] else "-" for ck in ("C", "Cs"))
            nd = 4 if nm.startswith("k") else 1
            lines.append(f"| {SHORT[e]} | {nm} | {a['truth'][j]:.{nd}f} | {a['bias'][j]:+.{nd}f} +- {a['bias_se'][j]:.{nd}f} | "
                         f"{a['scatter'][j]:.{nd}f} ({a['scatter_robust'][j]:.{nd}f}) | {' / '.join(cl)} | {zs} | {cv} |")
    return "\n".join(lines)


def mtab(scen):
    lines = ["| estimator | region | actual RMS error | of which bias | scatter | claimed: cov / sandwich / bootstrap | actual / claimed |",
             "|---|---|---|---|---|---|---|"]
    for e in EST:
        a = A[scen].get(e)
        if a is None:
            continue
        mp = a["mapping"]
        for rg in REG:
            cl = []
            ratios = []
            for ck in ("C", "Cs", "boot"):
                c = mp["claimed"].get(ck)
                v = c[rg]["median"] if c else None
                cl.append(fmt(v, 2) if v is not None else "-")
                if v:
                    ratios.append(f"{mp['total'][rg]['median'] / v:.1f}")
            lines.append(f"| {SHORT[e]} | {rg} | {mp['total'][rg]['median']:.2f} | {mp['bias'][rg]['median']:.2f} | {mp['scatter'][rg]['median']:.2f} | "
                         f"{' / '.join(cl)} | {' / '.join(ratios) if ratios else '-'} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ plots
cols = {"T1_a": "#1f77b4", "T2_a": "#2ca02c", "T1_b": "#ff7f0e", "T2_b": "#d62728", "T2_c": "#9467bd"}


def plot_params():
    pars = ["f", "cx", "cy", "k1"]
    fig, axs = plt.subplots(len(pars), 1, figsize=(15, 13), dpi=110)
    for ax, pn in zip(axs, pars):
        for i, e in enumerate(EST):
            for j, s in enumerate(SCEN):
                a = A[s].get(e)
                if a is None or pn not in a["names"]:
                    continue
                k = a["names"].index(pn)
                xpos = i + (j - (len(SCEN) - 1) / 2) * 0.15
                b, sc = a["bias"][k], a["scatter"][k]
                ax.errorbar(xpos, b, yerr=sc, fmt="o", color=cols[s], ms=4, capsize=2, label=s if (i == 0 or pn not in A[s][EST[0]]["names"]) and ax is axs[0] else None)
                cl = a["claimed"].get("Cs") or a["claimed"].get("C")
                if cl:
                    ax.plot([xpos - 0.05, xpos + 0.05], [b + cl["median"][k]] * 2, "-", color="k", lw=1.2)
                    ax.plot([xpos - 0.05, xpos + 0.05], [b - cl["median"][k]] * 2, "-", color="k", lw=1.2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlim(-0.5, len(EST) - 0.5)
        ax.set_xticks(range(len(EST)))
        ax.set_xticklabels([SHORT[e] for e in EST], fontsize=9)
        ax.set_ylabel(f"{pn}: estimate - truth" + (" [px]" if pn != "k1" else "\n(plumb-line: k1 at f0 = 1300)"))
        ax.grid(alpha=0.3)
        if pn in ("f", "cx", "cy"):
            lim = ax.get_ylim()
            ax.set_ylim(max(lim[0], -250), min(lim[1], 250))
    axs[0].legend(ncol=5, fontsize=8, loc="upper left")
    fig.suptitle("Synthetic round trip: bias (dot) +- actual 1-sigma scatter (bar) vs claimed 1-sigma (black ticks: sandwich, else covariance)\n"
                 "a = detection noise only, b = + unknown sticker / row placement errors + edge bows / direction errors, c = diagnosed board heights", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/synthetic_params.png")
    plt.close(fig)


def plot_mapping():
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), dpi=110, sharey=True)
    for ax, rg in zip(axs, REG):
        for i, e in enumerate(EST):
            for j, s in enumerate(SCEN):
                a = A[s].get(e)
                if a is None:
                    continue
                xpos = i + (j - (len(SCEN) - 1) / 2) * 0.15
                tot = a["mapping"]["total"][rg]["median"]
                ax.bar(xpos, tot, width=0.14, color=cols[s])
                for ck, mk in (("C", "_"), ("Cs", "x"), ("boot", "^")):
                    c = a["mapping"]["claimed"].get(ck)
                    if c and c[rg]["median"]:
                        ax.plot(xpos, c[rg]["median"], mk, color="k", ms=5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(EST)))
        ax.set_xticklabels([SHORT[e] for e in EST], rotation=60, fontsize=8)
        ax.set_title(f"{rg}: actual RMS mapping error vs truth (bars)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("px (rotation-compensated, median over region)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Patch(color=cols[s], label=f"actual, {s}") for s in SCEN] + [
        Line2D([], [], marker=m, ls="", color="k", label=l) for m, l in (("_", "claimed: covariance"), ("x", "claimed: sandwich"), ("^", "claimed: bootstrap"))]
    axs[0].legend(handles=hd, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Synthetic round trip: actual mapping error vs the uncertainty the estimator claims (markers: cov -, sandwich x, bootstrap ^)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/synthetic_mapping.png")
    plt.close(fig)


def plot_bias_fields():
    sel = [(s, e) for s in ("T2_a", "T2_b") for e in ("markers_k1k2_ppfree", "lines_joint", "combined_k1k2")]
    fig, axs = plt.subplots(2, 3, figsize=(18, 7.5), dpi=110)
    uv = L.UV
    step = 2
    shp = L.UV_SHAPE
    sub = np.zeros(shp, bool)
    sub[::step, ::step] = True
    sub = sub.ravel()
    for ax, (s, e) in zip(axs.ravel(), sel):
        a = A.get(s, {}).get(e)
        if a is None:
            continue
        mf = a["_fields"]["mean"]
        tot = a["_fields"]["tot"]
        im = ax.imshow(tot.reshape(shp), extent=(uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].max(), uv[:, 1].min()), cmap="viridis",
                       vmin=0, vmax=np.nanpercentile(tot, 95))
        sc = 30.0 / max(np.nanpercentile(np.linalg.norm(mf, axis=1), 95), 1e-3)
        ax.quiver(uv[sub, 0], uv[sub, 1], mf[sub, 0] * sc, mf[sub, 1] * sc, angles="xy", scale_units="xy", scale=1, color="w", width=0.002)
        plt.colorbar(im, ax=ax, fraction=0.03, label="RMS error [px]")
        ax.set_title(f"{s} {SHORT[e]}: RMS mapping error (colour), mean error (arrows x{sc:.0f})", fontsize=9)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/synthetic_mapping_fields.png")
    plt.close(fig)


def plot_sensitivity():
    rows = []
    for kind in ("T2", "real"):
        for key, v in SENS[kind].items():
            rows.append((kind, key, v))
    perts = [k for k in SENS["T2"].keys() if not k.startswith("frame")] + ["frame (7 stills, RMS)"]
    ests = ["markers_k1_ppfix", "markers_k1k2_ppfix", "markers_k1k2_ppfree", "plumb_fixed", "plumb_free", "vp", "lines_joint", "combined_k1k2"]
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), dpi=110)
    for ax, kind in zip(axs, ("T2", "real")):
        for i, p in enumerate(perts):
            for j, e in enumerate(ests):
                if p.startswith("frame"):
                    vals = [SENS[kind][f"frame{f}"]["result"].get(e, {}).get("map", {}).get("cart_band") for f in range(7)]
                    vals = [v for v in vals if v is not None]
                    v = float(np.sqrt(np.mean(np.square(vals)))) if vals else None
                else:
                    v = SENS[kind][p]["result"].get(e, {}).get("map", {}).get("cart_band")
                if v is None:
                    continue
                ax.bar(i + (j - 3.5) * 0.1, v, width=0.1, color=plt.cm.tab10(j), label=SHORT[e] if i == 0 or (p.startswith("edge") and not any(
                    SENS[kind][q]["result"].get(e) for q in perts[:i] if not q.startswith("frame"))) else None)
        ax.set_yscale("log")
        ax.set_xticks(range(len(perts)))
        ax.set_xticklabels(perts, rotation=30, fontsize=8, ha="right")
        ax.set_ylabel("mapping change, cart band [px]\n(median |displacement|, rot.-compensated)")
        ax.set_title(f"{'synthetic T2 (noise-free)' if kind == 'T2' else 'REAL data'}: change of the estimated mapping caused by each perturbation (absolute, not per 0.1 px)")
        ax.grid(alpha=0.3, which="both")
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/synthetic_sensitivity.png")
    plt.close(fig)


plot_params()
plot_mapping()
plot_bias_fields()
if SENS:
    plot_sensitivity()

# ------------------------------------------------------------------ sensitivity tables
def stab(kind):
    lines = ["| perturbation | size | estimator | d f | d cx | d cy | d k1 | d k2 | map change centre / cart band / corners [px] | per 0.1 px (cart band) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in SENS[kind].items():
        if key.startswith("frame") and key != "frame0":
            continue
        sz = v.get("perturbation_size")
        szs = f"rms {sz['rms_px']:.2f} px" if sz else ("0.3 px along the normal" if key.startswith("edge") else "-")
        for e, r in v["result"].items():
            if e in ("markers_k1_ppfree",):
                continue
            dp = dict(zip(r["names"], r["dparams"]))
            mp = r["map"]
            per = r.get("map_per_0p1px", {}).get("cart_band")
            lines.append(f"| {key} | {szs} | {SHORT[e]} | {fmt(dp.get('f'), 2)} | {fmt(dp.get('cx'), 2)} | {fmt(dp.get('cy'), 2)} | {fmt(dp.get('k1'), 4)} | "
                         f"{fmt(dp.get('k2'), 4)} | {fmt(mp['centre'], 2)} / {fmt(mp['cart_band'], 2)} / {fmt(mp['corners'], 2)} | {fmt(per, 2)} |")
    return "\n".join(lines)


def frame_summary(kind):
    out = {}
    for e in EST:
        vals = [SENS[kind][f"frame{f}"]["result"].get(e) for f in range(7)]
        vals = [v for v in vals if v]
        if not vals:
            continue
        dp = np.array([v["dparams"] for v in vals])
        out[e] = dict(names=vals[0]["names"], rms_dparams=np.sqrt(np.mean(dp ** 2, 0)).tolist(),
                      rms_map={r: float(np.sqrt(np.mean([v["map"][r] ** 2 for v in vals if v["map"][r] is not None]))) for r in REG})
    return out


FR = {k: frame_summary(k) for k in ("T2", "T1", "real")} if SENS else {}
SUB = SENS.get("subsample", {}) if SENS else {}


def sub_summary():
    out = {}
    for kind, d in SUB.items():
        for key, reps in d.items():
            for e in EST:
                vals = [reps[r].get(e) for r in reps if reps[r].get(e)]
                if not vals:
                    continue
                dp = np.array([v["dparams"] for v in vals])
                out.setdefault(kind, {}).setdefault(key, {})[e] = dict(names=vals[0]["names"], n=len(vals), rms_dparams=np.sqrt(np.mean(dp ** 2, 0)).tolist(),
                                                                       rms_map={r: float(np.sqrt(np.mean([v["map"][r] ** 2 for v in vals if v["map"][r] is not None]))) for r in REG})
    return out


SUBS = sub_summary()

# ------------------------------------------------------------------ JSON
rep_json = dict(
    description="Synthetic round trip and sensitivity of the orchestrator estimators (scripts work/44_synth_*.py)",
    edge_files_review_status=SETUP["review_status"], n_edges=SETUP["n_edges"], n_sticker_sides=SETUP["n_sides"], n_corners=SETUP["n_corners"],
    truths={k: {kk: vv for kk, vv in v.items() if kk not in ("poses",)} for k, v in SETUP["truths"].items()},
    noise_models=dict(
        a="corners: per-coordinate sigma = per-frame radial std / sqrt(2*7) (median %.3f px); edges: AR(1) white noise with the per-edge sigma "
          "(median %.3f px) and lag-1 correlation (median %.2f) of the high-frequency part of the real residuals" % (
              SETUP["corner_sigma_median"], SETUP["edge_stats_summary"]["sigma_median"], SETUP["edge_stats_summary"]["rho_median"]),
        b="a + per-sticker 3D placement error sigma %.2f mm (isotropic; moves corners and traced sides) + per cart-row height error sigma %.1f mm "
          "(calibrated: sticker-only part -> %.2f px, total drawing-fit RMS median %.2f px vs real %.2f px) + per-edge bow = real low-frequency residual profile "
          "(median rms %.3f px) with random sign + per-edge direction deviation (VP-group edges) = real angle to the reference VP (median %.3f deg, max %.2f deg) "
          "with random sign" % (SETUP["geometry_noise"]["sigma_sticker_mm"], SETUP["geometry_noise"]["sigma_row_mm"], SETUP["geometry_noise"]["target_sticker_part_px"],
                                SETUP["geometry_noise"]["synthetic_median_rms_px"], SETUP["geometry_noise"]["real_rms_px"], SETUP["edge_stats_summary"]["bow_rms_median"],
                                SETUP["edge_stats_summary"]["theta_abs_median_deg"], SETUP["edge_stats_summary"]["theta_abs_max_deg"]),
        c="deterministic what-if: top stickers raised by the diagnosed end-board heights (+64/+71/+61/+81 mm) + noise a; median drawing-fit RMS %.2f px" %
          SETUP["geometry_noise"]["diagnosed_whatif_rms_px"]),
    honesty_summary=HON,
    roundtrip={s: {e: {k: v for k, v in a.items() if k != "_fields"} for e, a in A[s].items() if a is not None} for s in SCEN},
    sensitivity=dict(T2=SENS["T2"], T1=SENS["T1"], real=SENS["real"], frames_rms=FR, subsample=SUBS, notes=SENS["notes"]) if SENS else None,
    real_reference=SETUP["real"],
)
save_json(rep_json, f"{CACHE}/synthetic_report.json")

md = ["# Synthetic round trip + sensitivity of the pipeline (work/44_synth_*.py)", "",
      f"Edge files: {SETUP['review_status']}; {SETUP['n_edges']} structural edges + {SETUP['n_sides']} sticker sides; {SETUP['n_corners']} valid corners of 20 stickers.",
      "", "## Truths and noise models", ""]
for k, v in SETUP["truths"].items():
    md.append(f"* {k}: f={v['f']:.0f}, pp=({v['cx']:.0f},{v['cy']:.0f}), k1={v['k1']}, k2={v['k2']} ({v['note']}); poses = pose-only fit to the real corners "
              f"(synthetic vs real corners {v['synthetic_vs_real_corner_rms_px']:.1f} px RMS)")
for k, v in rep_json["noise_models"].items():
    md.append(f"* scenario {k}: {v}")
def realism():
    rows = ["| quantity | real data | " + " | ".join(SCEN) + " |", "|---|---|" + "---|" * len(SCEN)]
    rl = SETUP["real"]
    q = [("sticker RMS, drawing fit k1k2 ppfree [px]", rl["markers_k1k2_ppfree"]["rms"], lambda s: A[s]["markers_k1k2_ppfree"].get("rms_median")),
         ("sticker RMS, drawing fit k1 ppfix [px]", rl["markers_k1_ppfix"]["rms"], lambda s: A[s]["markers_k1_ppfix"].get("rms_median")),
         ("plumb-line (centre fixed) edge RMS [px]", rl["plumb_fixed"]["rms"], lambda s: A[s]["plumb_fixed"].get("rms_median") if A[s].get("plumb_fixed") else None),
         ("lines joint edge RMS [px]", rl["lines_joint"]["rms"], lambda s: A[s]["lines_joint"].get("rms_median") if A[s].get("lines_joint") else None)]
    for nm, rv, fn in q:
        vals = []
        for s in SCEN:
            try:
                v = fn(s)
            except Exception:
                v = None
            vals.append(f"{v:.3f}" if v is not None else "-")
        rows.append(f"| {nm} | {rv:.3f} | " + " | ".join(vals) + " |")
    for b in "MLVS":
        vals = [f"{A[s]['combined_k1k2']['block_rms_median'][b]:.3f}" if A[s]["combined_k1k2"]["block_rms_median"].get(b) is not None else "-" for s in SCEN]
        rows.append(f"| combined block {b} RMS (per coordinate / per point) [px] | see method_combined | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def failtab():
    rows = []
    for s in SCEN:
        for e in EST:
            a = A[s].get(e)
            if a and a.get("n_failed_from_perturbed_start"):
                rows.append(f"* {s} / {SHORT[e]}: {a['n_failed_from_perturbed_start']} of {a['n']} fits from the perturbed start failed "
                            f"(edge RMS / result: {'; '.join(a['failed_examples'])}); re-run from the orchestrator's default start, statistics use the re-run.")
    return "\n".join(rows) if rows else "* none"


md += ["", "## Realism check (synthetic residual levels vs the real data; medians over replicates)", "", realism(), "",
       "## Convergence failures from perturbed starts", "", failtab()]
md += ["", "## Honesty summary (f, or k1 for the plumb-line; mapping = rotation-compensated error vs truth, median over the region)", "", htab()]
for s in SCEN:
    md += ["", f"## Round trip {s} ({A[s][EST[0]]['n']} replicates)", "", "Parameters (plumb-line k's are k/f0^2, k/f0^4 at f0 = 1300 in both estimate and truth):", "", ptab(s), "",
           "Mapping error vs truth (rotation compensated; median over the region of the per-point RMS over replicates):", "", mtab(s)]
if SENS:
    md += ["", "## Sensitivity (noise-free synthetic T2)", "", stab("T2"), "", "## Sensitivity (REAL data)", "", stab("real"), "",
           "Single frames instead of the 7-frame mean: RMS over the 7 frames of the parameter / mapping change: see synthetic_report.json sensitivity.frames_rms.",
           "Subsampling: see sensitivity.subsample."]
open(f"{CACHE}/synthetic_report.md", "w").write("\n".join(md) + "\n")
print("written")
