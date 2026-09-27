"""Cuboid cart model, part 3 (DIAGNOSTIC ONLY - never a main estimate): is the mismatch between the drawing
and the image consistent with ONE dimension being off?

Joint fit of sticker corners + sticker sides + exact top-plate edges + the ten shelf-front lips (one outline
per shelf and cart; every lip at its drawing shelf height, with ONE common lip offset (dY, dZ) per cart as a
nuisance - the lip profile is not in the drawing) with exactly one drawing dimension freed at a time:
  dz_top        top-plate height relative to all shelves (top stickers + top edges shifted in Z)
  dz_A..dz_E    one shelf height
  sz_shelves    common scale of the shelf depths below the top plate
  sx / sy       cart width / depth (relative scale, 1/1000)
  code_mm       sticker code size (not a cart dimension; for comparison)
  dz_top@cart   top-plate height per cart (2 parameters)
The same with the lips left out (stickers + exact edges only) shows what the lips add.  Validation: edges
NOT used in any fit (back rail / member, posts, non-exact board edges) - implied shift of their drawing
line under each hypothesis camera.
Camera model: the main model of 41_cuboid_fit.py (method_cuboid.json) and k1k2 with a free pp.
Outputs: work/cache/cuboid_diag.json, results/cuboid_diag.png
"""
import importlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS, H, W, K_from, load_json, save_json  # noqa: E402

L = importlib.import_module("41_cuboid_lib")
NPROC = 3
RAW = L.load_cart_edges_raw()
CL = L.cart_lines()
M, PTS, SIDES, EXACT, P0 = L.base_data()
LIPS = {310: ["cart310_x_A_front_fall", "cart310_x_B_front_fall", "cart310_x_C_front_fall", "cart310_x_D_front_fall", "cart310_x_E_front_fall"],
        80: ["cart80_x_A_front_out", "cart80_x_B_front", "cart80_x_C_front", "cart80_x_D_front", "cart80_x_E_front"]}
NUIS = ["lipZ@310", "lipZ@80"]  # one common lip height per cart (lip outline at Y = 0); dY is degenerate with dZ
VALID = ["cart80_x_back_rail_out", "cart80_x_back_rail_in", "cart80_x_back_low", "cart310_x_back_inner", "cart310_z_post_front_left_right",
         "cart310_z_post_front_right", "cart80_z_post1600_sil", "cart80_z_post1600_sil_low", "cart310_board1600_outer", "cart80_y_board0_outer",
         "cart80_y_board1600_outer", "cart310_x_board1600_front", "cart310_x_board0_front"]
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def lip_lines():
    out = []
    for c, ids in LIPS.items():
        for eid in ids:
            nm = L.EDGE_MAP[eid][0]
            ln = CL[nm]
            out.append(dict(id=eid, cart=c, points=RAW[eid]["points"], A=ln["A"].copy(), D=ln["D"], level=nm[5], kind="lip",
                            nuis=[(f"lipZ@{c}", [0, 0, 1])]))
    return out


LIPL = lip_lines()
HYP = {
    "drawing": [],
    "dz_top": ["dz_top"],
    "dz_top@cart": ["dz_top@80", "dz_top@310"],
    "dz_A": ["dz_A"], "dz_B": ["dz_B"], "dz_C": ["dz_C"], "dz_D": ["dz_D"], "dz_E": ["dz_E"],
    "sz_shelves": ["sz_shelves"],
    "sx_width": ["sx"], "sy_depth": ["sy"],
    "code_mm": ["code_mm"],
    "dz_A..E (5)": ["dz_A", "dz_B", "dz_C", "dz_D", "dz_E"],
    "dz_top+dz_E (2)": ["dz_top", "dz_E"],
    "dz_top+code_mm (2)": ["dz_top", "code_mm"],
}


def fit_one(args):
    hname, spec, with_lips = args
    geo = HYP[hname]
    if not with_lips:
        geo = [g for g in geo if g not in ("dz_D", "dz_E")]
    lines = SIDES + EXACT + (LIPL if with_lips else [])
    nuis = NUIS if with_lips else []
    best = None
    for f0 in (1300.0, 1450.0):
        cf = L.CuboidFit(PTS, lines, spec, geo=geo, nuis=nuis, weight_mode="cluster")
        x0 = cf.x0(K_from(f0, f0, (W - 1) / 2, (H - 1) / 2), np.array([-0.25, 0.05]), P0)
        try:
            r = cf.solve(x0)
        except Exception as ex:  # noqa
            print("fail", hname, ex)
            continue
        if best is None or r.cost < best[1].cost:
            best = (cf, r)
    cf, r = best
    K, d, poses, g, nu = cf.unpack(r.x)
    C = L.cov_from(r)
    sd = dict(zip(cf.names, np.sqrt(np.diag(C))))
    br = cf.block_rms(r.x)
    pl = cf.per_line_rms(r.x)
    # effective-observation AIC with fixed block sigmas of the drawing fit is done by the caller
    cam = cf.camera(r.x)
    val = {}
    for eid in VALID:
        nm, axes, lab = L.EDGE_MAP[eid]
        ln = CL[nm]
        e = RAW[eid]
        r0, _ = L.line_offsets(cam, e["cart"], ln["A"], ln["D"], e["points"])
        sg = L.physical_sign(cam, e["cart"], ln["A"], ln["D"], e["points"], AX[axes[0]])
        s1 = L.implied_shift(cam, e["cart"], ln["A"], ln["D"], e["points"], AX[axes[0]])
        s2 = L.implied_shift(cam, e["cart"], ln["A"], ln["D"], e["points"], AX[axes[1]])
        val[eid] = dict(mean_px=float(np.mean(r0 * sg)), **{f"d{axes[0]}_mm": s1[0], f"d{axes[1]}_mm": s2[0]})
    # chi2 with the per-block sigmas (cluster weights) for model comparison
    return dict(hyp=hname, spec=spec, with_lips=with_lips, intr=dict(zip(cf.inames, r.x[: cf.ni].tolist())),
                intr_sd={n: float(sd[n]) for n in cf.inames}, geo={k: float(v) for k, v in g.items()},
                geo_sd={k: float(sd[k]) for k in g}, nuis={k: float(v) for k, v in nu.items()}, nuis_sd={k: float(sd[k]) for k in nu},
                blocks=br, sig=dict(cf.sig), per_line=pl, validation=val, k=len(r.x), cost=float(r.cost),
                x=r.x.tolist(), names=cf.names)


def main():
    t0 = time.time()
    mc = load_json(f"{CACHE}/method_cuboid.json")
    main_key = mc["details"]["main_key"].split("|")[0]
    specs = {"k1_ppfix": dict(f="single", pp="fixed", dist=["k1"]), "k1k2_ppfix": dict(f="single", pp="fixed", dist=["k1", "k2"]),
             "k1_ppfree": dict(f="single", pp="free", dist=["k1"]), "k1k2_ppfree": dict(f="single", pp="free", dist=["k1", "k2"])}
    spec_main = specs.get(main_key, specs["k1k2_ppfix"])
    runs = []
    quick = os.environ.get("CUBOID_QUICK", "0") == "1"
    spec_list = (("main:" + main_key, spec_main),) if quick else (("main:" + main_key, spec_main), ("k1k2_ppfree", specs["k1k2_ppfree"]))
    for sname, spec in spec_list:
        for with_lips in (True, False):
            for h in (["drawing", "dz_top", "dz_E"] if quick else HYP):
                if not with_lips and h in ("dz_D", "dz_E", "dz_top+dz_E (2)"):
                    continue
                runs.append((h, spec, with_lips, sname))
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = pool.map(fit_one, [(h, s, wl) for h, s, wl, _ in runs], chunksize=1)
    out = {}
    for (h, s, wl, sname), rr in zip(runs, res):
        rr["spec_name"] = sname
        out[f"{sname}|{'lips' if wl else 'nolips'}|{h}"] = rr
    # n_eff per block (same for all hypotheses)
    cf0 = L.CuboidFit(PTS, SIDES + EXACT + LIPL, spec_main, nuis=NUIS)
    neff = {"S": sum(ln["neff"] for ln in cf0.lines if ln["blk"] == "S"), "E": sum(ln["neff"] for ln in cf0.lines if ln["blk"] == "E"),
            "X": sum(ln["neff"] for ln in cf0.lines if ln["blk"] == "X")}
    # comparable chi2 (fixed sigmas of the drawing fit of the same configuration) and AIC
    for key, rr in out.items():
        sname, lp, h = key.split("|")
        ref = out[f"{sname}|{lp}|drawing"]["sig"]
        b = rr["blocks"]
        chi = b["M"]["n"] * 2 * b["M"]["rms_coord"] ** 2 / ref["M"] ** 2
        n = b["M"]["n"] * 2
        for k in ("S", "E", "X"):
            if k in b:
                chi += neff[k] * b[k]["rms_w"] ** 2 / ref[k] ** 2
                n += neff[k]
        rr["chi2_ref"] = float(chi)
        rr["n_eff"] = float(n)
        rr["aic_ref"] = float(chi + 2 * len(rr["geo"]))
    # print table
    for key, rr in out.items():
        b = rr["blocks"]
        geo = " ".join(f"{k}={v:+.1f}+-{rr['geo_sd'][k]:.1f}" for k, v in rr["geo"].items())
        nus = " ".join(f"{k}={v:+.1f}" for k, v in rr["nuis"].items())
        intr = " ".join(f"{k}={v:.4g}" for k, v in rr["intr"].items())
        print(f"{key:40s} chi2ref {rr['chi2_ref']:7.1f} | M {b['M']['rms_point']:.2f} S {b['S']['rms']:.2f} E {b['E']['rms']:.2f}"
              + (f" lips {b['X']['rms']:.2f}" if 'X' in b else "") + f" | {intr} | {geo} | {nus}")
    save_json(dict(hypotheses=HYP, runs=out, n_eff=neff, validation_edges=VALID, lips=LIPS,
                   note="DIAGNOSTIC: one drawing dimension freed at a time; the main result uses the drawing unchanged."),
              f"{CACHE}/cuboid_diag.json")
    # markdown summary
    md = ["# Cuboid model - DIAGNOSTIC: one drawing dimension freed at a time (not a main estimate)", "",
          "Script `work/41_cuboid_diag.py`. Joint fit of sticker corners + sticker sides + exact top-plate edges (+ the ten shelf-front lips with "
          "one common lip height per cart as nuisance); cluster weighting; chi2_ref = chi2 with the block sigmas of the drawing fit of the same "
          "configuration (lower = better, the drawing fit has chi2_ref = n_eff). Values +- = Gauss-Newton 1-sigma (scaled). "
          "Validation = cart-80 back top rail (outer silhouette), NOT used in any fit: offset from the drawing line top Y=450, Z=0.", ""]
    for sname in sorted({k.split("|")[0] for k in out}):
        for lp in ("lips", "nolips"):
            md += [f"## {sname}, {'with' if lp == 'lips' else 'without'} shelf lips", "",
                   "| hypothesis | freed [mm, permille] | chi2_ref | corners px/pt | sides px | exact edges px | lips px | f | pp | k1, k2 | rail offset px (dZ mm) |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for h in HYP:
                rr = out.get(f"{sname}|{lp}|{h}")
                if rr is None:
                    continue
                b = rr["blocks"]
                geo = ", ".join(f"{k} {v:+.1f}+-{rr['geo_sd'][k]:.1f}" for k, v in rr["geo"].items()) or "-"
                it = rr["intr"]
                pp = f"({it['cx']:.0f}, {it['cy']:.0f})" if "cx" in it else "centre"
                rail = rr["validation"]["cart80_x_back_rail_out"]
                md.append(f"| {h} | {geo} | {rr['chi2_ref']:.1f} | {b['M']['rms_point']:.2f} | {b['S']['rms']:.2f} | {b['E']['rms']:.2f} | "
                          f"{b['X']['rms'] if 'X' in b else float('nan'):.2f} | {it['f']:.0f} | {pp} | {it['k1']:.3f}, {it['k2']:.3f} | "
                          f"{rail['mean_px']:+.1f} ({rail['dZ_mm']:+.0f}) |")
            md.append("")
    open(f"{CACHE}/cuboid_diag.md", "w").write("\n".join(md) + "\n")
    sec = ["Script `work/41_cuboid_diag.py` (full table `work/cache/cuboid_diag.md`, plot `results/cuboid_diag.png`). DIAGNOSTIC ONLY: one drawing "
           "dimension freed at a time; the main result above keeps the drawing.", ""]
    for key in (f"main:{main_key}|lips|drawing", f"main:{main_key}|lips|dz_top", f"main:{main_key}|lips|dz_top@cart", f"main:{main_key}|lips|dz_A",
                f"main:{main_key}|lips|dz_E", f"main:{main_key}|lips|sz_shelves", f"main:{main_key}|lips|sy_depth", f"main:{main_key}|lips|code_mm",
                f"main:{main_key}|lips|dz_top+dz_E (2)", "k1k2_ppfree|lips|dz_top", "k1k2_ppfree|nolips|dz_top"):
        rr = out.get(key)
        if rr is None:
            continue
        b = rr["blocks"]
        geo = ", ".join(f"{k} {v:+.1f} +- {rr['geo_sd'][k]:.1f}" for k, v in rr["geo"].items()) or "none"
        rail = rr["validation"]["cart80_x_back_rail_out"]
        it = rr["intr"]
        sec.append(f"* {key}: {geo}; chi2_ref {rr['chi2_ref']:.0f}; corners {b['M']['rms_point']:.2f} px/pt, exact edges {b['E']['rms']:.2f} px"
                   + (f", lips {b['X']['rms']:.2f} px" if 'X' in b else "") + f"; f {it['f']:.0f}, k1 {it['k1']:.3f}, k2 {it['k2']:.3f}"
                   + (f", pp ({it['cx']:.0f}, {it['cy']:.0f})" if 'cx' in it else "") + f"; back rail (not fitted) {rail['mean_px']:+.1f} px ({rail['dZ_mm']:+.0f} mm in Z)")
    L.md_replace_section(f"{CACHE}/method_cuboid.md", "## Diagnostic: which single dimension explains the mismatch", sec)
    # plot: chi2_ref per hypothesis (main spec, with / without lips) + freed value
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=110)
    sname = "main:" + main_key
    hyps = [h for h in HYP if f"{sname}|lips|{h}" in out]
    for j, lp in enumerate(("lips", "nolips")):
        vals = [out.get(f"{sname}|{lp}|{h}", {}).get("chi2_ref", np.nan) for h in hyps]
        ax[j].barh(np.arange(len(hyps)), vals, color="tab:blue" if lp == "lips" else "tab:gray")
        ax[j].set_yticks(np.arange(len(hyps)))
        ax[j].set_yticklabels(hyps, fontsize=8)
        ax[j].set_xlabel("chi2 with the block sigmas of the drawing fit (effective observations)")
        ax[j].set_title(f"{sname}, {'with' if lp == 'lips' else 'without'} shelf lips")
        for i, h in enumerate(hyps):
            rr = out.get(f"{sname}|{lp}|{h}")
            if rr and rr["geo"]:
                txt = ", ".join(f"{k}={v:+.0f}" for k, v in rr["geo"].items())
                ax[j].text(0, i, " " + txt, va="center", fontsize=7, color="w")
        ax[j].invert_yaxis()
    # block RMS of the lips vs hypothesis
    for lp, col in (("lips", "tab:blue"),):
        per = [[out[f"{sname}|{lp}|{h}"]["per_line"][eid]["rms"] for eid in sum(LIPS.values(), [])] for h in hyps]
        ax[2].imshow(np.array(per), aspect="auto", cmap="viridis", vmin=0, vmax=4)
        ax[2].set_yticks(np.arange(len(hyps)))
        ax[2].set_yticklabels(hyps, fontsize=8)
        ax[2].set_xticks(np.arange(10))
        ax[2].set_xticklabels([e.replace("cart", "").replace("_front", "").replace("_x", "") for e in sum(LIPS.values(), [])], rotation=60, fontsize=7)
        for i in range(len(hyps)):
            for k in range(10):
                ax[2].text(k, i, f"{per[i][k]:.1f}", ha="center", va="center", fontsize=6, color="w")
        ax[2].set_title("per-lip RMS [px] (common lip offset per cart)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/cuboid_diag.png")
    plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
