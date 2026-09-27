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
        v = cf.nll(r.x) + 1e6 * max(0.0, cf.FOLD_MIN - 0.005 - cf.fold_margin(r.x))  # review fix: rank starts by the profiled likelihood
        if best is None or v < best[2]:
            best = (cf, r, v)
    cf, r, _ = best
    K, d, poses, g, nu = cf.unpack(r.x)
    C = L.cov_from(r, cf)  # review fix: dof from the effective number of observations
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
                blocks=br, sig=dict(cf.sig), per_line=pl, validation=val, k=len(r.x), cost=float(r.cost), fold_margin=cf.fold_margin(r.x),
                x=r.x.tolist(), names=cf.names)


def main():
    t0 = time.time()
    mc = load_json(f"{CACHE}/method_cuboid.json")
    main_key = mc["details"]["main_key"].split("|")[0]
    specs = {"k1_ppfix": dict(f="single", pp="fixed", dist=["k1"]), "k1k2_ppfix": dict(f="single", pp="fixed", dist=["k1", "k2"]),
             "k1_ppfree": dict(f="single", pp="free", dist=["k1"]), "k1k2_ppfree": dict(f="single", pp="free", dist=["k1", "k2"]),
             "k1k2k3_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "k3"]), "k1k2_ppfree_fxfy": dict(f="fxfy", pp="free", dist=["k1", "k2"]),
             "k1k2p1p2_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "p1", "p2"])}
    assert main_key in specs, main_key  # review fix: never label a fallback spec as the main model
    spec_main = specs.get(main_key, specs["k1k2_ppfix"])
    runs = []
    quick = os.environ.get("CUBOID_QUICK", "0") == "1"
    spec_list = (("main:" + main_key, spec_main),) if quick else (("main:" + main_key, spec_main), ("k1k2_ppfree", specs["k1k2_ppfree"]))
    if not quick and main_key != "k1k2_ppfix":  # review: keep the k1k2 / pp-fixed lens as a comparison (the main k1 lens sits on the fold barrier)
        spec_list = spec_list + (("k1k2_ppfix", specs["k1k2_ppfix"]),)
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
                          f"{b['X']['rms'] if 'X' in b else float('nan'):.2f} | {it.get('f', it.get('fx', float('nan'))):.0f} | {pp} | {it['k1']:.3f}, {it.get('k2', 0.0):.3f} | "
                          f"{rail['mean_px']:+.1f} ({rail['dZ_mm']:+.0f}) |")
            md.append("")
    open(f"{CACHE}/cuboid_diag.md", "w").write("\n".join(md) + "\n")
    sec = ["Script `work/41_cuboid_diag.py` (full table `work/cache/cuboid_diag.md`, plot `results/cuboid_diag.png`). DIAGNOSTIC ONLY: one drawing "
           "dimension freed at a time; the main result above keeps the drawing.", ""]
    for key in (f"main:{main_key}|lips|drawing", f"main:{main_key}|lips|dz_top", f"main:{main_key}|lips|dz_top@cart", f"main:{main_key}|lips|dz_A",
                f"main:{main_key}|lips|dz_E", f"main:{main_key}|lips|sz_shelves", f"main:{main_key}|lips|sy_depth", f"main:{main_key}|lips|code_mm",
                f"main:{main_key}|lips|dz_top+dz_E (2)", f"main:{main_key}|lips|dz_A..E (5)", "k1k2_ppfix|lips|drawing", "k1k2_ppfix|lips|dz_top",
                "k1k2_ppfix|lips|dz_A..E (5)", "k1k2_ppfree|lips|drawing", "k1k2_ppfree|lips|dz_top", "k1k2_ppfree|lips|dz_A..E (5)", "k1k2_ppfree|nolips|dz_top"):
        rr = out.get(key)
        if rr is None:
            continue
        b = rr["blocks"]
        geo = ", ".join(f"{k} {v:+.1f} +- {rr['geo_sd'][k]:.1f}" for k, v in rr["geo"].items()) or "none"
        rail = rr["validation"]["cart80_x_back_rail_out"]
        it = rr["intr"]
        sec.append(f"* {key}: {geo}; chi2_ref {rr['chi2_ref']:.0f}; corners {b['M']['rms_point']:.2f} px/pt, exact edges {b['E']['rms']:.2f} px"
                   + (f", lips {b['X']['rms']:.2f} px" if 'X' in b else "") + f"; f {it.get('f', it.get('fx', float('nan'))):.0f}, k1 {it['k1']:.3f}, k2 {it.get('k2', 0.0):.3f}"
                   + (f", pp ({it['cx']:.0f}, {it['cy']:.0f})" if 'cx' in it else "") + f"; back rail (not fitted) {rail['mean_px']:+.1f} px ({rail['dZ_mm']:+.0f} mm in Z)")
    L.md_replace_section(f"{CACHE}/method_cuboid.md", "## Diagnostic: which single dimension explains the mismatch", sec)
    # ------------------------------------------------------------------ conclusions (numbers from the three scripts)
    mc = load_json(f"{CACHE}/method_cuboid.json")
    off = load_json(f"{CACHE}/cuboid_offsets.json")
    u = mc["uncertainty"]
    d0 = mc["dist_coeffs"]
    br = mc["details"]["block_rms"]
    fl = list(mc["details"]["f_over_models_and_weightings"].values())
    E = off["edges"]
    lips = sum(LIPS.values(), [])
    lip_j = [E["joint"][e]["mean_px"] for e in lips]
    lip_s = [E["joint"][e].get("camera_sigma_mean_px_robust", np.nan) for e in lips]
    sensZ = [E["joint"][e]["px_per_mm_Z"] for e in lips]
    sensY = [E["joint"][e]["px_per_mm_Y"] for e in lips]
    rail = {k: E[k]["cart80_x_back_rail_out"] for k in ("markers_only", "joint", "edge_lens_shelf_poses")}
    dt = out[f"main:{main_key}|lips|dz_top"]
    dtc = out[f"main:{main_key}|lips|dz_top@cart"]
    dtf = out["k1k2_ppfree|lips|dz_top"]
    dtk = out.get("k1k2_ppfix|lips|dz_top", out.get("main:k1k2_ppfix|lips|dz_top"))
    dzr = [rr["geo"]["dz_top"] for k, rr in out.items() if k.endswith("|lips|dz_top")]
    rail_by = {k.split("|")[0] + "/" + k.split("|")[2]: rr["validation"]["cart80_x_back_rail_out"] for k, rr in out.items()
               if k.split("|")[1] == "lips" and k.split("|")[2] in ("drawing", "dz_top", "dz_A..E (5)")}
    dte = out[f"main:{main_key}|lips|dz_top+dz_E (2)"]
    dr = out[f"main:{main_key}|lips|drawing"]
    others = {h: out[f"main:{main_key}|lips|{h}"]["chi2_ref"] for h in ("dz_A", "dz_B", "dz_C", "dz_D", "dz_E", "sz_shelves", "sx_width", "sy_depth", "code_mm")}
    sr = off["sticker_rows"]["edge_lens_shelf_poses"]
    lip_rms = [off["lip_pattern"][cn][str(c)]["common_dZ"]["rms_px"] if str(c) in off["lip_pattern"][cn] else off["lip_pattern"][cn][c]["common_dZ"]["rms_px"]
               for cn in ("markers_only", "joint", "edge_lens_shelf_poses") for c in (80, 310)]
    bd_off = [abs(E["edge_lens_shelf_poses"][e]["mean_px"]) for e in E["edge_lens_shelf_poses"] if "board" in e]
    cxs = [v["x"][v["names"].index("cx")] for k, v in mc["details"]["grid"].items() if "|" in k and "cx" in v["names"]]
    fm_main = mc["details"].get("fold_margin", float("nan"))
    nob = mc["details"].get("without_fold_barrier", {}).get("grid", {}).get(main_key)
    fold_txt = (f" The fit uses a fold barrier (fold margin >= {mc['details'].get('fold_barrier_min', 0.03)}); the main solution has fold margin {fm_main:.3f}"
                + (" i.e. it SITS ON THE BARRIER: the drawing geometry pushes the distortion towards a lens that folds inside the image, so k1, k2 are set by the "
                   "validity constraint and not by the data" if fm_main < mc["details"].get("fold_barrier_min", 0.03) + 0.005 else "")
                + (f" (without the barrier: f {nob['x'][0]:.0f}, k1 {nob['x'][nob['names'].index('k1')]:.3f}"
                   + (f", k2 {nob['x'][nob['names'].index('k2')]:.3f}" if 'k2' in nob['names'] else "") + f", fold margin {nob['fold_margin']:.3f} = folded)" if nob else "") + ".")
    conc = [
        f"1. Main cuboid fit (drawing exact, {mc['model']}): f = {mc['camera_matrix'][0][0]:.0f} +- {u['f_1sigma']:.0f} px (cluster bootstrap; over 14 model x weighting "
        f"variants {min(fl):.0f}..{max(fl):.0f}), k1 = {d0[0]:.3f} +- {u['k1_1sigma']:.3f}" + (f", k2 = {d0[1]:.3f} +- {u['k2_1sigma']:.3f}" if "k2_1sigma" in u else "") + f"; corners {br['M']['rms_point']:.2f} px per point "
        f"(max {br['M']['max_point']:.1f}), sticker sides {br['S']['rms']:.2f} px, exact edges {br['E']['rms']:.2f} px (corner RMS ~{br['M']['rms_point'] / 0.13:.0f}x the per-frame corner scatter). The exact edges add almost nothing "
        f"(3 short edges); free-pp variants are unstable (cx {min(cxs):.0f}..{max(cxs):.0f}) because pp is the knob that trades the exact edges against the stickers." + fold_txt + " The "
        "unconstrained distortion contradicts the straight-edge (plumb-line) distortion, as for the marker-only fit: the drawing geometry biases the estimate - only CONSISTENT with the data at the 5-8 px level.",
        "2. The three exact edges are the edges they claim to be (crops: top-face outer edge of the X=0 board of cart 310 against the floor; top-face front edges "
        "of both cart-80 boards, a 1-2 px dark front face beside them) and agree with their own board's stickers to <= 4.6 mm, but every global drawing camera misses them by "
        + ", ".join(f"{k.split('_', 1)[1]} {v['joint']['mean_offset_px']:+.1f} px ({v['joint']['implied_shift_mm']:+.0f} mm in {v['joint']['axis']})" for k, v in mc["details"]["exact_edge_check"].items())
        + " - they inherit the sticker inconsistency.",
        f"3. Shelf lips vs prediction (joint camera): {min(lip_j):+.1f}..{max(lip_j):+.1f} px (camera 1-sigma {np.nanmin(lip_s):.1f}..{np.nanmax(lip_s):.1f} px); the lips are "
        f"sensitive mainly to Y ({min(sensY):.2f}..{max(sensY):.2f} px/mm) and only weakly to Z ({min(sensZ):.2f}..{max(sensZ):.2f} px/mm), so a single lip fixes its height only to "
        f"~+-{2.5 / max(sensZ):.0f}..{2.5 / min(sensZ):.0f} mm (for 2.5 px). With one common lip height per cart the five lips agree only to {min(lip_rms):.1f}..{max(lip_rms):.1f} px rms (markers-only, joint and shelf-anchored cameras): lip outlines / fronts differ by a few mm between shelves.",
        f"4. The clearest misfit is at the TOP level: the cart-80 back top rail lies {rail['markers_only']['mean_px']:+.1f} px (markers-only) / {rail['joint']['mean_px']:+.1f} px (joint) "
        f"from the drawing top-back edge (= {rail['joint']['implied_dZ_mm']:+.0f} mm in Z), but only {rail['edge_lens_shelf_poses']['mean_px']:+.1f} px when the camera is anchored to the shelf "
        f"stickers (edge-only lens) - which in turn puts the top stickers {sr['80:top']['implied_dZ_mm']:+.0f} / {sr['310:top']['implied_dZ_mm']:+.0f} mm (cart 80 / 310) above the drawing "
        f"and the top-board edges {min(bd_off):.0f}..{max(bd_off):.0f} px off.",
        f"5. Single dimension: the top-plate surface (end boards with the top stickers) {dt['geo']['dz_top']:+.0f} +- {dt['geo_sd']['dz_top']:.0f} mm above the drawing relative to shelves and "
        f"frame (per cart {dtc['geo']['dz_top@80']:+.0f} / {dtc['geo']['dz_top@310']:+.0f} mm; pp free {dtf['geo']['dz_top']:+.0f} mm" + (f"; k1k2 pp fixed {dtk['geo']['dz_top']:+.0f} mm" if dtk else "") + f"; range over the lens models {min(dzr):+.0f}..{max(dzr):+.0f} mm - the k1-only lens sits on the fold barrier and gives the low end; formal +- = Gauss-Newton with n_eff dof) explains most of it: chi2_ref {dr['chi2_ref']:.0f} -> "
        f"{dt['chi2_ref']:.0f}, corners {dr['blocks']['M']['rms_point']:.1f} -> {dt['blocks']['M']['rms_point']:.1f} px, exact edges {dr['blocks']['E']['rms']:.1f} -> {dt['blocks']['E']['rms']:.1f} px, "
        f"and the NOT fitted back rail moves to {dt['validation']['cart80_x_back_rail_out']['mean_px']:+.1f} px ({dt['validation']['cart80_x_back_rail_out']['dZ_mm']:+.0f} mm). "
        "No other single dimension comes close (chi2_ref " + ", ".join(f"{k} {v:.0f}" for k, v in others.items()) + "). Stickers alone cannot tell this from 'all shelves ~65 mm lower'; the back top rail can, IF it really is at the top-plate level (its height is not in the drawing): with all shelf rows lowered instead (dz_A..E) the rail stays off (" + ", ".join(f"{k}: {v['mean_px']:+.1f} px / {v['dZ_mm']:+.0f} mm" for k, v in rail_by.items()) + "), so it groups with the shelves/frame and puts the offset on the end boards with the top stickers. Caveat: the rail edge was first read as a wall conduit by the scene tracing and re-assigned to cart 80 in the review. Observation only - the main result keeps the drawing.",
        f"6. Second order: shelf E's lip lies below its drawing height relative to A-D: dz_E {dte['geo']['dz_E']:+.0f} +- {dte['geo_sd']['dz_E']:.0f} mm on top of dz_top (lips "
        f"{dt['blocks']['X']['rms']:.2f} -> {dte['blocks']['X']['rms']:.2f} px), in both carts; E has no sticker, so this rests on the identification of the lowest traced lip.",
        f"7. With dz_top freed (diagnostic) the lens moves to f = {dt['intr']['f']:.0f} (pp fixed) / {dtf['intr']['f']:.0f} with pp ({dtf['intr']['cx']:.0f}, {dtf['intr']['cy']:.0f}), "
        f"k1 {dtf['intr']['k1']:.2f}, k2 {dtf['intr']['k2']:.2f} (fold margin {dtf.get('fold_margin', float('nan')):.2f}; main-lens dz_top fit: fold margin {dt.get('fold_margin', float('nan')):.3f}" + (" = on the barrier" if dt.get("fold_margin", 1) < 0.035 else "") + f") - the drawing-geometry f ({mc['camera_matrix'][0][0]:.0f}) is pulled down by "
        f"~{dt['intr']['f'] - mc['camera_matrix'][0][0]:.0f} px (pp fixed) by the top-plate mismatch; this bias is NOT contained in the main-result uncertainty.",
    ]
    L.md_replace_section(f"{CACHE}/method_cuboid.md", "## Conclusions (determined / consistent / not determinable)", conc)
    # review addition: the geometry systematic of f (diagnostic dz_top fit, same lens model) as an explicit uncertainty term
    shift = float(dt["intr"]["f"] - mc["camera_matrix"][0][0])
    mc["uncertainty"]["f_geometry_shift_px_diag"] = shift
    mc["uncertainty"]["f_1sigma_incl_geometry"] = float(np.hypot(mc["uncertainty"]["f_1sigma"], shift / 2.0))
    mc["uncertainty"]["geometry_note"] = ("f_geometry_shift_px_diag = f(top-plate height freed, diagnostic) - f(drawing), same lens model; "
                                          "f_1sigma_incl_geometry = sqrt(bootstrap^2 + (shift/2)^2) - half the shift as a 1-sigma systematic of the "
                                          "drawing-geometry estimate (the drawing is used unchanged for the value itself)")
    save_json(mc, f"{CACHE}/method_cuboid.json")
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
