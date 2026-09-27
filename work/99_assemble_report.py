"""Assemble results/REPORT.md from the text parts (REPORT_head.md, REPORT_part_*.md) and the result files.

Every number that depends on the final computation is a placeholder {{name}} in the text and is filled here from
results/lens_result.json, work/cache/combined_fit.json, combined_full.json, combined_cv_*.json, synthetic_main.json,
sensitivity_main.json, final_budget.json; tables TABLE_* are generated here (TABLE_METHODS by 97_report_tables.py).
Run after 95_final.py, 98_finalize_json.py, 90_compare_methods.py and 97_report_tables.py."""
import glob
import os
import re

import numpy as np

from common import CACHE, RESULTS, K_from, load_json
from evaltools import grid, mapping_displacement, region_defs

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def num(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−").replace(".", ",")


def sci(x, nd=2):
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return num(m, nd) + "·10" + str(e).translate(SUP)


L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"])
d = np.array(L["dist_coeffs"])
f, cx, cy, k1, k2 = K[0, 0], K[0, 2], K[1, 2], d[0], d[1]
U = L["uncertainty"]
UD = L["uncertainty_details"]
MU = L["mapping_uncertainty_px"]
MD = L["mapping_uncertainty_details"]
RD = L["rms_details"]
FIT = load_json(f"{CACHE}/combined_fit.json")
S = FIT["summary"]
FULL = load_json(f"{CACHE}/combined_full.json")
BUD = load_json(f"{CACHE}/final_budget.json")
SYN = load_json(f"{CACHE}/synthetic_main.json")["summary"]
SENS = load_json(f"{CACHE}/sensitivity_main.json")["results"] if os.path.exists(f"{CACHE}/sensitivity_main.json") else {}
V = {}

# ---------------- main result
V.update(f1=num(f, 1), f0=num(f, 0), f2=f"{f:.2f}", cx1=num(cx, 1), cx0=num(cx, 0), cx2=f"{cx:.2f}", cy1=num(cy, 1), cy0=num(cy, 0),
         cy2=f"{cy:.2f}", k1_3=num(k1, 3, True), k2_3=num(k2, 3, True), k1_4=f"{k1:.4f}", k2_4=f"{k2:.4f}",
         sf0=num(U["fx_px_1sigma"], 0), sfpct=num(100 * U["fx_px_1sigma"] / f, 1), scx0=num(U["cx_px_1sigma"], 0),
         scy0=num(U["cy_px_1sigma"], 0), sk1_3=num(U["k1_1sigma"], 3), sk2_3=num(U["k2_1sigma"], 3),
         rho_k1k2=num(UD["correlation_bootstrap"][3][4], 2), k1f2=sci(k1 / f ** 2), k2f4=sci(k2 / f ** 4),
         mc=num(MU["centre"]), mb=num(MU["cart_band"]), mk=num(MU["corners"]),
         mc_max=num(MD["max_in_region"]["centre"]), mb_max=num(MD["max_in_region"]["cart_band"]), mk_max=num(MD["max_in_region"]["corners"]),
         raw_c=num(MD["without_rotation_compensation"]["centre"], 0), raw_b=num(MD["without_rotation_compensation"]["cart_band"], 0),
         raw_k=num(MD["without_rotation_compensation"]["corners"], 0))
# 1 % in f with the distortion fixed in pixel units
uv, _ = grid(30)
regs = region_defs()
f2 = f * 1.01
D1 = mapping_displacement(K, d, K_from(f2, f2, cx, cy), np.array([k1 * 1.01 ** 2, k2 * 1.01 ** 4, 0, 0, 0]), uv)
r1 = np.hypot(D1[:, 0], D1[:, 1])
V.update(f1pct_band=num(np.median(r1[regs["cart_band"](uv)]), 1), f1pct_band_max=num(np.max(r1[regs["cart_band"](uv)]), 1),
         f1pct_corners=num(np.median(r1[regs["corners"](uv)]), 1))

# ---------------- blocks, variants, influence
sig = FIT["sig_main"]
V.update(sigM=num(sig["M"], 1), sigL=num(sig["L"], 2), sigV=num(sig["V"], 2), sigS=num(sig["S"], 2))
bl = RD["in_joint_fit_block_rms_px"]
V.update(bS=num(bl["edge_straightness"], 2), bV=num(bl["vanishing_point_groups"], 2), bL=num(bl["sticker_sides"], 1),
         bM_pt=num(bl["sticker_corners_per_coordinate"] * np.sqrt(2), 1), rms=num(RD["sticker_corners_px"], 1),
         rms_max=num(RD["sticker_corners_max_px"], 1))


def px(name):
    return dict(zip(S[name]["names"], S[name]["x"]))


for tag, key in (("inf_nocorners", "k1k2_influence_no_corners"), ("inf_nosides", "k1k2_influence_no_sides"), ("inf_novp", "k1k2_influence_no_vp"),
                 ("v_rowblocks", "k1k2_rowblocks"), ("v_commonv", "k1k2_common_vertical"), ("v_floor", "k1k2_floor_straight_only"),
                 ("v_nosides", "k1k2_no_80_3_7_sides"), ("v_nobent", "k1k2_no_bent_lip"), ("v_robust", "k1k2_robust"), ("inf_edges", "k1k2_edges_only")):
    p = px(key)
    V[f"{tag}_f"], V[f"{tag}_cx"], V[f"{tag}_cy"] = num(p["f"], 0), num(p["cx"], 0), num(p["cy"], 0)
    V[f"{tag}_k1"] = num(p.get("k1", np.nan), 3, True)
    V[f"{tag}_df"] = num(p["f"] - f, 0, True)
V["v_nosides_df"] = num(f - px("k1k2_no_80_3_7_sides")["f"], 0)
V["v_floor_dcy"] = num(abs(px("k1k2_floor_straight_only")["cy"] - cy), 0)
rb = S["k1k2_rowblocks"].get("rowblock_rms") or {}
V.update(rb_top=num(rb.get("top"), 1), rb_shelf=num(rb.get("shelves"), 1))
V["v_k3"] = num(px("k1k2k3")["k3"], 3, True)
pf = px("k1k2_fxfy")
V["v_fxfy"] = num(100 * (pf["fx"] / pf["fy"] - 1), 1, True) + " %"
V["n_methods"] = str(len(load_json(f"{RESULTS}/lens_by_method.json")))
boot = np.array(FULL["boot"])
V["boot_std_f"] = num(boot[:, 0].std(ddof=1), 0)
V["n_boot"] = str(len(boot))
V["n_alts"] = str(len(BUD["alternatives"]))
# angle between the Z axes of the two carts in the main fit
from common import rodrigues  # noqa: E402

xm = np.array(FIT["x_main"])
ni = len(FIT["inames"])
R80, R310 = rodrigues(xm[ni:ni + 3]), rodrigues(xm[ni + 6:ni + 9])
V["zz_angle"] = num(np.degrees(np.arccos(np.clip(R80[:, 2] @ R310[:, 2], -1, 1))), 2)
# sticker-only k1k2 pp-free lens: straightness of the edges
from linedata import load_edges  # noqa: E402
from lineselfcal import LineCal  # noqa: E402

mb = load_json(f"{CACHE}/markers_ba_full.json")["summary"]["k1k2_ppfree"]
Km, dm = np.array(mb["K"]), np.array(mb["dist"])
lc = LineCal(load_edges(verified_only=True), "plumb", dist_free=(), f0=Km[0, 0], centre_fixed=[Km[0, 2], Km[1, 2]], dist_fixed=dm[:5], subsample=1)
V["mk_ppfree_straight"] = num(float(np.sqrt(np.mean(lc.residuals(np.zeros(0))[:-1] ** 2))), 1)

# ---------------- f profile
prof = np.array(FIT["profile"])
i0 = int(np.argmin(prof[:, 1]))
lo, hi = max(i0 - 1, 0), min(i0 + 1, len(prof) - 1)
ch = prof[:, 1] - prof[i0, 1]
V.update(prof_min=num(prof[i0, 0], 0) + " px (mriežka po " + num(prof[1, 0] - prof[0, 0], 0) + " px)", prof_lo=num(prof[lo, 0], 0), prof_hi=num(prof[hi, 0], 0),
         prof_dchi_lo=num(ch[lo], 0), prof_dchi_hi=num(ch[hi], 0), sig_cov_f=num(S["k1k2"]["sigma_cov"][0], 1))
for j, b in enumerate("MLVS"):
    nd = 2 if b in "LVS" else 1
    if b == "S":
        nd = 4
    V[f"prof_{b}_lo"], V[f"prof_{b}_hi"], V[f"prof_{b}_0"] = num(prof[lo, 2 + j], nd), num(prof[hi, 2 + j], nd), num(prof[i0, 2 + j], nd)

# ---------------- tables: residuals
T = {}
t = ["| skupina | RMS [px] | max [px] | počet rohov |", "|---|---|---|---|", f"| všetky | {num(RD['sticker_corners_px'])} | {num(RD['sticker_corners_max_px'])} | 62 |"]
for c, v in RD["per_cart"].items():
    t.append(f"| vozík {c} | {num(v['rms_px'])} | {num(v['max_px'])} | {v['n_corners']} |")
for r, v in RD["per_row"].items():
    t.append(f"| rad {r} | {num(v['rms_px'])} | {num(v['max_px'])} | {v['n_corners']} |")
ph = RD["per_photo"]
t.append(f"| každá zo 7 fotiek zvlášť (póza pre fotku) | {num(min(v['rms_px'] for v in ph.values()), 2)}–{num(max(v['rms_px'] for v in ph.values()), 2)} | "
         f"{num(min(v['max_px'] for v in ph.values()))}–{num(max(v['max_px'] for v in ph.values()))} | "
         f"{min(v['n_corners'] for v in ph.values())}–{max(v['n_corners'] for v in ph.values())} |")
T["TABLE_RESID"] = "Reziduá rohov nálepiek s hlavným objektívom (geometria z výkresu, póza vozíka metódou najmenších štvorcov):\n\n" + "\n".join(t)
DET = {(m["cart"], m["id"]): m for m in load_json(f"{RESULTS}/detections.json")["markers"]}
t = ["| vozík:id | poloha | RMS [px] | max [px] | rohov |", "|---|---|---|---|---|"]
for k, v in RD["per_sticker"].items():
    c, i = map(int, k.split(":"))
    t.append(f"| {k} | {DET[(c, i)]['role']} | {num(v['rms_px'])} | {num(v['max_px'])} | {v['n_corners']} |")
T["TABLE_STICKERS"] = "Po nálepkách (rovnaká definícia):\n\n" + "\n".join(t)

# ---------------- tables: cross-validation
CV = {os.path.basename(p)[12:-5]: load_json(p) for p in glob.glob(f"{CACHE}/combined_cv_*.json")}
lab = {"row_top": "rad top", "row_A": "rad A", "row_B": "rad B", "row_C": "rad C", "cart_80": "vozík 80", "cart_310": "vozík 310",
       "edges_cart310": "hrany vozíka 310", "edges_cart80": "hrany vozíka 80", "edges_scene": "hrany scény"}
t = ["| vynechané | f | cx | cy | k1 | k2 | predikcia vynechaných [px] | tie isté body v hlavnom fite [px] | trénovacie rohy v spoločnom fite [px/bod] |",
     "|---|---|---|---|---|---|---|---|---|"]
fo = []
for k, nm in lab.items():
    j = CV[k]
    i = j["intrinsics"]
    fo.append([i["f"], i["cx"], i["cy"], i["k1"], i["k2"]])
    if "heldout_edges_straightness_rms_px" in j:
        pr = f"priamosť {num(j['heldout_edges_straightness_rms_px'], 2)}"
        inf = f"{num(bl['edge_straightness'], 2)} (všetky hrany)"
    else:
        pr = num(j.get("heldout_stickers_pred_rms_px", j.get("heldout_stickers_pose_refit_rms_px")))
        if "heldout_stickers_pose_refit_rms_px" in j:
            pr += " (póza refit.)"
            inf = num(RD["per_cart"][k[5:]]["rms_px"]) + " (póza z nálepiek)"
        else:
            inf = num(j["heldout_points_rms_in_main_fit_px"])
    tr = j["train_block_rms"].get("M")
    t.append(f"| {nm} | {num(i['f'], 0)} | {num(i['cx'], 0)} | {num(i['cy'], 0)} | {num(i['k1'], 3, True)} | {num(i['k2'], 3, True)} | {pr} | {inf} | "
             f"{num(tr * np.sqrt(2), 1) if tr else '–'} |")
JK = UD.get("jackknife_cv", {})
for pn, lab_ in (("stickers", "nálepky (20)"), ("rows", "rady (4)"), ("carts", "vozíky (2)"), ("edge_regions", "oblasti hrán (3)")):
    if pn in JK:
        q = JK[pn]["params"]
        t.append(f"| **jackknife σ: {lab_}** | {num(q['f'], 0)} | {num(q['cx'], 0)} | {num(q['cy'], 0)} | {num(q['k1'], 3)} | {num(q['k2'], 3)} | | | |")
T["TABLE_CV"] = "\n".join(t)
V["cv_dcx_310"] = num(CV["edges_cart310"]["intrinsics"]["cx"] - cx, 0, True)
V["cv_dcx_80"] = num(CV["edges_cart80"]["intrinsics"]["cx"] - cx, 0, True)
V["jk_st_f"] = num(JK["stickers"]["params"]["f"], 0) if "stickers" in JK else "–"
V["cv_top_train"] = num(CV["row_top"]["train_block_rms"]["M"] * np.sqrt(2), 1)
V["cv_top_held"] = num(CV["row_top"]["heldout_stickers_pred_rms_px"], 1)
st = {k: v for k, v in CV.items() if k.startswith("sticker_")}
fs = np.array([v["intrinsics"]["f"] for v in st.values()])
cys = np.array([v["intrinsics"]["cy"] for v in st.values()])
V.update(loo_f_min=num(fs.min(), 0), loo_f_max=num(fs.max(), 0), loo_f_std=num(fs.std(ddof=1), 0), loo_cy_min=num(cys.min(), 0), loo_cy_max=num(cys.max(), 0))
mx = {k: 0.0 for k in ("centre", "cart_band", "corners")}
for v in st.values():
    i = v["intrinsics"]
    Dv = mapping_displacement(K, d, K_from(i["f"], i["f"], i["cx"], i["cy"]), np.array([i["k1"], i["k2"], 0, 0, 0]), uv)
    rv = np.hypot(Dv[:, 0], Dv[:, 1])
    for k in mx:
        mx[k] = max(mx[k], float(np.median(rv[regs[k](uv)])))
V.update(loo_map_c=num(mx["centre"]), loo_map_b=num(mx["cart_band"]), loo_map_k=num(mx["corners"]))
top_ids = {(m["cart"], m["id"]) for m in DET.values() if m["role"].startswith("TOP")}
held_top = [v["heldout_stickers_pred_rms_px"] for k, v in st.items() if (int(k.split("_")[1]), int(k.split("_")[2])) in top_ids]
held_sh = [v["heldout_stickers_pred_rms_px"] for k, v in st.items() if (int(k.split("_")[1]), int(k.split("_")[2])) not in top_ids]
V.update(loo_top_lo=num(min(held_top), 0), loo_top_hi=num(max(held_top), 0), loo_shelf_lo=num(min(held_sh), 1), loo_shelf_hi=num(max(held_sh), 1))
hh = np.array([v["heldout_stickers_pred_rms_px"] for v in st.values()])
ii = np.array([v["heldout_points_rms_in_main_fit_px"] for v in st.values()])
big = [k[8:].replace("_", ":") + f" ({num(v['intrinsics']['f'] - f, 1, True)} px)" for k, v in st.items() if abs(v["intrinsics"]["f"] - f) > 5]
V["loo_big"] = ", ".join(big) if big else "žiadnej"
V["loo_infl"] = num(100 * (np.sqrt(np.mean(hh ** 2)) / np.sqrt(np.mean(ii ** 2)) - 1), 0)

# ---------------- synthetic
nm5 = ["f", "cx", "cy", "k1", "k2"]
t = ["| scenár | pravda | replík | odchýlka f (± chyba priemeru) | rozptyl f | RMSE f | RMSE cx / cy | RMSE k1 / k2 | RMSE zobrazenia stred / pás / rohy [px] |",
     "|---|---|---|---|---|---|---|---|---|"]
for sc, label in (("T2_a", "a"), ("T2_b", "**b**"), ("T2_c", "c"), ("T1_b", "b")):
    s = SYN[sc]
    m = s["mapping_rmse"]
    t.append(f"| {label} | {sc[:2]} | {s['n']} | {num(s['bias'][0], 1, True)} ± {num(s['bias_se'][0], 1)} | {num(s['std'][0], 1)} | {num(s['rmse'][0], 1)} | "
             f"{num(s['rmse'][1], 1)} / {num(s['rmse'][2], 1)} | {num(s['rmse'][3], 3)} / {num(s['rmse'][4], 3)} | "
             f"{num(m['centre']['median'])} / {num(m['cart_band']['median'])} / {num(m['corners']['median'])} |")
T["TABLE_SYNTH"] = "\n".join(t)
sb, s1 = SYN["T2_b"], SYN["T1_b"]
V.update(syn_a_bias_f=num(SYN["T2_a"]["bias"][0], 1, True), syn_c_bias_f=num(SYN["T2_c"]["bias"][0], 1, True),
         syn_b_bias_f=num(sb["bias"][0], 1, True), syn_b_bias_se=num(sb["bias_se"][0], 1), syn_t1_bias_f=num(s1["bias"][0], 1, True),
         syn_t1_bias_se=num(s1["bias_se"][0], 1), syn_b_std_f=num(sb["std"][0], 0), syn_b_n=str(sb["n"]),
         syn_rmse_relerr=num(100 / np.sqrt(2 * (sb["n"] - 1)), 0), syn_t1_rmse_f=num(s1["rmse"][0], 0),
         syn_t1_map=" / ".join(num(s1["mapping_rmse"][k]["median"]) for k in ("centre", "cart_band", "corners")))
z2, z1 = abs(sb["bias"][0]) / sb["bias_se"][0], abs(s1["bias"][0]) / s1["bias_se"][0]
V["syn_bias_comment"] = (("pri T2 je odchýlka nevýznamná" if z2 < 2 else f"pri T2 je odchýlka {num(z2, 1)}σ") + ", " +
                         ("pri T1 tiež." if z1 < 2 else f"pri T1 je malá, ale štatisticky významná ({num(z1, 1)}σ, "
                          f"{num(100 * abs(s1['bias'][0]) / s1['truth'][0], 1)} % f); voči rozptylu je malá a RMSE ju obsahuje."))
bs = boot[:, 0].std(ddof=1)
V["boot_vs_syn"] = (f"v scenári b je rozptyl f {num(sb['std'][0], 0)} px." )
syn_small = all((BUD["regions"][k].get("synthetic_rmse") or 0) <= BUD["regions"][k].get("budget_median", BUD["regions"][k]["total_median"]) for k in ("centre", "cart_band", "corners"))
V["syn_vs_budget_old"] = ("Syntetický test dáva vo všetkých oblastiach menšie hodnoty než empirický rozpočet, výsledkom je teda rozpočet "
                      "(bootstrap ⊕ alternatívy)." if syn_small else
                      "Tam, kde syntetický test dáva viac než empirický rozpočet, berie sa hodnota zo syntetického testu.")

# ---------------- sensitivity
if SENS:
    labs = [("dilate_+0.3", "dilatácia čierneho štvorca", "+0,3 px na stranu"), ("dilate_-0.3", "erózia čierneho štvorca", "−0,3 px na stranu"),
            ("shift_x_+0.2", "posun všetkých rohov a strán v x", "0,2 px"), ("shift_y_+0.2", "posun všetkých rohov a strán v y", "0,2 px"),
            ("yscale_corners_+0.0005", "zvislá mierka rohov (rolling-shutter)", "5·10⁻⁴"), ("yscale_all_+0.0005", "zvislá mierka celého obrazu", "5·10⁻⁴"),
            ("aruco", "ArUco rohy namiesto hranových", "0,44 px RMS"), ("bias_corrected", "rohy opravené o posun hrán k čiernej", "0,96 px RMS"),
            ("sides_bias_+0.66", "strany nálepiek opravené o posun hrán k čiernej", "0,66 px von"),
            ("all_bias_corrected", "rohy aj strany opravené", "–"),
            ("edge_dark_+0.3", "hrany posunuté k tmavej strane", "+0,3 px"), ("edge_dark_-0.3", "hrany posunuté k svetlej strane", "0,3 px"),
            ("edge_outward_+0.3", "hrany posunuté von od stredu", "0,3 px")]
    t = ["| porucha | veľkosť | Δf [px] | Δcx, Δcy [px] | Δk1 | zmena zobrazenia stred / pás / rohy [px] |", "|---|---|---|---|---|---|"]
    for k, a, b in labs:
        if k not in SENS:
            continue
        r = SENS[k]
        dp = dict(zip(r["names"], r["dparams"]))
        t.append(f"| {a} | {b} | {num(dp['f'], 2, True)} | {num(dp['cx'], 2, True)}; {num(dp['cy'], 2, True)} | {num(dp['k1'], 4, True)} | "
                 f"{num(r['map']['centre'], 2)} / {num(r['map']['cart_band'], 2)} / {num(r['map']['corners'], 2)} |")
    fr = SENS.get("_frames_summary")
    if fr:
        t.append(f"| jednotlivé fotky namiesto priemeru 7 | 7 fotiek | {num(fr['df_range'][0], 2, True)} … {num(fr['df_range'][1], 2, True)} | | | "
                 f"max {num(fr['map_max']['centre'], 2)} / {num(fr['map_max']['cart_band'], 2)} / {num(fr['map_max']['corners'], 2)} |")
    T["TABLE_SENS"] = "\n".join(t)
    small = [SENS[k]["map"]["cart_band"] for k, _, _ in labs if k in SENS and "bias" not in k]
    V["sens_max_band"] = num(max(small), 2)
    V["sens_sides_df"] = num(dict(zip(SENS["sides_bias_+0.66"]["names"], SENS["sides_bias_+0.66"]["dparams"]))["f"], 1, True) if "sides_bias_+0.66" in SENS else "–"
else:
    T["TABLE_SENS"] = "(citlivosti neboli prepočítané)"
    V["sens_max_band"] = V["sens_sides_df"] = "–"

# ---------------- mapping budget
t = ["| oblasť | štatistická | systematická | rozpočet spolu | syntetický test (scenár b, T2) | jackknife CV | **výsledok (1σ)** | max v oblasti |",
     "|---|---|---|---|---|---|---|---|"]
for k, nm in (("centre", "stred"), ("cart_band", "pás vozíkov"), ("corners", "rohy")):
    r = BUD["regions"][k]
    t.append(f"| {nm} | {num(r['stat_median'])} | {num(r['sys_median'])} | {num(r.get('budget_median', r['total_median']))} | "
             f"{num(r.get('synthetic_rmse'))} | {num(r.get('jackknife_cv'))} | **{num(r['total_median'])} px** | {num(r['total_max'])} |")
T["TABLE_BUDGET"] = "\n".join(t)

# ---------------- extra values (review round 2)
pp = px("k1k2p1p2")
V.update(v_p1p2_f=num(pp["f"], 0), v_p1p2_cy=num(pp["cy"], 0), t2_diff=num(f - 1420, 0),
         syn_t2_rmse_f=num(sb["rmse"][0], 0))
ms = sb["median_block_sigmas"]
V.update(syn_sig_M=num(ms["M"], 1), syn_sig_L=num(ms["L"], 2), syn_sig_V=num(ms["V"], 2), syn_sig_S=num(ms["S"], 2))
V["syn_vs_budget"] = "Výsledná neistota zobrazenia je v každej oblasti najväčšia z troch stĺpcov (rozpočet, syntetický test, jackknife)."
# f-profile: chi2 contribution of each block relative to the minimum
cnt = {k: S["k1k2"]["block_rms"][k][1] for k in "MLVS"}
chi = {k: cnt[k] * prof[:, 2 + j] ** 2 / sig[k] ** 2 for j, k in enumerate("MLVS")}
for k in "MLV":
    V[f"dchi_{k}_lo"] = num(chi[k][lo] - chi[k][i0], 0, True)
    V[f"dchi_{k}_hi"] = num(chi[k][hi] - chi[k][i0], 0, True)
# radii: farthest used edge point and the image corners from the principal point
EJ = load_json(f"{RESULTS}/edges.json")
rr = [np.hypot(*(np.array(e["points"], float) - [cx, cy]).T).max() for b_ in EJ["regions"] for e in b_["edges"]
      if e.get("straight_3d", False) and e.get("verified", "yes") != "no" and len(e["points"])]
cr = [np.hypot(x - cx, y - cy) for x in (0, 1919) for y in (0, 1079)]
V.update(edge_rmax=num(max(rr), 0), corner_rmin=num(min(cr), 0), corner_rmax=num(max(cr), 0))
# bootstrap asymmetry: replay the random streams of 50_combined.py boot (same seeds) to know which stickers were drawn
bf = boot[:, 0]
V.update(boot_med_f=num(np.median(bf), 0), boot_lo_f=num(np.percentile(bf, 16), 0), boot_hi_f=num(np.percentile(bf, 84), 0))
try:
    import sys as _sys
    _argv = _sys.argv
    _sys.argv = ["50_combined.py", "none"]
    _ns = {}
    exec(compile(open("50_combined.py").read().split('if MODE == "fit":')[0], "50c", "exec"), _ns)
    _sys.argv = _argv
    from linedata import stratified_resample  # noqa: E402
    mis = sorted({p_["mi"] for p_ in _ns["PTS"]})
    i33 = [i for i, m in enumerate(_ns["M"]) if (m["cart"], m["id"]) in ((80, 3), (80, 7))]
    has = []
    for sd_ in sorted(int(os.path.basename(q)[14:-5]) for q in glob.glob(f"{CACHE}/combined_boot_*.json")):
        rng_ = np.random.default_rng(sd_)
        nb = len(load_json(f"{CACHE}/combined_boot_{sd_}.json")["boot"])
        k_ = 0
        while k_ < nb:
            pick = rng_.choice(mis, len(mis), replace=True)
            carts_ = {p_["cart"] for q_, mi_ in enumerate(pick) for p_ in _ns["PTS"] if p_["mi"] == mi_}
            if len(carts_) < 2:
                continue
            stratified_resample(_ns["LINES"], rng_)
            has.append(all(i_ in set(pick) for i_ in i33))
            k_ += 1
    has = np.array(has)
    if len(has) == len(bf):
        V.update(boot_miss_pct=num(100 * np.mean(~has), 0), boot_miss_f=num(np.median(bf[~has]), 0), boot_both_f=num(np.median(bf[has]), 0))
    else:
        raise ValueError("replay length mismatch")
except Exception as ex:  # noqa
    print("bootstrap replay failed:", ex)
    V.update(boot_miss_pct="–", boot_miss_f="–", boot_both_f="–")

# ---------------- methods table
mt = open(f"{CACHE}/report_tables.md").read().split("\n")
T["TABLE_METHODS"] = "\n".join(re.sub(r"(?<![\w/:])-(?=\d)", "−", re.sub(r"(?<=\d)\.(?=\d)", ",", x)) for x in mt if x.startswith("|"))

# ---------------- files section
ALT = load_json(f"{RESULTS}/lens_alternatives.json")
FILES = f"""## 14. Súbory vo `results/`

* **Výsledky:** `lens_result.json` (hlavný; `uncertainty_details` s kovarianciou, `mapping_uncertainty_details`, `rms_details`
  s reziduami po nálepkách / radoch / vozíkoch / fotkách, `determination`), `lens_alternatives.json` ({len(ALT)} rovnocenne
  podporených variantov v rovnakom tvare), `lens_by_method.json` (každá metóda a variant zvlášť, pole `method` a `method_id`,
  jednotné kľúče neistôt a rovnaká definícia `rms_reprojection_error_px`).
* **Detekcie a hrany:** `detections.json` (rohy nálepiek s priradením k fyzickým rohom, strany čiastočne zakrytých nálepiek),
  `edges.json` (všetky hrany s bodmi a overením).
* **Obrázky:**
  - hrany: `edges_all.png`, `edges_cart310.png`, `edges_cart80.png`, `edges_scene.png`;
  - nálepky: `markers_overlay.png`, `markers_crops.png`, `markers_rect_views.png`, `markers_edge_bias.png`;
  - reziduá a predikcia: `residuals_stickers_main.png`, `predicted_stickers_crops.png`, `residuals_edges_main.png`,
    `markers_residuals.png`, `markers_complete_residuals.png`;
  - profil ohniska: `f_profiles.png`, `combined_f_profile.png`, `markers_f_profile.png`, `cuboid_f_profile.png`;
  - neistota a porovnanie: `mapping_uncertainty.png`, `methods_comparison.png`, `undistorted_mean.png`;
  - diagnostika geometrie: `geomdiag_*.png`, `cuboid_*.png` (`cuboid_overlay.png` je s objektívom metódy M5, nie hlavným);
  - čiarové metódy (nezávislá implementácia): `40_lines_indep_*.png`;
  - modely objektívu: `altmodels_*.png`;
  - syntetický test a citlivosti (štúdia všetkých metód): `synthetic_*.png`.
"""

# ---------------- assemble
parts = ["REPORT_head.md", "REPORT_part_data.md", "REPORT_part_geom.md", "REPORT_part_methods.md", "REPORT_part_results.md",
         "REPORT_part_improve.md", "REPORT_part_repro.md"]
txt = "\n".join(open(p).read().rstrip() + "\n" for p in parts) + "\n" + FILES
for k, v in T.items():
    txt = txt.replace(k, v)
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", txt)) - set(V))
if missing:
    raise SystemExit(f"missing placeholders: {missing}")
txt = re.sub(r"\{\{(\w+)\}\}", lambda m: V[m.group(1)], txt)
open(f"{RESULTS}/REPORT.md", "w").write(txt)
print("REPORT.md", len(txt), "chars; placeholders", len(V))
