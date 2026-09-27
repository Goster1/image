"""Markdown tables for REPORT.md from the method files and the final budget -> work/cache/report_tables.md"""
import os

import numpy as np

from common import CACHE, RESULTS, fold_margin, load_json
from evaltools import grid, mapping_displacement, region_defs

main = load_json(f"{RESULTS}/lens_result.json")
K0 = np.array(main["camera_matrix"])
d0 = np.array(main["dist_coeffs"])
uv, _ = grid(30)
regs = region_defs()

ROWS = [  # (label, file, method-name-in-list or None, uses dimensions?, note)
    ("M6 kombinovaný – HLAVNÝ", "method_combined.json", None, "áno (všetky nálepky, plná váha)", "nálepky + 61 hrán + strany zakrytých nálepiek"),
    ("M6 kombinovaný, robustné váhy nálepiek", "method_combined_alternatives.json", "combined_k1k2_robust", "áno (top-nálepky ~0 váha)", "Cauchy váhy po nálepkách"),
    ("M6 kombinovaný, k1,k2,k3", "method_combined_alternatives.json", "combined_k1k2k3", "áno", "k3 len konzistentné"),
    ("M6 kombinovaný, pp v strede obrazu", "method_combined_alternatives.json", "combined_k1k2_ppfixed", "áno", ""),
    ("M6 kombinovaný, fx≠fy", "method_combined_alternatives.json", "combined_k1k2_fxfy", "áno", "fx≠fy neurčené"),
    ("M6 kombinovaný, +p1,p2", "method_combined_alternatives.json", "combined_k1k2p1p2", "áno", "p1,p2 pohltia nesúlad geometrie"),
    ("M4 spoločný fit z čiar (nezávislá impl., spoločná zvislica)", "method_lines_joint_indep.json", None, "nie", ""),
    ("M4 spoločný fit z čiar, podlahová čiara ako 1 priamka", "method_lines_joint_indep_mergedfloor.json", None, "nie", ""),
    ("M4 spoločný fit z čiar (orchestrátor, samostatné zvislice)", "method_lines_joint_lc.json", None, "nie", ""),
    ("M3 úbežníky (nezávislá impl.)", "method_vanishing_indep.json", None, "nie", "pp = stred skreslenia"),
    ("M3 úbežníky (orchestrátor)", "method_vanishing_lc.json", None, "nie", ""),
    ("M2 plumb-line (skreslenie; f z M3)", "method_plumbline_indep.json", None, "nie", "len k1/f², k2/f⁴ a stred"),
    ("M9 Brown k1,k2 (štúdia modelov, spoločné dáta)", "method_alt_brown_k1k2.json", None, "áno", ""),
    ("M9 Brown k1,k2,k3", "method_alt_brown_k1k2k3.json", None, "áno", ""),
    ("M9 divízny l1,l2", "method_alt_division_l1l2.json", None, "áno", "prevedený na Brown"),
    ("M9 Kannala-Brandt k1,k2", "method_alt_kb_k1k2.json", None, "áno", "prevedený na Brown"),
    ("M7 dva vozíky (normály podlahy)", "method_twocart.json", None, "áno", "len konzistencia"),
    ("M8 tvar nálepiek", "method_markershape.json", None, "nie (len štvorce)", "príliš slabé"),
    ("M5 kváder s policami (výkres)", "method_cuboid.json", None, "áno", "vychýlené geometriou"),
    ("M1 len nálepky (výkres)", "method_markers.json", None, "áno", "vychýlené, prehnuté"),
]


def load(fn, name):
    p = f"{CACHE}/{fn}"
    if not os.path.exists(p):
        return None
    j = load_json(p)
    if isinstance(j, list):
        for it in j:
            if it.get("method") == name:
                return it
        return None
    return j


def g(u, *keys):
    for k in keys:
        if u and u.get(k) is not None:
            try:
                return float(u[k])
            except Exception:  # noqa
                pass
    return None


lines = ["| metóda | rozmery z výkresu | f [px] | cx | cy | k1 | k2 | k3 | fold | Δ mapovania vs hlavný: stred / pás / rohy [px] | neistota zobrazenia metódy (stred / pás / rohy) | pozn. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for lab, fn, nm, dims, note in ROWS:
    j = load(fn, nm)
    if j is None:
        lines.append(f"| {lab} | {dims} | – | | | | | | | | | chýba |")
        continue
    K = np.array(j["camera_matrix"])
    d = np.array(j["dist_coeffs"], float)
    u = j.get("uncertainty") or {}
    fs = g(u, "fx_px_1sigma", "f_1sigma", "fx_1sigma")
    cxs = g(u, "cx_px_1sigma", "cx_1sigma")
    cys = g(u, "cy_px_1sigma", "cy_1sigma")
    fm = fold_margin(K, d)
    try:
        D = mapping_displacement(K0, d0, K, d, uv)
        r = np.hypot(D[:, 0], D[:, 1])
        dm = " / ".join(f"{np.nanmedian(r[f(uv)]):.1f}" for f in regs.values())
    except Exception:  # noqa
        dm = "–"
    mu = j.get("mapping_uncertainty_px") or {}
    mus = " / ".join(f"{float(mu[k]):.1f}" if isinstance(mu.get(k), (int, float)) else "–" for k in ("centre", "cart_band", "corners"))
    fstr = f"{K[0,0]:.0f}" + (f" ± {fs:.0f}" if fs else "")
    if abs(K[0, 0] - K[1, 1]) > 0.5:
        fstr = f"{K[0,0]:.0f}/{K[1,1]:.0f}"
    lines.append(f"| {lab} | {dims} | {fstr} | {K[0,2]:.0f}" + (f" ± {cxs:.0f}" if cxs else "") + f" | {K[1,2]:.0f}" + (f" ± {cys:.0f}" if cys else "") +
                 f" | {d[0]:+.3f} | {d[1]:+.3f} | {d[4]:+.3f} | {'ok' if fm > 0 else 'PREHNUTÝ'} | {dm} | {mus} | {note} |")
out = ["## Tabuľka metód (automaticky, 97_report_tables.py)", "", *lines, ""]

# sticker residual table of the main fit
if os.path.exists(f"{CACHE}/final_sticker_residuals.json"):
    sr = load_json(f"{CACHE}/final_sticker_residuals.json")["per_sticker"]
    out += ["## Reziduá nálepiek s hlavným objektívom (geometria z výkresu, pózy vozíkov robustne)", "",
            "| vozík | id | poloha | RMS [px] | priemerný posun (x, y) [px] |", "|---|---|---|---|---|"]
    for s in sr:
        mo = s["mean_offset_px"]
        out.append(f"| {s['cart']} | {s['id']} | {s['role']} | {s['rms_px']:.1f} | " + (f"({mo[0]:+.1f}, {mo[1]:+.1f})" if mo else "–") + " |")
    out.append("")
bud = load_json(f"{CACHE}/final_budget.json")
out += ["## Rozpočet neistoty zobrazenia (1σ, px, medián oblasti)", "", "| oblasť | štatistická | systematická | spolu (medián) | spolu (max) |", "|---|---|---|---|---|"]
for k, v in bud["regions"].items():
    out.append(f"| {k} | {v['stat_median']:.1f} | {v['sys_median']:.1f} | {v['total_median']:.1f} | {v['total_max']:.1f} |")
out.append("")
open(f"{CACHE}/report_tables.md", "w").write("\n".join(out))
print("\n".join(out))
