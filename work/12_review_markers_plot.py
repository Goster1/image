"""Figure for the marker edge-bias finding of the review: results/markers_edge_bias.png
Left: sticker 80:0 rectified with the nominal 6x6 cell grid built from its measured corners (white
cells overflow the grid lines = black eroded). Right: fitted bias b per sticker (+-1 se), fully
visible vs partly occluded stickers, and the outer-edge estimate for fully visible ones.
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, RESULTS, load_json

L = importlib.import_module("11_markers_lib")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"


def main():
    M = {(m["cart"], m["id"]): m for m in load_json(f"{CACHE}/markers_final.json")}
    rv = load_json(f"{CACHE}/review_markers/review.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    q = np.array(M[(80, 0)]["corners_px"], float)
    C, Mg = 50, 0.8
    Hm, N = L.rect_H(q, C=C, M=Mg)
    R = L.warp(L.mean_gray(), Hm, N)
    a0.imshow(R, cmap="gray", vmin=0, vmax=255, interpolation="bicubic")
    for k in range(7):
        p = Mg * C + k * C - 0.5
        a0.axvline(p, color=ORANGE, lw=0.8)
        a0.axhline(p, color=ORANGE, lw=0.8)
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_title("80:0 rectified, nominal cell grid from its corners\n(white cells overflow the lines: black looks eroded)",
                 fontsize=9.5, color=INK)
    keys = list(rv["bias"].keys())
    x = np.arange(len(keys))
    b = np.array([rv["bias"][k]["b_px"] for k in keys])
    se = np.array([rv["bias"][k]["b_se_px"] for k in keys])
    full = np.array([rv["bias"][k]["all_valid"] for k in keys])
    bo = np.array([rv["bias"][k]["b2_outer_px"] if rv["bias"][k]["all_valid"] else np.nan for k in keys])
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.errorbar(x[full], b[full], yerr=se[full], fmt="o", ms=7, color=BLUE, lw=1.2, label="b, fully visible sticker (interior edges)")
    a1.errorbar(x[~full], b[~full], yerr=se[~full], fmt="s", ms=7, color=ORANGE, lw=1.2, label="b, partly occluded sticker (visible cells)")
    a1.plot(x[full], bo[full], "_", ms=14, mew=2, color=INK, label="outer sides of the black square (2-parameter fit)")
    bs = rv["bias_summary"]
    a1.axhline(bs["full_mean_px"], color=BLUE, lw=0.8, ls="--")
    a1.text(len(keys) - 0.6, bs["full_mean_px"] + 0.03, f"mean {bs['full_mean_px']:+.2f} px", color=INK, fontsize=8.5, ha="right")
    a1.set_xticks(x)
    a1.set_xticklabels(keys, rotation=60, fontsize=8.5)
    a1.set_ylabel("edge bias b [px]  (< 0: edge lies towards black)", fontsize=9.5)
    a1.set_ylim(-1.7, 0.15)
    a1.grid(axis="y", color="#e6e6e3", lw=0.6)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.legend(fontsize=8.5, frameon=False, loc="lower left")
    a1.set_title("Edge-location bias of the corner measurement per sticker\n(corners lie ~|b| inside the true 90 mm square along each side normal)",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/markers_edge_bias.png", dpi=110)
    print("wrote", f"{RESULTS}/markers_edge_bias.png")


if __name__ == "__main__":
    main()
