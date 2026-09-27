"""Zoomed crops of the four end boards and of the shelf-sticker cards (mean colour image) for the visual
plausibility check of the geometry diagnosis. Valid corners are marked (red), invalid completions (grey).
Output: results/geomdiag_crops.png
"""
import importlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")

PANELS = [
    ("cart 310, X=0 end board (ids 0, 322) + stack 3/5/7", (530, 620, 860, 1005), 310),
    ("cart 310, X=1600 end board (ids 1, 2) + cards 4/6/8", (200, 0, 680, 215), 310),
    ("cart 80, X=0 end board (ids 0, 92) + cards 3/5/7", (1230, 0, 1720, 160), 80),
    ("cart 80, X=1600 end board (ids 1, 2) + stack 4/6/8", (1160, 630, 1560, 1000), 80),
    ("cart 80 X=1600: inner board edge (tan end face ~4-5 px)", (1270, 860, 1540, 925), 80),
    ("cart 310 X=0: inner board edge (brown end face ~4-6 px)", (560, 850, 830, 935), 310),
]

NOTES = {
    0: "thin white plate, brown inner end face ~4-6 px (~10 mm);\nshelf cards: small white cards with visible front faces",
    1: "white plate, left (back) part bends down / folded;\nno 60-90 mm side face visible at the inner edge",
    2: "white plate, back corner curled up (right);\nshelf cards partly hidden under the plate",
    3: "white plate, back corner curled up (right), tan inner\nend face ~4-5 px; shelf cards flat-looking",
    4: "a 65 mm tall inner face would be ~24 px wide here\n(0.37 px/mm vertical scale)",
    5: "a 60 mm tall inner face would be ~21 px wide here\n(0.36 px/mm vertical scale)",
}


def main():
    im = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
    ms = G.markers()
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=110)
    for i, (title, (x0, y0, x1, y1), cart) in enumerate(PANELS):
        ax = axs.flat[i]
        ax.imshow(im[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="bicubic")
        for m in ms:
            if m["cart"] != cart:
                continue
            c = m["corners_px"]
            inside = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)
            if not inside.any():
                continue
            v = m["valid"]
            ax.plot(c[v, 0], c[v, 1], "o", mfc="none", mec="r", ms=5, mew=1)
            ax.plot(c[~v, 0], c[~v, 1], "x", color="0.6", ms=4)
            cc = c.mean(0)
            if x0 < cc[0] < x1 and y0 < cc[1] < y1:
                ax.text(cc[0], cc[1], f"{m['id']}", color="yellow", fontsize=9, ha="center", va="center",
                        bbox=dict(fc="k", alpha=0.4, pad=1))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(title, fontsize=9)
        ax.text(0.01, 0.01, NOTES[i], transform=ax.transAxes, fontsize=7.5, color="w", va="bottom", bbox=dict(fc="k", alpha=0.55))
        ax.tick_params(labelsize=7)
    fig.suptitle("Visual check of the end boards and shelf-sticker cards (mean of the 7 stills; red o = valid corners, grey x = completions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_crops.png")
    print("saved")


if __name__ == "__main__":
    main()
