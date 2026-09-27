"""Cart 80: review montages of automatic candidates (by aid) -> cache/edgecrops_cart80/review_<name>.png
usage: python3 10_edges_cart80_review.py name aid1 aid2 ..."""
import sys

import cv2
import numpy as np

from common import CACHE, load_json
import edges80lib as L


def main():
    name = sys.argv[1]
    aids = [int(a) for a in sys.argv[2:]]
    auto = {e["aid"]: e for e in load_json(f"{CACHE}/edges80_auto.json")["edges"]}
    strips = []
    for a in aids:
        e = auto[a]
        P = np.asarray(e["points"])
        strips.append(L.edge_strip(P, label=f"a{a} L{e['length']:.0f}", win=90, scale=4, nwin=3))
    cv2.imwrite(f"{CACHE}/edgecrops_cart80/review_{name}.png", L.montage(strips))


if __name__ == "__main__":
    main()
