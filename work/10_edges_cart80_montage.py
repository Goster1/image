"""Cart 80: stack several per-edge verification crops into one image for review.
usage: python3 10_edges_cart80_montage.py outname id1 id2 ...   (ids without the 'cart80_' prefix)"""
import sys

import cv2

from common import CACHE
import edges80lib as L


def main():
    out = sys.argv[1]
    ims = [cv2.imread(f"{CACHE}/edgecrops_cart80/cart80_{i}.png") for i in sys.argv[2:]]
    ims = [im for im in ims if im is not None]
    cv2.imwrite(f"{CACHE}/edgecrops_cart80/montage_{out}.png", L.montage(ims))


if __name__ == "__main__":
    main()
