"""Cart 80: zoomed view of an image window with selected automatic candidates drawn (for inspection).
usage: python3 10_edges_cart80_zoom.py name x0 y0 x1 y1 scale aid1 aid2 ..."""
import sys

import cv2
import numpy as np

from common import CACHE, load_json
import edges80lib as L

COLS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255),
        (128, 0, 255)]


def main():
    name = sys.argv[1]
    x0, y0, x1, y1 = [int(v) for v in sys.argv[2:6]]
    sc = float(sys.argv[6])
    aids = [int(a) for a in sys.argv[7:]]
    auto = {e["aid"]: e for e in load_json(f"{CACHE}/edges80_auto.json")["edges"]}
    img = L.color_image()
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for i, a in enumerate(aids):
        P = np.asarray(auto[a]["points"])
        col = COLS[i % len(COLS)]
        for x, y in P:
            u = int(round((x - x0 + 0.5) * sc - 0.5))
            v = int(round((y - y0 + 0.5) * sc - 0.5))
            if 1 <= u < crop.shape[1] - 1 and 1 <= v < crop.shape[0] - 1:
                crop[v - 1:v + 1, u - 1:u + 1] = col
        ins = P[(P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)]
        if len(ins):
            m = ins[len(ins) // 2]
            cv2.putText(crop, str(a), (int((m[0] - x0) * sc) + 5, int((m[1] - y0) * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(f"{CACHE}/edgecrops_cart80/zoom_{name}.png", crop)


if __name__ == "__main__":
    main()
