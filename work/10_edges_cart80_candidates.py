"""Cart 80: automatic LSD candidates on the mean image, classified by the angle to the three cart
vanishing directions (initial calibration).  Writes cache/edgecrops_cart80/cand_tile*.png and
cache/edges80_lsd.npy (x0,y0,x1,y1,contrast,class,angle)."""
import cv2
import numpy as np

from common import CACHE
import edges80lib as L

OUT = f"{CACHE}/edgecrops_cart80"
CLS = ["cartX", "cartY", "cartZ"]
COL = {0: (0, 0, 255), 1: (0, 200, 0), 2: (255, 80, 0), 3: (160, 160, 160)}


def main():
    segs = L.lsd_segments(minlen=10)
    rows = []
    for s in segs:
        ang = L.classify_dir(s[:2], s[2:4])
        a = np.array([ang[c] for c in CLS])
        k = int(np.argmin(a))
        srt = np.sort(a)
        cls = k if (srt[0] < 3.0 and srt[1] - srt[0] > 2.0) else 3
        rows.append(np.r_[s, cls, srt[0]])
    rows = np.array(rows)
    np.save(f"{CACHE}/edges80_lsd.npy", rows)
    lens = np.hypot(rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1])
    print(len(rows), "segments;", {CLS[i] if i < 3 else "other": int(((rows[:, 5] == i) & (lens > 25)).sum())
                                    for i in range(4)}, "(len>25)")
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    tiles = [(1120, 0, 1720, 260), (1120, 230, 1720, 490), (1120, 460, 1720, 720), (1120, 700, 1720, 990)]
    sc = 2
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        crop = (crop * 0.75).astype(np.uint8)
        for i, r in enumerate(rows):
            if lens[i] < 15:
                continue
            a = (np.array(r[:2]) - [x0, y0]) * sc
            b = (np.array(r[2:4]) - [x0, y0]) * sc
            if not (0 <= (a[0] + b[0]) / 2 < crop.shape[1] and 0 <= (a[1] + b[1]) / 2 < crop.shape[0]):
                continue
            cv2.line(crop, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), COL[int(r[5])], 1,
                     cv2.LINE_AA)
            if lens[i] > 30:
                m = ((a + b) / 2).astype(int)
                cv2.putText(crop, str(i), (int(m[0]) + 2, int(m[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                            (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUT}/cand_tile{ti}.png", crop)


if __name__ == "__main__":
    main()
