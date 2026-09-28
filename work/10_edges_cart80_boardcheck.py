"""Cart 80: independent position check of the top-board edges with a plane homography from the
marker corners on the same board (markers 0 + 92 on the X=0 board, 1 + 2 on the X=1600 board; drawing
positions, sticker face in the top-plate plane Z=0).  The traced edge points are mapped to the board
plane; the mean X (for cart-Y edges) or Y (for cart-X edges) is compared with the drawing value.
Done with raw pixels and with points undistorted by the initial calibration (the homography is only
locally valid; the edges lie just outside the marker area).  Output: cache/edges80_boardcheck.json"""
import cv2
import numpy as np

from common import CACHE, load_json, save_json, marker_corners_3d
import edges80lib as L

BOARDS = {"X0": [0, 92], "X1600": [1, 2]}
EDGES = {"cart80_y_board0_outer": ("X0", 0, 0.0), "cart80_y_board0_inner": ("X0", 0, None),
         "cart80_x_board0_front": ("X0", 1, 0.0),
         "cart80_y_board1600_outer": ("X1600", 0, 1600.0), "cart80_y_board1600_inner": ("X1600", 0, None),
         "cart80_y_board1600_inner_top": ("X1600", 0, None), "cart80_x_board1600_front": ("X1600", 1, 0.0)}


def main():
    ic = load_json(f"{CACHE}/initial_calib.json")
    keys = [tuple(k) for k in ic["keys"]]
    corners = {k: np.array(c) for k, c in zip(keys, ic["corners_mean"])}
    edges = {e["id"]: np.array(e["points"]) for e in load_json(f"{CACHE}/edges_cart80.json")["edges"]}
    out = {}
    for eid, (board, coord, nominal) in EDGES.items():
        if eid not in edges:
            continue
        img_pts, obj = [], []
        for m in BOARDS[board]:
            img_pts.append(corners[(80, m)])
            obj.append(marker_corners_3d(m, 80, ic["rot"][f"80_{m}"])[:, :2])
        img_pts = np.vstack(img_pts)
        obj = np.vstack(obj)
        res = {}
        for mode in ("raw", "undist"):
            ip = img_pts if mode == "raw" else L.undist_px(img_pts)
            Hm, _ = cv2.findHomography(ip, obj, 0)
            P = edges[eid] if mode == "raw" else L.undist_px(edges[eid])
            Q = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            back = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2)
            fit_rms = float(np.sqrt(np.mean(np.sum((back - ip) ** 2, axis=1))))
            v = Q[:, coord]
            res[mode] = dict(mean_mm=round(float(np.mean(v)), 1), std_mm=round(float(np.std(v)), 1),
                             range_other_mm=[round(float(Q[:, 1 - coord].min())), round(float(Q[:, 1 - coord].max()))],
                             homography_fit_rms_px=round(fit_rms, 2))
        res["nominal_mm"] = nominal
        res["coordinate"] = "X" if coord == 0 else "Y"
        out[eid] = res
        print(eid, res)
    save_json(out, f"{CACHE}/edges80_boardcheck.json")


if __name__ == "__main__":
    main()
