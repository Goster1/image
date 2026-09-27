# Review of sub-task 43 (lens-model choice): uncertainty of the recommended models

Script `work/43_altmodels_review.py` (run after `43_altmodels_report.py`; ~7 min with 3 processes). Numbers: `work/cache/altmodels_review.json`. Same estimator, data and fixed block sigmas as `43_altmodels_fit.py`.

## What was wrong

1. **Clusters.** The sandwich treated every traced edge as an independent cluster; 13 edges are the second boundary of the same physical member (`pair_of`: both sides of a lip, tape, rod, pole). Here clusters = 48 physical members (+ 20 stickers).
2. **Few influential clusters.** The cluster-robust sandwich is only reliable when the information on a parameter is spread over many clusters. It is not: cy is carried by the floor_H group (3 members: white tape between the carts, blue line above it, blue line on the left, assumed parallel and perpendicular to the free scene vertical); the edges-only focal length by the two short cart-310 post edges (cart 80's two Z edges are nearly collinear outlines of one post). Delete-one-cluster jackknife (clusters as above; also of the rotation-compensated mapping directly) is used as the check.

## Results (Brown k1,k2 = recommended, Brown k1,k2,k3 = alternative)

| model / data / VP policy | estimate | 1 sigma: member sandwich / jackknife | f over delete-one folds [px] | mapping sandwich (member) c / band / corners [px] | mapping jackknife [px] |
|---|---|---|---|---|---|
| B_k1k2 / E / noboard | f=1436, cx=927.6, cy=502.1, k1=-0.3366, k2=0.09133 | f 32.9 / 156, cx 6.93 / 9.57, cy 2.87 / 12.3, k1 0.0163 / 0.0759, k2 0.00984 / 0.0445 | 1412-1544 | 2.23 / 9.54 / 13.51 | 11.64 / 49.31 / 65.71 |
| B_k1k2 / E / nofloor | f=1429, cx=925.8, cy=479.9, k1=-0.331, k2=0.08699 | f 35.1 / 5.75e+03, cx 6.67 / 7.93, cy 24 / 39.3, k1 0.0171 / 8.13, k2 0.0102 / 57.9 | 1405-7297 | 2.91 / 10.98 / 15.23 | 401.72 / 1909.81 / 178621.98 |
| B_k1k2 / ME / noboard | f=1434, cx=927.5, cy=501.8, k1=-0.3355, k2=0.0906 | f 29 / 43.3, cx 6.87 / 9.68, cy 2.87 / 15, k1 0.0146 / 0.0213, k2 0.00899 / 0.0128 | 1413-1457 | 2.33 / 9.87 / 13.43 | 3.38 / 13.93 / 19.33 |
| B_k1k2 / ME / nofloor | f=1426, cx=925.4, cy=477.5, k1=-0.329, k2=0.08561 | f 30.4 / 47.9, cx 6.44 / 7.78, cy 22.8 / 42.7, k1 0.0147 / 0.0235, k2 0.00887 / 0.0139 | 1404-1450 | 2.79 / 10.22 / 14.42 | 4.32 / 15.34 / 21.93 |
| B_k1k2k3 / E / noboard | f=1432, cx=931.1, cy=505.6, k1=-0.355, k2=0.1417, k3=-0.0337 | f 32.8 / 205, cx 5.5 / 9.28, cy 4.1 / 7.72, k1 0.0221 / 0.109, k2 0.0331 / 0.103, k3 0.018 / 0.0439 | 1409-1578 | 2.72 / 11.53 / 15.73 | 15.43 / 64.80 / 83.20 |
| B_k1k2k3 / E / nofloor | f=1427, cx=930, cy=488.4, k1=-0.3482, k2=0.13, k3=-0.02767 | f 34.4 / 1.66e+03, cx 5.45 / 7.43, cy 25.3 / 31.6, k1 0.024 / 1.27, k2 0.0348 / 2.61, k3 0.0182 / 2.43 | 1404-3111 | 3.17 / 11.74 / 16.43 | 132.01 / 424.74 / 341.45 |
| B_k1k2k3 / ME / noboard | f=1432, cx=930.7, cy=505, k1=-0.3528, k2=0.1368, k3=-0.03074 | f 29.1 / 44.3, cx 5.52 / 9.57, cy 3.98 / 8.57, k1 0.0211 / 0.0307, k2 0.0326 / 0.0463, k3 0.0178 / 0.0249 | 1411-1455 | 2.35 / 9.99 / 13.93 | 3.39 / 14.23 / 19.57 |
| B_k1k2k3 / ME / nofloor | f=1426, cx=929.3, cy=485, k1=-0.3449, k2=0.1239, k3=-0.0244 | f 30.4 / 46.8, cx 5.26 / 7.25, cy 24 / 34.8, k1 0.0219 / 0.0306, k2 0.0322 / 0.0409, k3 0.0169 / 0.0208 | 1405-1451 | 3.01 / 10.92 / 15.22 | 4.11 / 14.95 / 20.92 |

(policy `nofloor` = floor lines used for straightness only, no floor VP groups.) Edge-cluster sandwich of the original fits (ME, Brown k1,k2): f 26.3, cx 7.3, cy 4.0, k1 0.013, k2 0.0084; mapping 2.14 / 9.08 / 12.67 px.

## Where the cy information comes from (Brown k1,k2, edges only)

chi2 (fixed block sigmas) with cy fixed and everything else refitted:

| cy | dchi2 | f | 310 | 80 | floor_H | floor_V | wv | S |
|---|---|---|---|---|---|---|---|---|
| 420 | 922.5 | 1406.5 | 875 | 1176 | 805 | 22 | 22 | 1262 |
| 440 | 514.3 | 1416.4 | 884 | 1151 | 474 | 22 | 21 | 1202 |
| 460 | 231.5 | 1424.9 | 886 | 1135 | 246 | 22 | 20 | 1161 |
| 480 | 63.9 | 1431.4 | 874 | 1133 | 115 | 21 | 19 | 1141 |
| 490 | 19.4 | 1433.8 | 864 | 1138 | 81 | 21 | 18 | 1137 |
| 502 | 0.0 | 1436.2 | 854 | 1147 | 65 | 20 | 17 | 1136 |
| 510 | 8.8 | 1437.6 | 848 | 1154 | 72 | 20 | 16 | 1137 |
| 520 | 46.3 | 1439.3 | 843 | 1164 | 101 | 20 | 16 | 1141 |
| 540 | 219.9 | 1442.7 | 839 | 1183 | 247 | 20 | 15 | 1155 |
| 560 | 540.0 | 1446.8 | 842 | 1197 | 530 | 20 | 14 | 1176 |

Variants (member sandwich 1 sigma):

* all (orchestrator policy): f 1436.3 +- 32.9, cx 927.6 +- 6.9, cy 502.1 +- 2.9
* floor lines straightness only: f 1428.6 +- 35.1, cx 925.8 +- 6.7, cy 479.9 +- 24.0
* no floor lines: f 1434.6 +- 34.5, cx 927.0 +- 7.8, cy 497.7 +- 27.8
* no floor lines, no scene verticals: f 1434.1 +- 34.6, cx 926.4 +- 7.8, cy 495.8 +- 28.0
* floor_H without the left blue line: f 1434.1 +- 34.3, cx 927.0 +- 7.5, cy 495.4 +- 26.5
* floor_H without the right tape / blue line: f 1432.3 +- 34.6, cx 926.3 +- 7.2, cy 489.5 +- 25.9

Fit without floor VP groups: direction of each floor line in the plane perpendicular to the scene vertical, relative to scene_tapeH_top (mod 180 deg): scene_tapeV_left +89.87, scene_tapeV_right +89.89, scene_tapeH_top +0.00, scene_tapeH_bottom +0.21, scene_blueH_top +0.10, scene_blueL_top +0.37, scene_blueL_bottom +0.41 deg. Vertical directions: scene_vertical_vs_cart80_Z 2.17 deg, scene_vertical_vs_cart310_Z 0.59 deg, cart80_Z_vs_cart310_Z 1.59 deg.

## Conclusions of the review

* The model choice (Brown k1,k2; fx = fy; pp free; p1 = p2 = k3 = 0) and the point estimates hold (reproduced exactly; the E fit reaches the same minimum from starts f 1300-1500, cy 386-540).
* The parameter uncertainties were too small. Recommended model (ME): f 1434.4 +- 43 (was +-26), cx 927.5 +- 9.7 (7.3), cy 501.8 +- 15 (4.0) if the floor lines are exactly parallel, +- 43 without that assumption (then cy = 477.5), k1 -0.3355 +- 0.021 (0.013), k2 0.0906 +- 0.013 (0.0084).
* Parameter part of the mapping uncertainty (jackknife, rot.-comp., median): 3.4 / 13.9 / 19.3 px with the floor groups, 4.3 / 15.3 / 21.9 px without (was 2.14 / 9.08 / 12.67). The method file now carries the larger one combined in quadrature with the model-choice part.
* Edges only: the focal length is NOT robustly determined: deleting either of the two cart-310 post edges moves f by about +108 px; jackknife sd 156 px (sandwich 33); without the floor VP groups one deletion sends f to 7297 px. The stickers stabilise f in the joint fit (fold range 1413-1457 px).
* cy is only *consistent* at the 15-40 px level: floor groups 502, no floor groups 478-480, plumb-line distortion centre 426 +- 25, relaxed-geometry stickers 392 (diagnostic). The +22 px pull of the floor groups corresponds to a ~0.4 deg non-parallelism of the left blue line and the white tape, far below what painted lines / the 0.5-2 deg spread of the verticals can guarantee.
* The other method_alt_*.json files keep the edge-cluster sandwich uncertainties; they are too small by the same factors (f ~1.5x, cy ~4x, mapping ~1.5x); a note was added to each.
