# Method: cuboid cart model (stickers + exact edges, drawing dimensions exact)

Script `work/41_cuboid_fit.py` (helpers `41_cuboid_lib.py`). Data: 62 sticker corners, 13 sticker sides, 3 exact structure edges (cart310_board0_outer, cart80_x_board0_front, cart80_x_board1600_front). Edge files reviewed: {310: True, 80: True}.

Main model (min. leave-one-sticker-out error among the pp-fixed, fx=fy models, cluster weighting): **k1_ppfix**. Free-pp variants reach a lower LOSO (k1_ppfree 8.16 px, k1k2_ppfree 7.18 px, k1k2k3_ppfree 8.88 px, k1k2_ppfree_fxfy 6.78 px, k1k2p1p2_ppfree 10.40 px) but their pp is not determined: k1k2_ppfree: all (887, 396), corners_only (983, 287), corners+sides (994, 293), corners+exact_edges (879, 362); k1k2_ppfree_fxfy: all (873, 401), corners_only (979, 300), corners+sides (990, 303), corners+exact_edges (974, 301).

| param | value | bootstrap 1-sigma | cov 1-sigma |
|---|---|---|---|
| f | 1359.3 | 35.9 | 12.1 |
| k1 | -0.21013 | 0.0112 | 0.00888 |

Block RMS: corners 6.09 px per point (max 10.9), sticker sides 2.13 px, exact edges 8.79 px. LOSO 8.30 px.
Mapping uncertainty (rot.-comp., bootstrap): centre 2.90 px, cart_band 13.42 px, corners 21.87 px, whole_image 15.76 px

Fold barrier (fold margin >= 0.03) ON in every fit; main-model fold margin 0.030 - the solution SITS ON THE BARRIER: the data (drawing geometry) push the distortion towards a lens that folds inside the image; k1, k2 are then fixed by the validity constraint, not by the data. Bootstrap: 86 % of the replicates on the barrier.

| model / weighting | f | pp | dist | corners px/pt | sides | edges | LOSO | fold margin |
|---|---|---|---|---|---|---|---|---|
| k1_ppfix|cluster | 1359 | centre | k1=-0.210 | 6.09 | 2.13 | 8.79 | 8.30 | 0.030 |
| k1k2_ppfix|cluster | 1345 | centre | k1=-0.242, k2=0.021 | 5.94 | 2.25 | 7.92 | 8.31 | 0.030 |
| k1_ppfree|cluster | 1372 | (959, 476) | k1=-0.202 | 6.02 | 2.14 | 8.07 | 8.16 | 0.030 |
| k1k2_ppfree|cluster | 1358 | (887, 396) | k1=-0.232, k2=0.024 | 6.36 | 2.56 | 3.34 | 7.18 | 13.077 |
| k1k2k3_ppfree|cluster | 1338 | (675, 437) | k1=0.054, k2=-0.434, k3=0.233 | 6.68 | 2.34 | 1.41 | 8.88 | 3381.711 |
| k1k2_ppfree_fxfy|cluster | 1339 | (873, 401) | k1=-0.208, k2=0.019 | 6.18 | 2.52 | 3.23 | 6.78 | 9.719 |
| k1k2p1p2_ppfree|cluster | 1363 | (955, 365) | k1=-0.448, k2=0.417, p1=0.011, p2=-0.003 | 5.33 | 2.28 | 6.02 | 10.40 | 401.145 |
| k1_ppfix|point | 1301 | centre | k1=-0.193 | 9.92 | 4.34 | 0.59 | nan | 0.031 |
| k1k2_ppfix|point | 1308 | centre | k1=-0.319, k2=0.133 | 8.88 | 3.15 | 0.95 | nan | 119.233 |
| k1_ppfree|point | 1395 | (566, 312) | k1=-0.068 | 19.64 | 2.00 | 0.32 | nan | 0.365 |
| k1k2_ppfree|point | 1338 | (694, 360) | k1=-0.198, k2=0.018 | 7.78 | 2.48 | 1.18 | nan | 8.316 |
| k1k2k3_ppfree|point | 1344 | (585, 434) | k1=0.187, k2=-0.442, k3=0.133 | 8.78 | 1.87 | 0.60 | nan | 1737.570 |
| k1k2_ppfree_fxfy|point | 1325 | (715, 391) | k1=-0.200, k2=0.018 | 7.58 | 2.57 | 1.16 | nan | 8.536 |
| k1k2p1p2_ppfree|point | 1354 | (814, 276) | k1=-0.261, k2=0.031, p1=0.013, p2=0.024 | 9.58 | 2.16 | 0.80 | nan | 17.674 |

Without the fold barrier (original run; all k1 / k1k2 lenses fold inside the image - not valid):

| model (cluster) | f | pp | dist | corners px/pt | sides | edges | fold margin |
|---|---|---|---|---|---|---|---|
| k1_ppfix | 1341 | centre | k1=-0.231 | 5.89 | 2.30 | 7.80 | -0.020 |
| k1k2_ppfix | 1327 | centre | k1=-0.118, k2=-0.177 | 5.66 | 2.49 | 7.64 | -0.118 |
| k1_ppfree | 1353 | (885, 453) | k1=-0.222 | 5.75 | 2.28 | 6.91 | -0.077 |
| k1k2_ppfree | 1355 | (749, 423) | k1=-0.096, k2=-0.126 | 6.21 | 2.44 | 3.02 | -0.211 |
| k1k2k3_ppfree | 1338 | (675, 437) | k1=0.054, k2=-0.434, k3=0.233 | 6.68 | 2.34 | 1.41 | 3381.718 |
| k1k2_ppfree_fxfy | 1338 | (838, 439) | k1=-0.116, k2=-0.124 | 5.82 | 2.44 | 4.40 | -0.174 |
| k1k2p1p2_ppfree | 1363 | (955, 365) | k1=-0.448, k2=0.416, p1=0.011, p2=-0.003 | 5.33 | 2.28 | 6.01 | 400.324 |

Exact-edge check (implied shift of the traced edge from its drawing line with the fitted camera, mm):
* cart310_board0_outer: joint -22.0 mm along X (-10.38 px); markers-only -20.5 mm (-9.61 px)
* cart80_x_board0_front: joint +7.6 mm along Y (-5.09 px); markers-only +7.8 mm (-5.32 px)
* cart80_x_board1600_front: joint +7.0 mm along Y (-3.82 px); markers-only +7.7 mm (-4.25 px)

## Prediction check: modelled cart lines vs traced edges

Script `work/41_cuboid_predict.py`; full table `work/cache/cuboid_offsets.md/.json`; plots `results/cuboid_overlay.png`, `results/cuboid_crops.png`, `results/cuboid_crops_anchor.png`, `results/cuboid_offsets.png`.
Offset traced - predicted [px] (+ = traced edge on the + side of the axis in [..]) +- camera 1-sigma (bootstrap, robust) / implied shift [mm]:

| edge | markers-only | joint | edge lens + shelf-sticker poses (diag.) |
|---|---|---|---|
| cart310_board0_outer | -9.6 +-3.5 / dX -21 | -10.4 +-4.8 / dX -22 | -26.2 / dX -55 |
| cart310_board1600_outer | +7.5 +-2.8 / dX +12 | +6.2 +-1.5 / dX +9 | +16.9 / dX +27 |
| cart310_x_board0_front | -1.1 +-1.7 / dY -2 | +2.2 +-3.8 / dY +4 | +6.1 / dY +11 |
| cart310_x_board1600_front | +4.3 +-3.8 / dY +6 | +5.0 +-4.0 / dY +7 | +14.3 / dY +22 |
| cart310_x_A_front_fall | -2.1 +-2.6 / dZ -13 | -1.8 +-2.3 / dZ -12 | +0.3 / dZ +2 |
| cart310_x_B_front_fall | +4.2 +-1.6 / dZ +36 | +3.7 +-1.7 / dZ +32 | +2.8 / dZ +25 |
| cart310_x_B_front_rise | +1.5 +-1.6 / dZ +12 | +1.0 +-1.7 / dZ +8 | +0.0 / dZ +0 |
| cart310_x_C_front_fall | +1.9 +-1.8 / dZ +22 | +1.4 +-2.5 / dZ +17 | -0.7 / dZ -8 |
| cart310_x_C_front_rise | +0.3 +-1.8 / dZ +4 | -0.4 +-2.8 / dZ -4 | -2.7 / dZ -31 |
| cart310_x_D_front_fall | +3.7 +-2.7 / dZ +51 | +2.9 +-3.6 / dZ +39 | -0.0 / dZ -0 |
| cart310_x_E_front_fall | -1.6 +-3.4 / dZ -27 | -1.9 +-3.8 / dZ -32 | -4.6 / dZ -76 |
| cart310_x_B_inner_fall | +13.9 +-1.7 / dZ +121 | +13.4 +-1.6 / dZ +117 | +12.1 / dZ +110 |
| cart310_x_back_inner | -79.2 +-2.5 / dZ -254 | -74.9 +-3.1 / dZ -249 | -59.1 / dZ -224 |
| cart310_z_post_front_left_right | -2.9 +-2.5 / dX -16 | -0.7 +-2.7 / dX -3 | -0.8 / dX -4 |
| cart310_z_post_front_right | -4.5 +-1.5 / dX -10 | -5.1 +-1.8 / dX -12 | -2.1 / dX -5 |
| cart80_x_board0_front | +5.3 +-3.7 / dY +8 | +5.1 +-4.4 / dY +8 | +15.4 / dY +24 |
| cart80_x_board1600_front | +4.2 +-2.9 / dY +8 | +3.8 +-2.5 / dY +7 | +13.3 / dY +25 |
| cart80_y_board0_outer | -9.1 +-1.3 / dX -15 | -7.7 +-2.2 / dX -11 | -17.3 / dX -28 |
| cart80_y_board1600_outer | +8.0 +-2.7 / dX +18 | +9.1 +-3.6 / dX +21 | +25.7 / dX +59 |
| cart80_x_back_rail_out | -19.3 +-3.1 / dZ -63 | -17.0 +-2.9 / dZ -56 | +3.2 / dZ +12 |
| cart80_x_back_rail_in | -31.2 +-3.1 / dZ -102 | -28.6 +-2.7 / dZ -95 | -9.5 / dZ -37 |
| cart80_x_back_low | -78.8 +-3.2 / dZ -254 | -76.4 +-3.1 / dZ -247 | -55.7 / dZ -210 |
| cart80_x_A_front_out | +2.4 +-3.3 / dZ +14 | +0.8 +-1.8 / dZ +5 | +3.6 / dZ +22 |
| cart80_x_A_front_in | +9.8 +-3.7 / dZ +57 | +7.8 +-1.7 / dZ +46 | +10.2 / dZ +64 |
| cart80_x_B_front | +1.8 +-2.5 / dZ +14 | +0.8 +-1.9 / dZ +6 | -0.9 / dZ -8 |
| cart80_x_C_front | +7.6 +-2.9 / dZ +80 | +7.6 +-3.1 / dZ +78 | +3.3 / dZ +35 |
| cart80_x_D_front | +8.0 +-3.6 / dZ +102 | +8.7 +-3.8 / dZ +108 | +3.5 / dZ +45 |
| cart80_x_E_front | +2.0 +-4.2 / dZ +31 | +3.4 +-4.9 / dZ +51 | -2.2 / dZ -32 |
| cart80_x_E_front_in | +4.8 +-4.2 / dZ +74 | +6.2 +-4.9 / dZ +92 | +0.6 / dZ +10 |
| cart80_z_post1600_sil | +1.7 +-2.5 / dX +11 | -0.2 +-3.4 / dX -1 | +0.2 / dX +1 |
| cart80_z_post1600_sil_low | -3.9 +-1.1 / dX -20 | -4.5 +-1.3 / dX -23 | -5.3 / dX -26 |

* markers_only: sticker rows RMS px / best Z shift mm: 80:top 5.0/+7, 310:top 5.8/+5, 80:A 7.2/-24, 310:A 8.6/-33, 80:B 3.5/-10, 310:B 3.5/-7, 80:C 6.0/+32, 310:C 3.6/+17
  * lips cart 310: common lip height +8 mm, per-lip deviation A -3.5, B +3.2, C +1.1, D +3.1, E -2.1 px (rms 2.98); + depth scale -> rms 2.63; + D/E shift +13 mm -> rms 2.92
  * lips cart 80: common lip height +32 mm, per-lip deviation A -3.3, B -2.3, C +4.6, D +5.5, E -0.1 px (rms 3.74); + depth scale -> rms 2.65; + D/E shift +48 mm -> rms 3.09
* joint: sticker rows RMS px / best Z shift mm: 80:top 6.2/+7, 310:top 6.5/+9, 80:A 8.0/-27, 310:A 8.3/-32, 80:B 4.0/-18, 310:B 4.3/-13, 80:C 5.6/+22, 310:C 2.0/+5
  * lips cart 310: common lip height +6 mm, per-lip deviation A -2.8, B +3.0, C +0.9, D +2.4, E -2.3 px (rms 2.61); + depth scale -> rms 2.43; + D/E shift +5 mm -> rms 2.60
  * lips cart 80: common lip height +28 mm, per-lip deviation A -4.2, B -2.8, C +4.8, D +6.4, E +1.5 px (rms 4.37); + depth scale -> rms 2.50; + D/E shift +67 mm -> rms 3.14
* edge_lens_shelf_poses: sticker rows RMS px / best Z shift mm: 80:top 22.0/+75, 310:top 20.9/+71, 80:A 2.6/+6, 310:A 1.6/+2, 80:B 2.7/-10, 310:B 2.0/-0, 80:C 2.6/+7, 310:C 1.4/-7
  * lips cart 310: common lip height +0 mm, per-lip deviation A +0.2, B +2.8, C -0.7, D -0.0, E -4.6 px (rms 2.48); + depth scale -> rms 2.14; + D/E shift -39 mm -> rms 2.00
  * lips cart 80: common lip height +15 mm, per-lip deviation A +1.1, B -2.8, C +1.9, D +2.3, E -3.2 px (rms 2.35); + depth scale -> rms 2.29; + D/E shift -3 mm -> rms 2.34
* edge_lens_top_poses: sticker rows RMS px / best Z shift mm: 80:top 2.2/+0, 310:top 3.9/-0, 80:A 18.2/-71, 310:A 13.7/-49, 80:B 19.3/-101, 310:B 16.6/-4, 80:C 14.5/-98, 310:C 20.3/+45
  * lips cart 310: common lip height +48 mm, per-lip deviation A -14.1, B +0.9, C +6.2, D +12.0, E +11.3 px (rms 10.24); + depth scale -> rms 1.37; + D/E shift +180 mm -> rms 6.15
  * lips cart 80: common lip height -92 mm, per-lip deviation A +4.7, B -2.3, C +0.2, D -0.6, E -6.9 px (rms 4.20); + depth scale -> rms 2.32; + D/E shift -57 mm -> rms 3.13

## Diagnostic: which single dimension explains the mismatch

Script `work/41_cuboid_diag.py` (full table `work/cache/cuboid_diag.md`, plot `results/cuboid_diag.png`). DIAGNOSTIC ONLY: one drawing dimension freed at a time; the main result above keeps the drawing.

* main:k1_ppfix|lips|drawing: none; chi2_ref 223; corners 6.25 px/pt, exact edges 9.47 px, lips 2.54 px; f 1365, k1 -0.212, k2 0.000; back rail (not fitted) -16.6 px (-56 mm in Z)
* main:k1_ppfix|lips|dz_top: dz_top +54.3 +- 3.8; chi2_ref 117; corners 3.95 px/pt, exact edges 5.38 px, lips 2.30 px; f 1451, k1 -0.239, k2 0.000; back rail (not fitted) -3.0 px (-11 mm in Z)
* main:k1_ppfix|lips|dz_top@cart: dz_top@80 +54.5 +- 4.9, dz_top@310 +54.2 +- 4.6; chi2_ref 117; corners 3.95 px/pt, exact edges 5.39 px, lips 2.30 px; f 1451, k1 -0.239, k2 0.000; back rail (not fitted) -3.0 px (-11 mm in Z)
* main:k1_ppfix|lips|dz_A: dz_A -33.5 +- 3.8; chi2_ref 163; corners 5.02 px/pt, exact edges 8.68 px, lips 2.43 px; f 1363, k1 -0.211, k2 0.000; back rail (not fitted) -18.3 px (-61 mm in Z)
* main:k1_ppfix|lips|dz_E: dz_E -91.5 +- 11.7; chi2_ref 200; corners 6.55 px/pt, exact edges 9.66 px, lips 1.60 px; f 1360, k1 -0.210, k2 0.000; back rail (not fitted) -16.7 px (-56 mm in Z)
* main:k1_ppfix|lips|sz_shelves: sz_shelves +68.9 +- 16.0; chi2_ref 201; corners 5.67 px/pt, exact edges 6.74 px, lips 2.54 px; f 1454, k1 -0.239, k2 0.000; back rail (not fitted) -16.0 px (-57 mm in Z)
* main:k1_ppfix|lips|sy_depth: sy -6.3 +- 8.1; chi2_ref 223; corners 6.22 px/pt, exact edges 9.10 px, lips 2.55 px; f 1362, k1 -0.211, k2 0.000; back rail (not fitted) -17.2 px (-58 mm in Z)
* main:k1_ppfix|lips|code_mm: code_mm +87.2 +- 1.5; chi2_ref 220; corners 6.14 px/pt, exact edges 9.49 px, lips 2.55 px; f 1363, k1 -0.211, k2 0.000; back rail (not fitted) -16.6 px (-55 mm in Z)
* main:k1_ppfix|lips|dz_top+dz_E (2): dz_top +55.8 +- 3.8, dz_E -78.3 +- 10.6; chi2_ref 91; corners 4.01 px/pt, exact edges 5.43 px, lips 1.58 px; f 1449, k1 -0.238, k2 0.000; back rail (not fitted) -2.8 px (-10 mm in Z)
* main:k1_ppfix|lips|dz_A..E (5): dz_A -55.3 +- 4.0, dz_B -60.2 +- 7.1, dz_C -61.0 +- 10.9, dz_D -52.5 +- 16.8, dz_E -136.7 +- 21.5; chi2_ref 90; corners 3.97 px/pt, exact edges 5.24 px, lips 1.55 px; f 1457, k1 -0.240, k2 0.000; back rail (not fitted) -18.3 px (-65 mm in Z)
* k1k2_ppfix|lips|drawing: none; chi2_ref 223; corners 6.15 px/pt, exact edges 8.81 px, lips 2.55 px; f 1355, k1 -0.238, k2 0.017; back rail (not fitted) -16.4 px (-55 mm in Z)
* k1k2_ppfix|lips|dz_top: dz_top +65.0 +- 3.8; chi2_ref 100; corners 3.60 px/pt, exact edges 3.72 px, lips 2.24 px; f 1468, k1 -0.434, k2 0.291; back rail (not fitted) +1.5 px (+5 mm in Z)
* k1k2_ppfix|lips|dz_A..E (5): dz_A -66.0 +- 4.3, dz_B -69.3 +- 8.0, dz_C -70.7 +- 12.2, dz_D -58.6 +- 17.4, dz_E -138.9 +- 21.6; chi2_ref 75; corners 3.63 px/pt, exact edges 3.59 px, lips 1.48 px; f 1474, k1 -0.429, k2 0.285; back rail (not fitted) -16.6 px (-59 mm in Z)
* k1k2_ppfree|lips|drawing: none; chi2_ref 223; corners 5.93 px/pt, exact edges 8.06 px, lips 2.44 px; f 1365, k1 -0.435, k2 0.398, pp (990, 301); back rail (not fitted) -14.0 px (-46 mm in Z)
* k1k2_ppfree|lips|dz_top: dz_top +67.1 +- 3.5; chi2_ref 98; corners 3.10 px/pt, exact edges 4.41 px, lips 2.24 px; f 1442, k1 -0.440, k2 0.314, pp (978, 399); back rail (not fitted) +1.5 px (+5 mm in Z)
* k1k2_ppfree|lips|dz_A..E (5): dz_A -72.7 +- 4.6, dz_B -85.4 +- 9.5, dz_C -92.9 +- 14.7, dz_D -86.3 +- 19.8, dz_E -174.7 +- 24.9; chi2_ref 69; corners 3.04 px/pt, exact edges 3.93 px, lips 1.46 px; f 1476, k1 -0.490, k2 0.411, pp (977, 396); back rail (not fitted) -16.9 px (-60 mm in Z)
* k1k2_ppfree|nolips|dz_top: dz_top +67.8 +- 3.8; chi2_ref 55; corners 3.08 px/pt, exact edges 4.37 px; f 1438, k1 -0.449, k2 0.331, pp (971, 396); back rail (not fitted) +1.7 px (+6 mm in Z)

## Conclusions (determined / consistent / not determinable)

1. Main cuboid fit (drawing exact, k1_ppfix: fx=fy, pp fixed at image centre, k1 only): f = 1359 +- 36 px (cluster bootstrap; over 14 model x weighting variants 1301..1395), k1 = -0.210 +- 0.011; corners 6.09 px per point (max 10.9), sticker sides 2.13 px, exact edges 8.79 px (corner RMS ~47x the per-frame corner scatter). The exact edges add almost nothing (3 short edges); free-pp variants are unstable (cx 566..959) because pp is the knob that trades the exact edges against the stickers. The fit uses a fold barrier (fold margin >= 0.03); the main solution has fold margin 0.030 i.e. it SITS ON THE BARRIER: the drawing geometry pushes the distortion towards a lens that folds inside the image, so k1, k2 are set by the validity constraint and not by the data (without the barrier: f 1341, k1 -0.231, fold margin -0.020 = folded). The unconstrained distortion contradicts the straight-edge (plumb-line) distortion, as for the marker-only fit: the drawing geometry biases the estimate - only CONSISTENT with the data at the 5-8 px level.
2. The three exact edges are the edges they claim to be (crops: top-face outer edge of the X=0 board of cart 310 against the floor; top-face front edges of both cart-80 boards, a 1-2 px dark front face beside them) and agree with their own board's stickers to <= 4.6 mm, but every global drawing camera misses them by board0_outer -10.4 px (-22 mm in X), x_board0_front -5.1 px (+8 mm in Y), x_board1600_front -3.8 px (+7 mm in Y) - they inherit the sticker inconsistency.
3. Shelf lips vs prediction (joint camera): -1.9..+8.7 px (camera 1-sigma 1.7..4.9 px); the lips are sensitive mainly to Y (0.37..0.60 px/mm) and only weakly to Z (0.06..0.18 px/mm), so a single lip fixes its height only to ~+-15..40 mm. With one common lip height per cart the five lips agree only to 2.5..4.3 px rms (markers-only, joint and shelf-anchored cameras): lip outlines / fronts differ by a few mm between shelves.
4. The clearest misfit is at the TOP level: the cart-80 back top rail lies -19.3 px (markers-only) / -17.0 px (joint) from the drawing top-back edge (= -56 mm in Z), but only +3.2 px when the camera is anchored to the shelf stickers (edge-only lens) - which in turn puts the top stickers +75 / +71 mm (cart 80 / 310) above the drawing and the top-board edges 11..23 px off.
5. Single dimension: the top-plate surface (end boards with the top stickers) +54 +- 4 mm above the drawing relative to shelves and frame (per cart +55 / +54 mm; pp free +67 mm; k1k2 pp fixed +65 mm; range over the lens models +54..+67 mm - the k1-only lens sits on the fold barrier and gives the low end; formal +- = Gauss-Newton with n_eff dof) explains most of it: chi2_ref 223 -> 117, corners 6.2 -> 4.0 px, exact edges 9.5 -> 5.4 px, and the NOT fitted back rail moves to -3.0 px (-11 mm). No other single dimension comes close (chi2_ref dz_A 163, dz_B 216, dz_C 218, dz_D 206, dz_E 200, sz_shelves 201, sx_width 222, sy_depth 223, code_mm 220). Stickers alone cannot tell this from 'all shelves ~65 mm lower'; the back top rail can, IF it really is at the top-plate level (its height is not in the drawing): with all shelf rows lowered instead (dz_A..E) the rail stays off (main:k1_ppfix/drawing: -16.6 px / -56 mm, main:k1_ppfix/dz_top: -3.0 px / -11 mm, main:k1_ppfix/dz_A..E (5): -18.3 px / -65 mm, k1k2_ppfree/drawing: -14.0 px / -46 mm, k1k2_ppfree/dz_top: +1.5 px / +5 mm, k1k2_ppfree/dz_A..E (5): -16.9 px / -60 mm, k1k2_ppfix/drawing: -16.4 px / -55 mm, k1k2_ppfix/dz_top: +1.5 px / +5 mm, k1k2_ppfix/dz_A..E (5): -16.6 px / -59 mm), so it groups with the shelves/frame and puts the offset on the end boards with the top stickers. Caveat: the rail edge was first read as a wall conduit by the scene tracing and re-assigned to cart 80 in the review. Observation only - the main result keeps the drawing.
6. Second order: shelf E's lip lies below its drawing height relative to A-D: dz_E -78 +- 11 mm on top of dz_top (lips 2.30 -> 1.58 px), in both carts; E has no sticker, so this rests on the identification of the lowest traced lip.
7. With dz_top freed (diagnostic) the lens moves to f = 1451 (pp fixed) / 1442 with pp (978, 399), k1 -0.44, k2 0.31 (fold margin 296.98; main-lens dz_top fit: fold margin 0.030 = on the barrier) - the drawing-geometry f (1359) is pulled down by ~92 px (pp fixed) by the top-plate mismatch; this bias is NOT contained in the main-result uncertainty.
