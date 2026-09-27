# Method: cuboid cart model (stickers + exact edges, drawing dimensions exact)

Script `work/41_cuboid_fit.py` (helpers `41_cuboid_lib.py`). Data: 62 sticker corners, 13 sticker sides, 3 exact structure edges (cart310_board0_outer, cart80_x_board0_front, cart80_x_board1600_front). Edge files reviewed: {310: True, 80: True}.

Main model (min. leave-one-sticker-out error, cluster weighting): **k1k2_ppfix**

| param | value | bootstrap 1-sigma | cov 1-sigma |
|---|---|---|---|
| f | 1326.6 | 45.7 | 8.93 |
| k1 | -0.11842 | 0.152 | 0.0329 |
| k2 | -0.17727 | 0.282 | 0.0522 |

Block RMS: corners 5.66 px per point (max 10.4), sticker sides 2.49 px, exact edges 7.64 px. LOSO 8.13 px.
Mapping uncertainty (rot.-comp., bootstrap): centre 49.40 px, cart_band 51.14 px, corners 543.68 px, whole_image 52.21 px

| model / weighting | f | pp | dist | corners px/pt | sides | edges | LOSO |
|---|---|---|---|---|---|---|---|
| k1_ppfix|cluster | 1341 | centre | k1=-0.231 | 5.89 | 2.30 | 7.80 | 8.21 |
| k1k2_ppfix|cluster | 1327 | centre | k1=-0.118, k2=-0.177 | 5.66 | 2.49 | 7.64 | 8.13 |
| k1_ppfree|cluster | 1353 | (885, 452) | k1=-0.222 | 5.75 | 2.28 | 6.90 | 8.54 |
| k1k2_ppfree|cluster | 1355 | (749, 423) | k1=-0.096, k2=-0.126 | 6.21 | 2.44 | 3.02 | 8.47 |
| k1k2k3_ppfree|cluster | 1338 | (675, 437) | k1=0.053, k2=-0.434, k3=0.233 | 6.68 | 2.34 | 1.42 | 9.11 |
| k1k2_ppfree_fxfy|cluster | 1337 | (835, 438) | k1=-0.115, k2=-0.123 | 5.83 | 2.44 | 4.33 | 8.59 |
| k1k2p1p2_ppfree|cluster | 1363 | (955, 365) | k1=-0.448, k2=0.416, p1=0.011, p2=-0.003 | 5.33 | 2.28 | 6.01 | 10.40 |
| k1_ppfix|point | 1294 | centre | k1=-0.226 | 8.69 | 3.70 | 0.83 | nan |
| k1k2_ppfix|point | 1308 | centre | k1=-0.319, k2=0.133 | 8.88 | 3.15 | 0.95 | nan |
| k1_ppfree|point | 1345 | (632, 365) | k1=-0.159 | 8.55 | 2.33 | 1.03 | nan |
| k1k2_ppfree|point | 1376 | (548, 400) | k1=0.221, k2=-0.363 | 9.93 | 1.79 | 0.44 | nan |
| k1k2k3_ppfree|point | 1386 | (539, 358) | k1=-0.097, k2=0.383, k3=-0.492 | 10.24 | 1.83 | 0.34 | nan |
| k1k2_ppfree_fxfy|point | 1347 | (529, 401) | k1=0.195, k2=-0.301 | 10.39 | 1.70 | 0.43 | nan |
| k1k2p1p2_ppfree|point | 1348 | (774, 432) | k1=0.288, k2=-1.057, p1=-0.006, p2=0.061 | 10.35 | 1.74 | 0.25 | nan |

Exact-edge check (implied shift of the traced edge from its drawing line with the fitted camera, mm):
* cart310_board0_outer: joint -18.8 mm along X (-8.85 px); markers-only -20.5 mm (-9.61 px)
* cart80_x_board0_front: joint +6.5 mm along Y (-4.41 px); markers-only +7.8 mm (-5.32 px)
* cart80_x_board1600_front: joint +6.4 mm along Y (-3.49 px); markers-only +7.7 mm (-4.25 px)

## Prediction check: modelled cart lines vs traced edges

Script `work/41_cuboid_predict.py`; full table `work/cache/cuboid_offsets.md/.json`; plots `results/cuboid_overlay.png`, `results/cuboid_crops.png`, `results/cuboid_crops_anchor.png`, `results/cuboid_offsets.png`.
Offset traced - predicted [px] (+ = traced edge on the + side of the axis in [..]) +- camera 1-sigma (bootstrap, robust) / implied shift [mm]:

| edge | markers-only | joint | edge lens + shelf-sticker poses (diag.) |
|---|---|---|---|
| cart310_board0_outer | -9.6 +-3.5 / dX -21 | -8.9 +-4.3 / dX -19 | -23.3 / dX -49 |
| cart310_board1600_outer | +7.5 +-2.8 / dX +12 | +7.7 +-3.2 / dX +12 | +10.8 / dX +17 |
| cart310_x_board0_front | -1.1 +-1.7 / dY -2 | +1.2 +-4.1 / dY +2 | +5.6 / dY +10 |
| cart310_x_board1600_front | +4.3 +-3.8 / dY +6 | +4.8 +-3.8 / dY +7 | +13.1 / dY +20 |
| cart310_x_A_front_fall | -2.1 +-2.6 / dZ -13 | -1.5 +-3.2 / dZ -10 | +0.3 / dZ +2 |
| cart310_x_B_front_fall | +4.2 +-1.6 / dZ +36 | +4.2 +-2.1 / dZ +36 | +3.1 / dZ +28 |
| cart310_x_B_front_rise | +1.5 +-1.6 / dZ +12 | +1.5 +-2.1 / dZ +13 | +0.4 / dZ +3 |
| cart310_x_C_front_fall | +1.9 +-1.8 / dZ +22 | +1.9 +-2.4 / dZ +22 | -0.8 / dZ -9 |
| cart310_x_C_front_rise | +0.3 +-1.8 / dZ +4 | +0.0 +-2.7 / dZ +0 | -2.7 / dZ -32 |
| cart310_x_D_front_fall | +3.7 +-2.7 / dZ +51 | +3.2 +-3.4 / dZ +44 | -0.5 / dZ -7 |
| cart310_x_E_front_fall | -1.6 +-3.4 / dZ -27 | -1.8 +-4.0 / dZ -30 | -5.6 / dZ -97 |
| cart310_x_B_inner_fall | +13.9 +-1.7 / dZ +121 | +14.1 +-2.2 / dZ +123 | +12.5 / dZ +114 |
| cart310_x_back_inner | -79.2 +-2.5 / dZ -254 | -77.6 +-3.6 / dZ -253 | -63.0 / dZ -227 |
| cart310_z_post_front_left_right | -2.9 +-2.5 / dX -16 | -1.2 +-2.6 / dX -6 | -2.6 / dX -13 |
| cart310_z_post_front_right | -4.5 +-1.5 / dX -10 | -5.3 +-2.0 / dX -12 | -1.1 / dX -3 |
| cart80_x_board0_front | +5.3 +-3.7 / dY +8 | +4.4 +-3.6 / dY +7 | +13.9 / dY +22 |
| cart80_x_board1600_front | +4.2 +-2.9 / dY +8 | +3.5 +-2.6 / dY +6 | +13.0 / dY +24 |
| cart80_y_board0_outer | -9.1 +-1.3 / dX -15 | -10.2 +-2.3 / dX -16 | -9.0 / dX -14 |
| cart80_y_board1600_outer | +8.0 +-2.7 / dX +18 | +7.8 +-3.6 / dX +18 | +22.6 / dX +52 |
| cart80_x_back_rail_out | -19.3 +-3.1 / dZ -63 | -18.7 +-3.1 / dZ -61 | -1.8 / dZ -7 |
| cart80_x_back_rail_in | -31.2 +-3.1 / dZ -102 | -30.5 +-3.1 / dZ -101 | -14.3 / dZ -53 |
| cart80_x_back_low | -78.8 +-3.2 / dZ -254 | -78.0 +-3.1 / dZ -253 | -60.9 / dZ -218 |
| cart80_x_A_front_out | +2.4 +-3.3 / dZ +14 | +1.5 +-3.7 / dZ +8 | +3.0 / dZ +18 |
| cart80_x_A_front_in | +9.8 +-3.7 / dZ +57 | +8.9 +-4.2 / dZ +51 | +9.5 / dZ +59 |
| cart80_x_B_front | +1.8 +-2.5 / dZ +14 | +1.4 +-2.7 / dZ +11 | -0.9 / dZ -8 |
| cart80_x_C_front | +7.6 +-2.9 / dZ +80 | +7.9 +-3.4 / dZ +82 | +3.1 / dZ +34 |
| cart80_x_D_front | +8.0 +-3.6 / dZ +102 | +8.7 +-3.9 / dZ +110 | +3.0 / dZ +39 |
| cart80_x_E_front | +2.0 +-4.2 / dZ +31 | +3.2 +-4.4 / dZ +49 | -3.2 / dZ -49 |
| cart80_x_E_front_in | +4.8 +-4.2 / dZ +74 | +6.0 +-4.3 / dZ +91 | -0.4 / dZ -5 |
| cart80_z_post1600_sil | +1.7 +-2.5 / dX +11 | -0.0 +-3.2 / dX -0 | +2.0 / dX +12 |
| cart80_z_post1600_sil_low | -3.9 +-1.1 / dX -20 | -4.4 +-1.3 / dX -22 | -5.4 / dX -27 |

* markers_only: sticker rows RMS px / best Z shift mm: 80:top 5.0/+7, 310:top 5.8/+5, 80:A 7.2/-24, 310:A 8.6/-33, 80:B 3.5/-10, 310:B 3.5/-7, 80:C 6.0/+32, 310:C 3.6/+17
  * lips cart 310: common lip height +8 mm, per-lip deviation A -3.5, B +3.2, C +1.1, D +3.1, E -2.1 px (rms 2.98); + depth scale -> rms 2.63; + D/E shift +13 mm -> rms 2.92
  * lips cart 80: common lip height +32 mm, per-lip deviation A -3.3, B -2.3, C +4.6, D +5.5, E -0.1 px (rms 3.74); + depth scale -> rms 2.65; + D/E shift +48 mm -> rms 3.09
* joint: sticker rows RMS px / best Z shift mm: 80:top 4.7/+6, 310:top 5.9/+7, 80:A 8.0/-28, 310:A 8.7/-33, 80:B 3.9/-15, 310:B 4.4/-12, 80:C 5.8/+26, 310:C 2.4/+9
  * lips cart 310: common lip height +9 mm, per-lip deviation A -3.0, B +3.2, C +1.1, D +2.5, E -2.4 px (rms 2.79); + depth scale -> rms 2.59; + D/E shift +5 mm -> rms 2.78
  * lips cart 80: common lip height +31 mm, per-lip deviation A -4.1, B -2.6, C +4.9, D +6.2, E +1.1 px (rms 4.27); + depth scale -> rms 2.55; + D/E shift +63 mm -> rms 3.19
* edge_lens_shelf_poses: sticker rows RMS px / best Z shift mm: 80:top 18.4/+60, 310:top 17.1/+55, 80:A 2.1/+1, 310:A 1.7/-4, 80:B 2.4/-6, 310:B 2.0/+4, 80:C 3.0/+16, 310:C 2.6/+7
  * lips cart 310: common lip height -1 mm, per-lip deviation A +0.4, B +3.2, C -0.7, D -0.4, E -5.6 px (rms 2.93); + depth scale -> rms 2.47; + D/E shift -51 mm -> rms 2.24
  * lips cart 80: common lip height +12 mm, per-lip deviation A +1.1, B -2.3, C +2.1, D +2.1, E -3.9 px (rms 2.51); + depth scale -> rms 2.40; + D/E shift -11 mm -> rms 2.44
* edge_lens_top_poses: sticker rows RMS px / best Z shift mm: 80:top 2.3/-0, 310:top 3.7/-0, 80:A 18.4/-71, 310:A 13.1/-50, 80:B 21.5/-103, 310:B 7.6/-11, 80:C 21.2/-110, 310:C 8.7/+25
  * lips cart 310: common lip height +28 mm, per-lip deviation A -10.9, B +1.8, C +5.1, D +9.0, E +7.3 px (rms 7.74); + depth scale -> rms 1.82; + D/E shift +132 mm -> rms 5.15
  * lips cart 80: common lip height -99 mm, per-lip deviation A +5.5, B -1.9, C -0.2, D -1.5, E -8.3 px (rms 5.03); + depth scale -> rms 2.45; + D/E shift -77 mm -> rms 3.45
