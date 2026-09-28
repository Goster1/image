# Cuboid model: predicted (drawing) vs traced cart edges

Script `work/41_cuboid_predict.py`. Offset = traced - predicted along the image normal [px], sign + = traced edge on the + side of
the first axis in [..]; implied shift = move of the drawing line (camera fixed) that puts it onto the traced edge [mm];
cam-sigma = 1-sigma of the mean offset over the bootstrap cameras. Cameras: markers_only = markers-only (fx=fy, pp fixed, dist k1+k2); joint = joint (k1_ppfix: fx=fy, pp fixed at image centre, k1 only); edge_lens = edge-only lens (method_lines_joint) + poses from all stickers; edge_lens_shelf_poses = edge-only lens (method_lines_joint) + poses from stickers of rows A+B+C only; edge_lens_top_poses = edge-only lens (method_lines_joint) + poses from stickers of rows top only

## camera: markers_only

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -9.61 (-13.51 / -5.39) | 4.25 | dX -20.5 (0.470) | dZ +29.0 |
| cart310_board1600_outer | top_x1600 [X] | +7.52 (+11.02 / +5.24) | 3.97 | dX +11.7 (0.637) | dZ +74.6 |
| cart310_x_board0_front | top_front [Y] | -1.08 (-4.38 / +1.96) | 2.63 | dY -1.9 (0.565) | dZ -6.5 |
| cart310_x_board1600_front | top_front [Y] | +4.35 (+3.97 / +5.37) | 3.78 | dY +6.3 (0.692) | dZ +20.9 |
| cart310_x_A_front_fall | shelfA_front [Z] | -2.12 (-4.02 / -1.48) | 2.68 | dZ -13.1 (0.164) | dY -3.6 |
| cart310_x_B_front_fall | shelfB_front [Z] | +4.18 (+2.72 / +4.57) | 1.67 | dZ +35.7 (0.116) | dY +8.1 |
| cart310_x_B_front_rise | shelfB_front [Z] | +1.47 (+0.06 / +2.06) | 1.67 | dZ +12.4 (0.117) | dY +2.8 |
| cart310_x_C_front_fall | shelfC_front [Z] | +1.85 (+3.42 / +0.35) | 2.42 | dZ +22.1 (0.086) | dY +4.3 |
| cart310_x_C_front_rise | shelfC_front [Z] | +0.34 (+1.14 / -1.38) | 2.39 | dZ +4.2 (0.088) | dY +0.8 |
| cart310_x_D_front_fall | shelfD_front [Z] | +3.67 (+3.89 / +3.08) | 3.22 | dZ +50.6 (0.073) | dY +8.9 |
| cart310_x_E_front_fall | shelfE_front [Z] | -1.62 (-1.16 / -2.76) | 4.01 | dZ -26.7 (0.060) | dY -4.3 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +13.91 (+13.79 / +13.95) | 1.88 | dZ +121.0 (0.115) | dY +27.5 |
| cart310_x_back_inner | top_back [Z] | -79.20 (-82.96 / -75.55) | 2.75 | dZ -254.2 (0.312) | dY -137.9 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | -2.95 (-1.29 / -4.29) | 2.68 | dX -16.4 (0.170) | dY -7.0 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -4.48 (-2.46 / -6.53) | 2.82 | dX -10.3 (0.446) | dY +19.8 |
| cart80_x_board0_front | top_front [Y] | +5.32 (+7.25 / +3.58) | 4.50 | dY +7.8 (0.679) | dZ +23.7 |
| cart80_x_board1600_front | top_front [Y] | +4.25 (+3.76 / +5.00) | 3.08 | dY +7.7 (0.549) | dZ +23.4 |
| cart80_y_board0_outer | top_x0 [X] | -9.13 (-6.89 / -12.43) | 10.19 | dX -14.5 (0.622) | dZ +175.1 |
| cart80_y_board1600_outer | top_x1600 [X] | +7.96 (+10.22 / +5.90) | 3.66 | dX +18.1 (0.440) | dZ +23.3 |
| cart80_x_back_rail_out | top_back [Z] | -19.35 (-23.92 / -14.87) | 3.46 | dZ -63.0 (0.308) | dY -36.1 |
| cart80_x_back_rail_in | top_back [Z] | -31.21 (-36.79 / -27.68) | 3.42 | dZ -102.0 (0.306) | dY -58.4 |
| cart80_x_back_low | top_back [Z] | -78.76 (-82.98 / -72.36) | 3.65 | dZ -254.0 (0.310) | dY -145.4 |
| cart80_x_A_front_out | shelfA_front [Z] | +2.38 (+0.43 / -0.38) | 3.57 | dZ +13.6 (0.175) | dY +4.1 |
| cart80_x_A_front_in | shelfA_front [Z] | +9.81 (+9.38 / +7.83) | 3.88 | dZ +57.4 (0.171) | dY +17.2 |
| cart80_x_B_front | shelfB_front [Z] | +1.82 (-0.02 / +1.68) | 2.87 | dZ +14.2 (0.126) | dY +3.5 |
| cart80_x_C_front | shelfC_front [Z] | +7.63 (+7.63 / +6.10) | 3.76 | dZ +80.2 (0.095) | dY +17.3 |
| cart80_x_D_front | shelfD_front [Z] | +7.98 (+7.31 / +6.82) | 4.64 | dZ +102.0 (0.078) | dY +19.8 |
| cart80_x_E_front | shelfE_front [Z] | +1.98 (+2.40 / -0.62) | 5.45 | dZ +30.9 (0.065) | dY +5.5 |
| cart80_x_E_front_in | shelfE_front [Z] | +4.75 (+5.04 / +2.17) | 5.46 | dZ +73.5 (0.065) | dY +13.0 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | +1.67 (+2.15 / +1.44) | 4.15 | dX +10.7 (0.154) | dY -4.5 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -3.86 (-3.84 / -3.92) | 1.78 | dX -20.2 (0.191) | dY +8.6 |

* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): 80:top 5.0/+7, 310:top 5.8/+5, 80:A 7.2/-24, 310:A 8.6/-33, 80:B 3.5/-10, 310:B 3.5/-7, 80:C 6.0/+32, 310:C 3.6/+17
* lips cart 310: common lip height dZ +8.3 mm (formal +-6.2); per-lip deviation from it [px]: A -3.47, B +3.22, C +1.14, D +3.07, E -2.11 (rms 2.98 px); + shelf-depth scale -25.8 +- 12.6 permille -> rms 2.63 px; + D/E shift +13.4 +- 17.3 mm -> rms 2.92 px
* lips cart 80: common lip height dZ +32.2 mm (formal +-7.5); per-lip deviation from it [px]: A -3.25, B -2.25, C +4.56, D +5.46, E -0.12 (rms 3.74 px); + shelf-depth scale -50.9 +- 12.1 permille -> rms 2.65 px; + D/E shift +48.4 +- 18.0 mm -> rms 3.09 px

## camera: joint

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -10.38 (-14.39 / -5.98) | 5.15 | dX -22.0 (0.473) | dZ +32.3 |
| cart310_board1600_outer | top_x1600 [X] | +6.19 (+6.47 / +5.94) | 2.56 | dX +9.2 (0.676) | dZ +61.1 |
| cart310_x_board0_front | top_front [Y] | +2.23 (-1.46 / +5.66) | 3.99 | dY +3.9 (0.565) | dZ +13.3 |
| cart310_x_board1600_front | top_front [Y] | +5.04 (+4.29 / +6.40) | 3.96 | dY +7.3 (0.687) | dZ +25.0 |
| cart310_x_A_front_fall | shelfA_front [Z] | -1.82 (-3.65 / -0.92) | 2.34 | dZ -11.5 (0.160) | dY -3.1 |
| cart310_x_B_front_fall | shelfB_front [Z] | +3.73 (+1.95 / +4.74) | 3.87 | dZ +32.0 (0.116) | dY +7.2 |
| cart310_x_B_front_rise | shelfB_front [Z] | +0.98 (-0.68 / +2.10) | 3.88 | dZ +8.2 (0.116) | dY +1.8 |
| cart310_x_C_front_fall | shelfC_front [Z] | +1.44 (+2.27 / +0.62) | 6.13 | dZ +17.0 (0.086) | dY +3.3 |
| cart310_x_C_front_rise | shelfC_front [Z] | -0.39 (-0.02 / -1.46) | 6.24 | dZ -4.3 (0.088) | dY -0.8 |
| cart310_x_D_front_fall | shelfD_front [Z] | +2.86 (+2.45 / +3.13) | 7.80 | dZ +39.1 (0.073) | dY +6.9 |
| cart310_x_E_front_fall | shelfE_front [Z] | -1.90 (-2.47 / -1.80) | 8.99 | dZ -31.6 (0.060) | dY -5.1 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +13.43 (+13.17 / +13.60) | 3.82 | dZ +117.4 (0.114) | dY +26.4 |
| cart310_x_back_inner | top_back [Z] | -74.92 (-78.60 / -71.49) | 3.53 | dZ -249.4 (0.300) | dY -131.5 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | -0.66 (+1.38 / -2.28) | 7.02 | dX -2.9 (0.175) | dY -1.2 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -5.12 (-4.05 / -6.32) | 3.73 | dX -11.6 (0.444) | dY +22.8 |
| cart80_x_board0_front | top_front [Y] | +5.09 (+6.83 / +3.39) | 3.72 | dY +7.6 (0.670) | dZ +22.8 |
| cart80_x_board1600_front | top_front [Y] | +3.82 (+3.07 / +4.82) | 2.78 | dY +7.0 (0.544) | dZ +21.0 |
| cart80_y_board0_outer | top_x0 [X] | -7.66 (-8.92 / -6.62) | 3.20 | dX -11.5 (0.668) | dZ +157.2 |
| cart80_y_board1600_outer | top_x1600 [X] | +9.14 (+10.83 / +7.24) | 4.42 | dX +20.7 (0.442) | dZ +27.4 |
| cart80_x_back_rail_out | top_back [Z] | -16.97 (-24.34 / -12.56) | 3.33 | dZ -55.7 (0.307) | dY -31.6 |
| cart80_x_back_rail_in | top_back [Z] | -28.58 (-35.56 / -25.38) | 3.21 | dZ -94.5 (0.303) | dY -53.6 |
| cart80_x_back_low | top_back [Z] | -76.35 (-83.47 / -69.82) | 3.43 | dZ -247.0 (0.309) | dY -140.0 |
| cart80_x_A_front_out | shelfA_front [Z] | +0.79 (-0.01 / -1.83) | 1.86 | dZ +4.6 (0.175) | dY +1.4 |
| cart80_x_A_front_in | shelfA_front [Z] | +7.83 (+7.87 / +6.08) | 1.75 | dZ +45.7 (0.172) | dY +13.8 |
| cart80_x_B_front | shelfB_front [Z] | +0.82 (-0.12 / +0.30) | 2.12 | dZ +6.3 (0.128) | dY +1.6 |
| cart80_x_C_front | shelfC_front [Z] | +7.56 (+8.23 / +5.65) | 3.41 | dZ +77.6 (0.098) | dY +17.1 |
| cart80_x_D_front | shelfD_front [Z] | +8.69 (+8.71 / +7.22) | 4.32 | dZ +107.8 (0.081) | dY +21.5 |
| cart80_x_E_front | shelfE_front [Z] | +3.41 (+4.36 / +0.66) | 5.11 | dZ +51.1 (0.067) | dY +9.3 |
| cart80_x_E_front_in | shelfE_front [Z] | +6.21 (+7.01 / +3.46) | 5.12 | dZ +92.5 (0.068) | dY +16.8 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | -0.19 (-0.17 / -0.06) | 4.38 | dX -1.2 (0.161) | dY +0.5 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -4.48 (-4.74 / -4.26) | 1.90 | dX -22.6 (0.197) | dY +10.0 |

* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): 80:top 6.2/+7, 310:top 6.5/+9, 80:A 8.0/-27, 310:A 8.3/-32, 80:B 4.0/-18, 310:B 4.3/-13, 80:C 5.6/+22, 310:C 2.0/+5
* lips cart 310: common lip height dZ +6.3 mm (formal +-5.5); per-lip deviation from it [px]: A -2.83, B +3.00, C +0.90, D +2.40, E -2.28 (rms 2.61 px); + shelf-depth scale -17.1 +- 11.6 permille -> rms 2.43 px; + D/E shift +4.9 +- 15.2 mm -> rms 2.60 px
* lips cart 80: common lip height dZ +28.3 mm (formal +-8.6); per-lip deviation from it [px]: A -4.18, B -2.82, C +4.79, D +6.40, E +1.50 (rms 4.37 px); + shelf-depth scale -66.3 +- 11.3 permille -> rms 2.50 px; + D/E shift +67.1 +- 18.0 mm -> rms 3.14 px

## camera: edge_lens

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -9.47 (-11.42 / -7.72) | nan | dX -19.8 (0.479) | dZ +31.3 |
| cart310_board1600_outer | top_x1600 [X] | +11.63 (+13.61 / +9.63) | nan | dX +18.3 (0.636) | dZ +125.0 |
| cart310_x_board0_front | top_front [Y] | -1.22 (-5.52 / +2.85) | nan | dY -2.2 (0.571) | dZ -7.1 |
| cart310_x_board1600_front | top_front [Y] | +1.08 (+0.62 / +2.13) | nan | dY +1.6 (0.658) | dZ +5.4 |
| cart310_x_A_front_fall | shelfA_front [Z] | -7.09 (-8.24 / -5.82) | nan | dZ -42.6 (0.167) | dY -11.9 |
| cart310_x_B_front_fall | shelfB_front [Z] | +1.30 (-0.83 / +3.37) | nan | dZ +10.3 (0.123) | dY +2.4 |
| cart310_x_B_front_rise | shelfB_front [Z] | -1.52 (-3.48 / +0.54) | nan | dZ -12.5 (0.123) | dY -3.0 |
| cart310_x_C_front_fall | shelfC_front [Z] | +2.22 (+1.76 / +2.57) | nan | dZ +23.7 (0.093) | dY +4.9 |
| cart310_x_C_front_rise | shelfC_front [Z] | -0.19 (-0.54 / -0.07) | nan | dZ -2.1 (0.095) | dY -0.4 |
| cart310_x_D_front_fall | shelfD_front [Z] | +4.82 (+3.46 / +6.64) | nan | dZ +60.7 (0.079) | dY +11.4 |
| cart310_x_E_front_fall | shelfE_front [Z] | +2.54 (+0.21 / +4.80) | nan | dZ +37.8 (0.066) | dY +6.5 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +10.97 (+10.35 / +11.43) | nan | dZ +89.6 (0.122) | dY +21.2 |
| cart310_x_back_inner | top_back [Z] | -73.88 (-75.11 / -72.41) | nan | dZ -259.3 (0.285) | dY -136.0 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | +5.79 (+3.63 / +7.81) | nan | dX +28.1 (0.198) | dY +13.6 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -7.91 (-9.35 / -6.61) | nan | dX -17.8 (0.437) | dY +37.3 |
| cart80_x_board0_front | top_front [Y] | +1.53 (+4.09 / -1.10) | nan | dY +2.4 (0.642) | dZ +7.0 |
| cart80_x_board1600_front | top_front [Y] | +3.41 (+1.71 / +5.47) | nan | dY +6.3 (0.537) | dZ +18.6 |
| cart80_y_board0_outer | top_x0 [X] | -12.98 (-12.79 / -13.11) | nan | dX -20.4 (0.636) | dZ +356.7 |
| cart80_y_board1600_outer | top_x1600 [X] | +12.21 (+10.59 / +14.04) | nan | dX +28.0 (0.434) | dZ +38.7 |
| cart80_x_back_rail_out | top_back [Z] | -11.85 (-14.19 / -9.03) | nan | dZ -42.0 (0.283) | dY -23.6 |
| cart80_x_back_rail_in | top_back [Z] | -24.21 (-26.77 / -21.82) | nan | dZ -86.4 (0.280) | dY -48.5 |
| cart80_x_back_low | top_back [Z] | -71.05 (-73.27 / -66.83) | nan | dZ -248.9 (0.286) | dY -139.6 |
| cart80_x_A_front_out | shelfA_front [Z] | -5.01 (-5.12 / -5.19) | nan | dZ -28.1 (0.178) | dY -8.7 |
| cart80_x_A_front_in | shelfA_front [Z] | +1.90 (+1.24 / +1.80) | nan | dZ +10.8 (0.175) | dY +3.4 |
| cart80_x_B_front | shelfB_front [Z] | -3.53 (-4.30 / -2.87) | nan | dZ -26.6 (0.133) | dY -7.0 |
| cart80_x_C_front | shelfC_front [Z] | +5.11 (+5.66 / +4.43) | nan | dZ +49.5 (0.103) | dY +11.4 |
| cart80_x_D_front | shelfD_front [Z] | +7.96 (+7.72 / +7.86) | nan | dZ +92.3 (0.086) | dY +19.2 |
| cart80_x_E_front | shelfE_front [Z] | +4.40 (+4.62 / +3.36) | nan | dZ +60.6 (0.073) | dY +11.6 |
| cart80_x_E_front_in | shelfE_front [Z] | +7.19 (+7.28 / +6.18) | nan | dZ +98.9 (0.073) | dY +18.9 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | -3.79 (-4.40 / -3.18) | nan | dX -21.4 (0.176) | dY +10.1 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -6.43 (-7.06 / -5.88) | nan | dX -30.1 (0.213) | dY +14.1 |

* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): 80:top 7.1/+20, 310:top 6.0/+12, 80:A 10.7/-39, 310:A 12.3/-51, 80:B 9.0/-38, 310:B 8.0/-35, 80:C 7.9/+1, 310:C 6.4/-20
* lips cart 310: common lip height dZ -4.4 mm (formal +-8.8); per-lip deviation from it [px]: A -6.36, B +1.84, C +2.63, D +5.17, E +2.83 (rms 4.40 px); + shelf-depth scale -74.4 +- 8.6 permille -> rms 1.91 px; + D/E shift +67.2 +- 18.1 mm -> rms 3.31 px
* lips cart 80: common lip height dZ +2.3 mm (formal +-10.3); per-lip deviation from it [px]: A -5.42, B -3.84, C +4.87, D +7.76, E +4.23 (rms 5.48 px); + shelf-depth scale -86.7 +- 9.9 permille -> rms 2.24 px; + D/E shift +93.0 +- 17.2 mm -> rms 3.13 px

## camera: edge_lens_shelf_poses

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -26.21 (-29.32 / -23.25) | nan | dX -55.0 (0.477) | dZ +91.2 |
| cart310_board1600_outer | top_x1600 [X] | +16.85 (+18.24 / +15.42) | nan | dX +27.3 (0.617) | dZ +184.4 |
| cart310_x_board0_front | top_front [Y] | +6.10 (+2.11 / +9.85) | nan | dY +10.8 (0.561) | dZ +39.0 |
| cart310_x_board1600_front | top_front [Y] | +14.30 (+13.95 / +15.22) | nan | dY +22.2 (0.643) | dZ +80.0 |
| cart310_x_A_front_fall | shelfA_front [Z] | +0.27 (+0.57 / +0.06) | nan | dZ +1.8 (0.149) | dY +0.5 |
| cart310_x_B_front_fall | shelfB_front [Z] | +2.80 (+1.81 / +3.67) | nan | dZ +25.1 (0.111) | dY +5.5 |
| cart310_x_B_front_rise | shelfB_front [Z] | +0.05 (-0.87 / +0.95) | nan | dZ +0.3 (0.111) | dY +0.1 |
| cart310_x_C_front_fall | shelfC_front [Z] | -0.70 (-0.01 / -1.08) | nan | dZ -8.2 (0.085) | dY -1.6 |
| cart310_x_C_front_rise | shelfC_front [Z] | -2.67 (-2.29 / -3.48) | nan | dZ -31.1 (0.086) | dY -5.9 |
| cart310_x_D_front_fall | shelfD_front [Z] | -0.01 (-0.64 / +0.93) | nan | dZ -0.2 (0.072) | dY -0.0 |
| cart310_x_E_front_fall | shelfE_front [Z] | -4.59 (-6.08 / -3.14) | nan | dZ -76.5 (0.060) | dY -12.1 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +12.15 (+12.07 / +12.29) | nan | dZ +110.3 (0.110) | dY +24.0 |
| cart310_x_back_inner | top_back [Z] | -59.06 (-59.43 / -58.41) | nan | dZ -223.7 (0.264) | dY -109.4 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | -0.76 (-0.63 / -0.73) | nan | dX -3.9 (0.189) | dY -1.8 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -2.09 (-1.19 / -3.13) | nan | dX -5.1 (0.422) | dY +9.5 |
| cart80_x_board0_front | top_front [Y] | +15.37 (+18.02 / +12.62) | nan | dY +24.5 (0.628) | dZ +78.7 |
| cart80_x_board1600_front | top_front [Y] | +13.32 (+11.84 / +15.14) | nan | dY +25.1 (0.531) | dZ +80.7 |
| cart80_y_board0_outer | top_x0 [X] | -17.30 (-17.12 / -17.41) | nan | dX -28.1 (0.616) | dZ +412.9 |
| cart80_y_board1600_outer | top_x1600 [X] | +25.70 (+25.88 / +25.78) | nan | dX +58.8 (0.437) | dZ +85.9 |
| cart80_x_back_rail_out | top_back [Z] | +3.23 (+2.70 / +3.84) | nan | dZ +12.2 (0.263) | dY +6.4 |
| cart80_x_back_rail_in | top_back [Z] | -9.54 (-10.09 / -8.99) | nan | dZ -36.6 (0.261) | dY -19.1 |
| cart80_x_back_low | top_back [Z] | -55.67 (-56.35 / -53.39) | nan | dZ -209.8 (0.265) | dY -109.5 |
| cart80_x_A_front_out | shelfA_front [Z] | +3.56 (+4.38 / +2.18) | nan | dZ +22.4 (0.160) | dY +6.4 |
| cart80_x_A_front_in | shelfA_front [Z] | +10.16 (+10.24 / +9.39) | nan | dZ +64.4 (0.158) | dY +18.3 |
| cart80_x_B_front | shelfB_front [Z] | -0.94 (-1.32 / -0.71) | nan | dZ -7.8 (0.121) | dY -1.9 |
| cart80_x_C_front | shelfC_front [Z] | +3.29 (+3.96 / +2.49) | nan | dZ +35.2 (0.094) | dY +7.5 |
| cart80_x_D_front | shelfD_front [Z] | +3.52 (+3.30 / +3.47) | nan | dZ +44.9 (0.078) | dY +8.7 |
| cart80_x_E_front | shelfE_front [Z] | -2.15 (-2.04 / -3.01) | nan | dZ -32.3 (0.066) | dY -5.7 |
| cart80_x_E_front_in | shelfE_front [Z] | +0.63 (+0.61 / -0.18) | nan | dZ +9.6 (0.066) | dY +1.7 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | +0.24 (+0.59 / +0.08) | nan | dX +1.3 (0.169) | dY -0.6 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -5.28 (-5.21 / -5.40) | nan | dX -25.6 (0.206) | dY +11.7 |

* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): 80:top 22.0/+75, 310:top 20.9/+71, 80:A 2.6/+6, 310:A 1.6/+2, 80:B 2.7/-10, 310:B 2.0/-0, 80:C 2.6/+7, 310:C 1.4/-7
* lips cart 310: common lip height dZ +0.2 mm (formal +-5.7); per-lip deviation from it [px]: A +0.24, B +2.77, C -0.72, D -0.02, E -4.61 (rms 2.48 px); + shelf-depth scale +29.4 +- 10.6 permille -> rms 2.14 px; + D/E shift -38.7 +- 12.6 mm -> rms 2.00 px
* lips cart 80: common lip height dZ +15.2 mm (formal +-5.0); per-lip deviation from it [px]: A +1.14, B -2.77, C +1.87, D +2.33, E -3.16 (rms 2.35 px); + shelf-depth scale +6.6 +- 11.0 permille -> rms 2.29 px; + D/E shift -3.0 +- 13.5 mm -> rms 2.34 px

## camera: edge_lens_top_poses

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -7.00 (-6.78 / -7.43) | nan | dX -15.0 (0.466) | dZ +21.6 |
| cart310_board1600_outer | top_x1600 [X] | +7.25 (+8.63 / +5.85) | nan | dX +11.0 (0.658) | dZ +97.9 |
| cart310_x_board0_front | top_front [Y] | -2.11 (-6.44 / +1.98) | nan | dY -3.7 (0.571) | dZ -11.5 |
| cart310_x_board1600_front | top_front [Y] | -0.27 (-0.74 / +0.79) | nan | dY -0.4 (0.674) | dZ -1.2 |
| cart310_x_A_front_fall | shelfA_front [Z] | -5.45 (-6.39 / -4.40) | nan | dZ -30.4 (0.180) | dY -9.0 |
| cart310_x_B_front_fall | shelfB_front [Z] | +7.28 (+5.83 / +8.58) | nan | dZ +55.0 (0.132) | dY +13.8 |
| cart310_x_B_front_rise | shelfB_front [Z] | +4.51 (+3.17 / +5.83) | nan | dZ +34.0 (0.132) | dY +8.5 |
| cart310_x_C_front_fall | shelfC_front [Z] | +10.98 (+11.74 / +10.46) | nan | dZ +110.2 (0.100) | dY +23.9 |
| cart310_x_C_front_rise | shelfC_front [Z] | +9.06 (+9.45 / +8.16) | nan | dZ +89.5 (0.101) | dY +19.4 |
| cart310_x_D_front_fall | shelfD_front [Z] | +16.04 (+15.58 / +16.58) | nan | dZ +189.8 (0.084) | dY +37.4 |
| cart310_x_E_front_fall | shelfE_front [Z] | +14.70 (+13.79 / +15.32) | nan | dZ +208.7 (0.070) | dY +37.6 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +16.78 (+16.50 / +17.03) | nan | dZ +128.0 (0.131) | dY +32.1 |
| cart310_x_back_inner | top_back [Z] | -77.61 (-79.69 / -75.35) | nan | dZ -258.0 (0.301) | dY -142.4 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | +7.69 (+5.46 / +9.77) | nan | dX +37.8 (0.197) | dY +17.8 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -25.99 (-32.70 / -19.39) | nan | dX -53.7 (0.473) | dY +156.2 |
| cart80_x_board0_front | top_front [Y] | -1.20 (+1.39 / -3.86) | nan | dY -1.9 (0.643) | dZ -6.0 |
| cart80_x_board1600_front | top_front [Y] | +0.60 (-1.12 / +2.67) | nan | dY +1.1 (0.548) | dZ +3.4 |
| cart80_y_board0_outer | top_x0 [X] | -10.90 (-10.18 / -11.46) | nan | dX -17.2 (0.632) | dZ +196.5 |
| cart80_y_board1600_outer | top_x1600 [X] | +3.51 (+4.21 / +3.15) | nan | dX +7.8 (0.448) | dZ +11.5 |
| cart80_x_back_rail_out | top_back [Z] | -18.00 (-18.92 / -16.42) | nan | dZ -66.1 (0.272) | dY -35.1 |
| cart80_x_back_rail_in | top_back [Z] | -30.63 (-31.76 / -29.22) | nan | dZ -113.4 (0.270) | dY -60.2 |
| cart80_x_back_low | top_back [Z] | -77.03 (-77.98 / -73.98) | nan | dZ -280.6 (0.275) | dY -148.9 |
| cart80_x_A_front_out | shelfA_front [Z] | -10.55 (-10.91 / -10.33) | nan | dZ -63.8 (0.165) | dY -18.3 |
| cart80_x_A_front_in | shelfA_front [Z] | -3.57 (-4.46 / -3.42) | nan | dZ -21.9 (0.163) | dY -6.3 |
| cart80_x_B_front | shelfB_front [Z] | -13.67 (-15.05 / -12.24) | nan | dZ -110.6 (0.124) | dY -27.0 |
| cart80_x_C_front | shelfC_front [Z] | -8.62 (-8.81 / -8.24) | nan | dZ -90.1 (0.096) | dY -19.1 |
| cart80_x_D_front | shelfD_front [Z] | -7.91 (-9.18 / -6.75) | nan | dZ -99.1 (0.080) | dY -19.1 |
| cart80_x_E_front | shelfE_front [Z] | -13.11 (-14.07 / -12.72) | nan | dZ -194.5 (0.067) | dY -34.4 |
| cart80_x_E_front_in | shelfE_front [Z] | -10.36 (-11.42 / -9.89) | nan | dZ -153.6 (0.067) | dY -27.2 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | +2.15 (+2.67 / +1.86) | nan | dX +12.6 (0.169) | dY -5.8 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -3.87 (-3.68 / -4.11) | nan | dX -18.6 (0.207) | dY +8.6 |

* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): 80:top 2.2/+0, 310:top 3.9/-0, 80:A 18.2/-71, 310:A 13.7/-49, 80:B 19.3/-101, 310:B 16.6/-4, 80:C 14.5/-98, 310:C 20.3/+45
* lips cart 310: common lip height dZ +48.2 mm (formal +-19.4); per-lip deviation from it [px]: A -14.11, B +0.93, C +6.18, D +11.98, E +11.31 (rms 10.24 px); + shelf-depth scale -181.0 +- 5.8 permille -> rms 1.37 px; + D/E shift +179.6 +- 32.2 mm -> rms 6.15 px
* lips cart 80: common lip height dZ -92.1 mm (formal +-8.0); per-lip deviation from it [px]: A +4.68, B -2.27, C +0.20, D -0.55, E -6.90 (rms 4.20 px); + shelf-depth scale +59.1 +- 10.8 permille -> rms 2.32 px; + D/E shift -56.5 +- 17.1 mm -> rms 3.13 px

