# Cuboid model: predicted (drawing) vs traced cart edges

Script `work/41_cuboid_predict.py`. Offset = traced - predicted along the image normal [px], sign + = traced edge on the + side of
the first axis in [..]; implied shift = move of the drawing line (camera fixed) that puts it onto the traced edge [mm];
cam-sigma = 1-sigma of the mean offset over the bootstrap cameras. Cameras: markers_only = markers-only (fx=fy, pp fixed, dist k1+k2); joint = joint (k1k2_ppfix: fx=fy, pp fixed at image centre, k1+k2); edge_lens = edge-only lens (method_lines_joint) + poses from all stickers; edge_lens_shelf_poses = edge-only lens (method_lines_joint) + poses from stickers of rows A+B+C only; edge_lens_top_poses = edge-only lens (method_lines_joint) + poses from stickers of rows top only

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

* lips cart 310: common lip height dZ +8.3 mm (formal +-6.2); per-lip deviation from it [px]: A +3.47, B -3.22, C -1.14, D -3.07, E +2.11 (rms 2.98 px); + shelf-depth scale -25.8 +- 12.6 permille -> rms 2.63 px; + D/E shift +13.4 +- 17.3 mm -> rms 2.92 px
* lips cart 80: common lip height dZ +32.2 mm (formal +-7.5); per-lip deviation from it [px]: A +3.25, B +2.25, C -4.56, D -5.46, E +0.12 (rms 3.74 px); + shelf-depth scale -50.9 +- 12.1 permille -> rms 2.65 px; + D/E shift +48.4 +- 18.0 mm -> rms 3.09 px

## camera: joint

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -8.85 (-13.11 / -4.27) | 6.92 | dX -18.8 (0.473) | dZ +27.1 |
| cart310_board1600_outer | top_x1600 [X] | +7.70 (+10.73 / +5.63) | 6.54 | dX +11.9 (0.643) | dZ +77.7 |
| cart310_x_board0_front | top_front [Y] | +1.19 (-2.24 / +4.35) | 3.52 | dY +2.1 (0.567) | dZ +7.0 |
| cart310_x_board1600_front | top_front [Y] | +4.78 (+4.28 / +5.91) | 3.15 | dY +6.9 (0.690) | dZ +23.4 |
| cart310_x_A_front_fall | shelfA_front [Z] | -1.52 (-3.88 / -0.39) | 5.12 | dZ -9.6 (0.162) | dY -2.6 |
| cart310_x_B_front_fall | shelfB_front [Z] | +4.24 (+2.19 / +5.27) | 14.41 | dZ +36.3 (0.116) | dY +8.2 |
| cart310_x_B_front_rise | shelfB_front [Z] | +1.49 (-0.44 / +2.69) | 14.39 | dZ +12.5 (0.116) | dY +2.8 |
| cart310_x_C_front_fall | shelfC_front [Z] | +1.86 (+2.62 / +0.93) | 22.04 | dZ +22.0 (0.086) | dY +4.3 |
| cart310_x_C_front_rise | shelfC_front [Z] | +0.03 (+0.33 / -1.02) | 21.74 | dZ +0.5 (0.088) | dY +0.1 |
| cart310_x_D_front_fall | shelfD_front [Z] | +3.19 (+2.82 / +3.36) | 26.24 | dZ +43.8 (0.073) | dY +7.7 |
| cart310_x_E_front_fall | shelfE_front [Z] | -1.82 (-2.18 / -2.06) | 30.25 | dZ -30.4 (0.060) | dY -4.9 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +14.10 (+13.72 / +14.31) | 14.81 | dZ +123.4 (0.114) | dY +27.8 |
| cart310_x_back_inner | top_back [Z] | -77.64 (-81.41 / -74.01) | 4.47 | dZ -253.5 (0.306) | dY -135.2 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | -1.21 (+0.82 / -2.88) | 26.13 | dX -6.1 (0.171) | dY -2.6 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -5.29 (-3.69 / -6.93) | 2.11 | dX -12.0 (0.444) | dY +23.4 |
| cart80_x_board0_front | top_front [Y] | +4.41 (+6.29 / +2.68) | 4.02 | dY +6.5 (0.674) | dZ +19.6 |
| cart80_x_board1600_front | top_front [Y] | +3.49 (+2.94 / +4.30) | 5.32 | dY +6.4 (0.548) | dZ +19.1 |
| cart80_y_board0_outer | top_x0 [X] | -10.22 (-8.63 / -12.64) | 12.89 | dX -16.2 (0.629) | dZ +204.3 |
| cart80_y_board1600_outer | top_x1600 [X] | +7.82 (+9.87 / +5.90) | 6.16 | dX +17.7 (0.442) | dZ +23.2 |
| cart80_x_back_rail_out | top_back [Z] | -18.65 (-23.42 / -14.59) | 4.65 | dZ -61.1 (0.306) | dY -34.9 |
| cart80_x_back_rail_in | top_back [Z] | -30.54 (-36.01 / -27.41) | 5.14 | dZ -100.6 (0.304) | dY -57.4 |
| cart80_x_back_low | top_back [Z] | -78.01 (-82.50 / -71.94) | 4.45 | dZ -253.1 (0.308) | dY -144.4 |
| cart80_x_A_front_out | shelfA_front [Z] | +1.46 (-0.39 / -1.15) | 4.23 | dZ +8.3 (0.176) | dY +2.5 |
| cart80_x_A_front_in | shelfA_front [Z] | +8.86 (+8.43 / +6.99) | 4.50 | dZ +51.5 (0.172) | dY +15.6 |
| cart80_x_B_front | shelfB_front [Z] | +1.45 (-0.37 / +1.39) | 1.88 | dZ +11.1 (0.128) | dY +2.8 |
| cart80_x_C_front | shelfC_front [Z] | +7.90 (+7.88 / +6.47) | 2.07 | dZ +81.6 (0.097) | dY +17.8 |
| cart80_x_D_front | shelfD_front [Z] | +8.75 (+8.01 / +7.71) | 2.95 | dZ +109.5 (0.080) | dY +21.7 |
| cart80_x_E_front | shelfE_front [Z] | +3.23 (+3.55 / +0.79) | 3.66 | dZ +49.0 (0.066) | dY +8.8 |
| cart80_x_E_front_in | shelfE_front [Z] | +6.00 (+6.19 / +3.59) | 3.71 | dZ +90.5 (0.067) | dY +16.3 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | -0.03 (+0.05 / +0.06) | 2.83 | dX -0.2 (0.158) | dY +0.1 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -4.39 (-4.65 / -4.18) | 1.76 | dX -22.5 (0.195) | dY +9.8 |

* lips cart 310: common lip height dZ +9.3 mm (formal +-5.8); per-lip deviation from it [px]: A +3.02, B -3.16, C -1.06, D -2.51, E +2.37 (rms 2.79 px); + shelf-depth scale -19.0 +- 12.4 permille -> rms 2.59 px; + D/E shift +5.5 +- 16.4 mm -> rms 2.78 px
* lips cart 80: common lip height dZ +31.4 mm (formal +-8.4); per-lip deviation from it [px]: A +4.06, B +2.57, C -4.86, D -6.24, E -1.14 (rms 4.27 px); + shelf-depth scale -64.0 +- 11.5 permille -> rms 2.55 px; + D/E shift +63.5 +- 18.4 mm -> rms 3.19 px

## camera: edge_lens

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -9.78 (-11.03 / -8.86) | nan | dX -20.9 (0.469) | dZ +30.9 |
| cart310_board1600_outer | top_x1600 [X] | +9.05 (+9.61 / +8.33) | nan | dX +13.8 (0.656) | dZ +91.7 |
| cart310_x_board0_front | top_front [Y] | -0.78 (-5.06 / +3.27) | nan | dY -1.4 (0.571) | dZ -4.6 |
| cart310_x_board1600_front | top_front [Y] | +1.63 (+1.10 / +2.75) | nan | dY +2.5 (0.666) | dZ +8.0 |
| cart310_x_A_front_fall | shelfA_front [Z] | -6.06 (-7.11 / -5.03) | nan | dZ -36.3 (0.167) | dY -10.1 |
| cart310_x_B_front_fall | shelfB_front [Z] | +1.98 (+0.12 / +3.66) | nan | dZ +16.1 (0.121) | dY +3.8 |
| cart310_x_B_front_rise | shelfB_front [Z] | -0.82 (-2.53 / +0.88) | nan | dZ -6.9 (0.121) | dY -1.6 |
| cart310_x_C_front_fall | shelfC_front [Z] | +1.91 (+1.90 / +1.93) | nan | dZ +21.0 (0.091) | dY +4.3 |
| cart310_x_C_front_rise | shelfC_front [Z] | -0.31 (-0.39 / -0.58) | nan | dZ -3.4 (0.092) | dY -0.7 |
| cart310_x_D_front_fall | shelfD_front [Z] | +3.98 (+2.92 / +5.35) | nan | dZ +51.8 (0.077) | dY +9.5 |
| cart310_x_E_front_fall | shelfE_front [Z] | +0.73 (-1.13 / +2.47) | nan | dZ +11.1 (0.064) | dY +1.9 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +11.58 (+11.13 / +11.93) | nan | dZ +96.1 (0.121) | dY +22.6 |
| cart310_x_back_inner | top_back [Z] | -75.18 (-76.85 / -73.34) | nan | dZ -254.3 (0.296) | dY -137.1 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | +2.33 (+2.48 / +2.28) | nan | dX +12.4 (0.187) | dY +5.6 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -5.52 (-5.91 / -5.31) | nan | dX -12.5 (0.437) | dY +25.6 |
| cart80_x_board0_front | top_front [Y] | +1.92 (+4.53 / -0.77) | nan | dY +3.0 (0.650) | dZ +8.6 |
| cart80_x_board1600_front | top_front [Y] | +3.98 (+2.14 / +6.19) | nan | dY +7.4 (0.539) | dZ +21.4 |
| cart80_y_board0_outer | top_x0 [X] | -8.76 (-10.01 / -7.58) | nan | dX -13.4 (0.654) | dZ +207.2 |
| cart80_y_board1600_outer | top_x1600 [X] | +11.67 (+9.71 / +13.98) | nan | dX +27.2 (0.427) | dZ +35.8 |
| cart80_x_back_rail_out | top_back [Z] | -14.26 (-17.44 / -10.36) | nan | dZ -48.8 (0.294) | dY -28.1 |
| cart80_x_back_rail_in | top_back [Z] | -26.43 (-29.89 / -23.12) | nan | dZ -91.1 (0.291) | dY -52.5 |
| cart80_x_back_low | top_back [Z] | -73.60 (-76.52 / -68.48) | nan | dZ -248.3 (0.296) | dY -143.0 |
| cart80_x_A_front_out | shelfA_front [Z] | -4.52 (-4.22 / -4.78) | nan | dZ -25.1 (0.179) | dY -7.8 |
| cart80_x_A_front_in | shelfA_front [Z] | +2.25 (+1.79 / +2.14) | nan | dZ +12.7 (0.177) | dY +4.0 |
| cart80_x_B_front | shelfB_front [Z] | -3.02 (-3.28 / -2.74) | nan | dZ -22.9 (0.132) | dY -6.0 |
| cart80_x_C_front | shelfC_front [Z] | +4.99 (+6.05 / +3.86) | nan | dZ +49.5 (0.101) | dY +11.2 |
| cart80_x_D_front | shelfD_front [Z] | +7.18 (+7.60 / +6.58) | nan | dZ +85.7 (0.084) | dY +17.6 |
| cart80_x_E_front | shelfE_front [Z] | +2.82 (+3.66 / +1.31) | nan | dZ +40.4 (0.070) | dY +7.6 |
| cart80_x_E_front_in | shelfE_front [Z] | +5.63 (+6.33 / +4.13) | nan | dZ +80.3 (0.070) | dY +15.1 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | -0.81 (-0.69 / -0.78) | nan | dX -4.8 (0.168) | dY +2.2 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -5.64 (-5.74 / -5.61) | nan | dX -27.4 (0.205) | dY +12.4 |

* lips cart 310: common lip height dZ -4.2 mm (formal +-7.4); per-lip deviation from it [px]: A +5.35, B -2.48, C -2.29, D -4.30, E -1.00 (rms 3.70 px); + shelf-depth scale -57.4 +- 9.7 permille -> rms 2.12 px; + D/E shift +47.1 +- 17.4 mm -> rms 3.12 px
* lips cart 80: common lip height dZ +1.4 mm (formal +-9.2); per-lip deviation from it [px]: A +4.76, B +3.20, C -4.85, D -7.06, E -2.72 (rms 4.82 px); + shelf-depth scale -75.2 +- 10.2 permille -> rms 2.30 px; + D/E shift +78.7 +- 17.2 mm -> rms 3.09 px

## camera: edge_lens_shelf_poses

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -23.28 (-25.73 / -21.12) | nan | dX -49.5 (0.471) | dZ +77.0 |
| cart310_board1600_outer | top_x1600 [X] | +10.84 (+11.04 / +10.47) | nan | dX +17.0 (0.639) | dZ +107.5 |
| cart310_x_board0_front | top_front [Y] | +5.64 (+1.63 / +9.42) | nan | dY +9.9 (0.565) | dZ +35.3 |
| cart310_x_board1600_front | top_front [Y] | +13.10 (+12.67 / +14.11) | nan | dY +20.0 (0.654) | dZ +71.1 |
| cart310_x_A_front_fall | shelfA_front [Z] | +0.26 (+0.40 / +0.07) | nan | dZ +1.7 (0.151) | dY +0.4 |
| cart310_x_B_front_fall | shelfB_front [Z] | +3.11 (+2.13 / +3.88) | nan | dZ +28.0 (0.110) | dY +6.1 |
| cart310_x_B_front_rise | shelfB_front [Z] | +0.36 (-0.55 / +1.18) | nan | dZ +3.2 (0.111) | dY +0.7 |
| cart310_x_C_front_fall | shelfC_front [Z] | -0.78 (+0.02 / -1.28) | nan | dZ -9.3 (0.083) | dY -1.7 |
| cart310_x_C_front_rise | shelfC_front [Z] | -2.69 (-2.26 / -3.62) | nan | dZ -31.8 (0.084) | dY -6.0 |
| cart310_x_D_front_fall | shelfD_front [Z] | -0.47 (-1.04 / +0.31) | nan | dZ -6.7 (0.070) | dY -1.1 |
| cart310_x_E_front_fall | shelfE_front [Z] | -5.64 (-6.97 / -4.41) | nan | dZ -96.8 (0.058) | dY -15.1 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +12.47 (+12.43 / +12.58) | nan | dZ +113.6 (0.110) | dY +24.7 |
| cart310_x_back_inner | top_back [Z] | -63.02 (-63.86 / -61.96) | nan | dZ -227.4 (0.277) | dY -115.2 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | -2.58 (-0.49 / -4.31) | nan | dX -13.2 (0.180) | dY -5.8 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -1.10 (+1.36 / -3.75) | nan | dX -3.0 (0.419) | dY +5.4 |
| cart80_x_board0_front | top_front [Y] | +13.91 (+16.58 / +11.13) | nan | dY +21.8 (0.639) | dZ +68.9 |
| cart80_x_board1600_front | top_front [Y] | +12.99 (+11.33 / +15.00) | nan | dY +24.2 (0.536) | dZ +76.6 |
| cart80_y_board0_outer | top_x0 [X] | -8.99 (-10.03 / -7.99) | nan | dX -14.1 (0.636) | dZ +173.7 |
| cart80_y_board1600_outer | top_x1600 [X] | +22.59 (+22.47 / +23.10) | nan | dX +52.1 (0.434) | dZ +72.6 |
| cart80_x_back_rail_out | top_back [Z] | -1.80 (-3.26 / +0.07) | nan | dZ -6.6 (0.276) | dY -3.6 |
| cart80_x_back_rail_in | top_back [Z] | -14.35 (-15.93 / -12.73) | nan | dZ -52.5 (0.274) | dY -28.4 |
| cart80_x_back_low | top_back [Z] | -60.86 (-62.33 / -57.54) | nan | dZ -218.5 (0.279) | dY -118.0 |
| cart80_x_A_front_out | shelfA_front [Z] | +2.96 (+3.84 / +1.90) | nan | dZ +18.3 (0.163) | dY +5.3 |
| cart80_x_A_front_in | shelfA_front [Z] | +9.54 (+9.56 / +8.97) | nan | dZ +59.4 (0.161) | dY +17.1 |
| cart80_x_B_front | shelfB_front [Z] | -0.93 (-1.10 / -0.77) | nan | dZ -7.8 (0.121) | dY -1.9 |
| cart80_x_C_front | shelfC_front [Z] | +3.13 (+4.05 / +2.19) | nan | dZ +33.9 (0.093) | dY +7.2 |
| cart80_x_D_front | shelfD_front [Z] | +3.01 (+3.14 / +2.78) | nan | dZ +39.1 (0.077) | dY +7.5 |
| cart80_x_E_front | shelfE_front [Z] | -3.16 (-2.74 / -4.17) | nan | dZ -48.6 (0.065) | dY -8.5 |
| cart80_x_E_front_in | shelfE_front [Z] | -0.37 (-0.07 / -1.34) | nan | dZ -5.5 (0.065) | dY -1.0 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | +1.99 (+2.96 / +1.34) | nan | dX +12.0 (0.163) | dY -5.3 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -5.37 (-4.85 / -5.92) | nan | dX -26.8 (0.201) | dY +11.8 |

* lips cart 310: common lip height dZ -0.9 mm (formal +-6.8); per-lip deviation from it [px]: A -0.40, B -3.21, C +0.70, D +0.40, E +5.59 (rms 2.93 px); + shelf-depth scale +37.3 +- 12.3 permille -> rms 2.47 px; + D/E shift -51.2 +- 14.4 mm -> rms 2.24 px
* lips cart 80: common lip height dZ +11.6 mm (formal +-5.3); per-lip deviation from it [px]: A -1.08, B +2.33, C -2.06, D -2.12, E +3.91 (rms 2.51 px); + shelf-depth scale +10.4 +- 11.4 permille -> rms 2.40 px; + D/E shift -10.5 +- 14.2 mm -> rms 2.44 px

## camera: edge_lens_top_poses

| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |
|---|---|---|---|---|---|
| cart310_board0_outer | top_x0 [X] | -6.88 (-7.11 / -6.99) | nan | dX -14.7 (0.467) | dZ +21.4 |
| cart310_board1600_outer | top_x1600 [X] | +7.15 (+7.94 / +6.21) | nan | dX +10.8 (0.662) | dZ +73.0 |
| cart310_x_board0_front | top_front [Y] | -1.81 (-6.16 / +2.32) | nan | dY -3.2 (0.572) | dZ -9.9 |
| cart310_x_board1600_front | top_front [Y] | -0.94 (-1.50 / +0.21) | nan | dY -1.4 (0.669) | dZ -4.3 |
| cart310_x_A_front_fall | shelfA_front [Z] | -5.91 (-7.29 / -4.56) | nan | dZ -33.5 (0.177) | dY -9.9 |
| cart310_x_B_front_fall | shelfB_front [Z] | +5.48 (+3.45 / +7.30) | nan | dZ +42.5 (0.128) | dY +10.5 |
| cart310_x_B_front_rise | shelfB_front [Z] | +2.68 (+0.81 / +4.51) | nan | dZ +20.6 (0.129) | dY +5.1 |
| cart310_x_C_front_fall | shelfC_front [Z] | +7.84 (+7.77 / +7.85) | nan | dZ +81.6 (0.096) | dY +17.4 |
| cart310_x_C_front_rise | shelfC_front [Z] | +5.60 (+5.47 / +5.37) | nan | dZ +57.4 (0.098) | dY +12.3 |
| cart310_x_D_front_fall | shelfD_front [Z] | +11.34 (+10.24 / +12.69) | nan | dZ +140.1 (0.081) | dY +27.1 |
| cart310_x_E_front_fall | shelfE_front [Z] | +9.25 (+7.42 / +10.84) | nan | dZ +137.4 (0.067) | dY +24.3 |
| cart310_x_B_inner_fall | shelfB_front [Z] | +15.14 (+14.63 / +15.53) | nan | dZ +118.7 (0.128) | dY +29.4 |
| cart310_x_back_inner | top_back [Z] | -77.08 (-79.04 / -74.96) | nan | dZ -252.1 (0.306) | dY -141.2 |
| cart310_z_post_front_left_right | post_x0_y0 [X] | +7.37 (+5.45 / +9.13) | nan | dX +36.9 (0.194) | dY +17.4 |
| cart310_z_post_front_right | post_x1600_y0 [X] | -10.61 (-12.43 / -8.95) | nan | dX -23.1 (0.451) | dY +50.9 |
| cart80_x_board0_front | top_front [Y] | -2.06 (+0.54 / -4.73) | nan | dY -3.2 (0.637) | dZ -10.1 |
| cart80_x_board1600_front | top_front [Y] | +0.88 (-1.02 / +3.14) | nan | dY +1.6 (0.548) | dZ +4.9 |
| cart80_y_board0_outer | top_x0 [X] | -10.32 (-10.42 / -10.13) | nan | dX -16.3 (0.635) | dZ +134.1 |
| cart80_y_board1600_outer | top_x1600 [X] | +3.31 (+4.44 / +2.68) | nan | dX +7.4 (0.450) | dZ +10.9 |
| cart80_x_back_rail_out | top_back [Z] | -18.12 (-18.59 / -16.82) | nan | dZ -64.8 (0.280) | dY -35.3 |
| cart80_x_back_rail_in | top_back [Z] | -30.85 (-31.50 / -29.62) | nan | dZ -111.2 (0.277) | dY -60.6 |
| cart80_x_back_low | top_back [Z] | -77.12 (-77.64 / -74.41) | nan | dZ -273.6 (0.282) | dY -149.1 |
| cart80_x_A_front_out | shelfA_front [Z] | -11.00 (-11.52 / -10.22) | nan | dZ -66.0 (0.167) | dY -19.3 |
| cart80_x_A_front_in | shelfA_front [Z] | -3.96 (-5.07 / -3.49) | nan | dZ -24.1 (0.165) | dY -7.0 |
| cart80_x_B_front | shelfB_front [Z] | -14.14 (-15.77 / -12.28) | nan | dZ -115.0 (0.123) | dY -28.4 |
| cart80_x_C_front | shelfC_front [Z] | -9.56 (-10.03 / -8.69) | nan | dZ -101.3 (0.094) | dY -21.6 |
| cart80_x_D_front | shelfD_front [Z] | -9.28 (-10.95 / -7.55) | nan | dZ -118.6 (0.078) | dY -23.0 |
| cart80_x_E_front | shelfE_front [Z] | -14.88 (-16.32 / -13.82) | nan | dZ -225.9 (0.066) | dY -40.1 |
| cart80_x_E_front_in | shelfE_front [Z] | -12.14 (-13.69 / -10.97) | nan | dZ -184.3 (0.066) | dY -32.7 |
| cart80_z_post1600_sil | post_x1600_y0 [X] | +0.00 (+0.19 / -0.03) | nan | dX -0.1 (0.168) | dY +0.0 |
| cart80_z_post1600_sil_low | post_x1600_y0 [X] | -5.05 (-5.09 / -5.07) | nan | dX -24.0 (0.210) | dY +11.3 |

* lips cart 310: common lip height dZ +28.5 mm (formal +-15.0); per-lip deviation from it [px]: A +10.95, B -1.83, C -5.11, D -9.04, E -7.34 (rms 7.74 px); + shelf-depth scale -138.3 +- 7.9 permille -> rms 1.82 px; + D/E shift +131.9 +- 27.8 mm -> rms 5.15 px
* lips cart 80: common lip height dZ -99.3 mm (formal +-9.6); per-lip deviation from it [px]: A -5.54, B +1.91, C +0.18, D +1.49, E +8.34 (rms 5.03 px); + shelf-depth scale +76.1 +- 11.4 permille -> rms 2.45 px; + D/E shift -76.6 +- 19.1 mm -> rms 3.45 px

