# What-if: top stickers tilted (angle scan)

Additional analysis on request; the main result (results/lens_result.json, drawing geometry = all stickers flat) is NOT changed. Script: `work/60_top_tilt_scan_fixed.py` (reviewed and corrected copy of `work/60_top_tilt_scan.py`; the scan, plates, common and check phases are identical, the review added the phases/sections marked [review fix]); all numbers: `work/cache/top_tilt_scan.json`; plot: `results/top_tilt_scan.png`.

## Definitions and sign conventions

* **cart_frame**: spec/cart-marker-layout.json: origin top front-left corner, X along the 1600 mm front, Y into the cart (0 front, 450 back), Z up, Z = 0 at the top plate
* **top_stickers**: ids 0 (82.5,60,0), 1 (1517.5,60,0), 2 (1517.5,390,0), 92/322 (82.5,390,0); plate X0 = ids 0 + 92/322, plate X1600 = ids 1 + 2; shelf stickers unchanged
* **Y_centre**: rotation about an axis parallel to cart Y through the sticker centre; angle > 0: the OUTER side of the plate goes UP (towards X=0 at the X0 plate, towards X=1600 at the X1600 plate); centres unchanged
* **X_centre**: rotation about an axis parallel to cart X through the sticker centre; angle > 0: the BACK side (larger Y) goes UP; centres unchanged
* **Y_hinge**: as Y_centre, but about a hinge line parallel to Y at 55.0 mm from the sticker centre towards the cart middle (edge of the 110 mm sticker incl. white border); angle > 0: outer side up, the centre rises by 55.0 sin(angle) mm
* **per_plate**: one angle per plate (80/X0, 80/X1600, 310/X0, 310/X1600) in the Y_centre convention (ay_*), optionally + one X_centre angle per plate (bx_*); both stickers of a plate share the angles; composite rotation R = Rx(bx) Ry(ay)
* **per_plate_uncertainty**: [review fix] 1-sigma of a plate tilt = [max(delete-one-sticker jackknife, stratified cluster bootstrap) (+) lens propagation over the 100 bootstrap lenses] (+) half the range over 6 definitions of the cart reference (A LS pose all corners, B + free plate heights, C/D/E shape-only with free top-sticker positions, J joint fit); key refdef.<kind>.sigma_total_incl_definition
* **tilt_matrix**: offset (dx,dy,dz) from the pivot -> R (dx,dy,dz); Ry: (o,0,0) -> (o cos a, 0, sin a) with o = -1 (X0 plate) / +1 (X1600 plate); Rx: (0,1,0) -> (0, cos b, sin b)
* **combined**: 50_combined.py main estimator (k1k2, fx=fy, pp free), same start and variance-component block weighting; only the top-sticker 3D corners differ
* **sig**: converged block sigmas of the combined fit [px]: M sticker corners (per coordinate), L traced sides of partly hidden stickers, V vanishing-point groups, S straightness
* **sticker_rms**: radial corner residuals [px] of all 62 valid corners with the given lens FIXED and one least-squares 6-DoF pose per cart (same definition as lens_result.json rms_details); 'top' = 32 corners of the 8 top stickers, 'shelf' = 30 corners of rows A-C
* **sticker_only**: bundle adjustment of the 62 corners alone (f single, k1, k2; pp fixed at the image centre (959.5, 539.5) or free; one pose per cart); no fold barrier unless 'fold_barrier'; fold_margin < 0 or min_radial_slope < 0.02 = radial map folds inside the image (not a usable lens)
* **mapping**: rotation-compensated displacement of the lens relative to the reference (main result or edges-only lens) on a 30 px grid, median / max |d| per region (evaltools.region_defs: centre, cart_band, corners)

## Verification

Sign convention (Y_centre +10 deg; corners with Z > 0 are on the outer side, centres unchanged):

```
sticker 80:0 plate 80/X0 corner 0: X    37.5 ->   38.18, Y 105.0, Z 0.0 ->  +7.81 (outer side)
sticker 80:0 plate 80/X0 corner 1: X   127.5 ->  126.82, Y 105.0, Z 0.0 ->  -7.81 (inner side)
sticker 80:0 plate 80/X0 corner 2: X   127.5 ->  126.82, Y  15.0, Z 0.0 ->  -7.81 (inner side)
sticker 80:0 plate 80/X0 corner 3: X    37.5 ->   38.18, Y  15.0, Z 0.0 ->  +7.81 (outer side)
sticker 80:1 plate 80/X1600 corner 0: X  1472.5 -> 1473.18, Y 105.0, Z 0.0 ->  -7.81 (inner side)
sticker 80:1 plate 80/X1600 corner 1: X  1562.5 -> 1561.82, Y 105.0, Z 0.0 ->  +7.81 (outer side)
sticker 80:1 plate 80/X1600 corner 2: X  1562.5 -> 1561.82, Y  15.0, Z 0.0 ->  +7.81 (outer side)
sticker 80:1 plate 80/X1600 corner 3: X  1472.5 -> 1473.18, Y  15.0, Z 0.0 ->  -7.81 (inner side)
X_centre +10: sticker 80:0 corner 0: Y  105.0 (centre 60) -> Z  +7.81
X_centre +10: sticker 80:0 corner 1: Y  105.0 (centre 60) -> Z  +7.81
X_centre +10: sticker 80:0 corner 2: Y   15.0 (centre 60) -> Z  -7.81
X_centre +10: sticker 80:0 corner 3: Y   15.0 (centre 60) -> Z  -7.81
Y_hinge +10: sticker 80:0 (80/X0) centre Z +9.551 mm (expected +9.551), centre X 83.34 (drawing 82.5)
Y_hinge +10: sticker 80:1 (80/X1600) centre Z +9.551 mm (expected +9.551), centre X 1516.66 (drawing 1517.5)
cart 80: +10 deg Y_centre moves the top-sticker corners in the image by 2.27 px on average (max 2.48)
cart 310: +10 deg Y_centre moves the top-sticker corners in the image by 2.15 px on average (max 2.47)
```
0 deg reproduces the main fit: f = 1468.631 px (main 1468.631), cx 928.21, cy 503.23, sticker RMS 7.886 px (main 7.886).

References: main f 1468.6 +- 46.6, cx 928.2, cy 503.2, k1 -0.3516, k2 0.0994; edges-only (V + S blocks) f 1436.4, cx 926.9, cy 501.1, k1 -0.3357, k2 0.0904. Main mapping 1-sigma: centre 4.0 / cart band 15.3 / corners 21.6 px.


## Y_centre

Combined fit (main estimator) with the tilted top stickers; sticker RMS = radial, combined lens fixed, LS pose per cart; dmap = rotation-compensated median |displacement| vs the main result (centre / cart band / corners).

| angle [deg] | f | cx | cy | k1 | k2 | sig M | sig L | sig V | sig S | RMS all | RMS top | RMS shelf | max all | dmap vs main c/b/k [px] | dmap vs edges-only band | fold margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -15 | 1468.8 | 928.2 | 503.2 | -0.3516 | 0.0994 | 11.04 | 1.36 | 0.365 | 0.231 | 8.70 | 8.33 | 9.08 | 14.8 | 0.0 / 0.1 / 0.1 | 10.3 | 9.99 |
| -12 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.94 | 1.37 | 0.365 | 0.231 | 8.46 | 7.83 | 9.08 | 13.9 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -10 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.89 | 1.37 | 0.365 | 0.231 | 8.32 | 7.54 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -8 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.84 | 1.37 | 0.365 | 0.231 | 8.20 | 7.29 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -6 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.80 | 1.37 | 0.365 | 0.231 | 8.10 | 7.06 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -4 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.77 | 1.37 | 0.365 | 0.231 | 8.01 | 6.87 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -3 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.76 | 1.37 | 0.365 | 0.231 | 7.98 | 6.79 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -2 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.74 | 1.37 | 0.365 | 0.231 | 7.94 | 6.71 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -1 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.73 | 1.37 | 0.365 | 0.231 | 7.91 | 6.64 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +0 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 1.37 | 0.365 | 0.231 | 7.89 | 6.58 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +1 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.71 | 1.37 | 0.365 | 0.231 | 7.86 | 6.53 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +2 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.70 | 1.37 | 0.365 | 0.231 | 7.84 | 6.49 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +3 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.70 | 1.37 | 0.365 | 0.231 | 7.83 | 6.45 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +4 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.69 | 1.37 | 0.365 | 0.231 | 7.81 | 6.42 | 9.06 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +6 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 1.37 | 0.365 | 0.231 | 7.80 | 6.39 | 9.06 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +8 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 1.37 | 0.365 | 0.231 | 7.79 | 6.38 | 9.06 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +10 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 1.37 | 0.365 | 0.231 | 7.80 | 6.40 | 9.05 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +12 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 1.37 | 0.365 | 0.231 | 7.81 | 6.44 | 9.05 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +15 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.69 | 1.37 | 0.365 | 0.231 | 7.85 | 6.55 | 9.04 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |

Per plate (top-sticker RMS, combined lens) and sticker RMS with the main / edges-only lens fixed; sticker-only fits (no fold barrier; pp fixed = image centre):

| angle | 80/X0 | 80/X1600 | 310/X0 | 310/X1600 | RMS main lens | RMS edges lens | only pp fixed: f / k1 / k2 / RMS | only pp free: f / cx / cy / RMS / fold | only pp free + barrier: f / cy / RMS |
|---|---|---|---|---|---|---|---|---|---|
| -15 | 7.09 | 9.83 | 8.66 | 7.47 | 8.70 | 8.15 | 1320 / -0.095 / -0.203 / 6.70 | 1331 / 928 / 471 / 6.51 / FOLDED (-0.18) | 1356 / 396 / 6.68 (at barrier) |
| -12 | 6.70 | 9.25 | 7.86 | 7.30 | 8.46 | 7.89 | 1316 / -0.095 / -0.203 / 6.37 | 1326 / 931 / 474 / 6.18 / FOLDED (-0.18) | 1347 / 430 / 6.37 |
| -10 | 6.46 | 8.92 | 7.36 | 7.22 | 8.32 | 7.73 | 1313 / -0.096 / -0.203 / 6.17 | 1323 / 932 / 476 / 5.98 / FOLDED (-0.18) | 1342 / 455 / 6.20 |
| -8 | 6.24 | 8.63 | 6.89 | 7.17 | 8.20 | 7.60 | 1310 / -0.096 / -0.203 / 5.99 | 1320 / 934 / 478 / 5.81 / FOLDED (-0.18) | 1334 / 432 / 6.05 |
| -6 | 6.03 | 8.38 | 6.46 | 7.15 | 8.10 | 7.49 | 1307 / -0.096 / -0.203 / 5.84 | 1317 / 935 / 480 / 5.66 / FOLDED (-0.18) | 1333 / 427 / 5.91 |
| -4 | 5.83 | 8.17 | 6.07 | 7.15 | 8.01 | 7.39 | 1305 / -0.096 / -0.203 / 5.70 | 1314 / 936 / 482 / 5.54 / FOLDED (-0.18) | 1332 / 423 / 5.79 |
| -3 | 5.74 | 8.09 | 5.89 | 7.16 | 7.98 | 7.35 | 1304 / -0.095 / -0.203 / 5.65 | 1313 / 937 / 482 / 5.48 / FOLDED (-0.18) | 1332 / 421 / 5.74 |
| -2 | 5.65 | 8.01 | 5.71 | 7.17 | 7.94 | 7.32 | 1302 / -0.095 / -0.203 / 5.59 | 1311 / 937 / 483 / 5.43 / FOLDED (-0.18) | 1311 / 287 / 5.41 |
| -1 | 5.57 | 7.94 | 5.55 | 7.19 | 7.91 | 7.28 | 1301 / -0.095 / -0.203 / 5.54 | 1310 / 938 / 484 / 5.38 / FOLDED (-0.18) | 1310 / 287 / 5.37 |
| +0 | 5.49 | 7.88 | 5.40 | 7.21 | 7.89 | 7.25 | 1300 / -0.095 / -0.203 / 5.50 | 1309 / 938 / 485 / 5.34 / FOLDED (-0.18) | 1309 / 287 / 5.34 |
| +1 | 5.41 | 7.83 | 5.26 | 7.24 | 7.86 | 7.23 | 1299 / -0.095 / -0.203 / 5.46 | 1308 / 939 / 485 / 5.31 / FOLDED (-0.18) | 1308 / 286 / 5.31 |
| +2 | 5.34 | 7.79 | 5.13 | 7.27 | 7.84 | 7.21 | 1298 / -0.095 / -0.203 / 5.43 | 1307 / 939 / 486 / 5.28 / FOLDED (-0.18) | 1307 / 286 / 5.29 |
| +3 | 5.27 | 7.76 | 5.01 | 7.30 | 7.83 | 7.19 | 1297 / -0.095 / -0.203 / 5.40 | 1305 / 939 / 486 / 5.25 / FOLDED (-0.18) | 1306 / 286 / 5.27 |
| +4 | 5.21 | 7.74 | 4.90 | 7.34 | 7.81 | 7.17 | 1296 / -0.095 / -0.203 / 5.37 | 1304 / 940 / 487 / 5.23 / FOLDED (-0.18) | 1305 / 286 / 5.26 |
| +6 | 5.11 | 7.72 | 4.71 | 7.43 | 7.80 | 7.15 | 1294 / -0.095 / -0.203 / 5.34 | 1302 / 940 / 488 / 5.20 / FOLDED (-0.18) | 1329 / 423 / 5.50 |
| +8 | 5.03 | 7.73 | 4.56 | 7.54 | 7.79 | 7.15 | 1293 / -0.095 / -0.203 / 5.32 | 1301 / 941 / 488 / 5.18 / FOLDED (-0.18) | 1329 / 435 / 5.50 |
| +10 | 4.99 | 7.76 | 4.46 | 7.66 | 7.80 | 7.15 | 1291 / -0.095 / -0.203 / 5.32 | 1299 / 941 / 489 / 5.19 / FOLDED (-0.18) | 1328 / 445 / 5.51 |
| +12 | 4.98 | 7.82 | 4.39 | 7.79 | 7.81 | 7.17 | 1290 / -0.095 / -0.203 / 5.34 | 1298 / 941 / 489 / 5.20 / FOLDED (-0.18) | 1341 / 478 / 5.58 |
| +15 | 5.04 | 7.93 | 4.34 | 8.02 | 7.85 | 7.21 | 1288 / -0.095 / -0.203 / 5.39 | 1296 / 942 / 489 / 5.26 / FOLDED (-0.18) | 1333 / 502 / 5.61 |

## X_centre

Combined fit (main estimator) with the tilted top stickers; sticker RMS = radial, combined lens fixed, LS pose per cart; dmap = rotation-compensated median |displacement| vs the main result (centre / cart band / corners).

| angle [deg] | f | cx | cy | k1 | k2 | sig M | sig L | sig V | sig S | RMS all | RMS top | RMS shelf | max all | dmap vs main c/b/k [px] | dmap vs edges-only band | fold margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -10 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.79 | 1.37 | 0.365 | 0.231 | 8.04 | 6.90 | 9.09 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -8 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.76 | 1.37 | 0.365 | 0.231 | 7.97 | 6.75 | 9.09 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -6 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.74 | 1.37 | 0.365 | 0.231 | 7.92 | 6.65 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -4 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.73 | 1.37 | 0.365 | 0.231 | 7.89 | 6.59 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| -2 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 1.37 | 0.365 | 0.231 | 7.88 | 6.57 | 9.08 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +0 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 1.37 | 0.365 | 0.231 | 7.89 | 6.58 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +2 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 1.37 | 0.365 | 0.231 | 7.90 | 6.63 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +4 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.73 | 1.37 | 0.365 | 0.231 | 7.93 | 6.71 | 9.06 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +6 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.74 | 1.37 | 0.365 | 0.231 | 7.97 | 6.80 | 9.06 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +8 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.76 | 1.37 | 0.365 | 0.231 | 8.02 | 6.92 | 9.05 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +10 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.77 | 1.37 | 0.365 | 0.231 | 8.07 | 7.04 | 9.04 | 13.6 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |

Per plate (top-sticker RMS, combined lens) and sticker RMS with the main / edges-only lens fixed; sticker-only fits (no fold barrier; pp fixed = image centre):

| angle | 80/X0 | 80/X1600 | 310/X0 | 310/X1600 | RMS main lens | RMS edges lens | only pp fixed: f / k1 / k2 / RMS | only pp free: f / cx / cy / RMS / fold | only pp free + barrier: f / cy / RMS |
|---|---|---|---|---|---|---|---|---|---|
| -10 | 5.87 | 8.33 | 5.66 | 7.40 | 8.03 | 7.41 | 1303 / -0.079 / -0.226 / 5.56 | 1313 / 938 / 491 / 5.44 / FOLDED (-0.17) | 1314 / 292 / 5.35 |
| -8 | 5.70 | 8.16 | 5.49 | 7.29 | 7.97 | 7.34 | 1302 / -0.082 / -0.221 / 5.49 | 1312 / 938 / 489 / 5.36 / FOLDED (-0.18) | 1312 / 291 / 5.28 |
| -6 | 5.58 | 8.03 | 5.38 | 7.23 | 7.92 | 7.29 | 1302 / -0.086 / -0.216 / 5.45 | 1311 / 938 / 488 / 5.31 / FOLDED (-0.18) | 1310 / 289 / 5.24 |
| -4 | 5.51 | 7.94 | 5.33 | 7.19 | 7.89 | 7.26 | 1301 / -0.089 / -0.212 / 5.45 | 1310 / 938 / 487 / 5.30 / FOLDED (-0.18) | 1309 / 288 / 5.24 |
| -2 | 5.48 | 7.89 | 5.34 | 7.19 | 7.88 | 7.25 | 1301 / -0.092 / -0.207 / 5.46 | 1309 / 938 / 486 / 5.31 / FOLDED (-0.18) | 1309 / 287 / 5.28 |
| +0 | 5.49 | 7.88 | 5.40 | 7.21 | 7.89 | 7.25 | 1300 / -0.095 / -0.203 / 5.50 | 1309 / 938 / 485 / 5.34 / FOLDED (-0.18) | 1309 / 287 / 5.34 |
| +2 | 5.52 | 7.91 | 5.51 | 7.25 | 7.90 | 7.27 | 1300 / -0.098 / -0.199 / 5.56 | 1308 / 938 / 483 / 5.40 / FOLDED (-0.18) | 1330 / 418 / 5.67 |
| +4 | 5.58 | 7.97 | 5.66 | 7.30 | 7.93 | 7.31 | 1300 / -0.101 / -0.196 / 5.63 | 1308 / 938 / 482 / 5.47 / FOLDED (-0.18) | 1330 / 419 / 5.74 |
| +6 | 5.66 | 8.05 | 5.84 | 7.36 | 7.97 | 7.35 | 1300 / -0.104 / -0.192 / 5.72 | 1308 / 938 / 481 / 5.55 / FOLDED (-0.18) | 1330 / 421 / 5.82 |
| +8 | 5.74 | 8.17 | 6.05 | 7.43 | 8.02 | 7.41 | 1300 / -0.106 / -0.189 / 5.81 | 1308 / 938 / 479 / 5.64 / FOLDED (-0.19) | 1330 / 421 / 5.92 |
| +10 | 5.82 | 8.31 | 6.28 | 7.49 | 8.07 | 7.47 | 1300 / -0.109 / -0.186 / 5.92 | 1308 / 938 / 478 / 5.75 / FOLDED (-0.19) | 1330 / 422 / 6.01 |

## Y_hinge

(ext.) = beyond the requested range, added because the metrics were still falling at the range edge.

Combined fit (main estimator) with the tilted top stickers; sticker RMS = radial, combined lens fixed, LS pose per cart; dmap = rotation-compensated median |displacement| vs the main result (centre / cart band / corners).

| angle [deg] | f | cx | cy | k1 | k2 | sig M | sig L | sig V | sig S | RMS all | RMS top | RMS shelf | max all | dmap vs main c/b/k [px] | dmap vs edges-only band | fold margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -10 | 1469.1 | 928.2 | 503.2 | -0.3517 | 0.0995 | 12.26 | 1.36 | 0.365 | 0.231 | 9.28 | 8.15 | 10.35 | 16.1 | 0.0 / 0.1 / 0.2 | 10.3 | 9.99 |
| -8 | 1469.0 | 928.2 | 503.2 | -0.3517 | 0.0994 | 11.93 | 1.36 | 0.365 | 0.231 | 8.96 | 7.77 | 10.07 | 15.6 | 0.0 / 0.1 / 0.2 | 10.3 | 9.99 |
| -6 | 1469.0 | 928.2 | 503.2 | -0.3517 | 0.0995 | 11.61 | 1.36 | 0.365 | 0.231 | 8.66 | 7.42 | 9.81 | 15.1 | 0.0 / 0.1 / 0.1 | 10.3 | 9.99 |
| -4 | 1468.9 | 928.2 | 503.2 | -0.3517 | 0.0994 | 11.30 | 1.36 | 0.365 | 0.231 | 8.38 | 7.11 | 9.55 | 14.6 | 0.0 / 0.1 / 0.1 | 10.3 | 9.99 |
| -2 | 1468.7 | 928.2 | 503.2 | -0.3516 | 0.0994 | 11.00 | 1.36 | 0.365 | 0.231 | 8.12 | 6.83 | 9.31 | 14.1 | 0.0 / 0.0 / 0.1 | 10.2 | 9.99 |
| +0 | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 1.37 | 0.365 | 0.231 | 7.89 | 6.58 | 9.07 | 13.7 | 0.0 / 0.0 / 0.0 | 10.2 | 9.99 |
| +2 | 1468.5 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.45 | 1.37 | 0.365 | 0.231 | 7.67 | 6.38 | 8.84 | 13.2 | 0.0 / 0.0 / 0.1 | 10.2 | 9.99 |
| +4 | 1468.4 | 928.2 | 503.2 | -0.3515 | 0.0993 | 10.20 | 1.37 | 0.365 | 0.231 | 7.48 | 6.21 | 8.63 | 12.8 | 0.0 / 0.1 / 0.1 | 10.1 | 9.99 |
| +6 | 1468.3 | 928.2 | 503.2 | -0.3514 | 0.0993 | 9.96 | 1.37 | 0.365 | 0.231 | 7.31 | 6.08 | 8.42 | 12.4 | 0.0 / 0.1 / 0.2 | 10.1 | 9.99 |
| +8 | 1468.1 | 928.2 | 503.2 | -0.3513 | 0.0993 | 9.73 | 1.38 | 0.365 | 0.231 | 7.16 | 5.99 | 8.23 | 12.0 | 0.0 / 0.2 / 0.2 | 10.1 | 9.99 |
| +10 | 1468.0 | 928.2 | 503.2 | -0.3513 | 0.0992 | 9.53 | 1.38 | 0.365 | 0.231 | 7.04 | 5.93 | 8.05 | 11.7 | 0.0 / 0.2 / 0.3 | 10.0 | 9.99 |
| +12 (ext.) | 1467.9 | 928.2 | 503.2 | -0.3512 | 0.0992 | 9.33 | 1.38 | 0.365 | 0.231 | 6.94 | 5.92 | 7.88 | 11.4 | 0.1 / 0.2 / 0.3 | 10.0 | 9.99 |
| +15 (ext.) | 1467.7 | 928.2 | 503.2 | -0.3511 | 0.0991 | 9.08 | 1.39 | 0.365 | 0.231 | 6.83 | 5.96 | 7.66 | 11.6 | 0.1 / 0.3 / 0.4 | 9.9 | 9.99 |
| +20 (ext.) | 1467.4 | 928.2 | 503.1 | -0.3510 | 0.0990 | 8.77 | 1.39 | 0.365 | 0.231 | 6.79 | 6.21 | 7.36 | 12.0 | 0.1 / 0.4 / 0.5 | 9.8 | 9.99 |
| +30 (ext.) | 1467.2 | 928.1 | 503.0 | -0.3508 | 0.0989 | 8.60 | 1.40 | 0.365 | 0.231 | 7.19 | 7.28 | 7.10 | 12.7 | 0.1 / 0.4 / 0.6 | 9.8 | 9.99 |

Per plate (top-sticker RMS, combined lens) and sticker RMS with the main / edges-only lens fixed; sticker-only fits (no fold barrier; pp fixed = image centre):

| angle | 80/X0 | 80/X1600 | 310/X0 | 310/X1600 | RMS main lens | RMS edges lens | only pp fixed: f / k1 / k2 / RMS | only pp free: f / cx / cy / RMS / fold | only pp free + barrier: f / cy / RMS |
|---|---|---|---|---|---|---|---|---|---|
| -10 | 6.91 | 10.08 | 7.63 | 7.63 | 9.27 | 8.68 | 1288 / -0.051 / -0.257 / 6.79 | 1300 / 937 / 476 / 6.59 / FOLDED (-0.19) | 1303 / 283 / 6.50 |
| -8 | 6.59 | 9.56 | 7.11 | 7.49 | 8.95 | 8.35 | 1290 / -0.060 / -0.246 / 6.48 | 1302 / 938 / 478 / 6.30 / FOLDED (-0.19) | 1304 / 283 / 6.22 |
| -6 | 6.29 | 9.08 | 6.63 | 7.38 | 8.65 | 8.05 | 1293 / -0.069 / -0.236 / 6.20 | 1304 / 938 / 480 / 6.03 / FOLDED (-0.19) | 1333 / 413 / 6.29 |
| -4 | 6.00 | 8.64 | 6.18 | 7.29 | 8.38 | 7.76 | 1296 / -0.078 / -0.225 / 5.94 | 1306 / 938 / 482 / 5.78 / FOLDED (-0.19) | 1324 / 420 / 6.05 |
| -2 | 5.74 | 8.24 | 5.77 | 7.24 | 8.12 | 7.50 | 1298 / -0.087 / -0.214 / 5.71 | 1307 / 938 / 484 / 5.55 / FOLDED (-0.18) | 1308 / 286 / 5.52 |
| +0 | 5.49 | 7.88 | 5.40 | 7.21 | 7.89 | 7.25 | 1300 / -0.095 / -0.203 / 5.50 | 1309 / 938 / 485 / 5.34 / FOLDED (-0.18) | 1309 / 287 / 5.34 |
| +2 | 5.26 | 7.57 | 5.08 | 7.21 | 7.67 | 7.03 | 1302 / -0.104 / -0.192 / 5.32 | 1310 / 938 / 485 / 5.16 / FOLDED (-0.18) | 1310 / 288 / 5.18 |
| +4 | 5.06 | 7.30 | 4.80 | 7.23 | 7.48 | 6.84 | 1304 / -0.113 / -0.181 / 5.16 | 1312 / 938 / 485 / 5.01 / FOLDED (-0.18) | 1311 / 289 / 5.04 |
| +6 | 4.88 | 7.08 | 4.58 | 7.27 | 7.31 | 6.66 | 1306 / -0.121 / -0.170 / 5.03 | 1313 / 938 / 484 / 4.87 / FOLDED (-0.17) | 1338 / 426 / 5.13 |
| +8 | 4.74 | 6.90 | 4.41 | 7.34 | 7.17 | 6.51 | 1308 / -0.130 / -0.159 / 4.93 | 1314 / 937 / 483 / 4.77 / FOLDED (-0.17) | 1333 / 429 / 4.99 |
| +10 | 4.64 | 6.76 | 4.29 | 7.42 | 7.05 | 6.39 | 1309 / -0.138 / -0.148 / 4.85 | 1315 / 937 / 481 / 4.68 / FOLDED (-0.17) | 1341 / 450 / 4.93 |
| +12 (ext.) | 4.59 | 6.67 | 4.22 | 7.53 | 6.95 | 6.29 | 1310 / -0.147 / -0.137 / 4.80 | 1316 / 937 / 478 / 4.62 / FOLDED (-0.17) | 1337 / 417 / 4.86 |
| +15 (ext.) | 4.60 | 6.60 | 4.20 | 7.74 | 6.85 | 6.19 | 1312 / -0.159 / -0.121 / 4.78 | 1317 / 936 / 473 / 4.58 / FOLDED (-0.17) | 1334 / 437 / 4.78 |
| +20 (ext.) | 4.89 | 6.66 | 4.37 | 8.19 | 6.82 | 6.15 | 1314 / -0.179 / -0.094 / 4.88 | 1318 / 936 / 461 / 4.63 / FOLDED (-0.18) | 1332 / 413 / 4.78 |
| +30 (ext.) | 6.51 | 7.22 | 5.17 | 9.52 | 7.22 | 6.59 | 1313 / -0.216 / -0.041 / 5.54 | 1312 / 937 / 434 / 5.15 / FOLDED (-0.19) | 1319 / 382 / 5.23 |

## Best-fitting common angle per family (argmin over the scan; refined = parabola through the 3 grid points around the minimum)

| family | metric | grid argmin [deg] | refined [deg] | value there | value at 0 deg |
|---|---|---|---|---|---|
| Y_centre | M_sigma | +10 | 9.25 | 10.677 | 10.720 |
| Y_centre | sticker_rms_combined_lens | +8 | 8.02 | 7.791 | 7.886 |
| Y_centre | top_rms_combined_lens | +8 | 7.66 | 6.378 | 6.584 |
| Y_centre | sticker_rms_main_lens | +8 | 8.03 | 7.791 | 7.886 |
| Y_centre | sticker_rms_edges_only_lens | +8 | 8.19 | 7.145 | 7.253 |
| Y_centre | sticker_only_ppfixed_rms | +10 | 9.04 | 5.323 | 5.500 |
| Y_centre | sticker_only_ppfree_rms | +8 | 8.65 | 5.183 | 5.343 |
| X_centre | M_sigma | +0 | -0.71 | 10.720 | 10.720 |
| X_centre | sticker_rms_combined_lens | -2 | -1.63 | 7.881 | 7.886 |
| X_centre | top_rms_combined_lens | -2 | -1.94 | 6.567 | 6.584 |
| X_centre | sticker_rms_main_lens | -2 | -1.62 | 7.881 | 7.886 |
| X_centre | sticker_rms_edges_only_lens | -2 | -1.83 | 7.246 | 7.253 |
| X_centre | sticker_only_ppfixed_rms | -4 | -4.25 | 5.445 | 5.500 |
| X_centre | sticker_only_ppfree_rms | -4 | -3.83 | 5.298 | 5.343 |
| Y_hinge | M_sigma | +30 (range edge) | - | 8.603 | 10.720 |
| Y_hinge | sticker_rms_combined_lens | +20 | 18.87 | 6.790 | 7.886 |
| Y_hinge | top_rms_combined_lens | +12 | 11.83 | 5.917 | 6.584 |
| Y_hinge | sticker_rms_main_lens | +20 | 18.70 | 6.815 | 7.886 |
| Y_hinge | sticker_rms_edges_only_lens | +20 | 18.54 | 6.155 | 7.253 |
| Y_hinge | sticker_only_ppfixed_rms | +15 | 14.54 | 4.783 | 5.500 |
| Y_hinge | sticker_only_ppfree_rms | +15 | 15.82 | 4.583 | 5.343 |

Combined fits at the refined argmin angles:

| configuration | chosen by | f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k | dmap vs edges-only band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Y_centre_common_+9.25 | M_sigma | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 7.79 | 6.39 | 0.0 / 0.0 / 0.0 | 10.2 |
| Y_centre_common_+7.66 | top_rms_combined_lens | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.68 | 7.79 | 6.38 | 0.0 / 0.0 / 0.0 | 10.2 |
| X_centre_common_-0.71 | M_sigma | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 7.88 | 6.57 | 0.0 / 0.0 / 0.0 | 10.2 |
| X_centre_common_-1.63 | sticker_rms_combined_lens | 1468.6 | 928.2 | 503.2 | -0.3516 | 0.0994 | 10.72 | 7.88 | 6.57 | 0.0 / 0.0 / 0.0 | 10.2 |
| Y_hinge_common_+18.87 | sticker_rms_combined_lens | 1467.5 | 928.2 | 503.1 | -0.3510 | 0.0991 | 8.82 | 6.79 | 6.14 | 0.1 / 0.4 / 0.5 | 9.8 |

## Per-plate fitted tilts

Tilts fitted to the 62 sticker corners with the lens FIXED (iteration 0: main result) and one LS pose per cart; then the combined fit with these tilts; then alternation (tilts refitted with the new combined lens) until |delta f| < 1 px. ay = Y_centre convention (+ = outer side up), bx = X_centre convention (+ = back side up).

Uncertainty columns: 'formal' = Gauss-Newton covariance scaled by the global residual variance (5.5 px per coordinate, dominated by the top-vs-shelf height mismatch, not by the sticker shapes, which are clean squares to ~0.2 px) - an upper bound; jackknife = delete one of the 20 stickers; bootstrap = stratified cluster bootstrap (top stickers resampled within their plate, shelf stickers within their cart); lens-propagated = refit with the 100 bootstrap lenses of the main estimator; total = max(jackknife, bootstrap) (+) lens; definition half-range = half the range of the tilt over 6 definitions of the cart reference (section 'Reference-definition systematic' below); total incl. definition = total (+) definition half-range = the 1-sigma to quote [review fix: the original quoted 'total' without the definition term].

For comparison (work/cache/geometry_diagnosis.md, sect. 5, per-sticker free pose, f = 1400, plumb distortion; converted to this sign convention): 80/X0 +10.5 / +13.4, 80/X1600 +7.2 / +6.2, 310/X0 +11.2 / +13.3, 310/X1600 +3.9 / +4.8 deg (+- 1.4-2.4); the metric board-tilt fits of sect. 3 also gave the opposite sign at 310/X1600. The sect. 5 values use a different lens and cart reference; see 'Reference-definition systematic' below.


### Y (iteration 0, main lens fixed)

| tilt | value [deg] | formal | jackknife (sticker) | bootstrap (stratified) | lens-propagated | total | definition half-range | **total incl. definition** | range over definitions | d tilt / 100 px f |
|---|---|---|---|---|---|---|---|---|---|---|
| ay_80/X0 | +12.35 | 8.92 | 1.44 | 0.94 | 0.31 | 1.47 | 0.97 | **1.77** | +11.4..+13.3 | -0.57 |
| ay_80/X1600 | +6.11 | 8.68 | 1.42 | 0.90 | 0.42 | 1.48 | 1.46 | **2.08** | +4.6..+7.5 | -0.88 |
| ay_310/X0 | +15.74 | 11.15 | 1.28 | 0.89 | 0.36 | 1.33 | 3.08 | **3.35** | +14.9..+21.0 | -0.63 |
| ay_310/X1600 | -4.71 | 9.88 | 2.33 | 1.32 | 0.84 | 2.48 | 1.24 | **2.77** | -5.3..-2.8 | +1.84 |

Sticker RMS with the main lens: 7.72 px (top 6.21, shelf 9.07; plates 80/X0 4.97, 80/X1600 7.69, 310/X0 4.36, 310/X1600 7.15) vs 7.89 px flat.

| iteration | lens used for the tilts (f) | tilts [deg] | combined f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k | dmap vs edges band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1468.6 | +12.3, +6.1, +15.7, -4.7 | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.72 | 6.20 | 0.0 / 0.0 / 0.0 | 10.2 |
| 1 | 1468.6 | +12.3, +6.1, +15.7, -4.7 | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.72 | 6.20 | 0.0 / 0.0 / 0.0 | 10.2 |

Sticker-only fits with the final tilts: pp fixed f 1294, k1 -0.104, k2 -0.189, RMS 5.23; pp free f 1302, pp (933, 489), RMS 5.06, FOLDED (margin -0.18).


### YX (iteration 0, main lens fixed)

| tilt | value [deg] | formal | jackknife (sticker) | bootstrap (stratified) | lens-propagated | total | definition half-range | **total incl. definition** | range over definitions | d tilt / 100 px f |
|---|---|---|---|---|---|---|---|---|---|---|
| ay_80/X0 | +12.20 | 9.75 | 1.65 | 0.74 | 0.36 | 1.69 | 1.09 | **2.01** | +11.6..+13.8 | +0.68 |
| ay_80/X1600 | +6.11 | 8.87 | 1.59 | 1.02 | 0.44 | 1.65 | 1.35 | **2.13** | +4.6..+7.3 | -0.90 |
| ay_310/X0 | +14.32 | 10.45 | 1.05 | 0.72 | 0.50 | 1.16 | 2.60 | **2.85** | +13.3..+18.5 | -1.03 |
| ay_310/X1600 | -4.18 | 9.35 | 2.78 | 1.77 | 0.57 | 2.84 | 0.74 | **2.93** | -4.2..-2.7 | +1.03 |
| bx_80/X0 | -0.30 | 9.95 | 2.94 | 1.39 | 1.03 | 3.11 | 3.40 | **4.61** | -2.0..+4.8 | +2.39 |
| bx_80/X1600 | +0.05 | 7.77 | 2.00 | 1.23 | 0.20 | 2.01 | 0.74 | **2.14** | -1.4..+0.1 | -0.16 |
| bx_310/X0 | -3.32 | 7.44 | 1.19 | 0.72 | 0.39 | 1.26 | 1.22 | **1.75** | -4.5..-2.0 | -0.92 |
| bx_310/X1600 | -2.63 | 9.57 | 4.45 | 2.61 | 0.73 | 4.51 | 3.92 | **5.98** | -5.5..+2.4 | +1.90 |

Sticker RMS with the main lens: 7.71 px (top 6.18, shelf 9.07; plates 80/X0 4.97, 80/X1600 7.69, 310/X0 4.24, 310/X1600 7.12) vs 7.89 px flat.

| iteration | lens used for the tilts (f) | tilts [deg] | combined f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k | dmap vs edges band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1468.6 | +12.2, +6.1, +14.3, -4.2, -0.3, +0.1, -3.3, -2.6 | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.71 | 6.18 | 0.0 / 0.0 / 0.0 | 10.2 |
| 1 | 1468.6 | +12.2, +6.1, +14.3, -4.2, -0.3, +0.1, -3.3, -2.6 | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.71 | 6.18 | 0.0 / 0.0 / 0.0 | 10.2 |

Sticker-only fits with the final tilts: pp fixed f 1294, k1 -0.102, k2 -0.192, RMS 5.20; pp free f 1303, pp (933, 489), RMS 5.03, FOLDED (margin -0.18).


## Joint fit: plate tilts as free parameters inside the combined estimator

| kind | tilts [deg] (formal sigma) | f | cx | cy | k1 | k2 | sig M | RMS all (LS poses) | RMS top | dmap vs main band | dmap vs edges band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Y | ay 80/X0 +11.7 (18.3), ay 80/X1600 +7.5 (17.6), ay 310/X0 +21.0 (24.3), ay 310/X1600 -3.0 (19.8) | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.73 | 6.23 | 0.0 | 10.2 |
| YX | ay 80/X0 +13.8 (23.8), ay 80/X1600 +7.3 (17.3), ay 310/X0 +18.5 (23.4), ay 310/X1600 -3.2 (21.9), bx 80/X0 +4.8 (23.3), bx 80/X1600 -1.2 (14.6), bx 310/X0 -2.0 (14.0), bx 310/X1600 +2.4 (22.6) | 1468.6 | 928.2 | 503.2 | -0.3515 | 0.0994 | 10.65 | 7.73 | 6.25 | 0.0 | 10.2 |

## Reference-definition systematic of the per-plate tilts [review fix]

A tilt is only defined relative to the cart frame, i.e. relative to an estimated cart pose. With the main lens fixed the tilts were refitted under six definitions of that reference: A = LS pose per cart from all 62 corners (the values above); B = A + a free height offset per plate (diagnostic only: stops the top-vs-shelf height mismatch from bending the pose; the heights are NOT free in any lens fit); C = shape-only (every top sticker gets a free 3D position, so only its shape carries the tilt; pose = LS from the shelf stickers + the top-sticker shapes); D = shape-only relative to the cart rotation of the main combined fit (edge-constrained; only the 32 top-sticker corners); E = as D with the edges-only lens (f 1436.4) and its cart rotation; J = joint fit inside the combined estimator. Half the range over A-E, J is added in quadrature to the statistical total.


### Y

| definition | ay_80/X0 | ay_80/X1600 | ay_310/X0 | ay_310/X1600 | RMS [px] (corners used) |
|---|---|---|---|---|---|
| A: LS pose per cart from all 62 corners (reported value) | +12.35 | +6.11 | +15.74 | -4.71 | 7.72 (62) |
| B: A + free height offset per plate (diagnostic) (dz +81, +71, +56, +86 mm) | +13.32 | +4.61 | +14.87 | -5.26 | 3.00 (62) |
| C: shape-only, free top-sticker positions, pose from shelf stickers + top shapes | +11.38 | +6.51 | +17.29 | -2.88 | 1.65 (62) |
| D: shape-only relative to the cart rotation of the main combined fit | +11.50 | +6.21 | +17.02 | -2.79 | 0.80 (32) |
| E: as D with the edges-only lens and its cart rotation | +11.56 | +6.17 | +16.96 | -3.19 | 0.77 (32) |
| J: joint fit inside the combined estimator | +11.72 | +7.53 | +21.02 | -3.01 | - |
| half range | 0.97 | 1.46 | 3.08 | 1.24 | |
| 1-sigma without / **incl.** definition | 1.47 / **1.77** | 1.48 / **2.08** | 1.33 / **3.35** | 2.48 / **2.77** | |

Lens variants (information only, not in the +-; A / C definitions):

| lens | ay_80/X0 | ay_80/X1600 | ay_310/X0 | ay_310/X1600 |
|---|---|---|---|---|
| pp at the image centre (f main) | +11.5 / +10.3 | +5.4 / +6.0 | +15.7 / +17.6 | -3.3 / -2.0 |
| shape-diagnostic lens f 1300 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +12.6 / +9.7 | +7.6 / +7.9 | +17.7 / +18.9 | -6.1 / -3.0 |
| shape-diagnostic lens f 1400 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +12.1 / +10.5 | +6.5 / +6.9 | +16.6 / +18.4 | -3.4 / -1.8 |
| shape-diagnostic lens f 1500 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +11.5 / +11.1 | +5.5 / +6.0 | +16.0 / +18.0 | -1.8 / -0.9 |

### YX

| definition | ay_80/X0 | ay_80/X1600 | ay_310/X0 | ay_310/X1600 | bx_80/X0 | bx_80/X1600 | bx_310/X0 | bx_310/X1600 | RMS [px] (corners used) |
|---|---|---|---|---|---|---|---|---|---|
| A: LS pose per cart from all 62 corners (reported value) | +12.20 | +6.11 | +14.32 | -4.18 | -0.30 | +0.05 | -3.32 | -2.63 | 7.71 (62) |
| B: A + free height offset per plate (diagnostic) (dz +82, +71, +56, +87 mm) | +12.28 | +4.57 | +13.30 | -3.82 | -2.04 | -1.43 | -4.49 | -5.47 | 2.93 (62) |
| C: shape-only, free top-sticker positions, pose from shelf stickers + top shapes | +11.59 | +6.30 | +14.86 | -2.80 | +0.87 | -0.57 | -3.03 | +1.54 | 1.63 (62) |
| D: shape-only relative to the cart rotation of the main combined fit | +11.79 | +5.89 | +14.90 | -2.70 | +1.13 | -0.90 | -2.66 | +1.59 | 0.71 (32) |
| E: as D with the edges-only lens and its cart rotation | +12.00 | +5.74 | +15.17 | -3.11 | +1.54 | -1.23 | -2.27 | +1.94 | 0.68 (32) |
| J: joint fit inside the combined estimator | +13.77 | +7.27 | +18.50 | -3.23 | +4.77 | -1.22 | -2.05 | +2.38 | - |
| half range | 1.09 | 1.35 | 2.60 | 0.74 | 3.40 | 0.74 | 1.22 | 3.92 | |
| 1-sigma without / **incl.** definition | 1.69 / **2.01** | 1.65 / **2.13** | 1.16 / **2.85** | 2.84 / **2.93** | 3.11 / **4.61** | 2.01 / **2.14** | 1.26 / **1.75** | 4.51 / **5.98** | |

Lens variants (information only, not in the +-; A / C definitions):

| lens | ay_80/X0 | ay_80/X1600 | ay_310/X0 | ay_310/X1600 | bx_80/X0 | bx_80/X1600 | bx_310/X0 | bx_310/X1600 |
|---|---|---|---|---|---|---|---|---|
| pp at the image centre (f main) | +10.7 / +10.2 | +5.4 / +5.6 | +14.1 / +14.7 | -3.0 / -1.8 | -1.7 / -0.4 | -0.6 / -1.1 | -3.8 / -3.6 | -2.4 / +1.5 |
| shape-diagnostic lens f 1300 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +9.9 / +10.0 | +7.7 / +7.6 | +16.8 / +16.7 | -4.2 / -2.8 | -5.0 / +1.2 | +0.3 / -0.9 | -1.9 / -2.8 | -5.6 / +2.6 |
| shape-diagnostic lens f 1400 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +10.9 / +10.6 | +6.5 / +6.6 | +15.3 / +15.8 | -2.9 / -1.6 | -2.4 / +0.3 | +0.0 / -0.8 | -3.0 / -3.1 | -3.2 / +2.4 |
| shape-diagnostic lens f 1500 (plumb k1 -0.292, k2 0.0716 at f0 1300, pp centre) | +11.4 / +10.9 | +5.5 / +5.8 | +14.2 / +15.1 | -1.8 / -0.7 | -0.2 / -0.7 | -0.2 / -0.7 | -4.0 / -3.5 | -0.7 / +1.8 |

## Hinge position sensitivity [review fix]

Y_hinge with the hinge 45 mm from the sticker centre (edge of the black code square) instead of 55 mm (sticker edge); centre rise 45 sin(angle) mm.

| angle | f | cx | cy | k1 | k2 | sig M | RMS all | RMS top | dmap vs main c/b/k |
|---|---|---|---|---|---|---|---|---|---|
| +10 | 1468.1 | 928.2 | 503.2 | -0.3513 | 0.0993 | 9.73 | 7.17 | 6.01 | 0.04 / 0.16 / 0.22 |
| +20 | 1467.7 | 928.2 | 503.2 | -0.3511 | 0.0991 | 9.11 | 6.99 | 6.31 | 0.06 / 0.29 / 0.40 |

## Conclusions

* Y_centre (-15..+15 deg): combined f changes by -0.03..+0.18 px, cy by -0.03..+0.00 px, k1 by -0.00005..+0.00001; mapping change vs the main result <= 0.06 px (cart band median) / 0.09 px (corners); M-block sigma 10.68..11.04 px (0 deg: 10.72). Sticker-only fits: pp fixed f 1288..1320, pp free f 1296..1331, RMS 5.18..6.51 px, folded in 19/19 (fold margin -0.18..-0.18).
* X_centre (-10..+10 deg): combined f changes by -0.00..+0.07 px, cy by -0.00..+0.00 px, k1 by -0.00004..+0.00001; mapping change vs the main result <= 0.02 px (cart band median) / 0.03 px (corners); M-block sigma 10.72..10.79 px (0 deg: 10.72). Sticker-only fits: pp fixed f 1300..1303, pp free f 1308..1313, RMS 5.30..5.75 px, folded in 11/11 (fold margin -0.19..-0.17).
* Y_hinge (-10..+10 deg): combined f changes by -0.62..+0.44 px, cy by -0.04..+0.00 px, k1 by -0.00016..+0.00028; mapping change vs the main result <= 0.20 px (cart band median) / 0.27 px (corners); M-block sigma 9.53..12.26 px (0 deg: 10.72). Sticker-only fits: pp fixed f 1288..1309, pp free f 1300..1315, RMS 4.68..6.59 px, folded in 11/11 (fold margin -0.19..-0.17).
* Y_hinge beyond the requested range: +12 deg: f 1467.9, sig M 9.33, RMS all 6.94, sticker-only pp free f 1316 (folded); +15 deg: f 1467.7, sig M 9.08, RMS all 6.83, sticker-only pp free f 1317 (folded); +20 deg: f 1467.4, sig M 8.77, RMS all 6.79, sticker-only pp free f 1318 (folded); +30 deg: f 1467.2, sig M 8.60, RMS all 7.19, sticker-only pp free f 1312 (folded) (centre rise 55 sin(angle) mm = 11.4, 14.2, 18.8, 27.5 mm).
* Why the combined lens hardly moves: with the converged block weights the 62 sticker corners carry little weight (sigma M = 10.7 px vs V 0.36 px on 2103 edge points and S 0.23 px on 1136); removing ALL corners at fixed block weights moves f only from 1468.6 to 1471.1 px (combined_fit.json, k1k2_influence_no_corners). A tilt changes the top corners by a few px only, and the M-block sigma stays far above the detection noise (0.1-0.2 px), so the stickers' weight never rises.
* Y_centre best common angle: M sigma argmin +10 deg (refined 9.25) 10.68 vs 10.72 px at 0; overall sticker RMS argmin +8 (refined 8.02) 7.79 vs 7.89; top-row RMS argmin +8 (refined 7.66) 6.38 vs 6.58; with the edges-only lens fixed: argmin +8, 7.15 vs 7.25 px.
* X_centre best common angle: M sigma argmin +0 deg (refined -0.71) 10.72 vs 10.72 px at 0; overall sticker RMS argmin -2 (refined -1.63) 7.88 vs 7.89; top-row RMS argmin -2 (refined -1.94) 6.57 vs 6.58; with the edges-only lens fixed: argmin -2, 7.25 vs 7.25 px.
* Y_hinge best common angle: M sigma argmin +30 deg (refined -, range edge) 8.60 vs 10.72 px at 0; overall sticker RMS argmin +20 (refined 18.87) 6.79 vs 7.89; top-row RMS argmin +12 (refined 11.83) 5.92 vs 6.58; with the edges-only lens fixed: argmin +20, 6.15 vs 7.25 px.
* Per-plate Y tilts (main lens fixed, definition A): ay_80/X0 +12.3 +- 1.8, ay_80/X1600 +6.1 +- 2.1, ay_310/X0 +15.7 +- 3.4, ay_310/X1600 -4.7 +- 2.8 deg (1-sigma = [max(jackknife, bootstrap) (+) lens propagation] (+) half range over 6 definitions of the cart reference; without the definition term: 1.5, 1.5, 1.3, 2.5); range over the definitions A-E, J: ay_80/X0 +11.4..+13.3, ay_80/X1600 +4.6..+7.5, ay_310/X0 +14.9..+21.0, ay_310/X1600 -5.3..-2.8; sticker RMS 7.72 px (top 6.21) vs 7.89 flat. Combined fit with the tilts (after 2 step(s)): f 1468.6, cx 928.2, cy 503.2, k1 -0.3515, k2 0.0994, sig M 10.65; mapping vs main 0.02 px (cart band).
* Per-plate YX tilts (main lens fixed, definition A): ay_80/X0 +12.2 +- 2.0, ay_80/X1600 +6.1 +- 2.1, ay_310/X0 +14.3 +- 2.8, ay_310/X1600 -4.2 +- 2.9, bx_80/X0 -0.3 +- 4.6, bx_80/X1600 +0.1 +- 2.1, bx_310/X0 -3.3 +- 1.8, bx_310/X1600 -2.6 +- 6.0 deg (1-sigma = [max(jackknife, bootstrap) (+) lens propagation] (+) half range over 6 definitions of the cart reference; without the definition term: 1.7, 1.6, 1.2, 2.8, 3.1, 2.0, 1.3, 4.5); range over the definitions A-E, J: ay_80/X0 +11.6..+13.8, ay_80/X1600 +4.6..+7.3, ay_310/X0 +13.3..+18.5, ay_310/X1600 -4.2..-2.7, bx_80/X0 -2.0..+4.8, bx_80/X1600 -1.4..+0.1, bx_310/X0 -4.5..-2.0, bx_310/X1600 -5.5..+2.4; sticker RMS 7.71 px (top 6.18) vs 7.89 flat. Combined fit with the tilts (after 2 step(s)): f 1468.6, cx 928.2, cy 503.2, k1 -0.3515, k2 0.0994, sig M 10.65; mapping vs main 0.02 px (cart band).
* Joint fit (Y tilts free inside the combined estimator): ay_80/X0 +11.7, ay_80/X1600 +7.5, ay_310/X0 +21.0, ay_310/X1600 -3.0 deg; f 1468.6, cx 928.2, cy 503.2, k1 -0.3515, k2 0.0994, sig M 10.65. Difference joint - LS-pose fit: ay_80/X0 -0.6, ay_80/X1600 +1.4, ay_310/X0 +5.3, ay_310/X1600 +1.7 deg (how the cart pose is defined: edge-constrained vs sticker-only LS).
* Joint fit (YX tilts free inside the combined estimator): ay_80/X0 +13.8, ay_80/X1600 +7.3, ay_310/X0 +18.5, ay_310/X1600 -3.2, bx_80/X0 +4.8, bx_80/X1600 -1.2, bx_310/X0 -2.0, bx_310/X1600 +2.4 deg; f 1468.6, cx 928.2, cy 503.2, k1 -0.3515, k2 0.0994, sig M 10.65. Difference joint - LS-pose fit: ay_80/X0 +1.6, ay_80/X1600 +1.2, ay_310/X0 +4.2, ay_310/X1600 +1.0, bx_80/X0 +5.1, bx_80/X1600 -1.3, bx_310/X0 +1.3, bx_310/X1600 +5.0 deg (how the cart pose is defined: edge-constrained vs sticker-only LS).
* [review fix] The per-plate tilts depend on how the cart reference is defined (A LS pose from all corners, B + free plate heights, C/D/E shape-only with free top-sticker positions, J joint): half the range over these definitions (ay_80/X0 1.0, ay_80/X1600 1.5, ay_310/X0 3.1, ay_310/X1600 1.2 deg) is of the same size as the statistical 1-sigma and is now included in the quoted +-. The shape-only definition D (cart rotation of the main combined fit, only the top-sticker shapes) gives ay_80/X0 +11.5, ay_80/X1600 +6.2, ay_310/X0 +17.0, ay_310/X1600 -2.8 deg. At 310/X1600 all definitions with the main lens give a small NEGATIVE tilt (-5.3..-2.8 deg, outer side down) and so do the lens variants incl. the plumb lens of the earlier shape diagnostic (-6.1..-0.9 deg); the earlier shape diagnostic (geometry_diagnosis.md sect. 5: +3.9 / +4.8 deg at f 1400) used its own cart reference (nadir + sticker-shape rotation, free in-plane yaw per sticker). The sign at 310/X1600 is therefore reference-dependent and only weakly determined; the outer-side-up tilts of the other three plates are robust (all definitions and lens variants >= +4.6 deg).
* [review fix] Hinge at 45 mm (edge of the black code square) instead of 55 mm: +10 deg: f 1468.13, cy 503.22, k1 -0.3513, sig M 9.73, mapping vs main 0.16 / 0.22 px (band / corners); +20 deg: f 1467.69, cy 503.16, k1 -0.3511, sig M 9.11, mapping vs main 0.29 / 0.40 px (band / corners) - the same conclusion as with 55 mm.
* Sticker-only pp free WITH the fold barrier is multimodal: over the scan it lands at f 1303..1356, cy 283..502, k1 -0.464..-0.222 (RMS 4.78..6.68), i.e. the stickers alone do not define a valid lens for any tilt.
* Consistency with the edges: over all 52 tilt configurations the sticker-only fits give f = 1288..1320 (pp fixed) / 1296..1331 (pp free) vs 1436.4 from the edges alone, the pp-free sticker lens folds inside the image in 52/52 cases, and the sticker RMS with the edges-only lens fixed stays at 6.15..8.68 px (flat: 7.25). No tested tilt makes the stickers consistent with the edges; the dominant sticker mismatch (top stickers ~60-80 mm high relative to the shelf stickers, REPORT.md ch. 5) is a height offset that a tilt about the sticker centre cannot produce and the hinge tilt produces only as 55 sin(angle) mm.
