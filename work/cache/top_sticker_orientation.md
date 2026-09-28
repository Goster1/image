# Measured orientation of the stickers (top vs shelf) - what-if analysis, branch B (reviewed)

Script `work/61_top_sticker_orientation_fixed.py` (reviewed, corrected copy of `work/61_top_sticker_orientation.py`: same method and central values, corrected uncertainty budget, added board-edge cross-check - see 'Review changes' at the end); numbers in `work/cache/top_sticker_orientation.json`; plot `results/top_sticker_orientation.png`. The main result is NOT changed by this analysis.

**Method.** Main lens (f 1468.6, pp (928.2, 503.2), k1 -0.3516, k2 0.0994). Every sticker = flat 90 mm square; its pose from its own 4 corners (IPPE_SQUARE on undistorted corners, both solutions LM-refined in the distorted image; the one closer to the cart Z axis kept). Normal expressed in the cart frame with two cart rotations that do not use the top stickers: (a) LS pose from the shelf-sticker corners only, (b) vanishing points of the cart-X edges (shelf lips, rails) and cart-Z edges (posts) of that cart. Headline = mean of (a) and (b).

**Sign conventions.** `pitch_outer` = tilt about the cart Y axis, **positive = outer end of the plate up** (outer = towards X=0 for the X0 end, towards X=1600 for the X1600 end); `roll` = tilt about the cart X axis, **positive = back side (Y=450) up**; `total` = angle between the sticker normal and the cart Z axis; `yaw` = in-plane rotation about +Z vs the nominal rot_k*90 deg. Degrees.

**1-sigma.** `total` = joint Monte Carlo (corner noise + lens drawn from the FINAL lens covariance of lens_result.json (458 draws, 2 folded draws rejected) + shelf-sticker / VP-edge resampling, robust half 16-84 % range) (+) half the difference of the two rotation sources (+) raw-vs-bias-corrected corner difference (+) empirical per-sticker systematic s = 0.75 deg (from the within-plate pitch consistency, divided by ~sqrt(2) for a plate mean).

## Per sticker (4-corner stickers)

| sticker | row | end | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | (a) shelf: pitch / roll | (b) VP: pitch / roll | sigma noise / lens (stat-only bootstrap) / rot.src / extra (pitch) | fit RMS px |
|---|---|---|---|---|---|---|---|---|---|---|
| 80:0 | top | X0 | +12.6 ± 1.3 | +1.4 ± 1.5 | 12.7 ± 1.2 | +0.8 ± 0.3 | +12.28 / +1.49 | +12.91 / +1.35 | 0.42 / 0.71 (0.47) / 0.19 / 0.75 | 0.143 |
| 80:1 | top | X1600 | +6.6 ± 1.1 | +1.1 ± 1.1 | 6.7 ± 1.1 | +0.5 ± 0.4 | +6.92 / +1.10 | +6.29 / +1.03 | 0.46 / 0.59 (0.22) / 0.19 / 0.75 | 0.336 |
| 80:2 | top | X1600 | +5.1 ± 1.1 | -1.4 ± 1.1 | 5.3 ± 1.1 | +0.3 ± 0.4 | +5.40 / -1.41 | +4.78 / -1.48 | 0.40 / 0.65 (0.20) / 0.19 / 0.75 | 0.114 |
| 80:4 | A | X1600 | +0.3 ± 1.1 | -0.2 ± 1.1 | 0.4 ± 0.9 | -0.4 ± 0.3 | +0.63 / -0.12 | +0.01 / -0.20 | 0.50 / 0.44 (0.19) / 0.19 / 0.75 | 0.124 |
| 80:5 | B | X0 | -1.3 ± 1.6 | -0.7 ± 1.4 | 1.5 ± 1.2 | -0.9 ± 0.3 | -1.62 / -0.62 | -1.00 / -0.70 | 0.66 / 1.07 (0.81) / 0.19 / 0.75 | 0.085 |
| 80:6 | B | X1600 | +0.6 ± 1.1 | -0.3 ± 1.0 | 0.7 ± 1.0 | +1.6 ± 0.3 | +0.89 / -0.26 | +0.27 / -0.34 | 0.52 / 0.34 (0.12) / 0.19 / 0.75 | 0.214 |
| 80:92 | top | X0 | +14.1 ± 1.5 | -1.3 ± 1.4 | 14.2 ± 1.5 | +1.1 ± 0.4 | +13.81 / -1.23 | +14.42 / -1.38 | 0.27 / 1.18 (0.58) / 0.19 / 0.75 | 0.244 |
| 310:0 | top | X0 | +13.0 ± 1.2 | +1.6 ± 1.2 | 13.1 ± 1.2 | -1.8 ± 0.4 | +12.93 / +1.50 | +13.17 / +1.66 | 0.52 / 0.79 (0.56) / 0.36 / 0.75 | 0.060 |
| 310:1 | top | X1600 | +1.9 ± 1.9 | +2.1 ± 1.7 | 2.9 ± 1.4 | -2.1 ± 0.4 | +1.99 / +2.04 | +1.76 / +2.25 | 0.69 / 1.26 (1.17) / 0.36 / 0.75 | 0.135 |
| 310:2 | top | X1600 | +2.8 ± 1.6 | -0.5 ± 1.5 | 2.8 ± 1.5 | -2.3 ± 0.5 | +2.90 / -0.62 | +2.67 / -0.42 | 0.55 / 0.96 (0.97) / 0.36 / 0.75 | 0.184 |
| 310:6 | B | X1600 | +2.9 ± 2.0 | -1.6 ± 2.2 | 3.3 ± 1.9 | -1.2 ± 0.4 | +2.99 / -1.70 | +2.76 / -1.49 | 1.45 / 0.98 (0.80) / 0.36 / 0.75 | 0.352 |
| 310:322 | top | X0 | +14.3 ± 1.3 | -2.8 ± 1.3 | 14.5 ± 1.3 | -1.0 ± 0.5 | +14.17 / -2.83 | +14.39 / -2.68 | 0.56 / 0.84 (0.53) / 0.35 / 0.75 | 0.110 |

## Per end plate (two top stickers combined)

| plate | stickers (front Y=60, back Y=390) | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | front minus back sticker: pitch / roll (joint MC sigma; incl. extra) |
|---|---|---|---|---|---|---|
| 310/X0 | 310:0, 310:322 | +13.6 ± 1.1 | -0.4 ± 1.0 | 13.8 ± 1.0 | -1.4 ± 0.4 | -1.2 / +4.3 (0.8 / 0.8; 1.3 / 1.4) |
| 310/X1600 | 310:1, 310:2 | +2.4 ± 1.5 | +0.5 ± 1.5 | 2.8 ± 1.2 | -2.2 ± 0.5 | -0.9 / +2.7 (1.1 / 0.9; 1.5 / 1.4) |
| 80/X0 | 80:0, 80:92 | +13.7 ± 1.3 | -0.5 ± 1.3 | 13.7 ± 1.3 | +1.0 ± 0.4 | -1.5 / +2.7 (0.8 / 0.6; 1.4 / 1.2) |
| 80/X1600 | 80:1, 80:2 | +5.7 ± 0.9 | -0.4 ± 0.9 | 5.9 ± 0.9 | +0.4 ± 0.4 | +1.5 / +2.5 (0.6 / 0.7; 1.2 / 1.2) |

## Control: shelf stickers

| sticker | row | end | kind (constraints) | pitch_outer [deg] | roll [deg] | total [deg] | yaw [deg] | usable |
|---|---|---|---|---|---|---|---|---|
| 80:3 | A | X0 | partial_prior (5) | +2.5 ± 7.7 | +29.3 ± 53.3 | 29.4 ± 13.5 | -3.3 ± 1.9 | no (not determined) |
| 80:4 | A | X1600 | full (8) | +0.3 ± 1.1 | -0.2 ± 1.1 | 0.4 ± 0.9 | -0.4 ± 0.3 | yes |
| 80:5 | B | X0 | full (8) | -1.3 ± 1.6 | -0.7 ± 1.4 | 1.5 ± 1.2 | -0.9 ± 0.3 | yes |
| 80:6 | B | X1600 | full (8) | +0.6 ± 1.1 | -0.3 ± 1.0 | 0.7 ± 1.0 | +1.6 ± 0.3 | yes |
| 80:7 | C | X0 | partial_prior (5) | +5.7 ± 10.8 | +41.0 ± 68.9 | 41.2 ± 22.1 | +0.2 ± 0.7 | no (not determined) |
| 80:8 | C | X1600 | partial (6) | -66.2 ± 89.6 | +1.7 ± 38.9 | 66.2 ± 35.8 | +10.8 ± 22.1 | no (not determined) |
| 310:3 | A | X0 | partial (6) | +13.2 ± 49.7 | -35.1 ± 54.3 | 36.5 ± 33.0 | +12.0 ± 22.9 | no (not determined) |
| 310:4 | A | X1600 | partial_prior (5) | +23.7 ± 20.2 | +23.0 ± 9.8 | 31.4 ± 11.6 | -1.0 ± 2.1 | no (not determined) |
| 310:5 | B | X0 | partial_prior (4) | +8.8 ± 9.2 | -16.5 ± 11.2 | 18.5 ± 10.1 | +5.3 ± 3.2 | no (not determined) |
| 310:6 | B | X1600 | full (8) | +2.9 ± 2.0 | -1.6 ± 2.2 | 3.3 ± 1.9 | -1.2 ± 0.4 | yes |
| 310:7 | C | X0 | partial_prior (4) | -0.8 ± 6.4 | -3.1 ± 6.4 | 3.2 ± 4.6 | +1.2 ± 1.4 | no (not determined) |
| 310:8 | C | X1600 | partial (6) | +5.9 ± 4.5 | +52.0 ± 63.7 | 52.1 ± 44.6 | +2.1 ± 2.2 | no (not determined) |

'partial' = 2 corners + traced sides (6 constraints, no prior); 'partial_prior' = fewer than 6 independent constraints, centre tied to the shelf-sticker cart pose (30 mm) -> not independent. The visible parts of the partly hidden stickers (one or two corners + 25-35 px of side) do NOT determine the sticker normal: their solutions scatter by tens of degrees (Monte Carlo sigma above) and several have near-equal alternative minima -> only the four 4-corner shelf stickers (80:4, 80:5, 80:6, 310:6) are used as the control.

## Sanity checks

* Shelf-sticker normals mutually parallel (camera frame, 4-corner stickers, same cart): 80:4-80:5 1.11 deg (joint sigma 0.90); 80:4-80:6 0.29 deg (joint sigma 0.55); 80:5-80:6 0.82 deg (joint sigma 0.90)
* Top minus 4-corner shelf sticker at the same cart end (cart rotation cancels, the lens does NOT fully cancel; total sigma, lens-only part in brackets): 310:1-310:6 pitch -1.0 ± 2.1 (0.5), roll +3.7 ± 1.8; 310:2-310:6 pitch -0.1 ± 2.0 (0.5), roll +1.1 ± 1.8; 80:0-80:5 pitch +13.9 ± 1.5 (0.7), roll +2.1 ± 1.5; 80:92-80:5 pitch +15.4 ± 1.8 (1.2), roll -0.6 ± 1.4; 80:1-80:4 pitch +6.3 ± 1.3 (0.2), roll +1.2 ± 1.3; 80:2-80:4 pitch +4.8 ± 1.3 (0.2), roll -1.3 ± 1.3; 80:1-80:6 pitch +6.0 ± 1.3 (0.2), roll +1.4 ± 1.3; 80:2-80:6 pitch +4.5 ± 1.3 (0.3), roll -1.1 ± 1.3
* IPPE ambiguity (4-corner stickers): chosen RMS / other solution RMS [px] and its angle to cart Z: 80:0 0.143/0.467 (38 deg); 80:1 0.336/0.708 (67 deg); 80:2 0.114/0.255 (76 deg); 80:4 0.124/0.311 (66 deg); 80:5 0.085/0.250 (30 deg); 80:6 0.214/0.056 (58 deg); 80:92 0.244/0.684 (56 deg); 310:0 0.060/0.298 (55 deg); 310:1 0.135/0.254 (35 deg); 310:2 0.184/0.494 (52 deg); 310:6 0.352/0.412 (29 deg); 310:322 0.110/0.300 (65 deg)
* Scale (PnP with the 90 mm code): distance between the two sticker centres of a plate vs the drawing 330 mm: 310/X0 338.4 mm (bias-corr. 326.4; implied code 87.8 / 91.0 mm), chord roll -0.6 ± 1.2 deg; 310/X1600 358.9 mm (bias-corr. 334.2; implied code 82.8 / 88.9 mm), chord roll -5.1 ± 1.6 deg; 80/X0 328.6 mm (bias-corr. 324.9; implied code 90.4 / 91.4 mm), chord roll +4.1 ± 2.6 deg; 80/X1600 335.6 mm (bias-corr. 327.7; implied code 88.5 / 90.6 mm), chord roll +0.5 ± 1.6 deg
* Implied sticker centre (PnP depth) in the shelf-sticker cart frame minus the drawing [mm] (dX, dY, dZ; bias-corrected dZ): 80:0 (-8, +22, +16; +64); 80:1 (+20, +14, +35; +79); 80:2 (+18, +20, +38; +84); 80:4 (+12, +6, -18; +35); 80:5 (+1, +1, -28; +34); 80:6 (+21, +10, -46; +27); 80:92 (-12, +19, +40; +80); 310:0 (-33, +7, +18; +71); 310:1 (+5, +26, +1; +73); 310:2 (+23, +53, -32; +65); 310:4 (+1, +1, +0; +0); 310:5 (-2, +0, +1; +2); 310:6 (+5, +18, -77; +2); 310:322 (-24, +15, +14; +75)
* Synthetic round trip (main lens + main poses, top stickers tilted by a known angle, corner noise as in the MC, 40 reps): truth pitch +0.0 / roll +0.0 -> noise-free top pitch -0.03..+0.03, roll -0.02..+0.01; noisy mean error -0.04 / +0.02, RMS error 0.45 / 0.46 deg; truth pitch +5.0 / roll +2.0 -> noise-free top pitch +4.97..+5.03, roll +1.98..+2.01; noisy mean error -0.04 / +0.03, RMS error 0.51 / 0.55 deg; truth pitch +14.0 / roll -1.5 -> noise-free top pitch +13.97..+14.03, roll -1.52..-1.49; noisy mean error -0.04 / +0.04, RMS error 0.47 / 0.55 deg; truth pitch -8.0 / roll +4.0 -> noise-free top pitch -8.03..-7.97, roll +3.98..+4.01; noisy mean error -0.04 / +0.01, RMS error 0.46 / 0.41 deg (signs and magnitudes recovered; residual offsets of the VP source come from its ~0.05 deg difference to the main-fit pose).
* Cart rotation sources: {'80': {'shelf_vs_vp_deg_about_XYZ': [0.08874664208797554, -0.6192798561486321, -0.2257040104719486], 'main_vs_vp_deg_about_XYZ': [0.03362869146716686, -0.05162571139745411, -0.016209680479244485]}, '310': {'shelf_vs_vp_deg_about_XYZ': [-0.19619691265288747, -0.2315208333673488, -0.15852163828647178], 'main_vs_vp_deg_about_XYZ': [-0.01856931929947054, -0.03934087152983796, -0.00909888815546523]}}; shelf-pose RMS {'80': 2.5227357504232915, '310': 1.6497877122853934} px, VP RMS 0.405 px.
* Frame-to-frame (7 stills) std of pitch_outer vs the corner-noise MC sigma: 80:0 0.16/0.42, 80:1 0.16/0.46, 80:2 0.28/0.40, 80:4 0.24/0.50, 80:5 0.34/0.66, 80:6 0.28/0.52, 80:92 0.08/0.27, 310:0 0.36/0.52, 310:1 0.34/0.69, 310:2 0.46/0.55, 310:6 0.81/1.45, 310:322 0.47/0.56
  (the corner-noise MC uses the single-still corner std per coordinate, i.e. it is conservative: the 7-still mean scatters ~sqrt(7) less than one still; detection systematics are covered by the raw-vs-bias-corrected term)
* 310/X0 has no 4-corner shelf sticker at its end (310:3, 310:5, 310:7 are partly hidden), so its tilt rests on the cart rotation (shelf pose / VPs) only; 80/X0 is additionally checked against 80:5 (same image region).

## Dependence on f

Plate pitch_outer / roll [deg]; (i) cx, cy, k1, k2 and the poses refitted by the combined estimator at fixed f; (ii) distortion fixed in pixel units.

| mode | f | 310/X0 | 310/X1600 | 80/X0 | 80/X1600 |
|---|---|---|---|---|---|
| refit (pp 930,500; k1 -0.332) | 1420 | +14.3 / -0.3 | +1.8 / +1.2 | +13.5 / +0.0 | +6.0 / -0.6 |
| refit (pp 928,503; k1 -0.352) | 1469 | +13.6 / -0.4 | +2.4 / +0.5 | +13.7 / -0.5 | +5.7 / -0.4 |
| refit (pp 925,502; k1 -0.371) | 1520 | +13.0 / -0.7 | +2.8 / -0.1 | +13.7 / -0.9 | +5.6 / -0.1 |
| pixel | 1300 | +15.6 / -0.3 | -1.2 / +2.3 | +12.2 / +1.4 | +6.2 / -1.7 |
| pixel | 1400 | +14.4 / -0.3 | +1.1 / +1.3 | +13.3 / +0.2 | +5.9 / -0.9 |
| pixel | 1420 | +14.1 / -0.4 | +1.5 / +1.1 | +13.4 / +0.0 | +5.8 / -0.7 |
| pixel | 1469 | +13.6 / -0.4 | +2.4 / +0.5 | +13.7 / -0.5 | +5.7 / -0.4 |
| pixel | 1520 | +13.1 / -0.5 | +3.3 / -0.1 | +13.9 / -1.0 | +5.6 / -0.0 |
| pixel | 1600 | +12.3 / -0.6 | +4.5 / -1.2 | +14.0 / -1.8 | +5.5 / +0.5 |

Slope (pixel mode, 1400-1550): 310/X0 pitch -1.06, roll -0.13 deg / 100 px; 310/X1600 pitch +1.80, roll -1.24 deg / 100 px; 80/X0 pitch +0.46, roll -1.04 deg / 100 px; 80/X1600 pitch -0.20, roll +0.70 deg / 100 px

## Other lens models (nominal data; plate pitch_outer / roll [deg]; control = mean |pitch|, |roll| of the 4-corner shelf stickers)

| lens | f | pp | k1, k2 | 310/X0 | 310/X1600 | 80/X0 | 80/X1600 | control |
|---|---|---|---|---|---|---|---|---|
| main (lens_result.json) | 1469 | (928, 503) | -0.352, 0.099 | +13.6 / -0.4 | +2.4 / +0.5 | +13.7 / -0.5 | +5.7 / -0.4 | - |
| combined_k1k2k3 | 1470 | (930, 505) | -0.364, 0.132 | +13.7 / -0.3 | +2.6 / +0.5 | +13.5 / -0.6 | +5.7 / -0.3 | 1.3 / 0.7 |
| alt_division_l1l2 | 1431 | (935, 508) | -0.356, 0.138 | +14.3 / +0.0 | +2.4 / +0.9 | +13.3 / -0.3 | +5.9 / -0.5 | 1.3 / 0.7 |
| combined_k1k2_rowblocks | 1473 | (929, 504) | -0.354, 0.101 | +13.6 / -0.4 | +2.6 / +0.4 | +13.7 / -0.6 | +5.7 / -0.3 | 1.3 / 0.7 |
| combined_k1k2_robust | 1476 | (930, 505) | -0.357, 0.103 | +13.6 / -0.4 | +2.8 / +0.4 | +13.8 / -0.7 | +5.7 / -0.3 | 1.3 / 0.7 |
| combined_k1k2_common_vertical | 1467 | (931, 507) | -0.354, 0.102 | +13.6 / -0.3 | +2.9 / +0.4 | +13.8 / -0.7 | +5.7 / -0.4 | 1.3 / 0.8 |
| lines_joint | 1473 | (931, 438) | -0.349, 0.095 | +14.7 / -0.0 | +0.8 / +2.4 | +12.3 / +1.5 | +7.0 / +0.2 | 2.0 / 0.4 |
| lines_joint_indep_mergedfloor | 1479 | (939, 509) | -0.362, 0.109 | +13.5 / -0.2 | +3.5 / +0.3 | +13.6 / -1.0 | +5.7 / -0.4 | 1.4 / 0.9 |
| alt_model_brown_k1k2 | 1434 | (927, 502) | -0.336, 0.091 | +14.0 / -0.4 | +1.7 / +1.0 | +13.5 / -0.1 | +5.8 / -0.6 | 1.3 / 0.6 |
| combined_k1k2p1p2 | 1398 | (946, 453) | -0.314, 0.079 | +13.1 / -1.4 | +1.1 / -0.8 | +14.1 / -0.5 | +5.2 / -0.4 | 0.7 / 0.9 |
| combined_k1k2_ppfixed | 1467 | (960, 540) | -0.372, 0.116 | +13.4 / +0.4 | +6.5 / -0.9 | +14.2 / -2.4 | +5.6 / -0.6 | 1.6 / 1.9 |
| lines_joint@lines_joint_lc | 1425 | (927, 503) | -0.333, 0.090 | +14.2 / -0.3 | +1.7 / +1.0 | +13.6 / -0.0 | +5.9 / -0.6 | 1.3 / 0.6 |
| markers_only | 1326 | (960, 540) | -0.246, 0.025 | +14.1 / -1.2 | -0.7 / +1.7 | +10.7 / +0.1 | +5.0 / -3.4 | 1.6 / 2.0 |
| markers_only_k1_ppfix | 1358 | (960, 540) | -0.211, 0.000 | +12.7 / -2.4 | -4.7 / +0.8 | +6.6 / +0.1 | +3.9 / -4.9 | 3.3 / 2.7 |
| sticker_shape | 1601 | (979, 471) | -0.443, 0.165 | +13.6 / +1.8 | +8.5 / +0.5 | +12.6 / -1.5 | +6.9 / +0.3 | 3.0 / 1.5 |
| plumb_line@plumbline_lc | 1419 | (960, 540) | -0.348, 0.102 | +14.0 / +0.4 | +5.4 / +0.0 | +14.0 / -1.8 | +5.7 / -0.9 | 1.5 / 1.7 |
| two_cart_consistency | 1378 | (955, 522) | -0.328, 0.090 | +14.8 / +0.4 | +3.9 / +1.3 | +13.7 / -0.6 | +6.1 / -0.9 | 1.6 / 1.1 |
| vanishing_points@vanishing_lc | 1419 | (941, 513) | -0.348, 0.102 | +14.5 / +0.3 | +4.0 / +0.9 | +14.3 / -0.7 | +6.1 / -0.3 | 1.5 / 0.9 |

## What-if: sticker residuals when the top stickers carry the measured tilt

Main lens fixed, one least-squares pose per cart, drawing positions; top stickers rotated about their own centres by the measured orientation (this is NOT a refit of the lens).

| variant | RMS all [px] | max | top | A | B | C | top-corner shift by the tilt (RMS / max px) |
|---|---|---|---|---|---|---|---|
| flat_drawing | 7.89 | 13.66 | 6.58 | 11.04 | 8.34 | 7.01 | - |
| per_sticker_pitch_roll | 7.73 | 13.61 | 6.24 | 11.03 | 8.32 | 7.02 | 2.14 / 3.88 |
| per_sticker_pitch_roll_yaw | 7.70 | 13.57 | 6.15 | 11.03 | 8.33 | 7.01 | 2.29 / 4.43 |
| plate_mean_pitch_roll | 7.74 | 13.61 | 6.25 | 11.04 | 8.33 | 7.02 | 2.11 / 3.16 |
| plate_mean_pitch_only | 7.74 | 13.61 | 6.25 | 11.04 | 8.32 | 7.02 | 2.10 / 3.10 |
| plate_pitch_hinged_at_inner_edge | 6.99 | 11.60 | 6.00 | 9.19 | 7.65 | 6.20 | 4.42 / 8.83 |
| plate_pitch_hinged_at_outer_edge | 8.67 | 17.31 | 6.69 | 13.08 | 9.09 | 7.93 | 4.34 / 8.59 |

## Uncertainty budget (review)

| plate | pitch_outer | noise | lens: final covariance (stat-only bootstrap) | rot. source MC | joint MC | shelf-vs-VP/2 | raw-vs-biascorr | extra per-sticker | total |
|---|---|---|---|---|---|---|---|---|---|
| 310/X0 | +13.62 | 0.39 | 0.80 (0.54) | 0.36 | 0.91 | 0.12 | 0.01 | 0.53 | 1.06 |
| 310/X1600 | +2.44 | 0.42 | 1.07 (1.08) | 0.36 | 1.42 | 0.12 | 0.02 | 0.55 | 1.52 |
| 80/X0 | +13.67 | 0.22 | 1.06 (0.53) | 0.19 | 1.14 | 0.31 | 0.09 | 0.58 | 1.32 |
| 80/X1600 | +5.74 | 0.30 | 0.62 (0.21) | 0.19 | 0.70 | 0.31 | 0.03 | 0.54 | 0.94 |

Within-plate consistency (rigid plate: both stickers must have the same pitch): chi2 = 12.5 for 4 plates with the Monte Carlo sigma alone -> extra per-sticker systematic s = 0.75 deg (reduced chi2 = 1), applied to every sticker normal. The same test on the roll gives chi2 = 69.3 (s = 2.16 deg if the plates were flat across Y); that part is NOT applied because all four plates show the same sign (possible bow). A uniform corner offset of +-1 px (erosion / dilation, the effect of the bias correction) moves the plate pitch by < 0.2 deg, so the raw-vs-bias-corrected term does not measure the detection systematic; plain ArUco corners instead of the edge corners move single sticker normals by up to ~2.4 deg.


## Cross-check without sticker shapes: board-front edges

The traced front edge of each end board (a cart-X line, 60-110 px long) gives dZ/dX of the board from its interpretation plane and the headline cart rotation. Control: the same angle of the long cart-X edges (shelf lips, rails) is 0.62 deg RMS. The edge's yaw is not measured by the edge itself and enters with 3-4 deg of pitch per deg of yaw.

| plate | edge | length [px] | pitch_outer, yaw 0 ± 2 deg | d pitch / d yaw | sticker yaw used [deg] | pitch_outer with sticker yaw | sticker-shape plate pitch |
|---|---|---|---|---|---|---|---|
| 80/X0 | cart80_x_board0_front | 108 | +9.7 ± 6.2 | -3.04 | +1.02 | +12.8 ± 1.8 | +13.7 |
| 80/X1600 | cart80_x_board1600_front | 74 | +9.3 ± 6.8 | -3.35 | +0.41 | +7.9 ± 2.0 | +5.7 |
| 310/X0 | cart310_x_board0_front | 84 | +20.1 ± 7.7 | -3.79 | -1.43 | +14.7 ± 2.3 | +13.6 |
| 310/X1600 | cart310_x_board1600_front | 59 | -4.7 ± 7.0 | -3.40 | -2.24 | +2.9 ± 2.7 | +2.4 |

## Conclusion

**Answer:** yes - the top stickers are measurably tilted about the cart Y axis (dZ/dX, along the cart length), outer end up, at 310/X0 +13.6 ± 1.1 deg, 80/X0 +13.7 ± 1.3 deg, 80/X1600 +5.7 ± 0.9 deg; not significant at 310/X1600 +2.4 ± 1.5 deg. About the cart X axis no plate is significantly tilted (plate roll -0.4 ± 1.0, +0.5 ± 1.5, -0.5 ± 1.3, -0.4 ± 0.9 deg).

Per end plate (pitch_outer: + = outer end up; roll: + = back side up):
- **310/X0**: pitch +13.6 ± 1.1 deg (+12.8 sigma, f 1420..1520: +13.0..+14.3), roll -0.4 ± 1.0 deg (-0.4 sigma), total 13.8 deg -> tilted (measurable)
- **310/X1600**: pitch +2.4 ± 1.5 deg (+1.6 sigma, f 1420..1520: +1.8..+2.8), roll +0.5 ± 1.5 deg (+0.3 sigma), total 2.8 deg -> not measurably tilted
- **80/X0**: pitch +13.7 ± 1.3 deg (+10.3 sigma, f 1420..1520: +13.5..+13.7), roll -0.5 ± 1.3 deg (-0.4 sigma), total 13.7 deg -> tilted (measurable)
- **80/X1600**: pitch +5.7 ± 0.9 deg (+6.1 sigma, f 1420..1520: +5.6..+6.0), roll -0.4 ± 0.9 deg (-0.4 sigma), total 5.9 deg -> tilted (measurable)

Control (4-corner shelf stickers 80:4, 80:5, 80:6, 310:6): pitch_outer 80:4 +0.3 ± 1.1, 80:5 -1.3 ± 1.6, 80:6 +0.6 ± 1.1, 310:6 +2.9 ± 2.0; roll 80:4 -0.2 ± 1.1, 80:5 -0.7 ± 1.4, 80:6 -0.3 ± 1.0, 310:6 -1.6 ± 2.2 deg (mean pitch +0.6, mean roll -0.7); max |value|/sigma = 1.5 -> the method returns horizontal for the shelf stickers within about 1-1.5 sigma.

Relative form (top minus 4-corner shelf sticker at the same cart end; the cart rotation cancels, the lens only partly): 310:1-310:6 -1.0 ± 2.1 deg (lens part 0.5); 310:2-310:6 -0.1 ± 2.0 deg (lens part 0.5); 80:0-80:5 +13.9 ± 1.5 deg (lens part 0.7); 80:92-80:5 +15.4 ± 1.8 deg (lens part 1.2); 80:1-80:4 +6.3 ± 1.3 deg (lens part 0.2); 80:2-80:4 +4.8 ± 1.3 deg (lens part 0.2); 80:1-80:6 +6.0 ± 1.3 deg (lens part 0.2); 80:2-80:6 +4.5 ± 1.3 deg (lens part 0.3).

Within each plate the front (Y=60) and back (Y=390) sticker differ in roll by 310/X0 +4.3 ± 1.4, 310/X1600 +2.7 ± 1.4, 80/X0 +2.7 ± 1.2, 80/X1600 +2.5 ± 1.2 deg (always front more 'back side up' than back) - consistent with plates slightly bowed up across Y (ridge between the stickers) of ~1-2 deg per sticker; the plate means of the roll are small.

Sticker-shape-free cross-check (board-front edges, 60-110 px): pitch_outer with yaw 0 ± 2 deg 80/X0 +9.7 ± 6.2, 80/X1600 +9.3 ± 6.8, 310/X0 +20.1 ± 7.7, 310/X1600 -4.7 ± 7.0; with the plate's sticker yaw 80/X0 +12.8 ± 1.8, 80/X1600 +7.9 ± 2.0, 310/X0 +14.7 ± 2.3, 310/X1600 +2.9 ± 2.7 deg. Difference to the sticker-shape plate pitch (with sticker yaw): 80/X0 -0.4 sigma, 80/X1600 +1.0 sigma, 310/X0 +0.4 sigma, 310/X1600 +0.1 sigma (the edge yaw is taken from the stickers, so this part is only semi-independent). Edges alone (yaw 0 ± 2 deg): X0 plates 80/X0 +1.6 sigma, 310/X0 +2.6 sigma from flat.

Uncertainty (review): lens from the final lens covariance and an empirical per-sticker systematic of 0.75 deg (within-plate pitch chi2 12.5 / 4 without it) are included in all sigmas above.

Other lens models (17; nominal data): plate pitch_outer range 310/X0 +12.7..+14.8; 310/X1600 -4.7..+8.5; 80/X0 +6.6..+14.3; 80/X1600 +3.9..+7.0. Without the two sticker-only drawing-geometry lenses (markers_only*, f 1326-1358, which absorb the geometry mismatch): 310/X0 +13.1..+14.8; 310/X1600 +0.8..+8.5; 80/X0 +12.3..+14.3; 80/X1600 +5.2..+7.0.

What-if (main lens, one LS pose per cart): tilting the top stickers about their own centres by the measured angles lowers the 62-corner RMS only from 7.89 to 7.73 px (top row 6.58 -> 6.24); the tilt moves the top corners by 2.1 px RMS (max 3.9). A rigid plate tilted about its INNER edge (sticker centres lifted by 82.5 mm x tan(pitch), ~20 mm at the X0 ends) gives 6.99 px (rows top/A/B/C 6.00/9.19/7.65/6.20), about the OUTER edge 8.67 px. The measured tilt is real but explains only a small part of the sticker misfit; together with a lifted outer end it points the same way as the known 'top stickers too high' mismatch (REPORT ch. 5), which needs ~60-80 mm, not ~20 mm.

## Review changes (vs work/61_top_sticker_orientation.py)

1. Lens term: the Monte Carlo lens draws come from the final lens covariance (statistical (+) systematic; cx / cy 1-sigma 18.8 / 29.1 px) instead of the statistical-only cluster bootstrap (cx / cy 5.6 / 8.3 px); the lens part of the plate pitch changes from 0.21-1.08 to 0.62-1.07 deg, the lens part of the X0 top-minus-shelf differences is 0.7-1.2 deg (the original text quoted 0.1-0.3 deg for all pairs).
2. Added the empirical per-sticker systematic s = 0.75 deg from the within-plate pitch consistency (without it chi2 = 12.5 for 4 plates).
3. Added the sticker-shape-free board-front-edge cross-check.
4. Central values, sign conventions, the method and all other checks are unchanged (independently re-implemented in the review: per-sticker angles reproduced within 0.05 deg; corner heights in the cart frame confirm the sign: at the X0 plates the X-smaller corners are the higher ones).
