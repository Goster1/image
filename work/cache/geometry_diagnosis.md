# Sticker geometry: diagnosis of the inconsistency (observation, not a model change)

Scripts (each runs end-to-end from `work/`): `42_geomdiag_lib.py` (helpers), `42_geomdiag_a_columns.py` (1),
`42_geomdiag_b_planes.py` (2), `42_geomdiag_c_fits.py` (3), `42_geomdiag_f_crops.py` (visual check),
`42_geomdiag_d_twocart.py` (4), `42_geomdiag_e_markershape.py` (5), `42_geomdiag_g_effects.py` (px effects).
Numbers: `work/cache/geomdiag_{columns,planes,fits,twocart,markershape,effects}.json`; plots `results/geomdiag_*.png`.

Per CLAUDE.md the drawing dimensions and the sticker positions are exact and the main result must use them.
Everything below that frees a dimension is a **diagnostic** to locate and size the mismatch; it is **not**
a proposal for the main result, and "sticker glued elsewhere" is not used as an explanation.

## Inputs and common settings

* 20 stickers, 62 valid corners (edge-based, mean of 7 jitter-corrected stills). Per-frame corner scatter
  0.07-0.33 px (radial); the MC uses sigma = corners_std_px / sqrt 2 per coordinate (per-frame scatter,
  i.e. ~2.6x the std of the 7-frame mean - conservative). Traced side lines: rms / sqrt(n/3) offset.
* Distortion for all projective tests: plumb-line (LineCal 'plumb') on 72 verified straight edges
  (cart 310, cart 80, scene), f0 = 1300, in pixel units (so every projective statement is independent of f and pp):
  centre fixed: k1 = -0.2954, k2 = 0.0732 (edge rms 0.285 px); centre free: (926.5, 467.1) +- (5.5, 23.9),
  k1 = -0.277, k2 = 0.061; k1-only: -0.2075 (rms 0.58). Variant "none" (no undistortion) as an extreme check.
* Vertical vanishing point (undistorted px, plumb centre fixed): from the 4 cart-post edges (cart Z axis)
  (930.6, 47.4) +- (0.2, 0.4) formal (a 3 px floor is used in the MC); from the 6 scene verticals
  (939.7, 32.5) +- (1.0, 0.9): **17 px apart (~0.6 deg)** - the carts' Z axes and the building verticals are not
  exactly parallel (floor slope or leaning structure). The cart-post VP is used for the sticker stacks (they run
  along cart Z).

## 1. Projective invariants of the sticker stacks (independent of f, pp and pose)

Corresponding corners of top / A / B / C stickers at the same (X, Y) must lie on one image line (a vertical 3D
line) with cross ratio CR(0, -195, -595, -995) = 1.1960. Hidden corners were replaced by the intersection of
a valid traced side line with the column line when possible. "Z_top from A,B,C" = height of the top-sticker
corner implied by the cross ratio with the shelves at the drawing heights; "+ VP" uses the cart-post vanishing
point as the 5th point (Z = infinity), which also gives the affine ratios of the gaps.

| cart / end | column XY [mm] | levels: corners (+ side line, angle) | collinearity RMS [px] meas / noise q95 | cart-Z VP off line [px] | CR (exp. 1.1960) | Z_top from A,B,C [mm] | Z_top from A,C + VP | gap top-A / A-B [mm] if B-C = 400 | VP(A,B,C) - VP(posts) along column [px] | Z_top (plumb centre free) |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 / X=0 | (127.5, 15) | top,A,B,C | 0.75 / 0.13 | 13.6 | 1.1962 | 0.3 +- 1.6 | 72.9 +- 1.3 | 305 / 511 | -178 +- 4 | 0.1 |
| 80 / X=0 | (37.5, 15) | top,A,B,C | 0.37 / 0.11 | 7.7 | 1.1904 | -6.9 +- 1.2 | 59.0 +- 1.3 | 288 / 506 | -175 +- 3 | -7.2 |
| 80 / X=1600 | (1472.5, 105) | top,A,B + C (60 deg) | 0.23 / 0.15 | 9.6 | 1.2335 | 48.7 +- 1.4 | 59.0 +- 0.8 | 258 / 413 | -86 +- 9 | 48.8 |
| 80 / X=1600 | (1562.5, 105) | top,A,B + C (62 deg) | 0.83 / 0.13 | 4.3 | 1.2398 | 57.3 +- 1.1 | 66.6 +- 0.7 | 265 / 411 | -79 +- 7 | 57.6 |
| 80 / X=1600 | (1562.5, 15) | top,A,B,C | 1.18 / 0.10 | -1.9 | 1.2444 | 63.7 +- 0.8 | 71.7 +- 0.6 | 270 / 409 | -66 +- 6 | 63.8 |
| 80 / X=1600 | (1472.5, 15) | top,A,B,C | 1.25 / 0.15 | -3.2 | 1.2346 | 50.2 +- 1.3 | 60.1 +- 0.8 | 259 / 412 | -82 +- 8 | 50.2 |
| 310 / X=0 | (127.5, 105) | top,A + B (65 deg), C (65 deg) | - | 57.2 (2-pt line) | 1.2442 | 63.4 +- 1.3 | 65.7 +- 0.8 | 262 / 403 | -18 +- 8 | 63.4 |
| 310 / X=0 | (127.5, 15) | top,A,B,C | 2.55 / 0.16 | 18.2 | 1.2521 | 74.7 +- 1.7 | 58.3 +- 0.8 | 247 / 382 | 154 +- 17 | 74.7 |
| 310 / X=1600 | (1562.5, 15) | top,A,B,C | 1.33 / 0.20 | 1.1 | 1.2379 | 54.7 +- 3.5 | 72.2 +- 1.4 | 274 / 421 | -58 +- 10 | 57.1 |
| 310 / X=1600 | (1472.5, 15) | top,A,B,C | 1.87 / 0.11 | 1.2 | 1.2594 | 85.2 +- 2.0 | 82.6 +- 1.3 | 277 / 397 | 9 +- 7 | 88.9 |
| *unreliable:* 310 / X=0 | (37.5, 15) | top + A,B,C side lines at 29-31 deg, line through VP | - | - | 1.2937 | 137.6 +- 3.6 | 64.7 +- 1.4 | 238 / 334 | - | 137.7 |
| *unreliable:* 310 / X=1600 | (1472.5, 105) | top,B + A,C side lines at 18 deg | - | -4.3 | 1.2113 | 19.3 +- 5.7 | 50.5 +- 3.7 | 259 / 445 | -108 +- 13 | 18.6 |
| *unreliable:* 310 / X=1600 | (1562.5, 105) | top,B + A,C side lines at 20-24 deg | - | -6.1 | 1.1546 | -48.8 +- 5.0 | -0.7 +- 4.4 | 220 / 506 | -189 +- 10 | -49.7 |

(80 / X=0 columns at Y = 105 have only top and B corners; 310 / X=0 (37.5, 105) only the top corner.
Grazing side-line intersections turn the 1-3 px lateral misalignment of the column into 5-15 px along the
column and are not used for conclusions. MC: N = 3000, corner + side-line noise + VP (3 px floor).)

Sensitivity to the distortion choice (Z_top from A,B,C, per column): plumb centre fixed / centre free / k1 only
agree within 0-4 mm at every column; even no undistortion changes it by at most 12 mm (310 / X=1600).

Observations (1):
* **Collinearity fails everywhere:** the stacked corners deviate from one image line by 0.23-2.55 px RMS
  (per level up to 3.0 px) vs 0.10-0.20 px expected (95 % quantile of the noise-only MC); identical for all
  distortion variants (0.2-0.3 px change). With the local horizontal scale 0.37-0.71 px/mm this is a lateral
  scatter of the stacked stickers of about 1-6 mm around a common vertical line. The largest deviations are
  mostly at shelf A (up to -3.0 px) and B, e.g. 310/X0: top -2.4, A +2.2, B +2.9, C -2.7 px; 310/X1600:
  top +1.5, A -3.0, B +1.7, C -0.2 px.
* **Cross ratio:** the 4-corner columns imply the top sticker at +50..+85 mm (80/X1600: 50, 64; 310/X0: 75;
  310/X1600: 55, 85) and at -7 / 0 mm at 80/X0. MC noise is only 1-3.5 mm; the spread between the columns
  of one end (up to 30 mm) is the realistic uncertainty and itself shows local irregularity.
* **With the cart-Z vanishing point:** the shelf gaps A-B : B-C are 409-421 : 400 at the two X=1600 ends and
  382-403 : 400 at 310/X0 (consistent to ~5 %), but 506-511 : 400 at 80/X0, where the drawing heights of A, B, C
  put the vanishing point 176 +- 3 px from the post VP along the column (MC 1-sigma 3-4 px). The top-A gap
  comes out 247-305 mm instead of 195 at **all four** ends (incl. 80/X0). So 80/X0 passes the cross-ratio test
  but fails the (independent) vanishing-point test; with the VP all four ends show the top sticker
  50-110 mm too far above shelf A.
* **Vertical VP of the column lines:** the 2-4-point column lines pass 1-24 px beside the cart-post VP
  (MC noise +-3 px): X=1600 ends 1-10 px, 80/X0 8-24 px, 310/X0 18 px. Intersection of each cart's column lines:
  cart 80 (934.7, 62.3), cart 310 (892.9, 47.2), post edges (930.6, 47.4), scene verticals (939.7, 32.5) - the
  per-cart column VPs are shifted by 15-38 px, consistent with the lateral scatter of the stacks, not with a
  sharper vertical.

## 2. Planarity / rectangle tests of the top stickers (undistorted, homographies)

In-board test (one end board = 2 stickers on one plate, 8 corners, drawing layout; homography + one freed element):

| board | plain H RMS [px] | + code size: RMS / code [mm] | ArUco-subpixel corners: plain -> code size / code [mm] | equivalent: Y spacing too large by [mm] | + relative in-plane rotation: RMS / deg | + relative height of 2nd sticker: RMS |
|---|---|---|---|---|---|---|
| 80 / X=0 | 0.66 | 0.27 / 87.87 | 0.53 -> 0.38 / 88.71 | 8.0 | 0.65 / 0.44 | 0.25 |
| 80 / X=1600 | 0.62 | 0.37 / 88.02 | 0.58 -> 0.36 / 88.17 | 7.4 | 0.60 / 0.63 | 0.55 |
| 310 / X=0 | 0.76 | 0.31 / 87.40 | 0.69 -> 0.31 / 87.71 | 9.8 | 0.76 / -0.19 | 0.71 |
| 310 / X=1600 | 1.03 | 0.21 / 86.57 | 0.81 -> 0.16 / 87.32 | 13.1 | 1.01 / -0.60 | 0.35 |

* Two coplanar squares at the drawing layout should fit a homography to ~0.1 px; they fit to 0.62-1.03 px
  (plumb centre free: 0.60-1.03; no undistortion 0.89-1.32). **No radial distortion fixes it**: a joint fit of
  k1, k2 (and a free centre) to the four boards bottoms out at 0.59-0.91 px.
* One number per board explains most of it: the ratio code size / sticker spacing. Within one board a code of
  86.6-88.0 mm (ArUco corners 87.3-88.7 mm) at 330 mm spacing is indistinguishable from a 90 mm code at
  337.4-343.1 mm spacing (a homography absorbs the common scale). A relative rotation does not help; a relative
  height helps only at 80/X0.
* Whole top of a cart (16 corners, one plane, 1435 x 330 mm rectangle of centres): H RMS 2.03 px (cart 80),
  2.73 px (cart 310); with a vertical shift of the second board along the post-VP direction: 2.00 / 1.87 px.
  The two end boards are **not consistent with one plane in the drawing layout**, and a pure relative height
  does not explain it either. (Cross-prediction of one board from the other board's homography is off by 43-147 px,
  but that is a 1435 mm extrapolation of a 0.6-1 px in-board misfit and not meaningful on its own.)

General projective camera per cart (3x4 P, 11 DoF: arbitrary f, pp, skew, aspect, pose; plumb distortion;
absorbs any affine change of the 3D coordinates, so only non-affine inconsistencies remain):

| geometry | cart 80 RMS [px] | cart 310 RMS [px] | freed values (cart 80 ; cart 310) |
|---|---|---|---|
| drawing | 5.10 | 4.87 | - |
| end-board heights | 2.30 | 2.48 | X0 +76 +- 1, X1600 +62 +- 5 mm ; X0 +56 +- 8, X1600 +96 +- 12 mm |
| top-plate height | 2.35 | 2.77 | +64 +- 5 ; +68 +- 8 mm |
| shelf-stack height per end | 2.29 | 2.61 | -78, -61 ; -60, -90 mm |
| end-board horizontal offsets | 2.40 | 2.47 | dY +26 / +25 mm, dX +37 mm at X1600 ; dX -38 mm at X0, dY +11 / +26 mm |
| end-board tilts | 3.33 | 2.78 | 10-20 deg (implausible) |
| code size | 5.04 | 4.87 | 87.8 ; 89.5 mm |
| code size + board heights | 2.18 | 2.44 | 87.9 mm, +76, +62 ; 88.9 mm, +57, +96 |
| board heights + tilts | 1.90 | 1.97 | +92, +73 (tilts 3-11 deg) ; +67, +94 (2-11 deg) |
| shelf stickers only, drawing | 2.04 | 1.36 | - |

The drawing geometry is inconsistent with **every** pinhole camera (5 px, ~40x noise) - this is not an
intrinsics problem. Freeing the top heights halves the residual, but even the shelf stickers alone (A, B, C)
are inconsistent at 1.4-2.0 px under a general camera.

## 3. Which element explains it? Diagnostic bundle adjustments (both carts, 62 corners, NOT the main result)

Camera fx = fy = f, pp free, k1, k2 free, one pose per cart; one element freed, rest = drawing. AIC = n ln(RSS/n) + 2k
(n = 124). Last columns: same fit with the plumb-line distortion fixed in pixel units.

| freed element | k | RMS [px] | AIC | f [px] | pp [px] | k1, k2 scaled to f0 = 1300 (plumb: -0.295, 0.073) | RMS / f with plumb dist. | row RMS top/A/B/C | diagnosed values |
|---|---|---|---|---|---|---|---|---|---|
| drawing | 17 | 5.34 | 364 | 1309 +- 20 | (938, 485) | **-0.087, -0.231** | 5.93 / 1294 | 4.9/8.0/3.6/5.4 | - |
| end-board heights (4) | 21 | 2.88 | 219 | 1440 +- 14 | (939, 390) | -0.319, 0.138 | 2.99 / 1430 | 3.4/1.9/2.5/2.3 | 80/X0 +64 +- 8, 80/X1600 +71 +- 5, 310/X0 +61 +- 6, 310/X1600 +81 +- 8 mm |
| shelf-row heights A,B,C (3) | 20 | 2.90 | 218 | 1475 +- 27 | (978, 390) | -0.405, 0.286 | 3.13 / 1410 | 3.5/1.8/2.1/2.0 | A -71 +- 5, B -92 +- 12, C -92 +- 19 mm |
| row heights per cart (6) | 23 | 2.83 | 218 | 1481 +- 27 | (985, 394) | -0.415, 0.302 | 3.08 / 1410 | 3.5/1.6/1.9/1.8 | -71..-102 mm |
| top-plate height per cart (2) | 19 | 3.01 | 225 | 1434 +- 15 | (972, 382) | -0.373, 0.233 | 3.19 / 1416 | 3.6/2.1/2.3/2.2 | 80: +68 +- 5, 310: +67 +- 6 mm |
| end-board tilts (8) | 25 | 4.54 | 339 | 1305 +- 20 | (933, 521) | 0.008, -0.328 | 5.48 / 1278 | 3.5/7.1/4.2/4.9 | 4-11 deg |
| end-board horizontal offsets (8) | 25 | 2.82 | 221 | 1432 +- 17 | (890, 428) | -0.222, 0.025 | 3.39 / 1371 | 3.2/2.3/2.5/2.4 | dY +9..+27 mm, dX up to +-36 mm |
| shelf-stack height per end (4) | 21 | 2.96 | 225 | 1440 +- 14 | (929, 386) | -0.299, 0.110 | 3.11 / 1422 | 3.5/2.1/2.4/2.1 | -58..-85 mm |
| every shelf sticker height (12) | 29 | 2.66 | 214 | 1495 +- 34 | (930, 356) | -0.297, 0.110 | 2.86 / 1455 | 3.4/1.3/1.4/1.3 | -68..-154 mm |
| code size (1) | 18 | 5.21 | 359 | 1309 +- 20 | (937, 483) | -0.065, -0.261 | 5.90 / 1291 | 4.7/8.1/3.2/5.1 | 86.8 +- 1.4 mm |
| code size + board heights (5) | 22 | 2.74 | **208** | 1442 +- 14 | (939, 392) | **-0.295, 0.105** | 2.89 / 1429 | 3.2/1.9/2.3/2.1 | 87.5 +- 0.8 mm; +66, +70, +60, +82 mm |
| code size + top height per cart (3) | 20 | 2.88 | 216 | 1435 +- 14 | (973, 385) | -0.361, 0.219 | 3.10 / 1414 | 3.4/2.0/2.2/2.0 | 87.6 mm; +66, +68 mm |
| board heights + tilts (12) | 29 | 2.32 | 181 | 1434 +- 14 | (915, 370) | -0.265, 0.025 | 2.33 / 1430 | 2.3/2.1/2.5/2.5 | +59..+83 mm, tilts 1-13 deg |
| code + board heights + tilts (13) | 30 | 1.94 | **138** | 1441 +- 11 | (908, 395) | -0.206, -0.044 | 2.00 / 1428 | 1.6/2.2/2.2/2.3 | 86.2 mm; +54..+82 mm; tilts about Y 5-14 deg |

(f / pp uncertainties: Gauss-Newton covariance scaled by the residual variance; they do not contain the
model error. Camera height above the floor agrees between the carts in every fit (differences < 25 mm, insensitive - see 4);
the floor normals differ by 0.7-2.4 deg.)

Answer to (3):
* **No single element, and no compact combination, brings the RMS near the detection noise (0.1-0.2 px).**
  The best 1-2-element fits reach 2.7-3.0 px; with 13 extra parameters 1.9 px. Only a free pose for every
  sticker reaches 0.17 px (section 5), i.e. the corners themselves are fine and each sticker is a perfect square.
* The dominant, compact element is **the height of the top stickers relative to the shelf stickers**: top
  plate / end boards +60..+80 mm, or equivalently all shelves (or shelf-sticker cards) 60-90 mm lower. Heights of
  the four end boards individually: +61, +64, +71, +81 mm (+- 5-8). The metric fits cannot separate "top higher"
  from "shelves lower" (row_dz vs top_dz: equal RMS). Horizontal board offsets of ~25 mm fit similarly well
  (2.82 px) but then the distortion disagrees with the edges (see next point) - the height version is preferred.
* **Consistency with the straight edges (independent evidence):** with the drawing geometry the free distortion
  is k1 = -0.087, k2 = -0.231 (at f0 = 1300), incompatible with the plumb-line from 72 verified edges (-0.295,
  0.073); fixing the plumb distortion costs dAIC = +22 for the drawing. With the end-board heights + code size
  freed, the stickers alone return k1 = -0.295, k2 = 0.105 - the plumb-line distortion - and fixing the plumb
  distortion costs only dAIC = +9 (board heights alone +5, heights + tilts -3). So the geometry deviation, not the
  lens, is what distorts the sticker-only lens estimate.
* **Implied intrinsics:** drawing f = 1309 +- 20, pp (938, 485); every variant that frees the top/shelf heights gives
  f = 1430-1495 (+-11-34), pp_y 356-396, pp_x 890-985 - i.e. with the drawing geometry f comes out 120-190 px lower
  and pp_y 90-130 px larger (further down in the image). The code size alone moves nothing (f 1309).
* **Code size:** within the boards the effect is clear (see 2); globally it matters little (5.34 -> 5.21 px alone,
  2.88 -> 2.74 px with heights) and is equivalent to a 2-4 % larger sticker spacing inside each board.

### Pixel effect of the diagnosed deviations (camera of the diagnostic fit held fixed)

| element | measured (diagnostic) vs drawing | px effect on the affected corners (RMS / max) | where in the image | uncertainty |
|---|---|---|---|---|
| end board 80 / X=0 height | +64 mm vs 0 | 16.4 / 18.8 px | x 1404-1656, y 25-116 (top right) | +-8 mm (fit); cross ratio says -7..0 mm here |
| end board 80 / X=1600 height | +71 mm vs 0 | 23.8 / 24.4 px | x 1288-1516, y 899-952 (bottom right) | +-5 mm; cross ratio +50..+64 |
| end board 310 / X=0 height | +61 mm vs 0 | 19.8 / 20.3 px | x 563-807, y 886-977 (bottom left) | +-6 mm; cross ratio +63..+75 |
| end board 310 / X=1600 height | +81 mm vs 0 | 19.8 / 23.0 px | x 226-480, y 31-178 (top left) | +-8 mm; cross ratio +55..+85 |
| code size (in-board ratio) | 86.6-88.0 mm vs 90 (or spacing +7..+13 mm vs 330) | 1.0 / 1.2 px (corners of all stickers) | all stickers | +-0.8 mm (BA), board-to-board spread 1.5 mm |
| end-board tilt about the cart Y axis (section 5, f = 1400) | 4-14 deg vs 0 | 0.8-4.0 px per corner | top boards (image top and bottom edges) | +-1.4-2.4 deg; f-dependent by ~3 deg / 100 px |
| lateral scatter of the stacked stickers | 1-6 mm vs 0 (collinearity 0.23-2.55 px) | 0.2-3.0 px | all four sticker stacks | noise 0.1-0.2 px |
| remaining after all of the above | - | 1.9 px RMS | everywhere, largest on shelf rows A-C | - |

With the drawing geometry, the refitted camera leaves 5.34 px RMS / 9.6 px max; the largest residuals are on
row A (8.0 px RMS, the stickers just below the end boards) and row C (5.4 px).

### Visual check (results/geomdiag_crops.png; zoomed crops of mean_aligned_color.png)

* **All four end boards look like thin white plates** (a printed sheet on a board: faint printed guide lines
  between the two stickers) and not like boxes. The visible inner end faces are only ~4-6 px wide (brown at
  310/X0, tan at 80/X1600), i.e. ~10-15 mm at the local vertical scale of 0.36-0.37 px/mm; a 60-90 mm tall side face
  would be 20-35 px wide there. At 310/X1600 the plate edge next to sticker 1 is a sharp edge over the shelf-A card.
  **So a 60-90 mm raised box under the stickers is not seen.** (An earlier impression of "thick boxes" at cart 310
  does not survive high zoom.)
* Plate deformation: cart 80, both boards: the back (Y = 450) corner is visibly **curled up**; cart 310 X=0: a grey
  downward flap at the back end; cart 310 X=1600: the back part appears to bend down. The plates are therefore not
  guaranteed flat - compatible (qualitatively) with the in-board scale anomaly and the tilts of section 5.
* Shelf stickers sit on small white cards at the shelf fronts; at X=0 of cart 310 the cards show white front faces
  (folded/raised cards, thickness not resolvable), some card corners are hidden by the posts or by the next card.
  No obvious strong tilt of the cards.
* Whether the cart structure itself (top plate vs shelf heights) differs from the drawing cannot be seen in the image.

## 4. Two-cart consistency (both carts on the same floor)

Each cart gets its own 6-DoF pose from its stickers (drawing geometry) at FIXED f (1100-1800 px, step 20), with the
plumb-line distortion fixed in pixel units (k1 = k1px f^2, k2 = k2px f^4) and pp either fixed at the image centre
or free and common. Compared: camera height above the floor h = C_z + 1800 mm from each cart pose, and the two
floor normals (cart Z axes in the camera frame). Plot: `results/geomdiag_twocart.png`; method file
`work/cache/method_twocart.json` / `.md`.

| point set | pp | f where the floor normals agree (x-component = 0) | cov. 1-sigma | bootstrap over stickers (n) | min. angle between normals | sticker RMS there |
|---|---|---|---|---|---|---|
| (a) all stickers | fixed (959.5, 539.5) | 1376 | 24 | 34 (150) | 0.07 deg at f = 1380 | 6.6 px |
| (a) all stickers | free (-> (956, 524)) | 1377 | 24 | 55 (146) | 0.06 deg at f = 1380 | 6.6 px |
| (b) shelf stickers only | fixed | 1489 | 19 | 49 (148) | 0.28 deg at f = 1480 | 2.3 px |
| (b) shelf stickers only | free | 1488 | 24 | 67 (144) | 0.20 deg at f = 1480 | 2.3 px |
| (c) top stickers only | fixed | 1324 | 107 | - | 3.0 deg (never parallel) | 4.3 px |

* The **camera height above the floor is useless** as a constraint here: both carts are at similar depth and
  their heights scale together with f; |h80 - h310| < 5 mm over f = 1100-1800 (pp fixed; with pp free the
  dh "roots" at 1481/1630 are artefacts of the pp wandering off / a jump between local minima at f ~ 1620).
* The **floor-normal agreement** is sensitive (x-component slope ~ 2.5 deg per 100 px): all stickers
  f = 1377 +- 24 (cov) +- 55 (bootstrap); shelf stickers only f = 1488 +- 19 / +- 49..67. The 110 px difference
  between the two point sets is the geometry dependence; with the half difference as a systematic term:
  **f = 1377 +- 82 px** (main entry of method_twocart.json: all stickers, pp free; pp = (956, 524) from the fit at
  that f, not separately profiled).
* Top stickers alone: the two carts' tops never become parallel (>= 3 deg) and imply camera heights differing by
  50-115 mm - the top plates as planar targets are mutually inconsistent (consistent with tilted / bent plates).
* The drawing-geometry sticker RMS at the agreement point is 6.6 px, so this estimate inherits the geometry problem;
  it is an independent **consistency** check, not a precise measurement.

## 5. Sticker-shape method (horizontal squares; positions and heights not used)

Joint ML form of "per-sticker homography -> vanishing points of its sides + horizon, combined per cart with the
nadir": f, pp, one rotation per cart; every sticker is a flat horizontal square with a FREE 3D position (heights,
spacings and the code size are not used) and orientation = cart rotation x rot_k * 90 deg; the cart-post
vanishing point ties each cart's Z axis (weight: 3 px); plumb distortion fixed in pixel units; all valid corners.
Bootstrap: stickers resampled within each cart (150 replicates; robust sigma = half the 16-84 % range).
Files: `work/cache/method_markershape.json` / `.md`, `results/geomdiag_markershape.png`.

| variant | f [px] | cov. 1-sigma | bootstrap 16-84 % | pp [px] | corner RMS [px] |
|---|---|---|---|---|---|
| all stickers, nadir, pp free (method file) | 1601 | 178 | 1428-1775 | (978, 470) | 1.48 |
| all stickers, nadir, pp fixed | 1496 | 77 | 1373-1621 | (959.5, 539.5) | 1.50 |
| all stickers, no nadir, pp free | 3183 | 1437 | - | (949, 435) | 1.45 |
| all stickers, no nadir, pp fixed | 1128 | 280 | - | (959.5, 539.5) | 1.46 |
| shelf stickers only (10), nadir, pp free | 1417 | 134 | 1338-1491 | (989, 489) | 0.60 |
| shelf stickers only, nadir, pp fixed | 1388 | 63 | - | (959.5, 539.5) | 0.61 |
| top stickers only (8), nadir, pp free | 1619 | 255 | 1416-1815 (degenerate replicates) | (998, 477) | 1.84 |
| top stickers only, nadir, pp fixed | 1514 | 105 | - | (959.5, 539.5) | 1.87 |

* **Too weak to be useful for f:** the corner RMS vs fixed f is almost flat (1.69 px at 1150, 1.51 at 1400,
  1.48 at 1600, 1.49 at 1750); without the nadir f is unconstrained (1128-3183). 60-px squares seen under ~20 deg
  carry little perspective information; single-sticker poses are even two-fold ambiguous (IPPE).
* **Diagnostic value:** the shape model should fit to ~0.15 px but gives 1.5 px (top stickers 1.8-1.9 px, shelf
  stickers 0.6 px). With a free tilt + in-plane yaw per sticker (f fixed at 1300 / 1400 / 1500) it drops to
  **0.17 px** - every sticker is individually a clean square; they are just not mutually parallel. Tilts about the
  cart Y axis (dZ/dX, f = 1400; +- 1.4-2.4 deg for 4-corner stickers):

| board / sticker | tilt dZ/dX [deg] at f = 1300 / 1400 / 1500 | meaning |
|---|---|---|
| 80 / X=0: 0, 92 | -7.8 / -10.9 / -13.7 ; -10.5 / -13.7 / -16.6 | outer (X=0) end higher |
| 80 / X=1600: 1, 2 | +9.0 / +7.4 / +5.9 ; +8.1 / +6.3 / +4.7 | outer (X=1600) end higher |
| 310 / X=0: 0, 322 | -12.5 / -11.3 / -10.3 ; -14.5 / -13.4 / -12.3 | outer end higher |
| 310 / X=1600: 1, 2 | +1.5 / +4.2 / +6.8 ; +2.4 / +5.2 / +7.8 | outer end higher (weak) |
| shelf stickers with >= 3 corners (80: 4, 5, 6; 310: 6) | -0.8..+3.8 / +1.3..+5.4 / -0.8..+7.9 | ~horizontal within 2 sigma |

  The two stickers of one plate agree (the plate tilts, not the sticker), the pattern holds for f = 1300-1500
  (magnitude changes ~3 deg per 100 px), and top and shelf stickers at the same image position differ (80/X0 top
  -11..-14 deg vs shelf B +1.4 +- 3.8 deg), so it is not a pure lens effect. The in-plane yaws of the fully visible stickers
  spread by up to ~2.5 deg within a cart (gauge fixed by a weak prior), i.e. the plates are also slightly rotated.

## Conclusions

**Established (independent of f, pp and of the plumb-line variant):**
1. The drawing geometry of the stickers is inconsistent with any pinhole camera: a general 3x4 projective camera
   per cart (11 DoF, arbitrary intrinsics) leaves 4.9-5.1 px RMS (noise ~0.1-0.2 px); the full metric fit 5.3 px.
   This is a geometry/placement mismatch, not a lens-model deficit (radial k1, k2 of any value with a free centre
   cannot remove even the in-board part).
2. Each sticker individually is consistent with a projected flat square (0.17 px with a free pose per sticker) - detections
   and the local lens model are fine; the inconsistency is in the relative placement / orientation of stickers.
3. Within every end board the two stickers are inconsistent with the drawing layout by 0.62-1.03 px (5-10x noise);
   a code/spacing ratio 2.2-3.8 % below nominal (code 86.6-88.0 mm at 330 mm, or spacing 337-343 mm at 90 mm)
   removes most of it (0.21-0.37 px).
4. The stacked corners (top, A, B, C) are not collinear: 0.23-2.55 px RMS vs 0.10-0.20 px noise -> the stacked
   stickers scatter laterally by ~1-6 mm around a vertical line.
5. The cross ratio of the stacks puts the top stickers 50-85 mm above the drawing (relative to the shelf stickers)
   at three ends (80/X1600, 310/X0, 310/X1600); at 80/X0 the cross ratio fits (-7 / 0 mm) but the gap ratios with
   the post vanishing point do not (A-B : B-C = 1.27 instead of 1.00; top-A gap 288-305 mm instead of 195).

**Suggested (depends on a metric camera model or on combining elements):**
6. The most compact single explanation is a +60..+80 mm offset of the top stickers relative to the shelf stickers
   (end boards +61/+64/+71/+81 mm, or per cart +67/+68 mm; equivalently shelves / shelf cards ~70-90 mm lower).
   It halves the RMS (5.3 -> 2.9 px) and makes the sticker-derived distortion agree with the straight-edge
   distortion (with the code size: k1 = -0.295, k2 = 0.105 vs plumb -0.295, 0.073), while the drawing geometry
   forces an edge-incompatible distortion (k1 = -0.09, k2 = -0.23).
7. The end plates are tilted along the cart X axis, outer end up: 6-14 deg at three plates, 4-5 deg (weak) at
   310/X1600 in the shape diagnostic (f = 1400; ~3 deg per 100 px f-dependent); the metric board-tilt fits agree
   at three plates and give the opposite sign at 310/X1600. Qualitatively consistent with the visibly curled / bent plates.
8. The focal lengths implied: drawing geometry f ~ 1290-1310; any variant that frees the top/shelf heights
   f ~ 1410-1480, pp_y ~ 360-400; two-cart floor-normal agreement f = 1377 +- 82 (all stickers) / 1488 (shelf only); sticker shape f = 1601 (pp free; 16-84 %: 1428-1775) / 1417 +- 134 (shelf only) - too weak to decide.

**Cannot be decided from these data:**
9. Top too high vs. shelves (or shelf cards) too low - only the relative offset is observable.
10. Code size vs. sticker spacing inside a board (only their ratio is observable).
11. The physical cause: the visible board thickness is ~10 mm, so the 60-90 mm are not a box under the stickers;
    the images do not show where the offset sits (frame, shelf heights, card mounting).
12. A residual of ~2 px remains after every compact geometric element tested (and 1.4-2.0 px among the shelf
    stickers alone), so part of the mismatch is sticker-by-sticker (cards, bent plates) and not modelable compactly.

**Consequences for the main (drawing-geometry) result:** a sticker-only fit with the drawing geometry is biased,
not noisy: it gives f ~ 1300, pp_y ~ 480 and a distortion that contradicts the straight edges, with 5-6 px RMS
(9.6-12.4 px max). Its formal uncertainties (f +- 15-20 px) are meaningless; the spread of f between geometric
hypotheses (1290-1480 px) is the realistic model uncertainty of any sticker-based estimate. The edge-only methods
(plumb-line, vanishing points) do not use the sticker geometry and should carry the main weight; the stickers
constrain the result only through the drawing geometry, whose mismatch must be reported (tables above: where, how
many px / mm).
