# Working context for all sub-tasks (read this first)

Task: blind, independent intrinsic calibration of the station camera (see ../CLAUDE.md, in Slovak).
Everything must be derived from files in this repo. No web look-ups of numbers. Honest uncertainty
matters more than a pretty number. Code lives in `work/`, never modify `data/` or `spec/`.

## Shared code (do not break; add new modules instead of editing these, unless fixing a real bug)
* `work/common.py` – paths, spec loading (`SPECJ`, `SHELF_Z`, `CART_W/D`, `TOP_H`, `CODE_MM`),
  `marker_center(id, cart)`, `marker_corners_3d(id, cart, rot_k)` (OpenCV corner order TL,TR,BR,BL of
  the marker face; sticker face up, in-plane rotation rot_k*90 deg), `project()` (== cv2.projectPoints,
  Brown + optional rational k4..k6), `undistort_points()`, `save_json/load_json`.
* `work/calib.py` – `Model(spec)` (f single/fxfy, pp free/fixed, list of free dist coeffs),
  `Problem(model, points, carts)` bundle adjustment of intrinsics + one 6-DoF pose per cart,
  `covariance(res)`.
* `work/edgelib.py` – `trace_edge(poly, ...)`: sub-pixel edge localisation along the normal of an
  approximate polyline (gradient extremum + parabola), iterative; `line_fit`, `sagitta`.
* Images: `work/cache/mean_aligned_gray.npy` (float32, 7-frame jitter-compensated mean; use this for
  edges), `mean_aligned_gray.png`, `mean_aligned_color.png`, `std_aligned_gray.npy`.
  Raw stills: `data/stills/*.jpg` (7 frames, 1920x1080).

## Facts established so far (stage 01-03 scripts)
* Scene: fixed camera ~3.6 m above floor looking down, tilted ~20 deg; the nadir (vertical
  vanishing point) is near the TOP-centre of the image (~(930, 60) px). Two upright carts:
  - LEFT cart = No. 310 (unique id 322). Its X axis runs from image bottom (X=0 end, markers 0 @(774,915),
    322 @(594,950)) to image top (X=1600 end, markers 1 @(445,68), 2 @(256,143)). Front (Y=0) is its
    right side (towards the aisle), back (Y=450) its left side.
  - RIGHT cart = No. 80 (unique id 92). X runs from image top (X=0 end: 0 @(1435,57), 92 @(1628,86))
    to image bottom (X=1600 end: 1 @(1316,928), 2 @(1489,922)). Front (Y=0) is its left side (aisle).
  - Shelf stickers sit on small white cards at the front (Y=60) near each end, displaced towards the
    nadir the lower the shelf is. Expected but NOT yet detected by plain ArUco (partly occluded /
    oblique): cart 310: 5 (B-LEFT ~(800,740)), 7 (C-LEFT ~(818,660)), 4/6/8 (A/B/C-RIGHT ~(510,60),
    (560,65),(605,60)); cart 80: 3 (A-LEFT ~(1375,58)), 7 (C-LEFT ~(1270,58)), 8 (C-RIGHT ~(1210,680)).
    Detected: cart 310: 0,1,2,3,322 ; cart 80: 0,1,2,4,5,6,92 (12 of 20).
  - Top-plate stickers: rot_k = 0; the detected shelf stickers: rot_k = 2 (determined by fit).
  - The cart "top plate" is visible as a white board at each END of the cart (each carries 2 stickers,
    spanning the full 450 mm depth); between the ends the top is open and the shelves are visible.
* Frame jitter: the 7 stills differ by a vertical shift + vertical scale (rolling-shutter-like):
  dy = a_f + b_f*(y-540), |a_f| <= 1 px, |b_f| <= 6e-4; dx ~ 0.05 px. Stored in
  `cache/initial_calib.json["frame_jitter"]` ([ax0, ax1, ay0, ay1] per frame). After compensation the
  per-corner frame-to-frame std is ~0.18 px (so the 7-frame mean has ~0.07 px noise).
* Initial joint fit (plain detections, drawing geometry): f ~ 1290 px, k1/k2 poorly determined,
  **RMS 5 px, max 10 px** – far above detection noise. The inconsistency is NOT fixed by any lens
  model tried (Brown up to k3 + tangential, rational 8-coef RMS 3.5 px, fisheye KB 4.9 px), NOT by free
  in-plane sticker rotations, NOT by a free marker size, NOT by free sticker tilts. A single common
  offset of the shelf heights relative to the top plate (-60 mm) lowers RMS to 3 px. This is an open
  problem (lens vs geometry vs detection). Per CLAUDE.md the drawing dimensions must be used for the
  main result; mismatches are to be described as observations (where, how many px).
* Prototype plumb-line on automatic Canny chains (`proto_plumb*.py`): pieces straight to ~0.2 px after
  correction, k1 ~ -0.29, k2 ~ 0.08 at f0 = 1300 with centre fixed; with a free distortion centre the
  centre runs away (weakly constrained, lines mostly near-vertical) – needs long, verified lines.

## Conventions for outputs
* Lens JSON = format of results/lens_result.json in CLAUDE.md (OpenCV order [k1,k2,p1,p2,k3]).
* Every method writes `work/cache/method_<name>.json` = lens JSON + `"method"` + free-form
  `"details"` (numbers, checks) and a short markdown summary `work/cache/method_<name>.md`.
* Uncertainties: 1-sigma; say how they were obtained (covariance scaled by residual variance,
  bootstrap, profile, ...). Separate "determined" vs "consistent with" vs "not determinable".
* Plots go to `results/*.png` with descriptive names; keep them readable (dpi ~110).
* Do not write model names or tool names into repo files.

## Important pitfall found (plumb-line / any edge residual)
Measuring straightness residuals in the UNDISTORTED image (or rescaling them by a per-edge chord ratio)
creates a noise-induced shrinkage bias: the fit prefers weaker distortion because the undistortion
magnifies the noise (synthetic test: sigma 0.2 px -> k1 biased by +0.01...+0.04). Residuals must be
distances in the DISTORTED image: r_d = r_u / |J^T n| (J = Jacobian of the undistortion map, n = unit
normal of the fitted line in the undistorted image). lineselfcal.py and combined.py do this now.
With this correction the free distortion centre is recoverable in synthetic tests (+-20..50 px with
0.2-0.4 px per-edge bows). On the prototype automatic chains (unverified) the free centre goes to
~(926, 834), i.e. ~300 px below the image centre - to be checked with the verified edges.

## Phase-1 results (markers) and first geometry findings (orchestrator, 2026-09-27 ~17:10)
* `work/cache/markers_final.json` / `results/detections.json`: all 20 stickers found, ids confirmed from image
  content; 62/80 corners valid (edge-based, mean of 7 jitter-corrected stills, frame std ~0.13 px).
  corners_px of INVALID corners are completions (never use them); side_lines/side_valid give traced sides.
  rot_k: top stickers 0, shelf stickers 2. Edge corners sit 0.16 px inward along the diagonal vs ArUco rule.
* Joint drawing-geometry fit, all 62 corners: RMS 5.34 px (f~1309). Subsets (f single, pp free, k1,k2):
  shelf stickers only (A,B,C; 30 corners) RMS 1.88 px, f~1440; cart-310 shelves only 1.37 px f~1523;
  cart-80 shelves only 2.06 px f~1455; top stickers only 2.52 px f~1794(!). Even shelf-only is 10x noise.
* CROSS-RATIO test (projective invariant, independent of f/pp/pose, distortion barely matters): corresponding
  corners of the 4 stickers stacked at the same (X,Y) (top, A, B, C) lie on a vertical 3D line. Expected
  CR(Z=0,-195,-595,-995)=1.1960. Measured -> implied Z of the TOP sticker (others kept at drawing):
    cart 80 X=0 end (ids 0,3,5,7): -7 / 0 mm   (consistent with the drawing)
    cart 80 X=1600 end (1,4,6,8): +50 / +64 mm
    cart 310 X=0 end (0,3,5,7): +73 / +75 mm   (collinearity deviations up to 3 px - check 310:5/310:7!)
    cart 310 X=1600 end (1,4,6,8): +55 / +86 mm
  Visual impression: the end boards of cart 310 look like thick boxes (brown side faces), the cart-80 X=0 end
  board looks like a thin sheet. Needs careful verification (this is an OBSERVATION, not a model change).
* Edge-only (cart 310 edges only, interim): plumb-line k1=-0.272, k2=0.046 @ f0=1300 (centre fixed),
  free centre (947, 501)+-30; VP (plumb distortion fixed): f~1410-1418, pp~(947,467); joint lines f~1393.
* Edge-only with cart310+cart80 edges (orchestrator 30_lines_methods.py, interim): plumb (centre fixed)
  k1=-0.310, k2=0.097 @f0=1300; free centre (937, 515)+-(9,42); VP with plumb distortion: f=1394+-15,
  pp=(951, 416); VP with pp fixed at image centre: f=1433; joint lines: f=1397, pp=(913, 387).
  Per-edge VP-consistency: the END-BOARD edges (cartY inner edges, short cartX board fronts) violate the
  cart-axis VPs by 1.3-2.1 px while being straight (0.2-0.5 px) -> the boards are probably not exactly
  aligned with the cart frame; prefer structural edges (shelf lips, rails, posts) for VP groups.
* Marker-only BA with all 20 stickers (20_markers_ba.py): every model RMS 5.0-5.7 px, leave-one-sticker-out
  8.2-10.5 px, best by LOO k1k2_ppfix f=1300+-21 (bootstrap f std 34 px); leave-row-out: top 19-80 px.
