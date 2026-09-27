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
