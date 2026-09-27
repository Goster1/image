# Lens-model choice (sub-task 43): which model is adequate to the data

Scripts: `work/43_altmodels_lib.py` (lens models + estimator), `43_altmodels_fit.py` (fits + CV), `43_altmodels_report.py` (criteria, mappings, conversions, plots, method files), `43_altmodels_diaggeom.py` (diagnostics). Run order: fit (~40 min on a loaded 4-CPU box) -> diaggeom -> report.
Numbers: `work/cache/altmodels_fits.json`, `altmodels_summary.json`, `method_alt_*.json`; plots `results/altmodels_mapping_diff.png`, `results/altmodels_radial.png`, `results/altmodels_cv.png`.

Data: M = 62 sticker corners (drawing geometry exact), E = 61 verified straight edges (straightness S + VP groups V: cart X/Z axes without end-board edges, free scene vertical, floor_V / floor_H perpendicular to it), ME = both. Block sigmas fixed per data set from the Brown k1,k2 variance components: M 3.77 px; E: V 0.361, S 0.229 px; ME: M 6.14, V 0.361, S 0.229 px. Every fit carries a fold barrier (radial map must stay invertible to the image corners, margin >= 0.03).

## Conclusions

1. **Stickers only (M, drawing geometry) cannot choose a lens model.** All 16 models leave 4.96-5.99 px radial RMS (noise ~0.1-0.2 px); the Brown k1,k2 fit has distinct local minima of practically equal cost (pp_y ~287 px, and ~485 px which is folded inside the image and excluded by the fold barrier; multi-start used). Leave-one-sticker-out RMS is 7.5-9.1 px for every model. The sticker-only lenses do not straighten the edges (lens(M) -> edges straightness 0.40-2.83 px vs 0.229 px for the edge fit). The models M 'prefers' (p1,p2, rational, KB k1..k4) are those that absorb the geometry mismatch. DIAGNOSTIC (5 extra geometry parameters, not a main estimate): with board heights + code size free, all radial 2-3-parameter models give the same chi2 (32.7 vs drawing 124.0; RMS 2.74 px), i.e. the stickers carry no information on the radial-profile shape; only k1-only (chi2 55.6) and pp fixed (44.2) are worse.
2. **Edges (E) need two radial degrees of freedom; one is not enough.** Brown k1 alone runs into the fold barrier (unconstrained optimum folded inside the image), straightness 0.286 vs 0.229 px, VP 0.578 vs 0.361 px, dQAIC +302; division l1 (1 parameter, no fold) dQAIC +27 and worse in CV. All 2-3-parameter radial forms (Brown k1,k2 / k1,k2,k3, KB k1,k2, division l1,l2, rational k4) reach S 0.228-0.229 px, which is the floor set by the edges' own non-straightness (0.2-0.6 px bows of real cart members). KB k1..k4 over-fits (ME tile CV of VP consistency +0.2646 +- 0.0879 px^2 worse). The overdispersion (design effect) is c = 12.9 for edges: AIC with the detection noise (0.1 px) or with n = all points always picks the most flexible model and is not usable.
3. **Brown k1,k2 vs a softer periphery (k3-type) is only weakly decidable.** In-sample the block RMS differ by <= 0.005 px; dQAIC / dQBIC (E): Brown k1,k2 13.0 / 3.2, k1,k2,k3 12.4 / 6.1, KB k1,k2 9.8 / 0.0, division l1,l2 11.6 / 1.9. Extrapolation (train without a source region, predict its edges): Brown k1,k2 straightness RMS 0.278 px vs k1,k2,k3 0.229, KB 0.225, rational k4 0.224 (paired dMS KB -0.0134 +- 0.0071, k3 -0.0119 +- 0.0072 px^2, i.e. ~1.7-1.9 sigma; the difference comes from the held-out scene edges at the image periphery). k3 = -0.034 +- 0.017 (E) and puts the fold only 0.039 (normalised radius, ~56 px) beyond the farthest image corner. The softer-periphery forms agree among themselves (KB k1,k2 vs Brown k1,k2,k3: 0.03 / 0.40 / 0.40 px, KB vs division l1,l2: 0.25 / 0.21 / 1.12 px, KB vs rational k4: 0.05 / 0.06 / 0.13 px). KB k1,k2 has the lowest QBIC and the best straightness extrapolation, but its VP-consistency extrapolation (ME source-region CV) is worse by +0.1195 +- 0.0490 px^2, so it fails the strict adequacy rule. The cluster differs from Brown k1,k2 by 0.27 / 1.78 / 2.31 px (k3) and 0.30 / 2.17 / 2.73 px (KB) (rot.-comp. median centre / cart band / corners, ME), largely through a 3-4 px shift of f and pp (f 1434 vs 1431), i.e. inside the f uncertainty (+-26 px). -> k3 is only *consistent* with the data, not determined; it is fixed in the recommended model and the difference is carried as the model-choice part of the mapping uncertainty (Brown k1,k2,k3 is the natural alternative, `method_alt_brown_k1k2k3.json`).
4. **Tangential p1,p2: not determined, fix 0.** Edges: p1 = -0.0010 +- 0.0062, p2 = 0.0015 +- 0.0032, f +-44 px (vs +-30 without). Joint with drawing geometry: p1 = 0.0086 +- 0.0029, f 1374, pp_y 453: stickers improve (LOSO 7.9 vs 10.5 px) while the edges get worse in CV (ME tile VP +0.5891 +- 0.2275, E source straightness +0.0172 +- 0.0056 px^2) - p1,p2 absorb the sticker geometry mismatch. Without the rack tube edge the E fit jumps to f 1364, p1 0.0055, p2 0.0111; without the scene edges to f 516. DIAGNOSTIC: with the relaxed sticker geometry the joint fit gives p1 0.0011, p2 0.0004, f 1431 - the drawing-geometry value is an artefact. Mapping change caused by p1,p2 in ME: 4.37 / 17.57 / 23.19 px.
5. **fx != fy: not determined, fix fx = fy.** E: fx 1383 +- 91, fy 1426 +- 34; ME fx 1439 +- 54, fy 1435 +- 29; dQAIC(E) +1.3; CV neutral. (A 2 MP webcam with square pixels is expected; the data neither confirm nor refute an aspect ratio at the ~3 % level.)
6. **Principal point: free (determined by the edges).** E: (928 +- 7, 502 +- 4) px (cluster-robust); fixing it at the image centre costs dQAIC +21 (E) / +24 (ME) and worsens the tile CV of VP consistency (+22.7485 +- 11.1775 px^2, E). Over the leave-region-out folds (E) pp moves within cx 912-942, cy 492-509 px and f within 1420-1545 px (ME: f 1419-1499) - the focal length from edges is fragile (few VP constraints), which is a parameter, not a model-choice issue. Mapping change pp fixed vs free: 2.95 / 2.34 / 10.40 px. Observation: the relaxed-geometry sticker diagnostic puts pp_y at 392 px, ~110 px above the edge value - the stickers and the edges do not agree on pp_y even after the geometry relaxation.
7. **Non-Brown models convert to OpenCV Brown [k1,k2,p1,p2,k3] with small error** (median centre / band / corners): Kannala-Brandt k1,k2: 0.043 / 0.043 / 0.070 px (max 1.55); division l1,l2: 0.023 / 0.015 / 0.020 px (max 0.36); rational k1,k2 / k4: 0.053 / 0.051 / 0.078 px (max 1.73); rational k1..k6: 0.341 / 0.449 / 0.167 px (max 1.24). With k1,k2 only the conversion error of KB k1,k2 is 0.41 px (band) / 0.71 px (corners).
8. **Recommendation (model adequate to the data): Brown k1,k2 with fx = fy, pp free, p1 = p2 = k3 = 0.** Determined: f, cx, cy and the 2-parameter radial profile (k1, k2 are strongly correlated; the mapping, not the individual k's, is what the data fix). Only consistent: k3 / softer periphery (Brown k1,k2,k3, division l1,l2, rational k4 and - in-sample - KB k1,k2 fit equally well). Not determinable: p1, p2, fx/fy, rational / higher-order terms; they are fixed. Model-choice part of the mapping uncertainty (RMS over the adequate radial alternatives Brown k1,k2,k3, rational k1,k2 / k4, rational k1..k6, division l1,l2; ME, rot.-comp.): centre 0.37, cart band 1.96, corners 2.67 px, vs the parameter part of the same model (sandwich samples): centre 2.14, cart band 9.08, corners 12.67 px; add them in quadrature.

## 1. Fits (same data, same weights for every model)

| model | data | parameters (+- cluster-robust 1 sigma) | block RMS [px] | sticker radial RMS / max [px] | fold margin |
|---|---|---|---|---|---|
| Brown k1 | E | f=1536+-45, cx=910.4+-21, cy=539.5+-39, k1=-0.2722+-0.014 | V 0.578, S 0.286 | - | -0.007 **barrier** |
| Brown k1,k2 | E | f=1436+-30, cx=927.6+-7.3, cy=502.1+-4, k1=-0.3366+-0.015, k2=0.09133+-0.0091 | V 0.361, S 0.229 | - | 75.174 |
| Brown k1,k2,k3 | E | f=1432+-30, cx=931.1+-6.8, cy=505.6+-4.9, k1=-0.355+-0.021, k2=0.1417+-0.032, k3=-0.0337+-0.017 | V 0.358, S 0.229 | - | 0.039 |
| Brown k1,k2,p1,p2 | E | f=1438+-44, cx=919.1+-22, cy=507.6+-32, k1=-0.3386+-0.024, k2=0.09345+-0.014, p1=-0.0009879+-0.0062, p2=0.001537+-0.0032 | V 0.360, S 0.229 | - | 77.216 |
| Brown k1,k2, fx!=fy | E | fx=1383+-91, fy=1426+-34, cx=928+-7.6, cy=510.5+-13, k1=-0.3281+-0.022, k2=0.08508+-0.015 | V 0.360, S 0.229 | - | 69.303 |
| Brown k1,k2,k3,p1,p2, fx!=fy | E | fx=1315+-96, fy=1407+-45, cx=924.8+-15, cy=532.2+-26, k1=-0.3411+-0.033, k2=0.1276+-0.032, p1=-0.001098+-0.0053, p2=0.001605+-0.0021, k3=-0.02581+-0.012 | V 0.354, S 0.229 | - | 0.030 **barrier** |
| Brown k1, pp fixed | E | f=1528+-35, cx=959.5, cy=539.5, k1=-0.2638+-0.011 | V 0.721, S 0.384 | - | 0.029 **barrier** |
| Brown k1,k2, pp fixed | E | f=1437+-27, cx=959.5, cy=539.5, k1=-0.3587+-0.014, k2=0.1076+-0.0085 | V 0.382, S 0.236 | - | 90.495 |
| Brown k1,k2,k3, pp fixed | E | f=1429+-28, cx=959.5, cy=539.5, k1=-0.385+-0.019, k2=0.1821+-0.029, k3=-0.04628+-0.015 | V 0.379, S 0.228 | - | 0.071 |
| rational k1,k2 / k4 | E | f=1431+-30, cx=930.6+-6.5, cy=505.6+-4.5, k1=0.2039+-0.2, k2=-0.02264+-0.042, k4=0.5685+-0.22 | V 0.357, S 0.229 | - | 0.103 |
| rational k1..k6 | E | f=1435+-38, cx=927.8+-6.4, cy=499.9+-6.3, k1=-8.516+-1.1, k2=22.15+-5.6, k3=2.536+-1, k4=-8.14+-1.1, k5=18.94+-5.1, k6=11.01+-3.2 | V 0.332, S 0.234 | - | 0.478 |
| Kannala-Brandt k1,k2 | E | f=1432+-30, cx=931.6+-6.2, cy=506.2+-4.3, k1=-0.02828+-0.021, k2=-0.06198+-0.021 | V 0.357, S 0.229 | - | 0.211 |
| Kannala-Brandt k1..k4 | E | f=1437+-35, cx=927.8+-8, cy=503.2+-5.4, k1=-0.1108+-0.046, k2=0.512+-0.3, k3=-1.408+-0.8, k4=1.124+-0.67 | V 0.349, S 0.231 | - | 37.221 |
| division l1 | E | f=1417+-31, cx=961.6+-10, cy=527.3+-6.6, l1=-0.4352+-0.022 | V 0.384, S 0.239 | - | 0.733 |
| division l1,l2 | E | f=1431+-29, cx=935.1+-5.6, cy=508.6+-3.9, l1=-0.3535+-0.022, l2=-0.1411+-0.024 | V 0.360, S 0.228 | - | 0.505 |
| division l1, pp fixed | E | f=1417+-30, cx=959.5, cy=539.5, l1=-0.4372+-0.019 | V 0.387, S 0.238 | - | 0.735 |
| Brown k1 | ME | f=1536+-43, cx=908.2+-21, cy=539.5+-38, k1=-0.2727+-0.012 | M 6.645, V 0.573, S 0.285 | 9.4 / 15.3 | -0.009 **barrier** |
| Brown k1,k2 | ME | f=1434+-26, cx=927.5+-7.3, cy=501.8+-4, k1=-0.3355+-0.013, k2=0.0906+-0.0084 | M 6.139, V 0.361, S 0.229 | 8.68 / 14.5 | 74.506 |
| Brown k1,k2,k3 | ME | f=1432+-26, cx=930.7+-6.8, cy=505+-4.8, k1=-0.3528+-0.02, k2=0.1368+-0.032, k3=-0.03074+-0.017 | M 6.259, V 0.358, S 0.229 | 8.85 / 14.4 | 0.047 |
| Brown k1,k2,p1,p2 | ME | f=1374+-34, cx=933.9+-19, cy=452.7+-18, k1=-0.3055+-0.017, k2=0.07653+-0.0099, p1=0.008636+-0.0029, p2=-0.001522+-0.0025 | M 4.498, V 0.359, S 0.232 | 6.36 / 12.5 | 61.967 |
| Brown k1,k2, fx!=fy | ME | fx=1439+-54, fy=1435+-29, cx=927.5+-7.2, cy=501.1+-7.3, k1=-0.3363+-0.017, k2=0.09117+-0.012 | M 6.097, V 0.361, S 0.229 | 8.62 / 14.5 | 75.038 |
| Brown k1,k2,k3,p1,p2, fx!=fy | ME | fx=1355+-64, fy=1362+-40, cx=931.2+-15, cy=450.7+-19, k1=-0.3153+-0.026, k2=0.1119+-0.024, p1=0.009477+-0.0033, p2=-0.0007038+-0.002, k3=-0.02325+-0.0086 | M 4.497, V 0.355, S 0.232 | 6.36 / 12.3 | 0.030 **barrier** |
| Brown k1, pp fixed | ME | f=1522+-30, cx=959.5, cy=539.5, k1=-0.2617+-0.0095 | M 6.539, V 0.721, S 0.384 | 9.25 / 14.7 | 0.029 **barrier** |
| Brown k1,k2, pp fixed | ME | f=1436+-24, cx=959.5, cy=539.5, k1=-0.3579+-0.013, k2=0.1072+-0.0077 | M 6.161, V 0.382, S 0.236 | 8.71 / 14.2 | 90.066 |
| Brown k1,k2,k3, pp fixed | ME | f=1430+-25, cx=959.5, cy=539.5, k1=-0.3842+-0.019, k2=0.1798+-0.029, k3=-0.04488+-0.015 | M 6.300, V 0.379, S 0.228 | 8.91 / 14.1 | 0.074 |
| rational k1,k2 / k4 | ME | f=1431+-27, cx=930.4+-6.5, cy=505+-4.4, k1=0.1353+-0.2, k2=-0.008485+-0.04, k4=0.4959+-0.21 | M 6.315, V 0.357, S 0.229 | 8.93 / 14.4 | 0.145 |
| rational k1..k6 | ME | f=1436+-33, cx=927.6+-6.3, cy=498.7+-6, k1=-8.678+-1, k2=22.47+-5, k3=2.414+-0.95, k4=-8.305+-1, k5=19.22+-4.6, k6=10.94+-3 | M 6.349, V 0.332, S 0.234 | 8.98 / 14.3 | 0.450 |
| Kannala-Brandt k1,k2 | ME | f=1431+-27, cx=931.1+-6.2, cy=505.5+-4.2, k1=-0.02646+-0.02, k2=-0.06455+-0.02 | M 6.307, V 0.357, S 0.229 | 8.92 / 14.4 | 0.205 |
| Kannala-Brandt k1..k4 | ME | f=1438+-32, cx=927.5+-8, cy=502.2+-5.4, k1=-0.1017+-0.046, k2=0.4667+-0.3, k3=-1.326+-0.8, k4=1.072+-0.66 | M 6.517, V 0.349, S 0.230 | 9.22 / 14.5 | 35.732 |
| division l1 | ME | f=1419+-28, cx=962+-10, cy=527.3+-6.5, l1=-0.4365+-0.02 | M 6.646, V 0.384, S 0.239 | 9.4 / 14.7 | 0.732 |
| division l1,l2 | ME | f=1431+-26, cx=934.6+-5.6, cy=508+-3.9, l1=-0.351+-0.021, l2=-0.1444+-0.023 | M 6.246, V 0.360, S 0.228 | 8.83 / 14.4 | 0.502 |
| division l1, pp fixed | ME | f=1420+-27, cx=959.5, cy=539.5, l1=-0.4386+-0.018 | M 6.661, V 0.387, S 0.238 | 9.42 / 14.6 | 0.734 |
| Brown k1 | M | f=1374+-25, cx=959.5+-74, cy=462.9+-40, k1=-0.2003+-0.016 | M 4.150 | 5.87 / 10.4 | 0.030 **barrier** |
| Brown k1,k2 | M | f=1309+-17, cx=982.9+-26, cy=286.7+-26, k1=-0.4505+-0.066, k2=0.3686+-0.11 | M 3.773 | 5.34 / 11.7 | 351.677 |
| Brown k1,k2,k3 | M | f=1309+-17, cx=982.2+-23, cy=286.1+-23, k1=-0.5391+-0.081, k2=0.6206+-0.17, k3=-0.2302+-0.081 | M 3.724 | 5.27 / 11.6 | 0.030 **barrier** |
| Brown k1,k2,p1,p2 | M | f=1318+-20, cx=945.9+-27, cy=354.1+-27, k1=-0.4256+-0.084, k2=0.3457+-0.14, p1=0.01055+-0.0025, p2=-0.003194+-0.0033 | M 3.551 | 5.02 / 10.4 | 329.805 |
| Brown k1,k2, fx!=fy | M | fx=1298+-21, fy=1315+-18, cx=978.8+-22, cy=300.1+-24, k1=-0.4213+-0.074, k2=0.3521+-0.12 | M 3.717 | 5.26 / 10.9 | 336.615 |
| Brown k1,k2,k3,p1,p2, fx!=fy | M | fx=1320+-21, fy=1320+-21, cx=949+-24, cy=350.9+-25, k1=-0.5184+-0.11, k2=0.6161+-0.24, p1=0.01074+-0.0024, p2=-0.003151+-0.0036, k3=-0.248+-0.12 | M 3.505 | 4.96 / 10.6 | 0.030 **barrier** |
| Brown k1, pp fixed | M | f=1359+-24, cx=959.5, cy=539.5, k1=-0.2101+-0.007 | M 4.187 | 5.92 / 11.1 | 0.030 **barrier** |
| Brown k1,k2, pp fixed | M | f=1326+-20, cx=959.5, cy=539.5, k1=-0.2459+-0.017, k2=0.02493+-0.0087 | M 4.060 | 5.74 / 11.5 | 0.030 **barrier** |
| Brown k1,k2,k3, pp fixed | M | f=1281+-31, cx=959.5, cy=539.5, k1=0.03459+-0.14, k2=-0.7148+-0.47, k3=0.5764+-0.47 | M 3.860 | 5.46 / 9.4 | 8716.834 |
| rational k1,k2 / k4 | M | f=1418+-1.3e+02, cx=971.5+-15, cy=293.7+-17, k1=7.731+-9.3, k2=-0.4361+-1.2, k4=10.21+-12 | M 3.619 | 5.12 / 10.7 | 0.366 |
| rational k1..k6 | M | f=1313+-2e+02, cx=980.9+-25, cy=286.3+-23, k1=-0.01615+-3.8e+02, k2=0.5525+-2.9e+02, k3=-0.07671+-1e+03, k4=0.6237+-3.8e+02, k5=-0.06442+-4.9e+02, k6=0.21+-1.1e+03 | M 3.688 | 5.22 / 11.4 | 0.030 |
| Kannala-Brandt k1,k2 | M | f=1316+-16, cx=981.1+-20, cy=286.2+-21, k1=-0.3556+-0.1, k2=0.7277+-0.19 | M 3.681 | 5.21 / 11.4 | 6.173 |
| Kannala-Brandt k1..k4 | M | f=1362+-60, cx=972.5+-16, cy=293+-16, k1=-0.9045+-0.73, k2=3.149+-5.2, k3=-3.83+-14, k4=1.421+-14 | M 3.611 | 5.11 / 10.5 | 19.426 |
| division l1 | M | f=1339+-22, cx=961.4+-78, cy=336+-52, l1=-0.2781+-0.031 | M 3.965 | 5.61 / 10.8 | 0.989 |
| division l1,l2 | M | f=1320+-17, cx=977.5+-4.3e-06, cy=284.2+-19, l1=-0.5837+-0.026, l2=0.5562+-4.5e-09 | M 3.729 | 5.27 / 11.3 | 0.030 **barrier** |
| division l1, pp fixed | M | f=1346+-21, cx=959.5, cy=539.5, l1=-0.3221+-0.03 | M 4.234 | 5.99 / 12.2 | 0.944 |

## 2. Information criteria (delta to the best model of the data set)

AIC_own = sum_b n_b ln(RSS_b/n_b) + 2k (each model its own block variances); QAIC = chi2/c + 2k and QBIC = chi2/c + k ln(n/c) with chi2 at the fixed block sigmas and c = overdispersion / design effect (median ratio sandwich/naive variance of the intrinsics of the reference fit: M 1.8, E 12.9, ME 11.5) - residuals of one sticker / one edge are strongly correlated. AIC_det uses the detection noise only (0.1 px) - shown to demonstrate that it always picks the most flexible model.

| model | k_intr | E: dAIC_own / dQAIC / dQBIC / dAIC_det | ME: dAIC_own / dQAIC / dQBIC | M: dAIC_own / dQAIC / dQBIC |
|---|---|---|---|---|
| Brown k1 | 4 | 2781 / 315.4 / 302.1 / 50178 | 2744 / 346.2 / 330.9 | 32.6 / 16.3 / 13.5 |
| Brown k1,k2 | 5 | 291 / 13.0 / 3.2 / 3879 | 282 / 14.6 / 3.0 | 11.0 / 3.9 / 3.3 |
| Brown k1,k2,k3 | 6 | 259 / 12.4 / 6.1 / 3478 | 255 / 14.2 / 6.2 | 9.8 / 4.1 / 5.8 |
| Brown k1,k2,p1,p2 | 7 | 293 / 16.8 / 14.1 / 3835 | 225 / 15.1 / 10.8 | 0.0 / 0.0 / 3.9 |
| Brown k1,k2, fx!=fy | 6 | 284 / 14.3 / 8.0 / 3750 | 284 / 16.6 / 8.6 | 9.3 / 3.8 / 5.5 |
| Brown k1,k2,k3,p1,p2, fx!=fy | 9 | 225 / 15.3 / 19.7 / 2946 | 184 / 15.3 / 18.3 | 0.8 / 2.4 / 10.8 |
| Brown k1, pp fixed | 2 | 4372 / 652.8 / 632.5 / 96707 | 4379 / 734.1 / 711.4 | 30.8 / 13.8 / 6.6 |
| Brown k1,k2, pp fixed | 3 | 597 / 34.2 / 17.4 / 7581 | 589 / 39.0 / 20.0 | 25.2 / 10.7 / 5.7 |
| Brown k1,k2,k3, pp fixed | 4 | 487 / 27.0 / 13.8 / 6648 | 484 / 31.2 / 15.9 | 14.7 / 5.1 / 2.3 |
| rational k1,k2 / k4 | 6 | 249 / 11.6 / 5.3 / 3319 | 247 / 13.5 / 5.5 | 2.7 / 0.4 / 2.0 |
| rational k1..k6 | 9 | 0 / 0.0 / 4.3 / 0 | 0 / 0.0 / 3.1 | 13.4 / 8.8 / 17.2 |
| Kannala-Brandt k1,k2 | 5 | 249 / 9.8 / 0.0 / 3375 | 247 / 11.6 / 0.0 | 4.9 / 0.6 / 0.0 |
| Kannala-Brandt k1..k4 | 7 | 169 / 7.6 / 4.9 / 2199 | 177 / 9.6 / 5.3 | 4.1 / 2.1 / 6.0 |
| division l1 | 4 | 649 / 40.4 / 27.2 / 8061 | 660 / 47.5 / 32.2 | 21.3 / 9.0 / 6.3 |
| division l1,l2 | 5 | 274 / 11.6 / 1.9 / 3707 | 269 / 13.5 / 1.9 | 8.1 / 2.3 / 1.7 |
| division l1, pp fixed | 2 | 673 / 39.0 / 18.6 / 8551 | 685 / 46.4 / 23.8 | 33.6 / 15.8 / 8.5 |

## 3. Cross-validation

Leave-one-sticker-out (LOSO): refit without the sticker, predict its corners (pose of its cart from the other stickers). Leave-one-edge-region-out: source regions (cart310 / cart80 / scene) and 3x3 image tiles; held-out edges are predicted for straightness (S, own line) and for VP consistency (V, direction from the training fit). dMS = mean over held-out clusters of the difference in mean-square error vs Brown k1,k2 (paired), +- its standard error.

| model | LOSO M: RMS [px] | LOSO ME: RMS [px], dMS +- se | E tiles: S / V RMS [px] | E tiles: dMS S, V (+-se) [px^2] | E src: S / V | ME tiles: S / V | adequate |
|---|---|---|---|---|---|---|---|
| Brown k1 | 8.49 | 9.63, -5.29 +- 14.42 | 0.525 / 0.971 | +0.1079 +- 0.0352, -2.0418 +- 1.4597 | 0.528 / - | 0.526 / 1.134 | no |
| Brown k1,k2 | 9.11 | 10.45, +0.00 +- 0.00 | 0.235 / 1.450 | +0.0000 +- 0.0000, +0.0000 +- 0.0000 | 0.278 / - | 0.233 / 0.659 | yes |
| Brown k1,k2,k3 | 8.41 | 10.67, +4.33 +- 1.98 | 0.234 / 1.759 | -0.0007 +- 0.0007, +1.8345 +- 1.3377 | 0.229 / - | 0.232 / 0.617 | yes |
| Brown k1,k2,p1,p2 | 8.51 | 7.90, -46.03 +- 9.14 | 0.492 / 3.474 | +0.0739 +- 0.0635, +14.5804 +- 11.6816 | 0.317 / - | 0.366 / 1.036 | no |
| Brown k1,k2, fx!=fy | 8.99 | 10.66, +3.59 +- 1.92 | 0.238 / 1.332 | +0.0011 +- 0.0005, -0.4392 +- 0.2913 | 0.348 / - | 0.233 / 0.640 | yes |
| Brown k1,k2,k3,p1,p2, fx!=fy | 8.86 | 8.21, -41.27 +- 10.30 | 0.243 / 1.784 | +0.0015 +- 0.0023, +2.1726 +- 1.4913 | 0.313 / - | 0.295 / 0.954 | no |
| Brown k1, pp fixed | 8.59 | 11.11, +19.26 +- 12.41 | 23.549 / 6.603 | +210.3258 +- 210.2097, +96.7922 +- 43.6433 | 0.677 / - | 0.744 / 4.470 | no |
| Brown k1,k2, pp fixed | 8.40 | 10.61, +1.17 +- 4.83 | 0.238 / 3.589 | -0.0000 +- 0.0022, +22.7485 +- 11.1775 | 0.349 / - | 0.238 / 2.266 | no |
| Brown k1,k2,k3, pp fixed | 8.28 | 10.83, +5.94 +- 4.29 | 0.265 / 4.323 | +0.0054 +- 0.0055, +30.0458 +- 17.5921 | 0.244 / - | 0.272 / 2.082 | no |
| rational k1,k2 / k4 | 8.21 | 10.76, +6.48 +- 2.97 | 0.230 / 1.958 | -0.0016 +- 0.0008, +3.1200 +- 2.2076 | 0.224 / - | 0.228 / 0.616 | yes |
| rational k1..k6 | 7.88 | 10.81, +9.02 +- 6.25 | 0.229 / 2.272 | -0.0012 +- 0.0019, +4.8374 +- 3.9022 | 0.263 / - | 0.227 / 0.612 | yes |
| Kannala-Brandt k1,k2 | 8.11 | 10.74, +5.97 +- 2.68 | 0.225 / 3.227 | -0.0027 +- 0.0011, +12.5642 +- 10.3702 | 0.225 / - | 0.224 / 0.602 | no |
| Kannala-Brandt k1..k4 | 7.47 | 11.10, +14.61 +- 6.49 | 0.294 / 3.460 | +0.0126 +- 0.0099, +14.9303 +- 11.8494 | 0.399 / - | 0.274 / 0.742 | no |
| division l1 | 8.78 | 11.39, +18.57 +- 8.00 | 0.310 / 5.642 | +0.0191 +- 0.0076, +43.6296 +- 36.9552 | 0.322 / - | 0.307 / 1.291 | no |
| division l1,l2 | 7.49 | 10.65, +3.80 +- 1.73 | 0.233 / 2.070 | -0.0012 +- 0.0011, +3.7012 +- 2.6986 | 0.233 / - | 0.231 / 0.629 | yes |
| division l1, pp fixed | 8.83 | 11.43, +19.08 +- 8.44 | 0.251 / 5.963 | +0.0034 +- 0.0027, +52.5603 +- 39.4613 | 0.252 / - | 0.251 / 1.683 | no |

Cross-data prediction (lens of one data set, poses/rotations refit on the other):

| model | lens(E) -> stickers: radial RMS [px] | lens(M) -> edges: S / V RMS [px] |
|---|---|---|
| Brown k1 | 8.12 | 0.397 / 0.790 |
| Brown k1,k2 | 7.26 | 2.590 / 0.791 |
| Brown k1,k2,k3 | 7.21 | 1.902 / 0.950 |
| Brown k1,k2,p1,p2 | 7.34 | 2.085 / 0.599 |
| Brown k1,k2, fx!=fy | 8.55 | 2.507 / 0.716 |
| Brown k1,k2,k3,p1,p2, fx!=fy | 10.90 | 1.654 / 0.758 |
| Brown k1, pp fixed | 7.89 | 0.397 / 0.753 |
| Brown k1,k2, pp fixed | 7.60 | 0.595 / 0.557 |
| Brown k1,k2,k3, pp fixed | 7.42 | 2.267 / 0.848 |
| rational k1,k2 / k4 | 7.17 | 1.102 / 2.355 |
| rational k1..k6 | 6.92 | 1.561 / 1.129 |
| Kannala-Brandt k1,k2 | 7.20 | 1.741 / 1.194 |
| Kannala-Brandt k1..k4 | 7.06 | 2.831 / 1.984 |
| division l1 | 7.30 | 0.578 / 0.628 |
| division l1,l2 | 7.26 | 2.004 / 0.886 |
| division l1, pp fixed | 7.29 | 0.425 / 0.513 |

## 4. Mapping differences between models (not parameters)

Displacement of each model's mapping relative to Brown k1,k2 fitted to the same data (rays of the reference, projected by the model; rotation-compensated = best common camera rotation removed, raw = none). Median |d| over the region (centre r < 150 px, cart band = the two cart footprints, corners = 200 x 150 px corner boxes).

| model | ME rot-comp: centre / band / corners | ME raw: centre / band / corners | E rot-comp: centre / band / corners | M rot-comp: centre / band / corners |
|---|---|---|---|---|
| Brown k1 | 7.52 / 39.63 / 65.18 | 44.19 / 55.20 / 76.57 | 7.38 / 38.78 / 64.59 | 8.91 / 41.41 / 65.47 |
| Brown k1,k2,k3 | 0.27 / 1.78 / 2.31 | 4.48 / 4.62 / 5.17 | 0.36 / 2.31 / 2.99 | 6.28 / 6.65 / 28.64 |
| Brown k1,k2,p1,p2 | 4.37 / 17.57 / 23.19 | 51.11 / 50.03 / 43.48 | 1.15 / 0.92 / 2.86 | 6.79 / 8.21 / 20.57 |
| Brown k1,k2, fx!=fy | 0.19 / 1.36 / 2.71 | 0.78 / 1.61 / 2.80 | 2.07 / 14.81 / 29.63 | 1.94 / 3.22 / 8.65 |
| Brown k1,k2,k3,p1,p2, fx!=fy | 5.36 / 24.18 / 32.91 | 53.32 / 53.89 / 49.68 | 5.34 / 36.07 / 67.13 | 3.39 / 5.02 / 25.27 |
| Brown k1, pp fixed | 6.50 / 35.36 / 65.99 | 52.88 / 62.36 / 79.43 | 6.71 / 36.73 / 67.47 | 9.96 / 32.71 / 57.50 |
| Brown k1,k2, pp fixed | 2.95 / 2.34 / 10.40 | 49.48 / 49.41 / 48.57 | 2.95 / 2.37 / 10.45 | 9.07 / 17.60 / 57.98 |
| Brown k1,k2,k3, pp fixed | 2.97 / 4.66 / 12.10 | 49.27 / 49.11 / 48.56 | 2.99 / 5.38 / 12.68 | 7.46 / 16.18 / 74.02 |
| rational k1,k2 / k4 | 0.28 / 2.25 / 2.63 | 4.22 / 4.51 / 5.08 | 0.40 / 3.05 / 3.60 | 18.84 / 14.28 / 64.85 |
| rational k1..k6 | 0.30 / 0.65 / 0.68 | 3.10 / 3.15 / 3.31 | 0.25 / 1.75 / 1.22 | 9.30 / 10.20 / 44.16 |
| Kannala-Brandt k1,k2 | 0.30 / 2.17 / 2.73 | 5.07 / 5.26 / 5.62 | 0.40 / 2.73 / 3.56 | 9.97 / 11.05 / 45.64 |
| Kannala-Brandt k1..k4 | 0.21 / 1.64 / 1.43 | 0.43 / 1.63 / 1.39 | 0.11 / 2.84 / 2.87 | 16.58 / 14.45 / 77.27 |
| division l1 | 2.51 / 10.33 / 18.27 | 42.35 / 42.16 / 39.17 | 2.55 / 11.50 / 19.96 | 4.13 / 25.36 / 43.61 |
| division l1,l2 | 0.52 / 2.32 / 3.89 | 9.27 / 9.28 / 8.91 | 0.58 / 2.83 / 4.63 | 1.90 / 1.87 / 21.84 |
| division l1, pp fixed | 3.08 / 10.43 / 19.29 | 48.91 / 49.20 / 49.38 | 3.18 / 11.76 / 21.11 | 8.75 / 20.81 / 48.18 |

Same model, different data (rot-comp median centre / band / corners [px]):

* B_k1k2:E->ME: 0.13 / 0.57 / 0.74
* B_k1k2:M->ME: 11.34 / 51.37 / 57.37
* B_k1k2:M->E: 11.47 / 51.96 / 58.06
* KB_k1k2:E->ME: 0.04 / 0.04 / 0.14
* KB_k1k2:M->ME: 17.07 / 58.46 / 53.37
* KB_k1k2:M->E: 17.10 / 58.39 / 53.48
* B_k1k2k3:E->ME: 0.04 / 0.04 / 0.12
* B_k1k2k3:M->ME: 14.02 / 54.18 / 52.64
* B_k1k2k3:M->E: 14.07 / 54.20 / 52.62
* D_l1l2:E->ME: 0.05 / 0.05 / 0.10
* D_l1l2:M->ME: 12.56 / 47.52 / 55.66
* D_l1l2:M->E: 12.61 / 47.53 / 55.69

Parameter-sample mapping uncertainty (cluster-robust sandwich covariance, rotation-compensated, 1 sigma, median over region) [px]:

| model | E: centre / band / corners | ME: centre / band / corners |
|---|---|---|
| Brown k1 | n/a (fold barrier active) | n/a (fold barrier active) |
| Brown k1,k2 | 2.15 / 9.14 / 12.82 | 2.14 / 9.08 / 12.67 |
| Brown k1,k2,k3 | 2.27 / 9.67 / 13.47 | 1.89 / 8.06 / 11.25 |
| Brown k1,k2,p1,p2 | 3.43 / 11.78 / 17.19 | 3.63 / 12.34 / 18.73 |
| Brown k1,k2, fx!=fy | 4.62 / 27.90 / 52.78 | 3.01 / 15.98 / 27.67 |
| Brown k1,k2,k3,p1,p2, fx!=fy | n/a (fold barrier active) | n/a (fold barrier active) |
| Brown k1, pp fixed | n/a (fold barrier active) | n/a (fold barrier active) |
| Brown k1,k2, pp fixed | 2.00 / 8.84 / 12.67 | 1.67 / 7.39 / 10.92 |
| Brown k1,k2,k3, pp fixed | 2.04 / 9.00 / 13.00 | 1.88 / 8.33 / 12.05 |
| rational k1,k2 / k4 | 2.45 / 10.35 / 13.94 | 2.08 / 8.70 / 12.08 |
| rational k1..k6 | 3.24 / 12.53 / 19.37 | 2.71 / 10.67 / 14.95 |
| Kannala-Brandt k1,k2 | 2.25 / 9.61 / 13.42 | 1.82 / 7.72 / 10.77 |
| Kannala-Brandt k1..k4 | 2.61 / 10.85 / 15.27 | 2.14 / 8.76 / 12.39 |
| division l1 | 2.54 / 10.64 / 16.13 | 2.09 / 8.54 / 13.31 |
| division l1,l2 | 2.08 / 8.90 / 12.43 | 1.79 / 7.63 / 10.74 |
| division l1, pp fixed | 2.15 / 9.46 / 14.74 | 2.00 / 8.78 / 13.56 |

**Model-choice part of the mapping uncertainty** (RMS over the adequate models Brown k1,k2, Brown k1,k2,k3, rational k1,k2 / k4, rational k1..k6, division l1,l2 of the rot.-comp. difference to the recommended Brown k1,k2, median over region):

* E: centre 0.42, cart band 2.70, corners 3.38 px (max over models: 0.58 / 3.05 / 4.70 px)
* ME: centre 0.37, cart band 1.96, corners 2.67 px (max over models: 0.52 / 2.33 / 3.92 px)

## 5. Conversion of non-Brown models to OpenCV Brown

evaltools.brown_from_mapping on a 20 px grid over the full image (f, pp and the coefficients refitted); error = |pixel difference| of the same rays, median (max) over the region.

| model (data) | target | K f / pp | dist [k1,k2,p1,p2,k3] | error centre / band / corners: median (max) [px] |
|---|---|---|---|---|
| rational k1,k2 / k4 (ME) | full | 1430.5 / (930.4, 505.0) | [-0.3532, 0.1396, 0.0, -0.0, -0.0325] | 0.053 (0.08) / 0.051 (0.09) / 0.078 (1.73) |
| rational k1,k2 / k4 (ME) | k1k2k3 | 1430.5 / (930.4, 505.0) | [-0.3532, 0.1395, 0.0, 0.0, -0.0325] | 0.053 (0.08) / 0.051 (0.09) / 0.080 (1.73) |
| rational k1,k2 / k4 (ME) | k1k2 | 1426.5 / (930.4, 505.0) | [-0.3293, 0.0857, 0.0, 0.0, 0.0] | 0.337 (0.55) / 0.420 (0.63) / 0.726 (11.02) |
| rational k1..k6 (ME) | full | 1431.2 / (927.6, 498.7) | [-0.3324, 0.0999, 0.0, 0.0, -0.0124] | 0.341 (0.53) / 0.449 (0.75) / 0.167 (1.24) |
| rational k1..k6 (ME) | k1k2k3 | 1431.2 / (927.6, 498.7) | [-0.3324, 0.0999, 0.0, 0.0, -0.0124] | 0.342 (0.53) / 0.449 (0.75) / 0.169 (1.25) |
| rational k1..k6 (ME) | k1k2 | 1429.6 / (927.6, 498.7) | [-0.3232, 0.0793, 0.0, 0.0, 0.0] | 0.452 (0.71) / 0.398 (0.75) / 0.265 (6.26) |
| Kannala-Brandt k1,k2 (ME) | full | 1430.9 / (931.1, 505.5) | [-0.354, 0.1398, 0.0, -0.0, -0.0321] | 0.043 (0.07) / 0.043 (0.08) / 0.070 (1.55) |
| Kannala-Brandt k1,k2 (ME) | k1k2k3 | 1430.9 / (931.1, 505.5) | [-0.354, 0.1398, 0.0, 0.0, -0.0321] | 0.043 (0.07) / 0.043 (0.08) / 0.071 (1.55) |
| Kannala-Brandt k1,k2 (ME) | k1k2 | 1427.0 / (931.1, 505.5) | [-0.3305, 0.0869, 0.0, 0.0, 0.0] | 0.319 (0.52) / 0.405 (0.61) / 0.706 (10.69) |
| Kannala-Brandt k1..k4 (ME) | full | 1430.3 / (927.5, 502.1) | [-0.3353, 0.0946, 0.0, -0.0, -0.0025] | 0.498 (0.70) / 0.321 (0.70) / 0.308 (3.11) |
| Kannala-Brandt k1..k4 (ME) | k1k2k3 | 1430.3 / (927.5, 502.1) | [-0.3352, 0.0946, 0.0, 0.0, -0.0025] | 0.495 (0.70) / 0.320 (0.70) / 0.306 (3.12) |
| Kannala-Brandt k1..k4 (ME) | k1k2 | 1430.0 / (927.5, 502.1) | [-0.3335, 0.0905, 0.0, 0.0, 0.0] | 0.515 (0.73) / 0.328 (0.73) / 0.273 (2.36) |
| division l1 (ME) | full | 1416.3 / (962.0, 527.3) | [-0.4067, 0.2324, 0.0, -0.0, -0.0716] | 0.188 (0.26) / 0.163 (0.29) / 0.231 (2.85) |
| division l1 (ME) | k1k2k3 | 1416.3 / (962.0, 527.3) | [-0.4067, 0.2324, 0.0, 0.0, -0.0716] | 0.187 (0.25) / 0.163 (0.28) / 0.229 (2.86) |
| division l1 (ME) | k1k2 | 1408.4 / (962.0, 527.3) | [-0.3574, 0.1183, 0.0, 0.0, 0.0] | 0.752 (1.07) / 0.847 (1.33) / 1.307 (11.75) |
| division l1,l2 (ME) | full | 1430.9 / (934.6, 508.0) | [-0.3562, 0.1384, 0.0, 0.0, -0.0285] | 0.023 (0.03) / 0.015 (0.03) / 0.020 (0.36) |
| division l1,l2 (ME) | k1k2k3 | 1430.9 / (934.6, 508.0) | [-0.3562, 0.1384, 0.0, 0.0, -0.0285] | 0.023 (0.03) / 0.015 (0.03) / 0.020 (0.36) |
| division l1,l2 (ME) | k1k2 | 1427.6 / (934.6, 508.0) | [-0.336, 0.0922, 0.0, 0.0, 0.0] | 0.211 (0.35) / 0.313 (0.51) / 0.624 (9.40) |
| division l1, pp fixed (ME) | full | 1416.8 / (959.5, 539.5) | [-0.4086, 0.2343, 0.0, 0.0, -0.0724] | 0.189 (0.24) / 0.163 (0.28) / 0.230 (2.27) |
| division l1, pp fixed (ME) | k1k2k3 | 1416.8 / (959.5, 539.5) | [-0.4086, 0.2343, 0.0, 0.0, -0.0724] | 0.189 (0.24) / 0.163 (0.28) / 0.230 (2.27) |
| division l1, pp fixed (ME) | k1k2 | 1408.8 / (959.5, 539.5) | [-0.3588, 0.1191, 0.0, 0.0, 0.0] | 0.760 (1.01) / 0.856 (1.33) / 1.319 (10.16) |
| rational k1,k2 / k4 (E) | full | 1430.1 / (930.6, 505.6) | [-0.3553, 0.145, 0.0, -0.0, -0.0359] | 0.066 (0.10) / 0.064 (0.11) / 0.097 (2.13) |
| rational k1,k2 / k4 (E) | k1k2k3 | 1430.1 / (930.6, 505.6) | [-0.3553, 0.145, 0.0, 0.0, -0.0359] | 0.066 (0.10) / 0.064 (0.11) / 0.098 (2.13) |
| rational k1,k2 / k4 (E) | k1k2 | 1425.7 / (930.6, 505.6) | [-0.3287, 0.0854, 0.0, 0.0, 0.0] | 0.380 (0.62) / 0.471 (0.71) / 0.803 (12.19) |
| rational k1..k6 (E) | full | 1429.7 / (927.8, 499.9) | [-0.3331, 0.1015, 0.0, 0.0, -0.013] | 0.364 (0.57) / 0.465 (0.76) / 0.160 (1.02) |
| rational k1..k6 (E) | k1k2k3 | 1429.7 / (927.8, 499.9) | [-0.3331, 0.1015, 0.0, 0.0, -0.013] | 0.365 (0.56) / 0.465 (0.77) / 0.161 (1.03) |
| rational k1..k6 (E) | k1k2 | 1428.1 / (927.8, 499.9) | [-0.3234, 0.0799, 0.0, 0.0, 0.0] | 0.481 (0.75) / 0.417 (0.81) / 0.257 (6.24) |
| Kannala-Brandt k1,k2 (E) | full | 1431.1 / (931.6, 506.2) | [-0.3554, 0.1419, 0.0, -0.0, -0.0329] | 0.046 (0.07) / 0.046 (0.08) / 0.073 (1.61) |
| Kannala-Brandt k1,k2 (E) | k1k2k3 | 1431.1 / (931.6, 506.2) | [-0.3554, 0.1419, 0.0, 0.0, -0.0329] | 0.046 (0.07) / 0.046 (0.08) / 0.074 (1.61) |
| Kannala-Brandt k1,k2 (E) | k1k2 | 1427.1 / (931.6, 506.2) | [-0.3314, 0.0878, 0.0, 0.0, 0.0] | 0.326 (0.53) / 0.413 (0.62) / 0.716 (10.77) |
| Kannala-Brandt k1..k4 (E) | full | 1428.8 / (927.8, 503.2) | [-0.3361, 0.0972, 0.0, -0.0, -0.004] | 0.543 (0.76) / 0.352 (0.77) / 0.341 (3.56) |
| Kannala-Brandt k1..k4 (E) | k1k2k3 | 1428.8 / (927.8, 503.2) | [-0.336, 0.0971, 0.0, 0.0, -0.004] | 0.541 (0.76) / 0.352 (0.76) / 0.340 (3.57) |
| Kannala-Brandt k1..k4 (E) | k1k2 | 1428.4 / (927.8, 503.2) | [-0.3332, 0.0907, 0.0, 0.0, 0.0] | 0.574 (0.81) / 0.366 (0.82) / 0.292 (2.37) |
| division l1 (E) | full | 1414.2 / (961.6, 527.3) | [-0.4055, 0.231, 0.0, -0.0, -0.071] | 0.187 (0.26) / 0.163 (0.29) / 0.231 (2.82) |
| division l1 (E) | k1k2k3 | 1414.2 / (961.6, 527.3) | [-0.4055, 0.231, 0.0, 0.0, -0.071] | 0.187 (0.25) / 0.163 (0.28) / 0.231 (2.83) |
| division l1 (E) | k1k2 | 1406.2 / (961.6, 527.3) | [-0.3564, 0.1176, 0.0, 0.0, 0.0] | 0.752 (1.07) / 0.848 (1.33) / 1.309 (11.67) |
| division l1,l2 (E) | full | 1431.3 / (935.1, 508.6) | [-0.3578, 0.141, 0.0, -0.0, -0.0296] | 0.017 (0.02) / 0.012 (0.02) / 0.022 (0.48) |
| division l1,l2 (E) | k1k2k3 | 1431.3 / (935.1, 508.6) | [-0.3578, 0.141, 0.0, 0.0, -0.0296] | 0.017 (0.02) / 0.012 (0.02) / 0.022 (0.48) |
| division l1,l2 (E) | k1k2 | 1427.9 / (935.1, 508.5) | [-0.3368, 0.0931, 0.0, 0.0, 0.0] | 0.225 (0.37) / 0.326 (0.53) / 0.638 (9.54) |
| division l1, pp fixed (E) | full | 1414.3 / (959.5, 539.5) | [-0.4073, 0.2328, 0.0, 0.0, -0.0717] | 0.189 (0.24) / 0.164 (0.28) / 0.231 (2.27) |
| division l1, pp fixed (E) | k1k2k3 | 1414.3 / (959.5, 539.5) | [-0.4073, 0.2328, 0.0, 0.0, -0.0717] | 0.189 (0.24) / 0.164 (0.28) / 0.230 (2.28) |
| division l1, pp fixed (E) | k1k2 | 1406.3 / (959.5, 539.5) | [-0.3576, 0.1183, 0.0, 0.0, 0.0] | 0.761 (1.01) / 0.857 (1.33) / 1.321 (10.17) |
