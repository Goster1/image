# Plumb-line (independent, 40_lines_indep_a_plumb.py)

method: plumb_line (independent implementation 40_lines_indep; distortion from the straightness of verified straight edges)

model: k1,k2 radial (pixel units: k1/f^2, k2/f^4), distortion centre FREE (= principal point in the OpenCV form); f taken from the vanishing-point result (S1, pp = this centre) because straightness alone determines only k_i/f^(2i)

* f = 1479.3 px (1-sigma 47.5), pp = (929.3, 426.5) (1-sigma 9.9, 50.2)
* dist = [-0.3493, 0.0937, 0.0000, 0.0000, 0.0000]; k1 1-sigma 0.0248, k2 1-sigma 0.0201
* rms 0.202 px; mapping uncertainty (rotation-compensated, px) {'centre': 3.454, 'cart_band': 2.522, 'corners': 7.857}

**Review correction** (40_lines_indep_j_review.py): the 1-sigma values above are statistical (+) systematic; statistical only: f 46.7, cx 6.8, cy 26.7, k1 0.0239, k2 0.0187; mapping {'centre': 1.816, 'cart_band': 1.275, 'corners': 5.001}; systematic: f 8.6, cx 7.2, cy 42.5, mapping {'centre': 2.938, 'cart_band': 2.176, 'corners': 6.06}.

* numbers reproduced; no code bug that changes the results
* principal point / distortion-centre height is NOT determined to the bootstrap sd: it moves with the radial model and with the floor-line assumption (range of the credible variants: y = 426..512); x is determined to ~+-10 px
* blue floor line (one painted line visible left of cart 310 and between the carts): with the main lens the piece between the carts lies +10.1 px off the straight line through the left piece and is rotated by 1.12 deg; with the pieces merged into one straight edge the free distortion centre moves to y ~510 without cost for the other edges (k1k2k3)
* focal length is robust to these variants (half-range 8.6 px for this method)
* stratified leave-one-VP-member-out jackknife (joint): sd f/cx/cy = [30.9, 3.4, 13.0] (bootstrap [20.6, 4.5, 22.0]); jackknife calibrated on synthetic data is ~unbiased but very noisy (8..37 px for a true 21.6 px)
* cart split (joint, common vertical): cart-310 X only f = 1440, cart-80 X only f = 1489

Straightness determines only k_i/f^(2i): k1/f^2 = -1.5961e-07 +- 3.4e-09 px^-2, k2/f^4 = 1.957e-14 +- 2.0e-15 px^-4 (bootstrap); OpenCV k's use f from the VP method.
Model choice (leave-one-member-out CV rms, px): k1 0.5133, k1k2 0.2532, k1k2k3 0.2573, k1_c 0.4778, k1k2_c 0.2057, k1k2k3_c 0.2136, k1k2_t 0.3111, k1k2_c_t 0.3014, div1 0.2265, div2 0.2077, div1_c 0.2386, div2_c 0.2161
Distortion centre FREE: (929.3, 426.5) +- (6.8, 26.7) (bootstrap); the vertical position depends on the radial model (div2_c 468, k1k2k3_c 456, k1k2+tangential 516) -> cx determined (~930), cy only 'consistent' (430-520).
k1-only folds at the image corner (runs into the fold barrier, rms 0.51 px) -> inadequate; k3 not determinable (runs towards the fold).
Per-region: scene_k1k2_c: cx=930.6, cy=431, a1=-0.1577, a2=0.01862; cart310_k1k2_c: cx=947.9, cy=449.1, a1=-0.1553, a2=0.02221; cart80_k1k2_c: cx=944.9, cy=482.3, a1=-0.1807, a2=0.0376
Per-region with the centre FIXED at the image centre disagree (a1: scene -0.1686, cart310 -0.1688, cart80 -0.1916, +-0.004-0.006) but agree with a free centre -> evidence for a decentred distortion (or a different radial profile: div2 with fixed centre fits as well, LOCO 0.2077).
Rejected as curved: cart80_x_A_front_in, cart80_x_E_front, cart80_x_E_front_in (cart-80 lowest lip E: both boundaries arched by ~0.75-0.8 px -> physically bent lip; A_in = inner shading boundary of the rounded A lip highlight, wobbly; its sharp partner A_out is kept). Crops: results/40_lines_indep_rejected_strips.png.
Same edges, orchestrator LineCal plumb: k@1300 (-0.2890, 0.0698) fixed centre, (-0.2697, 0.0559) centre (929.5, 426.7) -> identical to ours within 0.001 / 1 px.
Synthetic round trip (40 trials, truth pp shifted +15,+15): centre error sd (6.1, 22.0) px, bias (-1.2, +1.1); claimed (bootstrap) (6.8, 26.7).
