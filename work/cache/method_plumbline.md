# Plumb-line (independent, 40_lines_indep_a_plumb.py)

method: plumb_line (independent implementation 40_lines_indep; distortion from the straightness of verified straight edges)

model: k1,k2 radial (pixel units: k1/f^2, k2/f^4), distortion centre FREE (= principal point in the OpenCV form); f taken from the vanishing-point result (S1, pp = this centre) because straightness alone determines only k_i/f^(2i)

* f = 1479.3 px (1-sigma 46.7), pp = (929.3, 426.5) (1-sigma 6.8, 26.7)
* dist = [-0.3493, 0.0937, 0.0000, 0.0000, 0.0000]; k1 1-sigma 0.0239, k2 1-sigma 0.0187
* rms 0.202 px; mapping uncertainty (rotation-compensated, px) {'centre': 1.816, 'cart_band': 1.275, 'corners': 5.001}

Straightness determines only k_i/f^(2i): k1/f^2 = -1.5961e-07 +- 3.4e-09 px^-2, k2/f^4 = 1.957e-14 +- 2.0e-15 px^-4 (bootstrap); OpenCV k's use f from the VP method.
Model choice (leave-one-member-out CV rms, px): k1 0.5133, k1k2 0.2532, k1k2k3 0.2573, k1_c 0.4778, k1k2_c 0.2057, k1k2k3_c 0.2136, k1k2_t 0.3111, k1k2_c_t 0.3014, div1 0.2265, div2 0.2077, div1_c 0.2386, div2_c 0.2161
Distortion centre FREE: (929.3, 426.5) +- (6.8, 26.7) (bootstrap); the vertical position depends on the radial model (div2_c 468, k1k2k3_c 456, k1k2+tangential 516) -> cx determined (~930), cy only 'consistent' (430-520).
k1-only folds at the image corner (runs into the fold barrier, rms 0.51 px) -> inadequate; k3 not determinable (runs towards the fold).
Per-region: scene_k1k2_c: cx=930.6, cy=431, a1=-0.1577, a2=0.01862; cart310_k1k2_c: cx=947.9, cy=449.1, a1=-0.1553, a2=0.02221; cart80_k1k2_c: cx=944.9, cy=482.3, a1=-0.1807, a2=0.0376
Per-region with the centre FIXED at the image centre disagree (a1: scene -0.1686, cart310 -0.1688, cart80 -0.1916, +-0.004-0.006) but agree with a free centre -> evidence for a decentred distortion (or a different radial profile: div2 with fixed centre fits as well, LOCO 0.2077).
Rejected as curved: cart80_x_A_front_in, cart80_x_E_front, cart80_x_E_front_in (cart-80 lowest lip E: both boundaries arched by ~0.75-0.8 px -> physically bent lip; A_in = inner shading boundary of the rounded A lip highlight, wobbly; its sharp partner A_out is kept). Crops: results/40_lines_indep_rejected_strips.png.
Same edges, orchestrator LineCal plumb: k@1300 (-0.2890, 0.0698) fixed centre, (-0.2697, 0.0559) centre (929.5, 426.7) -> identical to ours within 0.001 / 1 px.
Synthetic round trip (40 trials, truth pp shifted +15,+15): centre error sd (6.1, 22.0) px, bias (-1.2, +1.1); claimed (bootstrap) (6.8, 26.7).
