# Joint line self-calibration (independent, 40_lines_indep_c_joint.py)

method: lines_joint (independent implementation 40_lines_indep: straightness + VP consistency with a random member-direction error)

model: fx=fy, zero skew, pp free (= distortion centre), k1,k2; common vertical for both carts' posts, cart X axes _|_ vertical, scene verticals own direction

* f = 1473.0 px (1-sigma 20.6), pp = (930.9, 437.8) (1-sigma 4.5, 22.0)
* dist = [-0.3486, 0.0950, 0.0000, 0.0000, 0.0000]; k1 1-sigma 0.0119, k2 1-sigma 0.0120
* rms 0.202 px; mapping uncertainty (rotation-compensated, px) {'centre': 2.15, 'cart_band': 6.762, 'corners': 10.553}

Cost: straightness (points, distorted-image distances, weight 1/sig_pt with a within-edge correlation inflation) + one angular VP-consistency residual per member (random direction error sig_psi).
Variants (f, pp): JRE_k1k2_ppfree: 1473, (931,438); JRE_k1k2_ppC0: 1475, (960,540); JRE_div2_ppfree: 1478, (940,479); JRE_k1k2k3_ppfree: 1480, (936,467); JRE_k1k2_framesfree: 1438, (931,436); JRE_k1k2_wvtied: 1504, (934,460); JRE_k1k2_sigpt_noinfl: 1472, (930,432); JRE_k1k2_sigpt_2xinfl: 1475, (934,458); JRE_k1k2_noWV: 1472, (930,436); JPT_k1k2_ppfree: 1492, (935,533); JPT_k1k2_ppC0: 1486, (960,540); JPT_k1k2_framesfree: 1435, (932,505)
f is stable (1470-1490) under all straightness/VP weightings and distortion models with tied cart frames; separate cart frames give 1425-1440 (cart 80's own posts are 2 short edges); scene verticals tied to the posts: 1504.
pp_y depends on the radial model (k1k2 438, div2 479, k1k2k3 467); f does not.
Focal profile (delta chi2 = 1): +-21 px.
Synthetic round trip (40 trials, truth pp +15,+15): f bias +4.1 sd 21.4 (claimed 20.6); pp bias (-1.2, +1.5) sd (5.9, 21.3) (claimed 4.5, 22.0); mapping error vs truth centre 2.1, cart_band 6.7, corners 9.7 px (claimed centre 2.1, cart_band 6.8, corners 10.6).
Orchestrator LineCal joint on the same edges: joint_ppfree_forward f=1423 pp=(932,503), joint_ppfree_jacobian f=1425 pp=(933,509), joint_ppC0_forward f=1430 pp=(960,540), joint_ppC0_jacobian f=1429 pp=(960,540); our analogue (separate frames, point-level VP) JPT_k1k2_framesfree: 1435, (932, 505).
