# Vanishing points (independent, 40_lines_indep_b_vp.py)

method: vanishing_points (independent implementation 40_lines_indep; VP MLE per direction group, f from orthogonality)

model: fx=fy, zero skew; pp = distortion centre of the plumb-line fit (k1,k2, centre free); f from cart-X _|_ cart-vertical of both carts (common vertical from the posts of both carts); pp NOT determinable from the vanishing points alone

* f = 1479.3 px (1-sigma 46.7), pp = (929.3, 426.5) (1-sigma 5.3, 23.7)
* dist = [-0.3493, 0.0937, 0.0000, 0.0000, 0.0000]; k1 1-sigma 0.0239, k2 1-sigma 0.0187
* rms 0.388 px; mapping uncertainty (rotation-compensated, px) {'centre': 3.767, 'cart_band': 14.747, 'corners': 21.263}

f from cart-X _|_ common post vertical (both carts, chi2 1.6/1), pp = plumb distortion centre.
pp fixed at the image centre (with the centre-fixed plumb distortion): f = 1478.7 +- 38.2.
pp cannot be determined from the VPs: floor lines are inconsistent with the vertical (chi2 350-550/4), end boards not parallel to the cart axes.
For the OpenCV-consistent model (distortion centre = pp) f is nearly independent of pp_y: 1469 (pp_y 340) .. 1489 (540) .. 1503 (660) (results/40_lines_indep_ppscan.png); only mixed models (centre fixed, pp moved) trade 1.7 px f per px pp_y.
Only cart 310 (310X _|_ 310 posts): f = 1408; with the scene verticals instead of the posts: 1521.
Self-consistent (distortion centre = pp) solutions agree for all plumb variants (1479-1490); mixed (centre != pp) differ by up to 200 px.
Cart-post vertical vs scene verticals: 0.79 deg (plumb centre free), 0.38 deg (centre fixed) -> carts' verticals and building verticals not parallel within noise.
Synthetic round trip (40 trials): VP f bias +4.9, sd 24.8 px (claimed bootstrap sd 46.7, robust 33.8 -> conservative).
Sensitivity: +-0.5 px dark-side edge shift: df 0.08 px; random per-edge tilts (0.3 px at the ends): df 20.6 px rms.
Orchestrator LineCal on the same edges (separate cart frames, point-level VP): vp vp_ourplumb_c_ppfree f=1429, vp_ourplumb_c_ppC0 f=1446, vp_lsplumb_fixed_ppfree f=1438, vp_lsplumb_fixed_ppC0 f=1431 -> ~40 px lower, mainly because cart 80's vertical then rests on its own two short post fragments (our frames-free variant: 1438).
