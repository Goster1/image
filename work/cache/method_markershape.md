# Method: sticker shape (horizontal squares)

Every sticker is modelled as a flat horizontal square with a free 3D position; all stickers of a cart share the cart rotation (axis-aligned, rot_k * 90 deg), the nadir from the vertical post edges ties the cart Z axis; distortion from the plumb-line fit (pixel units). Positions, heights and spacings of the stickers are not used (42_geomdiag_e_markershape.py).

| variant | f [px] | +-cov | pp | corner RMS [px] |
|---|---|---|---|---|
| all_vp_ppfree | 1601 | 174 | (979, 471) | 1.461 |
| all_vp_ppfixed | 1500 | 77 | (960, 540) | 1.478 |
| all_novp_ppfree | 2966 | 1276 | (949, 440) | 1.424 |
| all_novp_ppfixed | 1191 | 277 | (960, 540) | 1.434 |
| top_vp_ppfree | 1619 | 249 | (999, 479) | 1.810 |
| shelf_vp_ppfree | 1417 | 129 | (989, 494) | 0.588 |
| top_vp_ppfixed | 1519 | 104 | (960, 540) | 1.841 |
| shelf_vp_ppfixed | 1391 | 63 | (960, 540) | 0.601 |

Bootstrap over stickers (robust sigma = half the 16-84 % range): all_vp_ppfree: f median 1607, sigma 169 px (16-84 %: 1433-1772); all_vp_ppfixed: f median 1488, sigma 121 px (16-84 %: 1380-1622); shelf_vp_ppfree: f median 1420, sigma 75 px (16-84 %: 1339-1490); top_vp_ppfree: f median 1619, sigma 199 px (16-84 %: 1421-1819)

Verdict: too weak (profile of the corner RMS vs f is almost flat: 1.69 px at 1150, 1.48 px at 1600, 1.49 px at 1750) and biased by the non-parallel stickers (shape-model RMS 1.5 px, 0.17 px when each sticker gets its own tilt). See geometry_diagnosis.md.
