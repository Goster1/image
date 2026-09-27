# Method: sticker shape (horizontal squares)

Every sticker is modelled as a flat horizontal square with a free 3D position; all stickers of a cart share the cart rotation (axis-aligned, rot_k * 90 deg), the nadir from the vertical post edges ties the cart Z axis; distortion from the plumb-line fit (pixel units). Positions, heights and spacings of the stickers are not used (42_geomdiag_e_markershape.py).

| variant | f [px] | +-cov | pp | corner RMS [px] |
|---|---|---|---|---|
| all_vp_ppfree | 1601 | 178 | (978, 470) | 1.484 |
| all_vp_ppfixed | 1496 | 77 | (960, 540) | 1.502 |
| all_novp_ppfree | 3183 | 1437 | (949, 435) | 1.450 |
| all_novp_ppfixed | 1128 | 280 | (960, 540) | 1.458 |
| top_vp_ppfree | 1619 | 255 | (998, 477) | 1.841 |
| shelf_vp_ppfree | 1417 | 134 | (989, 489) | 0.595 |
| top_vp_ppfixed | 1514 | 105 | (960, 540) | 1.872 |
| shelf_vp_ppfixed | 1388 | 63 | (960, 540) | 0.608 |

Bootstrap over stickers: all_vp_ppfree: f std 182 px (16-84 %: 1428-1775); all_vp_ppfixed: f std 128 px (16-84 %: 1373-1621); shelf_vp_ppfree: f std 154 px (16-84 %: 1338-1491); top_vp_ppfree: f std 33442 px (16-84 %: 1416-1815)
