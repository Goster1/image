# Method: two-cart consistency

Both carts stand on the same floor: the camera height above the floor and the floor normal from the two cart poses must agree. Profile over fixed f with the plumb-line distortion fixed in pixel units (script 42_geomdiag_d_twocart.py).

| point set / pp | f where floor normals agree (dnx = 0) [px] (+-cov) | RMS [px] | bootstrap std | f where dh = 0 |
|---|---|---|---|---|
| a_all|pp_fixed | 1377 +- 24 | 6.55 | 34 (n=150) | 1239 +- 364 |
| a_all|pp_free | 1378 +- 23 | 6.55 | 54 (n=146) | 1482 +- 203, 1630 +- 13 |
| b_shelf|pp_fixed | 1489 +- 19 | 2.30 | 49 (n=148) | no sign change in grid |
| b_shelf|pp_free | 1488 +- 24 | 2.25 | 67 (n=144) | 1374 +- 27 |
| c_top|pp_fixed | 1306 +- 106 | 4.29 |  | no sign change in grid |
| c_top|pp_free | none in grid |  |  | no sign change in grid |

Main (all stickers, pp free, floor normals parallel): f = 1378 +- 81 px (statistical 59, systematic all-vs-shelf-only half difference 55), pp = (955, 522). The camera height difference is insensitive to f. Geometry-dependent (drawing-geometry RMS ~6 px); see geometry_diagnosis.md.
