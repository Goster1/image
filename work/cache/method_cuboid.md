# Method: cuboid cart model (stickers + exact edges, drawing dimensions exact)

Script `work/41_cuboid_fit.py` (helpers `41_cuboid_lib.py`). Data: 62 sticker corners, 13 sticker sides, 3 exact structure edges (cart310_board0_outer, cart80_x_board0_front, cart80_x_board1600_front). Edge files reviewed: {310: True, 80: True}.

Main model (min. leave-one-sticker-out error, cluster weighting): **k1k2_ppfix**

| param | value | bootstrap 1-sigma | cov 1-sigma |
|---|---|---|---|
| f | 1326.6 | 30.6 | 8.93 |
| k1 | -0.11842 | 0.171 | 0.0329 |
| k2 | -0.17727 | 0.308 | 0.0522 |

Block RMS: corners 5.66 px per point (max 10.4), sticker sides 2.49 px, exact edges 7.64 px. LOSO 8.13 px.
Mapping uncertainty (rot.-comp., bootstrap): centre 46.02 px, cart_band 46.56 px, corners 536.12 px, whole_image 48.04 px

| model / weighting | f | pp | dist | corners px/pt | sides | edges | LOSO |
|---|---|---|---|---|---|---|---|
| k1_ppfix|cluster | 1341 | centre | k1=-0.231 | 5.89 | 2.30 | 7.80 | 8.21 |
| k1k2_ppfix|cluster | 1327 | centre | k1=-0.118, k2=-0.177 | 5.66 | 2.49 | 7.64 | 8.13 |
| k1_ppfree|cluster | 1353 | (885, 452) | k1=-0.222 | 5.75 | 2.28 | 6.90 | nan |
| k1k2_ppfree|cluster | 1355 | (749, 423) | k1=-0.096, k2=-0.126 | 6.21 | 2.44 | 3.02 | nan |
| k1k2k3_ppfree|cluster | 1338 | (675, 437) | k1=0.053, k2=-0.434, k3=0.233 | 6.68 | 2.34 | 1.42 | nan |
| k1k2_ppfree_fxfy|cluster | 1337 | (835, 438) | k1=-0.115, k2=-0.123 | 5.83 | 2.44 | 4.33 | nan |
| k1k2p1p2_ppfree|cluster | 1363 | (955, 365) | k1=-0.448, k2=0.416, p1=0.011, p2=-0.003 | 5.33 | 2.28 | 6.01 | nan |
| k1_ppfix|point | 1294 | centre | k1=-0.226 | 8.69 | 3.70 | 0.83 | nan |
| k1k2_ppfix|point | 1308 | centre | k1=-0.319, k2=0.133 | 8.88 | 3.15 | 0.95 | nan |
| k1_ppfree|point | 1345 | (632, 365) | k1=-0.159 | 8.55 | 2.33 | 1.03 | nan |
| k1k2_ppfree|point | 1376 | (548, 400) | k1=0.221, k2=-0.363 | 9.93 | 1.79 | 0.44 | nan |
| k1k2k3_ppfree|point | 1386 | (539, 358) | k1=-0.097, k2=0.383, k3=-0.492 | 10.24 | 1.83 | 0.34 | nan |
| k1k2_ppfree_fxfy|point | 1347 | (529, 401) | k1=0.195, k2=-0.301 | 10.39 | 1.70 | 0.43 | nan |
| k1k2p1p2_ppfree|point | 1348 | (774, 432) | k1=0.288, k2=-1.057, p1=-0.006, p2=0.061 | 10.35 | 1.74 | 0.25 | nan |

Exact-edge check (implied shift of the traced edge from its drawing line with the fitted camera, mm):
* cart310_board0_outer: joint -18.8 mm along X (-8.85 px); markers-only -20.5 mm (-9.61 px)
* cart80_x_board0_front: joint +6.5 mm along Y (-4.41 px); markers-only +7.8 mm (-5.32 px)
* cart80_x_board1600_front: joint +6.4 mm along Y (-3.49 px); markers-only +7.7 mm (-4.25 px)
