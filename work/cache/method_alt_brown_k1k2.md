# method_alt_brown_k1k2: Brown k1,k2 (joint stickers + edges)

* K: f = 1434.4 / 1434.4 px, pp = (927.5, 501.8); dist [k1,k2,p1,p2,k3] = [-0.33551, 0.0906, 0.0, 0.0, 0.0]
* native: fx = 1434.4 +- 26, fy = 1434.4, cx = 927.49 +- 7.3, cy = 501.81 +- 4, k1 = -0.33551 +- 0.013, k2 = 0.090604 +- 0.0084
* blocks: M 6.139 px, V 0.361 px, S 0.229 px; sticker corners radial RMS 8.68 px
* dQAIC (M/E/ME): 3.9 / 13.0 / 14.6; fold margin 74.506
* adequate: True
* mapping 1-sigma (parameter samples, rot.-comp.): centre 2.14 px, cart_band 9.08 px, corners 12.67 px

## Review (43_altmodels_review.py)

* 1-sigma (max of delete-one-member jackknife and member-cluster sandwich; cy without the floor-line VP groups): f 43.3, cx 9.68, cy 42.7, k1 0.0213, k2 0.0128 (cy 15 only if the blue floor line and the white tape are exactly parallel)
* previous (edge-cluster sandwich): f_1sigma 26.3, cx_1sigma 7.29, cy_1sigma 3.98, k1_1sigma 0.0132, k2_1sigma 0.00842
* mapping uncertainty (total = parameter part + model-choice part, rot.-comp.): centre 4.338 / cart band 15.461 / corners 22.096 px (previous, parameter part only: {'centre': 2.138, 'cart_band': 9.076, 'corners': 12.674})
* without the floor-line VP groups: f 1425.9, pp (925.4, 477.5); jackknife sd f 47.9, cy 42.7
* focal length over the delete-one-cluster folds: 1413-1457 px
