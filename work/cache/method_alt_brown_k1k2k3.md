# method_alt_brown_k1k2k3: Brown k1,k2,k3 (joint stickers + edges)

* K: f = 1431.8 / 1431.8 px, pp = (930.7, 505.0); dist [k1,k2,p1,p2,k3] = [-0.35284, 0.13684, 0.0, 0.0, -0.03074]
* native: fx = 1431.8 +- 26, fy = 1431.8, cx = 930.72 +- 6.8, cy = 505.05 +- 4.8, k1 = -0.35284 +- 0.02, k2 = 0.13684 +- 0.032, k3 = -0.030737 +- 0.017
* blocks: M 6.259 px, V 0.358 px, S 0.229 px; sticker corners radial RMS 8.85 px
* dQAIC (M/E/ME): 4.1 / 12.4 / 14.2; fold margin 0.047
* adequate: True
* mapping 1-sigma (parameter samples, rot.-comp.): centre 1.89 px, cart_band 8.06 px, corners 11.25 px
* mapping vs Brown k1,k2 (same data, rot.-comp. median): centre 0.271 px, cart_band 1.777 px, corners 2.305 px, whole_image 2.053 px

## Review (43_altmodels_review.py)

* 1-sigma (max of delete-one-member jackknife and member-cluster sandwich; cy without the floor-line VP groups): f 44.3, cx 9.57, cy 34.8, k1 0.0307, k2 0.0463, k3 0.0249 (cy 8.6 only if the blue floor line and the white tape are exactly parallel)
* previous (edge-cluster sandwich): f_1sigma 26.5, cx_1sigma 6.84, cy_1sigma 4.84, k1_1sigma 0.0202, k2_1sigma 0.0317, k3_1sigma 0.0169
* mapping uncertainty (total = parameter part + model-choice part, rot.-comp.): centre 4.124 / cart band 15.078 / corners 21.087 px (previous, parameter part only: {'centre': 1.89, 'cart_band': 8.062, 'corners': 11.252})
* without the floor-line VP groups: f 1425.8, pp (929.3, 485.0); jackknife sd f 46.8, cy 34.8
* focal length over the delete-one-cluster folds: 1411-1455 px
