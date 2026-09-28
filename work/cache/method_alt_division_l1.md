# method_alt_division_l1: division l1 (joint stickers + edges)

* K: f = 1416.3 / 1416.3 px, pp = (962.0, 527.3); dist [k1,k2,p1,p2,k3] = [-0.40675, 0.23242, 1e-05, -0.0, -0.07162]
* native: fx = 1419.1 +- 28, fy = 1419.1, cx = 961.96 +- 10, cy = 527.32 +- 6.5, l1 = -0.43647 +- 0.02
* blocks: M 6.646 px, V 0.384 px, S 0.239 px; sticker corners radial RMS 9.40 px
* dQAIC (M/E/ME): 9.0 / 40.4 / 47.5; fold margin 0.732
* adequate: False - region_tile_E S: held-out MS worse than B_k1k2 by 0.0191 +- 0.0076 px^2; region_tile_ME S: held-out MS worse than B_k1k2 by 0.0187 +- 0.0075 px^2; region_tile_ME V: held-out MS worse than B_k1k2 by 2.78 +- 1.2 px^2; region_src_ME S: held-out MS worse than B_k1k2 by 0.0155 +- 0.0077 px^2; E: dQAIC +27.5 vs B_k1k2
* mapping 1-sigma (parameter samples, rot.-comp.): centre 2.09 px, cart_band 8.54 px, corners 13.31 px
* mapping vs Brown k1,k2 (same data, rot.-comp. median): centre 2.512 px, cart_band 10.327 px, corners 18.268 px, whole_image 13.354 px
* conversion to OpenCV Brown (k1,k2,p1,p2,k3): error median centre 0.188 px, cart_band 0.163 px, corners 0.231 px, whole_image 0.17 px
