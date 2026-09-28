"""Adds mapping_uncertainty_px (centre / cart_band / corners) to work/cache/method_twocart.json and
work/cache/method_markershape.json (run after 42_geomdiag_d_twocart.py and 42_geomdiag_e_markershape.py).

Parameter samples: f ~ N(f, sigma_f) (and pp ~ N(pp, sigma_pp) where the method gives one), distortion
fixed in pixel units (k1 = k1px f^2, k2 = k2px f^4); displacement field relative to the method's own
lens after the best compensating rotation (evaltools.mapping_displacement, pose is estimated with the lens
in normal use). Reported: median over each region of the per-pixel RMS displacement (1-sigma), + max.
"""
import numpy as np

from common import CACHE, K_from, load_json, save_json
from evaltools import grid, mapping_displacement, mapping_stats

rng = np.random.default_rng(5)
NS = 60


def lens(f, cx, cy, dpx):
    return K_from(f, f, cx, cy), np.array([dpx[0] * f ** 2, dpx[1] * f ** 4, 0, 0, dpx[4] * f ** 6])


def add(fn, dpx, sig_f, sig_cx=None, sig_cy=None):
    d = load_json(fn)
    K = np.array(d["camera_matrix"])
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    K0, d0 = lens(f, cx, cy, dpx)
    uv, _ = grid(40)
    disp = []
    for _ in range(NS):
        fs = f + rng.normal(0, sig_f)
        cxs = cx + (rng.normal(0, sig_cx) if sig_cx else 0.0)
        cys = cy + (rng.normal(0, sig_cy) if sig_cy else 0.0)
        disp.append(mapping_displacement(K0, d0, *lens(fs, cxs, cys, dpx), uv))
    st, _ = mapping_stats(np.array(disp), uv)
    from common import fold_margin
    d["mapping_uncertainty_px"] = {k: v["rms_median_px"] for k, v in st.items() if k != "whole_image"}
    d["details"]["fold_margin"] = float(fold_margin(K0, d0))
    d.setdefault("details", {})["mapping_uncertainty_detail"] = dict(
        stats=st, samples=NS, sampled=dict(sigma_f=sig_f, sigma_cx=sig_cx, sigma_cy=sig_cy),
        note="rotation-compensated displacement vs. the method lens; distortion fixed in pixel units")
    save_json(d, fn)
    print(fn.split("/")[-1], d["mapping_uncertainty_px"], "fold margin", d["details"]["fold_margin"])


def main():
    tc = load_json(f"{CACHE}/method_twocart.json")
    dpx = tc["details"]["plumb"]["dist_px"]
    add(f"{CACHE}/method_twocart.json", dpx, tc["uncertainty"]["fx_px_1sigma"])
    ms = load_json(f"{CACHE}/method_markershape.json")
    u = ms["uncertainty"]
    add(f"{CACHE}/method_markershape.json", dpx, u["fx_px_1sigma"], u.get("cx_1sigma"), u.get("cy_1sigma"))


if __name__ == "__main__":
    main()
