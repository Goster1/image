"""Pixel effect of each geometric deviation found by the diagnosis (for the tables in geometry_diagnosis.md).

For the camera of a diagnostic fit (42_geomdiag_c_fits, distortion free, f/pp/k1/k2 + poses), each element
is switched from the drawing value to the diagnosed value and the displacement of the affected sticker
corners is reported (RMS / max px, and where in the image). Elements:
  * end-board heights (board_dz fit)
  * top-plate height per cart (top_dz_cart fit)
  * code size 90 -> diagnosed (code_size+board_dz fit)
  * board tilt along X (sticker-shape diagnostic at f = 1400: mean tilt of the two stickers of a board)
Also: the image displacement per mm of height / horizontal offset at every sticker (local scale), from the
drawing fit, to translate column-collinearity residuals (px) into mm.
Output: work/cache/geomdiag_effects.json (printed table)
"""
import importlib

import numpy as np

from common import CACHE, load_json, project, save_json

G = importlib.import_module("42_geomdiag_lib")
C = importlib.import_module("42_geomdiag_c_fits")


def fitted(P, variant):
    ba = C.BA(P, variant, dist_mode="free")
    best = None
    for x0 in (ba.x0(),):
        r = ba.solve(x0)
        best = r if best is None or r.cost < best.cost else best
    return ba, best


def proj_all(ba, x, g_override=None):
    K, d, poses, g = ba.unpack(x)
    if g_override is not None:
        g = g_override
    X = C.geom_X(ba.P, ba.names_g, g)
    pr = np.zeros_like(ba.uv)
    for k in range(2):
        s = ba.ci == k
        pr[s] = project(X[s], K, d, rvec=poses[k, :3], tvec=poses[k, 3:])
    return pr


def main():
    ms = G.markers()
    P = C.points(ms)
    out = {}
    # ---- end-board heights ----
    for variant in ("board_dz", "top_dz_cart", "code_size+board_dz"):
        ba, r = fitted(P, variant)
        K, d, poses, g = ba.unpack(r.x)
        full = proj_all(ba, r.x)
        eff = {}
        for i, nm in enumerate(ba.names_g):
            g0 = g.copy()
            g0[i] = 0.0
            p0 = proj_all(ba, r.x, g0)
            dd = np.hypot(*(full - p0).T)
            m = dd > 1e-6
            uv = ba.uv[m]
            val = g[i] if nm != "code_scale" else 90 * (1 + g[i])
            eff[nm] = dict(value=float(val), px_rms=float(np.sqrt(np.mean(dd[m] ** 2))), px_max=float(dd[m].max()), n_corners=int(m.sum()),
                           image_region=f"x {uv[:, 0].min():.0f}-{uv[:, 0].max():.0f}, y {uv[:, 1].min():.0f}-{uv[:, 1].max():.0f}")
        out[variant] = dict(f=float(K[0, 0]), pp=[float(K[0, 2]), float(K[1, 2])], k=[float(d[0]), float(d[1])], effects=eff)
        print(variant, "f", round(K[0, 0]), {k: (round(v["value"], 1), round(v["px_rms"], 1), round(v["px_max"], 1), v["image_region"]) for k, v in eff.items()})
    # ---- local scales (drawing fit): px per mm of height and of horizontal X / Y offset at each sticker ----
    ba, r = fitted(P, "drawing")
    K, d, poses, g = ba.unpack(r.x)
    scales = {}
    for m in ms:
        c = m["corners_3d"].mean(0)
        k = C.CARTS.index(m["cart"])
        p0 = project(c[None], K, d, rvec=poses[k, :3], tvec=poses[k, 3:])[0]
        s = {}
        for ax, nm in ((2, "Z"), (0, "X"), (1, "Y")):
            e = np.zeros(3)
            e[ax] = 1.0
            p1 = project((c + e)[None], K, d, rvec=poses[k, :3], tvec=poses[k, 3:])[0]
            s[nm] = (p1 - p0).tolist()
        scales[f"{m['cart']}_{m['id']}"] = dict(row=m["row"], uv=p0.tolist(), px_per_mm=s, px_per_mm_norm={a: float(np.hypot(*v)) for a, v in s.items()})
    out["local_scale_drawing_fit"] = dict(f=float(K[0, 0]), scales=scales)
    for k, v in scales.items():
        print("scale", k, v["row"], {a: round(x, 3) for a, x in v["px_per_mm_norm"].items()})
    # ---- board tilt (sticker-shape diagnostic, f = 1400): corner displacement of a tilt of theta over +-45 mm ----
    msd = load_json(f"{CACHE}/geomdiag_markershape.json")
    td = msd["tilt_diagnostic"]["1400"]["tilt_yaw_deg"]
    tilt_eff = {}
    for key, v in td.items():
        cart, sid, row = key.split("_")
        sc = scales.get(f"{cart}_{sid}")
        if sc is None:
            continue
        dzmm = 45.0 * np.tan(np.radians(abs(v[0])))  # height change of the corners at +-45 mm along X
        tilt_eff[key] = dict(tilt_x_deg=v[0], corner_dz_mm=float(dzmm), corner_px=float(dzmm * sc["px_per_mm_norm"]["Z"]))
    out["tilt_effect_f1400"] = tilt_eff
    print({k: (round(v["tilt_x_deg"], 1), round(v["corner_px"], 2)) for k, v in tilt_eff.items()})
    save_json(out, f"{CACHE}/geomdiag_effects.json")


if __name__ == "__main__":
    main()
