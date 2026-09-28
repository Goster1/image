"""(3) DIAGNOSTIC bundle adjustments: which geometric element explains the sticker inconsistency?

NOT for the main result (the main result must use the drawing geometry). Each fit keeps the rest of
the drawing and frees ONE element (or a compact combination):
  drawing           : nothing freed
  board_dz          : height of each END BOARD (4: cart x end)
  row_dz            : height per shelf row A/B/C (3, common to both carts)
  row_dz_cart       : height per shelf row and cart (6)
  top_dz_cart       : height of the whole top plate per cart (2)
  board_tilt        : tilt of each end board (2 small rotations each, 8)
  board_dxy         : horizontal offset of each end board (8)
  stack_dz          : height of the shelf-sticker stack at each cart end (4)
  card_dz           : height of every shelf sticker (12)
  code_size         : printed code size (1; drawing 90 mm)
  code_size+board_dz, code_size+top_dz_cart, code_size+board_dz+board_tilt, code_size+card_dz
Camera: fx = fy = f, pp free, k1, k2 free (+ one 6-DoF pose per cart); second series with the
plumb-line distortion fixed in pixel units (only f, pp free) to see which geometry is compatible
with the straight-edge distortion. Residuals in distorted px, all 62 valid corners.
Compared: RMS, max, AIC (Gaussian, unknown variance: n ln(RSS/n) + 2k), BIC, implied f / pp / k,
per-row RMS. Uncertainties: Gauss-Newton covariance scaled by the residual variance.
Each fitted lens also gets common.fold_margin (> 0 required for a lens that is invertible out to the image
corners; the free-distortion drawing fits are folded, i.e. not valid lens models near the corners).
Output: work/cache/geomdiag_fits.json, results/geomdiag_fits.png
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, H, RESULTS, W, K_from, load_json, project, save_json  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")
CARTS = (80, 310)
SHELF_ID = {80: [3, 4, 5, 6, 7, 8], 310: [3, 4, 5, 6, 7, 8]}


def geom_names(variant):
    v = variant.split("+")
    names = []
    for part in v:
        if part == "drawing":
            continue
        if part == "board_dz":
            names += [f"dz_board_{c}_{e}" for c in CARTS for e in (0, 1600)]
        elif part == "row_dz":
            names += [f"dz_row_{r}" for r in "ABC"]
        elif part == "row_dz_cart":
            names += [f"dz_row_{c}_{r}" for c in CARTS for r in "ABC"]
        elif part == "top_dz_cart":
            names += [f"dz_top_{c}" for c in CARTS]
        elif part == "board_tilt":
            names += [f"t{a}_board_{c}_{e}" for c in CARTS for e in (0, 1600) for a in "xy"]
        elif part == "board_dxy":
            names += [f"d{a}_board_{c}_{e}" for c in CARTS for e in (0, 1600) for a in "xy"]
        elif part == "stack_dz":
            names += [f"dz_stack_{c}_{e}" for c in CARTS for e in (0, 1600)]
        elif part == "card_dz":
            names += [f"dz_card_{c}_{i}" for c in CARTS for i in SHELF_ID[c]]
        elif part == "code_size":
            names += ["code_scale"]
        else:
            raise ValueError(part)
    return names


def points(ms):
    P = []
    for m in ms:
        for j in range(4):
            if m["valid"][j]:
                X = m["corners_3d"][j]
                P.append(dict(cart=m["cart"], id=m["id"], j=j, level=m["row"], end=0 if X[0] < 800 else 1600, X=X.copy(),
                              cen=m["corners_3d"].mean(0), uv=m["corners_px"][j].copy()))
    return P


def geom_X(P, names, g):
    X = np.array([p["X"] for p in P], float)
    gp = dict(zip(names, g))
    if not gp:
        return X
    if "code_scale" in gp:
        C = np.array([p["cen"] for p in P])
        X = C + (X - C) * (1.0 + gp["code_scale"])
    for i, p in enumerate(P):
        c, e, L = p["cart"], p["end"], p["level"]
        if L == "top":
            b = f"board_{c}_{e}"
            X[i, 2] += gp.get(f"dz_{b}", 0.0) + gp.get(f"dz_top_{c}", 0.0)
            X[i, 0] += gp.get(f"dx_{b}", 0.0)
            X[i, 1] += gp.get(f"dy_{b}", 0.0)
            tx, ty = gp.get(f"tx_{b}", 0.0), gp.get(f"ty_{b}", 0.0)
            if tx or ty:
                xc = 82.5 if e == 0 else 1517.5
                X[i, 2] += tx * (p["X"][1] - 225.0) - ty * (p["X"][0] - xc)
        else:
            X[i, 2] += gp.get(f"dz_row_{L}", 0.0) + gp.get(f"dz_row_{c}_{L}", 0.0) + gp.get(f"dz_stack_{c}_{e}", 0.0) + \
                gp.get(f"dz_card_{c}_{p['id']}", 0.0)
    return X


class BA:
    def __init__(self, P, variant, dist_mode="free", pl=None, pp_free=True):
        self.P = P
        self.names_g = geom_names(variant)
        self.uv = np.array([p["uv"] for p in P])
        self.ci = np.array([CARTS.index(p["cart"]) for p in P])
        self.dist_mode = dist_mode
        self.pl = pl
        self.pp_free = pp_free
        self.ni = 1 + (2 if pp_free else 0) + (2 if dist_mode == "free" else 0)
        self.gscale = np.array([1e-3 if (n.startswith("t") or n == "code_scale") else 1.0 for n in self.names_g])

    def unpack(self, x):
        f = x[0]
        i = 1
        if self.pp_free:
            cx, cy = x[1], x[2]
            i = 3
        else:
            cx, cy = self.pl["centre"] if self.pl is not None else ((W - 1) / 2, (H - 1) / 2)
        if self.dist_mode == "free":
            d = np.array([x[i], x[i + 1], 0, 0, 0])
            i += 2
        else:
            dp = self.pl["dist_px"]
            d = np.array([dp[0] * f ** 2, dp[1] * f ** 4, 0, 0, dp[4] * f ** 6])
        poses = x[i:i + 12].reshape(2, 6)
        g = x[i + 12:] * self.gscale
        return K_from(f, f, cx, cy), d, poses, g

    def res(self, x):
        K, d, poses, g = self.unpack(x)
        X = geom_X(self.P, self.names_g, g)
        pr = np.zeros_like(self.uv)
        for k in range(2):
            s = self.ci == k
            pr[s] = project(X[s], K, d, rvec=poses[k, :3], tvec=poses[k, 3:])
        return (pr - self.uv).ravel()

    def x0(self, base=None):
        init = load_json(f"{CACHE}/initial_calib.json")
        if base is not None:
            x = list(base[:self.ni + 12])
        else:
            x = [1300.0] + ([959.5, 539.5] if self.pp_free else []) + ([-0.25, 0.05] if self.dist_mode == "free" else [])
            x += list(init["poses"]["80"]) + list(init["poses"]["310"])
        return np.array(x + [0.0] * len(self.names_g), float)

    def solve(self, x0):
        return least_squares(self.res, x0, x_scale="jac", method="trf", max_nfev=4000, xtol=1e-12, ftol=1e-12)


def summarise(ba, r):
    K, d, poses, g = ba.unpack(r.x)
    rr = r.fun.reshape(-1, 2)
    n = len(r.fun)
    k = len(r.x)
    rss = float(np.sum(r.fun ** 2))
    J = r.jac
    dof = max(n - k, 1)
    C = np.linalg.pinv(J.T @ J) * (rss / dof)
    sd = np.sqrt(np.abs(np.diag(C)))
    names_i = ["f"] + (["cx", "cy"] if ba.pp_free else []) + (["k1", "k2"] if ba.dist_mode == "free" else [])
    intr = {nm: (float(r.x[i]), float(sd[i])) for i, nm in enumerate(names_i)}
    if ba.dist_mode != "free":
        intr["k1"] = (float(d[0]), 0.0)
        intr["k2"] = (float(d[1]), 0.0)
    gsd = sd[ba.ni + 12:] * ba.gscale
    geo = {nm: (float(v), float(s)) for nm, v, s in zip(ba.names_g, g, gsd)}
    per_row = {L: float(np.sqrt(np.mean(np.sum(rr[[p["level"] == L for p in ba.P]] ** 2, 1)))) for L in ("top", "A", "B", "C")}
    per_cart = {c: float(np.sqrt(np.mean(np.sum(rr[[p["cart"] == c for p in ba.P]] ** 2, 1)))) for c in CARTS}
    # camera height above the floor and floor-normal agreement between the carts (by-product)
    from common import rodrigues
    hs, zs = [], []
    for kk in range(2):
        R = rodrigues(poses[kk, :3])
        Cc = -R.T @ poses[kk, 3:]
        hs.append(float(Cc[2] + 1800.0))
        zs.append(R[:, 2])
    ang = float(np.degrees(np.arccos(np.clip(zs[0] @ zs[1], -1, 1))))
    from common import fold_margin
    fm = float(fold_margin(K, d))  # > 0: radial map monotonic out to the image corners (valid lens); < 0: folded
    return dict(fold_margin=fm, rms=float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))), max=float(np.max(np.hypot(*rr.T))), n=n, k=k, rss=rss,
                aic=float(G.aic_gauss(rss, n, k)), bic=float(G.bic_gauss(rss, n, k)), intr=intr, geom=geo, per_row=per_row,
                per_cart=per_cart, cam_height_mm={80: hs[0], 310: hs[1]}, floor_normal_angle_deg=ang, resid=rr)


VARIANTS = ["drawing", "board_dz", "row_dz", "row_dz_cart", "top_dz_cart", "board_tilt", "board_dxy", "stack_dz", "card_dz",
            "code_size", "code_size+board_dz", "code_size+top_dz_cart", "code_size+board_dz+board_tilt", "code_size+card_dz",
            "board_dz+board_tilt"]


def main():
    ms = G.markers()
    P = points(ms)
    edges = G.all_edges()
    pl = G.plumb("k1k2_centre_fixed", edges)
    out = {}
    base = {}
    for mode in ("free", "plumb_fixed"):
        out[mode] = {}
        for v in VARIANTS:
            ba = BA(P, v, dist_mode=mode, pl=pl)
            best = None
            starts = [ba.x0()]
            if "drawing" in base.get(mode, {}):
                starts.append(np.concatenate([base[mode]["drawing"][:ba.ni + 12], np.zeros(len(ba.names_g))]))
            for x0 in starts:
                try:
                    r = ba.solve(x0)
                except Exception as ex:  # noqa
                    print("fail", v, ex)
                    continue
                if best is None or r.cost < best.cost:
                    best = r
            base.setdefault(mode, {})[v] = best.x
            s = summarise(ba, best)
            out[mode][v] = s
            it = s["intr"]
            gtxt = ", ".join(f"{k}={val:.1f}+-{sd:.1f}" if not (k.startswith("t") or k == "code_scale") else
                             (f"{k}={np.degrees(val):.2f}deg" if k.startswith("t") else f"code={90 * (1 + val):.2f}+-{90 * sd:.2f}mm")
                             for k, (val, sd) in s["geom"].items())
            print(f"[{mode:11s}] {v:30s} rms {s['rms']:.2f} max {s['max']:.2f} k {s['k']:2d} AIC {s['aic']:7.1f} BIC {s['bic']:7.1f} "
                  f"f {it['f'][0]:.0f}+-{it['f'][1]:.0f} pp ({it['cx'][0]:.0f},{it['cy'][0]:.0f})+-({it['cx'][1]:.0f},{it['cy'][1]:.0f}) "
                  f"k1 {it['k1'][0]:.3f} k2 {it['k2'][0]:.3f} fold {s['fold_margin']:+.3f} | rows {({kk: round(vv, 2) for kk, vv in s['per_row'].items()})} "
                  f"| h {s['cam_height_mm'][80]:.0f}/{s['cam_height_mm'][310]:.0f} ang {s['floor_normal_angle_deg']:.2f} | {gtxt}")
    # the same with bias-corrected corners (dark-side edge-localisation bias removed; phase-1 review)
    Pb = points(G.markers("bias_corrected"))
    out["free_bias_corrected"] = {}
    for v in ("drawing", "board_dz", "top_dz_cart", "code_size", "code_size+board_dz"):
        ba = BA(Pb, v, dist_mode="free", pl=pl)
        r = ba.solve(ba.x0())
        s = summarise(ba, r)
        out["free_bias_corrected"][v] = s
        it = s["intr"]
        print(f"[bias-corr  ] {v:30s} rms {s['rms']:.2f} f {it['f'][0]:.0f} pp ({it['cx'][0]:.0f},{it['cy'][0]:.0f}) k1 {it['k1'][0]:.3f} "
              f"k2 {it['k2'][0]:.3f} fold {s['fold_margin']:+.3f} |", {k: round(val if k != "code_scale" else 90 * (1 + val), 1) for k, (val, sd) in s["geom"].items()})
    save_json({m: {v: {k: val for k, val in s.items() if k != "resid"} for v, s in d.items()} for m, d in out.items()},
              f"{CACHE}/geomdiag_fits.json")

    # plot: RMS / AIC / implied f per variant
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5), dpi=110)
    for mi, (mode, mk) in enumerate((("free", "o"), ("plumb_fixed", "s"))):
        xs = np.arange(len(VARIANTS)) + (mi - 0.5) * 0.2
        axs[0].plot(xs, [out[mode][v]["rms"] for v in VARIANTS], mk, label=f"dist {mode}")
        axs[1].plot(xs, [out[mode][v]["aic"] for v in VARIANTS], mk, label=f"dist {mode}")
        axs[2].errorbar(xs, [out[mode][v]["intr"]["f"][0] for v in VARIANTS], yerr=[out[mode][v]["intr"]["f"][1] for v in VARIANTS],
                        fmt=mk, capsize=2, label=f"dist {mode}")
    for ax, t in zip(axs, ("RMS [px] (detection noise ~0.1-0.2 px)", "AIC (Gaussian, unknown variance)", "implied f [px]")):
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS, rotation=70, fontsize=7, ha="right")
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[0].axhline(0.15, color="k", ls=":", lw=0.8)
    fig.suptitle("DIAGNOSTIC fits (not the main result): one geometric element freed at a time, rest = drawing", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_fits.png")
    print("saved")


if __name__ == "__main__":
    main()
