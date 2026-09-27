"""(5) STICKER-SHAPE method: every sticker is a horizontal square; its image shape (not its position or
height) constrains the viewing geometry.

Model: f (fx = fy), pp, one rotation per cart; every sticker = a square (drawing size; the size only
sets its depth, which is free) with FREE 3D position (so heights, spacings and placements of the
stickers are NOT used) and orientation = cart rotation x in-plane rot_k * 90 deg (all stickers of one
cart are parallel and axis-aligned: they share the vanishing points of the cart X and Y directions and
the horizon). The vertical vanishing point (nadir from the vertical cart-post edges, +-3 px floor)
ties the cart Z axis. Distortion: plumb-line fit, fixed in pixel units. All valid corners are used.
This is the joint (maximum-likelihood) form of "per-sticker homography -> VPs of its sides + horizon,
combined over the stickers of a cart with the nadir".
Variants: with / without the nadir, pp free / fixed, top stickers only, shelf stickers only; and a
diagnostic with a free tilt per sticker (are the shelf-card stickers horizontal?).
Uncertainty: bootstrap over stickers (resampled within each cart) + residual-scaled covariance.
Writes work/cache/method_markershape.json (+ .md), results/geomdiag_markershape.png
"""
import importlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, RESULTS, K_from, distort_normalized, lens_json, load_json, project, rodrigues, save_json  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")
rng = np.random.default_rng(11)
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CARTS = (80, 310)
VP_SIG = 3.0


class Shape:
    def __init__(self, ms, pl, v_dist, pp_free=True, use_vp=True, sel=None, tilt=False):
        self.pl = pl
        self.pp_free = pp_free
        self.use_vp = use_vp
        self.v_dist = np.asarray(v_dist, float)
        self.tilt = tilt
        self.st = []
        for m in ms:
            if sel is not None and not sel(m):
                continue
            if m["valid"].sum() < 2:
                continue
            c = m["corners_3d"].mean(0)
            self.st.append(dict(cart=m["cart"], id=m["id"], row=m["row"], off=m["corners_3d"] - c, valid=m["valid"].copy(),
                                uv=m["corners_px"].copy(), c=c))
        self.ns = len(self.st)
        self.carts = [c for c in CARTS if any(s["cart"] == c for s in self.st)]
        self.ni = 1 + (2 if pp_free else 0)

    def lens(self, x):
        f = x[0]
        pp = x[1:3] if self.pp_free else np.array(self.pl["centre"])
        dp = self.pl["dist_px"]
        return K_from(f, f, pp[0], pp[1]), np.array([dp[0] * f ** 2, dp[1] * f ** 4, 0, 0, dp[4] * f ** 6])

    def split(self, x):
        i = self.ni
        rots = {}
        for c in self.carts:
            rots[c] = x[i:i + 3]
            i += 3
        ts = x[i:i + 3 * self.ns].reshape(-1, 3)
        i += 3 * self.ns
        tl = x[i:i + 3 * self.ns].reshape(-1, 3) * 1e-2 if self.tilt else np.zeros((self.ns, 3))
        return rots, ts, tl

    def res(self, x):
        K, d = self.lens(x)
        rots, ts, tl = self.split(x)
        Rs = {c: rodrigues(r) for c, r in rots.items()}
        out = []
        for k, s in enumerate(self.st):
            R = Rs[s["cart"]]
            off = s["off"]
            if self.tilt:
                a, b, g = tl[k]  # tilt about the cart Y axis (a: dZ/dX), about X (b: dZ/dY), in-plane yaw g (rad)
                off = np.column_stack([off[:, 0] - g * off[:, 1], off[:, 1] + g * off[:, 0], a * off[:, 0] + b * off[:, 1]])
            Xc = off @ R.T + ts[k]
            pr = project(Xc, K, d)
            out.append(((pr - s["uv"])[s["valid"]]).ravel())
        if self.use_vp:
            for c in self.carts:
                z = Rs[c][:, 2]
                xn, yn = z[0] / z[2], z[1] / z[2]
                xd, yd = distort_normalized(np.array([xn]), np.array([yn]), d)
                vp = np.array([K[0, 0] * xd[0] + K[0, 2], K[1, 1] * yd[0] + K[1, 2]])
                out.append((vp - self.v_dist) / VP_SIG * 0.15)  # weighted like a 0.15 px corner when 3 px off
        if self.tilt:
            out.append(tl.ravel() * 0.15 / np.deg2rad(20.0))  # weak prior (20 deg ~ one 0.15 px corner) fixes the gauge
        return np.concatenate(out)

    def x0(self, f0=1400.0, prev=None):
        init = load_json(f"{CACHE}/initial_calib.json")
        x = [f0] + ([959.5, 539.5] if self.pp_free else [])
        K = K_from(f0, f0, 959.5, 539.5)
        for c in self.carts:
            x += list(init["poses"][str(c)][:3])
        for s in self.st:
            pose = np.array(init["poses"][str(s["cart"])])
            R = rodrigues(pose[:3])
            t = R @ s["c"] + pose[3:] * f0 / 1289.0
            x += list(t)
        if self.tilt:
            x += [0.0] * (3 * self.ns)
        x = np.array(x, float)
        if prev is not None:
            x[:len(prev)] = prev[:len(x)] if len(prev) >= len(x) else np.r_[prev, x[len(prev):]][:len(x)]
        return x

    def solve(self, x0):
        return least_squares(self.res, x0, x_scale="jac", method="trf", max_nfev=5000, xtol=1e-12, ftol=1e-12)


def run(sh, x0=None, f0=1400.0):
    best = None
    starts = [sh.x0(f0)] if x0 is None else [x0]
    if x0 is None:
        starts += [sh.x0(1250.0), sh.x0(1600.0)]
    for s0 in starts:
        try:
            r = sh.solve(s0)
        except Exception:  # noqa
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def summary(sh, r):
    n_corner = int(sum(s["valid"].sum() for s in sh.st)) * 2
    rc = r.fun[:n_corner]
    J = r.jac
    dof = max(len(r.fun) - len(r.x), 1)
    s2 = float(np.sum(rc ** 2) / max(n_corner - len(r.x), 1))
    Cx = np.linalg.pinv(J.T @ J) * s2
    sd = np.sqrt(np.abs(np.diag(Cx)))
    out = dict(f=float(r.x[0]), sd_f=float(sd[0]), rms=float(np.sqrt(np.mean(rc.reshape(-1, 2) ** 2) * 2)), n_st=sh.ns, n_obs=n_corner, dof=dof)
    if sh.pp_free:
        out.update(cx=float(r.x[1]), cy=float(r.x[2]), sd_cx=float(sd[1]), sd_cy=float(sd[2]))
    else:
        out.update(cx=float(sh.pl["centre"][0]), cy=float(sh.pl["centre"][1]))
    rots, ts, tl = sh.split(r.x)
    Rs = {c: rodrigues(v) for c, v in rots.items()}
    if len(Rs) == 2:
        out["angle_between_cart_Z_deg"] = float(np.degrees(np.arccos(np.clip(Rs[80][:, 2] @ Rs[310][:, 2], -1, 1))))
    if sh.tilt:
        out["tilt_yaw_deg"] = {f"{s['cart']}_{s['id']}_{s['row']}": [float(np.degrees(np.arctan(a))) for a in tl[k]] for k, s in enumerate(sh.st)}
        out["tilt_yaw_sd_deg"] = {f"{s['cart']}_{s['id']}_{s['row']}": [float(np.degrees(sd[len(r.x) - 3 * sh.ns + 3 * k + q] * 1e-2)) for q in (0, 1, 2)]
                                  for k, s in enumerate(sh.st)}
    # per-sticker implied heights (from the free positions) relative to the cart frame, for information
    return out


def main():
    ms = G.markers()
    edges = G.all_edges()
    pl = G.plumb("k1k2_centre_fixed", edges)
    und = G.Und(pl)
    vpe = G.fit_vp_lines(und, G.vertical_edges(edges))
    v_dist = und.dist_px(vpe["v"][None])[0]
    print("nadir (undistorted)", np.round(vpe["v"], 1), "-> distorted", np.round(v_dist, 1))
    variants = {
        "all_vp_ppfree": dict(pp_free=True, use_vp=True),
        "all_vp_ppfixed": dict(pp_free=False, use_vp=True),
        "all_novp_ppfree": dict(pp_free=True, use_vp=False),
        "all_novp_ppfixed": dict(pp_free=False, use_vp=False),
        "top_vp_ppfree": dict(pp_free=True, use_vp=True, sel=lambda m: m["row"] == "top"),
        "shelf_vp_ppfree": dict(pp_free=True, use_vp=True, sel=lambda m: m["row"] != "top"),
        "top_vp_ppfixed": dict(pp_free=False, use_vp=True, sel=lambda m: m["row"] == "top"),
        "shelf_vp_ppfixed": dict(pp_free=False, use_vp=True, sel=lambda m: m["row"] != "top"),
    }
    out = {"nadir_undist": vpe["v"].tolist(), "nadir_dist": v_dist.tolist(), "variants": {}}
    sols = {}
    for name, kw in variants.items():
        sh = Shape(ms, pl, v_dist, **kw)
        r = run(sh)
        sols[name] = (sh, r)
        s = summary(sh, r)
        out["variants"][name] = s
        print(f"{name:18s} f {s['f']:.0f}+-{s['sd_f']:.0f} pp ({s['cx']:.0f},{s['cy']:.0f}) rms {s['rms']:.3f} n_st {s['n_st']} "
              f"angle(Z80,Z310) {s.get('angle_between_cart_Z_deg', np.nan):.2f}")
    # profile of the corner RMS vs f (all, vp, pp free) - how sharply does the shape determine f
    prof = []
    sh0, r0 = sols["all_vp_ppfree"]
    for f in np.arange(1150, 1751, 50):
        sh = Shape(ms, pl, v_dist, pp_free=True, use_vp=True)
        x0 = r0.x.copy()
        x0[0] = f
        res = least_squares(lambda x: sh.res(np.r_[f, x]), x0[1:], x_scale="jac", method="trf", max_nfev=3000)
        n_corner = int(sum(s["valid"].sum() for s in sh.st)) * 2
        prof.append(dict(f=float(f), rms_corners=float(np.sqrt(np.mean(res.fun[:n_corner] ** 2) * 2)), cost=float(2 * res.cost)))
    out["profile_all_vp_ppfree"] = prof
    print("profile:", [(p["f"], round(p["rms_corners"], 3)) for p in prof])
    # diagnostic at FIXED K (f = 1400, pp = image centre): free tilt (2) + in-plane yaw (1) per sticker, weak prior
    out["tilt_diagnostic"] = {}
    for fk in (1300.0, 1400.0, 1500.0):
        sht = Shape(ms, pl, v_dist, pp_free=False, use_vp=True, tilt=True)
        shf = Shape(ms, pl, v_dist, pp_free=False, use_vp=True)
        r_nt = least_squares(lambda x: shf.res(np.r_[fk, x]), shf.x0(fk)[1:], x_scale="jac", method="trf", max_nfev=4000)
        rt = least_squares(lambda x: sht.res(np.r_[fk, x]), np.r_[r_nt.x, np.zeros(3 * sht.ns)], x_scale="jac", method="trf", max_nfev=6000)

        class _R:  # adapter for summary()
            pass
        rr = _R()
        rr.x = np.r_[fk, rt.x]
        rr.fun = rt.fun
        rr.jac = np.column_stack([np.zeros(len(rt.fun)), rt.jac])
        st = summary(sht, rr)
        n_corner = int(sum(q["valid"].sum() for q in sht.st)) * 2
        st["rms_without_tilt_yaw"] = float(np.sqrt(np.mean(r_nt.fun[:n_corner] ** 2) * 2))
        out["tilt_diagnostic"][str(int(fk))] = st
        print(f"tilt/yaw diagnostic at f={fk:.0f}: corner rms {st['rms_without_tilt_yaw']:.3f} -> {st['rms']:.3f} px with free tilt+yaw per sticker")
        if fk == 1400.0:
            for k in st["tilt_yaw_deg"]:
                print("   ", k, "tiltX/tiltY/yaw [deg]", np.round(st["tilt_yaw_deg"][k], 2), "+-", np.round(st["tilt_yaw_sd_deg"][k], 2))
    # bootstrap over stickers
    boot = {}
    for name in ("all_vp_ppfree", "all_vp_ppfixed", "shelf_vp_ppfree", "top_vp_ppfree"):
        kw = variants[name]
        base_sh, base_r = sols[name]
        vals = []
        for b in range(NBOOT):
            msb = []
            for c in CARTS:
                cand = [m for m in ms if m["cart"] == c and (kw.get("sel") is None or kw["sel"](m)) and m["valid"].sum() >= 2]
                pick = rng.integers(0, len(cand), len(cand))
                msb += [cand[i] for i in pick]
            sh = Shape(msb, pl, v_dist, pp_free=kw["pp_free"], use_vp=kw["use_vp"])
            x0 = sh.x0(base_r.x[0])
            x0[:sh.ni] = base_r.x[:sh.ni]
            try:
                r = sh.solve(x0)
                vals.append([r.x[0]] + (list(r.x[1:3]) if kw["pp_free"] else []))
            except Exception:  # noqa
                pass
        v = np.array(vals)
        q = lambda a: float((np.percentile(a, 84) - np.percentile(a, 16)) / 2)  # robust sigma (degenerate replicates occur)
        boot[name] = dict(n=len(v), f_median=float(np.median(v[:, 0])), f_mean=float(v[:, 0].mean()), f_std=float(v[:, 0].std()),
                          f_sd_robust=q(v[:, 0]), f_q16=float(np.percentile(v[:, 0], 16)), f_q84=float(np.percentile(v[:, 0], 84)),
                          **({"cx_sd_robust": q(v[:, 1]), "cy_sd_robust": q(v[:, 2])} if kw["pp_free"] else {}))
        print("bootstrap", name, {k: round(x, 1) for k, x in boot[name].items()})
    out["bootstrap"] = boot
    save_json(out, f"{CACHE}/geomdiag_markershape.json")

    # method json: all stickers, nadir, pp free
    s = out["variants"]["all_vp_ppfree"]
    f = s["f"]
    K = K_from(f, f, s["cx"], s["cy"])
    dp = pl["dist_px"]
    d = np.array([dp[0] * f ** 2, dp[1] * f ** 4, 0, 0, dp[4] * f ** 6])
    b = boot["all_vp_ppfree"]
    fs = [out["variants"][k]["f"] for k in ("all_vp_ppfree", "all_vp_ppfixed", "all_novp_ppfree", "shelf_vp_ppfree", "top_vp_ppfree")]
    lj = lens_json(K, d, method="sticker_shape (horizontal squares: shape + common cart orientation + nadir; positions/heights free)",
                   model="fx=fy, pp free; k1,k2 fixed from the plumb-line fit in pixel units",
                   rms_reprojection_error_px=s["rms"],
                   uncertainty={"fx_px_1sigma": float(max(b["f_sd_robust"], s["sd_f"])), "fx_px_1sigma_bootstrap_stickers": b["f_sd_robust"],
                                "fx_px_1sigma_covariance": s["sd_f"], "cx_1sigma": b.get("cx_sd_robust"), "cy_1sigma": b.get("cy_sd_robust"),
                                "fx_spread_over_variants_px": float(np.ptp(fs)),
                                "note": "too weak to constrain f usefully; the corner RMS of the shape model (1.5 px) shows that the stickers of one cart "
                                        "are not mutually parallel / axis-aligned (end plates tilted), which biases it; shelf-only variant is cleaner"},
                   data_used=f"{s['n_st']} stickers (all valid corners), nadir from {len(G.vertical_edges(edges))} vertical cart-post edges",
                   geometry_assumptions="only: stickers are flat horizontal squares, axis-aligned with their cart (rot_k * 90 deg); "
                                        "sticker positions, heights and the code size are NOT used",
                   details=out)
    save_json(lj, f"{CACHE}/method_markershape.json")
    with open(f"{CACHE}/method_markershape.md", "w") as fh:
        fh.write("# Method: sticker shape (horizontal squares)\n\n")
        fh.write("Every sticker is modelled as a flat horizontal square with a free 3D position; all stickers of a cart share the cart "
                 "rotation (axis-aligned, rot_k * 90 deg), the nadir from the vertical post edges ties the cart Z axis; distortion from the "
                 "plumb-line fit (pixel units). Positions, heights and spacings of the stickers are not used (42_geomdiag_e_markershape.py).\n\n")
        fh.write("| variant | f [px] | +-cov | pp | corner RMS [px] |\n|---|---|---|---|---|\n")
        for k, v in out["variants"].items():
            fh.write(f"| {k} | {v['f']:.0f} | {v['sd_f']:.0f} | ({v['cx']:.0f}, {v['cy']:.0f}) | {v['rms']:.3f} |\n")
        fh.write("\nBootstrap over stickers (robust sigma = half the 16-84 % range): " + "; ".join(
            f"{k}: f median {v['f_median']:.0f}, sigma {v['f_sd_robust']:.0f} px (16-84 %: {v['f_q16']:.0f}-{v['f_q84']:.0f})" for k, v in boot.items()) + "\n")
        fh.write("\nVerdict: too weak (profile of the corner RMS vs f is almost flat: 1.69 px at 1150, 1.48 px at 1600, 1.49 px at 1750) and biased by "
                 "the non-parallel stickers (shape-model RMS 1.5 px, 0.17 px when each sticker gets its own tilt). See geometry_diagnosis.md.\n")

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=110)
    axs[0].plot([p["f"] for p in prof], [p["rms_corners"] for p in prof], "o-")
    axs[0].set_xlabel("fixed f [px]")
    axs[0].set_ylabel("corner RMS [px] (shape-only model, nadir, pp free)")
    axs[0].grid(alpha=0.3)
    names = list(out["variants"])
    axs[1].errorbar(range(len(names)), [out["variants"][k]["f"] for k in names], yerr=[out["variants"][k]["sd_f"] for k in names], fmt="o", capsize=3)
    for k, v in boot.items():
        i = names.index(k)
        axs[1].errorbar(i + 0.15, v["f_median"], yerr=v["f_sd_robust"], fmt="s", color="C1", capsize=3)
    axs[1].set_xticks(range(len(names)))
    axs[1].set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    axs[1].set_ylabel("f [px] (blue: covariance, orange: bootstrap over stickers)")
    axs[1].grid(alpha=0.3)
    fig.suptitle("Sticker-shape method (squares horizontal & axis-aligned; positions free)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_markershape.png")
    print("saved")


if __name__ == "__main__":
    main()
