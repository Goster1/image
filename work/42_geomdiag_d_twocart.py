"""(4) TWO-CART consistency: both carts stand on the same floor, so the camera height above the floor and
the floor normal (cart Z axis in the camera frame) must agree between the two cart poses.

For a grid of FIXED focal lengths f (plumb-line distortion fixed in pixel units: k1 = k1px f^2,
k2 = k2px f^4), each cart gets its own 6-DoF pose from its stickers (drawing geometry); the principal
point is either fixed at the plumb-line distortion centre (image centre) or free and common to both
carts. Point sets: (a) all stickers, (b) shelf stickers only (A, B, C), (c) top stickers only.
Discrepancies vs f: dh = h(80) - h(310) (camera height above the floor, mm) and the angle between
the two floor normals (plus its two components: along the camera x axis and y axis after projecting
on the image plane directions). The f at which the carts agree (dh = 0) is found by root finding;
its uncertainty from (i) the pose covariance (Gauss-Newton, scaled by the residual variance) and
(ii) a bootstrap over stickers (resampled within each cart, everything refitted).
Writes work/cache/method_twocart.json (+ .md) and results/geomdiag_twocart.png; run 42_geomdiag_h_mapping.py afterwards
to add mapping_uncertainty_px. Optional: GEOMDIAG_EDGES_DIR = directory with a frozen copy of the edge files.
"""
import importlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq, least_squares  # noqa: E402

from common import CACHE, RESULTS, K_from, lens_json, load_json, project, rodrigues, save_json  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")
C = importlib.import_module("42_geomdiag_c_fits")
rng = np.random.default_rng(7)
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 150
FGRID = np.arange(1100.0, 1801.0, 20.0)
CARTS = (80, 310)


class TwoCart:
    def __init__(self, P, pl, pp_free):
        self.P = P
        self.pl = pl
        self.pp_free = pp_free
        self.uv = np.array([p["uv"] for p in P])
        self.X = np.array([p["X"] for p in P])
        self.ci = np.array([CARTS.index(p["cart"]) for p in P])

    def lens(self, f, pp):
        dp = self.pl["dist_px"]
        return K_from(f, f, pp[0], pp[1]), np.array([dp[0] * f ** 2, dp[1] * f ** 4, 0, 0, dp[4] * f ** 6])

    def res(self, x, f):
        pp = x[:2] if self.pp_free else self.pl["centre"]
        poses = x[2:].reshape(2, 6) if self.pp_free else x.reshape(2, 6)
        K, d = self.lens(f, pp)
        pr = np.zeros_like(self.uv)
        for k in range(2):
            s = self.ci == k
            if s.any():
                pr[s] = project(self.X[s], K, d, rvec=poses[k, :3], tvec=poses[k, 3:])
        return (pr - self.uv).ravel()

    def fit(self, f, x0):
        r = least_squares(self.res, x0, args=(f,), x_scale="jac", method="trf", max_nfev=3000, xtol=1e-12, ftol=1e-12)
        return r

    def derived(self, x):
        poses = x[2:].reshape(2, 6) if self.pp_free else x.reshape(2, 6)
        out = {}
        z = []
        for k, c in enumerate(CARTS):
            R = rodrigues(poses[k, :3])
            Cc = -R.T @ poses[k, 3:]
            out[f"h_{c}"] = float(Cc[2] + 1800.0)
            z.append(R[:, 2])
        out["dh"] = out["h_80"] - out["h_310"]
        out["angle_deg"] = float(np.degrees(np.arccos(np.clip(z[0] @ z[1], -1, 1))))
        dz = z[0] - z[1]  # small-angle components in the camera frame (x: image right, y: image down)
        out["dnx_deg"] = float(np.degrees(dz[0]))
        out["dny_deg"] = float(np.degrees(dz[1]))
        return out


def x0_for(tc, f, prev=None):
    if prev is not None:
        return prev.copy()
    init = load_json(f"{CACHE}/initial_calib.json")
    poses = list(init["poses"]["80"]) + list(init["poses"]["310"])
    x = ([959.5, 539.5] if tc.pp_free else []) + poses
    # rescale depth with f (initial poses at f ~ 1289)
    x = np.array(x, float)
    k = 2 if tc.pp_free else 0
    for c in range(2):
        x[k + 6 * c + 3:k + 6 * c + 6] *= f / 1289.0
    return x


def profile(tc):
    rows = []
    prev = None
    # start from the middle of the grid and walk out both ways for robust warm starts
    order = np.argsort(np.abs(FGRID - 1400))
    sol = {}
    for i in order:
        f = FGRID[i]
        near = min(sol, key=lambda g: abs(g - f)) if sol else None
        x0 = x0_for(tc, f, sol[near] if near is not None else None)
        r = tc.fit(f, x0)
        sol[f] = r.x
        d = tc.derived(r.x)
        J = r.jac
        dof = max(len(r.fun) - len(r.x), 1)
        s2 = 2 * r.cost / dof
        Cx = np.linalg.pinv(J.T @ J) * s2
        # delta-method sigma of dh and angle components
        eps = 1e-6
        g_dh, g_nx, g_ny = [], [], []
        for j in range(len(r.x)):
            xp = r.x.copy()
            h = eps * max(1.0, abs(r.x[j]))
            xp[j] += h
            dp = tc.derived(xp)
            g_dh.append((dp["dh"] - d["dh"]) / h)
            g_nx.append((dp["dnx_deg"] - d["dnx_deg"]) / h)
            g_ny.append((dp["dny_deg"] - d["dny_deg"]) / h)
        g_dh, g_nx, g_ny = map(np.array, (g_dh, g_nx, g_ny))
        d.update(f=float(f), rms=float(np.sqrt(np.mean(np.sum(r.fun.reshape(-1, 2) ** 2, 1)))),
                 sd_dh=float(np.sqrt(g_dh @ Cx @ g_dh)), sd_dnx=float(np.sqrt(g_nx @ Cx @ g_nx)), sd_dny=float(np.sqrt(g_ny @ Cx @ g_ny)),
                 pp=(r.x[:2].tolist() if tc.pp_free else list(tc.pl["centre"])))
        rows.append(d)
    rows.sort(key=lambda r: r["f"])
    return rows, sol


def root(rows, key):
    f = np.array([r["f"] for r in rows])
    y = np.array([r[key] for r in rows])
    s = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    out = []
    for i in s:
        # linear interpolation, then report local slope
        fr = f[i] - y[i] * (f[i + 1] - f[i]) / (y[i + 1] - y[i])
        slope = (y[i + 1] - y[i]) / (f[i + 1] - f[i])
        out.append(dict(f=float(fr), slope_per_px=float(slope), i=int(i)))
    return out


def exact_root(tc, sol, key, fa, fb):
    """Root of derived[key](f) by Brent with warm starts."""
    cache = {}

    def g(f):
        near = min(sol, key=lambda q: abs(q - f))
        r = tc.fit(f, sol[near])
        cache[f] = r
        return tc.derived(r.x)[key]

    try:
        fr = brentq(g, fa, fb, xtol=0.05)
    except ValueError:
        return None, None
    near = min(cache, key=lambda q: abs(q - fr))
    return float(fr), cache[near]


def main():
    ms = G.markers()
    P_all = C.points(ms)
    edges = G.all_edges()
    pl = G.plumb("k1k2_centre_fixed", edges)
    sets = {"a_all": lambda p: True, "b_shelf": lambda p: p["level"] != "top", "c_top": lambda p: p["level"] == "top"}
    out = {"plumb": {k: pl[k] for k in ("centre", "dist_px", "dist", "rms")}, "fgrid": FGRID.tolist(), "profiles": {}, "roots": {}}
    for sname, sel in sets.items():
        P = [p for p in P_all if sel(p)]
        for pp_free in (False, True):
            tc = TwoCart(P, pl, pp_free)
            rows, sol = profile(tc)
            key = f"{sname}|pp_{'free' if pp_free else 'fixed'}"
            out["profiles"][key] = rows
            rr = {}
            for q in ("dh", "dnx_deg", "dny_deg"):
                rts = root(rows, q)
                ex = []
                for rt in rts:
                    fr, r = exact_root(tc, sol, q, FGRID[rt["i"]], FGRID[rt["i"] + 1])
                    if fr is None:
                        continue
                    d = tc.derived(r.x)
                    row_i = rows[rt["i"]]
                    sd_key = {"dh": "sd_dh", "dnx_deg": "sd_dnx", "dny_deg": "sd_dny"}[q]
                    ex.append(dict(f=fr, slope_per_px=rt["slope_per_px"], sd_f_cov=float(row_i[sd_key] / abs(rt["slope_per_px"])),
                                   rms=float(np.sqrt(np.mean(np.sum(r.fun.reshape(-1, 2) ** 2, 1)))), derived=d,
                                   pp=(r.x[:2].tolist() if pp_free else list(pl["centre"]))))
                rr[q] = ex
            out["roots"][key] = rr
            print(key, {q: [(round(e["f"], 1), round(e["sd_f_cov"], 1), round(e["rms"], 2)) for e in v] for q, v in rr.items()})
            for r_ in rows[::5]:
                print(f"   f {r_['f']:.0f} rms {r_['rms']:.2f} h80 {r_['h_80']:.0f} h310 {r_['h_310']:.0f} dh {r_['dh']:.1f}+-{r_['sd_dh']:.1f} "
                      f"angle {r_['angle_deg']:.2f} dnx {r_['dnx_deg']:.2f}+-{r_['sd_dnx']:.2f} dny {r_['dny_deg']:.2f}+-{r_['sd_dny']:.2f} pp {np.round(r_['pp'], 0)}")

    # ---------------- bootstrap over stickers for the roots of dnx (floor normals agree) and dh ----------------
    boot = {}
    for sname in ("a_all", "b_shelf"):
        for pp_free in (False, True):
            key = f"{sname}|pp_{'free' if pp_free else 'fixed'}"
            for q in ("dnx_deg", "dh"):
                base = [e for e in out["roots"][key][q] if 1200 < e["f"] < 1650]
                if len(base) != 1:
                    boot[f"{key}|{q}"] = None
                    continue
                f0 = base[0]["f"]
                sel = sets[sname]
                vals = []
                for b in range(NBOOT):
                    P = []
                    for c in CARTS:
                        sids = sorted({p["id"] for p in P_all if p["cart"] == c and sel(p)})
                        pick = rng.choice(sids, len(sids), replace=True)
                        for qq, sid in enumerate(pick):
                            P += [dict(p, id=(sid, qq)) for p in P_all if p["cart"] == c and p["id"] == sid and sel(p)]
                    tc = TwoCart(P, pl, pp_free)
                    sol = {}
                    try:
                        grid = (f0 - 200, f0 - 100, f0, f0 + 100, f0 + 200)
                        prev = None
                        for f in (f0, f0 - 100, f0 - 200, f0 + 100, f0 + 200):
                            r = tc.fit(f, x0_for(tc, f, sol.get(f0)))
                            sol[f] = r.x
                        ys = [tc.derived(sol[f])[q] for f in grid]
                        idx = [i for i in range(4) if np.sign(ys[i]) != np.sign(ys[i + 1])]
                        if not idx:
                            vals.append(np.nan)
                            continue
                        i = min(idx, key=lambda i: abs(grid[i] + 50 - f0))
                        fr, _ = exact_root(tc, sol, q, grid[i], grid[i + 1])
                        vals.append(np.nan if fr is None else fr)
                    except Exception:  # noqa
                        vals.append(np.nan)
                v = np.array(vals, float)
                boot[f"{key}|{q}"] = dict(n=int(np.isfinite(v).sum()), n_fail=int((~np.isfinite(v)).sum()), mean=float(np.nanmean(v)),
                                          std=float(np.nanstd(v)), q16=float(np.nanpercentile(v, 16)), q84=float(np.nanpercentile(v, 84)))
                print("bootstrap", key, q, {k: round(x, 1) for k, x in boot[f"{key}|{q}"].items()})
    out["bootstrap_roots"] = boot
    save_json(out, f"{CACHE}/geomdiag_twocart.json")

    # ---------------- method json (main variant: all stickers, pp free; criterion: floor normals agree) ----------------
    key = "a_all|pp_free"
    rt = [e for e in out["roots"][key]["dnx_deg"] if 1200 < e["f"] < 1650][0]
    f = rt["f"]
    pp = rt["pp"]
    K = K_from(f, f, pp[0], pp[1])
    dp = pl["dist_px"]
    d = np.array([dp[0] * f ** 2, dp[1] * f ** 4, 0, 0, dp[4] * f ** 6])
    bs = boot.get(key + "|dnx_deg") or {}
    rt_b = [e for e in out["roots"]["b_shelf|pp_free"]["dnx_deg"] if 1200 < e["f"] < 1650]
    syst = abs(rt_b[0]["f"] - f) / 2 if rt_b else float("nan")
    sd_stat = float(np.hypot(rt["sd_f_cov"], bs.get("std", 0.0)))
    sd_f = float(np.hypot(sd_stat, syst))
    lj = lens_json(K, d, method="two_cart_consistency (floor normals of the two cart poses parallel)",
                   model="fx=fy; pp free (common); k1,k2 fixed from the plumb-line fit in pixel units",
                   rms_reprojection_error_px=rt["rms"],
                   uncertainty={"fx_px_1sigma": sd_f, "fx_px_1sigma_statistical": sd_stat, "fx_px_1sigma_covariance": rt["sd_f_cov"],
                                "fx_px_1sigma_bootstrap_stickers": bs.get("std"), "fx_px_systematic_half_diff_all_vs_shelf_only": syst,
                                "cx_cy_note": "pp from the joint fit at this f (not separately profiled)",
                                "note": "f where the x-component (camera frame) of the difference of the two cart Z axes vanishes; the y-component "
                                        "is weakly determined; the camera-height difference h(80)-h(310) is insensitive to f (|dh| < 5 mm over "
                                        "f = 1100..1800 with pp fixed) and gives no usable constraint"},
                   data_used="62 valid sticker corners of both carts (drawing geometry) + plumb-line distortion from the verified edges",
                   geometry_assumptions="drawing dimensions exactly as in the spec (main); shelf-only and top-only variants in details",
                   details=dict(roots={k: {q: [(e["f"], e["sd_f_cov"], e["rms"]) for e in v[q]] for q in v} for k, v in out["roots"].items()},
                                roots_full=out["roots"], bootstrap=boot, plumb=out["plumb"]))
    save_json(lj, f"{CACHE}/method_twocart.json")

    # ---------------- plot ----------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), dpi=110)
    for key, rows in out["profiles"].items():
        f_ = [r["f"] for r in rows]
        ls = "-" if "pp_free" in key else "--"
        col = {"a_all": "C0", "b_shelf": "C1", "c_top": "C2"}[key.split("|")[0]]
        axs[0].plot(f_, [r["dh"] for r in rows], ls, color=col, label=key)
        axs[0].fill_between(f_, [r["dh"] - r["sd_dh"] for r in rows], [r["dh"] + r["sd_dh"] for r in rows], color=col, alpha=0.08)
        axs[1].plot(f_, [r["dnx_deg"] for r in rows], ls, color=col, label=key)
        axs[1].fill_between(f_, [r["dnx_deg"] - r["sd_dnx"] for r in rows], [r["dnx_deg"] + r["sd_dnx"] for r in rows], color=col, alpha=0.08)
        axs[2].plot(f_, [r["rms"] for r in rows], ls, color=col, label=key)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_ylabel("h(80) - h(310)  [mm]  (camera height above the floor)")
    axs[1].set_ylabel("floor-normal difference, x component [deg] (Z80 - Z310)")
    axs[1].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel("sticker RMS [px]")
    for ax in axs:
        ax.set_xlabel("fixed f [px] (plumb distortion fixed in pixel units)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[0].set_ylim(-400, 400)
    fig.suptitle("Two-cart consistency vs focal length (drawing geometry; a: all stickers, b: shelf only, c: top only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_twocart.png")
    with open(f"{CACHE}/method_twocart.md", "w") as fh:
        fh.write("# Method: two-cart consistency\n\n")
        fh.write("Both carts stand on the same floor: the camera height above the floor and the floor normal from the two cart poses must agree. "
                 "Profile over fixed f with the plumb-line distortion fixed in pixel units (script 42_geomdiag_d_twocart.py).\n\n")
        fh.write("| point set / pp | f where floor normals agree (dnx = 0) [px] (+-cov) | RMS [px] | bootstrap std | f where dh = 0 |\n|---|---|---|---|---|\n")
        for k, v in out["roots"].items():
            b = boot.get(k)
            b = boot.get(k + "|dnx_deg")
            roots_txt = ", ".join("%.0f +- %.0f" % (e["f"], e["sd_f_cov"]) for e in v["dnx_deg"]) or "none in grid"
            rms_txt = ", ".join("%.2f" % e["rms"] for e in v["dnx_deg"])
            b_txt = "" if not b else "%.0f (n=%d)" % (b["std"], b["n"])
            dh_txt = ", ".join("%.0f +- %.0f" % (e["f"], e["sd_f_cov"]) for e in v["dh"]) or "no sign change in grid"
            fh.write(f"| {k} | {roots_txt} | {rms_txt} | {b_txt} | {dh_txt} |\n")
        fh.write(f"\nMain (all stickers, pp free, floor normals parallel): f = {f:.0f} +- {sd_f:.0f} px "
                 f"(statistical {sd_stat:.0f}, systematic all-vs-shelf-only half difference {syst:.0f}), pp = ({pp[0]:.0f}, {pp[1]:.0f}). "
                 "The camera height difference is insensitive to f. Geometry-dependent (drawing-geometry RMS ~6 px); see geometry_diagnosis.md.\n")
    print("saved")


if __name__ == "__main__":
    main()
