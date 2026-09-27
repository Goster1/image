"""Method 1: calibration from ArUco sticker corners only (drawing geometry, exact).

* model grid (f single/fxfy, pp fixed/free, radial k1..k3, tangential)
* goodness: RMS/max, per marker / row / cart, AIC/BIC, leave-one-marker-out, leave-one-row-out,
  leave-one-cart-out (intrinsics from one cart -> pose-only fit of the other cart)
* profile of f, cluster bootstrap over stickers, cluster-robust (sandwich) covariance
* mapping uncertainty (rotation compensated) from the bootstrap
Writes work/cache/method_markers*.json/.md and results/markers_*.png.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from calib import Model, Problem, covariance, dist_trim
from common import CACHE, RESULTS, H, W, K_from, lens_json, load_json, project, rodrigues, save_json
from evaltools import grid, mapping_displacement, mapping_stats
from markerdata import load_markers, to_points

rng = np.random.default_rng(12345)
WHICH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "final"
TAG = "" if WHICH == "final" else "_" + WHICH
markers = load_markers(which=WHICH)
init = load_json(f"{CACHE}/initial_calib.json")
K0, d0 = np.array(init["K"]), np.array(init["dist"])
P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
print(f"markers: {len(markers)} ({markers[0]['src']}), valid corners: {sum(m['valid'].sum() for m in markers)}")

SPECS = {
    "k1_ppfix": dict(f="single", pp="fixed", dist=["k1"]),
    "k1k2_ppfix": dict(f="single", pp="fixed", dist=["k1", "k2"]),
    "k1_ppfree": dict(f="single", pp="free", dist=["k1"]),
    "k1k2_ppfree": dict(f="single", pp="free", dist=["k1", "k2"]),
    "k1k2k3_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "k3"]),
    "k1k2_ppfree_fxfy": dict(f="fxfy", pp="free", dist=["k1", "k2"]),
    "k1k2p1p2_ppfree": dict(f="single", pp="free", dist=["k1", "k2", "p1", "p2"]),
}


FOLD = "--nofold" not in sys.argv  # forbid lenses that fold inside the image (default on)


def fit(spec, pts, carts=(80, 310), x0=None, fix=None):
    m = Model(spec)
    pr = Problem(m, pts, carts=list(carts))
    pr.fold_barrier = FOLD
    if x0 is None:
        x0 = np.r_[m.pack(K0, d0), np.ravel([P0[c] for c in carts])]
    if fix is None:
        best = None
        for fs in (1.0, 0.85, 1.15):
            xs = x0.copy()
            xs[0] *= fs
            if m.f == "fxfy":
                xs[1] *= fs
            r = pr.solve(xs)
            if best is None or r.cost < best.cost:
                best = r
        r = best
    else:  # fix = (index, value) of an intrinsic parameter
        i, v = fix
        keep = np.ones(len(x0), bool)
        keep[i] = False

        def res(y):
            x = np.empty(len(x0))
            x[keep] = y
            x[i] = v
            return pr.residuals(x)

        rr = least_squares(res, x0[keep], method="lm", x_scale="jac", max_nfev=20000, xtol=1e-12, ftol=1e-12)
        x = np.empty(len(x0))
        x[keep] = rr.x
        x[i] = v
        rr.x = x
        r = rr
    res = pr.predict(r.x) - pr.uv
    return m, pr, r, res  # note: r.fun may carry the extra fold-barrier element


def rms(res):
    return float(np.sqrt(np.mean(np.sum(res ** 2, 1)))) if len(res) else float("nan")


def cluster_sandwich(pr, r, res, groups):
    """Cluster-robust covariance: (J'J)^-1 (sum_g J_g' e_g e_g' J_g) (J'J)^-1."""
    g2 = np.repeat(groups, 2)
    J = r.jac[: len(g2)]  # drop the fold-barrier row (last) if present
    e = r.fun[: len(g2)]
    JTJi = np.linalg.pinv(r.jac.T @ r.jac)
    meat = np.zeros_like(JTJi)
    for g in np.unique(groups):
        s = g2 == g
        v = J[s].T @ e[s]
        meat += np.outer(v, v)
    G = len(np.unique(groups))
    return JTJi @ meat @ JTJi * G / max(G - 1, 1)


pts_all = to_points(markers)
summary = {}
fits = {}
for name, spec in SPECS.items():
    m, pr, r, res = fit(spec, pts_all)
    C, s2, dof = covariance(r)
    groups = np.array([p["mi"] for p in pts_all])
    Cs = cluster_sandwich(pr, r, res, groups)
    n_obs = 2 * len(pts_all)
    k = len(r.x)
    ssr = float(np.sum(res ** 2))
    aic = n_obs * np.log(ssr / n_obs) + 2 * k
    bic = n_obs * np.log(ssr / n_obs) + k * np.log(n_obs)
    # leave-one-marker-out
    loo = []
    for mi in sorted({p["mi"] for p in pts_all}):
        tr = [p for p in pts_all if p["mi"] != mi]
        te = [p for p in pts_all if p["mi"] == mi]
        m2, pr2, r2, _ = fit(spec, tr, x0=r.x.copy())
        Kc, dc, ps = pr2.split(r2.x)
        c = te[0]["cart"]
        pp = ps[pr2.cidx[c]]
        pred = project(np.array([p["X"] for p in te]), Kc, dc, rvec=pp[:3], tvec=pp[3:])
        loo.append(np.linalg.norm(pred - np.array([p["uv"] for p in te]), axis=1))
    loo_rms = float(np.sqrt(np.mean(np.concatenate(loo) ** 2)))
    # leave-one-row-out
    rows = {}
    for row in ("top", "A", "B", "C"):
        te_m = {mi for mi, mk in enumerate(markers) if mk["row"] == row}
        te = [p for p in pts_all if p["mi"] in te_m]
        if not te:
            continue
        tr = [p for p in pts_all if p["mi"] not in te_m]
        try:
            m2, pr2, r2, _ = fit(spec, tr, x0=r.x.copy())
            Kc, dc, ps = pr2.split(r2.x)
            e = []
            for p in te:
                pp = ps[pr2.cidx[p["cart"]]]
                e.append(np.linalg.norm(project(p["X"][None], Kc, dc, rvec=pp[:3], tvec=pp[3:])[0] - p["uv"]))
            rows[row] = float(np.sqrt(np.mean(np.square(e))))
        except Exception as ex:  # noqa
            rows[row] = None
    # leave-one-cart-out: intrinsics from one cart, pose-only fit on the other
    lco = {}
    for c_tr, c_te in ((80, 310), (310, 80)):
        tr = [p for p in pts_all if p["cart"] == c_tr]
        te = [p for p in pts_all if p["cart"] == c_te]
        try:
            m2, pr2, r2, res2 = fit(spec, tr, carts=(c_tr,), x0=np.r_[r.x[: m.n], P0[c_tr]])
            Kc, dc, _ = pr2.split(r2.x)
            mm = Model(dict(f="single", pp="fixed", dist=[], pp0=[Kc[0, 2], Kc[1, 2]], dist0=list(dc)))
            pr3 = Problem(mm, te, carts=[c_te])

            def rr3(y):
                return pr3.residuals(np.r_[Kc[0, 0], y])

            r3 = least_squares(rr3, P0[c_te], method="lm")
            lco[f"{c_tr}->{c_te}"] = dict(train_rms=rms(res2), test_rms=float(np.sqrt(np.mean(r3.fun ** 2) * 2)),
                                          f=float(Kc[0, 0]), intr=r2.x[: m.n].tolist())
        except Exception as ex:  # noqa
            lco[f"{c_tr}->{c_te}"] = None
    Kf, df, ps = pr.split(r.x)
    per_marker = {}
    for mi in sorted({p["mi"] for p in pts_all}):
        s = np.array([p["mi"] == mi for p in pts_all])
        mk = markers[mi]
        per_marker[f"{mk['cart']}:{mk['id']}({mk['role']})"] = rms(res[s])
    per_cart = {c: rms(res[np.array([p["cart"] == c for p in pts_all])]) for c in (80, 310)}
    from common import fold_margin
    summary[name] = dict(spec=m.describe(), names=m.names, fold_margin=fold_margin(*pr.split(r.x)[:2]), x=r.x[: m.n].tolist(), sigma_cov=np.sqrt(np.diag(C))[: m.n].tolist(),
                         sigma_sandwich=np.sqrt(np.diag(Cs))[: m.n].tolist(), rms=rms(res), max=float(np.max(np.linalg.norm(res, axis=1))),
                         n_points=len(pts_all), dof=dof, aic=float(aic), bic=float(bic), loo_marker_rms=loo_rms,
                         loo_row_rms=rows, leave_cart=lco, per_marker_rms=per_marker, per_cart_rms=per_cart,
                         K=Kf.tolist(), dist=dist_trim(df).tolist(), poses={80: ps[0].tolist(), 310: ps[1].tolist()})
    fits[name] = (m, pr, r, res)
    print(f"{name:18s} rms {rms(res):6.3f} max {np.max(np.linalg.norm(res,axis=1)):6.2f} LOO {loo_rms:6.3f} rows {({k: round(v,2) if v else v for k,v in rows.items()})} "
          f"AIC {aic:8.1f} BIC {bic:8.1f} | " + " ".join(f"{nm}={v:.4g}±{s:.3g}" for nm, v, s in zip(m.names, r.x[: m.n], np.sqrt(np.diag(Cs))[: m.n])))
    for kk, v in lco.items():
        if v:
            print(f"      leave-cart {kk}: train {v['train_rms']:.2f} test {v['test_rms']:.2f} f={v['f']:.0f}")

# model choice: smallest leave-one-marker-out error (prediction), ties -> fewer params
best = min(summary, key=lambda k: summary[k]["loo_marker_rms"] + 1e-3 * len(summary[k]["x"]))
print("best by LOO:", best)

# ---------------- profile of f (for the best model and the k1-only model) ----------------
prof = {}
for name in sorted({best, "k1_ppfix", "k1k2_ppfree"}):
    m, pr, r, res = fits[name]
    f0 = r.x[0]
    fs = np.linspace(f0 * 0.8, f0 * 1.2, 41)
    out = []
    x = r.x.copy()
    for fv in fs:
        m2, pr2, r2, res2 = fit(SPECS[name], pts_all, x0=x, fix=(0, fv))
        if m2.f == "fxfy":
            pass
        out.append((fv, rms(res2)))
    prof[name] = np.array(out)

# ---------------- cluster bootstrap over stickers (best model) ----------------
m, pr, r, res = fits[best]
mis = sorted({p["mi"] for p in pts_all})
B = 100
boot = []
for b in range(B):
    pick = rng.choice(mis, size=len(mis), replace=True)
    bp = []
    for q, mi in enumerate(pick):
        bp += [dict(p, mi=1000 * q + mi) for p in pts_all if p["mi"] == mi]
    carts_b = sorted({p["cart"] for p in bp})
    if len(carts_b) < 2:
        continue
    try:
        m2, pr2, r2, res2 = fit(SPECS[best], bp, x0=r.x.copy())
        boot.append(r2.x[: m.n])
    except Exception:
        pass
boot = np.array(boot)
print("bootstrap", len(boot), "std", np.round(boot.std(0), 4))

Kb, db, _ = pr.split(r.x)
uv, shp = grid(40)
disp = []
for xb in boot:
    Kc, dc = m.unpack(xb)
    try:
        disp.append(mapping_displacement(Kb, db, Kc, dc, uv, compensate=True))
    except Exception:
        pass
disp = np.array(disp)
mstats, mrms = mapping_stats(disp, uv)
disp_raw = np.array([mapping_displacement(Kb, db, *m.unpack(xb), uv, compensate=False) for xb in boot])
mstats_raw, _ = mapping_stats(disp_raw, uv)
print("mapping 1-sigma (rot. compensated):", {k: round(v["rms_median_px"], 2) for k, v in mstats.items()})

# ---------------- plots ----------------
img = plt.imread(f"{CACHE}/mean_aligned_color.png")[..., ::-1] if False else None
import cv2

col = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
ax.imshow(col)
P = pr.predict(r.x)
for mi in mis:
    s = np.array([p["mi"] == mi for p in pts_all])
    ax.plot(*np.vstack([pr.uv[s], pr.uv[s][:1]]).T, "-", color="lime", lw=0.8)
    ax.plot(*np.vstack([P[s], P[s][:1]]).T, "-", color="red", lw=0.8)
ax.quiver(pr.uv[:, 0], pr.uv[:, 1], res[:, 0] * 10, res[:, 1] * 10, angles="xy", scale_units="xy", scale=1, color="yellow", width=0.0015)
ax.set_title(f"Marker-only fit ({summary[best]['spec']}): detected (green) vs predicted (red); residual x10 (yellow). RMS {summary[best]['rms']:.2f} px")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(f"{RESULTS}/markers_residuals{TAG}.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
for name, pf in prof.items():
    ax.plot(pf[:, 0], pf[:, 1], "-o", ms=3, label=name)
ax.set_xlabel("focal length f [px] (fixed)")
ax.set_ylabel("RMS reprojection error [px] (others refit)")
ax.set_title("Marker-only: profile of f")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS}/markers_f_profile{TAG}.png")
plt.close(fig)

# ---------------- outputs ----------------
def lens_out(name, method_note):
    s = summary[name]
    unc = {f"{nm}_1sigma": float(max(a, b)) for nm, a, b in zip(s["names"], s["sigma_sandwich"], s["sigma_cov"])}
    if len(boot) and name == best:
        unc.update({f"{nm}_1sigma_bootstrap": float(v) for nm, v in zip(s["names"], boot.std(0))})
    K = np.array(s["K"])
    return lens_json(K, s["dist"], method="markers_only" + ("" if name == best else f"_{name}"),
                     model=s["spec"], rms_reprojection_error_px=s["rms"], uncertainty=unc,
                     mapping_uncertainty_px={k: v["rms_median_px"] for k, v in mstats.items() if k != "whole_image"} if name == best else None,
                     data_used=f"{len(mis)} stickers, {len(pts_all)} corners ({markers[0]['src']} detections, 7-frame mean)",
                     geometry_assumptions="drawing dimensions and sticker positions exactly as in spec/cart-marker-layout.json",
                     details=dict(summary=s, note=method_note))


save_json(dict(best=best, summary=summary, profile={k: v.tolist() for k, v in prof.items()}, bootstrap=boot.tolist(),
               mapping=mstats, mapping_raw=mstats_raw), f"{CACHE}/markers_ba_full{TAG}.json")
save_json(lens_out(best, "best model by leave-one-sticker-out prediction error"), f"{CACHE}/method_markers{TAG}.json")
save_json([lens_out(n, "alternative model") for n in summary if n != best], f"{CACHE}/method_markers_alternatives{TAG}.json")
print("done")
