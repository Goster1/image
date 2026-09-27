"""(1) PLUMB-LINE distortion from the straightness of the verified straight_3d edges (independent implementation).

Residuals are distances in the DISTORTED image (Sampson / first-order, see 40_lines_indep_lib.py).
Pixel-unit parameterisation (S = 1000 px): only k_i / f^(2i) (and p_i / f) are identifiable from straightness.

Steps
  1. robust edge rejection (iterative, flexible model k1k2k3 + free centre): edges that stay curved
  2. all models on the accepted set: k1; k1k2; k1k2k3; + free centre; + tangential; division 1-2 (+ centre)
     -> rms, classic and cluster-sandwich covariance, leave-one-cluster-out cross-validation
  3. cluster bootstrap over edges (resample physical members with replacement)
  4. per-region fits (scene / cart310 / cart80) with bootstrap
  5. distortion-centre profile (grid of fixed centres) and k-centre correlations
  6. plots + crops of rejected edges
Writes work/cache/40_lines_indep_plumb.json (+ bootstrap samples .npz), results/40_lines_indep_plumb_*.png
Usage: python3 40_lines_indep_a_plumb.py [NBOOT]
"""
import importlib
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(4001)
T0 = time.time()

edges, dropped = L.load_edges_indep()
status = L.file_status()
print("edge files reviewed:", status)
print(f"{len(edges)} straight edges after dropping duplicates {[(a, b) for a, b, _ in dropped]}")

MODELS = {
    "k1": L.PlumbModel("poly", 1),
    "k1k2": L.PlumbModel("poly", 2),
    "k1k2k3": L.PlumbModel("poly", 3),
    "k1_c": L.PlumbModel("poly", 1, centre_free=True),
    "k1k2_c": L.PlumbModel("poly", 2, centre_free=True),
    "k1k2k3_c": L.PlumbModel("poly", 3, centre_free=True),
    "k1k2_t": L.PlumbModel("poly", 2, tang=True),
    "k1k2_c_t": L.PlumbModel("poly", 2, tang=True, centre_free=True),
    "div1": L.PlumbModel("div", 1),
    "div2": L.PlumbModel("div", 2),
    "div1_c": L.PlumbModel("div", 1, centre_free=True),
    "div2_c": L.PlumbModel("div", 2, centre_free=True),
}
BOOT_MODELS = ["k1", "k1k2", "k1k2k3", "k1k2_c", "k1k2k3_c", "k1k2_t", "div2", "div2_c"]
F_REF = 1300.0  # only to quote OpenCV-style k's in the tables (k_i at f0 = 1300); invariants are f-free

# ---------------------------------------------------------------- 1. robust rejection
print("\n[1] robust rejection (k1k2k3 + free centre, fac 2.5, bow floor 0.6 px, rms floor 0.5 px)")
act, hist, st_all = L.robust_edge_rejection(edges, MODELS["k1k2k3_c"], fac=2.5)
acc = [e for e, a in zip(edges, act) if a]
rej = [e for e, a in zip(edges, act) if not a]
print("rejected:", [e["id"] for e in rej])
es = L.EdgeSet(acc)
es_all = L.EdgeSet(edges)


def summarize(fit):
    se_cl = np.sqrt(np.diag(fit["cov_cluster"]))
    se_c = np.sqrt(np.diag(fit["cov_classic"]))
    C = fit["cov_cluster"]
    corr = C / np.outer(se_cl, se_cl)
    d = fit["dist"]
    out = dict(names=fit["names"], x=fit["x"].tolist(), se_cluster=se_cl.tolist(), se_classic=se_c.tolist(),
               corr_cluster=np.round(corr, 3).tolist(), rms=fit["rms"], cost=fit["cost"], n_points=len(fit["r"]),
               centre=d.c.tolist())
    if d.kind == "poly":
        out["k_at_f1300"] = d.opencv(F_REF).tolist()
        out["invariants"] = d.invariants()
    return out


# ---------------------------------------------------------------- 2. models + LOCO-CV
print("\n[2] models on the accepted set")
cl_ids, cl_names = L.cluster_ids(acc)
fits, table = {}, {}
for nm, m in MODELS.items():
    fit = L.fit_plumb(es, m)
    fits[nm] = fit
    table[nm] = summarize(fit)
# leave-one-cluster-out: fit without the cluster, evaluate the left-out edges' straightness (own lines)
for nm, m in MODELS.items():
    sq, n = 0.0, 0
    per = []
    for c in range(len(cl_names)):
        tr = [e for e, ci in zip(acc, cl_ids) if ci != c]
        te = [e for e, ci in zip(acc, cl_ids) if ci == c]
        f = L.fit_plumb(L.EdgeSet(tr), m, x0=fits[nm]["x"])
        est = L.EdgeSet(te)
        U, JU, ok = f["dist"].undistort(est.P)
        r = est.residuals(U, JU)
        sq += float(r @ r)
        n += len(r)
        per.append(float(np.sqrt(np.mean(r ** 2))))
    table[nm]["loco_cv_rms"] = float(np.sqrt(sq / n))
    table[nm]["loco_cv_median_cluster_rms"] = float(np.median(per))
    t = table[nm]
    print(f"  {nm:9s} rms {t['rms']:.4f}  LOCO-CV {t['loco_cv_rms']:.4f}  ",
          " ".join(f"{a}={v:.5g}±{s:.2g}" for a, v, s in zip(t["names"], t["x"], t["se_cluster"])),
          " k@1300", np.round(t.get("k_at_f1300", [np.nan] * 5), 4)[[0, 1, 4]])

# per-edge residual stats for the main models (all edges, incl. rejected, evaluated with the accepted-set fit)
per_edge = {}
for nm in ["k1k2", "k1k2_c", "k1k2k3_c"]:
    d = fits[nm]["dist"]
    U, JU, _ = d.undistort(es_all.P)
    r = es_all.residuals(U, JU)
    st = es_all.per_edge_stats(r)
    per_edge[nm] = {e["id"]: dict(s, accepted=bool(a)) for e, s, a in zip(edges, st, act)}
# raw straightness (no correction) for reference
U0, J0, _ = L.Dist("poly", [0.0]).undistort(es_all.P)
r0 = es_all.residuals(U0, J0)
per_edge["none"] = {e["id"]: s for e, s in zip(edges, es_all.per_edge_stats(r0))}


# ---------------------------------------------------------------- 3. bootstrap over clusters
def resample(edge_list, rng_):
    ids, names = L.cluster_ids(edge_list)
    pick = rng_.integers(0, len(names), len(names))
    out = []
    for q, c in enumerate(pick):
        out += [dict(e, cluster=f"{e['cluster']}#{q}") for e, ci in zip(edge_list, ids) if ci == c]
    return out


def bootstrap(edge_list, model, x0, nb, rng_):
    xs = []
    for b in range(nb):
        sub = resample(edge_list, rng_)
        try:
            f = L.fit_plumb(L.EdgeSet(sub), model, x0=x0, max_nfev=100)
            xs.append(f["x"])
        except Exception as ex:  # noqa
            print("   boot fail", ex)
    return np.array(xs)


def robust_sd(a):
    a = np.asarray(a)
    q = np.percentile(a, [15.87, 84.13], axis=0)
    return (q[1] - q[0]) / 2


print(f"\n[3] cluster bootstrap ({NB} reps) over {len(cl_names)} clusters")
boot = {}
for nm in BOOT_MODELS:
    t = time.time()
    xs = bootstrap(acc, MODELS[nm], fits[nm]["x"], NB, rng)
    boot[nm] = xs
    table[nm]["boot_mean"] = xs.mean(0).tolist()
    table[nm]["boot_sd"] = xs.std(0, ddof=1).tolist()
    table[nm]["boot_sd_robust"] = robust_sd(xs).tolist()
    table[nm]["boot_corr"] = np.round(np.corrcoef(xs.T), 3).tolist()
    table[nm]["n_boot"] = len(xs)
    print(f"  {nm:9s} ({time.time() - t:.0f}s) boot sd", np.array2string(np.array(table[nm]["boot_sd"]), precision=4),
          " robust", np.array2string(np.array(table[nm]["boot_sd_robust"]), precision=4),
          " sandwich", np.array2string(np.array(table[nm]["se_cluster"]), precision=4))

# ---------------------------------------------------------------- 4. per-region fits
print("\n[4] per-region fits")
regions = {}
REG_BOOT = max(NB * 2 // 3, 150)
for reg in ["scene", "cart310", "cart80"]:
    sub = [e for e in acc if L.region_of(e) == reg]
    for nm in ["k1k2", "k1k2_c", "k1k2k3"]:
        f = L.fit_plumb(L.EdgeSet(sub), MODELS[nm], x0=fits[nm]["x"])
        s = summarize(f)
        xs = bootstrap(sub, MODELS[nm], f["x"], REG_BOOT, rng)
        s["boot_sd"] = xs.std(0, ddof=1).tolist()
        s["boot_sd_robust"] = robust_sd(xs).tolist()
        s["n_edges"] = len(sub)
        s["n_clusters"] = len({e["cluster"] for e in sub})
        regions[f"{reg}_{nm}"] = s
        boot[f"{reg}_{nm}"] = xs
        print(f"  {reg:8s} {nm:7s} n={len(sub):2d} rms {s['rms']:.3f} ",
              " ".join(f"{a}={v:.5g}±{q:.2g}" for a, v, q in zip(s["names"], s["x"], s["boot_sd_robust"])))

# ---------------------------------------------------------------- 5. centre profile
print("\n[5] distortion-centre profile")
gx = np.arange(760, 1161, 25.0)
gy = np.arange(240, 841, 25.0)
prof = {}
for nm, nrad in [("k1k2", 2), ("k1k2k3", 3)]:
    Z = np.zeros((len(gy), len(gx)))
    A1 = np.zeros_like(Z)
    x = fits[nm]["x"]
    for iy, cy in enumerate(gy):
        for ix, cx in enumerate(gx):
            m = L.PlumbModel("poly", nrad, centre=(cx, cy))
            f = L.fit_plumb(es, m, x0=x, max_nfev=60)
            Z[iy, ix] = f["cost"]
            A1[iy, ix] = f["x"][0]
    prof[nm] = dict(cost=Z, a1=A1)
    iy, ix = np.unravel_index(np.argmin(Z), Z.shape)
    print(f"  {nm}: grid minimum at ({gx[ix]:.0f}, {gy[iy]:.0f}); cost range {Z.min():.1f}..{Z.max():.1f}")


# ---------------------------------------------------------------- radial displacement curves
def disp_curve(d, rr):
    """undistortion displacement (px, outward positive) at DISTORTED radius rr along the +x axis from the centre."""
    P = d.c + np.column_stack([rr, np.zeros_like(rr)])
    U, _, _ = d.undistort(P)
    return U[:, 0] - P[:, 0]


rr = np.linspace(0, 1100, 56)
curves = {}
for nm in BOOT_MODELS:
    m = MODELS[nm]
    cs = np.array([disp_curve(m.dist(x), rr) for x in boot[nm]])
    curves[nm] = dict(best=disp_curve(fits[nm]["dist"], rr), lo=np.percentile(cs, 15.87, 0), hi=np.percentile(cs, 84.13, 0))
for key in [k for k in boot if "_" in k and k.split("_")[0] in ("scene", "cart310", "cart80")]:
    reg, nm = key.split("_", 1)
    m = MODELS[nm]
    cs = np.array([disp_curve(m.dist(x), rr) for x in boot[key]])
    xb = np.array(regions[key]["x"])
    curves[key] = dict(best=disp_curve(m.dist(xb), rr), lo=np.percentile(cs, 15.87, 0), hi=np.percentile(cs, 84.13, 0))

# ---------------------------------------------------------------- 6. plots
img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
# (a) map of edges coloured by residual bow after correction (main model), rejected edges marked
fig, ax = plt.subplots(1, 2, figsize=(18, 5.6))
for a, nm, title in [(ax[0], "none", "no correction"), (ax[1], "k1k2k3_c", "after k1k2k3 + free centre (accepted set)")]:
    a.imshow(img, alpha=0.55)
    for e in edges:
        s = per_edge[nm][e["id"]]
        b = s["bow"]
        col = plt.cm.coolwarm(np.clip(0.5 + b / (8.0 if nm == "none" else 1.6), 0, 1))
        P = e["points"]
        a.plot(P[:, 0], P[:, 1], "-", color=col, lw=2.2)
        if e in rej:
            a.plot(P[:, 0], P[:, 1], ":", color="k", lw=1.0)
            a.text(P[len(P) // 2, 0] + 6, P[len(P) // 2, 1], e["id"].replace("cart80_", "80:").replace("scene_", "s:"), fontsize=7, color="k",
                   bbox=dict(fc="yellow", alpha=0.7, lw=0))
    a.set_title(f"residual bow per edge, {title} (red +, blue -; scale {'+-4' if nm == 'none' else '+-0.8'} px)")
    a.set_xlim(0, 1920)
    a.set_ylim(1080, 0)
    a.set_axis_off()
if "k1k2k3_c" in fits:
    c = fits["k1k2k3_c"]["dist"].c
    ax[1].plot(*c, "m+", ms=16, mew=2)
    ax[1].plot(*L.C0, "gx", ms=10, mew=2)
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_plumb_edges.png", dpi=110)
plt.close()

# (b) displacement curves: models and regions
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for nm, col in zip(["k1", "k1k2", "k1k2k3", "k1k2_c", "k1k2k3_c", "div2", "div2_c", "k1k2_t"], plt.cm.tab10.colors):
    cv = curves[nm]
    ax[0].plot(rr, cv["best"], color=col, label=nm)
    ax[0].fill_between(rr, cv["lo"], cv["hi"], color=col, alpha=0.15)
ax[0].set_xlabel("distorted radius from the model's centre [px]")
ax[0].set_ylabel("undistortion displacement [px] (outward +)")
ax[0].set_title("plumb-line models (bands: bootstrap 16-84 %)")
ax[0].axvline(np.hypot(960, 540), color="k", ls=":", lw=0.8)
ax[0].legend(fontsize=8)
ref = curves["k1k2"]["best"]
for nm, col in zip(["k1", "k1k2k3", "k1k2_c", "k1k2k3_c", "div2", "div2_c", "k1k2_t"], plt.cm.tab10.colors):
    ax[1].plot(rr, curves[nm]["best"] - ref, color=col, label=nm)
ax[1].fill_between(rr, curves["k1k2"]["lo"] - ref, curves["k1k2"]["hi"] - ref, color="gray", alpha=0.3, label="k1k2 boot band")
ax[1].set_title("difference to k1k2 (centre fixed)")
ax[1].set_xlabel("distorted radius [px]")
ax[1].set_ylabel("px")
ax[1].set_ylim(-8, 8)
ax[1].legend(fontsize=8)
for reg, col in zip(["scene", "cart310", "cart80"], ["tab:green", "tab:red", "tab:blue"]):
    for nm, ls in [("k1k2", "-"), ("k1k2_c", "--")]:
        cv = curves[f"{reg}_{nm}"]
        ax[2].plot(rr, cv["best"] - ref, ls, color=col, label=f"{reg} {nm}")
        if nm == "k1k2":
            ax[2].fill_between(rr, cv["lo"] - ref, cv["hi"] - ref, color=col, alpha=0.12)
ax[2].fill_between(rr, curves["k1k2"]["lo"] - ref, curves["k1k2"]["hi"] - ref, color="gray", alpha=0.3, label="all, boot band")
ax[2].set_title("per-region fits minus all-edge k1k2")
ax[2].set_xlabel("distorted radius [px]")
ax[2].set_ylim(-10, 10)
ax[2].legend(fontsize=7, ncol=2)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_plumb_models.png", dpi=110)
plt.close()

# (c) centre profile + bootstrap centres
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
for a, nm in zip(ax, ["k1k2", "k1k2k3"]):
    Z = prof[nm]["cost"]
    a.imshow(img, alpha=0.4, extent=(0, 1920, 1080, 0))
    cs = a.contour(gx, gy, Z - Z.min(), levels=[0.5, 2, 4.5, 8, 18, 32, 50, 100], colors="k", linewidths=0.8)
    a.clabel(cs, fontsize=7, fmt="%.1f")
    bnm = nm + "_c"
    if bnm in boot:
        xb = boot[bnm]
        a.plot(xb[:, 0], xb[:, 1], ".", color="m", ms=2, alpha=0.5, label=f"bootstrap centres ({bnm})")
    for reg, col in zip(["scene", "cart310", "cart80"], ["tab:green", "tab:red", "tab:blue"]):
        k = f"{reg}_k1k2_c"
        if nm == "k1k2" and k in regions:
            a.plot(*regions[k]["x"][:2], "o", color=col, ms=8, label=f"{reg} only (k1k2_c)")
    a.plot(*L.C0, "gx", ms=12, mew=2, label="image centre")
    a.set_xlim(0, 1920)
    a.set_ylim(1080, 0)
    a.set_title(f"cost - min (Huber cost units ~ px^2/2) vs fixed distortion centre, model {nm}")
    a.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_plumb_centre.png", dpi=110)
plt.close()

# (d) crops of the rejected edges with residuals (after the main model) exaggerated x20
d_main = fits["k1k2k3_c"]["dist"]
n = len(rej)
if n:
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for a, e in zip(ax.ravel(), rej):
        P = e["points"]
        x0_, y0_ = np.floor(P.min(0) - 25).astype(int)
        x1_, y1_ = np.ceil(P.max(0) + 25).astype(int)
        x0_, y0_ = max(x0_, 0), max(y0_, 0)
        x1_, y1_ = min(x1_, 1920), min(y1_, 1080)
        a.imshow(img[y0_:y1_, x0_:x1_], extent=(x0_, x1_, y1_, y0_))
        es1 = L.EdgeSet([e])
        U, JU, _ = d_main.undistort(es1.P)
        r, Ls = es1.residuals(U, JU, return_lines=True)
        # direction of the residual in the distorted image ~ normal of the chord
        ch = P[-1] - P[0]
        nrm = np.array([-ch[1], ch[0]]) / np.hypot(*ch)
        a.plot(P[:, 0], P[:, 1], "c.", ms=2)
        Q = P + 20 * r[:, None] * nrm[None]
        a.plot(Q[:, 0], Q[:, 1], "r-", lw=1.2)
        st = per_edge["k1k2k3_c"][e["id"]]
        a.set_title(f"{e['id']}\nrms {st['rms']:.2f} px, bow {st['bow']:+.2f} px (red: residual x20)", fontsize=9)
        a.set_axis_off()
    for a in ax.ravel()[n:]:
        a.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/40_lines_indep_plumb_rejected.png", dpi=110)
    plt.close()

# ---------------------------------------------------------------- save
out = dict(
    edge_files_reviewed=status,
    n_edges_loaded=len(edges), duplicates_dropped=dropped,
    rejection=dict(model="k1k2k3 + free centre", fac=2.5, bow_floor_px=0.6, rms_floor_px=0.5, history=hist,
                   rejected=[e["id"] for e in rej]),
    accepted=[e["id"] for e in acc], n_clusters=len(cl_names), clusters=cl_names,
    models=table, regions=regions, per_edge=per_edge,
    centre_profile={nm: dict(gx=gx.tolist(), gy=gy.tolist(), cost=v["cost"].tolist(), a1=v["a1"].tolist()) for nm, v in prof.items()},
    curves_r=rr.tolist(), curves={k: {q: np.asarray(v).tolist() for q, v in c.items()} for k, c in curves.items()},
    scale_S=L.S, note="a_i are pixel-unit coefficients scaled by S=1000 px: k_i = a_i f^(2i) / S^(2i); t_i: p_i = t_i f / S",
)
L.save(L.jsonable(out), "40_lines_indep_plumb.json")
np.savez(f"{CACHE}/40_lines_indep_plumb_boot.npz", **{k: v for k, v in boot.items()})
print(f"done in {time.time() - T0:.0f}s")
