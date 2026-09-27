"""Review additions to the independent line pipeline (40_lines_indep_*): systematic budget of the line methods.

Findings that motivated this script (independent review):
  * The distortion centre / principal point height is NOT determined to the bootstrap +-22..27 px: it moves with
    the radial model (k1k2 438, k1k2k3 456-467, div2 468-479) and with an extra, plausible straight-line
    constraint the pipeline did not use: the blue floor line is visible in TWO pieces (left of cart 310,
    x 172-434, and between the carts, x 872-1148; one painted line running under cart 310). With the main
    lens (centre (929, 426)) the right piece lies ~9 px off the line through the left piece and is ~1 deg
    rotated; with the two pieces merged into single straight edges the free centre moves to y ~510 and the
    other 54 edges do not object (k1k2k3: identical rms of the other edges).
  * f is robust to all of this (1473..1487), the mapping differences are 5 px (centre) .. 10 px (corners).
Hence: the headline uncertainties of method_plumbline / method_vanishing / method_lines_joint get a systematic
component (half-range over the CREDIBLE variants: radial model k1k2 / k1k2k3 / division-2, floor-line pieces
separate / merged, straightness weighting) added in quadrature to the statistical (bootstrap) part.
Variants NOT in the budget (reported only): pp fixed at the image centre (not data-driven), point-level VP
residuals (ignores the measured member direction error), separate cart frames (= dropping cart 80's X axis,
a subset, not an alternative model), scene verticals tied to the posts (rejected by the data, 0.8 +- 0.2 deg).

Also: stratified leave-one-VP-member-out jackknife of the joint f/pp (real data) and a cart split.
Writes work/cache/40_lines_indep_review.json and results/40_lines_indep_review_floorline.png
Run after 40_lines_indep_a/b/c (reads their caches); 40_lines_indep_f_methods.py then uses the output.
"""
import importlib
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS, K_from, load_json  # noqa: E402
from evaltools import grid, mapping_displacement, region_defs  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(4010)
T0 = time.time()
A = load_json(f"{CACHE}/40_lines_indep_plumb.json")
B = load_json(f"{CACHE}/40_lines_indep_vp.json")
C = load_json(f"{CACHE}/40_lines_indep_joint.json")
uv, _ = grid(40)
REG = region_defs()

edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
for e in acc:
    e["vgroup"] = "CZ" if e["group"] in ("310Z", "80Z") else e["group"]
byid = {e["id"]: e for e in acc}
print("edge files reviewed:", L.file_status(), "| accepted", len(acc))

# the two visible pieces of the blue floor line: (left piece, piece between the carts)
FLOOR_PAIRS = [("scene_blueL_top", "scene_blueH_top", "floorline_top"),        # top edge of the blue paint
               ("scene_blueL_bottom", "scene_tapeH_top", "floorline_bottom")]  # bottom edge of the paint (white tape starts)
FL_IDS = {a for a, b, _ in FLOOR_PAIRS} | {b for a, b, _ in FLOOR_PAIRS}


def merged(el, pairs=FLOOR_PAIRS):
    ids = {a for a, b, _ in pairs} | {b for a, b, _ in pairs}
    out = [e for e in el if e["id"] not in ids]
    for a, b, nm in pairs:
        out.append(dict(byid[a], id=nm, points=np.vstack([byid[a]["points"], byid[b]["points"]]), cluster="floorline", group=None,
                        vgroup=None))
    return out


def opencv_of(f, pp, dist):
    """OpenCV (K, d) of a pixel-unit model; division models via a least-squares Brown k1,k2,k3 fit."""
    if dist.kind == "poly":
        return K_from(f, f, *pp), dist.opencv(f)
    K, d, _, _ = L.fit_opencv_to_model(f, np.asarray(pp, float), dist, nk=3)
    return K, d


def map_diff(ref, other):
    disp = mapping_displacement(ref[0], ref[1], other[0], other[1], uv=uv, compensate=True)
    m = np.hypot(disp[:, 0], disp[:, 1])
    return {k: float(np.median(m[fn(uv)])) for k, fn in REG.items()}


# ------------------------------------------------------------------------------------------------ 1. collinearity test
def collinearity(dist):
    out = {}
    for a, b, nm in FLOOR_PAIRS:
        Ua, _, _ = dist.undistort(byid[a]["points"])
        Ub, JUb, _ = dist.undistort(byid[b]["points"])
        c = Ua.mean(0)
        _, _, vt = np.linalg.svd(Ua - c)
        t = vt[0]
        n = np.array([-t[1], t[0]])
        db = (Ub - c) @ n
        g = np.hypot(JUb[0] * n[0] + JUb[2] * n[1], JUb[1] * n[0] + JUb[3] * n[1])
        cb = Ub.mean(0)
        _, _, vb = np.linalg.svd(Ub - cb)
        ang = float(np.degrees(np.arcsin(abs(t[0] * vb[0][1] - t[1] * vb[0][0]))))
        out[nm] = dict(offset_px=float(np.mean(db / g)), angle_deg=ang)
    return out


lens_tests = {}
for nm in ["k1k2", "k1k2_c", "k1k2k3_c", "div2", "div2_c"]:
    t = A["models"][nm]
    nc = 2 if "cx" in t["names"] else 0
    nr = len([q for q in t["names"] if q[0] in "al"])
    d = L.Dist("div" if nm.startswith("div") else "poly", t["x"][nc:nc + nr], None, np.array(t["centre"]))
    lens_tests[f"plumb:{nm}"] = dict(centre=t["centre"], **collinearity(d))
for nm, v in C["variants"].items():
    d = L.Dist(v["kind"], v["coeffs"], None, np.array(v["pp"]))
    lens_tests[f"joint:{nm}"] = dict(centre=v["pp"], **collinearity(d))
lens_tests["no correction"] = dict(centre=None, **collinearity(L.Dist("poly", [0.0], None, L.C0)))
print("\n[1] floor-line collinearity (offset of the piece between the carts from the line through the left piece, px; angle deg)")
for k, v in lens_tests.items():
    print(f"  {k:30s} " + "  ".join(f"{q}: {v[q]['offset_px']:+6.2f} px {v[q]['angle_deg']:.2f} deg" for q in ("floorline_top", "floorline_bottom")))

# ------------------------------------------------------------------------------------------------ 2. plumb variants
print("\n[2] plumb-line variants with the floor-line pieces merged")
acc_m = merged(acc)
acc_mt = merged(acc, FLOOR_PAIRS[:1])
PM = {"k1k2_c": L.PlumbModel("poly", 2, centre_free=True), "k1k2k3_c": L.PlumbModel("poly", 3, centre_free=True),
      "div2_c": L.PlumbModel("div", 2, centre_free=True)}
plumb_var = {}
others_mask = lambda el: np.array([e["id"] not in FL_IDS and not e["id"].startswith("floorline") for e in el])  # noqa: E731
for tag, el in [("sep", acc), ("merged", acc_m), ("merged_toponly", acc_mt)]:
    for mn, m in PM.items():
        if tag == "merged_toponly" and mn != "k1k2_c":
            continue
        f = L.fit_plumb(L.EdgeSet(el), m)
        es = L.EdgeSet(el)
        om = others_mask(el)[es.k]
        st = es.per_edge_stats(f["r"])
        fl = {e["id"]: dict(rms=s["rms"], bow=s["bow"]) for e, s in zip(el, st) if e["id"].startswith("floorline")}
        plumb_var[f"{mn}|{tag}"] = dict(x=f["x"].tolist(), names=f["names"], centre=f["dist"].c.tolist(), coeffs=f["dist"].a.tolist(),
                                        kind=f["dist"].kind, rms=f["rms"], rms_other_edges=float(np.sqrt(np.mean(f["r"][om] ** 2))),
                                        floorline=fl, dist=f["dist"])
        print(f"  {mn:9s} {tag:15s} centre ({f['dist'].c[0]:.1f},{f['dist'].c[1]:.1f}) coeffs {np.round(f['dist'].a, 4)} "
              f"rms other edges {plumb_var[f'{mn}|{tag}']['rms_other_edges']:.4f} floor line {({k: round(v['rms'], 3) for k, v in fl.items()})}")

# ------------------------------------------------------------------------------------------------ 3. VP f for the plumb variants
print("\n[3] VP focal length (cart X _|_ common post vertical, pp = distortion centre) for the plumb variants")
PAIRS = [("310X", "CZ"), ("80X", "CZ")]
pl0 = plumb_var["k1k2_c|sep"]["dist"]
vps0 = L.vp_estimates(acc, pl0, ["310X", "80X", "CZ"])
f00 = L.f_orthogonal(vps0, PAIRS, pl0.c)
rates0 = L.edge_angle_rates(acc, pl0, f00, pl0.c)
COVS = {g: L.vp_cov_mc(acc, pl0, g, vps0[g], rates0, np.radians(B["sigma_psi_deg"]), n=100, rng_=rng) for g in ("310X", "80X", "CZ")}
vp_var = {}
for k, v in plumb_var.items():
    el = acc if k.endswith("|sep") else (acc_m if k.endswith("|merged") else acc_mt)
    vps = L.vp_estimates(el, v["dist"], ["310X", "80X", "CZ"])
    fv = L.gls_f(vps, COVS, PAIRS, v["dist"].c, f0=f00)
    vp_var[k] = dict(f=fv, pp=v["dist"].c.tolist())
    print(f"  plumb {k:24s} -> f {fv:7.1f} pp ({v['dist'].c[0]:.1f},{v['dist'].c[1]:.1f})")
# the GLS weights here come from a new Monte-Carlo covariance -> align to the method's main value (40_lines_indep_b_vp.py):
# only the differences between the variants are used
f_vp_main = B["plumb_variants"]["k1k2_c"]["solutions"]["S1_cartX_cartZ|ppDist"]["f"]
dF = f_vp_main - vp_var["k1k2_c|sep"]["f"]
for k in vp_var:
    vp_var[k]["f_aligned"] = vp_var[k]["f"] + dF
print(f"  (aligned to the main VP value {f_vp_main:.1f}: shift {dF:+.1f} px)")

# ------------------------------------------------------------------------------------------------ 4. joint variants
print("\n[4] joint variants with the floor-line pieces merged")
pl = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
se_cl = np.sqrt(np.diag(pl["cov_cluster"]))[2:]
se_c = np.sqrt(np.diag(pl["cov_classic"]))[2:]
SIG_PT = pl["rms"] * float(np.median(se_cl / se_c))
SIG_PSI = C["variants"][C["main"]]["sig_psi_deg"]


def joint_fit(el, kind="poly", nrad=2, use=("310X", "80X", "310Z", "80Z", "WV"), x0=None):
    p = L.fit_plumb(L.EdgeSet(el), L.PlumbModel("poly", 2, centre_free=True))
    vps = L.vp_estimates(el, p["dist"], ["310X", "80X", "CZ", "WV"])
    fi = L.f_orthogonal(vps, PAIRS, p["dist"].c)
    a0 = list(p["dist"].a) + ([0.0] if nrad == 3 else []) if kind == "poly" else [-0.17, -0.03]
    init = dict(f=fi, pp=tuple(p["dist"].c), a=a0, vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in vps.items()})
    m = L.JointModel(el, kind=kind, nrad=nrad, init=init, sig_pt=SIG_PT, sig_psi_deg=SIG_PSI, use_groups=use)
    r = m.fit(x0=x0, loss="linear")
    f, pp, dist, dirs = m.unpack(r.x)
    return dict(f=float(f), pp=[float(pp[0]), float(pp[1])], coeffs=dist.a.tolist(), kind=kind, cost=float(2 * r.cost), dist=dist,
                x=r.x, model=m, init=init)


joint_var = {}
for mn, kind, nrad in [("JRE_k1k2", "poly", 2), ("JRE_k1k2k3", "poly", 3), ("JRE_div2", "div", 2)]:
    j = joint_fit(acc_m, kind, nrad)
    joint_var[f"{mn}|merged"] = j
    print(f"  {mn:11s} merged: f {j['f']:.1f} pp ({j['pp'][0]:.1f},{j['pp'][1]:.1f}) coeffs {np.round(j['coeffs'], 4)}")
j = joint_fit(acc_mt)
joint_var["JRE_k1k2|merged_toponly"] = j
print(f"  JRE_k1k2 merged (top edge only): f {j['f']:.1f} pp ({j['pp'][0]:.1f},{j['pp'][1]:.1f})")

# small bootstrap of the merged main variant (same stratified scheme as 40_lines_indep_c_joint.py)
rates = L.edge_angle_rates(acc, pl["dist"], C["variants"][C["main"]]["f"], pl["dist"].c)
jm0 = joint_var["JRE_k1k2|merged"]
bm = []
for b in range(NB):
    sub = L.stratified_resample(acc_m, rng)
    sub = L.perturb_directions(sub, rates, np.radians(SIG_PSI), rng, groups={"CZ", "WV"})
    try:
        m = L.JointModel(sub, init=jm0["init"], sig_pt=SIG_PT, sig_psi_deg=SIG_PSI)  # same direction bases as jm0 -> x0 valid
        r = m.fit(x0=jm0["x"], loss="linear", max_nfev=150)
        f, pp, dist, _ = m.unpack(r.x)
        bm.append([f, pp[0], pp[1]] + list(dist.a))
    except Exception as ex:  # noqa
        print("   replicate failed", ex)
bm = np.array(bm)
merged_boot = dict(n=len(bm), sd=bm.std(0, ddof=1).tolist(), median=np.median(bm, 0).tolist(), columns=["f", "cx", "cy", "a1", "a2"])
print(f"  bootstrap merged JRE_k1k2 ({len(bm)}): sd f {bm[:, 0].std(ddof=1):.1f} cx {bm[:, 1].std(ddof=1):.1f} cy {bm[:, 2].std(ddof=1):.1f}  ({time.time() - T0:.0f}s)")

# ------------------------------------------------------------------------------------------------ 5. jackknife + cart split (joint main)
print("\n[5] stratified leave-one-VP-member-out jackknife of the joint main fit; cart split")
jmain = C["variants"][C["main"]]
strata = {}
for e in acc:
    g = {"310Z": "CZ", "80Z": "CZ"}.get(e["group"], e["group"])
    if g in ("310X", "80X", "CZ"):
        strata.setdefault(g, set()).add(e["cluster"])
jk = {}
var = np.zeros(3)
for g, cls in strata.items():
    vals = []
    for c in sorted(cls):
        r = joint_fit([e for e in acc if e["cluster"] != c])
        vals.append([r["f"], r["pp"][0], r["pp"][1]])
        jk[c] = vals[-1]
    vals = np.array(vals)
    var += (len(vals) - 1) / len(vals) * np.sum((vals - vals.mean(0)) ** 2, 0)
jk_sd = np.sqrt(var).tolist()
print(f"  jackknife sd f {jk_sd[0]:.1f} cx {jk_sd[1]:.1f} cy {jk_sd[2]:.1f} (bootstrap: {jmain['boot']['sd'][:3]})")
split = {}
for nm, use in [("cart310_X_only", ("310X", "310Z", "80Z", "WV")), ("cart80_X_only", ("80X", "310Z", "80Z", "WV"))]:
    r = joint_fit(acc, use=use)
    split[nm] = dict(f=r["f"], pp=r["pp"])
    print(f"  {nm}: f {r['f']:.1f} pp ({r['pp'][0]:.1f},{r['pp'][1]:.1f})")

# ------------------------------------------------------------------------------------------------ 6. systematic budget
print("\n[6] systematic budget (half-range over credible variants, incl. the main)")


def budget(entries, main_key):
    """entries: name -> dict(f, pp, ocv=(K,d), k=[k1,k2] or None). Half-range of f, cx, cy (+ k1,k2 over poly-k1k2 entries);
    mapping: half of the largest median |displacement| vs the main per region."""
    fs = np.array([v["f"] for v in entries.values()])
    cx = np.array([v["pp"][0] for v in entries.values()])
    cy = np.array([v["pp"][1] for v in entries.values()])
    ks = np.array([v["k"] for v in entries.values() if v.get("k") is not None])
    ref = entries[main_key]["ocv"]
    md = {n: map_diff(ref, v["ocv"]) for n, v in entries.items()}
    out = dict(members=list(entries), values={n: dict(f=v["f"], pp=v["pp"], k=v.get("k"), mapping_vs_main=md[n]) for n, v in entries.items()},
               f_sys=float(np.ptp(fs) / 2), cx_sys=float(np.ptp(cx) / 2), cy_sys=float(np.ptp(cy) / 2),
               k1_sys=float(np.ptp(ks[:, 0]) / 2) if len(ks) > 1 else 0.0, k2_sys=float(np.ptp(ks[:, 1]) / 2) if len(ks) > 1 else 0.0,
               mapping_sys={r: float(max(m[r] for m in md.values()) / 2) for r in REG},
               rule="systematic 1-sigma = half-range of the central values over the credible variants (main included); mapping: half of "
                    "the largest median rotation-compensated displacement vs the main in the region")
    print(f"  f_sys {out['f_sys']:.1f}  cx_sys {out['cx_sys']:.1f}  cy_sys {out['cy_sys']:.1f}  k1_sys {out['k1_sys']:.4f}  k2_sys {out['k2_sys']:.4f}  "
          f"mapping_sys {({k: round(v, 2) for k, v in out['mapping_sys'].items()})}")
    return out


# plumb-line method (OpenCV k's at the VP focal length; f fixed -> distortion-only mapping)
ent = {}
for k, v in plumb_var.items():
    if k.endswith("toponly"):
        continue
    ocv = opencv_of(f_vp_main, v["dist"].c, v["dist"])
    ent[k] = dict(f=f_vp_main, pp=v["centre"], ocv=ocv, k=(ocv[1][:2].tolist() if k.startswith("k1k2_c") else None))
print(" plumbline:")
bud_plumb = budget(ent, "k1k2_c|sep")
# vanishing-point method (pp = distortion centre of each plumb variant, f from the VPs)
ent = {}
for k, v in plumb_var.items():
    if k.endswith("toponly"):
        continue
    fv = vp_var[k]["f_aligned"]
    ocv = opencv_of(fv, v["dist"].c, v["dist"])
    ent[k] = dict(f=fv, pp=v["centre"], ocv=ocv, k=(ocv[1][:2].tolist() if k.startswith("k1k2_c") else None))
print(" vanishing:")
bud_vp = budget(ent, "k1k2_c|sep")
# joint method
ent = {}
for n in [C["main"], "JRE_k1k2k3_ppfree", "JRE_div2_ppfree", "JRE_k1k2_sigpt_noinfl", "JRE_k1k2_sigpt_2xinfl"]:
    v = C["variants"][n]
    d = L.Dist(v["kind"], v["coeffs"], None, np.array(v["pp"]))
    ocv = opencv_of(v["f"], v["pp"], d)
    ent[n] = dict(f=v["f"], pp=v["pp"], ocv=ocv, k=(ocv[1][:2].tolist() if (v["kind"] == "poly" and len(v["coeffs"]) == 2) else None))
for n, v in joint_var.items():
    if n.endswith("toponly"):
        continue
    ocv = opencv_of(v["f"], v["pp"], v["dist"])
    ent[n] = dict(f=v["f"], pp=v["pp"], ocv=ocv, k=(ocv[1][:2].tolist() if (v["kind"] == "poly" and len(v["coeffs"]) == 2) else None))
print(" lines_joint:")
bud_joint = budget(ent, C["main"])
# not in the budget, reported
excl = {}
for n in ["JRE_k1k2_ppC0", "JRE_k1k2_framesfree", "JRE_k1k2_wvtied", "JPT_k1k2_ppfree", "JPT_k1k2_framesfree"]:
    v = C["variants"][n]
    d = L.Dist(v["kind"], v["coeffs"], None, np.array(v["pp"]))
    excl[n] = dict(f=v["f"], pp=v["pp"], mapping_vs_main=map_diff(ent[C["main"]]["ocv"], opencv_of(v["f"], v["pp"], d)))
    print(f"  (not in budget) {n:22s} f {v['f']:.1f} pp ({v['pp'][0]:.0f},{v['pp'][1]:.0f}) mapping vs main {({k: round(q, 1) for k, q in excl[n]['mapping_vs_main'].items()})}")

# ------------------------------------------------------------------------------------------------ 7. plot
img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
ax = fig.add_subplot(gs[0, :])
x0_, x1_, y0_, y1_ = 150, 1200, 590, 720
ax.imshow(img[y0_:y1_, x0_:x1_], extent=(x0_, x1_, y1_, y0_))
for lab, dist, col in [("main lens (plumb k1k2, centre (929,426))", plumb_var["k1k2_c|sep"]["dist"], "red"),
                       ("floor-line pieces merged (k1k2, centre (939,508))", plumb_var["k1k2_c|merged"]["dist"], "lime"),
                       ("centre fixed at the image centre (k1k2)", L.Dist("poly", A["models"]["k1k2"]["x"], None, L.C0), "cyan")]:
    for a, b, _ in FLOOR_PAIRS[:1]:
        Ua, _, _ = dist.undistort(byid[a]["points"])
        c = Ua.mean(0)
        _, _, vt = np.linalg.svd(Ua - c)
        t = vt[0] * np.sign(vt[0][0])
        s = np.linspace(-150, 1000, 400)
        Pd = dist.distort_px(c[None] + s[:, None] * t[None])
        ax.plot(Pd[:, 0], Pd[:, 1], "--", color=col, lw=1.2, label=f"straight 3D line through the left piece, {lab}")
for a, b, _ in FLOOR_PAIRS:
    for q in (a, b):
        P = byid[q]["points"]
        ax.plot(P[:, 0], P[:, 1], ".", color="yellow", ms=2)
ax.set_xlim(x0_, x1_)
ax.set_ylim(y1_, y0_)
ax.set_title("blue floor line: two visible pieces (yellow = traced edges); dashed = the left piece's straight line extended with each lens "
             "(top edge of the paint)", fontsize=9)
ax.legend(fontsize=8, loc="lower left")
ax2 = fig.add_subplot(gs[1, 0])
names, ys, es_, cols = [], [], [], []
for k, v in plumb_var.items():
    names.append("plumb " + k)
    ys.append(v["centre"][1])
    es_.append(A["models"][k.split("|")[0]]["boot_sd"][1] if (k.endswith("|sep") and "boot_sd" in A["models"][k.split("|")[0]]) else np.nan)
    cols.append("tab:blue" if k.endswith("|sep") else "tab:green")
for n in [C["main"], "JRE_k1k2k3_ppfree", "JRE_div2_ppfree", "JPT_k1k2_ppfree", "JRE_k1k2_ppC0"]:
    v = C["variants"][n]
    names.append("joint " + n)
    ys.append(v["pp"][1])
    es_.append(v["boot"]["sd"][2] if "boot" in v else np.nan)
    cols.append("tab:red" if n != "JPT_k1k2_ppfree" else "gray")
for n, v in joint_var.items():
    names.append("joint " + n)
    ys.append(v["pp"][1])
    es_.append(merged_boot["sd"][2] if n == "JRE_k1k2|merged" else np.nan)
    cols.append("tab:olive")
for i, (y, e, c) in enumerate(zip(ys, es_, cols)):
    ax2.errorbar(y, i, xerr=e if np.isfinite(e) else None, fmt="o", color=c, capsize=3)
ax2.axvline(L.C0[1], color="k", ls=":", lw=1)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=7)
ax2.set_xlabel("distortion centre / pp y [px] (bars: bootstrap sd where available)")
ax2.set_title("pp_y by model and floor-line assumption", fontsize=9)
ax2.grid(alpha=0.3)
ax3 = fig.add_subplot(gs[1, 1])
names, fs_, cols = [], [], []
for k, v in vp_var.items():
    names.append("VP, plumb " + k)
    fs_.append(v["f_aligned"])
    cols.append("tab:blue" if k.endswith("|sep") else "tab:green")
for n in [C["main"], "JRE_k1k2k3_ppfree", "JRE_div2_ppfree", "JRE_k1k2_framesfree", "JRE_k1k2_wvtied", "JPT_k1k2_ppfree"]:
    names.append("joint " + n)
    fs_.append(C["variants"][n]["f"])
    cols.append("tab:red" if n in (C["main"], "JRE_k1k2k3_ppfree", "JRE_div2_ppfree") else "gray")
for n, v in joint_var.items():
    names.append("joint " + n)
    fs_.append(v["f"])
    cols.append("tab:olive")
for n, v in split.items():
    names.append("joint " + n)
    fs_.append(v["f"])
    cols.append("gray")
fm = C["variants"][C["main"]]
ax3.axvspan(fm["f"] - fm["boot"]["sd"][0], fm["f"] + fm["boot"]["sd"][0], color="tab:red", alpha=0.12, label="joint main +- bootstrap sd")
tot = float(np.hypot(fm["boot"]["sd"][0], bud_joint["f_sys"]))
ax3.axvline(fm["f"] - tot, color="tab:red", ls=":", lw=1)
ax3.axvline(fm["f"] + tot, color="tab:red", ls=":", lw=1, label="+- total (stat + sys)")
ax3.plot(fs_, range(len(fs_)), "o", ms=5, color="k")
for i, c in enumerate(cols):
    ax3.plot(fs_[i], i, "o", color=c)
ax3.set_yticks(range(len(names)))
ax3.set_yticklabels(names, fontsize=7)
ax3.set_xlabel("f [px]")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)
ax3.set_title("focal length: variants (gray = not in the budget)", fontsize=9)
ax4 = fig.add_subplot(gs[1, 2])
regs = list(REG)
stat = [C["variants"][C["main"]]["mapping_uncertainty"][r]["rms_median_px"] for r in regs]
sysv = [bud_joint["mapping_sys"][r] for r in regs]
ax4.bar(np.arange(3) - 0.27, stat, 0.27, label="joint: statistical (bootstrap)")
ax4.bar(np.arange(3), sysv, 0.27, label="joint: systematic (credible variants)")
ax4.bar(np.arange(3) + 0.27, np.hypot(stat, sysv), 0.27, label="joint: total")
ax4.set_xticks(range(3))
ax4.set_xticklabels(regs)
ax4.set_ylabel("mapping uncertainty [px] (rotation-compensated)")
ax4.legend(fontsize=7)
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_review_floorline.png", dpi=110)
plt.close()


def strip(d):
    return {k: v for k, v in d.items() if k not in ("dist", "model", "x", "init")}


out = dict(
    note="review additions: floor-line collinearity, variants with the two blue floor-line pieces merged, jackknife, cart split, "
         "systematic budget per method (see module docstring)",
    floor_pairs=FLOOR_PAIRS, collinearity_tests=lens_tests,
    plumb_variants={k: strip(v) for k, v in plumb_var.items()}, vp_variants=vp_var,
    joint_variants={k: strip(v) for k, v in joint_var.items()}, joint_merged_bootstrap=merged_boot,
    jackknife=dict(sd_f_cx_cy=jk_sd, per_member=jk, bootstrap_sd=jmain["boot"]["sd"][:3]), cart_split=split,
    budget=dict(plumbline=bud_plumb, vanishing=bud_vp, lines_joint=bud_joint), not_in_budget_joint=excl,
)
L.save(L.jsonable(out), "40_lines_indep_review.json")
print(f"done in {time.time() - T0:.0f}s")
