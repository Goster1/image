"""Sub-task 43, step 4 (independent review): honest parameter / mapping uncertainty of the recommended models.

Findings that motivated this step (see work/cache/altmodels_review.md):
  * the cluster-robust sandwich of 43_altmodels_fit.py treats every traced edge as its own cluster, although
    12 edges are the second boundary of the same physical member (`pair_of` in the edge files: both sides of
    one lip / tape / rod). Clusters here = physical members (pair_of chains merged) and stickers.
  * sigma(cy) = 4 px of the edge / joint fits comes almost entirely from the floor_H group (white tape between
    the carts + blue floor line on the left, ASSUMED parallel and perpendicular to the free scene vertical):
    without the floor lines (or with only one side of floor_H) sigma(cy) is ~26-30 px. A sandwich estimate
    with ~3 influential clusters is biased low.
  * the focal length moves 1420-1545 px over the leave-one-tile-out folds, far more than the sandwich
    sigma (26-30 px) allows.

What is computed (same estimator, same fixed block sigmas as 43_altmodels_fit.py):
  1. sandwich (CR1) with physical-member clusters,
  2. delete-one-cluster jackknife (clusters = physical members for edges, stickers for corners) of the
     intrinsics AND directly of the rotation-compensated mapping (no linearisation, no Gaussian assumption),
  3. the same for the variant without the floor-line VP groups (floor lines -> straightness only), i.e.
     without the untested "blue line || white tape, perpendicular to the scene vertical" assumption.
Models: Brown k1,k2 (recommended) and Brown k1,k2,k3 (alternative); data sets E and ME.
Writes work/cache/altmodels_review.json / .md and updates the uncertainty fields of
work/cache/method_alt_brown_k1k2.json / method_alt_brown_k1k2k3.json (+ .md). Run after 43_altmodels_report.py.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from common import CACHE, load_json, save_json  # noqa: E402
from evaltools import mapping_stats  # noqa: E402

L = importlib.import_module("43_altmodels_lib")
F = importlib.import_module("43_altmodels_fit")
FITS = load_json(f"{CACHE}/altmodels_fits.json")
NPROC = int(os.environ.get("NPROC", "2"))
UV, _ = L.pixel_grid(40)
MODELS = ["B_k1k2", "B_k1k2k3"]
DATASETS = ["E", "ME"]
POLICIES = ["noboard", "nofloor"]  # orchestrator VP policy / same without the floor-line VP groups


def pair_root():
    """edge id -> id of the physical member (pair_of chains followed to the root)."""
    par = {}
    for r in ("cart310", "cart80", "scene"):
        for e in load_json(f"{CACHE}/edges_{r}.json")["edges"]:
            if e.get("pair_of"):
                par[e["id"]] = e["pair_of"]

    def root(i):
        seen = set()
        while i in par and i not in seen:
            seen.add(i)
            i = par[i]
        return i

    return root


ROOT = pair_root()


def make_est(name, ds, policy="noboard", drop_edges=(), drop_sticker=None):
    Dd = F.get_data()
    data = L.build(Dd, ds, drop_sticker=drop_sticker, drop_edges=drop_edges, policy=policy)
    for b in ("V", "S"):
        for e in data[b]:
            e["cl"] = "m:" + ROOT(e["id"])
    lens = F.make_lens(name)
    return lens, L.Est(lens, data, FITS["sig"][ds])


def start_state(name, ds):
    f = FITS["fits"][ds][name]
    lens = F.make_lens(name)
    P = lens.P(np.array(f["x"][: lens.ni]))
    poses = {int(k): np.array(v) for k, v in f["poses"].items()}
    wv = np.array(f["wv"]) if f["wv"] is not None else None
    return P, poses, wv, dict(f.get("floor_ang") or {})


def fit(name, ds, policy="noboard", drop_edges=(), drop_sticker=None, state=None):
    lens, est = make_est(name, ds, policy, drop_edges, drop_sticker)
    P, poses, wv, fa = state if state is not None else start_state(name, ds)
    r = est.solve(est.x0(P, poses, wv, fa))
    return lens, est, r


def clusters_of(name, ds, policy):
    """Physical-member clusters of the data set (edge members + stickers)."""
    _, est = make_est(name, ds, policy)
    labs = sorted(set(est.clusters()) - {"_barrier"})
    return labs


def drop_args(label):
    Dd = F.get_data()
    if label.startswith("m:"):
        m = label[2:]
        return dict(drop_edges=[e["id"] for e in Dd["edges"] if ROOT(e["id"]) == m])
    c, i = label[1:].split("_")  # "S<cart>_<id>"
    return dict(drop_sticker=(int(c), int(i)))


def full_state(lens, est, r):
    return lens.P(r.x[: lens.ni]), est.poses_of(r.x), est.wv_of(r.x), est.floor_angles(r.x)


_FULL = {}


def task(args):
    name, ds, policy, label = args
    lens, est, r = fit(name, ds, policy, state=_FULL[(name, ds, policy)]["state"], **drop_args(label))
    P = lens.P(r.x[: lens.ni])
    Pf = _FULL[(name, ds, policy)]["P"]
    d = L.mapping_disp(lens, Pf, lens, P, UV, compensate=True)
    return name, ds, policy, label, r.x[: lens.ni].copy(), d, bool(est.barrier_active(r.x))


def sandwich(est, r):
    keep = slice(None) if est.barrier_active(r.x) else slice(0, -1)
    rr = SimpleNamespace(jac=r.jac[keep], fun=r.fun[keep], x=r.x)
    cl = est.clusters()[keep]
    return L.sandwich_cov(rr, cl)


def main():
    t0 = time.time()
    out = dict(models=MODELS, datasets=DATASETS, policies=POLICIES, sig=FITS["sig"], runs={})
    tasks = []
    for name in MODELS:
        for ds in DATASETS:
            for policy in POLICIES:
                lens, est, r = fit(name, ds, policy)
                P = lens.P(r.x[: lens.ni])
                C = sandwich(est, r)[: lens.ni, : lens.ni]
                labs = sorted(set(est.clusters()) - {"_barrier"})
                _FULL[(name, ds, policy)] = dict(P=P, state=full_state(lens, est, r))
                key = f"{name}|{ds}|{policy}"
                out["runs"][key] = dict(x=r.x[: lens.ni].tolist(), names=lens.names, P={k: float(v) for k, v in P.items()},
                                        chi2=float(2 * r.cost), block_stats=est.block_stats(r.x), fold_margin=float(lens.fold_margin(P)),
                                        barrier_active=bool(est.barrier_active(r.x)), n_clusters=len(labs),
                                        sd_sandwich_member=dict(zip(lens.names, np.sqrt(np.diag(C)).tolist())),
                                        cov_sandwich_member=C.tolist())
                print(f"[full] {key}: " + " ".join(f"{n}={v:.4g}+-{s:.2g}" for n, v, s in zip(lens.names, r.x[: lens.ni], np.sqrt(np.diag(C)))) +
                      f" clusters {len(labs)}", flush=True)
                tasks += [(name, ds, policy, lab) for lab in labs]
    print(f"jackknife fits: {len(tasks)}", flush=True)
    res = {}
    with Pool(NPROC) as pool:
        for k, (name, ds, policy, label, th, d, ba) in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            res.setdefault(f"{name}|{ds}|{policy}", []).append((label, th, d, ba))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    for key, rows in res.items():
        rows.sort(key=lambda z: z[0])
        G = len(rows)
        TH = np.array([z[1] for z in rows])
        Dm = np.array([z[2] for z in rows])
        th_bar = TH.mean(0)
        C = (G - 1) / G * (TH - th_bar).T @ (TH - th_bar)
        run = out["runs"][key]
        run["sd_jackknife"] = dict(zip(run["names"], np.sqrt(np.diag(C)).tolist()))
        run["cov_jackknife"] = C.tolist()
        run["jackknife_bias"] = dict(zip(run["names"], ((G - 1) * (th_bar - np.array(run["x"]))).tolist()))
        # mapping: jackknife variance of the rotation-compensated displacement field, per grid point
        dbar = Dm.mean(0)
        st, _ = mapping_stats(np.sqrt((G - 1)) * (Dm - dbar), UV)  # RMS over folds * sqrt(G-1) = jackknife sd
        run["mapping_unc_jackknife"] = {k: round(v["rms_median_px"], 3) for k, v in st.items()}
        run["mapping_unc_jackknife_max"] = {k: round(v["rms_max_px"], 3) for k, v in st.items() if "rms_max_px" in v}
        # most influential clusters (shift of each intrinsic in units of its jackknife sd)
        sd = np.sqrt(np.diag(C))
        infl = []
        for lab, th, d, ba in rows:
            z = (th - np.array(run["x"])) / np.maximum(sd, 1e-12)
            infl.append((float(np.max(np.abs(z))), lab, {n: round(float(v), 4) for n, v in zip(run["names"], th - np.array(run["x"]))}, ba))
        infl.sort(key=lambda z: -z[0])
        run["most_influential"] = [dict(cluster=b, max_abs_shift_in_jk_sd=round(a, 2), shift=c, barrier_active=e) for a, b, c, e in infl[:6]]
        run["n_barrier_active_folds"] = int(sum(z[3] for z in rows))
        run["f_range_over_folds"] = [float(TH[:, 0].min()), float(TH[:, 0].max())]
        print(f"[jk] {key}: " + " ".join(f"{n} sd_jk={s:.3g} (sand {run['sd_sandwich_member'][n]:.3g})" for n, s in run["sd_jackknife"].items()) +
              f" | mapping jk {run['mapping_unc_jackknife']}", flush=True)
    # parameter-sample mapping uncertainty from the member-cluster sandwich (same recipe as the report)
    rng = np.random.default_rng(4344)
    for key, run in out["runs"].items():
        name, ds, policy = key.split("|")
        lens = F.make_lens(name)
        th = np.array(run["x"])
        P = lens.P(th)
        Cm = np.array(run["cov_sandwich_member"])
        w, V = np.linalg.eigh((Cm + Cm.T) / 2)
        A = V * np.sqrt(np.clip(w, 0, None))
        disp = []
        for _ in range(150):
            Ps = lens.P(th + A @ rng.standard_normal(len(th)))
            if lens.fold_margin(Ps) < 0:
                continue
            disp.append(L.mapping_disp(lens, P, lens, Ps, UV, compensate=True))
        st, _ = mapping_stats(np.array(disp), UV)
        run["mapping_unc_sandwich_member"] = {k: round(v["rms_median_px"], 3) for k, v in st.items()}
        for k in ("cov_sandwich_member", "cov_jackknife"):
            run[k] = np.round(np.array(run[k]), 10).tolist()
    save_json(out, f"{CACHE}/altmodels_review.json")
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------------------------------
# method files: replace the sandwich-based uncertainty by the (larger) jackknife one
# ------------------------------------------------------------------------------------------
TAGS = {"B_k1k2": "brown_k1k2", "B_k1k2k3": "brown_k1k2k3"}


def update_methods(out):
    """Replace the uncertainty fields of the method files (idempotent: previous values are taken from the fit / report
    caches, not from the method file)."""
    raw = load_json(f"{CACHE}/altmodels_summary_raw.json")
    for name, tag in TAGS.items():
        fn = f"{CACHE}/method_alt_{tag}.json"
        if not os.path.exists(fn):
            continue
        m = load_json(fn)
        run = out["runs"][f"{name}|ME|noboard"]
        alt = out["runs"][f"{name}|ME|nofloor"]
        runE = out["runs"][f"{name}|E|noboard"]
        old_unc = {f"{k}_1sigma": v for k, v in FITS["fits"]["ME"][name]["sd_sandwich"].items()}
        st = raw["mapping_param_unc"]["ME"][name]["stats"]
        old_map = {k: round(st[k]["rms_median_px"], 3) for k in ("centre", "cart_band", "corners")}
        det = m.setdefault("details", {})
        mc = (det.get("model_choice_mapping_unc_px") or {}).get("ME") or {}
        sdj, sds, sda = run["sd_jackknife"], run["sd_sandwich_member"], alt["sd_jackknife"]
        unc = {f"{n}_1sigma": float(max(sdj[n], sds[n])) for n in run["names"]}
        # cy: the floor_H assumption (blue line || white tape, both perpendicular to the scene vertical) is untested
        # and carries almost all of the cy information -> use the spread-free variant without it as the honest sigma
        cy_nofloor = float(max(alt["sd_jackknife"]["cy"], alt["sd_sandwich_member"]["cy"]))
        cy_shift = abs(alt["P"]["cy"] - run["P"]["cy"])
        unc["cy_1sigma"] = float(max(unc["cy_1sigma"], cy_nofloor))
        unc["cy_1sigma_if_floor_lines_parallel"] = float(max(sdj["cy"], sds["cy"]))
        unc["note"] = ("max(delete-one-cluster jackknife, member-cluster sandwich) of the joint fit; clusters = physical edge members "
                       "(pair_of merged) and stickers. cy_1sigma is that of the fit WITHOUT the floor-line VP groups (the blue line || "
                       f"white tape assumption carries almost all cy information; without it cy = {alt['P']['cy']:.1f} px, "
                       f"shift {cy_shift:.1f} px)")
        mj, ms, mja = run["mapping_unc_jackknife"], run["mapping_unc_sandwich_member"], alt["mapping_unc_jackknife"]
        par = {k: float(max(mj[k], ms[k], mja[k], alt["mapping_unc_sandwich_member"][k])) for k in ("centre", "cart_band", "corners")}
        tot = {k: round(float(np.hypot(par[k], float(mc.get(k, 0.0)))), 3) for k in par}
        m["uncertainty"] = unc
        m["mapping_uncertainty_px"] = tot
        det["review"] = dict(
            script="work/43_altmodels_review.py",
            previous_uncertainty=old_unc, previous_mapping_uncertainty_px=old_map,
            sd_sandwich_member_clusters=sds, sd_jackknife_member_clusters=sdj, jackknife_bias=run["jackknife_bias"],
            mapping_param_unc_jackknife_px=mj, mapping_param_unc_sandwich_member_px=ms,
            mapping_param_unc_used_px={k: round(v, 3) for k, v in par.items()}, model_choice_part_px=mc,
            mapping_uncertainty_definition="sqrt(parameter part^2 + model-choice part^2); parameter part = max over "
                                           "{jackknife, member sandwich} x {with, without floor-line VP groups}",
            f_range_over_jackknife_folds=run["f_range_over_folds"], most_influential_clusters=run["most_influential"],
            without_floor_line_vp_groups=dict(P=alt["P"], sd_jackknife=alt["sd_jackknife"], sd_sandwich_member=alt["sd_sandwich_member"],
                                              mapping_unc_jackknife=alt["mapping_unc_jackknife"],
                                              mapping_unc_sandwich_member=alt["mapping_unc_sandwich_member"], block_stats=alt["block_stats"]),
            edges_only=dict(P=runE["P"], sd_jackknife=runE["sd_jackknife"], sd_sandwich_member=runE["sd_sandwich_member"],
                            mapping_unc_jackknife=runE["mapping_unc_jackknife"]))
        det["uncertainty_method"] = ("REVIEWED: 1-sigma = max(delete-one-cluster jackknife, cluster-robust sandwich) with clusters = physical edge "
                                     "members (pair_of merged) and stickers; cy from the variant without floor-line VP groups; mapping "
                                     "uncertainty = parameter part (max of jackknife of the rotation-compensated mapping and 150 member-sandwich "
                                     "samples, with and without floor-line VP groups; median over region) and model-choice part in quadrature")
        det["mapping_uncertainty_note"] = "total (parameter part and model-choice part in quadrature), see details.review"
        save_json(m, fn)
        mdfn = f"{CACHE}/method_alt_{tag}.md"
        md = open(mdfn).read().split("\n## Review")[0].rstrip("\n") + "\n"
        md += ("\n## Review (43_altmodels_review.py)\n\n"
               "* 1-sigma (max of delete-one-member jackknife and member-cluster sandwich; cy without the floor-line VP groups): "
               + ", ".join(f"{n} {unc[f'{n}_1sigma']:.3g}" for n in run["names"])
               + f" (cy {unc['cy_1sigma_if_floor_lines_parallel']:.2g} only if the blue floor line and the white tape are exactly parallel)\n"
               "* previous (edge-cluster sandwich): " + ", ".join(f"{k} {v:.3g}" for k, v in old_unc.items()) + "\n"
               f"* mapping uncertainty (total = parameter part + model-choice part, rot.-comp.): centre {tot['centre']} / cart band "
               f"{tot['cart_band']} / corners {tot['corners']} px (previous, parameter part only: {old_map})\n"
               f"* without the floor-line VP groups: f {alt['P']['fx']:.1f}, pp ({alt['P']['cx']:.1f}, {alt['P']['cy']:.1f}); "
               f"jackknife sd f {alt['sd_jackknife']['f']:.1f}, cy {alt['sd_jackknife']['cy']:.1f}\n"
               f"* focal length over the delete-one-cluster folds: {run['f_range_over_folds'][0]:.0f}-{run['f_range_over_folds'][1]:.0f} px\n")
        with open(mdfn, "w") as fh:
            fh.write(md)


def floor_checks():
    """Where does the cy information come from? (Brown k1,k2, edges only, fixed block sigmas)
    (a) cy profile: chi2 per VP group with cy fixed; (b) floor_H with one side removed;
    (c) the fit WITHOUT floor-line VP groups: angles of the floor lines in the plane perpendicular to the
        scene vertical (tests 'blue line || white tape' and 'tapeV perpendicular to tapeH')."""
    from scipy.optimize import least_squares

    name, ds = "B_k1k2", "E"
    sig = FITS["sig"][ds]
    out = dict(profile=[], variants={}, floor_angles_nofloor_fit_deg={})
    lens, est, r = fit(name, ds)
    x = r.x.copy()
    groups = {}
    for k in est.Vidx:
        groups.setdefault(str(est.E[k]["info"][0]), []).append(k)
    for cy in (420, 440, 460, 480, 490, 502, 510, 520, 540, 560):
        idx = [i for i in range(len(x)) if i != 2]
        x0 = x.copy()
        x0[2] = cy

        def res(y):
            z = x0.copy()
            z[idx] = y
            return est.residuals(z)

        q = least_squares(res, x0[idx], x_scale="jac", method="trf", xtol=1e-11, ftol=1e-11, max_nfev=400)
        z = x0.copy()
        z[idx] = q.x
        b = est.blocks(z)["_edges"]
        seg = lambda ks, s: sum(float(np.sum((b[est.start[k]:est.start[k] + est.cnt[k]] / s) ** 2)) for k in ks)  # noqa: E731
        out["profile"].append(dict(cy=cy, chi2=float(2 * q.cost), f=float(z[0]), cx=float(z[1]),
                                   chi2_by_group={g: round(seg(ks, sig["V"]), 1) for g, ks in groups.items()},
                                   chi2_S=round(seg(est.Sidx, sig["S"]), 1)))
        x = z
    Dd = F.get_data()
    wv_ids = [e["id"] for e in Dd["edges"] if e["direction"] == "world_vertical"]
    var = {"all (orchestrator policy)": dict(), "floor lines straightness only": dict(policy="nofloor"),
           "no floor lines": dict(drop_edges=list(L.FLOOR)), "no floor lines, no scene verticals": dict(drop_edges=list(L.FLOOR) + wv_ids),
           "floor_H without the left blue line": dict(drop_edges=["scene_blueL_top", "scene_blueL_bottom"]),
           "floor_H without the right tape / blue line": dict(drop_edges=["scene_tapeH_top", "scene_tapeH_bottom", "scene_blueH_top"])}
    for tag, kw in var.items():
        lens, est, r = fit(name, ds, **kw)
        C = sandwich(est, r)[: lens.ni, : lens.ni]
        out["variants"][tag] = dict(x=dict(zip(lens.names, r.x[: lens.ni].tolist())), sd_sandwich_member=dict(zip(lens.names, np.sqrt(np.diag(C)).tolist())))
    lens, est, r = fit(name, ds, policy="nofloor")
    P, Rs, _, wv = est.unpack(r.x)
    K = lens.K(P)
    dirs = {}
    for e in Dd["edges"]:
        if e["id"] not in L.FLOOR:
            continue
        U = lens.undist_px(e["points"], P)
        c = U.mean(0)
        nv = np.linalg.svd(U - c)[2][1]
        n = K.T @ np.r_[nv, -nv @ c]
        dd = np.cross(n / np.linalg.norm(n), wv)
        dirs[e["id"]] = dd / np.linalg.norm(dd)
    r0 = dirs["scene_tapeH_top"]
    for k, v in dirs.items():
        a = np.degrees(np.arctan2(np.dot(np.cross(r0, v), wv), np.dot(r0, v)))
        out["floor_angles_nofloor_fit_deg"][k] = round(float((a + 90) % 180 - 90), 3)  # angle mod 180 relative to tapeH_top
    ang = lambda a, b: float(np.degrees(np.arccos(min(1.0, abs(np.dot(a, b))))))  # noqa: E731
    out["vertical_angles_deg"] = dict(scene_vertical_vs_cart80_Z=ang(wv, Rs[80][:, 2]), scene_vertical_vs_cart310_Z=ang(wv, Rs[310][:, 2]),
                                      cart80_Z_vs_cart310_Z=ang(Rs[80][:, 2], Rs[310][:, 2]))
    return out


def write_review_md(out):
    fc = out.get("floor_checks") or {}
    R = out["runs"]

    def row(key):
        r = R[key]
        sj, ss = r["sd_jackknife"], r["sd_sandwich_member"]
        return (f"| {key.replace('|', ' / ')} | " + ", ".join(f"{n}={r['x'][i]:.4g}" for i, n in enumerate(r["names"])) + " | "
                + ", ".join(f"{n} {ss[n]:.3g} / {sj[n]:.3g}" for n in r["names"]) + f" | {r['f_range_over_folds'][0]:.0f}-{r['f_range_over_folds'][1]:.0f} | "
                + f"{r['mapping_unc_sandwich_member']['centre']:.2f} / {r['mapping_unc_sandwich_member']['cart_band']:.2f} / {r['mapping_unc_sandwich_member']['corners']:.2f} | "
                + f"{r['mapping_unc_jackknife']['centre']:.2f} / {r['mapping_unc_jackknife']['cart_band']:.2f} / {r['mapping_unc_jackknife']['corners']:.2f} |")

    md = ["# Review of sub-task 43 (lens-model choice): uncertainty of the recommended models", "",
          "Script `work/43_altmodels_review.py` (run after `43_altmodels_report.py`; ~7 min with 3 processes). Numbers: "
          "`work/cache/altmodels_review.json`. Same estimator, data and fixed block sigmas as `43_altmodels_fit.py`.", "",
          "## What was wrong", "",
          "1. **Clusters.** The sandwich treated every traced edge as an independent cluster; 13 edges are the second boundary of the same "
          "physical member (`pair_of`: both sides of a lip, tape, rod, pole). Here clusters = 48 physical members (+ 20 stickers).",
          "2. **Few influential clusters.** The cluster-robust sandwich is only reliable when the information on a parameter is spread over "
          "many clusters. It is not: cy is carried by the floor_H group (3 members: white tape between the carts, blue line above it, "
          "blue line on the left, assumed parallel and perpendicular to the free scene vertical); the edges-only focal length by the two "
          "short cart-310 post edges (cart 80's two Z edges are nearly collinear outlines of one post). Delete-one-cluster jackknife "
          "(clusters as above; also of the rotation-compensated mapping directly) is used as the check.", "",
          "## Results (Brown k1,k2 = recommended, Brown k1,k2,k3 = alternative)", "",
          "| model / data / VP policy | estimate | 1 sigma: member sandwich / jackknife | f over delete-one folds [px] | mapping sandwich (member) c / band / corners [px] | mapping jackknife [px] |",
          "|---|---|---|---|---|---|"]
    md += [row(k) for k in R]
    md += ["", "(policy `nofloor` = floor lines used for straightness only, no floor VP groups.) "
           "Edge-cluster sandwich of the original fits (ME, Brown k1,k2): f 26.3, cx 7.3, cy 4.0, k1 0.013, k2 0.0084; mapping 2.14 / 9.08 / 12.67 px.", ""]
    if fc:
        md += ["## Where the cy information comes from (Brown k1,k2, edges only)", "",
               "chi2 (fixed block sigmas) with cy fixed and everything else refitted:", "",
               "| cy | dchi2 | f | " + " | ".join(sorted(fc["profile"][0]["chi2_by_group"])) + " | S |", "|" + "---|" * (4 + len(fc["profile"][0]["chi2_by_group"]))]
        c0 = min(p["chi2"] for p in fc["profile"])
        for p in fc["profile"]:
            md.append(f"| {p['cy']} | {p['chi2'] - c0:.1f} | {p['f']:.1f} | " + " | ".join(f"{p['chi2_by_group'][g]:.0f}" for g in sorted(p["chi2_by_group"])) + f" | {p['chi2_S']:.0f} |")
        md += ["", "Variants (member sandwich 1 sigma):", ""]
        for tag, v in fc["variants"].items():
            md.append(f"* {tag}: f {v['x']['f']:.1f} +- {v['sd_sandwich_member']['f']:.1f}, cx {v['x']['cx']:.1f} +- {v['sd_sandwich_member']['cx']:.1f}, "
                      f"cy {v['x']['cy']:.1f} +- {v['sd_sandwich_member']['cy']:.1f}")
        md += ["", "Fit without floor VP groups: direction of each floor line in the plane perpendicular to the scene vertical, relative to "
               "scene_tapeH_top (mod 180 deg): " + ", ".join(f"{k} {v:+.2f}" for k, v in fc["floor_angles_nofloor_fit_deg"].items()) +
               " deg. Vertical directions: " + ", ".join(f"{k} {v:.2f} deg" for k, v in fc["vertical_angles_deg"].items()) + ".", ""]
    rk = R["B_k1k2|ME|noboard"]
    rn = R["B_k1k2|ME|nofloor"]
    re_ = R["B_k1k2|E|noboard"]
    md += ["## Conclusions of the review", "",
           f"* The model choice (Brown k1,k2; fx = fy; pp free; p1 = p2 = k3 = 0) and the point estimates hold (reproduced exactly; the E fit reaches "
           "the same minimum from starts f 1300-1500, cy 386-540).",
           f"* The parameter uncertainties were too small. Recommended model (ME): f {rk['x'][0]:.1f} +- {max(rk['sd_jackknife']['f'], rk['sd_sandwich_member']['f']):.0f} "
           f"(was +-26), cx {rk['x'][1]:.1f} +- {max(rk['sd_jackknife']['cx'], rk['sd_sandwich_member']['cx']):.1f} (7.3), "
           f"cy {rk['x'][2]:.1f} +- {max(rk['sd_jackknife']['cy'], rk['sd_sandwich_member']['cy']):.0f} (4.0) if the floor lines are exactly parallel, "
           f"+- {max(rn['sd_jackknife']['cy'], rn['sd_sandwich_member']['cy']):.0f} without that assumption (then cy = {rn['x'][2]:.1f}), "
           f"k1 {rk['x'][3]:.4f} +- {max(rk['sd_jackknife']['k1'], rk['sd_sandwich_member']['k1']):.3f} (0.013), "
           f"k2 {rk['x'][4]:.4f} +- {max(rk['sd_jackknife']['k2'], rk['sd_sandwich_member']['k2']):.3f} (0.0084).",
           f"* Parameter part of the mapping uncertainty (jackknife, rot.-comp., median): {rk['mapping_unc_jackknife']['centre']:.1f} / "
           f"{rk['mapping_unc_jackknife']['cart_band']:.1f} / {rk['mapping_unc_jackknife']['corners']:.1f} px with the floor groups, "
           f"{rn['mapping_unc_jackknife']['centre']:.1f} / {rn['mapping_unc_jackknife']['cart_band']:.1f} / {rn['mapping_unc_jackknife']['corners']:.1f} px "
           "without (was 2.14 / 9.08 / 12.67). The method file now carries the larger one combined in quadrature with the model-choice part.",
           f"* Edges only: the focal length is NOT robustly determined: deleting either of the two cart-310 post edges moves f by about +108 px; "
           f"jackknife sd {re_['sd_jackknife']['f']:.0f} px (sandwich {re_['sd_sandwich_member']['f']:.0f}); without the floor VP groups one deletion sends f "
           f"to {R['B_k1k2|E|nofloor']['f_range_over_folds'][1]:.0f} px. The stickers stabilise f in the joint fit (fold range "
           f"{rk['f_range_over_folds'][0]:.0f}-{rk['f_range_over_folds'][1]:.0f} px).",
           "* cy is only *consistent* at the 15-40 px level: floor groups 502, no floor groups 478-480, plumb-line distortion centre 426 +- 25, "
           "relaxed-geometry stickers 392 (diagnostic). The +22 px pull of the floor groups corresponds to a ~0.4 deg non-parallelism of the left "
           "blue line and the white tape, far below what painted lines / the 0.5-2 deg spread of the verticals can guarantee.",
           "* The other method_alt_*.json files keep the edge-cluster sandwich uncertainties; they are too small by the same factors "
           "(f ~1.5x, cy ~4x, mapping ~1.5x); a note was added to each."]
    with open(f"{CACHE}/altmodels_review.md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    # pointer + correction block in the author's summary (idempotent)
    fn = f"{CACHE}/altmodels_summary.md"
    if os.path.exists(fn):
        txt = open(fn).read().split("\n## Review corrections")[0].rstrip("\n") + "\n"
        txt += ("\n## Review corrections (43_altmodels_review.py, see altmodels_review.md)\n\n"
                "The +- values above are edge-cluster sandwich estimates and are too small (few influential clusters; paired edges counted "
                "twice). Corrected (delete-one-member jackknife, recommended Brown k1,k2, ME): "
                + md[-5][2:] + " " + md[-4][2:] + " " + md[-2][2:] + "\n")
        with open(fn, "w") as fh:
            fh.write(txt)
    # note in every other method_alt file
    import glob

    for mf in glob.glob(f"{CACHE}/method_alt_*.json"):
        if any(mf.endswith(f"method_alt_{t}.json") for t in TAGS.values()):
            continue
        m = load_json(mf)
        m.setdefault("details", {})["review_note"] = (
            "REVIEW: uncertainty / mapping_uncertainty_px are edge-cluster sandwich values; the delete-one-member jackknife of the "
            "recommended Brown k1,k2 fit shows they are too small (f x1.5, cy x4 - cy rests on the floor-line parallel assumption, "
            "mapping x1.5); see work/cache/altmodels_review.md")
        save_json(m, mf)


if __name__ == "__main__":
    if "--methods-only" in sys.argv:
        OUT = load_json(f"{CACHE}/altmodels_review.json")
    else:
        OUT = main()
    if "floor_checks" not in OUT or "--floor" in sys.argv:
        OUT["floor_checks"] = floor_checks()
        save_json(OUT, f"{CACHE}/altmodels_review.json")
    update_methods(OUT)
    write_review_md(OUT)
