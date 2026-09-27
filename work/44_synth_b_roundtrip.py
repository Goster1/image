"""Synthetic study, step B: round trip of the orchestrator estimators on synthetic replicates.

Scenarios (truth x noise):
  T1_a, T2_a : detection-level noise only (corners: per-frame std / sqrt 7; edges: white AR(1) noise with the
               per-edge sigma / lag-1 correlation of the real high-frequency residuals)
  T1_b, T2_b : realistic: (a) + unknown per-sticker 3D placement error (sigma_sticker) + per cart-row height error
               (sigma_row), sized so that the sticker RMS of the drawing fit equals the real one; + per-edge bows
               (real low-frequency residual profile, random sign) + per-edge direction deviations for VP-group edges
               (real angle w.r.t. the reference VP, random sign)
  T2_c       : what-if: the end-board height offsets of the geometry diagnosis (+61..+81 mm, deterministic) +
               detection noise; sticker estimators and combined only (edges as in a)
Estimators: markers-only (k1_ppfix, k1k2_ppfix, k1_ppfree, k1k2_ppfree; 20_markers_ba fit logic, perturbed start
and in addition the orchestrator's start from initial_calib), plumb-line (centre fixed / free), VP (plumb
distortion fixed), joint lines, combined (markers + exact lines + VP groups + straightness + sticker sides).
Claimed uncertainties: covariance / cluster sandwich per replicate; bootstrap (stickers / edges) on a subset.

Usage: python3 44_synth_b_roundtrip.py [NREP=30] [NWORKERS=3] [--resume]
Writes work/cache/synthetic_roundtrip_raw.pkl (with --resume: skips finished tasks, writes synthetic_roundtrip_raw_partN.pkl)
"""
import importlib
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import CACHE, K_from, load_json
from linedata import stratified_resample
from markerdata import to_points

L = importlib.import_module("44_synth_lib")

NREP = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 30
NW = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 3
NB_MARK = (6, 40)  # (replicates with bootstrap, bootstrap size) markers
NB_LINE = (3, 12)
NB_COMB = (2, 8)

SETUP = load_json(f"{CACHE}/synthetic_setup.json")
M, E, S = L.load_real()
NE = len(E)
init = load_json(f"{CACHE}/initial_calib.json")
K_INIT, D_INIT = np.array(init["K"]), np.array(init["dist"])
P_INIT = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
CSIG = [np.array(c) for c in SETUP["corner_sigma_per_coord"]]
STATS = SETUP["edge_stats"]
GN = SETUP["geometry_noise"]


class _T:  # light-weight truth from the setup file (identical poses in every worker)
    def __init__(self, name, t):
        self.name = name
        self.K = K_from(t["f"], t["f"], t["cx"], t["cy"])
        self.d = np.array([t["k1"], t["k2"], 0, 0, 0.0])
        self.poses = {int(c): np.array(v) for c, v in t["poses"].items()}
        self.wv = np.array(t["wv"])
        self.floor = {g: np.array(v) for g, v in t["floor"].items()}
        self.params = {k: t[k] for k in ("f", "cx", "cy", "k1", "k2")}


TR = {k: _T(k, v) for k, v in SETUP["truths"].items()}
SCEN = {"T1_a": ("T1", "a"), "T1_b": ("T1", "b"), "T2_a": ("T2", "a"), "T2_b": ("T2", "b"), "T2_c": ("T2", "c")}
IDEAL = {tn: L.ideal_edges(T, E, S) for tn, T in TR.items()}


def truth_vec(T, names, kind):
    p = dict(T.params)
    f = p["f"]
    if kind == "plumb":
        p = dict(cx=p["cx"], cy=p["cy"], k1=p["k1"] * (L.F0 / f) ** 2, k2=p["k2"] * (L.F0 / f) ** 4)
    return np.array([p[n] for n in names])


def lens_general(names, x, T, kind, dpx=None):
    p = dict(zip(names, x))
    if kind == "plumb":
        f = T.params["f"]
        cx, cy = p.get("cx", L.CEN[0]), p.get("cy", L.CEN[1])
        d = np.array([p["k1"] * (f / L.F0) ** 2, p["k2"] * (f / L.F0) ** 4, 0, 0, 0])
        return K_from(f, f, cx, cy), d
    f = p["f"]
    cx, cy = p.get("cx", L.CEN[0]), p.get("cy", L.CEN[1])
    if kind == "vp":
        d = np.array([dpx[0] * f ** 2, dpx[1] * f ** 4, 0, 0, 0])
    else:
        d = np.array([p.get("k1", 0.0), p.get("k2", 0.0), 0, 0, 0])
    return K_from(f, f, cx, cy), d


def make_data(scen, rep):
    tn, kind = SCEN[scen]
    T = TR[tn]
    g = np.random.default_rng(1000 * (list(SCEN).index(scen) + 1) + rep)
    if kind == "b":
        off = L.geometry_offsets(M, g, GN["sigma_sticker_mm"], GN["sigma_row_mm"])
    elif kind == "c":
        off = L.diagnosed_offsets(M)
    else:
        off = None
    pts = L.sticker_obs(T, M, offsets=off, rng=g, corner_sigma=CSIG)
    ide = IDEAL[tn] if off is None else L.ideal_edges(T, E, S, side_offsets=off)
    edges = L.noisy_edges(ide, STATS, g, bows=(kind == "b"), directions=(kind == "b"))
    # perturbed start
    f0 = T.params["f"] * (1 + 0.08 * g.standard_normal())
    K0 = K_from(f0, f0, T.params["cx"] + 20 * g.standard_normal(), T.params["cy"] + 20 * g.standard_normal())
    d0 = T.d + np.array([0.04 * g.standard_normal(), 0.02 * g.standard_normal(), 0, 0, 0])
    P0 = {c: T.poses[c] + np.r_[0.01 * g.standard_normal(3), 30 * g.standard_normal(3)] for c in (80, 310)}
    return T, kind, pts, edges, (K0, d0, P0), g


def rec(names, x, T, kind_lens, C=None, Cs=None, extra=None, dpx=None, g=None, claim_from=("C",)):
    K, d = lens_general(names, x, T, kind_lens, dpx)
    out = dict(names=list(names), x=np.asarray(x, float), truth=truth_vec(T, names, "plumb" if kind_lens == "plumb" else "std"),
               disp=L.disp_field(T.K, T.d, K, d).astype(np.float32))
    for key, CC in (("C", C), ("Cs", Cs)):
        if CC is None:
            continue
        out["sig_" + key] = np.sqrt(np.clip(np.diag(CC), 0, None))
        if key in claim_from:
            fn = (lambda xx: lens_general(names, xx, T, kind_lens, dpx))
            cm = L.claimed_mapping(K, d, CC, fn, np.asarray(x, float), g, ns=12)
            out["claim_map_" + key] = None if cm is None else cm.astype(np.float32)
    if extra:
        out.update(extra)
    return out


def run_task(task):
    scen, rep = task
    t0 = time.time()
    T, kind, pts, edges, (K0, d0, P0), g = make_data(scen, rep)
    res = {}
    # ---------------- markers only
    for name, spec in L.MARKER_SPECS.items():
        r = L.marker_estimate(spec, pts, K0, d0, P0)
        r2 = L.marker_estimate(spec, pts, K_INIT, D_INIT, P_INIT)  # orchestrator's start
        extra = dict(rms=r["rms"], x_orchestrator_start=r2["x"], rms_orchestrator_start=r2["rms"])
        if rep < NB_MARK[0] and name in ("k1k2_ppfix", "k1k2_ppfree", "k1_ppfix"):
            mis = sorted({p["mi"] for p in pts})
            bx = []
            for b in range(NB_MARK[1]):
                pick = g.choice(mis, size=len(mis), replace=True)
                bp = []
                for q, mi in enumerate(pick):
                    bp += [dict(p, mi=1000 * q + mi) for p in pts if p["mi"] == mi]
                if len({p["cart"] for p in bp}) < 2:
                    continue
                Kx, dx = r["unpack"](r["x"])
                rb = L.marker_estimate(spec, bp, Kx, dx, {80: r["full_x"][len(r["x"]):len(r["x"]) + 6], 310: r["full_x"][len(r["x"]) + 6:]})
                bx.append(rb["x"])
            bx = np.array(bx)
            extra["boot_x"] = bx
            extra["sig_boot"] = bx.std(0)  # 20_markers_ba quotes the plain std
            extra["sig_boot_robust"] = L.rstd(bx)
            # bootstrap mapping claim (displacement of each bootstrap replicate vs the estimate)
            Kx, dx = r["unpack"](r["x"])
            Ds = np.array([L.disp_field(Kx, dx[:5], *[np.asarray(v)[:5] if i else v for i, v in enumerate(r["unpack"](xb))]) for xb in bx[:30]])
            extra["claim_map_boot"] = np.sqrt(np.nanmean(np.sum(Ds ** 2, axis=2), axis=0)).astype(np.float32)
        res["markers_" + name] = rec(r["names"], r["x"], T, "std", C=r["C"], Cs=r["Cs"], extra=extra, g=g, claim_from=("C", "Cs"))
    # ---------------- combined (50_combined: robust IRLS main result; plain variant for comparison)
    for cname, robust in (("combined_k1k2", True), ("combined_nonrobust", False)):
        cb, rc, Cc, Kc, dc = L.combined_estimate(pts, edges, K0, d0, P0, robust=robust)
        comb_retry = None
        if cb.block_rms(rc.x)["S"][0] > 1.0:  # failed start -> orchestrator's default start (50_combined.run)
            comb_retry = float(cb.block_rms(rc.x)["S"][0])
            cb, rc, Cc, Kc, dc = L.combined_estimate(pts, edges, K_from(1420.0, 1420.0, *L.CEN), np.array([-0.33, 0.09, 0, 0, 0]), P0, robust=robust)
        extra = dict(block_rms=cb.block_rms(rc.x), sig_blocks=dict(cb.sig), failed_start_S_rms=comb_retry,
                     sticker_rms=cb.sticker_rms(rc.x) if robust else None)
        if robust and rep < NB_COMB[0]:
            mis = sorted({p["mi"] for p in pts})
            bx = []
            for b in range(NB_COMB[1]):
                pick_m = g.choice(mis, len(mis), replace=True)
                bp = []
                for q, mi in enumerate(pick_m):
                    bp += [dict(p, mi=1000 * q + mi) for p in pts if p["mi"] == mi]
                if len({p["cart"] for p in bp}) < 2:
                    continue
                be = stratified_resample(edges[:NE], g) + edges[NE:]  # as 50_combined: lines within VP group, sides follow
                try:
                    cbb, rb, _, _, _ = L.combined_estimate(bp, be, None, None, None, sig=dict(cb.sig), x_init=rc.x.copy())
                    bx.append(rb.x[: cb.ni])
                except Exception as ex:  # noqa
                    pass
            bx = np.array(bx)
            extra["boot_x"] = bx
            extra["sig_boot"] = L.rstd(bx)  # 50_combined quotes the robust std
            extra["sig_boot_std"] = bx.std(0)
            Ds = np.array([L.disp_field(Kc, dc, *lens_general(cb.inames, xb, T, "std")) for xb in bx])
            extra["claim_map_boot"] = np.sqrt(np.nanmean(np.sum(Ds ** 2, axis=2), axis=0)).astype(np.float32)
        res[cname] = rec(cb.inames, rc.x[: cb.ni], T, "std", C=Cc, extra=extra, g=g, claim_from=("C",))
    if kind != "c":
        # ---------------- lines (structural edges only, as 30_lines_methods)
        ed = edges[:NE]
        f_t = T.params["f"]
        kx0 = np.array([T.d[0] * (L.F0 / f_t) ** 2 + 0.04 * g.standard_normal(), T.d[1] * (L.F0 / f_t) ** 4 + 0.02 * g.standard_normal()])
        # perturbed start; if the fit fails (edge RMS > 1 px: the start folds the distortion inside the image and
        # undistort_points diverges for the edges near the corners) -> retry from the orchestrator's default start
        lcp, rp, Ccp, Csp = L.plumb_estimate(ed, centre_free=False, x0=kx0)
        fail_p = float(np.sqrt(np.mean(rp.fun ** 2)))
        if fail_p > 1.0:
            lcp, rp, Ccp, Csp = L.plumb_estimate(ed, centre_free=False)
        else:
            fail_p = None
        res["plumb_fixed"] = rec(lcp.names[: lcp.n_intr], rp.x[: lcp.n_intr], T, "plumb", C=Ccp, Cs=Csp, g=g, claim_from=("Cs",),
                                 extra=dict(rms=float(np.sqrt(np.mean(rp.fun ** 2))), failed_start_rms=fail_p, start=kx0))
        x0f = np.r_[L.CEN + 20 * g.standard_normal(2), kx0]
        lcf, rf, Ccf, Csf = L.plumb_estimate(ed, centre_free=True, x0=x0f)
        fail_f = float(np.sqrt(np.mean(rf.fun ** 2)))
        if fail_f > 1.0 or np.hypot(*(rf.x[:2] - L.CEN)) > 600:
            fail_f = [fail_f, rf.x[:4].tolist()]
            lcf, rf, Ccf, Csf = L.plumb_estimate(ed, centre_free=True)
        else:
            fail_f = None
        res["plumb_free"] = rec(lcf.names[: lcf.n_intr], rf.x[: lcf.n_intr], T, "plumb", C=Ccf, Cs=Csf, g=g, claim_from=("Cs",),
                                extra=dict(rms=float(np.sqrt(np.mean(rf.fun ** 2))), failed_start=fail_f, start=x0f))
        dpx = np.array([rp.x[0] / L.F0 ** 2, rp.x[1] / L.F0 ** 4])
        lcv, rv, Ccv, Csv = L.vp_estimate(ed, dpx, K0, P0)
        res["vp"] = rec(lcv.names[: lcv.n_intr], rv.x[: lcv.n_intr], T, "vp", C=Ccv, Cs=Csv, dpx=dpx, g=g, claim_from=("Cs",))
        lcj, rj, Ccj, Csj = L.joint_estimate(ed, K0, d0, P0)
        fail_j = float(np.sqrt(np.mean(rj.fun ** 2)))
        if fail_j > 2.0:
            lcj, rj, Ccj, Csj = L.joint_estimate(ed, K_from(1300.0, 1300.0, *L.CEN), np.array([-0.28, 0.05, 0, 0, 0]), P0)
        else:
            fail_j = None
        res["lines_joint"] = rec(lcj.names[: lcj.n_intr], rj.x[: lcj.n_intr], T, "std", C=Ccj, Cs=Csj, g=g, claim_from=("Cs",),
                                 extra=dict(rms=float(np.sqrt(np.mean(rj.fun ** 2))), failed_start_rms=fail_j))
        if rep < NB_LINE[0]:
            bp_, bv_, bj_ = [], [], []
            for b in range(NB_LINE[1]):
                sub = stratified_resample(ed, g)  # as 30_lines_methods: within VP groups, subsample 4
                try:
                    _, r1, _, _ = L.plumb_estimate(sub, subsample=4)
                    d1 = np.array([r1.x[0] / L.F0 ** 2, r1.x[1] / L.F0 ** 4])
                    _, r2, _, _ = L.vp_estimate(sub, d1, None, None, subsample=4, x_init=rv.x.copy())
                    _, r3, _, _ = L.joint_estimate(sub, None, None, None, subsample=4, x_init=rj.x.copy())
                    bp_.append(r1.x[:2])
                    bv_.append(np.r_[r2.x[:3], d1])
                    bj_.append(r3.x[: lcj.n_intr])
                except Exception as ex:  # noqa
                    print("line boot fail", ex, flush=True)
            bp_, bv_, bj_ = np.array(bp_), np.array(bv_), np.array(bj_)
            res["plumb_fixed"]["sig_boot"] = L.rstd(bp_)  # 30_lines_methods quotes the robust std
            res["plumb_fixed"]["sig_boot_std"] = bp_.std(0)
            res["plumb_fixed"]["boot_x"] = bp_
            res["vp"]["sig_boot"] = L.rstd(bv_[:, :3])
            res["vp"]["sig_boot_std"] = bv_[:, :3].std(0)
            res["vp"]["boot_x"] = bv_
            res["lines_joint"]["sig_boot"] = L.rstd(bj_)
            res["lines_joint"]["sig_boot_std"] = bj_.std(0)
            res["lines_joint"]["boot_x"] = bj_
            # bootstrap mapping claims (each replicate re-estimates everything, like 30_lines_methods)
            Kv, dv = lens_general(lcv.names[:3], rv.x[:3], T, "vp", dpx)
            Ds = np.array([L.disp_field(Kv, dv, *lens_general(["f", "cx", "cy"], xb[:3], T, "vp", xb[3:])) for xb in bv_])
            res["vp"]["claim_map_boot"] = np.sqrt(np.nanmean(np.sum(Ds ** 2, axis=2), axis=0)).astype(np.float32)
            Kj2, dj2 = lens_general(lcj.names[: lcj.n_intr], rj.x[: lcj.n_intr], T, "std")
            Ds = np.array([L.disp_field(Kj2, dj2, *lens_general(lcj.names[: lcj.n_intr], xb, T, "std")) for xb in bj_])
            res["lines_joint"]["claim_map_boot"] = np.sqrt(np.nanmean(np.sum(Ds ** 2, axis=2), axis=0)).astype(np.float32)
            Kp2, dp2 = lens_general(["k1", "k2"], rp.x[:2], T, "plumb")
            Ds = np.array([L.disp_field(Kp2, dp2, *lens_general(["k1", "k2"], xb, T, "plumb")) for xb in bp_])
            res["plumb_fixed"]["claim_map_boot"] = np.sqrt(np.nanmean(np.sum(Ds ** 2, axis=2), axis=0)).astype(np.float32)
    return scen, rep, res, time.time() - t0


def load_done(paths):
    """(scen, rep) -> result from earlier (partial) runs."""
    done = {}
    for fn in paths:
        if os.path.exists(fn):
            with open(fn, "rb") as fh:
                d = pickle.load(fh)
            for sc, reps in d["out"].items():
                for rp, res in reps.items():
                    done[(sc, rp)] = res
    return done


if __name__ == "__main__":
    # optional: --resume  -> skip tasks already stored in synthetic_roundtrip_raw*.pkl, write to a new part file
    resume = "--resume" in sys.argv
    tasks = [(s, r) for r in range(NREP) for s in SCEN]
    # expensive (bootstrap) tasks first for load balancing
    tasks.sort(key=lambda t: 0 if t[1] < max(NB_MARK[0], NB_LINE[0], NB_COMB[0]) else 1)
    import glob

    parts = sorted(glob.glob(f"{CACHE}/synthetic_roundtrip_raw*.pkl"))
    redo = set()
    if "--redo" in sys.argv:
        for q in sys.argv[sys.argv.index("--redo") + 1].split(","):
            a, b = q.split(":")
            redo.add((a, int(b)))
    if resume:
        done = load_done(parts)
        tasks = [t for t in tasks if t not in done or t in redo]
        if "--reverse" in sys.argv:  # second helper process working from the end of the task list
            tasks = tasks[::-1]
        fn = f"{CACHE}/synthetic_roundtrip_raw_part{len(parts) + 1}.pkl"
        print("resume: already done", len(done), "remaining", len(tasks), "->", fn)
    else:
        for q in parts:
            os.remove(q)
        fn = f"{CACHE}/synthetic_roundtrip_raw.pkl"
    out = {s: {} for s in SCEN}
    t0 = time.time()
    meta = dict(setup_review=SETUP["review_status"], nrep=NREP, n_edges=NE, uv=L.UV, uv_shape=L.UV_SHAPE,
                truths={k: v.params for k, v in TR.items()})
    with Pool(NW) as pool:
        for i, (scen, rep, res, dt) in enumerate(pool.imap_unordered(run_task, tasks)):
            out[scen][rep] = res
            print(f"{i + 1}/{len(tasks)} {scen} rep {rep} {dt:.1f}s elapsed {time.time() - t0:.0f}s "
                  f"f(k1k2_ppfree)={res['markers_k1k2_ppfree']['x'][0]:.1f} f(comb)={res['combined_k1k2']['x'][0]:.1f}", flush=True)
            with open(fn + ".tmp", "wb") as fh:
                pickle.dump(dict(out=out, **meta), fh)
            os.replace(fn + ".tmp", fn)
    print("done", round(time.time() - t0), "s")
