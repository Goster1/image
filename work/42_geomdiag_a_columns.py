"""(1) Projective invariants of the sticker columns (independent of f, principal point and pose).

For every stacked column (same X, Y; Z = 0 / -195 / -595 / -995 for top / A / B / C) we collect the
corresponding corners (valid corners) and, for hidden corners, the intersection of a valid traced side
line of that sticker with the column line. Everything is done in the undistorted image (plumb-line
distortion from all verified straight edges, pixel units; variants: centre fixed / free / k1 only /
none). Tests:
  * collinearity of the column points (residual in distorted px vs. the expected noise, Monte Carlo);
  * cross ratio CR(top, A, B, C) (expected 1.1960) -> implied Z of the top sticker with the shelves at
    the drawing heights;
  * with the vertical vanishing point v (from the vertical cart-post edges): affine ratios along the
    column -> implied gaps (top-A, A-B) in mm when the B-C gap is taken as 400 mm, and the implied top
    height from each pair of shelf levels;
  * the vertical VP implied by the column lines (per column: distance of v from the line; per cart:
    intersection of all column lines) vs v from the edges.
Noise: corner per-frame scatter (corners_std_px / sqrt 2 per coordinate; the 7-frame mean is ~2.6x
better, so this is conservative); side lines: rms / sqrt(n/3) offset and matching angle; v: the
covariance of the edge fit with a floor of 3 px (post verticality) -> Monte Carlo (N = 3000).
Output: work/cache/geomdiag_columns.json, results/geomdiag_columns.png
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS, save_json  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")
rng = np.random.default_rng(42)
NMC = 3000
VP_FLOOR = 3.0  # px, systematic floor for the cart-Z vanishing point (verticality / straightness of the posts)
CR_EXP = G.cross_ratio(0.0, -195.0, -595.0, -995.0)
ZL = np.array([G.Z_LEVEL[l] for l in G.LEVELS])
MIN_ANGLE = np.deg2rad(15.0)


def column_obs_und(col, und):
    """Undistort the observations of one column: returns list per level of dicts with nominal
    undistorted geometry + linearised perturbation operators."""
    out = {}
    for lev in G.LEVELS:
        ob = col["obs"][lev]
        if ob is None:
            out[lev] = None
            continue
        if isinstance(ob, tuple) and ob[0] == "pt":
            uv, s = ob[1], ob[2]
            U, J = und.jac(uv[None])
            out[lev] = dict(kind="pt", U=U[0], J=J[0], s=s, uv=uv)
        else:
            lines = []
            for (_, p, d, s_o, s_t, half, side) in ob:
                n = np.array([-d[1], d[0]])
                P2 = np.array([p - half * d, p + half * d])
                U, J = und.jac(P2)
                lines.append(dict(kind="line", U=U, J=J, n=n, d=d, half=half, s_o=s_o, s_t=s_t, side=side, p=p))
            out[lev] = lines
    return out


def sample_column(ob, eps=None):
    """One (perturbed if eps) realisation: points {lev: U} and lines {lev: [(c, d)]}."""
    pts, lns = {}, {}
    for lev, o in ob.items():
        if o is None:
            continue
        if isinstance(o, dict):
            U = o["U"].copy()
            if eps is not None:
                U = U + o["J"] @ (rng.normal(0, o["s"], 2))
            pts[lev] = U
        else:
            L = []
            for li in o:
                U = li["U"].copy()
                if eps is not None:
                    do = rng.normal(0, li["s_o"])
                    dt = rng.normal(0, li["s_t"])
                    dD = np.array([li["n"] * (do - li["half"] * dt), li["n"] * (do + li["half"] * dt)])
                    U = U + np.einsum("nij,nj->ni", li["J"], dD)
                c = U.mean(0)
                d = (U[1] - U[0]) / np.linalg.norm(U[1] - U[0])
                L.append((c, d))
            lns[lev] = L
    return pts, lns


def analyse_column(pts, lns, v, und_jac_pts=None):
    """Column line from points (>= 2) or from the single point + v. Returns dict of t positions etc."""
    levs = [l for l in G.LEVELS if l in pts]
    res = dict(n_pt=len(levs))
    if len(levs) >= 2:
        P = np.array([pts[l] for l in levs])
        c, d, n, r = G.tls_line(P)
        res["resid_u"] = dict(zip(levs, r))
        res["line_from"] = "points"
    elif len(levs) == 1 and v is not None:
        c = pts[levs[0]]
        d = (v - c) / np.linalg.norm(v - c)
        n = np.array([-d[1], d[0]])
        res["line_from"] = "point+vp"
    else:
        return None
    if v is not None and np.dot(v - c, d) < 0:
        d, n = -d, -n
    elif v is None and len(levs) >= 2 and np.dot(pts[levs[-1]] - pts[levs[0]], d) < 0:
        d, n = -d, -n
    t = {l: float(np.dot(pts[l] - c, d)) for l in levs}
    used_line = {}
    for l, L in lns.items():
        best = None
        for (lc, ld) in L:
            ang = np.arccos(min(abs(np.dot(ld, d)), 1.0))
            if ang < MIN_ANGLE:
                continue
            if best is None or ang > best[0]:
                best = (ang, lc, ld)
        if best is None:
            continue
        X = G.intersect_lines(c, d, best[1], best[2])
        t[l] = float(np.dot(X - c, d))
        used_line[l] = float(np.rad2deg(best[0]))
    res.update(c=c, d=d, n=n, t=t, used_line=used_line)
    if v is not None:
        res["t_v"] = float(np.dot(v - c, d))
        res["vp_perp_u"] = float(np.dot(v - c, n))  # distance of v from the column line (undist px)
        res["vp_angle_deg"] = float(np.rad2deg(np.arctan2(np.dot(v - c, n), np.dot(v - c, d))))
    return res


def derived(res):
    """Cross ratio, implied top Z, VP-affine gaps."""
    t = res["t"]
    out = {}
    if all(l in t for l in G.LEVELS):
        tt = np.array([t[l] for l in G.LEVELS])
        out["CR"] = float(G.cross_ratio(*tt))
        abc = G.proj1d_fit(ZL[1:], tt[1:])
        out["Ztop_from_ABC"] = float(G.proj1d_inv(abc, tt[0]))
    if all(l in t for l in ("A", "B", "C")):
        tt = np.array([t[l] for l in ("A", "B", "C")])
        a, b, c = G.proj1d_fit(ZL[1:], tt)
        out["t_inf_from_ABC"] = float(a / c)
        if "t_v" in res:
            out["vp_along_ABC_minus_edge"] = float(a / c - res["t_v"])
    if all(l in t for l in G.LEVELS):
        a, b, c = G.proj1d_fit(ZL, np.array([t[l] for l in G.LEVELS]))
        if "t_v" in res:
            out["vp_along_all4_minus_edge"] = float(a / c - res["t_v"])
    if "t_v" in res:
        tv = res["t_v"]
        # with t_inf known: s(Z) = 1/(tv - t) is affine in Z  (1D projectivity with t(inf) = tv)
        s = {l: 1.0 / (tv - t[l]) for l in t}
        if "B" in s and "C" in s:
            k = 400.0 / (s["B"] - s["C"])  # mm per unit of s
            if "top" in s and "A" in s:
                out["gap_topA_mm_if_BC400"] = float((s["top"] - s["A"]) * k)
            if "A" in s:
                out["gap_AB_mm_if_BC400"] = float((s["A"] - s["B"]) * k)
        # implied top Z from each pair of shelf levels (drawing heights), affine in s
        for p, q in (("A", "C"), ("A", "B"), ("B", "C")):
            if "top" in s and p in s and q in s:
                Zp, Zq = G.Z_LEVEL[p], G.Z_LEVEL[q]
                out[f"Ztop_from_{p}{q}_vp"] = float(Zp + (s["top"] - s[p]) * (Zq - Zp) / (s[q] - s[p]))
        if all(l in s for l in ("A", "B", "C")):
            out["ZB_from_AC_vp"] = float(-195.0 + (s["B"] - s["A"]) * (-800.0) / (s["C"] - s["A"]))
    return out


def resid_distorted(res, ob):
    """Perpendicular residuals of the column points converted to distorted px."""
    out = {}
    if "resid_u" not in res:
        return out
    for l, r in res["resid_u"].items():
        J = ob[l]["J"]
        jt = J.T @ res["n"]
        out[l] = float(r / np.hypot(*jt))
    return out


def cart_vp(col_res, weights=None):
    """Least-squares intersection of column lines (c, d): minimise sum w (n.(v - c))^2."""
    A, b = [], []
    for k, r in enumerate(col_res):
        w = 1.0 if weights is None else weights[k]
        A.append(r["n"] * np.sqrt(w))
        b.append(np.dot(r["n"], r["c"]) * np.sqrt(w))
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def run_variant(variant, ms, edges, do_mc):
    pl = G.plumb(variant, edges)
    und = G.Und(pl)
    vpe = G.fit_vp_lines(und, G.vertical_edges(edges))
    v = vpe["v"]
    Cv = vpe["cov_classic"] + np.eye(2) * VP_FLOOR ** 2
    cols = G.columns(ms)
    rows = []
    for col in cols:
        ob = column_obs_und(col, und)
        pts, lns = sample_column(ob)
        res = analyse_column(pts, lns, v)
        if res is None:
            continue
        row = dict(cart=col["cart"], end=col["end"], XY=col["XY"], top_corner=col["top_corner"], n_pt=res["n_pt"],
                   line_from=res["line_from"], levels_pt=[l for l in G.LEVELS if l in pts],
                   levels_line={l: a for l, a in res["used_line"].items()}, t=res["t"], t_v=res.get("t_v"),
                   vp_perp_u=res.get("vp_perp_u"), vp_angle_deg=res.get("vp_angle_deg"),
                   resid_dist=resid_distorted(res, ob), nominal=derived(res), c=res["c"].tolist(), d=res["d"].tolist())
        if do_mc:
            samp = {}
            rms_s, vpp_s = [], []
            # ideal (exactly collinear) copy of the point observations for the noise-only collinearity test
            ob_ideal = None
            if res["line_from"] == "points" and res["n_pt"] >= 3:
                ob_ideal = {l: (dict(o, U=res["c"] + np.dot(o["U"] - res["c"], res["d"]) * res["d"]) if isinstance(o, dict) else None)
                            for l, o in ob.items() if o is not None}
                ob_ideal = {l: o for l, o in ob_ideal.items() if o is not None}
            for _ in range(NMC):
                vs = rng.multivariate_normal(v, Cv)
                p2, l2 = sample_column(ob, eps=True)
                r2 = analyse_column(p2, l2, vs)
                if r2 is None:
                    continue
                dd = derived(r2)
                for k, val in dd.items():
                    samp.setdefault(k, []).append(val)
                if ob_ideal is not None:
                    p3, _ = sample_column(ob_ideal, eps=True)
                    r3 = analyse_column(p3, {}, None)
                    rd = resid_distorted(r3, ob_ideal)
                    rms_s.append(np.sqrt(np.mean(np.square(list(rd.values())))))
                if r2["line_from"] == "points":
                    vpp_s.append(np.dot(vs - r2["c"], r2["n"]))
            row["mc"] = {k: dict(mean=float(np.mean(s)), std=float(np.std(s)), q16=float(np.percentile(s, 16)),
                                 q84=float(np.percentile(s, 84))) for k, s in samp.items()}
            if rms_s:
                meas = np.sqrt(np.mean(np.square(list(row["resid_dist"].values()))))
                row["collinearity"] = dict(rms_meas_px=float(meas), rms_noise_median_px=float(np.median(rms_s)),
                                           rms_noise_q95_px=float(np.percentile(rms_s, 95)),
                                           p_value=float(np.mean(np.array(rms_s) >= meas)))
            if vpp_s:
                row["vp_perp_noise_std_u"] = float(np.std(vpp_s))
        rows.append(row)
    # per-cart vertical VP from the column lines (only columns fitted from >= 2 points)
    cvp = {}
    for cart in (80, 310):
        sub = [r for r in rows if r["cart"] == cart and r["line_from"] == "points"]
        rr = [dict(c=np.array(r["c"]), n=np.array([-r["d"][1], r["d"][0]])) for r in sub]
        vv = cart_vp(rr)
        cvp[cart] = dict(v=vv.tolist(), n_columns=len(sub), minus_edge_vp=(vv - v).tolist())
    allr = [dict(c=np.array(r["c"]), n=np.array([-r["d"][1], r["d"][0]])) for r in rows if r["line_from"] == "points"]
    cvp["both"] = dict(v=cart_vp(allr).tolist(), minus_edge_vp=(cart_vp(allr) - v).tolist())
    wv = G.vertical_edges(edges, kinds=("world_vertical",))
    vpw = G.fit_vp_lines(und, wv)["v"].tolist() if len(wv) >= 2 else None
    return dict(variant=variant, plumb=dict(centre=pl["centre"], dist_at_f1300=pl["dist"][:3], rms=pl["rms"]),
                vp_world_vertical=vpw, n_world_vertical=len(wv),
                vp_edges=dict(v=v.tolist(), sd_formal=np.sqrt(np.diag(vpe["cov_classic"])).tolist(),
                              sd_cluster=np.sqrt(np.abs(np.diag(vpe["cov_cluster"]))).tolist(), floor=VP_FLOOR,
                              rms=vpe["rms"], per_edge=dict(zip(vpe["ids"], vpe["per_edge"]))),
                columns=rows, column_vp=cvp)


def summarise_end(rows, key):
    """Inverse-variance mean of a MC quantity over the columns of one cart end."""
    vals, sds = [], []
    for r in rows:
        if "mc" in r and key in r["mc"] and key in r["nominal"]:
            vals.append(r["nominal"][key])
            sds.append(r["mc"][key]["std"])
    if not vals:
        return None
    w = 1 / np.square(sds)
    m = float(np.sum(w * vals) / w.sum())
    chi2 = float(np.sum(w * (np.array(vals) - m) ** 2))
    return dict(mean=m, sd=float(1 / np.sqrt(w.sum())), n=len(vals), values=[float(x) for x in vals],
                sds=[float(s) for s in sds], chi2=chi2)


def main():
    ms = G.markers()
    edges = G.all_edges()
    out = {"cr_expected": CR_EXP, "n_mc": NMC, "edges_used": len(edges)}
    variants = ["k1k2_centre_fixed", "k1k2_centre_free", "k1_centre_fixed", "none"]
    for var in variants:
        R = run_variant(var, ms, edges, do_mc=(var == "k1k2_centre_fixed" or var == "k1k2_centre_free"))
        out[var] = R
    # ---- per-end summaries (main variant with MC) ----
    keys = ["vp_along_ABC_minus_edge", "vp_along_all4_minus_edge", "Ztop_from_ABC", "Ztop_from_AC_vp", "Ztop_from_AB_vp", "Ztop_from_BC_vp", "gap_topA_mm_if_BC400",
            "gap_AB_mm_if_BC400", "ZB_from_AC_vp", "CR"]
    for var in ("k1k2_centre_fixed", "k1k2_centre_free"):
        summ = {}
        for cart in (80, 310):
            for end in (0, 1600):
                rows = [r for r in out[var]["columns"] if r["cart"] == cart and r["end"] == end]
                summ[f"{cart}_{end}"] = {k: summarise_end(rows, k) for k in keys}
        out[var]["end_summary"] = summ
    # variant sensitivity of the nominal implied top height (no MC)
    sens = {}
    for var in variants:
        for cart in (80, 310):
            for end in (0, 1600):
                rows = [r for r in out[var]["columns"] if r["cart"] == cart and r["end"] == end]
                vals = [r["nominal"].get("Ztop_from_ABC") for r in rows if r["nominal"].get("Ztop_from_ABC") is not None]
                vals2 = [r["nominal"].get("gap_topA_mm_if_BC400") for r in rows if r["nominal"].get("gap_topA_mm_if_BC400") is not None]
                sens.setdefault(f"{cart}_{end}", {})[var] = dict(Ztop_from_ABC=[round(x, 1) for x in vals],
                                                               gap_topA_if_BC400=[round(x, 1) for x in vals2])
    out["variant_sensitivity"] = sens
    save_json(out, f"{CACHE}/geomdiag_columns.json")

    # ---- print ----
    main_v = out["k1k2_centre_fixed"]
    print("VP from cart-post edges:", {v: np.round(out[v]["vp_edges"]["v"], 1).tolist() for v in variants})
    print("VP from world-vertical edges:", {v: (np.round(out[v]["vp_world_vertical"], 1).tolist() if out[v]["vp_world_vertical"] else None) for v in variants})
    print("column VP per cart:", {v: {k: np.round(x["v"], 1).tolist() for k, x in out[v]["column_vp"].items()} for v in variants})
    for r in main_v["columns"]:
        nm = r["nominal"]
        mc = r.get("mc", {})
        col = r.get("collinearity", {})
        print(f"cart {r['cart']:3d} end {r['end']:4d} XY {np.round(r['XY'], 1)} pts {r['levels_pt']} lines {r['levels_line']}"
              f" | collin rms {col.get('rms_meas_px', np.nan):.2f} (noise med {col.get('rms_noise_median_px', np.nan):.2f}, q95 {col.get('rms_noise_q95_px', np.nan):.2f})"
              f" | v-perp {r['vp_perp_u']:.1f}+-{r.get('vp_perp_noise_std_u', np.nan):.1f}"
              f" | CR {nm.get('CR', np.nan):.4f} Ztop(ABC) {nm.get('Ztop_from_ABC', np.nan):.1f}+-{mc.get('Ztop_from_ABC', {}).get('std', np.nan):.1f}"
              f" Ztop(AC,v) {nm.get('Ztop_from_AC_vp', np.nan):.1f}+-{mc.get('Ztop_from_AC_vp', {}).get('std', np.nan):.1f}"
              f" gap topA {nm.get('gap_topA_mm_if_BC400', np.nan):.1f}+-{mc.get('gap_topA_mm_if_BC400', {}).get('std', np.nan):.1f}"
              f" gap AB {nm.get('gap_AB_mm_if_BC400', np.nan):.1f}+-{mc.get('gap_AB_mm_if_BC400', {}).get('std', np.nan):.1f}"
              f" | t_v {r['t_v']:.0f} VP(ABC)-VP(edges) along {nm.get('vp_along_ABC_minus_edge', np.nan):.0f}+-{mc.get('vp_along_ABC_minus_edge', {}).get('std', np.nan):.0f}"
              f" VP(all4)-edges {nm.get('vp_along_all4_minus_edge', np.nan):.0f}")
    for var in ("k1k2_centre_fixed", "k1k2_centre_free"):
        print("== end summaries", var)
        for k, s in out[var]["end_summary"].items():
            print(" ", k, {kk: (round(v["mean"], 1), round(v["sd"], 1), v["n"], round(v["chi2"], 1)) for kk, v in s.items() if v})
    print("collinearity residuals per level (distorted px), per variant:")
    for var in variants:
        for r in out[var]["columns"]:
            if len(r["resid_dist"]) >= 3:
                print(f"  {var:18s} {r['cart']:3d}/X{r['end']:<4d} {np.round(r['XY'], 1)}", {k: round(v, 2) for k, v in r["resid_dist"].items()})
    print("variant sensitivity (Ztop from ABC per column):")
    for k, s in sens.items():
        print(" ", k, {v: s[v]["Ztop_from_ABC"] for v in variants})

    # ---- plot ----
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), dpi=110)
    ax = axs[0]
    labels, xs = [], 0
    done = set()
    cols_c = {"Ztop_from_ABC": "C0", "Ztop_from_AC_vp": "C1", "Ztop_from_BC_vp": "C2", "Ztop_from_AB_vp": "C3"}
    for cart in (80, 310):
        for end in (0, 1600):
            for r in [r for r in main_v["columns"] if r["cart"] == cart and r["end"] == end]:
                for i, (k, cc) in enumerate(cols_c.items()):
                    if k in r["nominal"] and "mc" in r and k in r["mc"]:
                        grazing = any(a < 45 for a in r["levels_line"].values()) or r["line_from"] != "points"
                        ax.errorbar(xs + (i - 1.5) * 0.12, r["nominal"][k], yerr=r["mc"][k]["std"], fmt="o", color=cc, ms=4,
                                    mfc="none" if grazing else cc, capsize=2, label=None if k in done else k)
                        done.add(k)
                labels.append(f"{cart}/X{end} ({r['XY'][0]:.0f},{r['XY'][1]:.0f})")
                xs += 1
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=60, ha="right")
    ax.text(0.99, 0.98, "open symbols: column uses side-line intersections at < 45 deg or a line through the VP (unreliable)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_ylabel("implied Z of the top sticker [mm] (drawing: 0)")
    ax.set_title("Column test: implied top-sticker height (shelves at drawing Z)\nerror bars: MC 1-sigma (corner/side noise, nadir)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax = axs[1]
    for i, var in enumerate(variants):
        for j, (k, s) in enumerate(out[var]["variant_sensitivity"].items() if False else sens.items()):
            vals = s[var]["Ztop_from_ABC"]
            ax.plot([j + (i - 1.5) * 0.15] * len(vals), vals, "o", color=f"C{i}", ms=4, label=var if j == 0 else None)
    ax.text(0.01, 0.02, "all columns incl. unreliable ones (see left)", transform=ax.transAxes, fontsize=7)
    ax.set_xticks(range(len(sens)))
    ax.set_xticklabels([k.replace("_", " / X") for k in sens], fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("implied Z top [mm] (cross ratio, shelves at drawing)")
    ax.set_title("Sensitivity to the plumb-line distortion variant (per column, no MC)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_columns.png")
    print("saved")


if __name__ == "__main__":
    main()
