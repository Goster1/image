"""(2) VANISHING POINTS -> f and principal point (independent implementation).

With the plumb-line distortion (pixel units; re-estimated in every bootstrap replicate) all edge points are
undistorted; the undistorted image is an ideal pinhole image (f, pp).  For every direction group the
vanishing point is estimated by maximum likelihood from the edge POINTS (each edge = a line through the VP
with its own free angle, residuals = distances in the distorted image).  f and pp then follow from the
orthogonality of VP pairs (IAC):  (K^-1 v_i) . (K^-1 v_j) = 0.

Direction groups (see 40_lines_indep_lib.vp_group):
  310X / 80X : shelf lips, rails, back members along the cart X axis (rich groups, 6-7 physical members)
  310Z / 80Z : cart posts (2 edges each; 80Z = two short parallel outline pieces) -> CZ = both carts' posts
  WV         : scene verticals (blue column, bollard)
  FA / FC    : floor lines along / across the aisle (white tape, blue line); FJ = floor joint across the aisle
  xxYb, xxXb : end-board edges (NOT parallel to the cart axes to VP precision -> diagnostics only)
Orthogonal pairs used: cart X _|_ cart vertical; floor lines _|_ vertical; FA _|_ FC (tape T-junction).

Uncertainty
  * per-edge direction error: estimated from the redundancy of the rich groups (angular deviation of each
    edge's interpretation plane from its group direction, minus the point-noise part) -> sigma_psi
  * VP covariance: Monte Carlo (point noise + sigma_psi per edge)
  * f, pp: GLS over the pairs with the propagated VP covariance
  * bootstrap (NB replicates): resample physical members (clusters) within the rich strata (310X, 80X, all
    edges not used for VPs); edges of the small groups are kept but rotated by a random direction error
    (sigma_psi + their own noise); the plumb-line distortion is RE-ESTIMATED in every replicate from the
    resampled/perturbed straight edges, then VPs and f, pp.
Writes work/cache/40_lines_indep_vp.json (+ _boot.npz) and results/40_lines_indep_vp_*.png
Usage: python3 40_lines_indep_b_vp.py [NBOOT]
"""
import importlib
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, RESULTS, H, W  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(4002)
T0 = time.time()
F_NOM = 1450.0  # nominal f only for converting VP offsets to angles in the error model (weak dependence)

edges, dropped = L.load_edges_indep()
print("edge files reviewed:", L.file_status())
# same robust rejection as the plumb-line script (k1k2 + free centre)
act, hist, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
print("rejected (curved):", [e["id"] for e, a in zip(edges, act) if not a])
for e in acc:  # merged cart-vertical group
    e["vgroup"] = "CZ" if e["group"] in ("310Z", "80Z") else e["group"]

PLUMB = {
    "k1k2_c": L.PlumbModel("poly", 2, centre_free=True),   # best LOCO-CV of the poly models
    "k1k2": L.PlumbModel("poly", 2),                       # centre fixed at the image centre
    "div2": L.PlumbModel("div", 2),                        # best LOCO-CV overall (centre fixed)
    "div2_c": L.PlumbModel("div", 2, centre_free=True),
    "k1k2k3_c": L.PlumbModel("poly", 3, centre_free=True),
}
MAIN = "k1k2_c"
GROUPS = ["310X", "80X", "310Z", "80Z", "CZ", "WV", "FA", "FC", "FJ", "FCJ", "310Yb", "80Yb", "310Xb", "80Xb"]


def group_members(edge_list, g):
    if g == "CZ":
        return [e for e in edge_list if e["group"] in ("310Z", "80Z")]
    if g == "FCJ":
        return [e for e in edge_list if e["group"] in ("FC", "FJ")]
    return [e for e in edge_list if e["group"] == g]


# ---------------------------------------------------------------------------------------------- VPs
def estimate_vps(edge_list, dist, groups=GROUPS):
    out = {}
    for g in groups:
        sub = group_members(edge_list, g)
        if len(sub) < 2:
            continue
        es = L.EdgeSet(sub)
        U, JU, ok = dist.undistort(es.P)
        v = L.fit_vp_group(U, JU, es)
        v["edges"] = [e["id"] for e in sub]
        out[g] = v
    return out


def psi_of_line(l_px, d):
    """angle (rad) between the interpretation plane of image line l (undistorted px, homogeneous) and direction d."""
    n = KT_NOM @ l_px
    return float(np.arcsin(np.clip(n @ d / np.linalg.norm(n), -1, 1)))


def line_px_from_scaled(l):
    """line in scaled coords ((u-C0)/S) -> pixel coords."""
    a, b, c = l
    return np.array([a / L.S, b / L.S, c - (a * L.C0[0] + b * L.C0[1]) / L.S])


def edge_psi_stats(edge_list, dist, vps, pp):
    """per edge: psi (free line vs group direction), noise part, dpsi/dtheta (for the perturbation model)."""
    global KT_NOM
    K = np.array([[F_NOM, 0, pp[0]], [0, F_NOM, pp[1]], [0, 0, 1.0]])
    KT_NOM = K.T
    rows = {}
    for g, v in vps.items():
        sub = group_members(edge_list, g)
        d = L.ray(v["V"], F_NOM, pp)
        es = L.EdgeSet(sub)
        U, JU, _ = dist.undistort(es.P)
        r, Lf = es.residuals(U, JU, None, return_lines=True)
        st = es.per_edge_stats(r)
        for i, e in enumerate(sub):
            lp = line_px_from_scaled(Lf[i])
            psi = psi_of_line(lp, d)
            # rotate the line about the edge centroid (undistorted) by a small angle -> dpsi/dtheta
            m = es.k == i
            cen = U[m].mean(0)
            th = np.arctan2(-lp[0], lp[1])
            dth = 1e-4
            nn = np.array([-np.sin(th + dth), np.cos(th + dth)])
            lp2 = np.array([nn[0], nn[1], -nn @ cen])
            lp1 = np.array([-np.sin(th), np.cos(th), 0.0])
            lp1[2] = -lp1[:2] @ cen
            dpsi = (psi_of_line(lp2, d) - psi_of_line(lp1, d)) / dth
            Lpx = float(np.hypot(*(U[m][-1] - U[m][0])))
            n = int(m.sum())
            sig_theta = st[i]["noise"] * np.sqrt(12.0 / n) / max(Lpx, 1.0)
            rows.setdefault(g, []).append(dict(id=e["id"], psi_deg=np.degrees(psi), noise_psi_deg=np.degrees(abs(dpsi) * sig_theta),
                                               dpsi_dtheta=float(dpsi), length_px=Lpx, n=n))
    return rows


def sigma_psi_from_rich(rows, groups=("310X", "80X")):
    num, den, per = 0.0, 0, {}
    for g in groups:
        if g not in rows:
            continue
        ps = np.radians([r["psi_deg"] for r in rows[g]])
        nz = np.radians([r["noise_psi_deg"] for r in rows[g]])
        n = len(ps)
        ex = (np.sum(ps ** 2) - np.sum(nz ** 2)) / max(n - 2, 1)
        per[g] = float(np.degrees(np.sqrt(max(ex, 0.0))))
        num += np.sum(ps ** 2) - np.sum(nz ** 2)
        den += n - 2
    return float(np.sqrt(max(num / max(den, 1), 0.0))), per


def perturb_edges(edge_list, psi_rows, sig_psi, rng_, only_groups=None):
    """rotate each edge (distorted points, about their centroid) by a random image angle corresponding to a
    random direction error N(0, sig_psi^2 + own noise^2) of its interpretation plane."""
    lookup = {r["id"]: r for g in psi_rows.values() for r in g}
    out = []
    for e in edge_list:
        if (only_groups is not None and e.get("vgroup") not in only_groups) or e["id"] not in lookup:
            out.append(e)
            continue
        r = lookup[e["id"]]
        s = np.sqrt(sig_psi ** 2 + np.radians(r["noise_psi_deg"]) ** 2)
        dth = rng_.normal(0, s) / r["dpsi_dtheta"] if abs(r["dpsi_dtheta"]) > 1e-9 else 0.0
        P = e["points"]
        c = P.mean(0)
        R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
        out.append(dict(e, points=(P - c) @ R.T + c))
    return out


def vp_cov_mc(edge_list, dist, g, v, psi_rows, sig_psi, n=120, rng_=None):
    sub = group_members(edge_list, g)
    e1, e2 = v["basis"]
    V0 = v["V"]
    T = []
    for _ in range(n):
        pert = perturb_edges(sub, psi_rows, sig_psi, rng_)
        es = L.EdgeSet(pert)
        U, JU, _ = dist.undistort(es.P)
        try:
            w = L.fit_vp_group(U, JU, es)
        except Exception:  # noqa
            continue
        Vs = w["V"] * np.sign(w["V"] @ V0)
        T.append([Vs @ e1, Vs @ e2])
    T = np.array(T)
    return np.cov(T.T)


# ---------------------------------------------------------------------------------------------- f, pp solvers
def cos_pair(Va, Vb, f, pp):
    return float(L.ray(Va, f, pp) @ L.ray(Vb, f, pp))


def gls_solve(vps, covs, pairs, pp_fixed=None, x0=(1450.0, 940.0, 450.0), outer=4):
    """GLS on the orthogonality cosines with the propagated VP covariance. Returns dict."""
    pairs = [p for p in pairs if p[0] in vps and p[1] in vps]
    if not pairs:
        return None
    gl = sorted({g for p in pairs for g in p})
    npar = 1 if pp_fixed is not None else 3

    def unpack(x):
        return (x[0], np.asarray(pp_fixed, float)) if pp_fixed is not None else (x[0], np.array([x[1], x[2]]))

    def cvec(x, Vd=None):
        f, pp = unpack(x)
        Vd = {g: vps[g]["V"] for g in gl} if Vd is None else Vd
        return np.array([cos_pair(Vd[a], Vd[b], f, pp) for a, b in pairs])

    def Cmat(x):
        # dc/d(tangent coords) numerically
        G = np.zeros((len(pairs), 2 * len(gl)))
        base = {g: vps[g]["V"] for g in gl}
        c0 = cvec(x, base)
        for k, g in enumerate(gl):
            e1, e2 = vps[g]["basis"]
            for j, e in enumerate((e1, e2)):
                Vd = dict(base)
                V = base[g] + 1e-6 * e
                Vd[g] = V / np.linalg.norm(V)
                G[:, 2 * k + j] = (cvec(x, Vd) - c0) / 1e-6
        Sg = np.zeros((2 * len(gl), 2 * len(gl)))
        for k, g in enumerate(gl):
            Sg[2 * k:2 * k + 2, 2 * k:2 * k + 2] = covs[g]
        return G @ Sg @ G.T + 1e-14 * np.eye(len(pairs))

    x = np.array(x0[:npar], float)
    for _ in range(outer):
        C = Cmat(x)
        Li = np.linalg.inv(np.linalg.cholesky(C))
        res = least_squares(lambda z: Li @ cvec(z), x, x_scale=np.array([100.0, 30.0, 30.0])[:npar], method="lm" if len(pairs) >= npar else "trf")
        x = res.x
    C = Cmat(x)
    Li = np.linalg.inv(np.linalg.cholesky(C))
    J = np.zeros((len(pairs), npar))
    for i in range(npar):
        h = np.zeros(npar)
        h[i] = 1e-3 * (10 if i == 0 else 1)
        J[:, i] = (Li @ (cvec(x + h) - cvec(x - h))) / (2 * h[i])
    rr = Li @ cvec(x)
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.nan)
    f, pp = unpack(x)
    ang = {f"{a}-{b}": float(np.degrees(np.arccos(np.clip(abs(cos_pair(vps[a]['V'], vps[b]['V'], f, pp)), 0, 1)))) for a, b in pairs}
    return dict(f=float(f), pp=pp.tolist(), cov=cov.tolist(), se=np.sqrt(np.abs(np.diag(cov))).tolist(), chi2=float(rr @ rr),
                dof=len(pairs) - npar, pairs=[f"{a}-{b}" for a, b in pairs], angles_deg=ang)


def iac_linear(vps, pairs, pp_fixed=None):
    """closed-form IAC (fx=fy, zero skew): v_i^T w v_j = 0 linear in (px, py, w3 = px^2+py^2+f^2)."""
    rows, rhs = [], []
    for a, b in pairs:
        if a not in vps or b not in vps:
            continue
        va, vb = L.from_scaled_h(vps[a]["V"]), L.from_scaled_h(vps[b]["V"])
        va = va / np.linalg.norm(va)
        vb = vb / np.linalg.norm(vb)
        s = va[0] * vb[0] + va[1] * vb[1]
        cx_ = va[0] * vb[2] + va[2] * vb[0]
        cy_ = va[1] * vb[2] + va[2] * vb[1]
        z = va[2] * vb[2]
        rows.append([-cx_, -cy_, z])
        rhs.append(-s)
    A, bvec = np.array(rows), np.array(rhs)
    if pp_fixed is not None:
        px, py = pp_fixed
        w3 = np.linalg.lstsq(A[:, 2:], bvec - A[:, 0] * px - A[:, 1] * py, rcond=None)[0][0]
    else:
        if len(A) < 3:
            return None
        # row scaling to unit norm
        nr = np.linalg.norm(A, axis=1)
        sol = np.linalg.lstsq(A / nr[:, None], bvec / nr, rcond=None)[0]
        px, py, w3 = sol
    f2 = w3 - px * px - py * py
    return dict(f=float(np.sqrt(f2)) if f2 > 0 else float("nan"), pp=[float(px), float(py)])


PAIRSETS = {
    "S1_cartX_cartZ": [("310X", "CZ"), ("80X", "CZ")],
    "S1_310only": [("310X", "310Z")],
    "S1w_cartX_WV": [("310X", "WV"), ("80X", "WV")],
    "S2_floor_CZ": [("310X", "CZ"), ("80X", "CZ"), ("FA", "CZ"), ("FC", "CZ"), ("FA", "FC")],
    "S2_floor_WV": [("310X", "CZ"), ("80X", "CZ"), ("FA", "WV"), ("FC", "WV"), ("FA", "FC")],
    "S2j_floorjoint_CZ": [("310X", "CZ"), ("80X", "CZ"), ("FA", "CZ"), ("FCJ", "CZ"), ("FA", "FCJ")],
    "S3_boards": [("310X", "CZ"), ("80X", "CZ"), ("310Yb", "310X"), ("310Yb", "CZ"), ("80Yb", "80X"), ("80Yb", "CZ")],
    "S4_all": [("310X", "CZ"), ("80X", "CZ"), ("FA", "CZ"), ("FC", "CZ"), ("FA", "FC"), ("310Yb", "310X"), ("310Yb", "CZ"),
               ("80Yb", "80X"), ("80Yb", "CZ")],
}


def solve_all(vps, covs, dist_centre):
    out = {}
    for nm, pairs in PAIRSETS.items():
        for ppname, ppv in [("ppC0", L.C0), ("ppDist", dist_centre), ("ppFree", None)]:
            if ppname == "ppDist" and np.allclose(dist_centre, L.C0):
                continue
            if ppname == "ppFree" and nm in ("S1_cartX_cartZ", "S1_310only", "S1w_cartX_WV"):
                continue  # under-determined (pp moves along the principal vertical with f)
            try:
                r = gls_solve(vps, covs, pairs, pp_fixed=ppv)
            except Exception as ex:  # noqa
                r = dict(error=str(ex))
            if r is not None:
                r["iac_closed_form"] = iac_linear(vps, pairs, pp_fixed=ppv)
                out[f"{nm}|{ppname}"] = r
    return out


def f_of_ppy(vps, covs, ppx, ppys, pairs=PAIRSETS["S1_cartX_cartZ"]):
    return np.array([gls_solve(vps, covs, pairs, pp_fixed=(ppx, py))["f"] for py in ppys])


def horizon_nadir(vps, pp, horiz=("310X", "80X")):
    """horizon line through the horizontal VPs (px), nadir = CZ VP; f^2 = d(pp,N) d(pp,H)."""
    N = L.from_scaled_h(vps["CZ"]["V"])
    N = N[:2] / N[2]
    P = []
    for g in horiz:
        if g in vps:
            v = L.from_scaled_h(vps[g]["V"])
            P.append(v[:2] / v[2])
    P = np.array(P)
    if len(P) < 2:
        return None
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    dvec = vt[0]
    nrm = np.array([-dvec[1], dvec[0]])
    dH = abs((pp - c) @ nrm)
    dN = np.hypot(*(N - pp))
    ang = np.degrees(np.arccos(abs(nrm @ (N - pp)) / dN))
    # foot of the perpendicular from pp on the horizon vs the line pp->N
    return dict(nadir=N.tolist(), horizon_point=c.tolist(), horizon_dir=dvec.tolist(), d_pp_nadir=float(dN), d_pp_horizon=float(dH),
                f_from_product=float(np.sqrt(dN * dH)), angle_horizon_normal_vs_nadir_dir_deg=float(ang), horizontal_vps=list(horiz))


# ---------------------------------------------------------------------------------------------- nominal run
results = {}
for pn, pm in PLUMB.items():
    fit = L.fit_plumb(L.EdgeSet(acc), pm)
    dist = fit["dist"]
    vps = estimate_vps(acc, dist)
    results[pn] = dict(fit=fit, dist=dist, vps=vps)
    print(f"\nplumb {pn}: x={np.round(fit['x'], 5)} centre {np.round(dist.c, 1)}")

# error model from the main plumb variant
dmain = results[MAIN]["dist"]
vmain = results[MAIN]["vps"]
psi_rows = edge_psi_stats(acc, dmain, vmain, dmain.c)
SIG_PSI, sig_per = sigma_psi_from_rich(psi_rows)
print(f"\nper-edge direction error (excess over point noise) from rich groups: sigma_psi = {np.degrees(SIG_PSI):.4f} deg; per group {sig_per}")
for g in ["310X", "80X", "CZ", "WV", "FA", "FC"]:
    if g in psi_rows:
        print(f"  {g}: psi [deg] " + ", ".join(f"{r['id'].split('_', 1)[1]}={r['psi_deg']:+.3f}(n{r['noise_psi_deg']:.3f})" for r in psi_rows[g]))

summary = {}
for pn in PLUMB:
    R = results[pn]
    dist, vps = R["dist"], R["vps"]
    covs = {g: vp_cov_mc(acc, dist, g, v, psi_rows, SIG_PSI, n=100, rng_=rng) for g, v in vps.items()}
    R["covs"] = covs
    sol = solve_all(vps, covs, dist.c)
    hz = {nm: horizon_nadir(vps, pp) for nm, pp in [("ppC0", L.C0), ("ppDist", dist.c)]}
    hz3 = {nm: horizon_nadir(vps, pp, ("310X", "80X", "FA")) for nm, pp in [("ppC0", L.C0), ("ppDist", dist.c)]}
    vp_tab = {}
    for g, v in vps.items():
        vpx = v["V_px"]
        C = covs[g]
        # 1-sigma ellipse of the VP in px (if finite) via the tangent basis
        e1, e2 = v["basis"]
        J = np.zeros((2, 2))
        for j, e in enumerate((e1, e2)):
            V2 = v["V"] + 1e-7 * e
            p2 = L.from_scaled_h(V2)
            J[:, j] = (p2[:2] / p2[2] - vpx[:2] / vpx[2]) / 1e-7
        Cpx = J @ C @ J.T
        vp_tab[g] = dict(vp_px=(vpx[:2] / vpx[2]).tolist(), cov_px=Cpx.tolist(), sd_px=np.sqrt(np.diag(Cpx)).tolist(),
                         n_edges=v["n_edges"], n_clusters=v["n_clusters"], rms_constrained=v["rms"], rms_free=v["rms_free"],
                         per_edge_rms=[s["rms"] for s in v["per_edge"]], edges=v["edges"])
    summary[pn] = dict(x=R["fit"]["x"].tolist(), names=R["fit"]["names"], centre=dist.c.tolist(), vps=vp_tab, solutions=sol,
                       horizon_nadir=hz, horizon_nadir_withFA=hz3)
    print(f"\n=== plumb {pn} (centre {np.round(dist.c, 1)})")
    for g, t in vp_tab.items():
        print(f"  VP {g:6s} ({t['vp_px'][0]:9.1f},{t['vp_px'][1]:9.1f}) sd ({t['sd_px'][0]:.0f},{t['sd_px'][1]:.0f})  "
              f"edges {t['n_edges']} rms {t['rms_constrained']:.3f} (free {t['rms_free']:.3f})")
    for k, r in sol.items():
        if "error" in r:
            print("  ", k, r["error"])
            continue
        print(f"  {k:28s} f={r['f']:7.1f}±{r['se'][0]:5.1f} pp=({r['pp'][0]:6.1f},{r['pp'][1]:6.1f})"
              + (f"±({r['se'][1]:.1f},{r['se'][2]:.1f})" if len(r['se']) > 1 else "") + f" chi2/dof={r['chi2']:.1f}/{r['dof']}"
              + f"  IAC-closed f={r['iac_closed_form']['f'] if r['iac_closed_form'] else float('nan'):.1f}")
    for nm, h in hz.items():
        print(f"  horizon/nadir {nm}: d(pp,N)={h['d_pp_nadir']:.1f} d(pp,H)={h['d_pp_horizon']:.1f} -> f={h['f_from_product']:.1f}; "
              f"horizon normal vs pp->nadir {h['angle_horizon_normal_vs_nadir_dir_deg']:.1f} deg")

# vertical coincidence: 310Z, 80Z-lines, WV, CZ
vtest = {}
for pn in [MAIN, "k1k2"]:
    vps = results[pn]["vps"]
    dist = results[pn]["dist"]
    pp = dist.c
    f_use = summary[pn]["solutions"].get("S1_cartX_cartZ|ppDist", summary[pn]["solutions"].get("S1_cartX_cartZ|ppC0"))["f"]
    d = {g: L.ray(vps[g]["V"], f_use, pp) for g in ["310Z", "CZ", "WV"] if g in vps}
    ang = {f"{a}-{b}": float(np.degrees(np.arccos(np.clip(abs(d[a] @ d[b]), 0, 1)))) for a, b in [("310Z", "WV"), ("CZ", "WV"), ("310Z", "CZ")] if a in d and b in d}
    # every vertical edge: angle between its interpretation plane and the CZ / WV direction
    Kt = np.array([[f_use, 0, pp[0]], [0, f_use, pp[1]], [0, 0, 1.0]]).T
    per = {}
    for g in ["310Z", "80Z", "WV"]:
        sub = group_members(acc, g)
        es = L.EdgeSet(sub)
        U, JU, _ = dist.undistort(es.P)
        r, Lf = es.residuals(U, JU, None, return_lines=True)
        for i, e in enumerate(sub):
            n = Kt @ line_px_from_scaled(Lf[i])
            n /= np.linalg.norm(n)
            per[e["id"]] = {k: float(np.degrees(np.arcsin(n @ d[k]))) for k in d}
    vtest[pn] = dict(f_used=f_use, angles_deg=ang, per_edge_psi_deg=per)
    print(f"\nverticals ({pn}, f={f_use:.0f}): angles {ang}")
    for k, v in per.items():
        print("   ", k, {a: round(b, 3) for a, b in v.items()})

# f as a function of pp_y (S1 pairs), for the main plumb
ppys = np.arange(340, 661, 10.0)
curve = {}
for pn in [MAIN, "k1k2", "div2"]:
    R = results[pn]
    ppx = R["dist"].c[0] if pn != "k1k2" else summary[MAIN]["centre"][0]
    curve[pn] = dict(ppx=float(ppx), f=f_of_ppy(R["vps"], R["covs"], ppx, ppys).tolist())
print("\nf(pp_y) [S1, main plumb]:", {int(y): round(f, 1) for y, f in zip(ppys[::4], curve[MAIN]["f"][::4])})


# ---------------------------------------------------------------------------------------------- bootstrap
def strata_resample(edge_list, rng_):
    """resample clusters within rich strata; small VP groups kept (perturbed separately)."""
    small = {"CZ", "WV", "FA", "FC"}
    out = [e for e in edge_list if e["vgroup"] in small]
    for st in ["310X", "80X", "rest"]:
        pool = [e for e in edge_list if (e["vgroup"] == st if st != "rest" else (e["vgroup"] not in small and e["vgroup"] not in ("310X", "80X")))]
        names = sorted({e["cluster"] for e in pool})
        pick = rng_.integers(0, len(names), len(names))
        for q, c in enumerate(pick):
            out += [dict(e, cluster=f"{e['cluster']}#{st}{q}") for e in pool if e["cluster"] == names[c]]
    return out


BOOT_PLUMB = [MAIN, "k1k2", "div2"]
KEYS = ["S1_cartX_cartZ|ppC0", "S1_cartX_cartZ|ppDist", "S1_310only|ppC0", "S1_310only|ppDist", "S1w_cartX_WV|ppC0", "S1w_cartX_WV|ppDist",
        "S2_floor_CZ|ppFree", "S2_floor_WV|ppFree", "S2_floor_CZ|ppC0", "S2_floor_CZ|ppDist", "S3_boards|ppFree", "S4_all|ppFree", "S2j_floorjoint_CZ|ppFree"]
boot = {pn: dict(x=[], sol={k: [] for k in KEYS}, vp={g: [] for g in ["310X", "80X", "CZ", "WV", "FA", "FC", "310Z"]}, fcurve=[], vang=[]) for pn in BOOT_PLUMB}
print(f"\nbootstrap {NB} replicates ...")
tb = time.time()
for b in range(NB):
    sub = strata_resample(acc, rng)
    sub = perturb_edges(sub, psi_rows, SIG_PSI, rng, only_groups={"CZ", "WV", "FA", "FC"})
    for pn in BOOT_PLUMB:
        try:
            fit = L.fit_plumb(L.EdgeSet(sub), PLUMB[pn], x0=results[pn]["fit"]["x"], max_nfev=100)
            dist = fit["dist"]
            vps = estimate_vps(sub, dist, groups=["310X", "80X", "310Z", "CZ", "WV", "FA", "FC", "FCJ", "FJ", "310Yb", "80Yb"])
            # GLS weights from the nominal run, rotated into the replicate's tangent bases
            covs = {}
            for g, v in vps.items():
                if g not in results[pn]["covs"]:
                    continue
                b0 = results[pn]["vps"][g]["basis"]
                sgn = np.sign(v["V"] @ results[pn]["vps"][g]["V"])
                Tm = np.array([[sgn * (v["basis"][i] @ b0[j]) for j in range(2)] for i in range(2)])
                covs[g] = Tm @ results[pn]["covs"][g] @ Tm.T
            sol = solve_all(vps, covs, dist.c)
            B = boot[pn]
            B["x"].append(fit["x"])
            for k in KEYS:
                r = sol.get(k)
                B["sol"][k].append([r["f"]] + r["pp"] if r and "f" in r else [np.nan] * 3)
            for g in B["vp"]:
                vpx = vps[g]["V_px"] if g in vps else np.full(3, np.nan)
                B["vp"][g].append(vpx[:2] / vpx[2])
            if pn == MAIN:
                B["fcurve"].append(f_of_ppy(vps, covs, dist.c[0], ppys[::2]))
            fuse = sol["S1_cartX_cartZ|ppDist" if "S1_cartX_cartZ|ppDist" in sol else "S1_cartX_cartZ|ppC0"]["f"]
            d = {g: L.ray(vps[g]["V"], fuse, dist.c) for g in ["310Z", "CZ", "WV"]}
            B["vang"].append([np.degrees(np.arccos(min(1, abs(d["CZ"] @ d["WV"])))), np.degrees(np.arccos(min(1, abs(d["310Z"] @ d["WV"]))))])
        except Exception as ex:  # noqa
            print("  replicate failed", pn, ex)
    if b % 25 == 0:
        print(f"  {b} ({time.time() - tb:.0f}s)")


def rstat(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return None
    a = a[np.all(np.isfinite(a), axis=1)] if a.ndim == 2 else a[np.isfinite(a)]
    if len(a) < 2:
        return None
    q = np.percentile(a, [15.87, 50, 84.13], axis=0)
    return dict(mean=np.mean(a, 0).tolist(), sd=np.std(a, 0, ddof=1).tolist(), median=q[1].tolist(), sd_robust=((q[2] - q[0]) / 2).tolist(), n=int(len(a)))


boot_summary = {}
for pn in BOOT_PLUMB:
    B = boot[pn]
    bs = dict(plumb_x=rstat(B["x"]), solutions={k: rstat(v) for k, v in B["sol"].items()}, vp={g: rstat(v) for g, v in B["vp"].items()},
              vertical_angles_deg=dict(zip(["CZ-WV", "310Z-WV"], [rstat(np.array(B["vang"])[:, i]) for i in range(2)])))
    if pn == MAIN:
        fc = np.array(B["fcurve"])
        bs["f_of_ppy"] = dict(ppy=ppys[::2].tolist(), median=np.nanmedian(fc, 0).tolist(), lo=np.nanpercentile(fc, 15.87, 0).tolist(),
                              hi=np.nanpercentile(fc, 84.13, 0).tolist())
    boot_summary[pn] = bs
    print(f"\nbootstrap summary plumb {pn}: centre/coeffs median {np.round(bs['plumb_x']['median'], 4)} sd_rob {np.round(bs['plumb_x']['sd_robust'], 4)}")
    for k in KEYS:
        s = bs["solutions"][k]
        if s is None:
            continue
        print(f"   {k:28s} f med {s['median'][0]:7.1f} sd {s['sd'][0]:6.1f} (rob {s['sd_robust'][0]:6.1f})  pp med ({s['median'][1]:.0f},{s['median'][2]:.0f})"
              f" sd_rob ({s['sd_robust'][1]:.1f},{s['sd_robust'][2]:.1f}) n={s['n']}")
    print("   vertical angles", {k: (round(v["median"], 3), round(v["sd_robust"], 3)) for k, v in bs["vertical_angles_deg"].items() if v})

# ---------------------------------------------------------------------------------------------- plots
img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
R = results[MAIN]
dist = R["dist"]
colors = {"310X": "tab:red", "80X": "tab:blue", "CZ": "tab:green", "WV": "tab:purple", "FA": "tab:orange", "FC": "tab:olive",
          "FJ": "tab:brown", "310Yb": "tab:pink", "80Yb": "tab:cyan", "310Xb": "k", "80Xb": "k"}
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
# (left) undistorted image domain: extended edge lines towards the VPs
a = ax[0]
for g in ["310X", "80X", "CZ", "WV", "FA", "FC", "FJ", "310Yb", "80Yb"]:
    sub = group_members(acc, g)
    if not sub:
        continue
    es = L.EdgeSet(sub)
    U, JU, _ = dist.undistort(es.P)
    for i in range(es.E):
        Q = U[es.k == i]
        a.plot(Q[:, 0], Q[:, 1], "-", color=colors[g], lw=2)
    if g in R["vps"]:
        vpx = R["vps"][g]["V_px"]
        if abs(vpx[2]) > 1e-9:
            v = vpx[:2] / vpx[2]
            for i in range(es.E):
                c = U[es.k == i].mean(0)
                a.plot([c[0], v[0]], [c[1], v[1]], ":", color=colors[g], lw=0.6)
            a.plot(*v, "o", color=colors[g], ms=7, label=f"{g} VP ({v[0]:.0f},{v[1]:.0f})")
a.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "k-", lw=1)
a.plot(*dist.c, "m+", ms=14, mew=2, label="distortion centre")
a.set_xlim(-3000, 5000)
a.set_ylim(8000, -600)
a.set_aspect("equal")
a.legend(fontsize=7, loc="lower left")
a.set_title(f"undistorted edges (plumb {MAIN}) and their vanishing points")
# (right) f(pp_y) curve with bootstrap band + distortion-centre estimates
a = ax[1]
bs = boot_summary[MAIN]["f_of_ppy"]
a.fill_between(bs["ppy"], bs["lo"], bs["hi"], color="tab:blue", alpha=0.25, label="S1 (cart X _|_ cart Z), bootstrap 68 %")
a.plot(ppys, curve[MAIN]["f"], "b-", label=f"S1 GLS f(pp_y), pp_x={curve[MAIN]['ppx']:.0f} (plumb {MAIN})")
a.plot(ppys, curve["k1k2"]["f"], "c--", label="same, plumb k1k2 centre fixed")
a.plot(ppys, curve["div2"]["f"], "g--", label="same, plumb div2 centre fixed")
for pn, col in [("k1k2_c", "m"), ("div2_c", "tab:orange"), ("k1k2k3_c", "tab:brown")]:
    cy = summary[pn]["centre"][1]
    a.axvline(cy, color=col, lw=1, ls="-.", label=f"distortion centre y ({pn}) = {cy:.0f}")
a.axvline(L.C0[1], color="k", lw=1, ls=":", label="image centre y")
for k, mk in [("S2_floor_CZ|ppFree", "s"), ("S2_floor_WV|ppFree", "^"), ("S3_boards|ppFree", "D"), ("S4_all|ppFree", "*")]:
    s = summary[MAIN]["solutions"].get(k)
    if s and "f" in s:
        a.errorbar(s["pp"][1], s["f"], xerr=s["se"][2], yerr=s["se"][0], fmt=mk, ms=7, capsize=3, label=f"pp free: {k.split('|')[0]}")
a.set_xlabel("principal point y [px]")
a.set_ylabel("f [px]")
a.set_ylim(1100, 1900)
a.grid(alpha=0.3)
a.legend(fontsize=7)
a.set_title("f vs principal-point height (vanishing points only)")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_vp.png", dpi=110)
plt.close()

# bootstrap scatter
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for k, col in [("S2_floor_CZ|ppFree", "tab:red"), ("S2_floor_WV|ppFree", "tab:blue"), ("S3_boards|ppFree", "tab:green"), ("S4_all|ppFree", "k")]:
    arr = np.array(boot[MAIN]["sol"][k], float)
    ax[0].plot(arr[:, 2], arr[:, 0], ".", ms=3, color=col, alpha=0.5, label=k)
    ax[1].plot(arr[:, 1], arr[:, 2], ".", ms=3, color=col, alpha=0.5, label=k)
ax[0].set_xlabel("pp_y")
ax[0].set_ylabel("f")
ax[0].set_xlim(0, 1000)
ax[0].set_ylim(800, 2400)
ax[1].set_xlabel("pp_x")
ax[1].set_ylabel("pp_y")
ax[1].set_xlim(600, 1300)
ax[1].set_ylim(0, 1000)
for a in ax[:2]:
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
ax[0].set_title("bootstrap: pp free solutions")
for k, col in [("S1_cartX_cartZ|ppC0", "tab:red"), ("S1_cartX_cartZ|ppDist", "tab:blue"), ("S1w_cartX_WV|ppDist", "tab:purple"), ("S1_310only|ppDist", "tab:orange")]:
    arr = np.array(boot[MAIN]["sol"][k], float)[:, 0]
    ax[2].hist(arr[np.isfinite(arr)], bins=40, histtype="step", color=col, label=k)
ax[2].set_xlabel("f [px]")
ax[2].legend(fontsize=7)
ax[2].set_title("bootstrap f (pp fixed at image centre / distortion centre)")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_vp_boot.png", dpi=110)
plt.close()

out = dict(edge_files_reviewed=L.file_status(), rejected=[e["id"] for e, a in zip(edges, act) if not a], n_edges=len(acc),
           sigma_psi_deg=float(np.degrees(SIG_PSI)), sigma_psi_per_group_deg=sig_per, psi_rows=psi_rows, main_plumb=MAIN,
           plumb_variants=summary, vertical_test=vtest, f_of_ppy=dict(ppy=ppys.tolist(), curves=curve), bootstrap=boot_summary, n_boot=NB,
           pairsets={k: [f"{a}-{b}" for a, b in v] for k, v in PAIRSETS.items()})
L.save(L.jsonable(out), "40_lines_indep_vp.json")
np.savez(f"{CACHE}/40_lines_indep_vp_boot.npz", **{f"{pn}__{k}": np.array(v, float) for pn in BOOT_PLUMB for k, v in boot[pn]["sol"].items()},
         **{f"{pn}__plumbx": np.array(boot[pn]["x"], float) for pn in BOOT_PLUMB})
print(f"done in {time.time() - T0:.0f}s")
