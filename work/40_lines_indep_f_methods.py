"""Assemble the lens-JSON method files of the independent line pipeline:
  work/cache/method_plumbline.json / .md   (distortion from straightness; OpenCV k's with f from the VP result)
  work/cache/method_vanishing.json / .md   (f, pp from vanishing points; distortion from the plumb-line)
  work/cache/method_lines_joint.json / .md (joint line self-calibration)
Inputs: work/cache/40_lines_indep_{plumb,vp,joint,synth,compare}.json and the bootstrap .npz files
(run 40_lines_indep_a..e first).  Mapping uncertainty: evaltools (rotation-compensated), from bootstrap samples.
Also writes results/40_lines_indep_summary.png (all variants).
"""
import importlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS, K_from, lens_json, load_json, save_json  # noqa: E402
from evaltools import grid, mapping_displacement, mapping_stats  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
S = L.S
A = load_json(f"{CACHE}/40_lines_indep_plumb.json")
B = load_json(f"{CACHE}/40_lines_indep_vp.json")
C = load_json(f"{CACHE}/40_lines_indep_joint.json")
SY = load_json(f"{CACHE}/40_lines_indep_synth.json") if os.path.exists(f"{CACHE}/40_lines_indep_synth.json") else None
CMP = load_json(f"{CACHE}/40_lines_indep_compare.json") if os.path.exists(f"{CACHE}/40_lines_indep_compare.json") else None
PS = load_json(f"{CACHE}/40_lines_indep_ppscan.json") if os.path.exists(f"{CACHE}/40_lines_indep_ppscan.json") else None
SE = load_json(f"{CACHE}/40_lines_indep_sensitivity.json") if os.path.exists(f"{CACHE}/40_lines_indep_sensitivity.json") else None
bA = np.load(f"{CACHE}/40_lines_indep_plumb_boot.npz")
bB = np.load(f"{CACHE}/40_lines_indep_vp_boot.npz")
bC = np.load(f"{CACHE}/40_lines_indep_joint_boot.npz")
uv, _ = grid(40)


def kvec(a, f):
    a = list(a) + [0, 0, 0]
    return np.array([a[0] * f ** 2 / S ** 2, a[1] * f ** 4 / S ** 4, 0.0, 0.0, a[2] * f ** 6 / S ** 6])


def mapping_unc(K0, d0, samples):
    disp = [mapping_displacement(K0, d0, K1, d1, uv=uv, compensate=True) for K1, d1 in samples]
    ms, _ = mapping_stats(np.array(disp), uv)
    return ms


def mapping_unc_robust(K0, d0, samples):
    """per grid point the 68.3 % quantile of |displacement| over the samples; median over each region."""
    from evaltools import region_defs

    disp = np.array([mapping_displacement(K0, d0, K1, d1, uv=uv, compensate=True) for K1, d1 in samples])
    q = np.percentile(np.hypot(disp[..., 0], disp[..., 1]), 68.27, axis=0)
    return {k: float(np.median(q[fn(uv)])) for k, fn in region_defs().items()}


def reg_summary(ms):
    return {k: round(v["rms_median_px"], 3) for k, v in ms.items() if k != "whole_image"}


def rsd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [15.87, 84.13])
    return float(np.std(x, ddof=1)), float((q[1] - q[0]) / 2)


status = A["edge_files_reviewed"]
data_used = (f"{B['n_edges']} verified straight_3d edges (mean of the 7 jitter-corrected stills; files edges_cart310/cart80/scene.json, "
             f"reviewed={all(status.values())}); duplicates removed; rejected as curved: {', '.join(A['rejection']['rejected'])}")

# ============================================================ VP result (needed for the plumb conversion)
vpm = B["plumb_variants"]["k1k2_c"]
sol = vpm["solutions"]["S1_cartX_cartZ|ppDist"]
f_vp = sol["f"]
pp_vp = np.array(sol["pp"])
a_c = np.array(vpm["x"][2:4])
fb = bB["k1k2_c__S1_cartX_cartZ|ppDist"][:, 0]
px = bB["k1k2_c__plumbx"]
ok = np.isfinite(fb)
fb, px = fb[ok], px[ok]
f_sd, f_sdr = rsd(fb)

# ============================================================ method_plumbline
m = A["models"]["k1k2_c"]
cen = np.array(m["x"][:2])
a_pl = np.array(m["x"][2:4])
kpl = kvec(a_pl, f_vp)
K_pl = K_from(f_vp, f_vp, *cen)
# samples: (i) distortion only, f fixed = f_vp (plumb bootstrap of script a); (ii) with f from the VP bootstrap (script b)
xa = bA["k1k2_c"]
samp_i = [(K_from(f_vp, f_vp, x[0], x[1]), kvec(x[2:4], f_vp)) for x in xa]
samp_ii = [(K_from(f, f, x[0], x[1]), kvec(x[2:4], f)) for x, f in zip(px, fb)]
mu_i = mapping_unc(K_pl, kpl, samp_i)
mu_ii = mapping_unc(K_pl, kpl, samp_ii)
mu_ii_rob = mapping_unc_robust(K_pl, kpl, samp_ii)
k1s_i = np.array([s[1][0] for s in samp_i])
k2s_i = np.array([s[1][1] for s in samp_i])
k1s_ii = np.array([s[1][0] for s in samp_ii])
k2s_ii = np.array([s[1][1] for s in samp_ii])
inv_a = xa[:, 2] / S ** 2
inv_b = xa[:, 3] / S ** 4
alts = {}
for nm, t in A["models"].items():
    alts[nm] = dict(params=dict(zip(t["names"], t["x"])), se_cluster_sandwich=dict(zip(t["names"], t["se_cluster"])),
                    boot_sd=dict(zip(t["names"], t.get("boot_sd", []))) if "boot_sd" in t else None, rms_px=t["rms"],
                    loco_cv_rms_px=t["loco_cv_rms"], at_fold_barrier=t.get("at_barrier"), centre=t["centre"],
                    opencv_k_at_f_vp=(kvec(t["x"][(2 if "cx" in t["names"] else 0):][: len([n for n in t["names"] if n.startswith("a")])], f_vp).tolist()
                                      if "k_at_f1300" in t else None), invariants=t.get("invariants"))
regions = {k: dict(params=dict(zip(v["names"], v["x"])), boot_sd_robust=dict(zip(v["names"], v["boot_sd_robust"])), n_edges=v["n_edges"],
                   n_clusters=v["n_clusters"], rms=v["rms"]) for k, v in A["regions"].items()}
boot_c = A["models"]["k1k2_c"]
plumb = lens_json(
    K_pl, kpl,
    method="plumb_line (independent implementation 40_lines_indep; distortion from the straightness of verified straight edges)",
    model="k1,k2 radial (pixel units: k1/f^2, k2/f^4), distortion centre FREE (= principal point in the OpenCV form); f taken from the "
          "vanishing-point result (S1, pp = this centre) because straightness alone determines only k_i/f^(2i)",
    rms_reprojection_error_px=m["rms"],
    uncertainty={
        "k1_1sigma": rsd(k1s_ii)[0], "k2_1sigma": rsd(k2s_ii)[0],
        "k1_1sigma_f_fixed": rsd(k1s_i)[0], "k2_1sigma_f_fixed": rsd(k2s_i)[0],
        "k1_over_f2_px-2": float(a_pl[0] / S ** 2), "k1_over_f2_1sigma": rsd(inv_a)[0],
        "k2_over_f4_px-4": float(a_pl[1] / S ** 4), "k2_over_f4_1sigma": rsd(inv_b)[0],
        "cx_1sigma": boot_c["boot_sd"][0], "cy_1sigma": boot_c["boot_sd"][1],
        "fx_px_1sigma": f_sd, "fy_px_1sigma": f_sd,
        "how": "cluster bootstrap over physical edge members (resampled with replacement, 300 replicates, distortion re-fitted); "
               "k1,k2 with the f samples of the vanishing-point bootstrap (same replicates re-estimate plumb + VPs); f is NOT from straightness",
    },
    mapping_uncertainty_px=reg_summary(mu_i),
    data_used=data_used,
    geometry_assumptions="no dimensions and no sticker positions used; only: traced edges are straight 3D lines",
    details=dict(
        rms_meaning="rms of the distances of edge points to their own best line, measured in the distorted image (px)",
        mapping_uncertainty_meaning="main value: distortion-only (f fixed at the VP value); 'mapping_uncertainty_with_f' includes the f "
                                    "uncertainty of the VP method",
        mapping_uncertainty_full=mu_i, mapping_uncertainty_with_f=reg_summary(mu_ii), mapping_uncertainty_with_f_full=mu_ii,
        mapping_uncertainty_with_f_robust_q68=mu_ii_rob,
        f_used=f_vp, pixel_units_scale_S=S,
        coefficients_pixel_units=dict(a1=float(a_pl[0]), a2=float(a_pl[1]), note="a_i = k_i S^(2i)/f^(2i), S = 1000 px"),
        distortion_centre=dict(value=cen.tolist(), boot_sd=boot_c["boot_sd"][:2], boot_sd_robust=boot_c["boot_sd_robust"][:2],
                               sandwich_se=boot_c["se_cluster"][:2], corr_boot=boot_c["boot_corr"]),
        model_selection="leave-one-cluster-out CV of the straightness rms (px): " + ", ".join(f"{k} {v['loco_cv_rms']:.4f}" for k, v in A["models"].items()),
        models=alts, regions=regions, rejection=A["rejection"],
        comparison_orchestrator_linecal=(dict(ours=CMP.get("ours_plumb"), linecal=CMP.get("linecal_plumb")) if CMP else None),
        synthetic_round_trip=(dict((k, v) for k, v in SY["summary"].items() if k.startswith("plumb")) if SY else None),
        sensitivity=SE,
    ),
)
save_json(L.jsonable(plumb), f"{CACHE}/method_plumbline.json")

# ============================================================ method_vanishing
K_vp = K_from(f_vp, f_vp, *pp_vp)
k_vp = kvec(a_c, f_vp)
samp_vp = [(K_from(f, f, x[0], x[1]), kvec(x[2:4], f)) for x, f in zip(px, fb)]
mu_vp = mapping_unc(K_vp, k_vp, samp_vp)
mu_vp_rob = mapping_unc_robust(K_vp, k_vp, samp_vp)
cxs, cys = px[:, 0], px[:, 1]
k1v = np.array([s[1][0] for s in samp_vp])
k2v = np.array([s[1][1] for s in samp_vp])
bsum = B["bootstrap"]
alt_vp = {}
for pn in B["plumb_variants"]:
    for k, s in B["plumb_variants"][pn]["solutions"].items():
        if "f" not in s:
            continue
        bs = bsum.get(pn, {}).get("solutions", {}).get(k) if pn in bsum else None
        alt_vp[f"{pn}:{k}"] = dict(f=s["f"], pp=s["pp"], se_gls=s["se"], chi2=s["chi2"], dof=s["dof"], iac_closed_form=s.get("iac_closed_form"),
                                   boot_median=bs["median"] if bs else None, boot_sd=bs["sd"] if bs else None, boot_sd_robust=bs["sd_robust"] if bs else None)
fc0 = bB["k1k2__S1_cartX_cartZ|ppC0"][:, 0]
vanish = lens_json(
    K_vp, k_vp,
    method="vanishing_points (independent implementation 40_lines_indep; VP MLE per direction group, f from orthogonality)",
    model="fx=fy, zero skew; pp = distortion centre of the plumb-line fit (k1,k2, centre free); f from cart-X _|_ cart-vertical of both "
          "carts (common vertical from the posts of both carts); pp NOT determinable from the vanishing points alone",
    rms_reprojection_error_px=float(np.sqrt(np.mean(np.square([vpm["vps"][g]["rms_constrained"] for g in ("310X", "80X", "CZ")])))),
    uncertainty={
        "fx_px_1sigma": f_sd, "fy_px_1sigma": f_sd, "f_1sigma_robust": f_sdr,
        "cx_1sigma": rsd(cxs)[0], "cy_1sigma": rsd(cys)[0], "k1_1sigma": rsd(k1v)[0], "k2_1sigma": rsd(k2v)[0],
        "f_systematic_frames_assumption": abs(f_vp - C["variants"]["JRE_k1k2_framesfree"]["f"]),
        "how": "stratified bootstrap (300): cart-X members and all non-VP edges resampled by cluster, members of the small groups (posts, "
               "scene verticals) rotated by a random direction error (sigma_psi from the redundancy of the cart-X groups); plumb-line "
               "distortion AND centre re-estimated in every replicate, then VPs and f",
    },
    mapping_uncertainty_px=reg_summary(mu_vp),
    data_used=data_used + "; VP groups: 310X (9 edges / 7 members), 80X (7/6), cart posts 310Z+80Z (4), scene verticals (5, not used for f)",
    geometry_assumptions="no dimensions; cart X axes perpendicular to the cart posts; both carts' posts parallel (upright on one floor)",
    details=dict(
        mapping_uncertainty_full=mu_vp, mapping_uncertainty_robust_q68=mu_vp_rob,
        vp=vpm["vps"], sigma_psi_deg=B["sigma_psi_deg"], sigma_psi_per_group_deg=B["sigma_psi_per_group_deg"],
        f_of_ppy=B["f_of_ppy"], f_of_ppy_boot=bsum["k1k2_c"].get("f_of_ppy"),
        pp_fixed_image_centre=dict(note="with the centre-fixed plumb-line distortion (k1k2) and pp = image centre", f=B["plumb_variants"]["k1k2"]["solutions"]["S1_cartX_cartZ|ppC0"]["f"],
                                   boot_sd=rsd(fc0)[0], boot_sd_robust=rsd(fc0)[1]),
        horizon_nadir={pn: B["plumb_variants"][pn]["horizon_nadir"] for pn in B["plumb_variants"]},
        vertical_test=B["vertical_test"], vertical_angles_boot={pn: bsum[pn]["vertical_angles_deg"] for pn in bsum},
        all_solutions=alt_vp, pairsets=B["pairsets"],
        notes=["floor-line groups (FA along / FC across the aisle) are inconsistent with the vertical (chi2 350-550 for 4 dof) -> not used",
               "end-board groups give pp-free solutions that swing with the distortion variant (f 49..1475) -> not used",
               "pp free is under-determined with the structural pairs; if the distortion centre is kept fixed while pp moves ('mixed', "
               "details.f_of_ppy) f changes ~1.7 px per px of pp_y, but for the OpenCV-consistent model (distortion centre = pp) f is almost "
               "independent of pp_y (details.pp_scan: 1469..1503 for pp_y 340..660)",
               "self-consistent pairs (distortion centre = pp) give f 1479-1490 for all plumb variants (centre fixed or free, poly or division)"],
        pp_scan=PS, synthetic_round_trip=(SY["summary"] if SY else None), synthetic_truth=(SY["truth"] if SY else None),
        synthetic_claimed=(SY["claimed"] if SY else None), sensitivity=SE,
        comparison_orchestrator_linecal=(CMP.get("linecal_vp_joint") if CMP else None),
    ),
)
save_json(L.jsonable(vanish), f"{CACHE}/method_vanishing.json")

# ============================================================ method_lines_joint
jm = C["variants"][C["main"]]
fJ, ppJ = jm["f"], np.array(jm["pp"])
kJ = np.array(jm["opencv_dist"])
BJ = bC[C["main"]]
jb = jm["boot"]
k1j = BJ[:, 3] * BJ[:, 0] ** 2 / S ** 2
k2j = BJ[:, 4] * BJ[:, 0] ** 4 / S ** 4
alt_j = {n: {q: v.get(q) for q in ("f", "pp", "coeffs", "kind", "opencv_dist", "invariants", "sig_psi_deg", "sig_pt_px", "psi_rms_deg",
                                   "rms_straightness_px", "rms_px", "angle_CZ_WV_deg", "se_formal", "boot", "mapping_uncertainty", "camera_tilt_from_vertical_deg")}
         for n, v in C["variants"].items()}
joint = lens_json(
    K_from(fJ, fJ, *ppJ), kJ,
    method="lines_joint (independent implementation 40_lines_indep: straightness + VP consistency with a random member-direction error)",
    model="fx=fy, zero skew, pp free (= distortion centre), k1,k2; common vertical for both carts' posts, cart X axes _|_ vertical, "
          "scene verticals own direction",
    rms_reprojection_error_px=jm["rms_straightness_px"],
    uncertainty={"fx_px_1sigma": jb["sd"][0], "fy_px_1sigma": jb["sd"][0], "cx_1sigma": jb["sd"][1], "cy_1sigma": jb["sd"][2],
                 "k1_1sigma": rsd(k1j)[0], "k2_1sigma": rsd(k2j)[0], "f_1sigma_robust": jb["sd_robust"][0], "cy_1sigma_robust": jb["sd_robust"][2],
                 "f_profile_1sigma_formal": None,
                 "how": f"stratified bootstrap ({jb['n']} replicates; resampled members + random direction errors, everything re-fitted); "
                        "formal errors (Gauss-Newton) in details.se_formal"},
    mapping_uncertainty_px=reg_summary(jm["mapping_uncertainty"]),
    data_used=data_used,
    geometry_assumptions="no dimensions; cart X axes _|_ posts; both carts upright on one floor (common vertical)",
    details=dict(psi_rms_deg=jm["psi_rms_deg"], sig_psi_deg=jm["sig_psi_deg"], sig_pt_px=jm["sig_pt_px"], straightness_inflation=C["straightness_inflation"],
                 se_formal=jm.get("se_formal"), boot=jb, mapping_uncertainty_full=jm["mapping_uncertainty"], f_profile=C["f_profile"],
                 variants=alt_j, rms_meaning="straightness rms (px, distorted image); VP consistency: psi_rms_deg",
                 pp_scan=PS, synthetic_round_trip=(SY["summary"] if SY else None), synthetic_truth=(SY["truth"] if SY else None),
                 sensitivity=SE, comparison_orchestrator_linecal=(CMP.get("linecal_vp_joint") if CMP else None),
                 systematics=dict(frames_free_minus_main=C["variants"]["JRE_k1k2_framesfree"]["f"] - fJ,
                                  wv_tied_minus_main=C["variants"]["JRE_k1k2_wvtied"]["f"] - fJ,
                                  div2_minus_main=C["variants"]["JRE_div2_ppfree"]["f"] - fJ,
                                  pointlevel_vp_minus_main=C["variants"]["JPT_k1k2_ppfree"]["f"] - fJ,
                                  pp_y_div2_minus_main=C["variants"]["JRE_div2_ppfree"]["pp"][1] - ppJ[1])),
)
# f profile 1-sigma (delta chi2 = 1, sigma units of the joint cost)
fp = np.array(C["f_profile"]["f"])
dc = np.array(C["f_profile"]["dchi2"])
try:
    lo = np.interp(1.0, dc[: np.argmin(dc) + 1][::-1], fp[: np.argmin(dc) + 1][::-1])
    hi = np.interp(1.0, dc[np.argmin(dc):], fp[np.argmin(dc):])
    joint["uncertainty"]["f_profile_1sigma_formal"] = float((hi - lo) / 2)
except Exception:  # noqa
    pass
save_json(L.jsonable(joint), f"{CACHE}/method_lines_joint.json")


# ============================================================ markdown summaries
def md(fn, title, js, lines):
    u = js["uncertainty"]
    K = js["camera_matrix"]
    d = js["dist_coeffs"]
    txt = [f"# {title}", "", f"method: {js['method']}", "", f"model: {js['model']}", "",
           f"* f = {K[0][0]:.1f} px (1-sigma {u.get('fx_px_1sigma', float('nan')):.1f}), pp = ({K[0][2]:.1f}, {K[1][2]:.1f}) "
           f"(1-sigma {u.get('cx_1sigma', float('nan')):.1f}, {u.get('cy_1sigma', float('nan')):.1f})",
           f"* dist = [{', '.join(f'{v:.4f}' for v in d)}]; k1 1-sigma {u.get('k1_1sigma', float('nan')):.4f}, k2 1-sigma {u.get('k2_1sigma', float('nan')):.4f}",
           f"* rms {js['rms_reprojection_error_px']:.3f} px; mapping uncertainty (rotation-compensated, px) {js['mapping_uncertainty_px']}",
           ""] + lines
    with open(fn, "w") as fh:
        fh.write("\n".join(txt) + "\n")


mods = A["models"]
md(f"{CACHE}/method_plumbline.md", "Plumb-line (independent, 40_lines_indep_a_plumb.py)", plumb, [
    "Straightness determines only k_i/f^(2i): k1/f^2 = {:.4e} +- {:.1e} px^-2, k2/f^4 = {:.3e} +- {:.1e} px^-4 (bootstrap); OpenCV k's use f from the VP method.".format(
        a_pl[0] / S ** 2, rsd(inv_a)[0], a_pl[1] / S ** 4, rsd(inv_b)[0]),
    "Model choice (leave-one-member-out CV rms, px): " + ", ".join(f"{k} {v['loco_cv_rms']:.4f}" for k, v in mods.items()),
    f"Distortion centre FREE: ({cen[0]:.1f}, {cen[1]:.1f}) +- ({boot_c['boot_sd'][0]:.1f}, {boot_c['boot_sd'][1]:.1f}) (bootstrap); the vertical position depends on "
    f"the radial model (div2_c {mods['div2_c']['x'][1]:.0f}, k1k2k3_c {mods['k1k2k3_c']['x'][1]:.0f}, k1k2+tangential {mods['k1k2_c_t']['x'][1]:.0f}) -> cx determined (~930), cy only 'consistent' (430-520).",
    "k1-only folds at the image corner (runs into the fold barrier, rms 0.51 px) -> inadequate; k3 not determinable (runs towards the fold).",
    "Per-region: " + "; ".join(f"{k}: " + ", ".join(f"{n}={v:.4g}" for n, v in r['params'].items()) for k, r in regions.items() if k.endswith('_k1k2_c')),
    "Per-region with the centre FIXED at the image centre disagree (a1: scene {:.4f}, cart310 {:.4f}, cart80 {:.4f}, +-0.004-0.006) but agree with a free centre -> evidence for a decentred distortion (or a different radial profile: div2 with fixed centre fits as well, LOCO 0.2077).".format(
        A["regions"]["scene_k1k2"]["x"][0], A["regions"]["cart310_k1k2"]["x"][0], A["regions"]["cart80_k1k2"]["x"][0]),
    "Rejected as curved: " + ", ".join(A["rejection"]["rejected"]) + " (cart-80 lowest lip E: both boundaries arched by ~0.75-0.8 px -> physically bent lip; A_in = inner shading boundary of the rounded A lip highlight, wobbly; its sharp partner A_out is kept). Crops: results/40_lines_indep_rejected_strips.png.",
    "Same edges, orchestrator LineCal plumb: k@1300 (-0.2890, 0.0698) fixed centre, (-0.2697, 0.0559) centre (929.5, 426.7) -> identical to ours within 0.001 / 1 px.",
] + ([f"Synthetic round trip (40 trials, truth pp shifted +15,+15): centre error sd ({SY['summary']['plumb_centre_x']['sd']:.1f}, {SY['summary']['plumb_centre_y']['sd']:.1f}) px, bias ({SY['summary']['plumb_centre_x']['bias']:+.1f}, {SY['summary']['plumb_centre_y']['bias']:+.1f}); claimed (bootstrap) ({boot_c['boot_sd'][0]:.1f}, {boot_c['boot_sd'][1]:.1f})."] if SY else []))
md(f"{CACHE}/method_vanishing.md", "Vanishing points (independent, 40_lines_indep_b_vp.py)", vanish, [
    f"f from cart-X _|_ common post vertical (both carts, chi2 {sol['chi2']:.1f}/{sol['dof']}), pp = plumb distortion centre.",
    f"pp fixed at the image centre (with the centre-fixed plumb distortion): f = {vanish['details']['pp_fixed_image_centre']['f']:.1f} +- {vanish['details']['pp_fixed_image_centre']['boot_sd']:.1f}.",
    "pp cannot be determined from the VPs: floor lines are inconsistent with the vertical (chi2 350-550/4), end boards not parallel to the cart axes.",
    "For the OpenCV-consistent model (distortion centre = pp) f is nearly independent of pp_y: 1469 (pp_y 340) .. 1489 (540) .. 1503 (660) (results/40_lines_indep_ppscan.png); only mixed models (centre fixed, pp moved) trade 1.7 px f per px pp_y.",
    f"Only cart 310 (310X _|_ 310 posts): f = {B['plumb_variants']['k1k2_c']['solutions']['S1_310only|ppDist']['f']:.0f}; with the scene verticals instead of the posts: {B['plumb_variants']['k1k2_c']['solutions']['S1w_cartX_WV|ppDist']['f']:.0f}.",
    "Self-consistent (distortion centre = pp) solutions agree for all plumb variants (1479-1490); mixed (centre != pp) differ by up to 200 px.",
    f"Cart-post vertical vs scene verticals: {B['vertical_test']['k1k2_c']['angles_deg'].get('CZ-WV', float('nan')):.2f} deg (plumb centre free), "
    f"{B['vertical_test']['k1k2']['angles_deg'].get('CZ-WV', float('nan')):.2f} deg (centre fixed) -> carts' verticals and building verticals not parallel within noise.",
] + ([f"Synthetic round trip (40 trials): VP f bias {SY['summary']['vp_f_ppDist']['bias']:+.1f}, sd {SY['summary']['vp_f_ppDist']['sd']:.1f} px (claimed bootstrap sd {f_sd:.1f}, robust {f_sdr:.1f} -> conservative)."] if SY else [])
   + ([f"Sensitivity: +-0.5 px dark-side edge shift: df {SE['cases']['dark+0.5']['rms_changes']['vp_f']:.2f} px; random per-edge tilts (0.3 px at the ends): df {SE['cases']['tilt_rand0.3']['rms_changes']['vp_f']:.1f} px rms."] if SE else [])
   + (["Orchestrator LineCal on the same edges (separate cart frames, point-level VP): vp " + ", ".join(f"{k} f={v['f']:.0f}" for k, v in CMP['linecal_vp_joint'].items() if k.startswith('vp')) + " -> ~40 px lower, mainly because cart 80's vertical then rests on its own two short post fragments (our frames-free variant: 1438)."] if CMP else []))
md(f"{CACHE}/method_lines_joint.md", "Joint line self-calibration (independent, 40_lines_indep_c_joint.py)", joint, [
    "Cost: straightness (points, distorted-image distances, weight 1/sig_pt with a within-edge correlation inflation) + one angular VP-consistency residual per member (random direction error sig_psi).",
    "Variants (f, pp): " + "; ".join(f"{n}: {v['f']:.0f}, ({v['pp'][0]:.0f},{v['pp'][1]:.0f})" for n, v in C["variants"].items()),
    "f is stable (1470-1490) under all straightness/VP weightings and distortion models with tied cart frames; separate cart frames give 1425-1440 (cart 80's own posts are 2 short edges); scene verticals tied to the posts: 1504.",
    f"pp_y depends on the radial model (k1k2 {ppJ[1]:.0f}, div2 {C['variants']['JRE_div2_ppfree']['pp'][1]:.0f}, k1k2k3 {C['variants']['JRE_k1k2k3_ppfree']['pp'][1]:.0f}); f does not.",
    f"Focal profile (delta chi2 = 1): +-{joint['uncertainty']['f_profile_1sigma_formal']:.0f} px.",
] + ([f"Synthetic round trip (40 trials, truth pp +15,+15): f bias {SY['summary']['joint_f']['bias']:+.1f} sd {SY['summary']['joint_f']['sd']:.1f} (claimed {jb['sd'][0]:.1f}); pp bias ({SY['summary']['joint_cx']['bias']:+.1f}, {SY['summary']['joint_cy']['bias']:+.1f}) sd ({SY['summary']['joint_cx']['sd']:.1f}, {SY['summary']['joint_cy']['sd']:.1f}) (claimed {jb['sd'][1]:.1f}, {jb['sd'][2]:.1f}); mapping error vs truth " + ", ".join(f"{k} {v:.1f}" for k, v in SY['summary']['map_joint'].items()) + " px (claimed " + ", ".join(f"{k} {v:.1f}" for k, v in reg_summary(jm['mapping_uncertainty']).items()) + ")."] if SY else [])
   + (["Orchestrator LineCal joint on the same edges: " + ", ".join(f"{k} f={v['f']:.0f} pp=({v['pp'][0]:.0f},{v['pp'][1]:.0f})" for k, v in CMP['linecal_vp_joint'].items() if k.startswith('joint')) + "; our analogue (separate frames, point-level VP) JPT_k1k2_framesfree: 1435, (932, 505)."] if CMP else []))

# ============================================================ summary figure
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
rows = []
for k in ["S1_cartX_cartZ|ppDist", "S1_310only|ppDist", "S1w_cartX_WV|ppDist"]:
    s = B["plumb_variants"]["k1k2_c"]["solutions"][k]
    bs = bsum["k1k2_c"]["solutions"].get(k)
    rows.append((f"VP k1k2_c {k}", s["f"], bs["sd_robust"][0] if bs else s["se"][0], s["pp"][1], "tab:blue"))
for pn in ["k1k2", "div2"]:
    s = B["plumb_variants"][pn]["solutions"]["S1_cartX_cartZ|ppC0"]
    bs = bsum.get(pn, {}).get("solutions", {}).get("S1_cartX_cartZ|ppC0")
    rows.append((f"VP {pn} (centre fixed) S1|ppC0", s["f"], bs["sd_robust"][0] if bs else s["se"][0], s["pp"][1], "tab:cyan"))
for pn in ["div2_c", "k1k2k3_c"]:
    s = B["plumb_variants"][pn]["solutions"]["S1_cartX_cartZ|ppDist"]
    rows.append((f"VP {pn} S1|ppDist", s["f"], s["se"][0], s["pp"][1], "tab:cyan"))
for n, v in C["variants"].items():
    sd = v["boot"]["sd"][0] if "boot" in v else (v.get("se_formal") or {}).get("f", np.nan)
    rows.append((f"joint {n}", v["f"], sd, v["pp"][1], "tab:red" if n.startswith("JRE") else "tab:orange"))
if CMP:
    for k, v in CMP.get("linecal_vp_joint", {}).items():
        rows.append((f"LineCal {k}", v["f"], v["se_sandwich"][0], v["pp"][1], "gray"))
for i, (lab, f, sd, ppy, col) in enumerate(rows):
    ax[0].errorbar(f, i, xerr=sd, fmt="o", color=col, capsize=3)
    ax[1].plot(ppy, i, "o", color=col)
ax[0].set_yticks(range(len(rows)))
ax[0].set_yticklabels([r[0] for r in rows], fontsize=7)
ax[0].set_xlabel("f [px] (bars: bootstrap sd (VP: robust 68 % half-width) where available, else formal)")
ax[0].set_xlim(1300, 1650)
ax[0].grid(alpha=0.3)
ax[0].set_title("independent line methods: focal length")
ax[1].set_yticks(range(len(rows)))
ax[1].set_yticklabels([])
ax[1].axvline(539.5, color="k", ls=":")
ax[1].set_xlabel("pp_y [px]")
ax[1].grid(alpha=0.3)
ax[1].set_title("principal point y (fixed values shown as used)")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_summary.png", dpi=110)
plt.close()
print("plumb:", plumb["camera_matrix"], plumb["dist_coeffs"], plumb["mapping_uncertainty_px"])
print("vanishing:", vanish["camera_matrix"], vanish["dist_coeffs"], vanish["uncertainty"], vanish["mapping_uncertainty_px"])
print("joint:", joint["camera_matrix"], joint["dist_coeffs"], joint["uncertainty"], joint["mapping_uncertainty_px"])
