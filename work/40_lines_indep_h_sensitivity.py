"""Sensitivity of the independent line results to small systematic errors of the edge localisation.

Perturbations of the traced edge points (all accepted edges), each followed by the full re-fit of
  plumb-line k1k2 (free centre), VP f (cart X _|_ common post vertical, pp = distortion centre), joint (JRE main):
  dark+0.5 / dark-0.5 : every point moved 0.5 px along the local edge normal towards (away from) the darker side
                        (the measured black/white edge bias of the stickers is ~0.66 px towards black)
  offset_rand0.3      : a random constant normal offset per edge, N(0, 0.3 px)   (5 draws)
  tilt_rand0.3        : a random linear tilt per edge, end points +-N(0, 0.3 px)  (5 draws)
  bow_rand0.2         : a random quadratic bow per edge, sagitta N(0, 0.2 px)     (5 draws)
Writes work/cache/40_lines_indep_sensitivity.json
"""
import importlib
import time

import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE

L = importlib.import_module("40_lines_indep_lib")
rng = np.random.default_rng(4005)
T0 = time.time()
img = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(float)

edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]


def normals(P):
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack([-t[:, 1], t[:, 0]])


def dark_sign(P, n):
    a = map_coordinates(img, [P[:, 1] + 2 * n[:, 1], P[:, 0] + 2 * n[:, 0]], order=1)
    b = map_coordinates(img, [P[:, 1] - 2 * n[:, 1], P[:, 0] - 2 * n[:, 0]], order=1)
    return np.sign(np.median(b - a))  # +1: darker on the +n side


def run(edge_list):
    pl = L.fit_plumb(L.EdgeSet(edge_list), L.PlumbModel("poly", 2, centre_free=True))
    vps = L.vp_estimates(edge_list, pl["dist"], ["310X", "80X", "CZ", "WV"])
    f = L.f_orthogonal(vps, [("310X", "CZ"), ("80X", "CZ")], pl["dist"].c)
    ini = dict(f=f, pp=tuple(pl["dist"].c), a=list(pl["dist"].a), vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in vps.items()})
    se_cl = np.sqrt(np.diag(pl["cov_cluster"]))[2:]
    se_c = np.sqrt(np.diag(pl["cov_classic"]))[2:]
    m = L.JointModel(edge_list, init=ini, sig_pt=pl["rms"] * float(np.median(se_cl / se_c)), sig_psi_deg=0.155)
    r = m.fit(loss="linear")
    fj, ppj, dj, _ = m.unpack(r.x)
    return dict(plumb_centre=pl["dist"].c.tolist(), plumb_a=pl["dist"].a.tolist(), vp_f=f, joint_f=float(fj), joint_pp=ppj.tolist(), joint_a=dj.a.tolist(),
                joint_k=dj.opencv(fj).tolist())


base = run(acc)
print("base", base)
out = dict(base=base, cases={})


def perturb(kind, amp):
    new = []
    for e in acc:
        P = e["points"]
        n = normals(P)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        s = 2 * s / s[-1] - 1
        if kind == "dark":
            dP = amp * dark_sign(P, n) * n
        elif kind == "offset":
            dP = rng.normal(0, amp) * n
        elif kind == "tilt":
            dP = (rng.normal(0, amp) * s)[:, None] * n
        else:
            dP = (rng.normal(0, amp) * (1 - s * s))[:, None] * n
        new.append(dict(e, points=P + dP))
    return new


for name, kind, amp, nrep in [("dark+0.5", "dark", 0.5, 1), ("dark-0.5", "dark", -0.5, 1), ("offset_rand0.3", "offset", 0.3, 5),
                              ("tilt_rand0.3", "tilt", 0.3, 5), ("bow_rand0.2", "bow", 0.2, 5)]:
    res = [run(perturb(kind, amp)) for _ in range(nrep)]
    d = dict(vp_f=[r["vp_f"] - base["vp_f"] for r in res], joint_f=[r["joint_f"] - base["joint_f"] for r in res],
             joint_ppy=[r["joint_pp"][1] - base["joint_pp"][1] for r in res], joint_ppx=[r["joint_pp"][0] - base["joint_pp"][0] for r in res],
             plumb_cy=[r["plumb_centre"][1] - base["plumb_centre"][1] for r in res],
             plumb_a1=[r["plumb_a"][0] - base["plumb_a"][0] for r in res], joint_k1=[r["joint_k"][0] - base["joint_k"][0] for r in res])
    out["cases"][name] = dict(changes=d, rms_changes={k: float(np.sqrt(np.mean(np.square(v)))) for k, v in d.items()})
    print(name, {k: round(v, 4) for k, v in out["cases"][name]["rms_changes"].items()}, f"({time.time() - T0:.0f}s)")
L.save(L.jsonable(out), "40_lines_indep_sensitivity.json")
print(f"done in {time.time() - T0:.0f}s")
