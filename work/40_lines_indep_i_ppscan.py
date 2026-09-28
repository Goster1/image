"""Scan of the principal-point height: what do the lines say about pp_y, and how does f depend on it?

For pp = (930, y), y = 340 .. 660:
  * plumb-line k1k2 with the distortion centre FIXED at pp: straightness cost (Huber, px^2/2) -> profile of pp_y
  * VPs with that distortion, f from cart-X _|_ post vertical (GLS) with pp = distortion centre  ("self-consistent")
  * the same VPs with the distortion centre kept at the best free-centre plumb fit ("mixed", centre != pp)
  * joint line model (JRE main) with pp fixed at (930, y): f and total cost profile
Writes work/cache/40_lines_indep_ppscan.json, results/40_lines_indep_ppscan.png
"""
import importlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
T0 = time.time()
rng = np.random.default_rng(4006)
edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
PAIRS = [("310X", "CZ"), ("80X", "CZ")]

pl = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
vps0 = L.vp_estimates(acc, pl["dist"], ["310X", "80X", "CZ", "WV"])
f0 = L.f_orthogonal(vps0, PAIRS, pl["dist"].c)
rates = L.edge_angle_rates(acc, pl["dist"], f0, pl["dist"].c)
covs0 = {g: L.vp_cov_mc(acc, pl["dist"], g, vps0[g], rates, np.radians(0.155), n=60, rng_=rng) for g in ("310X", "80X", "CZ")}
se_cl = np.sqrt(np.diag(pl["cov_cluster"]))[2:]
se_c = np.sqrt(np.diag(pl["cov_classic"]))[2:]
SIG_PT = pl["rms"] * float(np.median(se_cl / se_c))
init = dict(f=f0, pp=tuple(pl["dist"].c), a=list(pl["dist"].a), vp_px={g: L.from_scaled_h(v["V"]).tolist() for g, v in vps0.items()})
jm0 = L.JointModel(acc, init=init, sig_pt=SIG_PT, sig_psi_deg=0.155)
jr0 = jm0.fit(loss="linear")
PPX = 930.0
ys = np.arange(340, 661, 20.0)
rows = []
for y in ys:
    pp = np.array([PPX, y])
    p = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre=pp), x0=pl["x"][2:])
    v = L.vp_estimates(acc, p["dist"], ["310X", "80X", "CZ"])
    f_sc = L.gls_f(v, covs0, PAIRS, pp, f0=f0)
    f_mix = L.gls_f(vps0, covs0, PAIRS, pp, f0=f0)
    m = L.JointModel(acc, init=dict(init, pp=tuple(pp)), pp_free=False, pp_fixed=pp, sig_pt=SIG_PT, sig_psi_deg=0.155)
    x0 = np.r_[jr0.x[0], jr0.x[3:]]
    r = m.fit(x0=x0, loss="linear")
    fj = m.unpack(r.x)[0]
    rows.append([y, 2 * p["cost"], f_sc, f_mix, fj, 2 * r.cost])
    print(f"pp_y {y:5.0f}: plumb cost {2 * p['cost']:8.2f}  f self-consistent {f_sc:7.1f}  f mixed {f_mix:7.1f}  joint f {fj:7.1f} cost {2 * r.cost:8.2f}"
          f" ({time.time() - T0:.0f}s)")
R = np.array(rows)
R[:, 1] -= R[:, 1].min()
R[:, 5] -= R[:, 5].min()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(R[:, 0], R[:, 1], "b.-", label="plumb-line k1k2 (centre = pp): delta cost [px^2]")
ax[0].plot(R[:, 0], R[:, 5], "r.-", label="joint (JRE main, pp fixed): delta chi^2")
ax[0].axvline(539.5, color="k", ls=":", label="image centre")
ax[0].set_xlabel("pp_y = distortion-centre y [px] (pp_x = 930)")
ax[0].set_ylim(0, 30)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_title("what the lines say about the principal-point height")
ax[1].plot(R[:, 0], R[:, 2], "b.-", label="VP f, distortion centre = pp (self-consistent)")
ax[1].plot(R[:, 0], R[:, 4], "r.-", label="joint f (pp fixed)")
ax[1].plot(R[:, 0], R[:, 3], "g--", label="VP f, distortion centre kept at the free-centre fit (mixed)")
ax[1].axvline(539.5, color="k", ls=":")
ax[1].set_xlabel("pp_y [px]")
ax[1].set_ylabel("f [px]")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
ax[1].set_title("f vs pp_y")
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_ppscan.png", dpi=110)
plt.close()
L.save(L.jsonable(dict(ppx=PPX, columns=["pp_y", "plumb_dcost", "f_vp_selfconsistent", "f_vp_mixed", "f_joint", "joint_dchi2"], rows=R)),
       "40_lines_indep_ppscan.json")
print(f"done in {time.time() - T0:.0f}s")
