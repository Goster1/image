"""Visual check of the edges rejected as curved by the plumb-line (and comparable accepted edges).

For each edge: the best straight 3D line (undistorted line of the main plumb-line fit, k1k2 + free centre) is
mapped back into the distorted image; the image (mean of the 7 stills) is resampled perpendicular to that
curve (+-8 px, 0.25 px steps) -> "straightened strip": a truly straight edge (with a correct lens) appears as a
horizontal line on the strip centre; the traced points are overlaid (vertical scale x6 relative to the length).
Writes results/40_lines_indep_rejected_strips.png
"""
import importlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from common import CACHE, RESULTS  # noqa: E402

L = importlib.import_module("40_lines_indep_lib")
edges, _ = L.load_edges_indep()
act, _, _ = L.robust_edge_rejection(edges, L.PlumbModel("poly", 2, centre_free=True), fac=2.5, verbose=False)
acc = [e for e, a in zip(edges, act) if a]
rej = [e for e, a in zip(edges, act) if not a]
fit = L.fit_plumb(L.EdgeSet(acc), L.PlumbModel("poly", 2, centre_free=True))
d = fit["dist"]
img = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(float)
col = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
show_ids = [e["id"] for e in rej] + [i for i in ["cart80_x_A_front_out", "cart80_x_D_front", "cart80_x_C_front", "cart80_x_B_front"]
                                      if any(e["id"] == i for e in acc)]
byid = {e["id"]: e for e in edges}

fig, axes = plt.subplots(len(show_ids), 2, figsize=(17, 2.3 * len(show_ids)), gridspec_kw=dict(width_ratios=[4, 1]))
for row, eid in zip(axes, show_ids):
    e = byid[eid]
    P = e["points"]
    U, JU, _ = d.undistort(P)
    c = U.mean(0)
    _, _, vt = np.linalg.svd(U - c)
    t = vt[0]
    n = np.array([-t[1], t[0]])
    s_u = (U - c) @ t
    # dense samples along the straight undistorted line, extended 15 % beyond the traced ends
    ext = 0.15 * (s_u.max() - s_u.min())
    ss = np.linspace(s_u.min() - ext, s_u.max() + ext, 900)
    line_u = c + ss[:, None] * t
    line_d = d.distort_px(line_u)
    tan = np.gradient(line_d, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    offs = np.arange(-8, 8.01, 0.25)
    X = line_d[:, 0][None] + offs[:, None] * nrm[:, 0][None]
    Y = line_d[:, 1][None] + offs[:, None] * nrm[:, 1][None]
    strip = map_coordinates(img, [Y, X], order=1)
    # traced points: signed offset from the distorted line curve (nearest sample)
    idx = np.argmin(np.hypot(P[:, None, 0] - line_d[None, :, 0], P[:, None, 1] - line_d[None, :, 1]), axis=1)
    off = np.sum((P - line_d[idx]) * nrm[idx], axis=1)
    a = row[0]
    a.imshow(strip, cmap="gray", aspect="auto", extent=(0, len(ss), offs[-1], offs[0]))
    a.plot(idx, off, "r.", ms=1.5)
    a.axhline(0, color="c", lw=0.6)
    st = L.EdgeSet([e]).per_edge_stats(L.EdgeSet([e]).residuals(U, JU))[0]
    a.set_title(f"{eid} ({'REJECTED' if e in rej else 'accepted'}): residual bow {st['bow']:+.2f} px, rms {st['rms']:.2f} px; "
                f"strip along the lens-corrected best straight line (cyan), traced points red", fontsize=8)
    a.set_ylabel("offset [px]", fontsize=7)
    a.set_xticks([])
    b = row[1]
    x0, y0 = np.floor(P.min(0) - 30).astype(int)
    x1, y1 = np.ceil(P.max(0) + 30).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    b.imshow(col[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    b.plot(P[:, 0], P[:, 1], "r-", lw=0.8)
    b.set_axis_off()
plt.tight_layout()
plt.savefig(f"{RESULTS}/40_lines_indep_rejected_strips.png", dpi=110)
plt.close()
print("written; rejected:", [e["id"] for e in rej])
