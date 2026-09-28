"""Prototype synthetic round-trip for the marker bundle adjustment (identifiability check)."""
import numpy as np
from scipy.optimize import least_squares
from common import *
from calib import *
d = load_json('cache/initial_calib.json')
poses = d['poses']
rng = np.random.default_rng(1)
Ktrue = K_from(1250, 1250, 955, 535); dtrue = np.array([-0.27, 0.065, 0, 0, 0])
# plausible true poses: refit poses of the initial data under the true lens
keys = [tuple(k) for k in d['keys']]; mean = np.array(d['corners_mean']); rot = {tuple(map(int, k.split('_'))): v for k, v in d['rot'].items()}
def all_markers(full):
    ks = []
    for c in (80, 310):
        for mid in cart_marker_ids(c):
            if not full and (c, mid) not in keys: continue
            ks.append((c, mid))
    return ks
def rk(k):
    return rot.get(k, 0 if marker_center(k[1], k[0])[2] == 0 else 2)
# poses under the true lens
m0 = Model(dict(f='single', pp='fixed', dist=[], dist0=list(dtrue), pp0=[955, 535]))
pts = [dict(cart=k[0], X=marker_corners_3d(k[1], k[0], rot[k])[j], uv=mean[i, j]) for i, k in enumerate(keys) for j in range(4)]
pr = Problem(m0, pts, carts=[80, 310])
r = least_squares(lambda y: pr.residuals(np.r_[1250, y]), np.r_[poses['80'], poses['310']], method='lm')
P = r.x.reshape(2, 6)
for full in (False, True):
    for sig_px, sig_mm in [(0.1, 0), (0.2, 0), (0.1, 5), (0.1, 15)]:
        est = []
        for rep in range(40):
            pts = []
            for k in all_markers(full):
                X = marker_corners_3d(k[1], k[0], rk(k))
                Xp = X + rng.normal(0, sig_mm, 3)  # whole-marker 3D displacement (unknown to the fit)
                uv = project(Xp, Ktrue, dtrue, rvec=P[0 if k[0] == 80 else 1, :3], tvec=P[0 if k[0] == 80 else 1, 3:]) + rng.normal(0, sig_px, (4, 2))
                for j in range(4): pts.append(dict(cart=k[0], X=X[j], uv=uv[j]))
            m = Model(dict(f='single', pp='free', dist=['k1', 'k2']))
            pr = Problem(m, pts, carts=[80, 310])
            rr = pr.solve(np.r_[m.pack(Ktrue, dtrue), P.ravel()])
            res = pr.predict(rr.x) - pr.uv
            est.append(np.r_[rr.x[:m.n], np.sqrt(np.mean(np.sum(res ** 2, 1)))])
        est = np.array(est)
        print(f"{'20 markers' if full else '12 markers'} noise {sig_px}px + {sig_mm}mm: rms {est[:,-1].mean():.2f}  f {est[:,0].mean():.1f}±{est[:,0].std():.1f}  cx {est[:,1].mean():.1f}±{est[:,1].std():.1f}  cy {est[:,2].mean():.1f}±{est[:,2].std():.1f}  k1 {est[:,3].mean():.3f}±{est[:,3].std():.3f}  k2 {est[:,4].mean():.3f}±{est[:,4].std():.3f}")
