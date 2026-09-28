"""Stage 2: first joint calibration from the plainly detected markers.

* per-frame jitter model: frames differ by a vertical offset + vertical scale (rolling shutter /
  vibration, see 01 output) -> every frame is aligned to the 7-frame mean with dy = a_f + b_f*(y-540)
  (dx likewise, small) before averaging.
* in-plane rotation of every sticker chosen by the best reprojection (4 candidates).
Writes work/cache/initial_calib.json (model, poses, rotations) used to seed the guided detection.
"""
import cv2
import numpy as np

from calib import Model, Problem, covariance, dist_trim
from common import CACHE, H, W, K_from, marker_center, marker_corners_3d, load_json, save_json, project

PREF = ["default", "clahe", "up2_persp", "clahe_persp"]


def gather():
    d = load_json(f"{CACHE}/markers_initial.json")
    keys = sorted({(x["cart"], x["id"]) for x in d})
    A = np.full((len(keys), 7, 4, 2), np.nan)
    for k_i, k in enumerate(keys):
        for f in range(7):
            ds = [x for x in d if (x["cart"], x["id"]) == k and x["frame"] == f]
            ds.sort(key=lambda x: PREF.index(x["config"]))
            if ds:
                A[k_i, f] = ds[0]["corners"]
    return keys, A


def align_frames(A):
    """Estimate per-frame x/y affine-in-y jitter and return aligned array + params."""
    mean = np.nanmean(A, 1)
    params = []
    Al = A.copy()
    for f in range(A.shape[1]):
        dev = (A[:, f] - mean).reshape(-1, 2)
        y = mean.reshape(-1, 2)[:, 1] - H / 2
        ok = ~np.isnan(dev[:, 0])
        M = np.column_stack([np.ones(ok.sum()), y[ok]])
        ax = np.linalg.lstsq(M, dev[ok, 0], rcond=None)[0]
        ay = np.linalg.lstsq(M, dev[ok, 1], rcond=None)[0]
        params.append([*ax, *ay])
        yy = A[:, f, :, 1] - H / 2
        Al[:, f, :, 0] = A[:, f, :, 0] - (ax[0] + ax[1] * yy)
        Al[:, f, :, 1] = A[:, f, :, 1] - (ay[0] + ay[1] * yy)
    return Al, np.array(params)


def main():
    keys, A = gather()
    Al, fp = align_frames(A)
    print("per-frame jitter [ax0, ax1, ay0, ay1]:\n", np.round(fp, 4))
    mean = np.nanmean(Al, 1)
    sd = np.nanstd(Al, 1)
    n = np.sum(~np.isnan(Al[:, :, 0, 0]), 1)
    print("per-corner frame std after alignment (px):", np.round(np.nanmean(sd), 3))

    # --- initial guess: f from a rough 110 deg FOV-free guess, solve by LM from several f seeds
    best = None
    for f0 in [700, 900, 1100, 1300, 1600]:
        K0 = K_from(f0, f0, W / 2, H / 2)
        poses = []
        for cart in (80, 310):
            idx = [i for i, k in enumerate(keys) if k[0] == cart]
            X = np.array([marker_center(keys[i][1], cart) for i in idx])
            uv = mean[idx].mean(1)
            ok, rv, tv = cv2.solvePnP(X, uv, K0, None, flags=cv2.SOLVEPNP_SQPNP)
            poses.append(np.r_[rv.ravel(), tv.ravel()])
        # centres-only LM with k1
        pts = [dict(cart=k[0], X=marker_center(k[1], k[0]), uv=mean[i].mean(0)) for i, k in enumerate(keys)]
        m = Model(dict(f="single", pp="fixed", dist=["k1"]))
        pr = Problem(m, pts, carts=[80, 310])
        x0 = np.r_[m.pack(K0, [-0.1]), np.ravel(poses)]
        r = pr.solve(x0)
        print(f"seed f={f0}: f={r.x[0]:.1f} k1={r.x[1]:.3f} cost={r.cost:.2f}")
        if best is None or r.cost < best[0].cost:
            best = (r, pr)
    r, pr = best
    K, dist, poses = pr.split(r.x)

    # --- rotations of stickers
    rot = {}
    for i, k in enumerate(keys):
        errs = []
        for rk in range(4):
            X = marker_corners_3d(k[1], k[0], rk)
            p = project(X, K, dist, rvec=poses[pr.cidx[k[0]], :3], tvec=poses[pr.cidx[k[0]], 3:])
            errs.append(np.sqrt(np.mean(np.sum((p - mean[i]) ** 2, 1))))
        rot[k] = int(np.argmin(errs))
        print(k, "rot", rot[k], "errs", np.round(errs, 1))

    # --- corner fit, a few models
    pts = []
    for i, k in enumerate(keys):
        X = marker_corners_3d(k[1], k[0], rot[k])
        for j in range(4):
            pts.append(dict(cart=k[0], id=k[1], corner=j, X=X[j], uv=mean[i, j]))
    results = {}
    x_prev = None
    for spec in [dict(f="single", pp="fixed", dist=["k1"]), dict(f="single", pp="fixed", dist=["k1", "k2"]),
                 dict(f="single", pp="free", dist=["k1", "k2"]), dict(f="single", pp="free", dist=["k1", "k2", "k3"]),
                 dict(f="fxfy", pp="free", dist=["k1", "k2", "k3"]), dict(f="single", pp="free", dist=["k1", "k2", "p1", "p2"])]:
        m = Model(spec)
        prb = Problem(m, pts, carts=[80, 310])
        x0 = np.r_[m.pack(K, dist), poses.ravel()]
        rr = prb.solve(x0)
        C, s2, dof = covariance(rr)
        Kf, df, pf = prb.split(rr.x)
        res = (prb.predict(rr.x) - prb.uv)
        rms = np.sqrt(np.mean(np.sum(res ** 2, 1)))
        sig = np.sqrt(np.diag(C))[: m.n]
        print(f"{m.describe():45s} rms={rms:.3f} max={np.max(np.linalg.norm(res, axis=1)):.2f} dof={dof} ",
              " ".join(f"{nm}={v:.4f}±{s:.4f}" for nm, v, s in zip(m.names, rr.x[: m.n], sig)))
        results[m.describe()] = dict(K=Kf, dist=dist_trim(df), poses=pf, rms=rms)
    # keep the k1,k2 pp free model as seed
    seed = results["fx=fy, pp free, dist k1+k2"]
    save_json(dict(keys=[list(k) for k in keys], rot={f"{k[0]}_{k[1]}": v for k, v in rot.items()},
                   K=seed["K"], dist=seed["dist"], poses={"80": seed["poses"][0], "310": seed["poses"][1]},
                   frame_jitter=fp, corners_mean=mean, corners_frame_std=sd, n_frames=n), f"{CACHE}/initial_calib.json")


if __name__ == "__main__":
    main()
