"""(2) Planarity / rectangle tests of the TOP stickers and per-cart general projective camera tests.

All in the undistorted image (plumb-line distortion, pixel units, centre fixed = main; centre free =
check). Residuals are converted back to distorted-image px with the local undistortion Jacobian.

A. Top stickers of one cart (4 stickers, 16 corners, nominally one plane Z = 0, centres on a
   1435 x 330 mm rectangle):
   * one homography for all 16 corners (coplanarity + drawing layout in the plane);
   * one homography per end board (2 stickers, 8 corners): internal consistency of each board;
   * cross prediction: the homography of one end board predicts the corners of the other board;
   * one homography + a shift of the second board along the vertical direction (1 parameter; the
     vertical VP from the posts is used as the direction): is the second board at another height?
B. General projective camera per cart (3x4 P, 11 DOF: no assumption on f, pp, skew, aspect) fitted to
   all sticker corners of the cart (drawing geometry) and with single geometry elements freed. A
   general P absorbs any affine change of the 3D coordinates (e.g. a common Z scale or shift), so
   what remains are the projectively detectable (non-affine) inconsistencies. The height offsets
   are converted to mm with the fitted P (reference = the shelf stickers at the drawing heights).
Output: work/cache/geomdiag_planes.json, results/geomdiag_planes.png
"""
import importlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common import CACHE, RESULTS, save_json  # noqa: E402

G = importlib.import_module("42_geomdiag_lib")
TOP_IDS = {80: {0: [0, 92], 1600: [1, 2]}, 310: {0: [0, 322], 1600: [1, 2]}}
SHELF_IDS = {0: [3, 5, 7], 1600: [4, 6, 8]}


def cart_points(ms, und, cart):
    """All valid corners of one cart: list of dict(id, j, X, uv, U, Jinv, level, end)."""
    pts = []
    for m in ms:
        if m["cart"] != cart:
            continue
        for j in range(4):
            if not m["valid"][j]:
                continue
            uv = m["corners_px"][j]
            U, J = und.jac(uv[None])
            X = m["corners_3d"][j]
            end = 0 if X[0] < 800 else 1600
            pts.append(dict(id=m["id"], j=j, X=X.copy(), uv=uv, U=U[0], Jinv=np.linalg.inv(J[0]), level=m["row"], end=end,
                            sig=float(m["sig_xy"][j]), cen=m["corners_3d"].mean(0)))
    return pts


def to_dist(r_u, pts):
    return np.array([p["Jinv"] @ r for p, r in zip(pts, r_u)])


# --------------------------------------------------------------------------------- homographies
def hom_apply(Hm, XY):
    p = np.column_stack([XY, np.ones(len(XY))]) @ Hm.T
    return p[:, :2] / p[:, 2:3]


def fit_h(pts, extra=None):
    """Homography from plane XY to undistorted px, refined by least squares on distorted-px residuals.
    extra: optional (mask, v_h) -> points in mask are shifted by delta * v (homogeneous addition)."""
    XY = np.array([p["X"][:2] for p in pts])
    U = np.array([p["U"] for p in pts])
    H0, _ = cv2.findHomography(XY, U, 0)
    H0 = H0 / H0[2, 2]

    def res(x):
        Hm = np.append(x[:8], 1.0).reshape(3, 3)
        ph = np.column_stack([XY, np.ones(len(XY))]) @ Hm.T
        if extra is not None:
            mask, vh = extra
            ph[mask] = ph[mask] + x[8] * vh
        pr = ph[:, :2] / ph[:, 2:3]
        return to_dist(pr - U, pts).ravel()

    x0 = H0.ravel()[:8] if extra is None else np.append(H0.ravel()[:8], 0.0)
    r = least_squares(res, x0, x_scale="jac", method="lm")
    Hm = np.append(r.x[:8], 1.0).reshape(3, 3)
    rr = r.fun.reshape(-1, 2)
    return dict(H=Hm, x=r.x, rms=float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))), max=float(np.max(np.hypot(*rr.T))),
                resid=rr, n=len(pts), dof=len(r.fun) - len(r.x))


def predict_h(Hm, pts):
    XY = np.array([p["X"][:2] for p in pts])
    U = np.array([p["U"] for p in pts])
    rr = to_dist(hom_apply(Hm, XY) - U, pts)
    return dict(rms=float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))), max=float(np.max(np.hypot(*rr.T))), mean_vec=rr.mean(0).tolist(), resid=rr)


# --------------------------------------------------------------------------------- general P
GEOM = {
    "drawing": [],
    "board_dz": ["dz_board0", "dz_board1600"],
    "top_dz": ["dz_top"],
    "row_dz": ["dz_A", "dz_B", "dz_C"],
    "stack_dz": ["dz_stack0", "dz_stack1600"],
    "board_dxy": ["dx_board0", "dy_board0", "dx_board1600", "dy_board1600"],
    "board_tilt": ["tx_board0", "ty_board0", "tx_board1600", "ty_board1600"],
    "board_dz+tilt": ["dz_board0", "dz_board1600", "tx_board0", "ty_board0", "tx_board1600", "ty_board1600"],
    "board_dz+dxy": ["dz_board0", "dz_board1600", "dx_board0", "dy_board0", "dx_board1600", "dy_board1600"],
    "card_dz": ["dz_card_%d" % i for i in (3, 4, 5, 6, 7, 8)],
    "board_dz+card_dz": ["dz_board0", "dz_board1600"] + ["dz_card_%d" % i for i in (3, 4, 5, 6, 7, 8)],
    "code_size": ["code_scale"],
    "top_yspacing": ["dy_back_top"],
    "code_size+board_dz": ["code_scale", "dz_board0", "dz_board1600"],
    "code_size+top_dz": ["code_scale", "dz_top"],
    "code_size+board_dz+tilt": ["code_scale", "dz_board0", "dz_board1600", "tx_board0", "ty_board0", "tx_board1600", "ty_board1600"],
    "code_size+card_dz": ["code_scale"] + ["dz_card_%d" % i for i in (3, 4, 5, 6, 7, 8)],
    "code_size+board_dz+card_dz": ["code_scale", "dz_board0", "dz_board1600"] + ["dz_card_%d" % i for i in (3, 4, 5, 6, 7, 8)],
}


def geom_X(pts, names, g):
    X = np.array([p["X"] for p in pts], float).copy()
    gp = dict(zip(names, g))
    if "code_scale" in gp:
        C = np.array([p["cen"] for p in pts])
        X = C + (X - C) * (1.0 + gp["code_scale"])
    for i, p in enumerate(pts):
        top = p["level"] == "top"
        e = p["end"]
        if top:
            X[i, 2] += gp.get(f"dz_board{e}", 0.0) + gp.get("dz_top", 0.0)
            X[i, 0] += gp.get(f"dx_board{e}", 0.0)
            X[i, 1] += gp.get(f"dy_board{e}", 0.0) + (gp.get("dy_back_top", 0.0) if p["X"][1] > 225 else 0.0)
            tx, ty = gp.get(f"tx_board{e}", 0.0), gp.get(f"ty_board{e}", 0.0)
            if tx or ty:
                xc = 82.5 if e == 0 else 1517.5
                # small rotations (rad) about the board's X axis (tx) and Y axis (ty) through (xc, 225, 0)
                X[i, 2] += tx * (p["X"][1] - 225.0) - ty * (p["X"][0] - xc)
        else:
            X[i, 2] += gp.get(f"dz_{p['level']}", 0.0) + gp.get(f"dz_stack{e}", 0.0) + gp.get(f"dz_card_{p['id']}", 0.0)
    return X


def proj_P(P, X):
    ph = np.column_stack([X, np.ones(len(X))]) @ P.T
    return ph[:, :2] / ph[:, 2:3]


def dlt(X, U):
    A = []
    for (x, y, z), (u, v) in zip(X, U):
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    # normalise
    _, _, vt = np.linalg.svd(np.array(A))
    P = vt[-1].reshape(3, 4)
    return P / P[2, 3]


def fit_P(pts, variant, P0=None):
    names = GEOM[variant]
    U = np.array([p["U"] for p in pts])
    X0 = np.array([p["X"] for p in pts])
    if P0 is None:
        P0 = dlt(X0, U)
    scale_g = np.array([1e-3 if (n.startswith("t") or n == "code_scale") else 1.0 for n in names])

    def res(x):
        P = np.append(x[:11], 1.0).reshape(3, 4)
        X = geom_X(pts, names, x[11:] * scale_g) if names else X0
        return to_dist(proj_P(P, X) - U, pts).ravel()

    x0 = np.concatenate([P0.ravel()[:11], np.zeros(len(names))])
    r = least_squares(res, x0, x_scale="jac", method="lm", max_nfev=20000)
    rr = r.fun.reshape(-1, 2)
    J = r.jac
    dof = max(len(r.fun) - len(r.x), 1)
    s2 = 2 * r.cost / dof
    try:
        C = np.linalg.pinv(J.T @ J) * s2
        sd = np.sqrt(np.abs(np.diag(C)))[11:] * scale_g
    except Exception:  # noqa
        sd = np.full(len(names), np.nan)
    rss = float(np.sum(r.fun ** 2))
    n = len(r.fun)
    k = len(r.x)
    return dict(variant=variant, P=np.append(r.x[:11], 1.0).reshape(3, 4), g=dict(zip(names, (r.x[11:] * scale_g).tolist())),
                g_sd=dict(zip(names, sd.tolist())), rms=float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))),
                max=float(np.max(np.hypot(*rr.T))), n_obs=n, k=k, rss=rss, aic=float(G.aic_gauss(rss, n, k)),
                bic=float(G.bic_gauss(rss, n, k)), resid=rr, per_level={lv: float(np.sqrt(np.mean(np.sum(rr[[p["level"] == lv for p in pts]] ** 2, 1))))
                                                                      for lv in ("top", "A", "B", "C") if any(p["level"] == lv for p in pts)})


def P_intrinsics(P):
    """RQ decomposition of the left 3x3 of a general P: fx, fy, skew, cx, cy (undistorted px frame)."""
    M = P[:, :3]
    Kq, Rq = np.linalg.qr(np.flipud(M).T)
    K = np.flipud(np.fliplr(Rq.T))
    K = K / K[2, 2]
    S = np.diag(np.sign(np.diag(K)))
    K = K @ S
    K = K / K[2, 2]
    return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), skew=float(K[0, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))


BOARD_KINDS = (("plain", 0), ("code_size", 1), ("code_size_each", 2), ("y_spacing", 1), ("offset_2nd", 2), ("rot_2nd", 1),
               ("height_2nd", 1))


def board_tests(ms, und, vh, corners="final"):
    """One end board (2 stickers, 8 corners): homography + one in-board element freed.
    code_size and y_spacing are equivalent inside one board (only their ratio is seen)."""
    raw = {(m["cart"], m["id"]): m for m in G.load_json(f"{CACHE}/markers_final.json")}
    out = {}
    for cart in (80, 310):
        for end, ids in TOP_IDS[cart].items():
            P = []
            for i in ids:
                m = [x for x in ms if x["cart"] == cart and x["id"] == i][0]
                c = np.array(raw[(cart, i)]["aruco_corners_px"]) if corners == "aruco" else m["corners_px"]
                cen = m["corners_3d"].mean(0)[:2]
                for j in range(4):
                    U, J = und.jac(c[j][None])
                    P.append(dict(sid=ids.index(i), X=m["corners_3d"][j][:2], cen=cen, U=U[0], Jinv=np.linalg.inv(J[0])))
            XY0 = np.array([p["X"] for p in P])
            U = np.array([p["U"] for p in P])
            sid = np.array([p["sid"] for p in P])
            CEN = np.array([p["cen"] for p in P])
            H0, _ = cv2.findHomography(XY0, U, 0)
            H0 = (H0 / H0[2, 2]).ravel()[:8]

            def model(x, kind):
                Hm = np.append(x[:8], 1).reshape(3, 3)
                g = x[8:]
                XY = XY0.copy()
                if kind == "code_size":
                    XY = CEN + (XY0 - CEN) * (1 + g[0])
                elif kind == "code_size_each":
                    XY = CEN + (XY0 - CEN) * (1 + np.where(sid == 1, g[1], g[0]))[:, None]
                elif kind == "y_spacing":
                    XY = XY0 + (sid[:, None] == 1) * np.array([0, g[0]])
                elif kind == "offset_2nd":
                    XY = XY0 + (sid[:, None] == 1) * np.array([g[0], g[1]])
                elif kind == "rot_2nd":
                    a = g[0]
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    XY = np.where(sid[:, None] == 1, CEN + (XY0 - CEN) @ R.T, XY0)
                ph = np.column_stack([XY, np.ones(len(XY))]) @ Hm.T
                if kind == "height_2nd":
                    ph = ph + np.where(sid == 1, g[0], 0)[:, None] * vh
                return ph[:, :2] / ph[:, 2:]

            def res(x, kind):
                r = model(x, kind) - U
                return np.array([p["Jinv"] @ rr for p, rr in zip(P, r)]).ravel()

            R = {}
            for kind, ng in BOARD_KINDS:
                r = least_squares(lambda x: res(x, kind), np.append(H0, np.zeros(ng)), method="lm", x_scale="jac")
                rr = r.fun.reshape(-1, 2)
                g = r.x[8:]
                val = {"code_size": lambda g: {"code_mm": 90 * (1 + g[0])},
                       "code_size_each": lambda g: {"code_mm_1": 90 * (1 + g[0]), "code_mm_2": 90 * (1 + g[1])},
                       "y_spacing": lambda g: {"extra_spacing_mm": abs(g[0])},
                       "offset_2nd": lambda g: {"dX_mm": g[0], "dY_mm": g[1]},
                       "rot_2nd": lambda g: {"rot_deg": float(np.rad2deg(g[0]))},
                       "height_2nd": lambda g: {"delta_h": g[0]}, "plain": lambda g: {}}[kind](g)
                R[kind] = dict(rms=float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))), max=float(np.max(np.hypot(*rr.T))), **{k: float(v) for k, v in val.items()})
            out[f"{cart}_{end}"] = R
    return out


def board_distortion_test(ms):
    """Can ANY radial distortion (k1, k2 at f0 = 1300, centre fixed or free) make the four end boards
    individually consistent with a plane homography of the drawing layout? (distances in undistorted px)"""
    from common import K_from, undistort_points
    boards = []
    for cart in (80, 310):
        for end, ids in TOP_IDS[cart].items():
            XY, uv = [], []
            for m in ms:
                if m["cart"] == cart and m["id"] in ids:
                    XY += [m["corners_3d"][j][:2] for j in range(4)]
                    uv += [m["corners_px"][j] for j in range(4)]
            boards.append((f"{cart}_{end}", np.array(XY), np.array(uv)))

    def res(x, free_c):
        c = x[2:4] if free_c else np.array([959.5, 539.5])
        K = K_from(G.F0, G.F0, c[0], c[1])
        out = []
        for _, XY, uv in boards:
            U = undistort_points(uv, K, np.array([x[0], x[1], 0, 0, 0]), iters=30) * G.F0 + c
            H0, _ = cv2.findHomography(XY, U, 0)
            rh = least_squares(lambda h: (hom_apply(np.append(h, 1).reshape(3, 3), XY) - U).ravel(), (H0 / H0[2, 2]).ravel()[:8], method="lm")
            out.append(rh.fun)
        return np.concatenate(out)

    out = {}
    for free_c in (False, True):
        x0 = [-0.29, 0.07] + ([959.5, 539.5] if free_c else [])
        r = least_squares(lambda x: res(x, free_c), x0, diff_step=1e-4)
        rr = r.fun.reshape(-1, 2)
        per = [float(np.sqrt(np.mean(np.sum(rr[i * 8:(i + 1) * 8] ** 2, 1)))) for i in range(4)]
        out["centre_free" if free_c else "centre_fixed"] = dict(x=r.x.tolist(), board_rms=dict(zip([b[0] for b in boards], per)))
        print("board planarity with the best radial distortion,", "centre free" if free_c else "centre fixed", np.round(r.x, 4), np.round(per, 3))
    return out


def main():
    ms = G.markers()
    edges = G.all_edges()
    out = {}
    for var in ("k1k2_centre_fixed", "k1k2_centre_free", "none"):
        pl = G.plumb(var, edges)
        und = G.Und(pl)
        vpe = G.fit_vp_lines(und, G.vertical_edges(edges))
        vh = np.append(vpe["v"], 1.0)
        R = {"vp": vpe["v"].tolist(), "board_tests": board_tests(ms, und, vh), "board_tests_aruco_corners": board_tests(ms, und, vh, "aruco")}
        for cart in (80, 310):
            pts = cart_points(ms, und, cart)
            top = [p for p in pts if p["level"] == "top"]
            b0 = [p for p in top if p["end"] == 0]
            b1 = [p for p in top if p["end"] == 1600]
            A = {}
            A["top_all_H"] = fit_h(top)
            A["board0_H"] = fit_h(b0)
            A["board1600_H"] = fit_h(b1)
            A["pred_board1600_from_board0"] = predict_h(A["board0_H"]["H"], b1)
            A["pred_board0_from_board1600"] = predict_h(A["board1600_H"]["H"], b0)
            mask = np.array([p["end"] == 1600 for p in top])
            A["top_all_H_plus_vertical_shift_of_board1600"] = fit_h(top, extra=(mask, vh))
            # general P on all points, variants
            Pres = {}
            P_d = fit_P(pts, "drawing")
            for v in GEOM:
                Pres[v] = fit_P(pts, v, P0=P_d["P"])
            shelf = [p for p in pts if p["level"] != "top"]
            Pres["shelf_only_drawing"] = fit_P(shelf, "drawing")
            Pres["shelf_only_row_dz"] = fit_P(shelf, "row_dz", P0=Pres["shelf_only_drawing"]["P"])
            Pres["shelf_only_card_dz"] = fit_P(shelf, "card_dz", P0=Pres["shelf_only_drawing"]["P"])
            # mm-conversion of the vertical board shift of the homography test with the P of 'board_dz'
            for kk, vv in Pres.items():
                vv["intr"] = P_intrinsics(vv["P"])
            R[cart] = dict(planar=A, P=Pres, n_pts=len(pts), n_top=len(top), n_shelf=len(shelf))
        out[var] = R

    # ------------- print + json -------------
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in ("resid", "H", "x")}
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d

    out["board_distortion_test"] = board_distortion_test(ms)
    save_json(strip(out), f"{CACHE}/geomdiag_planes.json")
    for var in [v for v in out if v != "board_distortion_test"]:
        print("=====", var, "VP", np.round(out[var]["vp"], 1))
        for nm in ("board_tests", "board_tests_aruco_corners"):
            for k, R in out[var][nm].items():
                print(f"  {nm:26s} {k:9s}", " | ".join(f"{kind} {v['rms']:.2f}" + ("" if kind == "plain" else " " + ",".join(f"{a}={b:.4g}" for a, b in v.items() if a not in ("rms", "max")))
                                                     for kind, v in R.items()))
        for cart in (80, 310):
            R = out[var][cart]
            A = R["planar"]
            print(f" cart {cart}: top 16-corner H rms {A['top_all_H']['rms']:.2f} max {A['top_all_H']['max']:.2f} | board0 H rms {A['board0_H']['rms']:.3f}"
                  f" | board1600 H rms {A['board1600_H']['rms']:.3f} | pred 1600<-0 rms {A['pred_board1600_from_board0']['rms']:.1f} max {A['pred_board1600_from_board0']['max']:.1f}"
                  f" | pred 0<-1600 rms {A['pred_board0_from_board1600']['rms']:.1f} | H+vertical shift rms {A['top_all_H_plus_vertical_shift_of_board1600']['rms']:.2f}"
                  f" (delta {A['top_all_H_plus_vertical_shift_of_board1600']['x'][8]:.4g})")
            for v, P in R["P"].items():
                gs = {k: (f"{val:.1f}+-{P['g_sd'][k]:.1f}" if not k.startswith("t") else f"{np.rad2deg(val):.2f}+-{np.rad2deg(P['g_sd'][k]):.2f}deg")
                      if k != "code_scale" else f"code {90 * (1 + val):.2f}+-{90 * P['g_sd'][k]:.2f}mm" for k, val in P["g"].items()}
                it = P["intr"]
                print(f"   P[{v:18s}] rms {P['rms']:.2f} max {P['max']:.2f} n {P['n_obs']} k {P['k']} AIC {P['aic']:.1f} per-level {({k: round(x, 2) for k, x in P['per_level'].items()})} "
                      f"| fx {it['fx']:.0f} fy {it['fy']:.0f} skew {it['skew']:.0f} pp ({it['cx']:.0f},{it['cy']:.0f}) | {gs}")

    # ------------- plot: residual vectors of the general P (drawing) and (board_dz) -------------
    var = "k1k2_centre_fixed"
    pl = G.plumb(var, edges)
    und = G.Und(pl)
    img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.2), dpi=110)
    for ax, v in zip(axs, ("drawing", "board_dz")):
        ax.imshow(img, alpha=0.6)
        for cart in (80, 310):
            pts = cart_points(ms, und, cart)
            P = fit_P(pts, "drawing")
            P = fit_P(pts, v, P0=P["P"]) if v != "drawing" else P
            uv = np.array([p["uv"] for p in pts])
            rr = P["resid"]
            col = ["r" if p["level"] == "top" else "c" for p in pts]
            ax.quiver(uv[:, 0], uv[:, 1], rr[:, 0] * 20, rr[:, 1] * 20, angles="xy", scale_units="xy", scale=1, color=col, width=0.003)
            ax.text(uv[:, 0].mean(), uv[:, 1].mean(), f"cart {cart}\nrms {P['rms']:.2f} px", color="y", fontsize=9, ha="center",
                    bbox=dict(fc="k", alpha=0.5))
        ax.set_title(f"general 3x4 camera per cart (no intrinsic assumptions), geometry '{v}'\nresidual x20 (red: top stickers, cyan: shelf stickers)", fontsize=9)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/geomdiag_planes.png")
    print("saved")


if __name__ == "__main__":
    main()
