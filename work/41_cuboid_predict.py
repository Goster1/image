"""Cuboid cart model, part 2: PREDICTION CHECK of all modelled cart lines against the traced edges.

Cameras (intrinsics + one pose per cart; drawing geometry in every case):
  markers_only : work/cache/method_markers.json (20_markers_ba.py best model) with its poses
  joint        : work/cache/method_cuboid.json (41_cuboid_fit.py: stickers + sticker sides + exact edges)
  edge_lens    : lens of the edge-only joint line fit (method_lines_joint.json, no sticker information)
                 with poses fitted to all sticker corners - a reference that isolates the lens effect
Modelled lines (drawing): top-plate outline (Z=0), 4 corner posts, shelf front (Y=0) / back (Y=450)
edges at the 5 shelf heights, floor outline (Z=-1800).  Every traced cart edge that belongs to one of
these elements (41_cuboid_lib.EDGE_MAP, incl. non-exact ones such as shelf lips, posts, rails) is
compared with its predicted line:
  * signed offset along the image normal [px]: mean, first / last 10 % of the edge, rms;
    sign: + = the traced edge lies on the +axis side of the predicted line (axis = 1st letter of 'axes')
  * implied 3D shift of the line [mm] along each of the two non-free axes (camera fixed), with the
    sensitivity px/mm, and the joint 2-D shift (diagnostic)
  * camera part of the uncertainty of the mean offset: std over the bootstrap cameras
    (joint: cluster bootstrap of 41_cuboid_fit.py; markers_only: bootstrap intrinsics of 20_markers_ba.py
    with the poses refit)
Pattern analysis of the shelf lips (per cart, camera fixed): one common lip offset (dY, dZ) for all five
lips, the per-lip deviations from it, and whether one extra 'single dimension' (shelf-depth scale,
top-plate height relative to the shelves) explains the rest.  Diagnostic only.
Outputs: work/cache/cuboid_offsets.json, results/cuboid_overlay.png (full resolution),
         results/cuboid_crops.png, results/cuboid_offsets.png
"""
import importlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, RESULTS, H, W, load_json, save_json  # noqa: E402

L = importlib.import_module("41_cuboid_lib")
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
RAW = L.load_cart_edges_raw()
CL = L.cart_lines()
M, PTS, SIDES, EXACT, P0 = L.base_data()
LIPS = {310: ["cart310_x_A_front_fall", "cart310_x_B_front_fall", "cart310_x_C_front_fall", "cart310_x_D_front_fall", "cart310_x_E_front_fall"],
        80: ["cart80_x_A_front_out", "cart80_x_B_front", "cart80_x_C_front", "cart80_x_D_front", "cart80_x_E_front"]}


# ------------------------------------------------------------------------------------------------ cameras
def camera_joint():
    full = load_json(f"{CACHE}/cuboid_fit_full.json")
    mc = load_json(f"{CACHE}/method_cuboid.json")
    x = np.array(full["x_main"])
    names = full["names"]
    ni = len([n for n in names if not n.startswith("pose")])
    ni = names.index("pose80_0")
    poses = {80: x[ni:ni + 6], 310: x[ni + 6:ni + 12]}
    cam = L.Camera(np.array(mc["camera_matrix"]), mc["dist_coeffs"], poses, name="joint (" + mc["model"] + ")")
    # bootstrap cameras
    boots = []
    key = full["main_key"].split("|")[0]
    for xb in full["boot"]:
        xb = np.array(xb)
        intr = dict(zip(names[:ni], xb[:ni]))
        f = intr.get("f", 1.0)
        cx, cy = intr.get("cx", (W - 1) / 2), intr.get("cy", (H - 1) / 2)
        d = [intr.get("k1", 0.0), intr.get("k2", 0.0), intr.get("p1", 0.0), intr.get("p2", 0.0), intr.get("k3", 0.0)]
        boots.append(L.Camera(L.K_from(intr.get("fx", f), intr.get("fy", f), cx, cy), d, {80: xb[ni:ni + 6], 310: xb[ni + 6:ni + 12]}))
    return cam, boots, key


def camera_markers_only(nboot=150, seed=2020):
    """Camera of method_markers.json; bootstrap cameras = cluster bootstrap over stickers (corners only, same
    model: f, k1, k2, pp fixed; poses refit in every sample) - same resampling as the joint fit."""
    cam = L.markers_only_camera()
    mm = load_json(f"{CACHE}/method_markers.json")
    spec = dict(f="single", pp="fixed" if "pp fixed" in mm["model"] else "free", dist=["k1", "k2"])
    rng = np.random.default_rng(seed)
    mis = sorted({p["mi"] for p in PTS})
    cf0 = L.CuboidFit(PTS, [], spec)
    x0 = cf0.x0(cam.K, cam.dist, cam.poses)
    boots = []
    for b in range(nboot):
        pick = rng.choice(mis, len(mis), replace=True)
        bp = []
        for q, mi in enumerate(pick):
            bp += [dict(p, mi=1000 * q + mi) for p in PTS if p["mi"] == mi]
        if len({p["cart"] for p in bp}) < 2:
            continue
        cf = L.CuboidFit(bp, [], spec)
        r = cf.solve(x0, reweight=False)
        boots.append(cf.camera(r.x))
    return cam, boots


def camera_edge_lens(rows=None):
    """Lens of the edge-only joint line fit; poses from the sticker corners (all, or only the given rows)."""
    lj = load_json(f"{CACHE}/method_lines_joint.json")
    K = np.array(lj["camera_matrix"])
    d = np.array(lj["dist_coeffs"])
    pts = PTS if rows is None else [p for p in PTS if M[p["mi"]]["row"] in rows]
    tag = "all stickers" if rows is None else "stickers of rows " + "+".join(rows) + " only"
    return L.Camera(K, d, L.fit_poses(K, d, pts, P0), name=f"edge-only lens (method_lines_joint) + poses from {tag}")


# ------------------------------------------------------------------------------------------------ offsets
def edge_offsets(cam, eid):
    nm, axes, lab = L.EDGE_MAP[eid]
    e = RAW[eid]
    ln = CL[nm]
    c = e["cart"]
    P = e["points"]
    r, _ = L.line_offsets(cam, c, ln["A"], ln["D"], P)
    sgn = L.physical_sign(cam, c, ln["A"], ln["D"], P, AX[axes[0]])
    off = r * sgn  # + = traced edge on the +axes[0] side of the prediction
    n = len(off)
    k = max(n // 10, 3)
    out = dict(model_line=nm, label=lab, cart=c, sign_axis=axes[0], n_points=n, mean_px=float(off.mean()),
               start_px=float(off[:k].mean()), end_px=float(off[-k:].mean()), rms_px=float(np.sqrt(np.mean(off ** 2))),
               start_xy=P[:k].mean(0).tolist(), end_xy=P[-k:].mean(0).tolist())
    for a in axes:
        s, rms_, sens = L.implied_shift(cam, c, ln["A"], ln["D"], P, AX[a])
        out[f"implied_d{a}_mm"] = s
        out[f"rms_after_d{a}_px"] = rms_
        out[f"px_per_mm_{a}"] = sens
    k2, sd2, corr, rms2 = L.implied_shift_2d(cam, c, ln["A"], ln["D"], P, AX[axes[0]], AX[axes[1]])
    out["implied_2d"] = {f"d{axes[0]}_mm": float(k2[0]), f"d{axes[1]}_mm": float(k2[1]), "sigma_formal_mm": sd2.tolist(), "corr": corr, "rms_after_px": rms2}
    return out


def offsets_table(cam, boots):
    tab = {}
    for eid in L.EDGE_MAP:
        tab[eid] = edge_offsets(cam, eid)
    if boots:
        for eid in L.EDGE_MAP:
            nm, axes, _ = L.EDGE_MAP[eid]
            ln = CL[nm]
            e = RAW[eid]
            vals, sh = [], []
            for cb in boots:
                r, _ = L.line_offsets(cb, e["cart"], ln["A"], ln["D"], e["points"])
                sg = L.physical_sign(cb, e["cart"], ln["A"], ln["D"], e["points"], AX[axes[0]])
                vals.append(float(np.mean(r * sg)))
                sh.append(L.implied_shift(cb, e["cart"], ln["A"], ln["D"], e["points"], AX[axes[0]])[0])
            tab[eid]["camera_sigma_mean_px"] = float(np.std(vals, ddof=1))
            tab[eid]["camera_sigma_mean_px_robust"] = float((np.percentile(vals, 84) - np.percentile(vals, 16)) / 2)
            tab[eid][f"camera_sigma_d{axes[0]}_mm"] = float(np.std(sh, ddof=1))
            tab[eid][f"camera_sigma_d{axes[0]}_mm_robust"] = float((np.percentile(sh, 84) - np.percentile(sh, 16)) / 2)
    return tab


def sticker_rows(cam):
    """Corner residuals of every sticker row with this camera, and (camera fixed) the Z shift of each row that
    best fits its corners (diagnostic: where the row 'is' in height for this camera)."""
    from scipy.optimize import least_squares as lsq
    out = {}
    for row in ("top", "A", "B", "C"):
        for c in (80, 310):
            pts = [p for p in PTS if p["cart"] == c and M[p["mi"]]["row"] == row]
            if not pts:
                continue
            X = np.array([p["X"] for p in pts])
            uv = np.array([p["uv"] for p in pts])
            r0 = cam.proj(X, c) - uv

            def res(dz):
                return (cam.proj(X + [0, 0, dz[0]], c) - uv).ravel()

            rr = lsq(res, [0.0])
            out[f"{c}:{row}"] = dict(rms_px=float(np.sqrt(np.mean(np.sum(r0 ** 2, 1)))), n=len(pts), implied_dZ_mm=float(rr.x[0]),
                                     rms_after_dZ_px=float(np.sqrt(np.mean(rr.fun ** 2) * 2)))
    return out


def lip_pattern(cam):
    """Per cart: common lip offset (dY, dZ) of the five shelf-front lips + per-lip deviation; single-dimension
    alternatives with the camera fixed (diagnostic)."""
    out = {}
    for c, ids in LIPS.items():
        R, t = cam.R(c), cam.t(c)
        rows = []
        for eid in ids:
            nm = L.EDGE_MAP[eid][0]
            ln = CL[nm]
            P = RAW[eid]["points"]
            s = L.foot_params(P, cam.K, cam.dist, R, t, ln["A"], ln["D"])
            r0, _ = L.tangent_residuals(P, s, cam.K, cam.dist, R, t, ln["A"], ln["D"])
            gY = L.tangent_residuals(P, s, cam.K, cam.dist, R, t, ln["A"] + AX["Y"], ln["D"])[0] - r0
            gZ = L.tangent_residuals(P, s, cam.K, cam.dist, R, t, ln["A"] + AX["Z"], ln["D"])[0] - r0
            sz = -np.sign(np.mean(gZ))  # physical sign: + = traced edge ABOVE (+Z side of) the predicted line
            rows.append(dict(eid=eid, row=nm[5], z=ln["A"][2], r0=r0 * sz, gY=gY * sz, gZ=gZ * sz, w=np.sqrt(4.0 / len(P))))

        def solve(cols):
            J = np.vstack([np.column_stack([rw[c_](rw) if callable(c_) else rw[c_] for c_ in cols]) * rw["w"] for rw in rows])
            b = np.concatenate([-rw["r0"] * rw["w"] for rw in rows])
            k, *_ = np.linalg.lstsq(J, b, rcond=None)
            res = [rw["r0"] + np.column_stack([rw[c_](rw) if callable(c_) else rw[c_] for c_ in cols]) @ k for rw in rows]
            dof = max(len(rows) * 4 - len(cols), 1)
            C = np.linalg.pinv(J.T @ J) * np.sum((J @ k - b) ** 2) / dof
            return k, np.sqrt(np.diag(C)), res

        res_c = {}

        def summ(k, sd, res, names):
            d_ = {n: float(v) for n, v in zip(names, k)}
            d_.update({f"{n}_sigma": float(v) for n, v in zip(names, sd)})
            d_["per_lip_mean_px"] = {rw["row"]: float(np.mean(rr)) for rw, rr in zip(rows, res)}
            d_["rms_px"] = float(np.sqrt(np.mean(np.concatenate(res) ** 2)))
            return d_

        for rw in rows:
            rw["gS"] = rw["gZ"] * rw["z"] / 1000.0  # shelf-depth scale (per mille of the depth below the top)
        # (a) one common lip height dZ (lip outline at Y = 0): the drawing + lip profile
        k, sd, res = solve(["gZ"])
        res_c["common_dZ"] = summ(k, sd, res, ["dZ_mm"])
        # (b) one common Y offset of the lip outline (at the nominal shelf height)
        k, sd, res = solve(["gY"])
        res_c["common_dY"] = summ(k, sd, res, ["dY_mm"])
        # (c) common lip height + a common scale of the shelf depths below the top plate (single dimension)
        k, sd, res = solve(["gZ", "gS"])
        res_c["common_dZ+depth_scale"] = summ(k, sd, res, ["dZ_mm", "scale_permille"])
        # (d) common lip height + an extra shift of shelves D, E (rows without stickers) relative to A, B, C
        for rw in rows:
            rw["gDE"] = rw["gZ"] * (1.0 if rw["row"] in ("D", "E") else 0.0)
        k, sd, res = solve(["gZ", "gDE"])
        res_c["common_dZ+DE_shift"] = summ(k, sd, res, ["dZ_mm", "dZ_DE_mm"])
        # (c) per-lip implied dZ and dY (1-D)
        res_c["per_lip"] = {rw["row"]: dict(dZ_mm=float(-np.sum(rw["gZ"] * rw["r0"]) / np.sum(rw["gZ"] ** 2)),
                                            dY_mm=float(-np.sum(rw["gY"] * rw["r0"]) / np.sum(rw["gY"] ** 2)),
                                            px_per_mm_Z=float(np.mean(np.abs(rw["gZ"]))), px_per_mm_Y=float(np.mean(np.abs(rw["gY"]))),
                                            mean_offset_px_up=float(np.mean(rw["r0"]))) for rw in rows}
        res_c["sign_note"] = "per-lip residuals [px] with physical sign: + = traced lip lies above (+Z side of) the fitted line"
        out[c] = res_c
    return out


# ------------------------------------------------------------------------------------------------ drawing
def draw_lines(img, cam, color, thick=1, label=False):
    for c in (80, 310):
        for nm, ln in CL.items():
            uv, _ = L.project_segment(cam, c, ln["A"], ln["D"], ln["t0"], ln["t1"], n=900)
            ok = (uv[:, 0] > -500) & (uv[:, 0] < W + 500) & (uv[:, 1] > -500) & (uv[:, 1] < H + 500)
            uv = uv[ok]
            if len(uv) < 2:
                continue
            cv2.polylines(img, [np.round(uv * 8).astype(np.int32)], False, color, thick, lineType=cv2.LINE_AA, shift=3)
            if label:
                j = len(uv) // 2
                p = tuple(int(v) for v in uv[j])
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.putText(img, f"{c}:{nm}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)


CROP_WINS = [("310 lips A..E (mid-height)", 310, (615, 370, 815, 470)),
             ("310 X=0 end: board outer edge, front post", 310, (735, 850, 885, 1005)),
             ("310 X=1600 end: board front / outer edge", 310, (370, 0, 560, 115)),
             ("310 back member (x_back_inner)", 310, (375, 420, 475, 530)),
             ("80 lips A..E (mid-height)", 80, (1155, 425, 1325, 525)),
             ("80 X=0 end: board front edge", 80, (1335, 0, 1485, 125)),
             ("80 X=1600 end: board front / outer edge, post", 80, (1175, 830, 1335, 990)),
             ("80 back top rail (Y=450, Z=0)", 80, (1530, 370, 1665, 490))]
SHORT = {"top_front": "top Y0", "top_back": "top Y450", "top_x0": "top X0", "top_x1600": "top X1600", "floor_front": "floor Y0",
         "floor_back": "floor Y450", "floor_x0": "floor X0", "floor_x1600": "floor X1600"}


def short(nm):
    if nm in SHORT:
        return SHORT[nm]
    if nm.startswith("shelf"):
        return nm[5] + (" front" if nm.endswith("front") else " back")
    return nm.replace("post_", "post ")


def render_crops(camlist, fname, title):
    base = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(2, 4, figsize=(22, 11.5), dpi=110)
    for ax, (ttl, cart, (x0, y0, x1, y1)) in zip(axs.ravel(), CROP_WINS):
        ax.imshow((base[y0:y1, x0:x1] * 0.8).astype(np.uint8), extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="bicubic")
        for eid, e in RAW.items():
            P = e["points"]
            m = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
            if m.any():
                ax.plot(P[m, 0], P[m, 1], ".", ms=2.2, color="lime" if eid in L.EDGE_MAP else "0.7")
        ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        for j, (cam, col, lab) in enumerate(camlist):
            for nm, ln in CL.items():
                uv, _ = L.project_segment(cam, cart, ln["A"], ln["D"], ln["t0"], ln["t1"], n=3000)
                m = (uv[:, 0] > x0) & (uv[:, 0] < x1) & (uv[:, 1] > y0) & (uv[:, 1] < y1)
                if m.sum() < 2:
                    continue
                ax.plot(uv[:, 0], uv[:, 1], "-", color=col, lw=1.0, alpha=0.9, label=lab if nm == "top_front" else None)
                if j == 0:
                    q = uv[m]
                    k = np.argmin(np.sum((q - ctr) ** 2, 1) + 0 * q[:, 0])
                    k = int(np.clip(k, 0, len(q) - 1))
                    ax.text(q[k, 0], q[k, 1], short(nm), color="white", fontsize=7, clip_on=True, ha="left", va="bottom",
                            bbox=dict(facecolor=col, alpha=0.55, pad=0.8, lw=0))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(ttl, fontsize=10)
        ax.tick_params(labelsize=7)
    h = [plt.Line2D([], [], color=col, lw=1.5, label=lab) for _, col, lab in camlist] + [plt.Line2D([], [], color="lime", marker=".", ls="", label="traced edge (compared)")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(fname)
    plt.close(fig)


def main():
    cams = {}
    cam_j, boots_j, key = camera_joint()
    cam_m, boots_m = camera_markers_only()
    cam_e = camera_edge_lens()
    cams = {"markers_only": (cam_m, boots_m), "joint": (cam_j, boots_j), "edge_lens": (cam_e, []),
            "edge_lens_shelf_poses": (camera_edge_lens(("A", "B", "C")), []), "edge_lens_top_poses": (camera_edge_lens(("top",)), [])}
    result = dict(sign_convention="offset + = traced edge lies on the + side (sign_axis) of the predicted line; implied_dA_mm = shift of the "
                                  "drawing line along A (camera fixed) that puts it onto the traced edge",
                  cameras={k: dict(name=v[0].name, K=v[0].K.tolist(), dist=v[0].dist[:5].tolist(), poses={c: p.tolist() for c, p in v[0].poses.items()},
                                   n_boot=len(v[1])) for k, v in cams.items()},
                  edges={}, lip_pattern={}, edge_files_reviewed=L.edges_review_state())
    for cname, (cam, boots) in cams.items():
        tab = offsets_table(cam, boots)
        result["edges"][cname] = tab
        result["lip_pattern"][cname] = lip_pattern(cam)
        result.setdefault("sticker_rows", {})[cname] = sticker_rows(cam)
        print("  sticker rows:", {k: (round(v["rms_px"], 2), round(v["implied_dZ_mm"], 1)) for k, v in result["sticker_rows"][cname].items()})
        print(f"==== camera {cname}: {cam.name}")
        for eid, v in tab.items():
            a0, a1 = L.EDGE_MAP[eid][1][0], L.EDGE_MAP[eid][1][1]
            print(f"  {eid:34s} {v['model_line']:14s} mean {v['mean_px']:+7.2f} (ends {v['start_px']:+6.2f} {v['end_px']:+6.2f}) px"
                  f" +-{v.get('camera_sigma_mean_px', float('nan')):.2f} | d{a0} {v[f'implied_d{a0}_mm']:+7.1f} mm ({v[f'px_per_mm_{a0}']:.3f} px/mm)"
                  f" | d{a1} {v[f'implied_d{a1}_mm']:+7.1f} mm | rms after {v[f'rms_after_d{a0}_px']:.2f}")
        for c, lp in result["lip_pattern"][cname].items():
            cd = lp["common_dZ"]
            print(f"  lips cart {c}: common lip height dZ {cd['dZ_mm']:+.1f} mm, per-lip dev "
                  + str({k: round(v, 2) for k, v in cd["per_lip_mean_px"].items()}) + f" rms {cd['rms_px']:.2f};"
                  f" dY only {lp['common_dY']['dY_mm']:+.1f} mm rms {lp['common_dY']['rms_px']:.2f};"
                  f" +depth scale {lp['common_dZ+depth_scale']['scale_permille']:+.1f} permille rms {lp['common_dZ+depth_scale']['rms_px']:.2f};"
                  f" +D/E shift {lp['common_dZ+DE_shift']['dZ_DE_mm']:+.1f} mm rms {lp['common_dZ+DE_shift']['rms_px']:.2f}")
    save_json(result, f"{CACHE}/cuboid_offsets.json")
    # markdown table
    md = ["# Cuboid model: predicted (drawing) vs traced cart edges", "",
          "Script `work/41_cuboid_predict.py`. Offset = traced - predicted along the image normal [px], sign + = traced edge on the + side of",
          "the first axis in [..]; implied shift = move of the drawing line (camera fixed) that puts it onto the traced edge [mm];",
          "cam-sigma = 1-sigma of the mean offset over the bootstrap cameras. Cameras: " +
          "; ".join(f"{k} = {v[0].name}" for k, v in cams.items()), ""]
    for cname in cams:
        tab = result["edges"][cname]
        md += [f"## camera: {cname}", "", "| edge | model line | mean px (start / end) | cam-sigma px | implied 1st axis [mm] (px/mm) | implied 2nd axis [mm] |",
               "|---|---|---|---|---|---|"]
        for eid, v in tab.items():
            a0, a1 = L.EDGE_MAP[eid][1][0], L.EDGE_MAP[eid][1][1]
            md.append(f"| {eid} | {v['model_line']} [{a0}] | {v['mean_px']:+.2f} ({v['start_px']:+.2f} / {v['end_px']:+.2f}) | "
                      f"{v.get('camera_sigma_mean_px', float('nan')):.2f} | d{a0} {v[f'implied_d{a0}_mm']:+.1f} ({v[f'px_per_mm_{a0}']:.3f}) | "
                      f"d{a1} {v[f'implied_d{a1}_mm']:+.1f} |")
        md.append("")
        md.append("* sticker rows (corner RMS px / Z shift of the row that fits its corners best, camera fixed, mm): "
                  + ", ".join(f"{k} {v['rms_px']:.1f}/{v['implied_dZ_mm']:+.0f}" for k, v in result["sticker_rows"][cname].items()))
        for c, lp in result["lip_pattern"][cname].items():
            cd = lp["common_dZ"]
            sc = lp["common_dZ+depth_scale"]
            de = lp["common_dZ+DE_shift"]
            md.append(f"* lips cart {c}: common lip height dZ {cd['dZ_mm']:+.1f} mm (formal +-{cd['dZ_mm_sigma']:.1f}); per-lip deviation from it [px]: "
                      + ", ".join(f"{k} {v:+.2f}" for k, v in cd["per_lip_mean_px"].items()) + f" (rms {cd['rms_px']:.2f} px); "
                      f"+ shelf-depth scale {sc['scale_permille']:+.1f} +- {sc['scale_permille_sigma']:.1f} permille -> rms {sc['rms_px']:.2f} px; "
                      f"+ D/E shift {de['dZ_DE_mm']:+.1f} +- {de['dZ_DE_mm_sigma']:.1f} mm -> rms {de['rms_px']:.2f} px")
        md.append("")
    open(f"{CACHE}/cuboid_offsets.md", "w").write("\n".join(md) + "\n")
    # short section in method_cuboid.md
    E = result["edges"]
    sec = ["Script `work/41_cuboid_predict.py`; full table `work/cache/cuboid_offsets.md/.json`; plots `results/cuboid_overlay.png`, "
           "`results/cuboid_crops.png`, `results/cuboid_crops_anchor.png`, `results/cuboid_offsets.png`.",
           "Offset traced - predicted [px] (+ = traced edge on the + side of the axis in [..]) +- camera 1-sigma (bootstrap, robust) / implied shift [mm]:", "",
           "| edge | markers-only | joint | edge lens + shelf-sticker poses (diag.) |", "|---|---|---|---|"]
    for eid in L.EDGE_MAP:
        a = L.EDGE_MAP[eid][1][0]
        cells = []
        for cname in ("markers_only", "joint", "edge_lens_shelf_poses"):
            v = E[cname][eid]
            s = v.get("camera_sigma_mean_px_robust")
            cells.append(f"{v['mean_px']:+.1f}" + (f" +-{s:.1f}" if s is not None else "") + f" / d{a} {v[f'implied_d{a}_mm']:+.0f}")
        sec.append(f"| {eid} | " + " | ".join(cells) + " |")
    sec.append("")
    for cname in ("markers_only", "joint", "edge_lens_shelf_poses", "edge_lens_top_poses"):
        sr = result["sticker_rows"][cname]
        sec.append(f"* {cname}: sticker rows RMS px / best Z shift mm: " + ", ".join(f"{k} {v['rms_px']:.1f}/{v['implied_dZ_mm']:+.0f}" for k, v in sr.items()))
        for c, lp in result["lip_pattern"][cname].items():
            cd = lp["common_dZ"]
            sec.append(f"  * lips cart {c}: common lip height {cd['dZ_mm']:+.0f} mm, per-lip deviation " + ", ".join(f"{k} {v:+.1f}" for k, v in cd["per_lip_mean_px"].items())
                       + f" px (rms {cd['rms_px']:.2f}); + depth scale -> rms {lp['common_dZ+depth_scale']['rms_px']:.2f}; + D/E shift {lp['common_dZ+DE_shift']['dZ_DE_mm']:+.0f} mm -> rms {lp['common_dZ+DE_shift']['rms_px']:.2f}")
    L.md_replace_section(f"{CACHE}/method_cuboid.md", "## Prediction check: modelled cart lines vs traced edges", sec)

    # -------------------------------------------------------------- full-resolution overlay
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    img = (img.astype(np.float32) * 0.62).astype(np.uint8)
    draw_lines(img, cam_m, (60, 60, 255), 1)          # red   markers-only
    draw_lines(img, cam_e, (0, 220, 255), 1)          # yellow edge lens
    draw_lines(img, cam_j, (255, 255, 0), 1, label=True)  # cyan  joint
    for eid, e in RAW.items():
        col = (0, 255, 0) if eid in L.EDGE_MAP else (170, 170, 170)
        for p in e["points"][::2]:
            cv2.circle(img, (int(round(p[0] * 8)), int(round(p[1] * 8))), 8, col, -1, lineType=cv2.LINE_AA, shift=3)
    for i, (txt, col) in enumerate([("predicted (drawing): markers-only camera", (60, 60, 255)), ("predicted: cuboid joint camera", (255, 255, 0)),
                                    ("predicted: edge-only lens + sticker poses", (0, 220, 255)), ("traced edges compared", (0, 255, 0)),
                                    ("other traced edges", (170, 170, 170))]):
        if i == 0:
            cv2.rectangle(img, (875, 962), (1275, 1072), (0, 0, 0), -1)
        cv2.putText(img, txt, (885, 982 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)
    cv2.imwrite(f"{RESULTS}/cuboid_overlay.png", img)

    # -------------------------------------------------------------- zoomed crops
    render_crops([(cam_m, "red", "markers-only camera"), (cam_j, "cyan", "cuboid joint camera")], f"{RESULTS}/cuboid_crops.png",
                 "Predicted cart lines (drawing dimensions) vs traced edges (green dots = compared edges, grey = other traced edges)")
    render_crops([(cams["edge_lens_shelf_poses"][0], "orange", "edge-only lens, poses from SHELF stickers (A,B,C)"),
                  (cams["edge_lens_top_poses"][0], "magenta", "edge-only lens, poses from TOP stickers")], f"{RESULTS}/cuboid_crops_anchor.png",
                 "DIAGNOSTIC: the same drawing lines with the camera anchored to the shelf stickers only vs to the top stickers only")

    # -------------------------------------------------------------- offsets plot
    ids = list(L.EDGE_MAP)
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), dpi=110, sharex=True)
    xs = np.arange(len(ids))
    series = (("markers_only", "tab:red", "markers-only camera"), ("joint", "tab:cyan", "cuboid joint camera"),
              ("edge_lens", "gold", "edge-only lens + all-sticker poses"), ("edge_lens_shelf_poses", "tab:orange", "edge-only lens + SHELF-sticker poses (diag.)"))
    for j, (cname, col, lab) in enumerate(series):
        tab = result["edges"][cname]
        mean = np.array([tab[e]["mean_px"] for e in ids])
        st = np.array([tab[e]["start_px"] for e in ids])
        en = np.array([tab[e]["end_px"] for e in ids])
        err = np.array([tab[e].get("camera_sigma_mean_px_robust", 0.0) for e in ids])
        dx = (j - 1.5) * 0.18
        ax[0].errorbar(xs + dx, mean, yerr=err, fmt="o", color=col, ms=4, label=lab, capsize=2)
        ax[0].vlines(xs + dx, np.minimum(st, en), np.maximum(st, en), color=col, lw=3, alpha=0.4)
        a0 = [L.EDGE_MAP[e][1][0] for e in ids]
        imp = np.array([tab[e][f"implied_d{a}_mm"] for e, a in zip(ids, a0)])
        ax[1].plot(xs + dx, np.clip(imp, -150, 150), "o", color=col, ms=4, label=lab)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("traced - predicted [px]")
    ax[0].set_title("dot = mean offset along the normal, bar = first..last 10 % of the edge, error bar = camera 1-sigma (bootstrap, robust)", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("implied shift of the 3D line [mm]")
    ax[1].set_title("implied shift along the axis in [..] (Z for lips / rails, X or Y otherwise), camera fixed; clipped at +-150 mm", fontsize=9)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{e.replace('cart', '')} [{L.EDGE_MAP[e][1][0]}]" for e in ids], rotation=75, fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/cuboid_offsets.png")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
