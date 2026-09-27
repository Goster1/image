"""Sub-task 43, DIAGNOSTIC ONLY (not a main estimate): which lens model do the stickers prefer once the
known geometry mismatch is relaxed, and how sensitive is the model choice to single edges?

(a) Sticker fits with 5 extra geometry parameters (labelled diagnostic, per geometry_diagnosis.md the most
    compact explanation): height offset of each of the 4 end boards (top stickers) + one common code size.
    Drawing dimensions are NOT changed in any main estimate; this only tells whether the lens-model
    preferences seen in the drawing-geometry fits (p1,p2 / k3 / rational) are artefacts of the geometry.
(b) Joint (stickers + edges) with the same relaxed geometry, Brown k1,k2 vs k1,k2,p1,p2.
(c) Edge sensitivity: edges-only fits without the long rack tube at the far-left border
    (scene_rack_tube_a, bow 16 px, not verified straight to 0.6 px) and without all scene edges.
Writes work/cache/altmodels_diaggeom.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib  # noqa: E402

import numpy as np  # noqa: E402

from common import CACHE, marker_center, save_json, load_json  # noqa: E402

L = importlib.import_module("43_altmodels_lib")
F = importlib.import_module("43_altmodels_fit")
FITS = load_json(f"{CACHE}/altmodels_fits.json")


class EstG(L.Est):
    """Est + 5 geometry parameters appended to x: dz of the 4 end boards [80/0, 80/1600, 310/0, 310/1600] (mm)
    and the code size scale s (code = 90 mm * s)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.X0 = self.mX.copy()
        cen = np.array([marker_center(p["id"], p["cart"]) for p in self.mp]).reshape(-1, 3)
        self.cen = cen
        self.top = cen[:, 2] == 0.0
        self.board = np.array([(0 if p["cart"] == 80 else 2) + (0 if c[0] < 800 else 1) for p, c in zip(self.mp, cen)], int)
        self.ng = 5

    def blocks(self, x):
        g = x[self.n:self.n + 5]
        X = self.cen + g[4] * (self.X0 - self.cen)
        X[:, 2] = self.cen[:, 2] + np.where(self.top, g[self.board], 0.0)
        self.mX = X
        return super().blocks(x[: self.n])

    def residuals(self, x):
        r = super().residuals(x)
        return r


def fit_geom(name, ds, sig, start):
    D = F.get_data()
    lens = F.make_lens(name)
    data = L.build(D, ds)
    est = EstG(lens, data, sig)
    best = None
    for Pi in ([start] if ds != "M" else F.start_variants(start)):
        x0 = np.r_[est.x0(Pi, {int(k): np.array(v) for k, v in FITS["fits"][ds][name]["poses"].items()},
                          np.array(FITS["fits"][ds][name]["wv"]) if FITS["fits"][ds][name]["wv"] else None,
                          FITS["fits"][ds][name].get("floor_ang")), np.zeros(4), 1.0]
        r = est.solve(x0)
        if best is None or r.cost < best.cost:
            best = r
    r = best
    P = lens.P(r.x[: lens.ni])
    g = r.x[est.n:est.n + 5]
    return dict(P={k: float(v) for k, v in P.items()}, chi2=float(np.sum(r.fun[:-1] ** 2)), k=int(len(r.x)),
                block_stats=est.block_stats(r.x), board_dz_mm=g[:4].tolist(), code_mm=float(90 * g[4]),
                fold_margin=lens.fold_margin(P))


def main():
    out = dict(note="DIAGNOSTIC: extra geometry parameters (board heights, code size) are NOT used in any main estimate",
               M={}, ME={}, edges={})
    for name in ("B_k1", "B_k1k2", "B_k1k2k3", "B_k1k2p1p2", "B_k1k2_fxfy", "B_k1k2_ppfix", "KB_k1k2", "D_l1", "D_l1l2", "R_k1k2k4"):
        P0 = F.make_lens(name).P(np.array(FITS["fits"]["E"][name]["x"][: FITS["fits"]["E"][name]["k_intr"]]))
        res = fit_geom(name, "M", FITS["sig"]["M"], P0)
        res["chi2_drawing"] = FITS["fits"]["M"][name]["chi2"]
        out["M"][name] = res
        print("M relaxed", name, {k: round(v, 4) for k, v in res["P"].items() if v}, "RMS", round(res["block_stats"]["M"]["rms_radial"], 3),
              "dz", np.round(res["board_dz_mm"], 1), "code", round(res["code_mm"], 2), flush=True)
    for name in ("B_k1k2", "B_k1k2p1p2", "B_k1k2k3", "KB_k1k2"):
        P0 = F.make_lens(name).P(np.array(FITS["fits"]["ME"][name]["x"][: FITS["fits"]["ME"][name]["k_intr"]]))
        res = fit_geom(name, "ME", FITS["sig"]["ME"], P0)
        res["chi2_drawing"] = FITS["fits"]["ME"][name]["chi2"]
        out["ME"][name] = res
        print("ME relaxed", name, {k: round(v, 4) for k, v in res["P"].items() if v}, res["block_stats"], flush=True)
    # ---- edge sensitivity
    D = F.get_data()
    variants = {"no_rack_tube": ["scene_rack_tube_a"],
                "no_scene": [e["id"] for e in D["edges"] if e["region"] == "scene"]}
    for vname, drop in variants.items():
        out["edges"][vname] = {}
        for name in ("B_k1k2", "B_k1k2k3", "B_k1k2p1p2", "KB_k1k2", "D_l1l2", "R_k1k2k4", "B_k1k2_ppfix"):
            f = FITS["fits"]["E"][name]
            lens = F.make_lens(name)
            P0 = lens.P(np.array(f["x"][: f["k_intr"]]))
            lens, est, r = F.fit_one(name, "E", FITS["sig"]["E"], P0, {int(k): np.array(v) for k, v in f["poses"].items()},
                                     np.array(f["wv"]) if f["wv"] else None, drop_edges=drop, floor_ang=f.get("floor_ang"))
            P = lens.P(r.x[: lens.ni])
            uv, _ = L.pixel_grid(40)
            d = L.mapping_disp(lens, P0, lens, P, uv, compensate=True)
            out["edges"][vname][name] = dict(P={k: float(v) for k, v in P.items()}, block_stats=est.block_stats(r.x),
                                             mapping_vs_all_edges=L.region_stats(d, uv), fold_margin=lens.fold_margin(P))
            print("edges", vname, name, {k: round(v, 4) for k, v in P.items() if v}, "map corners",
                  round(out["edges"][vname][name]["mapping_vs_all_edges"]["corners"]["median"], 2), flush=True)
    save_json(out, f"{CACHE}/altmodels_diaggeom.json")
    print("done")


if __name__ == "__main__":
    main()
