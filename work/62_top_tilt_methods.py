"""What-if (user request): the main combined estimator with the top stickers tilted by the MEASURED plate tilts
(61_top_sticker_orientation_fixed.py: pose of each sticker from its own four corners, main lens) and the method
entries of the tilt study for lens_by_method.json.

Tilt conventions as in 60_top_tilt_scan_fixed.py: pitch about the cart Y axis, + = outer end of the plate up;
roll about the cart X axis, + = back side up; pivot 'centre' = each sticker about its own centre (drawing position
kept), 'hinge' = the plate about a hinge 55 mm inward from the sticker centre (the centre rises by 55 sin(pitch)).
All other geometry is the drawing. The main result (drawing geometry) is NOT changed.

Writes work/cache/top_tilt_measured.json and work/cache/method_top_tilt.json.
"""
import importlib
import sys

import numpy as np

_argv = sys.argv
sys.argv = ["60_top_tilt_scan_fixed.py", "none"]
S = importlib.import_module("60_top_tilt_scan_fixed")
sys.argv = _argv
from common import CACHE, K_from, lens_json, load_json, save_json  # noqa: E402

O = load_json(f"{CACHE}/top_sticker_orientation.json")
J = load_json(f"{CACHE}/top_tilt_scan.json")
MEAS = {pl: (O["plates"][pl]["value"]["pitch_outer"], O["plates"][pl]["value"]["roll"]) for pl in S.PLATES}
print("measured plate tilts (pitch, roll):", {k: (round(a, 2), round(b, 2)) for k, (a, b) in MEAS.items()})

recs = {}
for piv in ("centre", "hinge"):
    cfg = dict(name=f"measured_{piv}", family=f"measured_{piv}", angle=None,
               plate_tilts={pl: (a, b, piv) for pl, (a, b) in MEAS.items()})
    recs[piv] = S.evaluate(cfg)
save_json(dict(measured_tilts_deg=MEAS, records=recs, conventions=__doc__), f"{CACHE}/top_tilt_measured.json")

GEOM = ("WHAT-IF on request, not the drawing: the top stickers are tilted ({}); every other dimension and position "
        "exactly as in spec/cart-marker-layout.json")


def entry(rec, method, geom_note):
    c = rec["combined"]
    K = K_from(c["f"], c["f"], c["cx"], c["cy"])
    d = np.array([c["k1"], c["k2"], 0.0, 0.0, 0.0])
    return lens_json(K, d, method=method, model="k1k2, fx = fy, pp free (main combined estimator)",
                     rms_reprojection_error_px=None, uncertainty={},
                     mapping_uncertainty_px={"centre": None, "cart_band": None, "corners": None},
                     data_used="as the main result (62 sticker corners, 13 sticker sides, 61 edges)",
                     geometry_assumptions=GEOM.format(geom_note),
                     details=dict(plate_tilts=rec["plate_tilts"], block_sigmas=c["sig"],
                                  sticker_rms_combined_lens=rec["stickers_combined_lens"]["overall"],
                                  sticker_rms_top=rec["stickers_combined_lens"]["top"],
                                  sticker_only_pp_fixed={k: rec["sticker_only"]["pp_fixed"][k] for k in ("f", "k1", "k2", "rms", "folds_inside_image")},
                                  mapping_vs_main=rec["mapping_vs_main"],
                                  uncertainty_note="not evaluated separately; the main-result uncertainty applies (the lens differs "
                                                   "from the main result by < 1 px of mapping)"))


pl_Y = J["per_plate"]["Y"]["steps"][-1]["combined_record"]
com = {c["name"]: c for c in J["common_angle"]}
out = [
    entry(recs["centre"], "top_tilt_measured_centre (main estimator, top stickers tilted by the measured plate tilts, each about its centre)",
          "measured plate pitch/roll from the sticker shapes, each sticker about its own centre"),
    entry(recs["hinge"], "top_tilt_measured_hinge (main estimator, top plates tilted by the measured tilts about a hinge 55 mm inward)",
          "measured plate pitch/roll, plate hinged 55 mm inward (sticker centres rise by up to 13 mm)"),
    entry(com["Y_centre_common_+9.25"], "top_tilt_common_Y_centre_+9.25 (main estimator, all top stickers pitched +9.25 deg, outer end up)",
          "common pitch +9.25 deg about each sticker centre (best common angle of the scan)"),
    entry(com["Y_hinge_common_+18.87"], "top_tilt_common_Y_hinge_+18.87 (main estimator, all top plates pitched +18.9 deg about a hinge 55 mm inward)",
          "common pitch +18.87 deg about a hinge 55 mm inward (best of this family; the centres rise by 18 mm)"),
    entry(pl_Y, "top_tilt_fitted_per_plate (main estimator, per-plate pitch fitted to the sticker corners with the main lens)",
          "per-plate pitch fitted to the corner positions (80/X0 +12.3, 80/X1600 +6.1, 310/X0 +15.7, 310/X1600 -4.7 deg)"),
]
save_json(out, f"{CACHE}/method_top_tilt.json")
for e in out:
    print(f"{e['method'][:60]:60s} f {e['camera_matrix'][0][0]:.2f} cx {e['camera_matrix'][0][2]:.2f} cy {e['camera_matrix'][1][2]:.2f} "
          f"k1 {e['dist_coeffs'][0]:+.4f} rms {e['details']['sticker_rms_combined_lens']['rms']:.2f} "
          f"band {e['details']['mapping_vs_main']['cart_band']['median']:.2f}")
