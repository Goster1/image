"""Scene region (everything that is not one of the two carts): final edge set.

Re-traces all curated edges (edgesscene_defs.py via 10_edges_scene_trace.run: sub-pixel tracing,
cleaning, verification crops work/cache/edgecrops_scene/<id>.png), applies the visual-verification table
below (what was checked on the zoomed crops, direction class, keep/reject) and writes
  work/cache/edges_scene.json       (edge JSON schema shared by all regions; accepted edges only)
  work/cache/edges_scene_diag.json  (per-edge diagnostics: length, noise, vanishing-direction angles, rejected edges)
  results/edges_scene.png           (full-resolution colour mean image with the edges, coloured by direction class)

Direction classes: floor lines = floor_plane; 'world_vertical' only where several edges of the same rigid
object all point to the nadir within ~0.2-3.3 deg (classification lens, see edgesscene_lib.class_calib);
'horizontal_other' = clearly NOT through the nadir (> 40 deg), physically horizontal members roughly along
the aisle (their exact 3D direction is not known, they are not assigned to a cart axis).
"""
import importlib.util
import os

import cv2
import numpy as np

import edgelib
import edgesscene_lib as L
from common import RESULTS, save_json
from edgesscene_defs import E

spec = importlib.util.spec_from_file_location("trace", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "10_edges_scene_trace.py"))
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)

# id -> (verified, note, direction, keep)
REVIEW = {
    "scene_tapeV_left": ("yes", "strip + 4 windows x7: points sit on the steep dark-halo -> white transition along the "
                         "whole tape (y 690-1078); a small floor pebble at y~835 does not disturb it", "floor_plane", True),
    "scene_tapeV_right": ("yes", "strip + windows: points on the white -> dark-halo transition over the full length "
                          "(y 686-1078), uniform", "floor_plane", True),
    "scene_tapeH_top": ("yes", "strip + windows: points on the blue -> white tape boundary, x 872-1150; blue paint "
                        "above is scratched but the tape edge itself is sharp", "floor_plane", True),
    "scene_tapeH_bottom": ("partly", "only x 975-1140 kept (further left the lit floor is as bright as the tape); "
                           "the tape edge is slightly ragged (dirt), noise ~0.25 px", "floor_plane", True),
    "scene_blueH_top": ("partly", "short (85 px, x 1060-1145): the rest of the blue line is worn / scratched and was "
                        "excluded; low gray contrast", "floor_plane", True),
    "scene_blueL_top": ("yes", "strip + windows: clean boundary floor -> blue paint line x 172-430; a floor crack "
                        "crosses at x~262 without visible disturbance", "floor_plane", True),
    "scene_blueL_bottom": ("yes", "strip + windows: clean boundary blue -> floor; slight undulation of the painted "
                           "edge (<0.3 px)", "floor_plane", True),
    "scene_joint450": ("partly", "centre of the saw-cut floor joint (dark line ~3 px wide) x 826-1155; the joint "
                       "filling is ragged -> point noise ~0.5 px, no systematic kink seen", "floor_plane", True),
    "scene_jointB_H": ("no", "diffuse wide dark band (y~970), no sharp edge; traced points scatter ~1 px", "floor_plane",
                       False),
    "scene_jointB_V_left": ("partly", "short (94 px) left edge of a filled floor joint near the bottom border; "
                            "clean but chipped in places", "floor_plane", True),
    "scene_jointB_V_right": ("partly", "short (80 px, one gap) right edge of the same floor joint", "floor_plane",
                             True),
    "scene_blue_left_upper": ("yes", "outer left edge of the tall blue object above the notice sheet (y 742-830); "
                              "red tape band at y~775 crosses it, edge continuous", "world_vertical", True),
    "scene_blue_left_lower": ("yes", "outer left edge below the white frame (y 1019-1077), row profiles confirm a "
                              "straight step of ~50 gray levels; NOTE: upper and lower parts were traced separately "
                              "because, taken together, they deviate ~0.9 px from one smooth curve (not simply "
                              "collinear) - keep them as two edges", "world_vertical", True),
    "scene_blue_right": ("yes", "outer right edge against the floor, y 993-1077 only: the upper part (y 900-990, "
                         "next to / partly under a translucent sheet) was traced too but is offset ~0.7-1.2 px from "
                         "the lower part, so it was excluded", "world_vertical", True),
    "scene_blue_face": ("no", "fold between the dark top face and the lighter side face: very low contrast "
                        "(strength ~4), local systematic deviations up to 0.7 px along the trace", "world_vertical",
                        False),
    "scene_rack_tube_a": ("partly", "right edge of the light tube of the tall rack at the far-left border, y 10-620 "
                          "(609 px); goods touch it in places (5 gaps) but all pieces lie on one smooth curve "
                          "(piece offsets < 0.1 px, residual 0.34 px); low contrast", "horizontal_other", True),
    "scene_rack_base": ("no", "thin dark line at the bottom of the rack side: a kink/step visible in the zoomed "
                        "window, cannot be verified as one straight member", "unknown", False),
    "scene_trolley_rodA": ("partly", "short (67 px) edge of a tube of the grey wire trolley in the bottom-left "
                           "corner, clean", "unknown", True),
    "scene_trolley_rodB": ("partly", "short (66 px) edge of a second trolley tube, clean", "unknown", True),
    "scene_rpost_rod": ("yes", "left edge of the long light pipe on the right wall, y 340-840 (460 px), sharp and "
                        "uniform; the white lamp at (1697,640) excluded", "horizontal_other", True),
    "scene_rpost_rodR": ("yes", "right edge of the same pipe, y 340-628 only (below, a light board rim runs "
                         "alongside)", "horizontal_other", True),
    "scene_beam_lower": ("partly", "lower edge of the light grey column/bar top right against the dark gap above "
                         "the notice board, x 1771-1835 (60 px); the continuation to the left (against the dark "
                         "opening, x 1717-1761) was traced too but is offset ~0.8 px -> excluded", "world_vertical",
                         True),
    "scene_beam_upper": ("no", "upper edge of the same bar against the concrete wall x 1632-1762: low contrast "
                         "(strength ~10), a bump at the bolt head, and it stays curved by ~1 px rms after a "
                         "k1,k2 straightening that fits the other scene edges to ~0.2 px -> not reliable",
                         "world_vertical", False),
    "scene_leg_left": ("yes", "short (65 px) sharp edge of the dark conveyor leg under the bollard", "horizontal_other",
                       True),
    "scene_leg_right": ("partly", "short (96 px) right edge of the conveyor leg (leg -> black gap), low contrast",
                        "horizontal_other", True),
    "scene_roller_1": ("no", "roller of the curved conveyor top right: the traced boundary is a highlight "
                       "boundary on a dark roller, not a silhouette", "unknown", False),
    "scene_roller_2": ("no", "roller top right: trace jumps between the highlight and the silhouette (step in the "
                       "strip)", "unknown", False),
    "scene_roller_br1": ("partly", "roller conveyor bottom right: sharp upper boundary of the bright roller "
                         "(interpreted as the occluding contour; a highlight boundary cannot be fully excluded), "
                         "108 px, noise 0.06 px", "horizontal_other", True),
    "scene_roller_br2": ("partly", "next roller, same kind of boundary, 105 px", "horizontal_other", True),
    "scene_pole_top": ("partly", "upper silhouette of the yellow-black bollard, yellow bands only (black bands have "
                       "no contrast), end cap excluded; 3 pieces over 78 px", "world_vertical", True),
    "scene_pole_bottom": ("partly", "lower silhouette of the bollard, yellow bands only, end cap excluded",
                          "world_vertical", True),
    "scene_frame_br": ("yes", "right edge of the light aluminium frame strip right of the poster board against the "
                       "dark machine, y 785-1076 (271 px), sharp; one small notch (dark object) near y~990",
                       "horizontal_other", True),
    "scene_wall_railA": ("partly", "right edge of the right one of two thin light rods on the wall, y 300-620, three "
                         "pieces (small crossing elements), local offsets up to 0.3 px", "horizontal_other", True),
    "scene_wall_railB": ("yes", "centre of a second thin light rod on the wall, y 300-690, clean over the whole "
                         "length", "horizontal_other", True),
}

DIR_NOTE = {
    "world_vertical": "every edge of this object points to the nadir within {a:.1f} deg (classification lens); "
                      "interpreted as physically vertical",
    "horizontal_other": "{a:.0f} deg away from the nadir direction -> not vertical; horizontal member roughly "
                        "along the aisle, exact 3D direction unknown (not assigned to a cart axis)",
}


def fragments(P, s, gap=10.0):
    cut = np.where(np.diff(s) > gap)[0]
    st = np.r_[0, cut + 1]
    en = np.r_[cut + 1, len(P)]
    return [P[a:b] for a, b in zip(st, en)]


def main():
    missing = [e["id"] for e in E if e["id"] not in REVIEW]
    assert not missing, missing
    rows = TR.run(E, render=True)
    edges, diag, rejected = [], [], []
    for r in rows:
        ver, note, direction, keep = REVIEW[r["id"]]
        P = np.asarray(r["points"], float)
        S = np.asarray(r["strength"], float)
        vpz = min(r.get("vp_angles", {}).get("80Z", 90), r.get("vp_angles", {}).get("310Z", 90))
        d = dict(id=r["id"], n=int(len(P)), length_px=r.get("length"), noise_px=r.get("noise"),
                 med_strength=r.get("med_strength"), vp_angles_deg=r.get("vp_angles"),
                 angle_to_nadir_deg=vpz, undist_rms_class_lens=r.get("undist_rms"), kind=r["kind"],
                 direction=direction, verified=ver, kept=keep)
        diag.append(d)
        if not keep or len(P) < 10:
            rejected.append(dict(id=r["id"], what=r["what"], reason=note))
            continue
        c, dd, rms, mx = edgelib.line_fit(P)
        bow, _ = edgelib.sagitta(P)
        dn = DIR_NOTE.get(direction)
        vnote = note + ("; direction: " + dn.format(a=vpz) if dn else "")
        what = r["what"]
        if r["kind"] != "edge":
            what += " [points = centre line of a thin line, not a step edge]"
        edges.append({
            "id": r["id"],
            "what": what,
            "class": r["cls"],
            "cart": None,
            "straight_3d": bool(r["straight_3d"]),
            "direction": direction,
            "model_line": None,
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in P],
            "strength": [round(float(v), 2) for v in S],
            "chord_dev_rms_px": round(float(rms), 3),
            "chord_bow_px": round(float(bow), 3),
            "verified": ver,
            "verification_note": vnote,
            "crop": f"work/cache/edgecrops_scene/{r['id']}.png",
        })
    out = {"region": "scene",
           "image": "work/cache/mean_aligned_gray.npy (jitter-compensated mean of the 7 stills)",
           "edges": edges}
    save_json(out, f"{L.CACHE}/edges_scene.json")
    K, dist, poses, src = L.class_calib()
    save_json({"classification_lens": {"source": src, "K": K, "dist": dist,
                                       "pnp_rms_px": {str(c): p[2] for c, p in poses.items()}},
               "edges": diag, "rejected": rejected}, f"{L.CACHE}/edges_scene_diag.json")

    # overview image
    img = L.color_image()
    for e, r in zip([x for x in edges], [x for x in rows if REVIEW[x["id"]][3] and len(x["points"]) >= 10]):
        col = L.DIRCOL[e["direction"]]
        P = np.asarray(r["points"], float)
        for F in fragments(P, np.asarray(r.get("s", np.arange(len(P)) * 2.0))):
            if len(F) >= 2:
                cv2.polylines(img, [np.round(F * 8).astype(np.int32)], False, col, 1, cv2.LINE_AA, shift=3)
        frac = (0.3, 0.7)[len(edges) and [x["id"] for x in edges].index(e["id"]) % 2]
        c = P[int(frac * (len(P) - 1))]
        lab = e["id"].replace("scene_", "")
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
        x0 = int(min(max(c[0] + 6, 2), 1918 - tw))
        y0 = int(min(max(c[1] + 4, th + 3), 1077))
        cv2.rectangle(img, (x0 - 1, y0 - th - 2), (x0 + tw + 1, y0 + 2), (0, 0, 0), -1)
        cv2.putText(img, lab, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 0.8, col, 1, cv2.LINE_AA)
    y = 20
    for k, col in L.DIRCOL.items():
        if k in {e["direction"] for e in edges}:
            (tw, th), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (758, y - th - 3), (762 + tw, y + 3), (0, 0, 0), -1)
            cv2.putText(img, k, (760, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y += 18
    cv2.imwrite(f"{RESULTS}/edges_scene.png", img)

    # summary
    by = {}
    for e in edges:
        P = np.asarray(e["points"])
        L_ = float(np.ptp((P - P.mean(0)) @ edgelib.line_fit(P)[1]))
        b = by.setdefault(e["direction"], [0, 0.0])
        b[0] += 1
        b[1] += L_
    print("accepted", len(edges), "rejected", len(rejected))
    for k, (n, l) in by.items():
        print(f"  {k:18s} n={n:3d} total chord length {l:7.0f} px")
    print("  verified:", {v: sum(e["verified"] == v for e in edges) for v in ("yes", "partly")})


if __name__ == "__main__":
    main()
