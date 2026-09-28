"""Cart 80: geometric interpretation of the traced automatic candidates with the initial calibration.
For each candidate (length > 50 px) print its direction class (angle to the 3 vanishing directions)
and where the corresponding 3D line would sit:
  cartX line: the level Z at which it would lie on the front plane Y=0 / back plane Y=450
  cartY line: X position if it lies at Z = 0 / A-level / B-level
  cartZ line: X if on the front plane Y=0, Y if on the end planes X=0 / X=1600
Output: printed table + cache/edges80_interpret.json"""
import numpy as np

from common import CACHE, load_json, save_json, SHELF_Z
import edges80lib as L

CLS = ["cartX", "cartY", "cartZ"]


def line_in_plane(P, axis, value, free_axis):
    """back-project points onto plane coord[axis]=value; return mean/std of the remaining fixed coord."""
    Q = L.backproject_plane(P, axis, value)
    other = [a for a in range(3) if a != axis and a != free_axis][0]
    return float(np.mean(Q[:, other])), float(np.std(Q[:, other])), Q


def solve_level(P, fixed_axis, fixed_value, level_axis, lo=-2500, hi=800):
    """find the value v of coordinate `level_axis` such that the X/Y/Z line through the edge has
    coordinate fixed_axis == fixed_value (linear relation, sampled)."""
    vs = np.array([lo, hi], float)
    m = []
    for v in vs:
        Q = L.backproject_plane(P, level_axis, v)
        m.append(np.mean(Q[:, fixed_axis]))
    # linear in v
    a = (m[1] - m[0]) / (vs[1] - vs[0])
    return float(vs[0] + (fixed_value - m[0]) / a)


def main():
    auto = load_json(f"{CACHE}/edges80_auto.json")["edges"]
    rows = []
    for e in auto:
        if e["length"] < 50:
            continue
        P = np.asarray(e["points"])
        ang = np.asarray(e["ang"])
        k = int(np.argmin(ang))
        info = dict(aid=e["aid"], length=round(e["length"]), mid=np.round(P[len(P) // 2]).tolist(),
                    ang=np.round(ang, 1).tolist(), cls=CLS[k] if ang[k] < 3 else "other", bow=round(e["bow"], 2))
        if info["cls"] == "cartX":
            info["Z_if_Y0"] = round(solve_level(P, 1, 0.0, 2))
            info["Z_if_Y450"] = round(solve_level(P, 1, 450.0, 2))
        elif info["cls"] == "cartY":
            for nm, z in [("top", 0.0), ("A", SHELF_Z["A"]), ("B", SHELF_Z["B"])]:
                Q = L.backproject_plane(P, 2, z)
                info[f"X_at_{nm}"] = round(float(np.mean(Q[:, 0])))
            Q = L.backproject_plane(P, 2, 0.0)
            info["Yrange_top"] = [round(float(Q[:, 1].min())), round(float(Q[:, 1].max()))]
        elif info["cls"] == "cartZ":
            Q = L.backproject_plane(P, 1, 0.0)
            info["X_if_Y0"] = round(float(np.mean(Q[:, 0])))
            info["Zrange_if_Y0"] = [round(float(Q[:, 2].min())), round(float(Q[:, 2].max()))]
            Q = L.backproject_plane(P, 0, 1600.0)
            info["Y_if_X1600"] = round(float(np.mean(Q[:, 1])))
            Q = L.backproject_plane(P, 0, 0.0)
            info["Y_if_X0"] = round(float(np.mean(Q[:, 1])))
        rows.append(info)
    rows.sort(key=lambda r: (r["cls"], r["mid"][1]))
    for r in rows:
        print(r)
    save_json(rows, f"{CACHE}/edges80_interpret.json")


if __name__ == "__main__":
    main()
