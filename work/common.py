"""Shared helpers for the blind lens calibration (camera model, spec, IO).

Conventions
-----------
* Pixel coordinates: OpenCV (x right, y down, pixel centre of the top-left pixel = (0, 0)).
* Camera model: OpenCV pinhole + Brown-Conrady, dist = [k1, k2, p1, p2, k3].
* Cart frame (spec/cart-marker-layout.json): origin top front-left corner, X along the 1600 mm
  front, Y into the cart (depth 450), Z up, Z = 0 at the top plate.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SPEC = os.path.join(ROOT, "spec")
RESULTS = os.path.join(ROOT, "results")
WORK = os.path.join(ROOT, "work")
CACHE = os.path.join(WORK, "cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

W, H = 1920, 1080


def still_paths():
    return sorted(glob.glob(os.path.join(DATA, "stills", "*.jpg")))


def load_gray(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def load_spec():
    with open(os.path.join(SPEC, "cart-marker-layout.json")) as f:
        return json.load(f)


SPECJ = load_spec()
CART_W = float(SPECJ["cart"]["widthMm"])  # 1600
CART_D = float(SPECJ["cart"]["depthMm"])  # 450
TOP_H = float(SPECJ["cart"]["heightAboveFloorMm_topPlate"])  # 1800
SHELF_H = {k: float(v) for k, v in SPECJ["cart"]["shelfSurfaceHeightAboveFloorMm"].items()}
SHELF_Z = {k: -(TOP_H - v) for k, v in SHELF_H.items()}  # A:-195 B:-595 C:-995 D:-1300 E:-1605
FLOOR_Z = -TOP_H
CODE_MM = float(SPECJ["codeMm"])  # 90 (black square incl. border)
HALF = CODE_MM / 2.0
CARTS = {int(k): int(v["uniqueId"]) for k, v in SPECJ["cartsInStills"].items()}  # {80: 92, 310: 322}


def marker_center(mid, cart):
    """3D centre (mm, cart frame) of marker id `mid` on cart number `cart`."""
    if mid == 12 + cart:
        return np.array(SPECJ["uniqueMarker"]["centerMm"], float)
    return np.array(SPECJ["markers"][str(mid)]["centerMm"], float)


def marker_role(mid, cart):
    if mid == 12 + cart:
        return SPECJ["uniqueMarker"]["role"]
    return SPECJ["markers"][str(mid)]["role"]


def cart_marker_ids(cart):
    return list(range(9)) + [12 + cart]


# canonical ArUco corner order in the marker frame (x right, y up, z out of the face):
# TL, TR, BR, BL  (OpenCV detectMarkers order)
_CANON = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], float)


def marker_corners_3d(mid, cart, rot_k, half=HALF):
    """3D corners (4x3, mm, cart frame) in OpenCV detection order.

    The sticker lies face up (marker z = +Z of the cart); its in-plane rotation is rot_k*90 deg
    (marker x-axis = R_z(rot_k*90) @ cart X). A mirror is impossible for a printed marker.
    """
    c = marker_center(mid, cart)
    a = np.deg2rad(90.0 * rot_k)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = (R @ (_CANON * half).T).T
    return np.column_stack([c[0] + xy[:, 0], c[1] + xy[:, 1], np.full(4, c[2])])


# ----------------------------------------------------------------------------------------
# camera model (numpy, vectorised) -- identical to cv2.projectPoints for the Brown model
# ----------------------------------------------------------------------------------------

def rodrigues(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def distort_normalized(xn, yn, dist):
    """Apply OpenCV distortion to normalised coords. dist = [k1,k2,p1,p2,k3,(k4,k5,k6)]."""
    d = np.zeros(8)
    d[: len(dist)] = dist
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r2 = xn * xn + yn * yn
    rad = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd


def project(X, K, dist, rvec=None, tvec=None, R=None):
    """Project Nx3 points; returns Nx2 pixels. Either (rvec,tvec) or (R,tvec)."""
    X = np.asarray(X, float).reshape(-1, 3)
    if R is None:
        R = rodrigues(rvec) if rvec is not None else np.eye(3)
    t = np.zeros(3) if tvec is None else np.asarray(tvec, float).ravel()
    Xc = X @ R.T + t
    xn = Xc[:, 0] / Xc[:, 2]
    yn = Xc[:, 1] / Xc[:, 2]
    xd, yd = distort_normalized(xn, yn, dist)
    return np.column_stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]])


def undistort_points(uv, K, dist, iters=50):
    """Pixel -> normalised undistorted coords (Nx2), robust fixed-point + Newton polish."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
    xd = (uv[:, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        fx, fy = distort_normalized(x, y, dist)
        ex, ey = fx - xd, fy - yd
        # numeric Jacobian (2x2 per point)
        h = 1e-7
        ax, ay = distort_normalized(x + h, y, dist)
        bx, by = distort_normalized(x, y + h, dist)
        j11, j21 = (ax - fx) / h, (ay - fy) / h
        j12, j22 = (bx - fx) / h, (by - fy) / h
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        dx = (j22 * ex - j12 * ey) / det
        dy = (-j21 * ex + j11 * ey) / det
        x -= dx
        y -= dy
        if np.max(np.abs(dx) + np.abs(dy)) < 1e-13:
            break
    return np.column_stack([x, y])


def undistort_to_pixels(uv, K, dist, Knew=None):
    Knew = K if Knew is None else Knew
    n = undistort_points(uv, K, dist)
    return np.column_stack([Knew[0, 0] * n[:, 0] + Knew[0, 2], Knew[1, 1] * n[:, 1] + Knew[1, 2]])


def K_from(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def lens_json(K, dist, **extra):
    d = list(np.asarray(dist, float).ravel()) + [0.0] * 5
    out = {
        "image_width": W,
        "image_height": H,
        "camera_matrix": [[float(K[0, 0]), 0.0, float(K[0, 2])], [0.0, float(K[1, 1]), float(K[1, 2])], [0.0, 0.0, 1.0]],
        "dist_coeffs": [float(v) for v in d[:5]],
    }
    out.update(extra)
    return out


def save_json(obj, path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# image regions for the mapping uncertainty: see evaltools.region_defs (single definition)


# ----------------------------------------------------------------------------------------
# fold check: a radial Brown model is only usable if r_d(r_u) is monotonic out to the image corners
# ----------------------------------------------------------------------------------------
def radial_fold(dist, r_max=4.0, n=4000):
    """Return (r_u_fold, r_d_max): first radius where d r_d / d r_u <= 0 (or r_max) and the largest
    distorted radius reachable before it (normalised units). Tangential terms are ignored."""
    d = np.zeros(8)
    d[: len(dist)] = dist
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r = np.linspace(0, r_max, n)
    r2 = r * r
    rd = r * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    dr = np.diff(rd)
    bad = np.nonzero(dr <= 0)[0]
    i = bad[0] if len(bad) else n - 1
    return float(r[i]), float(np.max(rd[: i + 1]))


def corner_radius_norm(K):
    c = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)
    x = (c[:, 0] - K[0, 2]) / K[0, 0]
    y = (c[:, 1] - K[1, 2]) / K[1, 1]
    return float(np.max(np.hypot(x, y)))


def fold_margin(K, dist):
    """> 0: model invertible out to the farthest image corner (margin in normalised radius); < 0: folded."""
    _, rdmax = radial_fold(dist)
    return rdmax - corner_radius_norm(K)
