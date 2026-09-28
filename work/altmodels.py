"""Alternative lens models behind one interface (for model comparison).

Each model: params dict -> project(Xc) (camera-frame points, N x 3 -> N x 2 px) and rays(uv).
  brown    : OpenCV pinhole + k1,k2,k3,p1,p2 (== common.project)
  kb       : Kannala-Brandt / cv2.fisheye: theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
  division : Fitzgibbon division model (1 or 2 params), defined by its UNdistortion in normalised
             units: x_u = x_d / (1 + l1 r_d^2 + l2 r_d^4); projection solved numerically.
"""
from __future__ import annotations

import numpy as np

from common import K_from, distort_normalized, undistort_points


def _norm(uv, f, cx, cy):
    return np.column_stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f])


class Brown:
    names = ["f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]

    def __init__(self, p):
        self.p = dict(p)

    def dist(self):
        g = self.p.get
        return np.array([g("k1", 0), g("k2", 0), g("p1", 0), g("p2", 0), g("k3", 0)])

    def K(self):
        return K_from(self.p["f"], self.p["f"], self.p["cx"], self.p["cy"])

    def project(self, Xc):
        x = Xc[:, 0] / Xc[:, 2]
        y = Xc[:, 1] / Xc[:, 2]
        xd, yd = distort_normalized(x, y, self.dist())
        return np.column_stack([self.p["f"] * xd + self.p["cx"], self.p["f"] * yd + self.p["cy"]])

    def rays(self, uv):
        n = undistort_points(uv, self.K(), self.dist())
        r = np.column_stack([n, np.ones(len(n))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)


class KB:
    names = ["f", "cx", "cy", "k1", "k2", "k3", "k4"]

    def __init__(self, p):
        self.p = dict(p)

    def _thd(self, th):
        g = self.p.get
        return th * (1 + g("k1", 0) * th ** 2 + g("k2", 0) * th ** 4 + g("k3", 0) * th ** 6 + g("k4", 0) * th ** 8)

    def project(self, Xc):
        a = Xc[:, 0] / Xc[:, 2]
        b = Xc[:, 1] / Xc[:, 2]
        r = np.hypot(a, b)
        th = np.arctan(r)
        s = np.where(r > 1e-12, self._thd(th) / np.maximum(r, 1e-12), 1.0)
        return np.column_stack([self.p["f"] * a * s + self.p["cx"], self.p["f"] * b * s + self.p["cy"]])

    def rays(self, uv):
        n = _norm(uv, self.p["f"], self.p["cx"], self.p["cy"])
        rd = np.hypot(n[:, 0], n[:, 1])
        th = rd.copy()
        for _ in range(60):
            g = self._thd(th) - rd
            h = 1e-7
            dg = (self._thd(th + h) - self._thd(th)) / h
            th = th - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
        r = np.tan(th)
        s = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
        v = np.column_stack([n[:, 0] * s, n[:, 1] * s, np.ones(len(n))])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Division:
    names = ["f", "cx", "cy", "l1", "l2"]

    def __init__(self, p):
        self.p = dict(p)

    def _und(self, xd, yd):
        r2 = xd * xd + yd * yd
        s = 1 + self.p.get("l1", 0) * r2 + self.p.get("l2", 0) * r2 * r2
        return xd / s, yd / s

    def rays(self, uv):
        n = _norm(uv, self.p["f"], self.p["cx"], self.p["cy"])
        xu, yu = self._und(n[:, 0], n[:, 1])
        v = np.column_stack([xu, yu, np.ones(len(n))])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def project(self, Xc):
        xu = Xc[:, 0] / Xc[:, 2]
        yu = Xc[:, 1] / Xc[:, 2]
        ru = np.hypot(xu, yu)
        rd = ru.copy()
        l1, l2 = self.p.get("l1", 0), self.p.get("l2", 0)
        for _ in range(60):  # solve rd / (1 + l1 rd^2 + l2 rd^4) = ru
            g = rd - ru * (1 + l1 * rd ** 2 + l2 * rd ** 4)
            dg = 1 - ru * (2 * l1 * rd + 4 * l2 * rd ** 3)
            rd = rd - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
        s = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
        return np.column_stack([self.p["f"] * xu * s + self.p["cx"], self.p["f"] * yu * s + self.p["cy"]])


MODELS = {"brown": Brown, "kb": KB, "division": Division}
