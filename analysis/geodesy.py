"""ECEF <-> geodetic and local ENU conversions (WGS84)."""
import numpy as np

from analysis import _paths  # noqa: F401  (puts processor/ on sys.path)
from CONSTANTS import WGS84_A, WGS84_E2


def ecef2lla(x, y, z):
    """ECEF (m) -> (lat, lon) in radians and ellipsoidal height (m). Iterative Bowring."""
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return lat, lon, h


def enu_rotation(lat, lon):
    """3x3 matrix that takes an ECEF vector to ENU at the point (lat, lon)."""
    sla, cla = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)
    return np.array([
        [-slo,        clo,       0.0],
        [-sla * clo, -sla * slo, cla],
        [cla * clo,   cla * slo, sla],
    ])


def ecef_delta_to_enu(dxyz, lat, lon):
    """Converts ECEF increments (N,3) to ENU (N,3) at the given reference point."""
    return np.asarray(dxyz) @ enu_rotation(lat, lon).T
