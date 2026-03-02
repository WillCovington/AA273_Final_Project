"""
gravity/gravity_model.py

- loads the mean GRGM1200a spherical harmonic coefficients saved in a .npz
  (made by clone_average.py).
- Then gives one main callable:

    accel_inertial(r_I_m, t_s, L)

  which returns the lunar gravity acceleration in a Moon-centered inertial frame.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import os
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

def list_available_npz(folder: str = "clone_averages"):
    """
    I list all available gravity coefficient files in clone_averages.
    """
    folder_path = Path(folder)
    return sorted(folder_path.glob("*.npz"))

COEFFS_NPZ_PATH = "clone_averages/grgm1200a_clone_mean_L660.npz"

# never let L < 2 (don't want accidental two-body runs)
MIN_ALLOWED_L = 2

# Simple Moon spin model (good enough if truth+filter use the same model):
# Lunar sidereal rotation rate ~ 2*pi / 27.321661 days
MOON_OMEGA_RAD_S = 2.6617e-6  # [rad/s]
T0_S = 0.0  # reference epoch for rotation angle [s]

# Units
# The .npz stores R_km and GM_km3_s2.
# convert once at load time and then everything is SI (meters, seconds)
KM_TO_M = 1000.0
KM3_TO_M3 = 1000.0 ** 3

# Safety
EPS = 1e-15

# For MakeGravGridPoint:
# omega controls centrifugal contribution. I set omega=0 so I get *gravitational only*
GRAVMAG_OMEGA = 0.0

# ============================================================
# pyshtools
# ============================================================

try:
    import pyshtools  # type: ignore
except ImportError as e:
    raise RuntimeError(
        "need pyshtools (spherical harmonic gravity evaluation)"
        "Install it with: pip install pyshtools"
    ) from e


# ============================================================
# rotation helpers
# ============================================================

def _rotz(theta: float) -> np.ndarray:
    """Euler rotation matrix about +Z."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def _R_I_to_BF(t_s: float) -> np.ndarray:
    """
    Inertial -> Moon body-fixed rotation.
    I’m using a simple constant-spin model. The key is consistency between truth and filter.
    """
    theta = MOON_OMEGA_RAD_S * (t_s - T0_S)
    return _rotz(theta)


# ============================================================
# Coordinate conversions (Cartesian <-> spherical)
# ============================================================

def _cart_to_latlonr(r_xyz: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert Cartesian position to (r, lat, lon).
    - r [m]
    - lat [deg]  geocentric latitude
    - lon [deg]  longitude
    """
    x, y, z = r_xyz
    r = float(np.sqrt(x*x + y*y + z*z) + EPS)
    lon = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    return r, lat, lon

def _sph_components_to_cart(lat_deg: float, lon_deg: float,
                           g_r: float, g_theta: float, g_phi: float) -> np.ndarray:
    """
    Convert spherical (r, theta, phi) into Cartesian (x,y,z).

    pyshtools.gravmag.MakeGravGridPoint returns gravity components in spherical coords:
      (g_r, g_theta, g_phi)
        - r     : radial outward
        - theta : colatitude direction (increasing theta = moving south)
        - phi   : longitude direction (increasing lon = east)

    convert to Cartesian using std unit vectors:
      e_r     = [cos lat cos lon, cos lat sin lon, sin lat]
      e_phi   = [-sin lon, cos lon, 0]
      e_theta = [-sin lat cos lon, -sin lat sin lon, cos lat]
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)

    e_r = np.array([clat * clon, clat * slon, slat], dtype=np.float64)
    e_phi = np.array([-slon, clon, 0.0], dtype=np.float64)
    e_theta = np.array([-slat * clon, -slat * slon, clat], dtype=np.float64)

    return g_r * e_r + g_theta * e_theta + g_phi * e_phi


# ============================================================
# GravityModel class
# ============================================================

@dataclass(frozen=True)
class GravityModel:
    """
    Holds coefficients + constants and provides acceleration

    Coefficient storage:
      cilm[0, n, m] = C[n,m]
      cilm[1, n, m] = S[n,m]
    """
    cilm: np.ndarray          # shape (2, Lmax+1, Lmax+1)
    gm_m3_s2: float           # GM in SI [m^3/s^2]
    r0_m: float               # reference radius in SI [m]
    lmax_data: int            # max degree available in the loaded file

    @staticmethod
    def from_npz(npz_path: str = COEFFS_NPZ_PATH) -> "GravityModel":
        """
        Load the mean GRGM1200a coefficients saved by clone_average.py.

        Expected keys:
          C, S : (L+1, L+1)
          R_km : float
          GM_km3_s2 : float
        """
        data = np.load(npz_path)

        C = np.array(data["C"], dtype=np.float64)
        S = np.array(data["S"], dtype=np.float64)

        # r0_m = float(data["R_km"]) * KM_TO_M
        # gm_m3_s2 = float(data["GM_km3_s2"]) * KM3_TO_M3

        # change
        # Some clone files don't store R_km or GM_km3_s2
        # If missing, use standard Moon constants.

        if "R_km" in data:
            r0_m = float(data["R_km"]) * KM_TO_M
        else:
            # I use mean lunar radius [km]
            r0_m = 1737.4 * KM_TO_M

        if "GM_km3_s2" in data:
            gm_m3_s2 = float(data["GM_km3_s2"]) * KM3_TO_M3
        else:
            # I use standard lunar GM [km^3/s^2]
            gm_m3_s2 = 4902.800066 * KM3_TO_M3

        lmax_data = C.shape[0] - 1
        if C.shape != S.shape:
            raise ValueError(f"C and S must have same shape, got {C.shape} vs {S.shape}")
        if C.shape[0] != C.shape[1]:
            raise ValueError(f"C and S must be square, got {C.shape}")

        # pyshtools gravmag routines use "cilm" coefficient array with shape (2, lmax+1, lmax+1)
        cilm = np.zeros((2, lmax_data + 1, lmax_data + 1), dtype=np.float64)
        cilm[0, :, :] = C
        cilm[1, :, :] = S

        # I saw C00=0 in the averaged clone files (they start at n=2), so I add the monopole term
        # Without this, I only get tiny perturbation accelerations (~1e-11 m/s^2) instead of ~1.6 m/s^2 in LLO
        cilm[0, 0, 0] = 1.0
        cilm[1, 0, 0] = 0.0

        # I keep degree-1 terms at zero (body-centered frame)
        if lmax_data >= 1:
            cilm[:, 1, :] = 0.0

        print("C00 =", cilm[0,0,0], "S00 =", cilm[1,0,0])

        return GravityModel(cilm=cilm, gm_m3_s2=gm_m3_s2, r0_m=r0_m, lmax_data=lmax_data)

    # -----------------------------
    # Main API: inertial acceleration
    # -----------------------------
    def accel_inertial(self, r_I_m: np.ndarray, t_s: float, L: int) -> np.ndarray:
        """
        Lunar gravitational acceleration in Moon-centered inertial frame

        Inputs:
          r_I_m : (3,) position [m] in Moon-centered inertial frame
          t_s   : time [s]
          L     : truncation degree (must satisfy MIN_ALLOWED_L <= L <= lmax_data)

        Output:
          a_I_mps2 : (3,) acceleration [m/s^2] in inertial frame
        """
        L = int(L)
        if L < MIN_ALLOWED_L:
            raise ValueError(
                f"L={L} is not allowed. Start at L>=2 to avoid accidental two-body runs"
            )
        if L > self.lmax_data:
            raise ValueError(f"L={L} exceeds coefficient degree available (lmax_data={self.lmax_data}).")

        r_I_m = np.asarray(r_I_m, dtype=np.float64).reshape(3,)

        # Inertial -> body-fixed (bc coefficients are tied to the rotating Moon)
        R_I2BF = _R_I_to_BF(t_s)
        r_BF_m = R_I2BF @ r_I_m

        # Body-fixed acceleration from spherical harmonics
        a_BF_mps2 = self._accel_bodyfixed(r_BF_m, L)

        # Rotate back to inertial
        a_I_mps2 = R_I2BF.T @ a_BF_mps2
        return a_I_mps2

    # -----------------------------
    # Internal: body-fixed acceleration
    # -----------------------------
    def _accel_bodyfixed(self, r_BF_m: np.ndarray, L: int) -> np.ndarray:
        """
        Compute gravitational acceleration in the Moon body-fixed frame using pyshtools

          pyshtools.gravmag.MakeGravGridPoint(cilm, gm, r0, r, lat, lon, lmax=L, omega=0)

        - I set omega=0 to exclude centrifugal acceleration.
        - Output is in spherical components (r, theta, phi) so convert to Cartesian
        """
        r, lat_deg, lon_deg = _cart_to_latlonr(r_BF_m)

        # Compute gravity vector components at a single point
        # Per PYSHTOOLS docs, output is (g_r, g_theta, g_phi) in spherical coordinates.
        g_sph = pyshtools.gravmag.MakeGravGridPoint(
            self.cilm, self.gm_m3_s2, self.r0_m,
            r, lat_deg, lon_deg,
            lmax=L,
            omega=GRAVMAG_OMEGA
        )

        g_r = float(g_sph[0])
        g_theta = float(g_sph[1])
        g_phi = float(g_sph[2])

        a_BF = _sph_components_to_cart(lat_deg, lon_deg, g_r, g_theta, g_phi)
        return a_BF


# ============================================================
# test 
# ============================================================

if __name__ == "__main__":

    npz_files = list_available_npz()

    if not npz_files:
        raise RuntimeError("No .npz files found in clone_averages/")

    for npz_path in npz_files:
        print("\n======================================")
        print(f"Testing file: {npz_path.name}")

        model = GravityModel.from_npz(str(npz_path))

        r0 = model.r0_m + 50_000.0
        r_I = np.array([r0, 0.0, 0.0], dtype=np.float64)

        t = 0.0
        L_truth = model.lmax_data

        a2 = model.accel_inertial(r_I, t, L=2)
        aT = model.accel_inertial(r_I, t, L=L_truth)

        print(f"Loaded lmax_data = {model.lmax_data}")
        print("a(L=2)   [m/s^2] =", a2)
        print("a(L=max) [m/s^2] =", aT)
        print("||aT - a2|| =", np.linalg.norm(aT - a2))
        print("Norm of position:", np.linalg.norm(r_I))