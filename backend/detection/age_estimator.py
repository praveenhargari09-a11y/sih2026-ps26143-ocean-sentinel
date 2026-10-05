"""
backend/detection/age_estimator.py
====================================
Oil Spill Age Estimation using Fay's Spreading Equations.

Physical Background
-------------------
When oil is released on the sea surface it undergoes three spreading phases
driven by different forces:

  Phase 1 (gravity-inertia):  A(t) = k1 · V^(4/3) · t^(2/3)
  Phase 2 (gravity-viscous):  A(t) = k2 · V^(5/6) · t^(3/4)  ← dominant
  Phase 3 (surface-tension):  A(t) = k3 · V^(3/4) · t^(3/4)

where:
  A  = slick area (m²)
  V  = volume of oil spilled (m³)
  t  = time since release (s)
  k1 ≈ 0.679  (Fay's constant, gravity-inertia)
  k2 ≈ 0.645  (Fay's constant, gravity-viscous)

By inverting Phase 2 (the most commonly observed phase for satellite
observation timescales of hours to days):
  t = (A / (k2 · V^(5/6)))^(4/3)

Volume Estimation from Thickness
---------------------------------
When spill volume is unknown (common case for satellite observations), we
use thickness appearance as a proxy:

  thin sheen (iridescent)  : ~0.1 mm → low backscatter contrast
  thick slick (dark)       : ~0.3 mm → high backscatter contrast

References
----------
Fay, J.A. (1969). "The Spread of Oil Slicks on a Calm Sea."
  Oil on the Sea, Springer, pp. 53-63.
EMSA / REMPEC Guidelines on SAR-based oil spill thickness estimation.
"""

from __future__ import annotations

import math
from loguru import logger


# ---------------------------------------------------------------------------
# Physical constants (Fay 1969)
# ---------------------------------------------------------------------------

# Gravity-inertia phase constant
K1_GRAVITY_INERTIA: float = 0.679

# Gravity-viscous phase constant (dominant for satellite observation windows)
K2_GRAVITY_VISCOUS: float = 0.645

# Surface-tension phase constant
K3_SURFACE_TENSION: float = 0.402

# Typical SAR observation timescales are in gravity-viscous / surface-tension
# phase → we use K2 as primary estimator

# Oil density (kg/m³) — used for volume estimation from thickness
OIL_DENSITY_KG_M3: float = 870.0  # typical crude oil

# Conversion factor km² → m²
KM2_TO_M2: float = 1e6


class SpillAgeEstimator:
    """
    Estimate oil spill age from observed slick area using Fay's spreading law.

    Usage
    -----
    estimator = SpillAgeEstimator()
    result = estimator.estimate_age(area_km2=12.5, volume_estimate_m3=500)
    print(result)  # {'age_hours_min': ..., 'age_hours_max': ..., ...}
    """

    def __init__(self) -> None:
        logger.debug("SpillAgeEstimator initialized (Fay spreading model)")

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def estimate_age(
        self,
        area_km2: float,
        volume_estimate_m3: float | None = None,
    ) -> dict:
        """
        Estimate the age of an oil spill from its observed area.

        When ``volume_estimate_m3`` is not provided, the method estimates a
        plausible range by trying thin-sheen and thick-slick volume bounds.

        Parameters
        ----------
        area_km2 : float
            Observed slick area in km².
        volume_estimate_m3 : float, optional
            Volume of oil spilled in m³.  If None, estimated from area × thickness.

        Returns
        -------
        dict with keys:
          - ``age_hours_min``  : float — lower age bound (hours)
          - ``age_hours_max``  : float — upper age bound (hours)
          - ``age_hours_mean`` : float — central estimate (hours)
          - ``method``         : str — always 'fay_spreading'
          - ``area_km2``       : float
          - ``volume_m3``      : float — volume used (or estimated range)
          - ``phase``          : str — spreading phase used

        Raises
        ------
        ValueError
            If ``area_km2`` <= 0.
        """
        if area_km2 <= 0:
            raise ValueError(f"area_km2 must be positive, got {area_km2}")

        area_m2 = area_km2 * KM2_TO_M2

        if volume_estimate_m3 is not None:
            # Single volume → compute one age, return ±20% uncertainty
            age_hours = self._fay_phase2_age(area_m2, volume_estimate_m3)
            return {
                "age_hours_min": max(0.0, age_hours * 0.80),
                "age_hours_max": age_hours * 1.20,
                "age_hours_mean": age_hours,
                "method": "fay_spreading",
                "area_km2": area_km2,
                "volume_m3": volume_estimate_m3,
                "phase": "gravity_viscous",
            }
        else:
            # Unknown volume → bound by thin/thick thickness assumptions
            vol_thin = self._volume_from_thickness(area_m2, thickness_mm=0.1)  # thin sheen
            vol_thick = self._volume_from_thickness(area_m2, thickness_mm=0.3)  # thick slick

            age_thin = self._fay_phase2_age(area_m2, vol_thin)   # thin → spreads faster → older
            age_thick = self._fay_phase2_age(area_m2, vol_thick) # thick → spreads slower → younger

            age_min = min(age_thin, age_thick)
            age_max = max(age_thin, age_thick)

            logger.debug(
                f"Age estimate: area={area_km2} km², "
                f"vol_thin={vol_thin:.1f} m³ → {age_thin:.1f} h, "
                f"vol_thick={vol_thick:.1f} m³ → {age_thick:.1f} h"
            )

            return {
                "age_hours_min": round(age_min, 2),
                "age_hours_max": round(age_max, 2),
                "age_hours_mean": round((age_min + age_max) / 2, 2),
                "method": "fay_spreading",
                "area_km2": area_km2,
                "volume_m3": f"{vol_thin:.1f}–{vol_thick:.1f}",
                "phase": "gravity_viscous",
            }

    # ------------------------------------------------------------------
    # Thickness-based backscatter heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def thickness_from_appearance(backscatter_diff: float) -> float:
        """
        Heuristic: estimate oil thickness (mm) from the backscatter difference
        between the slick and surrounding ocean.

        A stronger damping signal (larger negative difference) suggests a
        thicker oil layer:
          |Δσ⁰| < 3 dB  → thin sheen (~0.1 mm, iridescent)
          3–6 dB         → medium slick (~0.2 mm)
          |Δσ⁰| > 6 dB  → thick slick (~0.3 mm, dark patch)

        Parameters
        ----------
        backscatter_diff : float
            Difference (dB) between slick backscatter and ocean background.
            Should be negative (slick is darker).

        Returns
        -------
        float
            Estimated thickness in millimetres.
        """
        abs_diff = abs(backscatter_diff)
        if abs_diff < 3.0:
            thickness_mm = 0.1   # thin iridescent sheen
        elif abs_diff < 6.0:
            # Linear interpolation: 0.1–0.3 mm over 3–6 dB range
            thickness_mm = 0.1 + (abs_diff - 3.0) / 3.0 * 0.2
        else:
            thickness_mm = 0.3   # thick dark slick (emulsified / heavy crude)

        logger.debug(f"Thickness from backscatter diff {backscatter_diff:.2f} dB → {thickness_mm:.3f} mm")
        return thickness_mm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fay_phase2_age(area_m2: float, volume_m3: float) -> float:
        """
        Invert Fay's gravity-viscous spreading equation for time.

        Forward:  A(t) = K2 · V^(5/6) · t^(3/4)
        Inverse:  t = (A / (K2 · V^(5/6)))^(4/3)

        Parameters
        ----------
        area_m2 : float  — slick area in m²
        volume_m3 : float — spill volume in m³ (must be > 0)

        Returns
        -------
        float — age in hours
        """
        if volume_m3 <= 0:
            return 0.0

        # t in seconds
        # A = K2 · V^(5/6) · t^(3/4)
        # t^(3/4) = A / (K2 · V^(5/6))
        # t = [A / (K2 · V^(5/6))]^(4/3)
        v_term = K2_GRAVITY_VISCOUS * (volume_m3 ** (5.0 / 6.0))
        if v_term < 1e-12:
            return 0.0

        t_seconds = (area_m2 / v_term) ** (4.0 / 3.0)
        t_hours = t_seconds / 3600.0
        return max(0.0, t_hours)

    @staticmethod
    def _volume_from_thickness(area_m2: float, thickness_mm: float) -> float:
        """
        Estimate oil volume from area and assumed thickness.

        V (m³) = A (m²) × h (m)

        Parameters
        ----------
        area_m2 : float   — area in m²
        thickness_mm : float — thickness in mm

        Returns
        -------
        float — volume in m³
        """
        thickness_m = thickness_mm / 1000.0
        return area_m2 * thickness_m


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    area = float(sys.argv[1]) if len(sys.argv) > 1 else 12.5
    vol = float(sys.argv[2]) if len(sys.argv) > 2 else None

    estimator = SpillAgeEstimator()
    result = estimator.estimate_age(area_km2=area, volume_estimate_m3=vol)
    print("\n=== Spill Age Estimate ===")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    # Thickness demo
    for db in [-2.0, -4.5, -8.0]:
        t = SpillAgeEstimator.thickness_from_appearance(db)
        print(f"  backscatter_diff={db:.1f} dB → thickness={t:.3f} mm")
