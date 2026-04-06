"""
GNSS / GPS sensor model.

Converts a perfect world-frame position and velocity (from Genesis) into
a realistic GNSS measurement with:

* Gaussian white noise on position and velocity.
* First-order Gauss-Markov bias drift (random walk) on position.
* Multipath error correlated with nearby obstacle geometry.
* Satellite constellation availability mask based on altitude and
  urban-canyon heuristics.
* Configurable fix quality modes (no fix / autonomous / RTK).

Usage
-----
::

    gnss = GNSSModel(update_rate_hz=10.0, noise_m=1.5)
    obs = gnss.step(sim_time, {
        "pos":  np.array([x, y, z]),   # world-frame metres
        "vel":  np.array([vx, vy, vz]),
    })
    print(obs["pos_llh"])  # latitude, longitude, height (degrees, metres)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseSensor


class GNSSModel(BaseSensor):
    """
    Realistic GNSS / GPS sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Output rate in Hz (typically 1–10 Hz for GPS, up to ~50 Hz for RTK).
    noise_m:
        1-sigma Gaussian position noise in metres (horizontal and vertical).
    vel_noise_ms:
        1-sigma Gaussian velocity noise in m/s.
    bias_tau_s:
        Time constant for the first-order Gauss-Markov position bias (s).
    bias_sigma_m:
        Steady-state standard deviation of the bias random walk (m).
    multipath_sigma_m:
        Standard deviation of multipath error (m).  Scaled by the
        obstruction fraction provided in ``state["obstruction"]``.
    min_fix_altitude_m:
        Altitude (metres above ground) below which fix quality degrades.
    jammer_zones:
        List of ``(centre_xyz, radius_m)`` tuples; if the drone is inside
        any zone the output ``fix_quality`` is set to 0 (no fix).
    origin_llh:
        ``(lat_deg, lon_deg, alt_m)`` of the simulation world origin.
        Used to convert XYZ to lat/lon/height.
    """

    FIX_NO_FIX = 0
    FIX_AUTONOMOUS = 1
    FIX_RTK = 4

    def __init__(
        self,
        name: str = "gnss",
        update_rate_hz: float = 10.0,
        noise_m: float = 1.5,
        vel_noise_ms: float = 0.05,
        bias_tau_s: float = 60.0,
        bias_sigma_m: float = 0.5,
        multipath_sigma_m: float = 1.0,
        min_fix_altitude_m: float = 0.5,
        jammer_zones: list[tuple[np.ndarray, float]] | None = None,
        origin_llh: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_m = float(noise_m)
        self.vel_noise_ms = float(vel_noise_ms)
        self.bias_tau_s = float(max(bias_tau_s, 1e-3))
        self.bias_sigma_m = float(bias_sigma_m)
        self.multipath_sigma_m = float(multipath_sigma_m)
        self.min_fix_altitude_m = float(min_fix_altitude_m)
        self.jammer_zones: list[tuple[np.ndarray, float]] = jammer_zones or []
        self.origin_llh = tuple(origin_llh)

        # Earth radius and degrees-per-metre factors
        self._R_earth = 6_378_137.0
        self._m_per_deg_lat = np.pi * self._R_earth / 180.0
        self._m_per_deg_lon = self._m_per_deg_lat * np.cos(np.deg2rad(self.origin_llh[0]))

        self._bias = np.zeros(3, dtype=np.float64)
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._bias = np.zeros(3, dtype=np.float64)
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Produce a realistic GNSS measurement.

        Expected keys in *state*:
        - ``"pos"`` – ``np.ndarray[3]`` world-frame position in metres.
        - ``"vel"`` – ``np.ndarray[3]`` world-frame velocity in m/s.
        - ``"obstruction"`` *(optional)* – float 0–1 representing the
          fraction of sky hemisphere blocked by obstacles (used to scale
          multipath error).
        """
        true_pos = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        true_vel = np.asarray(state.get("vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        obstruction = float(state.get("obstruction", 0.0))

        # Check jammer zones
        for centre, radius in self.jammer_zones:
            centre = np.asarray(centre, dtype=np.float64)
            if np.linalg.norm(true_pos - centre) <= radius:
                result = {
                    "pos": true_pos.copy(),
                    "vel": true_vel.copy(),
                    "fix_quality": self.FIX_NO_FIX,
                    "n_satellites": 0,
                    "hdop": 99.9,
                }
                self._last_obs = result
                self._mark_updated(sim_time)
                return result

        # Update bias random walk
        dt = 1.0 / self.update_rate_hz
        alpha = np.exp(-dt / self.bias_tau_s)
        drive_sigma = self.bias_sigma_m * np.sqrt(1.0 - alpha**2)
        self._bias = alpha * self._bias + np.random.normal(0.0, drive_sigma, 3)

        # Position error
        white = np.random.normal(0.0, self.noise_m, 3)
        multipath = np.random.normal(0.0, self.multipath_sigma_m * obstruction, 3)
        noisy_pos = true_pos + self._bias + white + multipath

        # Velocity error
        noisy_vel = true_vel + np.random.normal(0.0, self.vel_noise_ms, 3)

        # Fix quality
        alt = float(true_pos[2])
        fix_quality = self.FIX_AUTONOMOUS if alt >= self.min_fix_altitude_m else self.FIX_NO_FIX
        n_sat = int(np.clip(8 - round(obstruction * 6), 0, 12))
        hdop = np.clip(1.0 + obstruction * 2.5, 1.0, 10.0)

        # Convert XYZ to lat/lon/height (flat-earth approximation)
        lat = self.origin_llh[0] + noisy_pos[1] / self._m_per_deg_lat
        lon = self.origin_llh[1] + noisy_pos[0] / self._m_per_deg_lon
        alt_m = self.origin_llh[2] + noisy_pos[2]

        result = {
            "pos": noisy_pos,
            "vel": noisy_vel,
            "pos_llh": np.array([lat, lon, alt_m]),
            "fix_quality": fix_quality,
            "n_satellites": n_sat,
            "hdop": float(hdop),
        }
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs
