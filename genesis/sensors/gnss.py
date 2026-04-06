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
        "pos":  np.array([x, y, z]),   # world-frame metres (ENU: x=East, y=North)
        "vel":  np.array([vx, vy, vz]),
    })
    print(obs["pos_llh"])  # latitude, longitude, height (degrees, metres)
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import Float64Array, GnssObservation, JammerZone

if TYPE_CHECKING:
    from .config import GNSSConfig

# Physical constants
_EARTH_RADIUS_M: Final[float] = 6_378_137.0
# Minimum metres-per-degree-longitude to guard against division by zero at the poles.
_MIN_M_PER_DEG_LON: Final[float] = 1.0
# Fallback HDOP reported when there is no fix or inside a jammer zone.
_HDOP_NO_FIX: Final[float] = 99.9
# Nominal satellites visible at zero obstruction.
_BASE_SATELLITES: Final[int] = 8
# Number of satellites lost per unit obstruction fraction.
_SAT_LOSS_PER_OBSTRUCTION: Final[int] = 6
# Maximum satellite count (clamp upper bound).
_MAX_SATELLITES: Final[int] = 12
# HDOP slope as a function of obstruction fraction.
_HDOP_OBSTRUCTION_SLOPE: Final[float] = 2.5
# Minimum HDOP (ideal open sky).
_HDOP_MIN: Final[float] = 1.0
# Maximum HDOP (very degraded).
_HDOP_MAX: Final[float] = 10.0


class GnssFixQuality(IntEnum):
    """NMEA GGA fix-quality indicator values."""

    NO_FIX = 0
    AUTONOMOUS = 1
    RTK = 4


class GNSSModel(BaseSensor):
    """
    Realistic GNSS / GPS sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Output rate in Hz (typically 1-10 Hz for GPS, up to ~50 Hz for RTK).
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
        any zone the output ``fix_quality`` is set to ``GnssFixQuality.NO_FIX``.
    origin_llh:
        ``(lat_deg, lon_deg, alt_m)`` of the simulation world origin.
        Used to convert XYZ to lat/lon/height.  The coordinate convention
        is ENU: world-frame x = East, y = North, z = Up.
    seed:
        Optional seed for the random-number generator (reproducibility).
    """

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
        jammer_zones: list[JammerZone] | None = None,
        origin_llh: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_m = float(noise_m)
        self.vel_noise_ms = float(vel_noise_ms)
        self.bias_tau_s = float(max(bias_tau_s, 1e-3))
        self.bias_sigma_m = float(bias_sigma_m)
        self.multipath_sigma_m = float(multipath_sigma_m)
        self.min_fix_altitude_m = float(min_fix_altitude_m)
        self.jammer_zones: list[JammerZone] = jammer_zones or []
        self.origin_llh = tuple(origin_llh)
        self._rng = np.random.default_rng(seed=seed)

        # Earth radius and degrees-per-metre factors (flat-earth ENU approximation).
        # Clamp m_per_deg_lon with a minimum value to prevent division by zero
        # when the simulation origin is at or very near the geographic poles.
        self._m_per_deg_lat = math.pi * _EARTH_RADIUS_M / 180.0
        self._m_per_deg_lon = max(
            self._m_per_deg_lat * math.cos(math.radians(self.origin_llh[0])),
            _MIN_M_PER_DEG_LON,
        )

        # ------------------------------------------------------------------
        # Pre-computed Gauss-Markov coefficients (constant for fixed update rate)
        # ------------------------------------------------------------------
        self._dt: float = 1.0 / self.update_rate_hz
        self._alpha: float = math.exp(-self._dt / self.bias_tau_s)
        # Steady-state noise injection per step: sigma * sqrt(1 - alpha^2)
        self._drive_sigma: float = self.bias_sigma_m * math.sqrt(1.0 - self._alpha**2)

        # Pre-convert jammer zone centres to numpy arrays to avoid repeated
        # np.asarray() calls inside the hot step() loop.
        self._jammer_centres: list[Float64Array] = [
            np.asarray(centre, dtype=np.float64) for centre, _ in self.jammer_zones
        ]
        self._jammer_radii: list[float] = [float(r) for _, r in self.jammer_zones]

        self._bias: Float64Array = np.zeros(3, dtype=np.float64)
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "GNSSConfig") -> "GNSSModel":
        """Construct a :class:`GNSSModel` from a :class:`~genesis.sensors.config.GNSSConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "GNSSConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.GNSSConfig`."""
        from .config import GNSSConfig

        return GNSSConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_m=self.noise_m,
            vel_noise_ms=self.vel_noise_ms,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_m=self.bias_sigma_m,
            multipath_sigma_m=self.multipath_sigma_m,
            min_fix_altitude_m=self.min_fix_altitude_m,
            jammer_zones=list(self.jammer_zones),
            origin_llh=self.origin_llh,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bias(self) -> Float64Array:
        """Current position bias vector (metres, world frame)."""
        return self._bias.copy()

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._bias = np.zeros(3, dtype=np.float64)
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> GnssObservation | dict[str, Any]:
        """
        Produce a realistic GNSS measurement.

        Expected keys in *state*:
        - ``"pos"`` -- ``np.ndarray[3]`` world-frame position in metres (ENU).
        - ``"vel"`` -- ``np.ndarray[3]`` world-frame velocity in m/s.
        - ``"obstruction"`` *(optional)* -- float 0-1 representing the
          fraction of sky hemisphere blocked by obstacles (used to scale
          multipath error).
        """
        true_pos: Float64Array = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        true_vel: Float64Array = np.asarray(state.get("vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        obstruction = float(state.get("obstruction", 0.0))

        # Check jammer zones using pre-converted centre arrays
        for centre_arr, radius in zip(self._jammer_centres, self._jammer_radii):
            if float(np.linalg.norm(true_pos - centre_arr)) <= radius:
                # Return stale last-known position/velocity when jammed so that
                # ground truth is never exposed through the sensor output.
                stale_pos = (
                    np.asarray(self._last_obs["pos"], dtype=np.float64)
                    if self._last_obs and "pos" in self._last_obs
                    else np.zeros(3, dtype=np.float64)
                )
                stale_vel = (
                    np.asarray(self._last_obs["vel"], dtype=np.float64)
                    if self._last_obs and "vel" in self._last_obs
                    else np.zeros(3, dtype=np.float64)
                )
                result: GnssObservation = {
                    "pos": stale_pos,
                    "vel": stale_vel,
                    "pos_llh": np.array([self.origin_llh[0], self.origin_llh[1], self.origin_llh[2]]),
                    "fix_quality": GnssFixQuality.NO_FIX,
                    "n_satellites": 0,
                    "hdop": _HDOP_NO_FIX,
                }
                self._last_obs = result
                self._mark_updated(sim_time)
                return result

        # Update bias random walk (Gauss-Markov) using pre-computed coefficients
        alpha = self._alpha
        self._bias = alpha * self._bias + self._rng.normal(0.0, self._drive_sigma, 3)

        # Position error: white noise + Gauss-Markov bias + multipath
        white: Float64Array = self._rng.normal(0.0, self.noise_m, 3)
        # Guard against zero-sigma when obstruction=0 to keep semantics clear
        if obstruction > 0.0:
            multipath: Float64Array = self._rng.normal(0.0, self.multipath_sigma_m * obstruction, 3)
        else:
            multipath = np.zeros(3, dtype=np.float64)
        noisy_pos: Float64Array = true_pos + self._bias + white + multipath

        # Velocity error
        noisy_vel: Float64Array = true_vel + self._rng.normal(0.0, self.vel_noise_ms, 3)

        # Fix quality and satellite count
        alt = float(true_pos[2])
        fix_quality = GnssFixQuality.AUTONOMOUS if alt >= self.min_fix_altitude_m else GnssFixQuality.NO_FIX
        n_sat = int(
            min(
                max(_BASE_SATELLITES - round(obstruction * _SAT_LOSS_PER_OBSTRUCTION), 0),
                _MAX_SATELLITES,
            )
        )
        hdop = min(max(_HDOP_MIN + obstruction * _HDOP_OBSTRUCTION_SLOPE, _HDOP_MIN), _HDOP_MAX)

        # Convert ENU XYZ to lat/lon/height (flat-earth approximation).
        # ENU convention: x=East → longitude, y=North → latitude.
        lat = self.origin_llh[0] + noisy_pos[1] / self._m_per_deg_lat
        lon = self.origin_llh[1] + noisy_pos[0] / self._m_per_deg_lon
        alt_m = self.origin_llh[2] + noisy_pos[2]

        obs: GnssObservation = {
            "pos": noisy_pos,
            "vel": noisy_vel,
            "pos_llh": np.array([lat, lon, alt_m]),
            "fix_quality": fix_quality,
            "n_satellites": n_sat,
            "hdop": hdop,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs
