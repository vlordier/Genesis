"""
IMU (Inertial Measurement Unit) sensor model.

Simulates a 6-DOF MEMS IMU (accelerometer + gyroscope) from ideal body-frame
kinematics produced by Genesis.  The error model is based on the standard
IEEE-1554 / Allan-deviation characterisation used by Isaac Lab and most
flight-sim frameworks:

* **White noise** (angle/velocity random walk) — ``noise_density_*`` in
  ``units/√Hz``, converted to per-sample sigma as ``σ = nd / √dt``.
* **Bias instability** (Gauss-Markov first-order drift) — ``bias_tau_*`` and
  ``bias_sigma_*``.
* **Scale-factor error** — a relative gain applied before noise, simulating
  manufacturing mismatch.

Gravity
-------
Genesis reports *pure* linear acceleration (no gravity).  Real IMUs measure
*specific force* = linear acceleration − gravity vector.  When
``add_gravity=True`` the model expects a ``"gravity_body"`` key in *state*
(gravity vector in body frame, m/s²) or uses the default ENU gravity
``[0, 0, 9.81]`` rotated to body frame (caller's responsibility to provide
the correct vector).

Usage
-----
::

    imu = IMUModel(update_rate_hz=200.0, noise_density_acc=2.0e-3)
    obs = imu.step(sim_time, {
        "lin_acc":     np.array([ax, ay, az]),   # body-frame m/s²
        "ang_vel":     np.array([wx, wy, wz]),   # body-frame rad/s
        "gravity_body": np.array([0.0, 0.0, 9.81]),  # body-frame gravity
    })
    print(obs["lin_acc"])   # noisy specific force (m/s²)
    print(obs["ang_vel"])   # noisy angular velocity (rad/s)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import Float64Array, ImuObservation

if TYPE_CHECKING:
    from .config import IMUConfig

# Standard gravity constant (m/s²)
_G: Final[float] = 9.80665
# Default gravity vector in ENU world frame: points down → -Z
# If no gravity_body is provided and add_gravity=True, this is used as a
# fallback assuming the vehicle is level (identity rotation).
_DEFAULT_GRAVITY_BODY: Final[Float64Array] = np.array([0.0, 0.0, _G], dtype=np.float64)

# Minimum allowable time-constant for Gauss-Markov processes (seconds).
_MIN_BIAS_TAU_S: Final[float] = 1e-3


class IMUModel(BaseSensor):
    """
    Realistic 6-DOF MEMS IMU (accelerometer + gyroscope) sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        IMU output rate in Hz (100–1000 Hz for typical MEMS devices).
    noise_density_acc:
        Accelerometer white-noise density in m/s²/√Hz (velocity random walk).
        Converted internally to per-sample sigma = ``noise_density_acc /
        sqrt(1 / update_rate_hz)``.  Typical MEMS value: 2–5 × 10⁻³ m/s²/√Hz.
    noise_density_gyr:
        Gyroscope white-noise density in rad/s/√Hz (angle random walk).
        Typical MEMS value: 1–5 × 10⁻⁴ rad/s/√Hz.
    bias_tau_acc_s:
        Accelerometer bias correlation time (Gauss-Markov first-order process),
        seconds.  Typical value: 300–3600 s.
    bias_sigma_acc:
        Steady-state accelerometer bias standard deviation (m/s²).
    bias_tau_gyr_s:
        Gyroscope bias correlation time (s).
    bias_sigma_gyr:
        Steady-state gyroscope bias standard deviation (rad/s).
    scale_factor_acc:
        Relative accelerometer scale-factor error (dimensionless, ≥ −1).
        Measurement = (1 + scale_factor_acc) × true_value + bias + noise.
        Default 0 = no scale error.
    scale_factor_gyr:
        Relative gyroscope scale-factor error (dimensionless, ≥ −1).
    add_gravity:
        When ``True`` the model adds the gravity vector (from
        ``state["gravity_body"]`` or the default ENU level-flight value) to
        the ideal acceleration before applying noise.  This replicates how a
        real accelerometer measures specific force.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "imu",
        update_rate_hz: float = 200.0,
        noise_density_acc: float = 2.0e-3,
        noise_density_gyr: float = 1.7e-4,
        bias_tau_acc_s: float = 300.0,
        bias_sigma_acc: float = 5.0e-3,
        bias_tau_gyr_s: float = 300.0,
        bias_sigma_gyr: float = 1.0e-4,
        scale_factor_acc: float = 0.0,
        scale_factor_gyr: float = 0.0,
        add_gravity: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_density_acc = float(noise_density_acc)
        self.noise_density_gyr = float(noise_density_gyr)
        self.bias_tau_acc_s = float(max(bias_tau_acc_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_acc = float(bias_sigma_acc)
        self.bias_tau_gyr_s = float(max(bias_tau_gyr_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_gyr = float(bias_sigma_gyr)
        self.scale_factor_acc = float(scale_factor_acc)
        self.scale_factor_gyr = float(scale_factor_gyr)
        self.add_gravity = bool(add_gravity)
        self._rng = np.random.default_rng(seed=seed)

        # ------------------------------------------------------------------
        # Pre-computed noise and Gauss-Markov coefficients
        # ------------------------------------------------------------------
        dt = 1.0 / self.update_rate_hz

        # White-noise sigma per sample: σ = noise_density / √dt
        self._sigma_acc: float = self.noise_density_acc / math.sqrt(dt)
        self._sigma_gyr: float = self.noise_density_gyr / math.sqrt(dt)

        # Gauss-Markov decay and drive-noise for accelerometer bias
        self._alpha_acc: float = math.exp(-dt / self.bias_tau_acc_s)
        self._drive_sigma_acc: float = self.bias_sigma_acc * math.sqrt(1.0 - self._alpha_acc**2)

        # Gauss-Markov decay and drive-noise for gyroscope bias
        self._alpha_gyr: float = math.exp(-dt / self.bias_tau_gyr_s)
        self._drive_sigma_gyr: float = self.bias_sigma_gyr * math.sqrt(1.0 - self._alpha_gyr**2)

        # Pre-compute (1 + scale_factor) for fast scalar multiply in step()
        self._gain_acc: float = 1.0 + self.scale_factor_acc
        self._gain_gyr: float = 1.0 + self.scale_factor_gyr

        # Mutable state
        self._bias_acc: Float64Array = np.zeros(3, dtype=np.float64)
        self._bias_gyr: Float64Array = np.zeros(3, dtype=np.float64)
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "IMUConfig") -> "IMUModel":
        """Construct an :class:`IMUModel` from an :class:`~genesis.sensors.config.IMUConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "IMUConfig":
        """Return the current parameters as an :class:`~genesis.sensors.config.IMUConfig`."""
        from .config import IMUConfig

        return IMUConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_density_acc=self.noise_density_acc,
            noise_density_gyr=self.noise_density_gyr,
            bias_tau_acc_s=self.bias_tau_acc_s,
            bias_sigma_acc=self.bias_sigma_acc,
            bias_tau_gyr_s=self.bias_tau_gyr_s,
            bias_sigma_gyr=self.bias_sigma_gyr,
            scale_factor_acc=self.scale_factor_acc,
            scale_factor_gyr=self.scale_factor_gyr,
            add_gravity=self.add_gravity,
        )

    # ------------------------------------------------------------------
    # Public properties (inspection / debugging)
    # ------------------------------------------------------------------

    @property
    def bias_acc(self) -> Float64Array:
        """Current accelerometer bias vector (m/s²).  Copy to prevent mutation."""
        return self._bias_acc.copy()

    @property
    def bias_gyr(self) -> Float64Array:
        """Current gyroscope bias vector (rad/s).  Copy to prevent mutation."""
        return self._bias_gyr.copy()

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._bias_acc = np.zeros(3, dtype=np.float64)
        self._bias_gyr = np.zeros(3, dtype=np.float64)
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> ImuObservation | dict[str, Any]:
        """
        Produce a realistic IMU measurement.

        Expected keys in *state*:
        - ``"lin_acc"``  -- ``np.ndarray[3]`` ideal linear acceleration in
          **body frame** (m/s²), NOT including gravity.  Genesis provides this
          as the rigid-body acceleration minus external forces / gravity.
        - ``"ang_vel"``  -- ``np.ndarray[3]`` ideal angular velocity in
          **body frame** (rad/s).
        - ``"gravity_body"`` *(optional)* -- gravity vector rotated to body
          frame (m/s²).  Only used when ``add_gravity=True``.  If absent and
          ``add_gravity=True``, the default ENU level-flight gravity
          ``[0, 0, 9.81]`` is used (valid when the vehicle is near-level).
        """
        true_acc: Float64Array = np.asarray(state.get("lin_acc", [0.0, 0.0, 0.0]), dtype=np.float64)
        true_gyr: Float64Array = np.asarray(state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        if true_acc.shape != (3,):
            raise ValueError(f"state['lin_acc'] must be a 3-element array, got shape {true_acc.shape}")
        if true_gyr.shape != (3,):
            raise ValueError(f"state['ang_vel'] must be a 3-element array, got shape {true_gyr.shape}")

        # ------------------------------------------------------------------
        # Gravity injection
        # ------------------------------------------------------------------
        if self.add_gravity:
            gravity_body = state.get("gravity_body")
            if gravity_body is not None:
                g_vec: Float64Array = np.asarray(gravity_body, dtype=np.float64)
            else:
                g_vec = _DEFAULT_GRAVITY_BODY
            true_acc = true_acc + g_vec

        # ------------------------------------------------------------------
        # Scale-factor error (applied before bias and noise)
        # ------------------------------------------------------------------
        true_acc = self._gain_acc * true_acc
        true_gyr = self._gain_gyr * true_gyr

        # ------------------------------------------------------------------
        # Gauss-Markov bias drift
        # ------------------------------------------------------------------
        self._bias_acc = self._alpha_acc * self._bias_acc + self._rng.normal(0.0, self._drive_sigma_acc, 3)
        self._bias_gyr = self._alpha_gyr * self._bias_gyr + self._rng.normal(0.0, self._drive_sigma_gyr, 3)

        # ------------------------------------------------------------------
        # White noise (angle/velocity random walk)
        # ------------------------------------------------------------------
        noisy_acc: Float64Array = true_acc + self._bias_acc + self._rng.normal(0.0, self._sigma_acc, 3)
        noisy_gyr: Float64Array = true_gyr + self._bias_gyr + self._rng.normal(0.0, self._sigma_gyr, 3)

        obs: ImuObservation = {
            "lin_acc": noisy_acc,
            "ang_vel": noisy_gyr,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["IMUModel"]
