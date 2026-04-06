"""
Pydantic v2 configuration models for every Genesis sensor.

Each ``*Config`` class is a ``pydantic.BaseModel`` with field-level
validation (range checks, type coercion).  Sensors can be constructed
directly from these models via their ``from_config()`` class-method or
serialised / de-serialised for experiment logging::

    import json
    from genesis.sensors.config import CameraConfig, SensorSuiteConfig

    cfg = CameraConfig(iso=800, jpeg_quality=70)
    print(cfg.model_dump_json(indent=2))     # JSON export
    cfg2 = CameraConfig.model_validate_json(json_str)  # JSON import

    suite_cfg = SensorSuiteConfig(rgb=cfg)
    suite = SensorSuite.from_config(suite_cfg)

All fields mirror the constructor parameters of the corresponding sensor
class exactly, so ``sensor_class(**config.model_dump())`` always works.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------


class CameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.CameraModel`."""

    name: str = "rgb_camera"
    update_rate_hz: float = Field(default=30.0, gt=0, description="Frame rate (Hz).")
    resolution: tuple[int, int] = Field(default=(640, 480), description="(width, height) in pixels.")
    distortion_coeffs: tuple[float, ...] | None = Field(
        default=None,
        description="OpenCV-style (k1, k2, p1, p2[, k3]).  None = no distortion.",
    )
    rolling_shutter_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = global shutter, 1 = full rolling shutter.",
    )
    motion_blur_kernel: int = Field(default=0, ge=0, description="Half-length of 1-D motion-blur kernel. 0 = off.")
    base_iso: float = Field(default=100.0, gt=0, description="Reference ISO.")
    iso: float = Field(default=100.0, gt=0, description="Effective ISO.")
    read_noise_sigma: float = Field(default=1.5, ge=0, description="Gaussian read-noise sigma (electrons).")
    dead_pixel_fraction: float = Field(default=0.0001, ge=0.0, le=1.0, description="Fraction of dead pixels.")
    hot_pixel_fraction: float = Field(default=0.00005, ge=0.0, le=1.0, description="Fraction of hot pixels.")
    jpeg_quality: int = Field(default=0, ge=0, le=100, description="JPEG quality (0 = disabled).")
    full_well_electrons: float = Field(default=3500.0, gt=0, description="Full-well capacity at base_iso (electrons).")
    vignetting_strength: float = Field(
        default=0.0,
        ge=0.0,
        description="Radial vignetting strength (0 = off, 0.5 = moderate, 1.0 = strong).",
    )
    chromatic_aberration_px: float = Field(
        default=0.0,
        ge=0.0,
        description="Lateral chromatic aberration: max channel shift in pixels at image corner (0 = off).",
    )
    seed: int | None = Field(default=None, description="RNG seed for reproducibility.")

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# EventCameraConfig
# ---------------------------------------------------------------------------


class EventCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.EventCameraModel`."""

    name: str = "event_camera"
    update_rate_hz: float = Field(default=1000.0, gt=0)
    threshold_pos: float = Field(default=0.2, gt=0, description="Positive contrast threshold (log-intensity).")
    threshold_neg: float = Field(default=0.2, gt=0, description="Negative contrast threshold (log-intensity).")
    refractory_period_s: float = Field(default=0.0, ge=0.0, description="Per-pixel minimum inter-event interval (s).")
    threshold_variation: float = Field(
        default=0.0,
        ge=0.0,
        description="Relative per-pixel threshold spread (sigma / nominal).",
    )
    background_activity_rate_hz: float = Field(
        default=0.0, ge=0.0, description="Mean spontaneous noise event rate per pixel per second."
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# ThermalCameraConfig
# ---------------------------------------------------------------------------


class ThermalCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ThermalCameraModel`."""

    name: str = "thermal_camera"
    update_rate_hz: float = Field(default=9.0, gt=0)
    resolution: tuple[int, int] = Field(default=(320, 240))
    temp_ambient_c: float = Field(default=20.0, description="Background pixel temperature (°C).")
    temp_sky_c: float = Field(default=-30.0, description="Sky pixel temperature (°C).")
    psf_sigma: float = Field(default=1.0, ge=0.0, description="Gaussian PSF sigma (pixels).")
    nuc_sigma: float = Field(default=0.5, ge=0.0, description="NUC residual offset sigma (°C).")
    noise_sigma: float = Field(default=0.05, ge=0.0, description="Per-frame detector noise sigma (°C).")
    bit_depth: int = Field(default=14, ge=1, le=32, description="Output quantisation bit depth.")
    fog_density: float = Field(default=0.0, ge=0.0, description="Fog extinction coefficient (1/m).")
    temp_range_c: tuple[float, float] = Field(
        default=(-20.0, 140.0), description="(t_min, t_max) for quantisation clipping."
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _temp_range_ordered(self) -> "ThermalCameraConfig":
        t_min, t_max = self.temp_range_c
        if t_min >= t_max:
            raise ValueError(f"temp_range_c must satisfy t_min < t_max, got {self.temp_range_c}")
        return self

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# LidarConfig
# ---------------------------------------------------------------------------


class LidarConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.LidarModel`."""

    name: str = "lidar"
    update_rate_hz: float = Field(default=10.0, gt=0)
    n_channels: int = Field(default=16, ge=1, description="Number of vertical scan beams.")
    v_fov_deg: tuple[float, float] = Field(default=(-15.0, 15.0), description="(min_elevation_deg, max_elevation_deg).")
    h_resolution: int = Field(default=1800, ge=1, description="Azimuth steps per revolution.")
    max_range_m: float = Field(default=100.0, gt=0, description="Maximum measurable range (m).")
    no_hit_value: float = Field(default=0.0, description="Value written for beams with no return.")
    range_noise_sigma_m: float = Field(default=0.02, ge=0.0, description="Gaussian range noise sigma (m).")
    intensity_noise_sigma: float = Field(default=0.01, ge=0.0, description="Gaussian intensity noise sigma (0-1).")
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0, description="Per-beam random dropout probability.")
    rain_rate_mm_h: float = Field(default=0.0, ge=0.0, description="Rain rate for two-way attenuation (mm/h).")
    fog_density: float = Field(default=0.0, ge=0.0, description="Fog extinction coefficient (1/m).")
    channel_offsets_m: list[float] | None = Field(default=None, description="Per-channel calibration offsets (m).")
    beam_divergence_mrad: float = Field(
        default=0.0,
        ge=0.0,
        description="Half-angle beam divergence (mrad).  0 = off; typical spinning LiDAR: 1.5–3.0 mrad.",
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _v_fov_ordered(self) -> "LidarConfig":
        lo, hi = self.v_fov_deg
        if lo >= hi:
            raise ValueError(f"v_fov_deg must satisfy min < max, got {self.v_fov_deg}")
        return self


# ---------------------------------------------------------------------------
# GNSSConfig
# ---------------------------------------------------------------------------


class GNSSConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.GNSSModel`."""

    name: str = "gnss"
    update_rate_hz: float = Field(default=10.0, gt=0)
    noise_m: float = Field(default=1.5, ge=0.0, description="1-sigma position noise (m).")
    vel_noise_ms: float = Field(default=0.05, ge=0.0, description="1-sigma velocity noise (m/s).")
    bias_tau_s: float = Field(
        default=60.0,
        gt=0,
        description="Gauss-Markov bias time constant (s).",
    )
    bias_sigma_m: float = Field(default=0.5, ge=0.0, description="Steady-state bias sigma (m).")
    multipath_sigma_m: float = Field(default=1.0, ge=0.0, description="Multipath error sigma (m).")
    min_fix_altitude_m: float = Field(default=0.5, description="Altitude below which fix degrades (m).")
    jammer_zones: list[tuple[list[float], float]] = Field(
        default_factory=list,
        description="List of (centre_xyz, radius_m) jammer zones.  "
        "``centre_xyz`` is a list of 3 floats [x, y, z] in world-frame metres.",
    )
    origin_llh: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="(lat_deg, lon_deg, alt_m) of world origin.",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# RadioConfig
# ---------------------------------------------------------------------------


class RadioConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.RadioLinkModel`."""

    name: str = "radio"
    update_rate_hz: float = Field(default=100.0, gt=0)
    tx_power_dbm: float = Field(default=20.0, description="Transmit power (dBm).")
    frequency_ghz: float = Field(default=2.4, gt=0, description="Carrier frequency (GHz).")
    noise_figure_db: float = Field(default=6.0, ge=0.0, description="Receiver noise figure (dB).")
    path_loss_exponent: float = Field(default=2.5, ge=2.0, description="Log-distance path-loss exponent.")
    shadowing_sigma_db: float = Field(default=4.0, ge=0.0, description="Log-normal shadow fading sigma (dB).")
    min_snr_db: float = Field(default=-5.0, description="SNR below which PER → 1 (dB).")
    snr_transition_db: float = Field(default=10.0, gt=0, description="SNR range for PER sigmoid transition (dB).")
    base_latency_s: float = Field(default=0.001, ge=0.0, description="Minimum delivery latency (s).")
    jitter_sigma_s: float = Field(default=0.0005, ge=0.0, description="Latency jitter sigma (s).")
    nlos_excess_loss_db: float = Field(default=20.0, ge=0.0, description="Extra path loss when no LoS (dB).")
    los_required: bool = Field(default=False, description="Drop packets immediately when no LoS.")
    seed: int | None = None


# ---------------------------------------------------------------------------
# IMUConfig
# ---------------------------------------------------------------------------


class IMUConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.IMUModel`."""

    name: str = "imu"
    update_rate_hz: float = Field(default=200.0, gt=0, description="IMU output rate (Hz).")
    noise_density_acc: float = Field(
        default=2.0e-3,
        gt=0,
        description="Accelerometer white-noise density (m/s²/√Hz).  Typical MEMS: 2–5 × 10⁻³.",
    )
    noise_density_gyr: float = Field(
        default=1.7e-4,
        gt=0,
        description="Gyroscope white-noise density (rad/s/√Hz).  Typical MEMS: 1–5 × 10⁻⁴.",
    )
    bias_tau_acc_s: float = Field(default=300.0, gt=0, description="Accelerometer bias correlation time (s).")
    bias_sigma_acc: float = Field(default=5.0e-3, ge=0.0, description="Steady-state accelerometer bias sigma (m/s²).")
    bias_tau_gyr_s: float = Field(default=300.0, gt=0, description="Gyroscope bias correlation time (s).")
    bias_sigma_gyr: float = Field(default=1.0e-4, ge=0.0, description="Steady-state gyroscope bias sigma (rad/s).")
    scale_factor_acc: float = Field(
        default=0.0, ge=-1.0, description="Relative accelerometer scale-factor error (≥ −1)."
    )
    scale_factor_gyr: float = Field(default=0.0, ge=-1.0, description="Relative gyroscope scale-factor error (≥ −1).")
    add_gravity: bool = Field(
        default=True,
        description="Add gravity vector (from state['gravity_body']) to acceleration, mimicking specific-force output.",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# SensorSuiteConfig
# ---------------------------------------------------------------------------


class SensorSuiteConfig(BaseModel):
    """
    Top-level configuration for a complete :class:`~genesis.sensors.SensorSuite`.

    Each field corresponds to one sensor in the suite.  Set a field to
    ``None`` to disable that sensor entirely.

    Example
    -------
    ::

        cfg = SensorSuiteConfig(
            rgb=CameraConfig(iso=400, jpeg_quality=60),
            gnss=GNSSConfig(noise_m=0.5),
            lidar=None,     # disabled
            event=None,
            thermal=None,
            radio=None,
        )
        suite = SensorSuite.from_config(cfg)
    """

    rgb: CameraConfig | None = Field(default_factory=CameraConfig, description="RGB camera (None = disabled).")
    event: EventCameraConfig | None = Field(default=None, description="Event camera (None = disabled).")
    thermal: ThermalCameraConfig | None = Field(default=None, description="Thermal camera (None = disabled).")
    lidar: LidarConfig | None = Field(default_factory=LidarConfig, description="LiDAR (None = disabled).")
    gnss: GNSSConfig | None = Field(default_factory=GNSSConfig, description="GNSS (None = disabled).")
    radio: RadioConfig | None = Field(default_factory=RadioConfig, description="Radio link (None = disabled).")
    imu: IMUConfig | None = Field(default_factory=IMUConfig, description="IMU (None = disabled).")

    @classmethod
    def minimal(cls) -> "SensorSuiteConfig":
        """Return a config with only GNSS enabled (lightest default)."""
        return cls(
            rgb=None,
            event=None,
            thermal=None,
            lidar=None,
            radio=None,
            imu=None,
        )

    @classmethod
    def all_disabled(cls) -> "SensorSuiteConfig":
        """Return a config with every sensor disabled."""
        return cls(rgb=None, event=None, thermal=None, lidar=None, gnss=None, radio=None, imu=None)

    @classmethod
    def full(cls) -> "SensorSuiteConfig":
        """Return a config with every sensor enabled at default settings."""
        return cls(
            rgb=CameraConfig(),
            event=EventCameraConfig(),
            thermal=ThermalCameraConfig(),
            lidar=LidarConfig(),
            gnss=GNSSConfig(),
            radio=RadioConfig(),
            imu=IMUConfig(),
        )


__all__ = [
    "CameraConfig",
    "EventCameraConfig",
    "GNSSConfig",
    "IMUConfig",
    "LidarConfig",
    "RadioConfig",
    "SensorSuiteConfig",
    "ThermalCameraConfig",
]
