"""
Real-world sensor presets for the Genesis sensor layer.

Each constant is a fully-validated ``*Config`` instance whose parameters are
taken from publicly available datasheets.  They are module-level singletons
and should be treated as read-only; pass them directly to the corresponding
sensor's ``from_config()`` factory or copy-with-changes via
``model.model_copy(update={...})`` before mutating.

Usage
-----
::

    from genesis.sensors.presets import RASPBERRY_PI_V2, get_preset, list_presets

    # Use directly
    cam = CameraModel.from_config(RASPBERRY_PI_V2)

    # Discover available presets
    print(list_presets())           # ["RASPBERRY_PI_V2", "INTEL_D435_RGB", ...]

    # Retrieve by name (case-insensitive)
    cfg = get_preset("velodyne_vlp16")

Sources are cited inline as ``# Source: <URL or spec>``.
"""

from __future__ import annotations

from .config import CameraConfig, GNSSConfig, IMUConfig, LidarConfig

# ---------------------------------------------------------------------------
# Type alias for all preset config types
# ---------------------------------------------------------------------------

PresetConfig = CameraConfig | LidarConfig | IMUConfig | GNSSConfig

# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

RASPBERRY_PI_V2 = CameraConfig(
    # Source: https://www.raspberrypi.com/products/camera-module-v2/
    # Sony IMX219, 8 MP, 30 fps @ 1080p, rolling-shutter CMOS.
    # Readout time ≈ 19 ms at 30 fps (33.3 ms frame period) → RS fraction ≈ 0.95.
    name="RASPBERRY_PI_V2",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.95,
    # IMX219 sensitivity: ~1.1 e⁻/ADU at ISO 100; read noise ≈ 2 e⁻.
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=2.0,
    full_well_electrons=4600.0,
    # Fixed-pattern noise: ~0.01 % dead pixels reported in production batches.
    dead_pixel_fraction=0.0001,
    hot_pixel_fraction=0.00005,
    # Typical CMOS lens shading effect — moderate vignetting.
    vignetting_strength=0.3,
    # IMX219 has small lateral CA at the edges — 0.5 px max shift.
    chromatic_aberration_px=0.5,
)

INTEL_D435_RGB = CameraConfig(
    # Source: https://www.intelrealsense.com/depth-camera-d435/
    # Intel RealSense D435 RGB imager: OV2740, 1920×1080 @ 30 fps, rolling shutter.
    # Readout ≈ 26.4 ms of 33.3 ms frame → RS fraction ≈ 0.80.
    name="INTEL_D435_RGB",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.80,
    base_iso=100.0,
    iso=200.0,
    # OV2740 read noise ≈ 3 e⁻ at low gain.
    read_noise_sigma=3.0,
    full_well_electrons=3200.0,
    dead_pixel_fraction=0.0002,
    hot_pixel_fraction=0.0001,
    # Moderate vignetting from narrow fixed-focus lens.
    vignetting_strength=0.35,
    chromatic_aberration_px=0.8,
)

GOPRO_HERO11_4K30 = CameraConfig(
    # Source: https://community.gopro.com/s/article/HERO11-Black-FAQ
    # GoPro HERO 11 Black: 4K30, CMOS rolling shutter.
    # Typical RS skew for action cameras ≈ 16–18 ms / 33 ms → fraction ≈ 0.85.
    name="GOPRO_HERO11_4K30",
    update_rate_hz=30.0,
    resolution=(3840, 2160),
    rolling_shutter_fraction=0.85,
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=2.5,
    full_well_electrons=5000.0,
    dead_pixel_fraction=0.00005,
    hot_pixel_fraction=0.00002,
    # Wide-angle lens shows stronger vignetting & CA than typical surveillance cam.
    vignetting_strength=0.45,
    chromatic_aberration_px=1.2,
)

ZED2_LEFT = CameraConfig(
    # Source: https://www.stereolabs.com/assets/datasheets/zed2-datasheet.pdf
    # ZED 2 stereo camera: 1080p @ 30 fps per eye, rolling-shutter Sony IMX326.
    # Readout ≈ 25 ms / 33.3 ms → RS fraction ≈ 0.75.
    name="ZED2_LEFT",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.75,
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=1.8,
    full_well_electrons=4000.0,
    dead_pixel_fraction=0.0001,
    hot_pixel_fraction=0.00004,
    vignetting_strength=0.25,
    chromatic_aberration_px=0.6,
)

# ---------------------------------------------------------------------------
# LiDAR presets
# ---------------------------------------------------------------------------

VELODYNE_VLP16 = LidarConfig(
    # Source: https://velodynelidar.com/wp-content/uploads/2019/12/63-9243-Rev-E-VLP-16-User-Manual.pdf
    # VLP-16 Puck: 16 channels, ±15° VFOV, 360° @ 10 Hz, ~1800 azimuth steps.
    # Range: 100 m, range noise: ±3 cm (1σ).
    name="VELODYNE_VLP16",
    update_rate_hz=10.0,
    n_channels=16,
    v_fov_deg=(-15.0, 15.0),
    h_resolution=1800,
    max_range_m=100.0,
    range_noise_sigma_m=0.03,
    intensity_noise_sigma=0.02,
    dropout_prob=0.001,
    beam_divergence_mrad=1.5,
)

VELODYNE_HDL64E = LidarConfig(
    # Source: https://velodynelidar.com/wp-content/uploads/2019/12/97-0038-Rev-N-HDL-64E-S3-S3D-DS.pdf
    # HDL-64E: 64 channels, -24.8° to +2° VFOV, 360° @ 10 Hz, 4500 azimuth steps.
    # Range: 120 m, range noise: ±2 cm.
    name="VELODYNE_HDL64E",
    update_rate_hz=10.0,
    n_channels=64,
    v_fov_deg=(-24.8, 2.0),
    h_resolution=4500,
    max_range_m=120.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.01,
    dropout_prob=0.0005,
    beam_divergence_mrad=1.3,
)

OUSTER_OS1_64 = LidarConfig(
    # Source: https://ouster.com/downloads/OS1_Lidar_Product_Datasheet.pdf
    # Ouster OS1-64: 64 channels, ±22.5° VFOV, 1024 pts/scan, 10/20 Hz.
    # Range: 120 m, range accuracy: ±3 cm.
    name="OUSTER_OS1_64",
    update_rate_hz=20.0,
    n_channels=64,
    v_fov_deg=(-22.5, 22.5),
    h_resolution=1024,
    max_range_m=120.0,
    range_noise_sigma_m=0.03,
    intensity_noise_sigma=0.01,
    dropout_prob=0.001,
    beam_divergence_mrad=2.0,
)

LIVOX_AVIA = LidarConfig(
    # Source: https://www.livoxtech.com/avia/specs
    # Livox Avia: 70.4°×77.2° non-repetitive FOV, 240k pts/s, range 450 m.
    # 6 channels modelled as equispaced over a 70° VFOV; 10 Hz rotation equivalent.
    # Range noise: 2 cm (1σ); beam divergence: 0.28 mrad (tight solid-state beam).
    name="LIVOX_AVIA",
    update_rate_hz=10.0,
    n_channels=6,
    v_fov_deg=(-35.0, 35.0),
    h_resolution=900,
    max_range_m=450.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.005,
    dropout_prob=0.0005,
    beam_divergence_mrad=0.28,
)

# ---------------------------------------------------------------------------
# IMU presets
# ---------------------------------------------------------------------------

PIXHAWK_ICM20689 = IMUConfig(
    # Source: TDK InvenSense ICM-20689 datasheet, rev 1.1
    # Pixhawk 4 primary IMU.
    # Accel noise density: 300 µg/√Hz ≈ 2.94e-3 m/s²/√Hz
    # Gyro noise density: 0.01 °/s/√Hz ≈ 1.75e-4 rad/s/√Hz
    # Bias instability (accel): 0.05 mg; (gyro): 3.8 °/hr.
    name="PIXHAWK_ICM20689",
    update_rate_hz=400.0,
    noise_density_acc=2.94e-3,
    noise_density_gyr=1.75e-4,
    bias_tau_acc_s=200.0,
    bias_sigma_acc=4.9e-4,  # 0.05 mg in m/s²
    bias_tau_gyr_s=200.0,
    bias_sigma_gyr=1.83e-5,  # 3.8 °/hr in rad/s
    scale_factor_acc=3e-3,
    scale_factor_gyr=3e-3,
    add_gravity=True,
)

VECTORNAV_VN100 = IMUConfig(
    # Source: VectorNav VN-100 datasheet (Rev 2)
    # Tactical-grade MEMS IMU.
    # Accel noise density: 0.14 mg/√Hz ≈ 1.37e-3 m/s²/√Hz
    # Gyro noise density: 0.0035 °/s/√Hz ≈ 6.11e-5 rad/s/√Hz
    # Bias instability (accel): 0.04 mg; (gyro): 10 °/hr.
    name="VECTORNAV_VN100",
    update_rate_hz=800.0,
    noise_density_acc=1.37e-3,
    noise_density_gyr=6.11e-5,
    bias_tau_acc_s=500.0,
    bias_sigma_acc=3.92e-4,  # 0.04 mg in m/s²
    bias_tau_gyr_s=500.0,
    bias_sigma_gyr=4.85e-5,  # 10 °/hr in rad/s
    scale_factor_acc=1e-3,
    scale_factor_gyr=1e-3,
    add_gravity=True,
)

XSENS_MTI_3 = IMUConfig(
    # Source: Xsens MTi-3 datasheet, Document MT0605P, Rev AE
    # Accel noise density: 120 µg/√Hz ≈ 1.18e-3 m/s²/√Hz
    # Gyro noise density: 0.007 °/s/√Hz ≈ 1.22e-4 rad/s/√Hz
    # Accel bias stability: 15 µg; gyro in-run bias: 2 °/hr.
    name="XSENS_MTI_3",
    update_rate_hz=400.0,
    noise_density_acc=1.18e-3,
    noise_density_gyr=1.22e-4,
    bias_tau_acc_s=300.0,
    bias_sigma_acc=1.47e-4,  # 15 µg in m/s²
    bias_tau_gyr_s=300.0,
    bias_sigma_gyr=9.70e-6,  # 2 °/hr in rad/s
    scale_factor_acc=5e-4,
    scale_factor_gyr=5e-4,
    add_gravity=True,
)

# ---------------------------------------------------------------------------
# GNSS presets
# ---------------------------------------------------------------------------

UBLOX_M8N = GNSSConfig(
    # Source: u-blox NEO-M8N datasheet, UBX-13003366
    # CEP: 2.5 m, velocity accuracy: 0.05 m/s (RMS), update rate: 10 Hz.
    # 1-sigma per-axis (2D isotropic Gaussian): σ = CEP / √(2 ln 2) ≈ 2.5 / 1.1774 ≈ 2.12 m.
    name="UBLOX_M8N",
    update_rate_hz=10.0,
    noise_m=2.12,
    vel_noise_ms=0.05,
    bias_tau_s=60.0,
    bias_sigma_m=0.5,
    multipath_sigma_m=1.5,
    min_fix_altitude_m=0.5,
)

UBLOX_F9P_RTK = GNSSConfig(
    # Source: u-blox ZED-F9P datasheet, UBX-17051259
    # RTK fix: 0.01 m CEP (horizontal), heading accuracy: 0.4 °.
    # 1-sigma per-axis (from CEP): σ = 0.01 m / 1.1774 ≈ 0.0085 m; using 0.012 m
    # to include residual tropospheric / multipath contributions.
    # Float solution degrades to ~0.1 m.
    name="UBLOX_F9P_RTK",
    update_rate_hz=20.0,
    noise_m=0.012,
    vel_noise_ms=0.005,
    bias_tau_s=120.0,
    bias_sigma_m=0.01,
    multipath_sigma_m=0.05,
    min_fix_altitude_m=0.1,
)

NOVATEL_OEM7 = GNSSConfig(
    # Source: NovAtel OEM7 Solutions datasheet, OM-20000129
    # Autonomous GNSS: 1.5 m RMS horizontal, 2.5 m vertical.
    # SBAS-corrected: 0.6 m horizontal.  Using autonomous here.
    # Velocity: 0.03 m/s RMS.
    name="NOVATEL_OEM7",
    update_rate_hz=20.0,
    noise_m=1.5,
    vel_noise_ms=0.03,
    bias_tau_s=90.0,
    bias_sigma_m=0.3,
    multipath_sigma_m=0.8,
    min_fix_altitude_m=0.3,
)

# ---------------------------------------------------------------------------
# Registry and helpers
# ---------------------------------------------------------------------------

# Category → list of preset names; used by list_presets(kind=...).
_PRESET_CATEGORIES: dict[str, list[str]] = {
    "camera": ["GOPRO_HERO11_4K30", "INTEL_D435_RGB", "RASPBERRY_PI_V2", "ZED2_LEFT"],
    "lidar": ["LIVOX_AVIA", "OUSTER_OS1_64", "VELODYNE_HDL64E", "VELODYNE_VLP16"],
    "imu": ["PIXHAWK_ICM20689", "VECTORNAV_VN100", "XSENS_MTI_3"],
    "gnss": ["NOVATEL_OEM7", "UBLOX_F9P_RTK", "UBLOX_M8N"],
}

_REGISTRY: dict[str, PresetConfig] = {
    # Cameras
    "RASPBERRY_PI_V2": RASPBERRY_PI_V2,
    "INTEL_D435_RGB": INTEL_D435_RGB,
    "GOPRO_HERO11_4K30": GOPRO_HERO11_4K30,
    "ZED2_LEFT": ZED2_LEFT,
    # LiDAR
    "VELODYNE_VLP16": VELODYNE_VLP16,
    "VELODYNE_HDL64E": VELODYNE_HDL64E,
    "OUSTER_OS1_64": OUSTER_OS1_64,
    "LIVOX_AVIA": LIVOX_AVIA,
    # IMU
    "PIXHAWK_ICM20689": PIXHAWK_ICM20689,
    "VECTORNAV_VN100": VECTORNAV_VN100,
    "XSENS_MTI_3": XSENS_MTI_3,
    # GNSS
    "UBLOX_M8N": UBLOX_M8N,
    "UBLOX_F9P_RTK": UBLOX_F9P_RTK,
    "NOVATEL_OEM7": NOVATEL_OEM7,
}


def list_presets(kind: str | None = None) -> list[str]:
    """
    Return a sorted list of all available preset names.

    Parameters
    ----------
    kind:
        Optional sensor-type filter.  Accepted values: ``"camera"``,
        ``"lidar"``, ``"imu"``, ``"gnss"``.  When *None* (default) all
        presets are returned.

    Raises
    ------
    KeyError
        If *kind* is not a recognised sensor category.

    Examples
    --------
    ::

        list_presets()                # all 14 presets, sorted
        list_presets(kind="lidar")    # ["LIVOX_AVIA", "OUSTER_OS1_64", ...]
        list_presets(kind="gnss")     # ["NOVATEL_OEM7", "UBLOX_F9P_RTK", "UBLOX_M8N"]
    """
    if kind is None:
        return sorted(_REGISTRY)
    key = kind.lower()
    if key not in _PRESET_CATEGORIES:
        available = ", ".join(sorted(_PRESET_CATEGORIES))
        raise KeyError(f"Unknown sensor kind {kind!r}.  Available kinds: {available}")
    return sorted(_PRESET_CATEGORIES[key])


def get_preset(name: str) -> PresetConfig:
    """
    Return a preset config by name (case-insensitive).

    Parameters
    ----------
    name:
        Preset identifier, e.g. ``"VELODYNE_VLP16"`` or ``"velodyne_vlp16"``.

    Returns
    -------
    PresetConfig
        One of :class:`~genesis.sensors.CameraConfig`,
        :class:`~genesis.sensors.LidarConfig`,
        :class:`~genesis.sensors.IMUConfig`, or
        :class:`~genesis.sensors.GNSSConfig`.

    Raises
    ------
    KeyError
        If *name* does not match any known preset.

    Examples
    --------
    ::

        from genesis.sensors.presets import get_preset
        cfg = get_preset("velodyne_vlp16")
        lidar = LidarModel.from_config(cfg)
    """
    key = name.upper()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown sensor preset {name!r}.  Available: {available}")
    return _REGISTRY[key]


__all__ = [
    # Type alias
    "PresetConfig",
    # Camera presets
    "RASPBERRY_PI_V2",
    "INTEL_D435_RGB",
    "GOPRO_HERO11_4K30",
    "ZED2_LEFT",
    # LiDAR presets
    "VELODYNE_VLP16",
    "VELODYNE_HDL64E",
    "OUSTER_OS1_64",
    "LIVOX_AVIA",
    # IMU presets
    "PIXHAWK_ICM20689",
    "VECTORNAV_VN100",
    "XSENS_MTI_3",
    # GNSS presets
    "UBLOX_M8N",
    "UBLOX_F9P_RTK",
    "NOVATEL_OEM7",
    # Helpers
    "get_preset",
    "list_presets",
]
