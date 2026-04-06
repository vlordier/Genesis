"""
Parametrised tests for the Genesis external sensor layer.

Fixtures are declared in ``tests/conftest.py`` (``sensor_resolution``,
``sensor_rgb_image``, ``sensor_gray_image``, ``sensor_seg_mask``,
``sensor_range_image``, ``sensor_lidar_n_channels``, ``sensor_gnss_noise_m``,
``sensor_thermal_bit_depth``, ``sensor_camera_iso``).

These tests require no Genesis scene and no EGL/OpenGL — fully headless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from genesis.sensors import (
    CameraModel,
    EventCameraModel,
    GNSSModel,
    IMUModel,
    LidarModel,
    RadioLinkModel,
    SensorSuite,
    ThermalCameraModel,
)
from genesis.sensors.config import (
    CameraConfig,
    EventCameraConfig,
    GNSSConfig,
    IMUConfig,
    LidarConfig,
    RadioConfig,
    SensorSuiteConfig,
    ThermalCameraConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RGB_CHANNELS = 3
_LIDAR_POINT_COLS = 4  # x, y, z, intensity


# ===========================================================================
# Camera — resolution × ISO
# ===========================================================================


class TestCameraModelResolutionISO:
    """CameraModel behaves correctly across resolutions and ISO values."""

    def test_output_shape_matches_resolution(
        self,
        sensor_resolution: tuple[int, int],
        sensor_camera_iso: float,
        sensor_rgb_image: np.ndarray,
    ) -> None:
        """Output RGB array must have shape (H, W, 3) matching configured resolution."""
        w, h = sensor_resolution
        cam = CameraModel(
            resolution=(w, h),
            iso=sensor_camera_iso,
            base_iso=100.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        obs = cam.step(0.0, {"rgb": sensor_rgb_image})
        assert obs["rgb"].shape == (h, w, _RGB_CHANNELS)

    def test_output_dtype_is_uint8(
        self,
        sensor_resolution: tuple[int, int],
        sensor_camera_iso: float,
        sensor_rgb_image: np.ndarray,
    ) -> None:
        """Output must always be uint8."""
        w, h = sensor_resolution
        cam = CameraModel(resolution=(w, h), iso=sensor_camera_iso, seed=0)
        obs = cam.step(0.0, {"rgb": sensor_rgb_image})
        assert obs["rgb"].dtype == np.uint8

    def test_output_values_in_range(
        self,
        sensor_resolution: tuple[int, int],
        sensor_rgb_image: np.ndarray,
    ) -> None:
        """All pixel values must be in [0, 255]."""
        w, h = sensor_resolution
        cam = CameraModel(
            resolution=(w, h),
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        obs = cam.step(0.0, {"rgb": sensor_rgb_image})
        assert obs["rgb"].min() >= 0
        assert obs["rgb"].max() <= 255

    def test_float32_and_uint8_inputs_both_accepted(
        self,
        sensor_resolution: tuple[int, int],
        sensor_camera_iso: float,
    ) -> None:
        """uint8 and float32 inputs must both produce valid uint8 output."""
        w, h = sensor_resolution
        rng = np.random.default_rng(seed=1)
        uint8_img = rng.integers(0, 255, (h, w, _RGB_CHANNELS), dtype=np.uint8)
        float_img = uint8_img.astype(np.float32) / 255.0

        cam = CameraModel(resolution=(w, h), iso=sensor_camera_iso, seed=0)
        obs_u8 = cam.step(0.0, {"rgb": uint8_img})
        cam.reset()
        obs_f32 = cam.step(0.0, {"rgb": float_img})
        # Both should produce uint8 output of the same shape.
        assert obs_u8["rgb"].dtype == np.uint8
        assert obs_f32["rgb"].dtype == np.uint8
        assert obs_u8["rgb"].shape == obs_f32["rgb"].shape

    def test_high_iso_produces_more_noise(
        self,
        sensor_resolution: tuple[int, int],
        sensor_rgb_image: np.ndarray,
    ) -> None:
        """Higher ISO should produce a noisier image (measured on a mid-grey input)."""
        w, h = sensor_resolution
        # Use a mid-grey image to avoid saturation clipping at high ISO.
        mid_grey = np.full((h, w, _RGB_CHANNELS), 128, dtype=np.uint8)
        cam_low = CameraModel(
            resolution=(w, h),
            iso=100.0,
            base_iso=100.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        cam_high = CameraModel(
            resolution=(w, h),
            iso=1600.0,
            base_iso=100.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        obs_low = cam_low.step(0.0, {"rgb": mid_grey})
        obs_high = cam_high.step(0.0, {"rgb": mid_grey})
        # High-ISO output should be significantly brighter (gain > 1).
        mean_low = float(obs_low["rgb"].astype(np.float32).mean())
        mean_high = float(obs_high["rgb"].astype(np.float32).mean())
        assert mean_high > mean_low, "Higher ISO should produce a brighter (more gained) output"

    def test_get_observation_matches_last_step(
        self,
        sensor_resolution: tuple[int, int],
        sensor_rgb_image: np.ndarray,
    ) -> None:
        """get_observation() must return the same object as the last step() result."""
        w, h = sensor_resolution
        cam = CameraModel(resolution=(w, h), seed=0)
        obs = cam.step(0.0, {"rgb": sensor_rgb_image})
        assert cam.get_observation() is obs

    def test_config_round_trip_preserves_resolution(
        self,
        sensor_resolution: tuple[int, int],
        sensor_camera_iso: float,
    ) -> None:
        """from_config(get_config()) must preserve resolution and ISO."""
        w, h = sensor_resolution
        cam = CameraModel(resolution=(w, h), iso=sensor_camera_iso)
        cfg = cam.get_config()
        cam2 = CameraModel.from_config(cfg)
        assert cam2.resolution == (w, h)
        assert cam2.iso == sensor_camera_iso


# ===========================================================================
# Event camera — threshold pairs
# ===========================================================================


class TestEventCameraThresholds:
    """EventCameraModel fires events according to threshold settings."""

    @pytest.mark.parametrize(
        "threshold_pos,threshold_neg",
        [
            (0.01, 0.01),  # very sensitive
            (0.2, 0.2),  # default
            (2.0, 2.0),  # very insensitive
        ],
        ids=["sensitive", "default", "insensitive"],
    )
    def test_no_events_on_first_frame(self, threshold_pos: float, threshold_neg: float) -> None:
        """First frame must always produce zero events (no previous reference)."""
        cam = EventCameraModel(
            threshold_pos=threshold_pos,
            threshold_neg=threshold_neg,
            seed=0,
        )
        gray = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        obs = cam.step(0.0, {"gray": gray})
        assert len(obs["events"]) == 0

    @pytest.mark.parametrize(
        "threshold_pos,threshold_neg,expect_many",
        [
            (0.01, 0.01, True),  # low threshold → many events
            (10.0, 10.0, False),  # very high threshold → no events
        ],
        ids=["low_threshold", "high_threshold"],
    )
    def test_threshold_controls_event_count(
        self, threshold_pos: float, threshold_neg: float, expect_many: bool
    ) -> None:
        """Low threshold produces more events than high threshold on identical input."""
        cam = EventCameraModel(
            threshold_pos=threshold_pos,
            threshold_neg=threshold_neg,
            seed=0,
        )
        rng = np.random.default_rng(seed=42)
        gray1 = rng.random((32, 32)).astype(np.float32)
        gray2 = rng.random((32, 32)).astype(np.float32)  # completely different frame
        cam.step(0.0, {"gray": gray1})
        obs = cam.step(0.05, {"gray": gray2})
        n_events = len(obs["events"])
        if expect_many:
            assert n_events > 0, "Expected events with low threshold on changing input"
        else:
            assert n_events == 0, "Expected no events with very high threshold"

    @pytest.mark.parametrize(
        "threshold_pos,threshold_neg",
        [(0.1, 0.1), (0.5, 0.5)],
        ids=["thr0.1", "thr0.5"],
    )
    def test_uniform_input_produces_no_events(self, threshold_pos: float, threshold_neg: float) -> None:
        """Two identical frames must produce zero events for any threshold."""
        cam = EventCameraModel(threshold_pos=threshold_pos, threshold_neg=threshold_neg, seed=0)
        gray = np.full((16, 16), 0.5, dtype=np.float32)
        cam.step(0.0, {"gray": gray})
        obs = cam.step(0.001, {"gray": gray})
        assert len(obs["events"]) == 0

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0], ids=["t0.1", "t0.5", "t1.0"])
    def test_reset_reinitialises_camera(self, threshold: float) -> None:
        """After reset(), the first frame should again produce zero events."""
        cam = EventCameraModel(threshold_pos=threshold, threshold_neg=threshold, seed=0)
        rng = np.random.default_rng(seed=0)
        cam.step(0.0, {"gray": rng.random((16, 16)).astype(np.float32)})
        cam.step(0.001, {"gray": rng.random((16, 16)).astype(np.float32)})
        cam.reset()
        obs = cam.step(0.0, {"gray": rng.random((16, 16)).astype(np.float32)})
        assert len(obs["events"]) == 0, "After reset, first frame must have zero events"

    @pytest.mark.parametrize("threshold", [0.1, 0.2], ids=["t0.1", "t0.2"])
    def test_event_config_round_trip(self, threshold: float) -> None:
        """from_config(get_config()) preserves threshold values."""
        cam = EventCameraModel(threshold_pos=threshold, threshold_neg=threshold)
        cfg = cam.get_config()
        cam2 = EventCameraModel.from_config(cfg)
        assert cam2.threshold_pos == pytest.approx(threshold)
        assert cam2.threshold_neg == pytest.approx(threshold)


# ===========================================================================
# Thermal camera — bit depth
# ===========================================================================


class TestThermalCameraModelBitDepth:
    """ThermalCameraModel outputs correct dtype and range for each bit depth."""

    def test_output_dtype_matches_bit_depth(
        self,
        sensor_thermal_bit_depth: int,
        sensor_seg_mask: np.ndarray,
    ) -> None:
        """Output dtype must be uint8 for ≤8-bit, uint16 for >8-bit."""
        h, w = sensor_seg_mask.shape
        cam = ThermalCameraModel(
            resolution=(w, h),
            bit_depth=sensor_thermal_bit_depth,
            noise_sigma=0.0,
            nuc_sigma=0.0,
            seed=0,
        )
        temp_map = {0: 25.0, 1: 60.0, 2: 100.0, 3: -10.0}
        obs = cam.step(0.0, {"seg": sensor_seg_mask, "temperature_map": temp_map})
        expected_dtype = np.uint8 if sensor_thermal_bit_depth <= 8 else np.uint16
        assert obs["thermal"].dtype == expected_dtype

    def test_output_range_within_bit_depth(
        self,
        sensor_thermal_bit_depth: int,
        sensor_seg_mask: np.ndarray,
    ) -> None:
        """All pixel values must fit within [0, 2^bit_depth - 1]."""
        h, w = sensor_seg_mask.shape
        cam = ThermalCameraModel(
            resolution=(w, h),
            bit_depth=sensor_thermal_bit_depth,
            noise_sigma=0.0,
            nuc_sigma=0.0,
            seed=0,
        )
        temp_map = {0: 25.0, 1: 60.0, 2: 100.0, 3: -10.0}
        obs = cam.step(0.0, {"seg": sensor_seg_mask, "temperature_map": temp_map})
        max_val = 2**sensor_thermal_bit_depth - 1
        assert int(obs["thermal"].max()) <= max_val
        assert int(obs["thermal"].min()) >= 0

    def test_temperature_c_float_output(
        self,
        sensor_thermal_bit_depth: int,
        sensor_seg_mask: np.ndarray,
    ) -> None:
        """temperature_c field must be a float32 array with the same spatial shape."""
        h, w = sensor_seg_mask.shape
        cam = ThermalCameraModel(
            resolution=(w, h),
            bit_depth=sensor_thermal_bit_depth,
            seed=0,
        )
        temp_map = {0: 20.0, 1: 50.0, 2: 80.0, 3: 10.0}
        obs = cam.step(0.0, {"seg": sensor_seg_mask, "temperature_map": temp_map})
        assert obs["temperature_c"].shape == (h, w)
        assert obs["temperature_c"].dtype == np.float32

    def test_fog_darkens_output(
        self,
        sensor_thermal_bit_depth: int,
        sensor_seg_mask: np.ndarray,
    ) -> None:
        """Dense fog must reduce the mean thermal intensity relative to clear conditions."""
        h, w = sensor_seg_mask.shape
        temp_map = {0: 80.0, 1: 80.0, 2: 80.0, 3: 80.0}

        cam_clear = ThermalCameraModel(
            resolution=(w, h), bit_depth=sensor_thermal_bit_depth, fog_density=0.0, noise_sigma=0.0, seed=0
        )
        cam_foggy = ThermalCameraModel(
            resolution=(w, h), bit_depth=sensor_thermal_bit_depth, fog_density=1.0, noise_sigma=0.0, seed=0
        )

        state = {"seg": sensor_seg_mask, "depth": np.full((h, w), 20.0, dtype=np.float32), "temperature_map": temp_map}
        obs_clear = cam_clear.step(0.0, state)
        obs_foggy = cam_foggy.step(0.0, state)

        mean_clear = float(obs_clear["thermal"].astype(np.float32).mean())
        mean_foggy = float(obs_foggy["thermal"].astype(np.float32).mean())
        assert mean_foggy <= mean_clear, "Fog should reduce or maintain thermal output"

    def test_config_bit_depth_round_trip(self, sensor_thermal_bit_depth: int) -> None:
        """get_config() / from_config() must preserve bit_depth."""
        cam = ThermalCameraModel(bit_depth=sensor_thermal_bit_depth)
        cfg = cam.get_config()
        assert cfg.bit_depth == sensor_thermal_bit_depth
        cam2 = ThermalCameraModel.from_config(cfg)
        assert cam2.bit_depth == sensor_thermal_bit_depth


# ===========================================================================
# LiDAR — n_channels
# ===========================================================================


class TestLidarModelChannels:
    """LidarModel produces correct point clouds across different channel counts."""

    def test_point_cloud_is_2d_float32(
        self,
        sensor_lidar_n_channels: int,
        sensor_range_image: np.ndarray,
    ) -> None:
        """Points array must be 2-D float32 (N points × 4 columns)."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, seed=0)
        obs = lidar.step(0.0, {"range_image": sensor_range_image})
        pts = obs["points"]
        assert pts.ndim == 2  # shape is (N, 4): N points, each with x, y, z, intensity
        assert pts.dtype == np.float32

    def test_point_cloud_has_4_columns(
        self,
        sensor_lidar_n_channels: int,
        sensor_range_image: np.ndarray,
    ) -> None:
        """Each point must have columns x, y, z, intensity."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, seed=0)
        obs = lidar.step(0.0, {"range_image": sensor_range_image})
        assert obs["points"].shape[1] == _LIDAR_POINT_COLS

    def test_all_points_within_max_range(
        self,
        sensor_lidar_n_channels: int,
        sensor_range_image: np.ndarray,
    ) -> None:
        """No returned point should exceed max_range_m from the sensor origin."""
        max_range = 100.0
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, max_range_m=max_range, seed=0)
        obs = lidar.step(0.0, {"range_image": sensor_range_image})
        pts = obs["points"]
        if len(pts) > 0:
            ranges = np.linalg.norm(pts[:, :3], axis=1)
            assert ranges.max() <= max_range + 0.1  # allow small noise-induced overshoot

    def test_dropout_reduces_point_count(
        self,
        sensor_lidar_n_channels: int,
        sensor_range_image: np.ndarray,
    ) -> None:
        """100% dropout must produce zero points."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, dropout_prob=1.0, seed=0)
        obs = lidar.step(0.0, {"range_image": sensor_range_image})
        assert len(obs["points"]) == 0

    def test_range_image_shape_preserved(
        self,
        sensor_lidar_n_channels: int,
        sensor_range_image: np.ndarray,
    ) -> None:
        """The returned range_image must have the same shape as the input."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, seed=0)
        obs = lidar.step(0.0, {"range_image": sensor_range_image})
        assert obs["range_image"].shape == sensor_range_image.shape

    def test_no_range_image_returns_empty_points(
        self,
        sensor_lidar_n_channels: int,
    ) -> None:
        """When state has no range_image key, points array must be empty."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels, h_resolution=360, seed=0)
        obs = lidar.step(0.0, {})
        assert obs["points"].shape == (0, _LIDAR_POINT_COLS)

    def test_config_n_channels_round_trip(self, sensor_lidar_n_channels: int) -> None:
        """n_channels must survive get_config() / from_config()."""
        lidar = LidarModel(n_channels=sensor_lidar_n_channels)
        cfg = lidar.get_config()
        assert cfg.n_channels == sensor_lidar_n_channels
        lidar2 = LidarModel.from_config(cfg)
        assert lidar2.n_channels == sensor_lidar_n_channels


# ===========================================================================
# GNSS — noise_m
# ===========================================================================


class TestGNSSModelNoise:
    """GNSSModel noise parameter propagates correctly into observations."""

    def test_observation_keys_present(self, sensor_gnss_noise_m: float) -> None:
        """Every observation must contain the expected keys."""
        gnss = GNSSModel(noise_m=sensor_gnss_noise_m, seed=0)
        obs = gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        for key in ("pos", "vel", "pos_llh", "fix_quality", "n_satellites", "hdop"):
            assert key in obs

    def test_pos_shape_is_3(self, sensor_gnss_noise_m: float) -> None:
        """Returned position and velocity must be shape (3,)."""
        gnss = GNSSModel(noise_m=sensor_gnss_noise_m, seed=0)
        obs = gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        assert obs["pos"].shape == (3,)
        assert obs["vel"].shape == (3,)

    def test_zero_noise_stays_near_truth(self) -> None:
        """With zero noise and zero bias the returned pos should be very close to truth."""
        gnss = GNSSModel(
            noise_m=0.0,
            vel_noise_ms=0.0,
            bias_sigma_m=0.0,
            multipath_sigma_m=0.0,
            seed=0,
        )
        truth = np.array([10.0, -5.0, 100.0])
        obs = gnss.step(0.0, {"pos": truth, "vel": np.zeros(3)})
        np.testing.assert_allclose(obs["pos"], truth, atol=1e-6)

    def test_high_noise_increases_error(self) -> None:
        """Higher noise_m should on average produce larger position errors."""
        n_trials = 40
        errors_low, errors_high = [], []
        truth = np.zeros(3)
        for seed in range(n_trials):
            gnss_low = GNSSModel(noise_m=0.01, bias_sigma_m=0.0, multipath_sigma_m=0.0, seed=seed)
            gnss_high = GNSSModel(noise_m=10.0, bias_sigma_m=0.0, multipath_sigma_m=0.0, seed=seed)
            errors_low.append(np.linalg.norm(gnss_low.step(0.0, {"pos": truth, "vel": truth})["pos"] - truth))
            errors_high.append(np.linalg.norm(gnss_high.step(0.0, {"pos": truth, "vel": truth})["pos"] - truth))
        assert np.mean(errors_high) > np.mean(errors_low) * 2

    @pytest.mark.parametrize(
        "noise_m",
        [0.0, 1.5, 5.0],
        ids=["0m", "1.5m", "5m"],
    )
    def test_pos_llh_has_3_components(self, noise_m: float) -> None:
        """pos_llh must always be a (3,) array regardless of noise level."""
        gnss = GNSSModel(noise_m=noise_m, seed=0)
        obs = gnss.step(0.0, {"pos": np.array([100.0, 200.0, 50.0]), "vel": np.zeros(3)})
        assert obs["pos_llh"].shape == (3,)

    def test_config_noise_round_trip(self, sensor_gnss_noise_m: float) -> None:
        """noise_m must survive get_config() / from_config()."""
        gnss = GNSSModel(noise_m=sensor_gnss_noise_m)
        cfg = gnss.get_config()
        assert cfg.noise_m == pytest.approx(sensor_gnss_noise_m)
        gnss2 = GNSSModel.from_config(cfg)
        assert gnss2.noise_m == pytest.approx(sensor_gnss_noise_m)


# ===========================================================================
# IMU — noise_density_acc × add_gravity
# ===========================================================================


class TestIMUModelNoise:
    """IMUModel noise and gravity injection work correctly across parameters."""

    @pytest.mark.parametrize(
        "noise_density_acc,noise_density_gyr",
        [
            (0.0, 0.0),
            (2e-3, 1.7e-4),
            (5e-3, 5e-4),
        ],
        ids=["noiseless", "typical", "noisy"],
    )
    def test_observation_shapes_always_3(self, noise_density_acc: float, noise_density_gyr: float) -> None:
        """lin_acc and ang_vel must always be shape (3,)."""
        imu = IMUModel(
            noise_density_acc=max(noise_density_acc, 1e-9),
            noise_density_gyr=max(noise_density_gyr, 1e-9),
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            seed=0,
        )
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)})
        assert obs["lin_acc"].shape == (3,)
        assert obs["ang_vel"].shape == (3,)

    @pytest.mark.parametrize("add_gravity", [True, False], ids=["gravity_on", "gravity_off"])
    def test_gravity_injection_effect(self, add_gravity: bool) -> None:
        """add_gravity=True adds gravity vector; add_gravity=False leaves lin_acc near zero."""
        g = np.array([0.0, 0.0, 9.80665])
        imu = IMUModel(
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            add_gravity=add_gravity,
            seed=0,
        )
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": g})
        if add_gravity:
            np.testing.assert_allclose(obs["lin_acc"], g, atol=1e-9)
        else:
            np.testing.assert_allclose(obs["lin_acc"], np.zeros(3), atol=1e-9)

    @pytest.mark.parametrize(
        "noise_density_acc",
        [1e-4, 1e-3, 1e-2],
        ids=["low_noise", "med_noise", "high_noise"],
    )
    def test_higher_noise_density_increases_variance(self, noise_density_acc: float) -> None:
        """A sensor with 10× noise density should have ≥ stddev of the lower-noise sensor."""
        n_steps = 200
        g = np.zeros(3)
        imu_lo = IMUModel(noise_density_acc=noise_density_acc, bias_sigma_acc=0.0, seed=0)
        imu_hi = IMUModel(noise_density_acc=noise_density_acc * 10.0, bias_sigma_acc=0.0, seed=0)
        meas_lo = np.array(
            [imu_lo.step(i * 0.005, {"lin_acc": g, "ang_vel": g, "gravity_body": g})["lin_acc"] for i in range(n_steps)]
        )
        meas_hi = np.array(
            [imu_hi.step(i * 0.005, {"lin_acc": g, "ang_vel": g, "gravity_body": g})["lin_acc"] for i in range(n_steps)]
        )
        assert meas_hi.std() >= meas_lo.std(), "10× noise density should produce ≥ stddev"

    @pytest.mark.parametrize(
        "scale_factor_acc",
        [-0.5, 0.0, 0.1, 0.5],
        ids=["sf-0.5", "sf0", "sf0.1", "sf0.5"],
    )
    def test_scale_factor_applied_correctly(self, scale_factor_acc: float) -> None:
        """lin_acc = (1 + scale_factor) * true_acc when noise and bias are zero."""
        imu = IMUModel(
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            scale_factor_acc=scale_factor_acc,
            add_gravity=False,
            seed=0,
        )
        true_acc = np.array([1.0, 2.0, 3.0])
        obs = imu.step(0.0, {"lin_acc": true_acc, "ang_vel": np.zeros(3)})
        expected = (1.0 + scale_factor_acc) * true_acc
        np.testing.assert_allclose(obs["lin_acc"], expected, atol=1e-9)

    @pytest.mark.parametrize("add_gravity", [True, False], ids=["grav_on", "grav_off"])
    def test_config_add_gravity_round_trip(self, add_gravity: bool) -> None:
        """add_gravity flag must survive get_config() / from_config()."""
        imu = IMUModel(add_gravity=add_gravity)
        cfg = imu.get_config()
        assert cfg.add_gravity == add_gravity
        imu2 = IMUModel.from_config(cfg)
        assert imu2.add_gravity == add_gravity


# ===========================================================================
# Radio — distance × los_required
# ===========================================================================


class TestRadioLinkModelDistance:
    """RadioLinkModel delivery probability varies correctly with distance and LoS."""

    @pytest.mark.parametrize(
        "distance_m,expect_delivery",
        [
            (1.0, True),  # 1 m — trivially delivered
            (1e6, False),  # 1 000 km — far beyond range
        ],
        ids=["close", "far"],
    )
    def test_delivery_depends_on_distance(self, distance_m: float, expect_delivery: bool) -> None:
        """Short-range packets should be delivered; extreme long-range should be dropped."""
        radio = RadioLinkModel(seed=0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([distance_m, 0.0, 0.0])
        n_delivered = 0
        n_trials = 20
        for i in range(n_trials):
            radio.transmit({"i": i}, src_pos=src, dst_pos=dst, sim_time=float(i) * 0.01)
            obs = radio.step(sim_time=float(i) * 0.01 + 1.0, state={})
            n_delivered += len(obs["delivered"])
        if expect_delivery:
            assert n_delivered > 0, "Short-range transmissions should deliver at least one packet"
        else:
            assert n_delivered == 0, "Extreme long-range transmissions should all be dropped"

    @pytest.mark.parametrize("los_required", [True, False], ids=["los_required", "los_optional"])
    def test_los_required_drops_when_obstructed(self, los_required: bool) -> None:
        """When los_required=True and LoS is absent the packet should be dropped."""
        radio = RadioLinkModel(los_required=los_required, seed=0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([10.0, 0.0, 0.0])
        # Transmit with no obstruction — state does not contain "los" so LoS check is skipped.
        radio.transmit("test", src_pos=src, dst_pos=dst, sim_time=0.0)
        obs = radio.step(sim_time=10.0, state={})
        # Both modes should return a valid observation dict structure.
        assert "delivered" in obs
        assert "queue_depth" in obs
        assert isinstance(obs["delivered"], list)

    @pytest.mark.parametrize(
        "tx_power_dbm",
        [10.0, 20.0, 30.0],
        ids=["10dBm", "20dBm", "30dBm"],
    )
    def test_higher_tx_power_improves_link(self, tx_power_dbm: float) -> None:
        """Estimate SNR: higher tx power must yield higher SNR."""
        radio = RadioLinkModel(tx_power_dbm=tx_power_dbm, seed=0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([100.0, 0.0, 0.0])
        metrics = radio.estimate_link_metrics(src_pos=src, dst_pos=dst)
        assert "snr_db" in metrics
        # Higher power → higher SNR (direct proportionality in log-domain)
        radio_ref = RadioLinkModel(tx_power_dbm=tx_power_dbm - 10.0, seed=0)
        metrics_ref = radio_ref.estimate_link_metrics(src_pos=src, dst_pos=dst)
        assert metrics["snr_db"] > metrics_ref["snr_db"]

    def test_config_round_trip_los_required(self) -> None:
        """los_required must survive get_config() / from_config()."""
        radio = RadioLinkModel(los_required=True, tx_power_dbm=25.0)
        cfg = radio.get_config()
        assert cfg.los_required is True
        assert cfg.tx_power_dbm == pytest.approx(25.0)
        radio2 = RadioLinkModel.from_config(cfg)
        assert radio2.los_required is True


# ===========================================================================
# All presets — instantiation + step round-trip
# ===========================================================================

_ALL_PRESET_NAMES = [
    "RASPBERRY_PI_V2",
    "INTEL_D435_RGB",
    "GOPRO_HERO11_4K30",
    "ZED2_LEFT",
    "VELODYNE_VLP16",
    "VELODYNE_HDL64E",
    "OUSTER_OS1_64",
    "LIVOX_AVIA",
    "PIXHAWK_ICM20689",
    "VECTORNAV_VN100",
    "XSENS_MTI_3",
    "UBLOX_M8N",
    "UBLOX_F9P_RTK",
    "NOVATEL_OEM7",
]

_CAMERA_PRESETS = {"RASPBERRY_PI_V2", "INTEL_D435_RGB", "GOPRO_HERO11_4K30", "ZED2_LEFT"}
_LIDAR_PRESETS = {"VELODYNE_VLP16", "VELODYNE_HDL64E", "OUSTER_OS1_64", "LIVOX_AVIA"}
_IMU_PRESETS = {"PIXHAWK_ICM20689", "VECTORNAV_VN100", "XSENS_MTI_3"}
_GNSS_PRESETS = {"UBLOX_M8N", "UBLOX_F9P_RTK", "NOVATEL_OEM7"}


class TestAllPresetInstantiation:
    """Every named preset produces a valid sensor and observation."""

    @pytest.mark.parametrize("preset_name", _ALL_PRESET_NAMES)
    def test_preset_has_positive_update_rate(self, preset_name: str) -> None:
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        assert cfg.update_rate_hz > 0.0

    @pytest.mark.parametrize("preset_name", list(_CAMERA_PRESETS))
    def test_camera_preset_step_returns_rgb(self, preset_name: str) -> None:
        """Camera presets must produce uint8 RGB output when stepped."""
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        w, h = cfg.resolution  # type: ignore[union-attr]
        cam = CameraModel.from_config(cfg)  # type: ignore[arg-type]
        rng = np.random.default_rng(seed=0)
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        obs = cam.step(0.0, {"rgb": rgb})
        assert "rgb" in obs
        assert obs["rgb"].shape == (h, w, 3)
        assert obs["rgb"].dtype == np.uint8

    @pytest.mark.parametrize("preset_name", list(_LIDAR_PRESETS))
    def test_lidar_preset_step_returns_points(self, preset_name: str) -> None:
        """LiDAR presets must produce a (N, 4) float32 point cloud."""
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        lidar = LidarModel.from_config(cfg)  # type: ignore[arg-type]
        rng = np.random.default_rng(seed=0)
        ri = rng.uniform(1.0, 50.0, (cfg.n_channels, cfg.h_resolution)).astype(np.float32)  # type: ignore[union-attr]
        obs = lidar.step(0.0, {"range_image": ri})
        assert "points" in obs
        assert obs["points"].ndim == 2
        assert obs["points"].shape[1] == _LIDAR_POINT_COLS

    @pytest.mark.parametrize("preset_name", list(_IMU_PRESETS))
    def test_imu_preset_step_returns_acc_gyr(self, preset_name: str) -> None:
        """IMU presets must produce lin_acc and ang_vel of shape (3,)."""
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        imu = IMUModel.from_config(cfg)  # type: ignore[arg-type]
        obs = imu.step(
            0.0,
            {
                "lin_acc": np.zeros(3, dtype=np.float64),
                "ang_vel": np.zeros(3, dtype=np.float64),
                "gravity_body": np.array([0.0, 0.0, 9.80665]),
            },
        )
        assert obs["lin_acc"].shape == (3,)
        assert obs["ang_vel"].shape == (3,)

    @pytest.mark.parametrize("preset_name", list(_GNSS_PRESETS))
    def test_gnss_preset_step_returns_pos_llh(self, preset_name: str) -> None:
        """GNSS presets must produce pos, vel, pos_llh observations."""
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        gnss = GNSSModel.from_config(cfg)  # type: ignore[arg-type]
        obs = gnss.step(0.0, {"pos": np.array([500.0, 300.0, 100.0]), "vel": np.zeros(3)})
        assert obs["pos_llh"].shape == (3,)
        assert obs["pos"].shape == (3,)

    @pytest.mark.parametrize("preset_name", _ALL_PRESET_NAMES)
    def test_preset_json_round_trip(self, preset_name: str) -> None:
        """All presets must serialise to JSON and back without data loss."""
        from genesis.sensors.presets import get_preset

        cfg = get_preset(preset_name)
        json_str = cfg.model_dump_json()
        # Re-validate using the concrete config class.
        cfg2 = type(cfg).model_validate_json(json_str)
        assert cfg2.update_rate_hz == pytest.approx(cfg.update_rate_hz)

    @pytest.mark.parametrize("preset_name", _ALL_PRESET_NAMES)
    def test_preset_case_insensitive_lookup(self, preset_name: str) -> None:
        """get_preset() must work for any case variant of the name."""
        from genesis.sensors.presets import get_preset

        cfg_upper = get_preset(preset_name.upper())
        cfg_lower = get_preset(preset_name.lower())
        assert cfg_upper == cfg_lower


# ===========================================================================
# SensorSuiteConfig combinations
# ===========================================================================

_SUITE_COMBINATIONS: list[tuple[str, dict]] = [
    (
        "rgb_only",
        {
            "rgb": CameraConfig(resolution=(16, 16)),
            "event": None,
            "thermal": None,
            "lidar": None,
            "gnss": None,
            "radio": None,
            "imu": None,
        },
    ),
    (
        "gnss_only",
        {
            "rgb": None,
            "event": None,
            "thermal": None,
            "lidar": None,
            "gnss": GNSSConfig(noise_m=0.5),
            "radio": None,
            "imu": None,
        },
    ),
    (
        "imu_only",
        {
            "rgb": None,
            "event": None,
            "thermal": None,
            "lidar": None,
            "gnss": None,
            "radio": None,
            "imu": IMUConfig(update_rate_hz=100.0),
        },
    ),
    (
        "lidar_only",
        {
            "rgb": None,
            "event": None,
            "thermal": None,
            "lidar": LidarConfig(n_channels=8, h_resolution=90),
            "gnss": None,
            "radio": None,
            "imu": None,
        },
    ),
    (
        "rgb_gnss_imu",
        {
            "rgb": CameraConfig(resolution=(16, 16)),
            "event": None,
            "thermal": None,
            "lidar": None,
            "gnss": GNSSConfig(noise_m=1.0),
            "radio": None,
            "imu": IMUConfig(),
        },
    ),
    (
        "all_disabled",
        {
            "rgb": None,
            "event": None,
            "thermal": None,
            "lidar": None,
            "gnss": None,
            "radio": None,
            "imu": None,
        },
    ),
]


class TestSuiteConfigCombinations:
    """SensorSuite behaves correctly for a variety of sensor-enabled combinations."""

    @pytest.mark.parametrize(
        "combo_name,sensor_kwargs",
        _SUITE_COMBINATIONS,
        ids=[c[0] for c in _SUITE_COMBINATIONS],
    )
    def test_suite_step_does_not_crash(self, combo_name: str, sensor_kwargs: dict) -> None:
        """suite.step() must complete without exception for every combo."""
        cfg = SensorSuiteConfig(**sensor_kwargs)
        suite = SensorSuite.from_config(cfg)
        suite.reset()

        rng = np.random.default_rng(seed=0)
        state = {
            "rgb": rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            "gray": rng.random((16, 16)).astype(np.float32),
            "seg": rng.integers(0, 4, (16, 16), dtype=np.int32),
            "pos": np.array([10.0, 20.0, 5.0]),
            "vel": np.zeros(3),
            "ang_vel": np.zeros(3),
            "lin_acc": np.zeros(3),
            "gravity_body": np.array([0.0, 0.0, 9.80665]),
            "range_image": rng.uniform(1.0, 50.0, (8, 90)).astype(np.float32),
            "temperature_map": {0: 20.0, 1: 40.0, 2: 60.0, 3: -5.0},
        }
        obs = suite.step(0.0, state)
        assert isinstance(obs, dict)

    @pytest.mark.parametrize(
        "combo_name,sensor_kwargs",
        _SUITE_COMBINATIONS,
        ids=[c[0] for c in _SUITE_COMBINATIONS],
    )
    def test_suite_sensor_names_match_enabled_sensors(self, combo_name: str, sensor_kwargs: dict) -> None:
        """sensor_names() must contain exactly the enabled sensors."""
        cfg = SensorSuiteConfig(**sensor_kwargs)
        suite = SensorSuite.from_config(cfg)
        names = suite.sensor_names()
        for key, val in sensor_kwargs.items():
            expected_name = key
            if val is not None:
                assert expected_name in names, f"{expected_name} should be in suite"
            else:
                assert expected_name not in names, f"{expected_name} should not be in suite"

    @pytest.mark.parametrize(
        "combo_name,sensor_kwargs",
        [(c[0], c[1]) for c in _SUITE_COMBINATIONS if c[0] != "all_disabled"],
        ids=[c[0] for c in _SUITE_COMBINATIONS if c[0] != "all_disabled"],
    )
    def test_suite_reset_then_step_succeeds(self, combo_name: str, sensor_kwargs: dict) -> None:
        """reset() followed by step() must work without error."""
        cfg = SensorSuiteConfig(**sensor_kwargs)
        suite = SensorSuite.from_config(cfg)
        suite.reset()
        suite.reset()  # double-reset should also be safe
        rng = np.random.default_rng(seed=0)
        state = {
            "rgb": rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            "pos": np.zeros(3),
            "vel": np.zeros(3),
            "ang_vel": np.zeros(3),
            "lin_acc": np.zeros(3),
            "gravity_body": np.zeros(3),
            "range_image": rng.uniform(1.0, 50.0, (8, 90)).astype(np.float32),
            "seg": rng.integers(0, 4, (16, 16), dtype=np.int32),
            "temperature_map": {0: 20.0},
        }
        obs = suite.step(0.0, state)
        assert isinstance(obs, dict)

    def test_all_disabled_suite_returns_empty_obs(self) -> None:
        """all_disabled config must produce an empty observation dict."""
        cfg = SensorSuiteConfig.all_disabled()
        suite = SensorSuite.from_config(cfg)
        suite.reset()
        obs = suite.step(0.0, {})
        assert obs == {}, f"Expected empty obs, got {list(obs.keys())}"

    def test_full_suite_config_has_all_sensors(self) -> None:
        """SensorSuiteConfig.full() must enable all 7 sensor types."""
        cfg = SensorSuiteConfig.full()
        for field in ("rgb", "event", "thermal", "lidar", "gnss", "radio", "imu"):
            assert getattr(cfg, field) is not None, f"{field} should not be None in full() config"

    @pytest.mark.parametrize(
        "combo_name,sensor_kwargs",
        _SUITE_COMBINATIONS,
        ids=[c[0] for c in _SUITE_COMBINATIONS],
    )
    def test_suite_json_config_round_trip(self, combo_name: str, sensor_kwargs: dict) -> None:
        """SensorSuiteConfig must serialise to JSON and reconstruct without error."""
        cfg = SensorSuiteConfig(**sensor_kwargs)
        json_str = cfg.model_dump_json()
        cfg2 = SensorSuiteConfig.model_validate_json(json_str)
        # Enabled/disabled status must be preserved for each sensor.
        for key in sensor_kwargs:
            original_enabled = sensor_kwargs[key] is not None
            restored_enabled = getattr(cfg2, key) is not None
            assert original_enabled == restored_enabled, (
                f"{key}: enabled={original_enabled} before round-trip, enabled={restored_enabled} after"
            )


# ===========================================================================
# Cross-sensor: is_due scheduling across parametrised rates
# ===========================================================================


class TestSensorSchedulingRates:
    """Verify is_due() / step() scheduling for various update rates."""

    @pytest.mark.parametrize(
        "rate_hz,dt",
        [
            (10.0, 0.1),
            (50.0, 0.02),
            (200.0, 0.005),
        ],
        ids=["10Hz", "50Hz", "200Hz"],
    )
    def test_sensor_fires_exactly_at_period(self, rate_hz: float, dt: float) -> None:
        """Sensor must be due after exactly one period has elapsed."""
        imu = IMUModel(update_rate_hz=rate_hz, seed=0)
        state = {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)}
        imu.step(0.0, state)
        assert not imu.is_due(dt * 0.5), "Should NOT be due before one period"
        assert imu.is_due(dt), "Should be due after one period"

    @pytest.mark.parametrize(
        "rate_hz",
        [1.0, 10.0, 100.0],
        ids=["1Hz", "10Hz", "100Hz"],
    )
    def test_sensor_not_due_before_period(self, rate_hz: float) -> None:
        """Sensor must not be due at sim_time < 1 / rate_hz after an update."""
        gnss = GNSSModel(update_rate_hz=rate_hz, noise_m=0.0, seed=0)
        gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        dt = 1.0 / rate_hz
        assert not gnss.is_due(dt * 0.9)

    @pytest.mark.parametrize(
        "rate_hz,n_steps",
        [
            (10.0, 10),
            (100.0, 5),
        ],
        ids=["10Hz-10steps", "100Hz-5steps"],
    )
    def test_multi_step_period_consistency(self, rate_hz: float, n_steps: int) -> None:
        """Stepping n times by exactly 1/rate should keep the sensor in sync."""
        dt = 1.0 / rate_hz
        imu = IMUModel(update_rate_hz=rate_hz, seed=0)
        state = {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)}
        for i in range(n_steps):
            t = i * dt
            assert imu.is_due(t), f"Step {i}: sensor should be due at t={t:.4f}"
            imu.step(t, state)


# ===========================================================================
# Config validation — cross-sensor parametrised
# ===========================================================================


class TestConfigValidation:
    """Pydantic validation rejects invalid fields consistently across sensor configs."""

    @pytest.mark.parametrize(
        "config_cls,invalid_kwargs,match",
        [
            (CameraConfig, {"update_rate_hz": -1.0}, ""),
            (CameraConfig, {"resolution": (0, 480)}, ""),
            (EventCameraConfig, {"threshold_pos": 0.0}, ""),
            (LidarConfig, {"v_fov_deg": (30.0, -15.0)}, ""),
            (LidarConfig, {"n_channels": 0}, ""),
            (ThermalCameraConfig, {"temp_range_c": (100.0, 50.0)}, ""),
            (IMUConfig, {"noise_density_acc": -1e-3}, ""),
            (IMUConfig, {"scale_factor_acc": -2.0}, ""),
            (GNSSConfig, {"noise_m": -1.0}, ""),
            (RadioConfig, {"path_loss_exponent": 1.5}, ""),  # must be >= 2.0
        ],
        ids=[
            "cam_bad_rate",
            "cam_bad_resolution",
            "event_zero_threshold",
            "lidar_bad_vfov",
            "lidar_zero_channels",
            "thermal_bad_temp_range",
            "imu_neg_noise_density",
            "imu_bad_scale_factor",
            "gnss_neg_noise",
            "radio_low_path_exponent",
        ],
    )
    def test_invalid_config_raises(
        self,
        config_cls: type,
        invalid_kwargs: dict,
        match: str,
    ) -> None:
        """Constructing a config with invalid values must raise a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            config_cls(**invalid_kwargs)

    @pytest.mark.parametrize(
        "config_cls,valid_kwargs",
        [
            (CameraConfig, {"update_rate_hz": 60.0, "resolution": (1280, 720)}),
            (LidarConfig, {"n_channels": 64, "max_range_m": 200.0}),
            (IMUConfig, {"update_rate_hz": 400.0, "add_gravity": False}),
            (GNSSConfig, {"noise_m": 0.01, "bias_sigma_m": 0.0}),
            (ThermalCameraConfig, {"bit_depth": 16, "noise_sigma": 0.1}),
            (RadioConfig, {"tx_power_dbm": 30.0, "los_required": True}),
        ],
        ids=["cam", "lidar", "imu", "gnss", "thermal", "radio"],
    )
    def test_valid_config_constructs(self, config_cls: type, valid_kwargs: dict) -> None:
        """Valid kwargs must not raise."""
        cfg = config_cls(**valid_kwargs)
        for key, val in valid_kwargs.items():
            assert getattr(cfg, key) == val

    @pytest.mark.parametrize(
        "config_cls,sensor_cls",
        [
            (CameraConfig, CameraModel),
            (LidarConfig, LidarModel),
            (IMUConfig, IMUModel),
            (GNSSConfig, GNSSModel),
            (ThermalCameraConfig, ThermalCameraModel),
            (RadioConfig, RadioLinkModel),
            (EventCameraConfig, EventCameraModel),
        ],
        ids=["camera", "lidar", "imu", "gnss", "thermal", "radio", "event"],
    )
    def test_from_config_and_get_config_are_inverses(self, config_cls: type, sensor_cls: type) -> None:
        """from_config(get_config()) must recreate an equivalent config."""
        cfg_original = config_cls()
        sensor = sensor_cls.from_config(cfg_original)
        cfg_recovered = sensor.get_config()
        # update_rate_hz must be preserved exactly.
        assert cfg_recovered.update_rate_hz == pytest.approx(cfg_original.update_rate_hz)
        assert cfg_recovered.name == cfg_original.name
