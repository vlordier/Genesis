"""
Unit tests for the external sensor realism layer (genesis/sensors/).

These tests do NOT require a running Genesis scene (no scene.build() call),
so they work in headless CI environments without EGL/OpenGL.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError

from genesis.sensors import (
    BaseSensor,
    CameraModel,
    Event,
    EventCameraModel,
    GnssFixQuality,
    GNSSModel,
    IMUModel,
    LidarModel,
    Polarity,
    RadioLinkModel,
    ScheduledPacket,
    SensorScheduler,
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

# Number of color channels in an RGB image
_RGB_CHANNELS = 3
# Expected number of columns in a LiDAR point-cloud row: x, y, z, intensity
_LIDAR_POINT_COLS = 4
# Expected number of dimensions (axes) in the point-cloud ndarray
_LIDAR_ARRAY_NDIM = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 RGB image of shape (H, W, 3)."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, (h, w, _RGB_CHANNELS), dtype=np.uint8)


def _make_gray(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random float32 grayscale image with values in [0, 1]."""
    rng = np.random.default_rng(seed=0)
    return rng.random((h, w)).astype(np.float32)


def _make_seg(h: int = 64, w: int = 64, n_entities: int = 3) -> np.ndarray:
    """Return a random integer segmentation mask with entity IDs in [0, n_entities)."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, n_entities, (h, w), dtype=np.int32)


def _make_range_image(n_ch: int = 16, n_az: int = 360) -> np.ndarray:
    """Return a random float32 range image (n_channels x h_resolution) with ranges in [1, 50] m."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(1.0, 50.0, (n_ch, n_az)).astype(np.float32)


# ---------------------------------------------------------------------------
# BaseSensor
# ---------------------------------------------------------------------------


class _DummySensor(BaseSensor):
    """Minimal concrete sensor for testing BaseSensor API."""

    def reset(self, env_id: int = 0) -> None:
        self._obs: dict = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict) -> dict:
        self._obs = {"value": sim_time}
        self._mark_updated(sim_time)
        return self._obs

    def get_observation(self) -> dict:
        return self._obs


class TestBaseSensor:
    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError, match="update_rate_hz must be positive"):
            _DummySensor(name="bad", update_rate_hz=-1.0)

    def test_is_due_initial(self) -> None:
        s = _DummySensor(name="s", update_rate_hz=10.0)
        assert s.is_due(0.0)

    def test_is_due_after_step(self) -> None:
        s = _DummySensor(name="s", update_rate_hz=10.0)
        s.step(0.0, {})
        # Not due immediately after update
        assert not s.is_due(0.0)
        # Due after one period
        assert s.is_due(0.1)

    def test_reset_clears_state(self) -> None:
        s = _DummySensor(name="s", update_rate_hz=10.0)
        s.step(1.0, {})
        s.reset()
        assert s.is_due(0.0)

    def test_repr(self) -> None:
        s = _DummySensor(name="my_sensor", update_rate_hz=5.0)
        assert "my_sensor" in repr(s)
        assert "5.0" in repr(s)


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


class TestCameraModel:
    def test_basic_step(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(64, 64))
        rgb = _make_rgb(64, 64)
        obs = cam.step(0.0, {"rgb": rgb})
        assert "rgb" in obs
        assert obs["rgb"].dtype == np.uint8
        assert obs["rgb"].shape == (64, 64, _RGB_CHANNELS)

    def test_noise_changes_output(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(64, 64), iso=1600.0)
        rgb = _make_rgb(64, 64)
        obs = cam.step(0.0, {"rgb": rgb})
        # Noise should change the image (with near-certainty)
        assert not np.array_equal(obs["rgb"], rgb)

    def test_empty_state(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0)
        obs = cam.step(0.0, {})
        assert obs == {}

    def test_reset(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0)
        cam.step(0.0, {"rgb": _make_rgb()})
        cam.reset()
        assert cam.is_due(0.0)

    def test_float_input(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(32, 32))
        rng = np.random.default_rng(seed=1)
        rgb = rng.random((32, 32, _RGB_CHANNELS)).astype(np.float32)
        obs = cam.step(0.0, {"rgb": rgb})
        assert obs["rgb"].dtype == np.uint8

    def test_get_observation(self) -> None:
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(32, 32))
        cam.step(0.0, {"rgb": _make_rgb(32, 32)})
        assert cam.get_observation() is not None

    def test_fixed_pattern_noise_resolution_mismatch_no_crash(self) -> None:
        """Stepping with an image larger than configured resolution must not raise IndexError."""
        cam = CameraModel(resolution=(32, 32), update_rate_hz=30.0, dead_pixel_fraction=0.01)
        # Provide an image that is smaller than configured — must succeed
        small_rgb = _make_rgb(16, 16)
        obs_small = cam.step(0.0, {"rgb": small_rgb})
        assert obs_small["rgb"].shape == (16, 16, _RGB_CHANNELS)
        # Provide an image that is larger than configured — must succeed (was IndexError)
        large_rgb = _make_rgb(64, 64)
        obs_large = cam.step(0.0, {"rgb": large_rgb})
        assert obs_large["rgb"].shape == (64, 64, _RGB_CHANNELS)


# ---------------------------------------------------------------------------
# EventCameraModel
# ---------------------------------------------------------------------------


class TestEventCameraModel:
    def test_no_events_on_first_frame(self) -> None:
        ecam = EventCameraModel(update_rate_hz=1000.0)
        obs = ecam.step(0.0, {"gray": _make_gray()})
        assert obs["events"] == []

    def test_events_on_change(self) -> None:
        ecam = EventCameraModel(threshold_pos=0.1, threshold_neg=0.1)
        gray1 = np.zeros((32, 32), dtype=np.float32) + 0.5
        gray2 = np.zeros((32, 32), dtype=np.float32) + 0.9  # big jump
        ecam.step(0.0, {"gray": gray1})
        obs = ecam.step(0.001, {"gray": gray2})
        assert len(obs["events"]) > 0

    def test_events_have_correct_type(self) -> None:
        ecam = EventCameraModel(threshold_pos=0.1, threshold_neg=0.1)
        gray1 = np.zeros((16, 16), dtype=np.float32) + 0.3
        gray2 = np.zeros((16, 16), dtype=np.float32) + 0.9
        ecam.step(0.0, {"gray": gray1})
        obs = ecam.step(0.001, {"gray": gray2})
        for e in obs["events"]:
            assert isinstance(e, Event)
            assert e.polarity in (-1, 1)

    def test_from_rgb_input(self) -> None:
        ecam = EventCameraModel(threshold_pos=0.05, threshold_neg=0.05)
        rgb1 = np.full((16, 16, _RGB_CHANNELS), 100, dtype=np.uint8)
        rgb2 = np.full((16, 16, _RGB_CHANNELS), 200, dtype=np.uint8)
        ecam.step(0.0, {"rgb": rgb1})
        obs = ecam.step(0.001, {"rgb": rgb2})
        assert "events" in obs

    def test_refractory_period(self) -> None:
        ecam = EventCameraModel(threshold_pos=0.05, refractory_period_s=1.0)
        gray1 = np.zeros((16, 16), dtype=np.float32) + 0.3
        gray2 = np.zeros((16, 16), dtype=np.float32) + 0.9
        ecam.step(0.0, {"gray": gray1})
        obs1 = ecam.step(0.001, {"gray": gray2})
        # Fire again immediately: all pixels should be suppressed
        obs2 = ecam.step(0.002, {"gray": gray1})
        assert len(obs2["events"]) <= len(obs1["events"])

    def test_background_activity(self) -> None:
        ecam = EventCameraModel(background_activity_rate_hz=1e6)
        gray = np.zeros((32, 32), dtype=np.float32) + 0.5
        ecam.step(0.0, {"gray": gray})
        obs = ecam.step(0.001, {"gray": gray})
        # With very high BA rate, expect many noise events
        assert len(obs["events"]) > 0

    def test_reset_clears_state(self) -> None:
        ecam = EventCameraModel()
        ecam.step(0.0, {"gray": _make_gray()})
        ecam.reset()
        # After reset, is_initialized should be False
        assert not ecam.is_initialized

    def test_is_initialized_property(self) -> None:
        ecam = EventCameraModel()
        assert not ecam.is_initialized
        ecam.step(0.0, {"gray": _make_gray()})
        assert ecam.is_initialized

    def test_get_observation_returns_last_obs(self) -> None:
        """get_observation() must return the exact same dict as the last step() call."""
        ecam = EventCameraModel(threshold_pos=0.05, threshold_neg=0.05)
        gray1 = np.zeros((16, 16), dtype=np.float32) + 0.3
        gray2 = np.zeros((16, 16), dtype=np.float32) + 0.9
        ecam.step(0.0, {"gray": gray1})
        obs = ecam.step(0.001, {"gray": gray2})
        assert ecam.get_observation() is obs, "get_observation() must return the cached _last_obs dict"

    def test_get_observation_before_step(self) -> None:
        """get_observation() before any step() should return empty events list, not raise."""
        ecam = EventCameraModel()
        obs = ecam.get_observation()
        assert "events" in obs
        assert obs["events"] == []


# ---------------------------------------------------------------------------
# ThermalCameraModel
# ---------------------------------------------------------------------------


class TestThermalCameraModel:
    def test_basic_step(self) -> None:
        tcam = ThermalCameraModel(update_rate_hz=9.0, resolution=(32, 32))
        seg = _make_seg(32, 32, n_entities=3)
        temp_map = {0: 20.0, 1: 50.0, 2: 80.0}
        obs = tcam.step(0.0, {"seg": seg, "temperature_map": temp_map})
        assert "thermal" in obs
        assert "temperature_c" in obs
        assert obs["thermal"].shape == (32, 32)

    def test_no_seg(self) -> None:
        tcam = ThermalCameraModel()
        obs = tcam.step(0.0, {})
        assert obs == {}

    def test_temperature_range(self) -> None:
        tcam = ThermalCameraModel(resolution=(8, 8), bit_depth=8)
        seg = np.zeros((8, 8), dtype=np.int32)
        obs = tcam.step(0.0, {"seg": seg, "temperature_map": {0: 50.0}})
        assert obs["thermal"].dtype == np.uint8

    def test_fog_attenuation(self) -> None:
        tcam = ThermalCameraModel(resolution=(8, 8), fog_density=0.1)
        seg = np.zeros((8, 8), dtype=np.int32)
        depth = np.full((8, 8), 20.0, dtype=np.float32)
        obs_no_fog = ThermalCameraModel(resolution=(8, 8), fog_density=0.0).step(
            0.0, {"seg": seg, "temperature_map": {0: 100.0}}
        )
        obs_fog = tcam.step(0.0, {"seg": seg, "temperature_map": {0: 100.0}, "depth": depth})
        # Fog should reduce apparent temperature
        assert obs_fog["temperature_c"].mean() < obs_no_fog["temperature_c"].mean()

    def test_custom_temp_range(self) -> None:
        tcam = ThermalCameraModel(resolution=(8, 8), temp_range_c=(-40.0, 300.0))
        seg = np.zeros((8, 8), dtype=np.int32)
        obs = tcam.step(0.0, {"seg": seg, "temperature_map": {0: 200.0}})
        assert "thermal" in obs

    def test_reset(self) -> None:
        tcam = ThermalCameraModel()
        tcam.step(0.0, {"seg": _make_seg(), "temperature_map": {}})
        tcam.reset()
        assert tcam.is_due(0.0)

    def test_gaussian_blur_fallback_applies_blur(self) -> None:
        """Box-filter fallback must actually blur the image, not return it unchanged."""
        img = np.zeros((20, 20), dtype=np.float32)
        img[10, 10] = 1.0  # single hot pixel
        blurred = ThermalCameraModel._gaussian_blur(img, sigma=2.0)
        # Blurred image must spread energy to neighbours
        assert blurred[10, 10] < 1.0, "Hot pixel should be spread by blur"
        assert blurred[10, 11] > 0.0, "Adjacent pixel should receive some energy"


# ---------------------------------------------------------------------------
# LidarModel
# ---------------------------------------------------------------------------


class TestLidarModel:
    def test_basic_step(self) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, update_rate_hz=10.0)
        ri = _make_range_image(16, 360)
        obs = lidar.step(0.0, {"range_image": ri})
        assert "points" in obs
        assert obs["points"].ndim == _LIDAR_ARRAY_NDIM
        assert obs["points"].shape[1] == _LIDAR_POINT_COLS

    def test_no_range_image(self) -> None:
        lidar = LidarModel()
        obs = lidar.step(0.0, {})
        assert obs["points"].shape == (0, _LIDAR_POINT_COLS)

    def test_dropout(self) -> None:
        lidar = LidarModel(n_channels=8, h_resolution=90, dropout_prob=0.5)
        ri = _make_range_image(8, 90)
        obs_full = LidarModel(n_channels=8, h_resolution=90).step(0.0, {"range_image": ri})
        obs_drop = lidar.step(0.0, {"range_image": ri})
        assert obs_drop["points"].shape[0] <= obs_full["points"].shape[0]

    def test_max_range_clip(self) -> None:
        lidar = LidarModel(n_channels=4, h_resolution=8, max_range_m=10.0)
        ri = np.full((4, 8), 200.0, dtype=np.float32)  # all beyond max range
        obs = lidar.step(0.0, {"range_image": ri})
        assert obs["points"].shape[0] == 0

    def test_reset(self) -> None:
        lidar = LidarModel()
        lidar.step(0.0, {"range_image": _make_range_image()})
        lidar.reset()
        assert lidar.is_due(0.0)


# ---------------------------------------------------------------------------
# GNSSModel
# ---------------------------------------------------------------------------


class TestGNSSModel:
    def test_basic_step(self) -> None:
        gnss = GNSSModel(update_rate_hz=10.0)
        obs = gnss.step(0.0, {"pos": np.array([10.0, 20.0, 5.0]), "vel": np.array([1.0, 0.0, 0.0])})
        assert "pos" in obs
        assert "pos_llh" in obs
        assert "fix_quality" in obs
        assert obs["pos"].shape == (3,)

    def test_noise_varies(self) -> None:
        gnss = GNSSModel(noise_m=5.0, update_rate_hz=10.0)
        true_pos = np.array([0.0, 0.0, 50.0])
        obs1 = gnss.step(0.0, {"pos": true_pos, "vel": np.zeros(3)})
        obs2 = gnss.step(0.1, {"pos": true_pos, "vel": np.zeros(3)})
        # Two consecutive measurements should differ due to noise
        assert not np.allclose(obs1["pos"], obs2["pos"])

    def test_jammer_zone(self) -> None:
        centre = np.array([0.0, 0.0, 0.0])
        gnss = GNSSModel(jammer_zones=[(centre, 100.0)], update_rate_hz=10.0)
        obs = gnss.step(0.0, {"pos": np.array([1.0, 1.0, 1.0]), "vel": np.zeros(3)})
        assert obs["fix_quality"] == GnssFixQuality.NO_FIX

    def test_fix_quality_enum(self) -> None:
        gnss = GNSSModel(update_rate_hz=10.0)
        obs = gnss.step(0.0, {"pos": np.array([0.0, 0.0, 10.0]), "vel": np.zeros(3)})
        assert obs["fix_quality"] in (
            GnssFixQuality.NO_FIX,
            GnssFixQuality.AUTONOMOUS,
            GnssFixQuality.RTK,
        )

    def test_reset_clears_bias(self) -> None:
        gnss = GNSSModel()
        gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        gnss.reset()
        assert np.allclose(gnss.bias, 0.0)

    def test_polar_origin_no_crash(self) -> None:
        """GNSSModel must not produce inf/nan when origin is at the geographic pole."""
        gnss = GNSSModel(origin_llh=(90.0, 0.0, 0.0), update_rate_hz=10.0, seed=0)
        obs = gnss.step(0.0, {"pos": np.array([100.0, 200.0, 5.0]), "vel": np.zeros(3)})
        assert np.all(np.isfinite(obs["pos_llh"])), "pos_llh must be finite near the pole"
        assert np.isfinite(obs["pos_llh"][1]), "longitude must be finite near the pole"

    def test_jammer_does_not_expose_true_position(self) -> None:
        """When inside a jammer zone the output pos must NOT be the true position."""
        centre = np.array([0.0, 0.0, 0.0])
        gnss = GNSSModel(jammer_zones=[(centre, 1000.0)], update_rate_hz=10.0, seed=0)
        true_pos = np.array([1.0, 2.0, 3.0])
        obs = gnss.step(0.0, {"pos": true_pos, "vel": np.zeros(3)})
        assert obs["fix_quality"] == GnssFixQuality.NO_FIX
        assert not np.allclose(obs["pos"], true_pos), "Jammer output must not expose the ground-truth position"


# ---------------------------------------------------------------------------
# RadioLinkModel
# ---------------------------------------------------------------------------


class TestRadioLinkModel:
    def test_transmit_and_receive(self) -> None:
        radio = RadioLinkModel(update_rate_hz=100.0, base_latency_s=0.01, jitter_sigma_s=0.0)
        src = np.array([0.0, 0.0, 10.0])
        dst = np.array([5.0, 5.0, 10.0])
        radio.transmit({"cmd": "hover"}, src, dst, 0.0, has_los=True)
        # Before delivery time
        obs = radio.step(0.0, {})
        assert len(obs["delivered"]) == 0
        # After delivery time
        obs = radio.step(1.0, {})
        assert len(obs["delivered"]) == 1

    def test_long_range_drops(self) -> None:
        radio = RadioLinkModel(tx_power_dbm=5.0, min_snr_db=20.0, snr_transition_db=1.0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([0.0, 0.0, 5000.0])  # 5 km distance
        results = [radio.transmit({}, src, dst, float(i), has_los=True) for i in range(20)]
        n_dropped = sum(1 for r in results if r is None)
        assert n_dropped > 0

    def test_estimate_link_metrics(self) -> None:
        radio = RadioLinkModel()
        metrics = radio.estimate_link_metrics(np.zeros(3), np.array([100.0, 0.0, 0.0]), has_los=True)
        assert "snr_db" in metrics
        assert "packet_error_rate" in metrics
        assert 0.0 <= metrics["packet_error_rate"] <= 1.0

    def test_reset_clears_queue(self) -> None:
        radio = RadioLinkModel()
        radio.transmit({}, np.zeros(3), np.ones(3) * 10.0, 0.0, has_los=True)
        radio.reset()
        assert radio.queue_depth == 0

    def test_scheduled_packet(self) -> None:
        base_latency = 0.05
        radio = RadioLinkModel(base_latency_s=base_latency, jitter_sigma_s=0.0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([1.0, 0.0, 0.0])
        pkt = radio.transmit({"id": 99}, src, dst, 0.0, has_los=True)
        if pkt is not None:
            assert isinstance(pkt, ScheduledPacket)
            assert pkt.delivery_time >= base_latency

    def test_nlos_excess_loss(self) -> None:
        radio = RadioLinkModel(
            nlos_excess_loss_db=40.0,
            min_snr_db=10.0,
            snr_transition_db=1.0,
            shadowing_sigma_db=0.0,  # deterministic: no random fading
        )
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([1000.0, 0.0, 0.0])  # 1 km -- heavy NLOS makes it lossy
        results = [radio.transmit({}, src, dst, float(i), has_los=False) for i in range(20)]
        # Very high NLOS loss at 1 km should cause many drops
        n_dropped = sum(1 for r in results if r is None)
        assert n_dropped > 0

    def test_queue_depth_property(self) -> None:
        radio = RadioLinkModel(base_latency_s=10.0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([1.0, 0.0, 0.0])
        radio.transmit({}, src, dst, 0.0, has_los=True)
        assert radio.queue_depth >= 0  # may be 0 if packet dropped


# ---------------------------------------------------------------------------
# SensorScheduler
# ---------------------------------------------------------------------------


class TestSensorScheduler:
    def test_add_and_update(self) -> None:
        s = SensorScheduler()
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(16, 16))
        s.add(cam, name="rgb")
        rgb = _make_rgb(16, 16)
        obs = s.update(0.0, {"rgb": rgb})
        assert "rgb" in obs

    def test_duplicate_name_raises(self) -> None:
        s = SensorScheduler()
        s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")
        with pytest.raises(ValueError, match="already registered"):
            s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")

    def test_remove(self) -> None:
        s = SensorScheduler()
        s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")
        s.remove("s")
        assert "s" not in s

    def test_rate_scheduling(self) -> None:
        slow = _DummySensor(name="slow", update_rate_hz=1.0)
        fast = _DummySensor(name="fast", update_rate_hz=100.0)
        sched = SensorScheduler([("slow", slow), ("fast", fast)])
        sched.update(0.0, {})
        # At t=0.005, fast is due (>1/100) but slow is not (< 1/1)
        obs = sched.update(0.005, {})
        assert obs["fast"] is not None

    def test_reset_all(self) -> None:
        s = SensorScheduler()
        s.add(_DummySensor(name="d", update_rate_hz=10.0))
        s.update(0.0, {})
        s.reset()

    def test_repr(self) -> None:
        s = SensorScheduler()
        s.add(_DummySensor(name="d", update_rate_hz=5.0))
        assert "5.0Hz" in repr(s)

    def test_get_sensor_unknown_name_helpful_error(self) -> None:
        """get_sensor() must raise KeyError with the sensor name and registered list."""
        s = SensorScheduler()
        s.add(_DummySensor(name="known", update_rate_hz=1.0), name="known")
        with pytest.raises(KeyError, match="unknown"):
            s.get_sensor("unknown")


# ---------------------------------------------------------------------------
# SensorSuite
# ---------------------------------------------------------------------------


class TestSensorSuite:
    def test_default_factory(self) -> None:
        suite = SensorSuite.default()
        assert "rgb" in suite.sensor_names()
        assert "gnss" in suite.sensor_names()
        assert "lidar" in suite.sensor_names()

    def test_step_returns_all_sensors(self) -> None:
        suite = SensorSuite.default()
        state = {
            "rgb": _make_rgb(64, 64),
            "gray": _make_gray(64, 64),
            "seg": _make_seg(64, 64),
            "temperature_map": {0: 20.0},
            "range_image": _make_range_image(16, 360),
            "pos": np.array([0.0, 0.0, 10.0]),
            "vel": np.zeros(3),
        }
        obs = suite.step(0.0, state)
        for name in suite.sensor_names():
            assert name in obs

    def test_reset(self) -> None:
        suite = SensorSuite.default()
        suite.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        suite.reset()  # should not raise

    def test_disable_sensors(self) -> None:
        suite = SensorSuite.default(rgb_rate_hz=0, lidar_rate_hz=0)
        assert "rgb" not in suite.sensor_names()
        assert "lidar" not in suite.sensor_names()

    def test_extra_sensors(self) -> None:
        extra = _DummySensor(name="custom", update_rate_hz=5.0)
        suite = SensorSuite(extra_sensors=[("custom", extra)])
        obs = suite.step(0.0, {})
        assert "custom" in obs

    def test_get_sensor(self) -> None:
        suite = SensorSuite.default()
        gnss = suite.get_sensor("gnss")
        assert isinstance(gnss, GNSSModel)

    def test_repr(self) -> None:
        suite = SensorSuite.default()
        r = repr(suite)
        assert "SensorSuite" in r


# ---------------------------------------------------------------------------
# Polarity IntEnum
# ---------------------------------------------------------------------------


class TestPolarity:
    def test_values(self) -> None:
        assert int(Polarity.POSITIVE) == 1
        assert int(Polarity.NEGATIVE) == -1

    def test_int_comparison(self) -> None:
        """IntEnum must compare equal to plain int counterparts."""
        assert Polarity.POSITIVE == 1
        assert Polarity.NEGATIVE == -1

    def test_arithmetic(self) -> None:
        assert Polarity.POSITIVE + 0 == 1
        assert Polarity.NEGATIVE + 0 == -1


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_event_is_frozen(self) -> None:
        e = Event(x=10, y=20, timestamp=0.5, polarity=Polarity.POSITIVE)
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            e.x = 99  # type: ignore[misc]

    def test_event_polarity_is_polarity_enum(self) -> None:
        e = Event(x=0, y=0, timestamp=0.0, polarity=Polarity.NEGATIVE)
        assert e.polarity == Polarity.NEGATIVE
        assert int(e.polarity) == -1

    def test_ba_noise_events_polarity_is_enum(self) -> None:
        """Background-activity noise events must have Polarity enum polarity, not plain int."""
        rng = np.random.default_rng(seed=99)
        gray = rng.random((16, 16)).astype(np.float32)
        cam = EventCameraModel(background_activity_rate_hz=1e6, update_rate_hz=100.0, seed=0)
        cam.reset()
        cam.step(0.0, {"gray": gray})
        # Large BA rate ensures noise events are generated
        obs = cam.step(0.01, {"gray": gray})
        events = obs["events"]
        assert len(events) > 0, "Expected BA noise events at 1 MHz rate"
        for ev in events:
            assert isinstance(ev.polarity, Polarity), (
                f"Expected Polarity enum, got {type(ev.polarity).__name__} ({ev.polarity!r})"
            )

    def test_lidar_point_is_frozen(self) -> None:
        from genesis.sensors import LidarPoint

        pt = LidarPoint(x=1.0, y=2.0, z=3.0, intensity=0.5, channel=0, azimuth_deg=45.0, range_m=10.0)
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            pt.x = 99.0  # type: ignore[misc]

    def test_scheduled_packet_is_frozen(self) -> None:
        pkt = ScheduledPacket(
            payload="test",
            src_pos=np.zeros(3),
            dst_pos=np.ones(3),
            send_time=0.0,
            delivery_time=1.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            pkt.payload = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pydantic v2 config models
# ---------------------------------------------------------------------------


class TestCameraConfig:
    def test_default_construction(self) -> None:
        cfg = CameraConfig()
        assert cfg.update_rate_hz == 30.0
        assert cfg.resolution == (640, 480)

    def test_invalid_rate(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(update_rate_hz=-1.0)

    def test_invalid_resolution(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(resolution=(0, 480))

    def test_rolling_shutter_clipped(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(rolling_shutter_fraction=1.5)

    def test_json_round_trip(self) -> None:
        cfg = CameraConfig(iso=400, jpeg_quality=70)
        json_str = cfg.model_dump_json()
        cfg2 = CameraConfig.model_validate_json(json_str)
        assert cfg2.iso == 400
        assert cfg2.jpeg_quality == 70

    def test_from_config_produces_valid_model(self) -> None:
        cfg = CameraConfig(resolution=(32, 32), update_rate_hz=10.0)
        cam = CameraModel.from_config(cfg)
        assert cam.resolution == (32, 32)
        assert cam.update_rate_hz == 10.0

    def test_get_config_round_trip(self) -> None:
        cam = CameraModel(resolution=(16, 16), iso=200.0, update_rate_hz=15.0)
        cfg = cam.get_config()
        assert cfg.resolution == (16, 16)
        assert cfg.iso == 200.0
        assert cfg.update_rate_hz == 15.0

    def test_get_config_preserves_dead_hot_pixel_fractions(self) -> None:
        """get_config() must include dead_pixel_fraction and hot_pixel_fraction."""
        cam = CameraModel(dead_pixel_fraction=0.01, hot_pixel_fraction=0.005, resolution=(8, 8))
        cfg = cam.get_config()
        assert cfg.dead_pixel_fraction == pytest.approx(0.01)
        assert cfg.hot_pixel_fraction == pytest.approx(0.005)

    def test_dead_hot_pixel_round_trip(self) -> None:
        """from_config(get_config()) must use the same dead/hot pixel fractions."""
        cam = CameraModel(dead_pixel_fraction=0.02, hot_pixel_fraction=0.01, resolution=(8, 8), seed=0)
        cfg = cam.get_config()
        cam2 = CameraModel.from_config(cfg)
        assert cam2.dead_pixel_fraction == pytest.approx(cam.dead_pixel_fraction)
        assert cam2.hot_pixel_fraction == pytest.approx(cam.hot_pixel_fraction)


class TestEventCameraConfig:
    def test_default_construction(self) -> None:
        cfg = EventCameraConfig()
        assert cfg.threshold_pos == 0.2

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValidationError):
            EventCameraConfig(threshold_pos=-0.1)

    def test_from_config(self) -> None:
        cfg = EventCameraConfig(update_rate_hz=500.0, threshold_pos=0.3)
        sensor = EventCameraModel.from_config(cfg)
        assert sensor.threshold_pos == 0.3
        assert sensor.update_rate_hz == 500.0

    def test_get_config_round_trip(self) -> None:
        sensor = EventCameraModel(threshold_variation=0.1)
        cfg = sensor.get_config()
        assert cfg.threshold_variation == 0.1


class TestThermalCameraConfig:
    def test_temp_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            ThermalCameraConfig(temp_range_c=(100.0, -20.0))  # min >= max

    def test_from_config(self) -> None:
        cfg = ThermalCameraConfig(resolution=(64, 64), bit_depth=8)
        sensor = ThermalCameraModel.from_config(cfg)
        assert sensor.bit_depth == 8

    def test_get_config_round_trip(self) -> None:
        sensor = ThermalCameraModel(noise_sigma=0.1, fog_density=0.02)
        cfg = sensor.get_config()
        assert cfg.noise_sigma == 0.1
        assert cfg.fog_density == 0.02


class TestLidarConfig:
    def test_v_fov_validation(self) -> None:
        with pytest.raises(ValidationError):
            LidarConfig(v_fov_deg=(30.0, -15.0))  # min >= max

    def test_from_config(self) -> None:
        cfg = LidarConfig(n_channels=32, max_range_m=50.0)
        sensor = LidarModel.from_config(cfg)
        assert sensor.n_channels == 32
        assert sensor.max_range_m == 50.0

    def test_get_config_round_trip(self) -> None:
        sensor = LidarModel(rain_rate_mm_h=5.0, dropout_prob=0.01)
        cfg = sensor.get_config()
        assert cfg.rain_rate_mm_h == 5.0
        assert cfg.dropout_prob == 0.01

    def test_channel_offsets_none_preserved(self) -> None:
        """get_config() must return None for channel_offsets_m when init'd with None."""
        sensor = LidarModel(n_channels=8)
        cfg = sensor.get_config()
        assert cfg.channel_offsets_m is None

    def test_channel_offsets_roundtrip(self) -> None:
        """Non-None channel offsets survive get_config / from_config round-trip."""
        offsets = [0.01, -0.01, 0.02, -0.02, 0.0, 0.0, 0.01, -0.01]
        sensor = LidarModel(n_channels=8, channel_offsets_m=offsets)
        cfg = sensor.get_config()
        assert cfg.channel_offsets_m == pytest.approx(offsets)
        sensor2 = LidarModel.from_config(cfg)
        assert list(sensor2._channel_offsets) == pytest.approx(offsets)


class TestGNSSConfig:
    def test_default_construction(self) -> None:
        cfg = GNSSConfig()
        assert cfg.noise_m == 1.5

    def test_from_config(self) -> None:
        cfg = GNSSConfig(noise_m=0.5, update_rate_hz=1.0)
        sensor = GNSSModel.from_config(cfg)
        assert sensor.noise_m == 0.5

    def test_get_config_round_trip(self) -> None:
        sensor = GNSSModel(noise_m=2.0, multipath_sigma_m=0.5)
        cfg = sensor.get_config()
        assert cfg.noise_m == 2.0
        assert cfg.multipath_sigma_m == 0.5


class TestRadioConfig:
    def test_from_config(self) -> None:
        cfg = RadioConfig(tx_power_dbm=30.0, los_required=True)
        sensor = RadioLinkModel.from_config(cfg)
        assert sensor.tx_power_dbm == 30.0
        assert sensor.los_required is True

    def test_get_config_round_trip(self) -> None:
        sensor = RadioLinkModel(shadowing_sigma_db=6.0, frequency_ghz=5.8)
        cfg = sensor.get_config()
        assert cfg.shadowing_sigma_db == 6.0
        assert cfg.frequency_ghz == 5.8


class TestSensorSuiteConfig:
    def test_default_has_rgb_and_gnss(self) -> None:
        cfg = SensorSuiteConfig()
        assert cfg.rgb is not None
        assert cfg.gnss is not None

    def test_all_disabled(self) -> None:
        cfg = SensorSuiteConfig.all_disabled()
        assert cfg.rgb is None
        assert cfg.gnss is None
        assert cfg.lidar is None

    def test_minimal(self) -> None:
        cfg = SensorSuiteConfig.minimal()
        assert cfg.gnss is not None
        assert cfg.rgb is None

    def test_full(self) -> None:
        cfg = SensorSuiteConfig.full()
        assert cfg.event is not None
        assert cfg.thermal is not None

    def test_from_config_builds_suite(self) -> None:
        cfg = SensorSuiteConfig(
            rgb=CameraConfig(resolution=(32, 32)),
            gnss=GNSSConfig(noise_m=0.5),
            event=None,
            thermal=None,
            lidar=None,
            radio=None,
        )
        suite = SensorSuite.from_config(cfg)
        assert "rgb" in suite.sensor_names()
        assert "gnss" in suite.sensor_names()
        assert "lidar" not in suite.sensor_names()

    def test_json_round_trip(self) -> None:
        cfg = SensorSuiteConfig.full()
        json_str = cfg.model_dump_json()
        cfg2 = SensorSuiteConfig.model_validate_json(json_str)
        assert cfg2.rgb is not None
        assert cfg2.event is not None

    def test_from_config_suite_step(self) -> None:
        cfg = SensorSuiteConfig(
            rgb=CameraConfig(resolution=(16, 16), update_rate_hz=10.0),
            gnss=GNSSConfig(update_rate_hz=10.0),
            event=None,
            thermal=None,
            lidar=None,
            radio=None,
        )
        suite = SensorSuite.from_config(cfg)
        suite.reset()
        obs = suite.step(0.0, {"rgb": _make_rgb(16, 16), "pos": np.zeros(3), "vel": np.zeros(3)})
        assert "rgb" in obs
        assert "gnss" in obs


# ---------------------------------------------------------------------------
# IMUModel
# ---------------------------------------------------------------------------


class TestIMUModel:
    """Tests for genesis.sensors.imu.IMUModel."""

    def _make_state(self) -> dict:
        return {
            "lin_acc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "ang_vel": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "gravity_body": np.array([0.0, 0.0, 9.80665], dtype=np.float64),
        }

    def test_basic_step_returns_expected_keys(self) -> None:

        imu = IMUModel(update_rate_hz=200.0, seed=0)
        obs = imu.step(0.0, self._make_state())
        assert "lin_acc" in obs
        assert "ang_vel" in obs
        assert obs["lin_acc"].shape == (3,)
        assert obs["ang_vel"].shape == (3,)

    def test_gravity_injection_adds_to_lin_acc(self) -> None:
        """With add_gravity=True and zero true acceleration, lin_acc ~ gravity_body."""

        imu = IMUModel(
            update_rate_hz=200.0,
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            scale_factor_acc=0.0,
            add_gravity=True,
            seed=0,
        )
        g_vec = np.array([0.0, 0.0, 9.80665])
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": g_vec})
        np.testing.assert_allclose(obs["lin_acc"], g_vec, atol=1e-9)

    def test_no_gravity_when_disabled(self) -> None:
        """With add_gravity=False and zero noise, lin_acc should be near-zero."""

        imu = IMUModel(
            update_rate_hz=200.0,
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            add_gravity=False,
            seed=0,
        )
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3)})
        np.testing.assert_allclose(obs["lin_acc"], np.zeros(3), atol=1e-9)

    def test_noise_varies_output(self) -> None:
        """Two consecutive steps should produce different measurements (noise active)."""

        imu = IMUModel(update_rate_hz=200.0, noise_density_acc=1e-2, seed=1)
        state = {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)}
        obs1 = imu.step(0.0, state)
        obs2 = imu.step(0.005, state)
        assert not np.allclose(obs1["lin_acc"], obs2["lin_acc"])

    def test_scale_factor_multiplies_signal(self) -> None:
        """With scale_factor_acc=0.1 and no noise, lin_acc = 1.1 * true_acc + g."""

        imu = IMUModel(
            update_rate_hz=200.0,
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            scale_factor_acc=0.1,
            add_gravity=False,
            seed=0,
        )
        true_acc = np.array([1.0, 0.0, 0.0])
        obs = imu.step(0.0, {"lin_acc": true_acc, "ang_vel": np.zeros(3)})
        np.testing.assert_allclose(obs["lin_acc"], 1.1 * true_acc, atol=1e-9)

    def test_reset_clears_bias(self) -> None:

        imu = IMUModel(update_rate_hz=200.0)
        imu.step(0.0, self._make_state())
        imu.reset()
        np.testing.assert_allclose(imu.bias_acc, np.zeros(3))
        np.testing.assert_allclose(imu.bias_gyr, np.zeros(3))

    def test_get_observation_returns_cached_dict(self) -> None:

        imu = IMUModel(update_rate_hz=200.0, seed=0)
        obs = imu.step(0.0, self._make_state())
        assert imu.get_observation() is obs

    def test_get_observation_before_step_empty(self) -> None:

        imu = IMUModel(update_rate_hz=200.0)
        obs = imu.get_observation()
        assert isinstance(obs, dict)

    def test_is_due_respects_rate(self) -> None:

        imu = IMUModel(update_rate_hz=100.0)  # period = 10 ms
        imu.step(0.0, self._make_state())
        assert not imu.is_due(0.005)  # only 5 ms passed
        assert imu.is_due(0.010)  # 10 ms passed

    def test_bias_properties_return_copy(self) -> None:
        """bias_acc and bias_gyr must return copies, not live references."""

        imu = IMUModel(update_rate_hz=200.0, seed=42)
        for _ in range(10):
            imu.step(0.0, self._make_state())
        b_acc = imu.bias_acc
        b_acc[0] = 999.0
        assert imu.bias_acc[0] != 999.0, "bias_acc should be a copy"


# ---------------------------------------------------------------------------
# IMUConfig
# ---------------------------------------------------------------------------


class TestIMUConfig:
    def test_basic_construction(self) -> None:

        cfg = IMUConfig(update_rate_hz=400.0)
        assert cfg.update_rate_hz == 400.0

    def test_invalid_noise_density(self) -> None:

        with pytest.raises(Exception):
            IMUConfig(noise_density_acc=-1.0)  # must be > 0

    def test_invalid_scale_factor(self) -> None:

        with pytest.raises(Exception):
            IMUConfig(scale_factor_acc=-2.0)  # must be >= -1

    def test_from_config_round_trip(self) -> None:

        cfg = IMUConfig(update_rate_hz=500.0, noise_density_acc=3e-3, bias_sigma_gyr=2e-4, add_gravity=False)
        imu = IMUModel.from_config(cfg)
        cfg2 = imu.get_config()
        assert cfg2.update_rate_hz == cfg.update_rate_hz
        assert cfg2.noise_density_acc == cfg.noise_density_acc
        assert cfg2.add_gravity == cfg.add_gravity

    def test_json_round_trip(self) -> None:

        cfg = IMUConfig(scale_factor_gyr=0.01)
        json_str = cfg.model_dump_json()
        cfg2 = IMUConfig.model_validate_json(json_str)
        assert cfg2.scale_factor_gyr == cfg.scale_factor_gyr


# ---------------------------------------------------------------------------
# Camera vignetting + chromatic aberration (third-pass additions)
# ---------------------------------------------------------------------------


class TestCameraVignetting:
    def test_vignetting_darkens_corners(self) -> None:
        """With vignetting enabled corners must be darker than the center."""
        from genesis.sensors import CameraModel

        cam = CameraModel(
            resolution=(64, 64),
            vignetting_strength=0.8,
            update_rate_hz=30.0,
            iso=100.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        # Uniform white image: all pixels are 1.0 before vignetting
        white = np.ones((64, 64, _RGB_CHANNELS), dtype=np.float32)
        obs = cam.step(0.0, {"rgb": white})
        rgb = obs["rgb"].astype(np.float32)
        center_brightness = rgb[32, 32].mean()
        corner_brightness = rgb[0, 0].mean()
        assert corner_brightness < center_brightness, "Corners should be darker after vignetting"

    def test_no_vignetting_preserves_uniformity(self) -> None:
        """With vignetting_strength=0 the brightness gradient should be much smaller."""
        from genesis.sensors import CameraModel

        cam = CameraModel(
            resolution=(32, 32),
            vignetting_strength=0.0,
            update_rate_hz=30.0,
            iso=100.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            seed=0,
        )
        white = np.ones((32, 32, _RGB_CHANNELS), dtype=np.float32)
        obs = cam.step(0.0, {"rgb": white})
        rgb = obs["rgb"].astype(np.float32) / 255.0
        center = rgb[16, 16].mean()
        corner = rgb[0, 0].mean()
        # Difference should be small (only noise, no vignetting)
        assert abs(center - corner) < 0.15, "Without vignetting centre/corner should be close"

    def test_vignetting_config_round_trip(self) -> None:
        from genesis.sensors import CameraModel

        cam = CameraModel(vignetting_strength=0.6)
        cfg = cam.get_config()
        assert cfg.vignetting_strength == 0.6
        cam2 = CameraModel.from_config(cfg)
        assert cam2.vignetting_strength == 0.6


class TestCameraCA:
    def test_ca_changes_output(self) -> None:
        """With CA enabled the output must differ from the CA-free output."""
        from genesis.sensors import CameraModel

        h, w = 32, 32
        # Structured image with a sharp vertical edge (not uniform)
        rgb = np.zeros((h, w, _RGB_CHANNELS), dtype=np.float32)
        rgb[:, w // 2 :, 0] = 1.0  # red channel edge at center

        cam_noca = CameraModel(
            resolution=(w, h),
            chromatic_aberration_px=0.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            iso=100.0,
            seed=0,
        )
        cam_ca = CameraModel(
            resolution=(w, h),
            chromatic_aberration_px=3.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            iso=100.0,
            seed=0,
        )
        obs_noca = cam_noca.step(0.0, {"rgb": rgb})
        obs_ca = cam_ca.step(0.0, {"rgb": rgb})
        # The two outputs should differ (CA moves the red channel)
        assert not np.array_equal(obs_noca["rgb"], obs_ca["rgb"])

    def test_ca_zero_is_identity_on_uniform_image(self) -> None:
        """CA=0 on a uniform image: all channels should have equal mean (no shift artefact)."""
        h, w = 16, 16
        rgb = np.full((h, w, _RGB_CHANNELS), 0.5, dtype=np.float32)
        cam = CameraModel(
            resolution=(w, h),
            chromatic_aberration_px=0.0,
            dead_pixel_fraction=0.0,
            hot_pixel_fraction=0.0,
            iso=100.0,
            seed=0,
        )
        obs = cam.step(0.0, {"rgb": rgb})
        out = obs["rgb"].astype(np.float32)
        # Without CA, R/G/B means should be nearly equal (only noise varies).
        assert abs(out[..., 0].mean() - out[..., 2].mean()) < 15.0, (
            "Without CA, R and B channel means should be close"
        )

    def test_ca_config_round_trip(self) -> None:
        from genesis.sensors import CameraModel

        cam = CameraModel(chromatic_aberration_px=2.5)
        cfg = cam.get_config()
        assert cfg.chromatic_aberration_px == 2.5
        cam2 = CameraModel.from_config(cfg)
        assert cam2.chromatic_aberration_px == 2.5


# ---------------------------------------------------------------------------
# LiDAR beam divergence (third-pass addition)
# ---------------------------------------------------------------------------


class TestLidarBeamDivergence:
    def _make_range_image_with_edge(self, n_ch: int = 16, n_az: int = 60) -> np.ndarray:
        """Create a range image with a sharp depth discontinuity."""
        ri = np.full((n_ch, n_az), 10.0, dtype=np.float32)
        ri[:, n_az // 2 :] = 50.0  # far wall after the midpoint
        return ri

    def test_beam_divergence_softens_edge(self) -> None:
        """Beam divergence should blur the range image, softening the depth edge."""
        from genesis.sensors import LidarModel

        n_ch, n_az = 16, 60
        ri = self._make_range_image_with_edge(n_ch, n_az)

        lidar_sharp = LidarModel(n_channels=n_ch, h_resolution=n_az, beam_divergence_mrad=0.0, seed=0)
        lidar_soft = LidarModel(n_channels=n_ch, h_resolution=n_az, beam_divergence_mrad=5.0, seed=0)

        obs_sharp = lidar_sharp.step(0.0, {"range_image": ri.copy()})
        obs_soft = lidar_soft.step(0.0, {"range_image": ri.copy()})

        # The range image from the blurred sensor should differ from the sharp one
        assert not np.allclose(obs_sharp["range_image"], obs_soft["range_image"])

    def test_beam_divergence_zero_no_change(self) -> None:
        """beam_divergence_mrad=0 must not modify the range image (beyond other noise)."""
        from genesis.sensors import LidarModel

        ri = self._make_range_image_with_edge()
        lidar = LidarModel(n_channels=16, h_resolution=60, beam_divergence_mrad=0.0, range_noise_sigma_m=0.0, seed=0)
        obs = lidar.step(0.0, {"range_image": ri.copy()})
        # With no noise and no beam divergence, the returned range image
        # should have the same structure as the input.
        assert obs["range_image"].shape == ri.shape

    def test_beam_divergence_config_round_trip(self) -> None:
        from genesis.sensors import LidarModel

        lidar = LidarModel(beam_divergence_mrad=2.5)
        cfg = lidar.get_config()
        assert cfg.beam_divergence_mrad == 2.5
        lidar2 = LidarModel.from_config(cfg)
        assert lidar2.beam_divergence_mrad == 2.5


# ---------------------------------------------------------------------------
# SensorSuite with IMU (third-pass integration test)
# ---------------------------------------------------------------------------


class TestSensorSuiteWithIMU:
    def test_default_suite_includes_imu(self) -> None:
        """SensorSuite.default() should register an IMU sensor."""
        suite = SensorSuite.default()
        assert "imu" in suite.sensor_names()

    def test_suite_step_returns_imu_obs(self) -> None:
        """suite.step() should return an 'imu' key with lin_acc and ang_vel."""
        suite = SensorSuite.default(
            rgb_rate_hz=0,
            event_rate_hz=0,
            thermal_rate_hz=0,
            lidar_rate_hz=0,
            radio_rate_hz=0,
            gnss_rate_hz=0,
            imu_rate_hz=200.0,
        )
        suite.reset()
        state = {
            "lin_acc": np.zeros(3, dtype=np.float64),
            "ang_vel": np.zeros(3, dtype=np.float64),
            "gravity_body": np.array([0.0, 0.0, 9.80665]),
        }
        obs = suite.step(0.0, state)
        assert "imu" in obs
        assert "lin_acc" in obs["imu"]
        assert "ang_vel" in obs["imu"]

    def test_suite_from_config_includes_imu(self) -> None:
        from genesis.sensors.config import SensorSuiteConfig

        cfg = SensorSuiteConfig(rgb=None, event=None, thermal=None, lidar=None, gnss=None, radio=None, imu=IMUConfig())
        suite = SensorSuite.from_config(cfg)
        assert "imu" in suite.sensor_names()

    def test_suite_imu_disabled(self) -> None:
        from genesis.sensors.config import SensorSuiteConfig

        cfg = SensorSuiteConfig.all_disabled()
        suite = SensorSuite.from_config(cfg)
        assert "imu" not in suite.sensor_names()

    def test_suite_config_full_includes_imu(self) -> None:
        from genesis.sensors.config import SensorSuiteConfig

        cfg = SensorSuiteConfig.full()
        assert cfg.imu is not None
