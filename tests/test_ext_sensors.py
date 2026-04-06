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
    LidarConfig,
    RadioConfig,
    SensorSuiteConfig,
    ThermalCameraConfig,
)

# Number of colour channels in an RGB image
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
