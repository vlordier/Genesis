"""
Unit tests for the external sensor realism layer (genesis/sensors/).

These tests do NOT require a running Genesis scene (no scene.build() call),
so they work in headless CI environments without EGL/OpenGL.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis.sensors import (
    BaseSensor,
    CameraModel,
    Event,
    EventCameraModel,
    GNSSModel,
    LidarModel,
    RadioLinkModel,
    ScheduledPacket,
    SensorScheduler,
    SensorSuite,
    ThermalCameraModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 RGB image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _make_gray(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.random((h, w)).astype(np.float32)


def _make_seg(h: int = 64, w: int = 64, n_entities: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, n_entities, (h, w), dtype=np.int32)


def _make_range_image(n_ch: int = 16, n_az: int = 360) -> np.ndarray:
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
    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            _DummySensor(name="bad", update_rate_hz=-1.0)

    def test_is_due_initial(self):
        s = _DummySensor(name="s", update_rate_hz=10.0)
        assert s.is_due(0.0)

    def test_is_due_after_step(self):
        s = _DummySensor(name="s", update_rate_hz=10.0)
        s.step(0.0, {})
        # Not due immediately after update
        assert not s.is_due(0.0)
        # Due after one period
        assert s.is_due(0.1)

    def test_reset_clears_state(self):
        s = _DummySensor(name="s", update_rate_hz=10.0)
        s.step(1.0, {})
        s.reset()
        assert s.is_due(0.0)

    def test_repr(self):
        s = _DummySensor(name="my_sensor", update_rate_hz=5.0)
        assert "my_sensor" in repr(s)
        assert "5.0" in repr(s)


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


class TestCameraModel:
    def test_basic_step(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(64, 64))
        rgb = _make_rgb(64, 64)
        obs = cam.step(0.0, {"rgb": rgb})
        assert "rgb" in obs
        assert obs["rgb"].dtype == np.uint8
        assert obs["rgb"].shape == (64, 64, 3)

    def test_noise_changes_output(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(64, 64), iso=1600.0)
        rgb = _make_rgb(64, 64)
        obs = cam.step(0.0, {"rgb": rgb})
        # Noise should change the image (with near-certainty)
        assert not np.array_equal(obs["rgb"], rgb)

    def test_empty_state(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0)
        obs = cam.step(0.0, {})
        assert obs == {}

    def test_reset(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0)
        cam.step(0.0, {"rgb": _make_rgb()})
        cam.reset()
        assert cam.is_due(0.0)

    def test_float_input(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(32, 32))
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        obs = cam.step(0.0, {"rgb": rgb})
        assert obs["rgb"].dtype == np.uint8

    def test_get_observation(self):
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(32, 32))
        cam.step(0.0, {"rgb": _make_rgb(32, 32)})
        assert cam.get_observation() is not None


# ---------------------------------------------------------------------------
# EventCameraModel
# ---------------------------------------------------------------------------


class TestEventCameraModel:
    def test_no_events_on_first_frame(self):
        ecam = EventCameraModel(update_rate_hz=1000.0)
        obs = ecam.step(0.0, {"gray": _make_gray()})
        assert obs["events"] == []

    def test_events_on_change(self):
        ecam = EventCameraModel(threshold_pos=0.1, threshold_neg=0.1)
        gray1 = np.zeros((32, 32), dtype=np.float32) + 0.5
        gray2 = np.zeros((32, 32), dtype=np.float32) + 0.9  # big jump
        ecam.step(0.0, {"gray": gray1})
        obs = ecam.step(0.001, {"gray": gray2})
        assert len(obs["events"]) > 0

    def test_events_have_correct_type(self):
        ecam = EventCameraModel(threshold_pos=0.1, threshold_neg=0.1)
        gray1 = np.zeros((16, 16), dtype=np.float32) + 0.3
        gray2 = np.zeros((16, 16), dtype=np.float32) + 0.9
        ecam.step(0.0, {"gray": gray1})
        obs = ecam.step(0.001, {"gray": gray2})
        for e in obs["events"]:
            assert isinstance(e, Event)
            assert e.polarity in (-1, 1)

    def test_from_rgb_input(self):
        ecam = EventCameraModel(threshold_pos=0.05, threshold_neg=0.05)
        rgb1 = np.full((16, 16, 3), 100, dtype=np.uint8)
        rgb2 = np.full((16, 16, 3), 200, dtype=np.uint8)
        ecam.step(0.0, {"rgb": rgb1})
        obs = ecam.step(0.001, {"rgb": rgb2})
        assert "events" in obs

    def test_refractory_period(self):
        ecam = EventCameraModel(threshold_pos=0.05, refractory_period_s=1.0)
        gray1 = np.zeros((16, 16), dtype=np.float32) + 0.3
        gray2 = np.zeros((16, 16), dtype=np.float32) + 0.9
        ecam.step(0.0, {"gray": gray1})
        obs1 = ecam.step(0.001, {"gray": gray2})
        # Fire again immediately: all pixels should be suppressed
        obs2 = ecam.step(0.002, {"gray": gray1})
        assert len(obs2["events"]) <= len(obs1["events"])

    def test_background_activity(self):
        ecam = EventCameraModel(background_activity_rate_hz=1e6)
        gray = np.zeros((32, 32), dtype=np.float32) + 0.5
        ecam.step(0.0, {"gray": gray})
        obs = ecam.step(0.001, {"gray": gray})
        # With very high BA rate, expect many noise events
        assert len(obs["events"]) > 0

    def test_reset(self):
        ecam = EventCameraModel()
        ecam.step(0.0, {"gray": _make_gray()})
        ecam.reset()
        assert ecam._prev_log is None


# ---------------------------------------------------------------------------
# ThermalCameraModel
# ---------------------------------------------------------------------------


class TestThermalCameraModel:
    def test_basic_step(self):
        tcam = ThermalCameraModel(update_rate_hz=9.0, resolution=(32, 32))
        seg = _make_seg(32, 32, n_entities=3)
        temp_map = {0: 20.0, 1: 50.0, 2: 80.0}
        obs = tcam.step(0.0, {"seg": seg, "temperature_map": temp_map})
        assert "thermal" in obs
        assert "temperature_c" in obs
        assert obs["thermal"].shape == (32, 32)

    def test_no_seg(self):
        tcam = ThermalCameraModel()
        obs = tcam.step(0.0, {})
        assert obs == {}

    def test_temperature_range(self):
        tcam = ThermalCameraModel(resolution=(8, 8), bit_depth=8)
        seg = np.zeros((8, 8), dtype=np.int32)
        obs = tcam.step(0.0, {"seg": seg, "temperature_map": {0: 50.0}})
        assert obs["thermal"].dtype == np.uint8

    def test_fog_attenuation(self):
        tcam = ThermalCameraModel(resolution=(8, 8), fog_density=0.1)
        seg = np.zeros((8, 8), dtype=np.int32)
        depth = np.full((8, 8), 20.0, dtype=np.float32)
        obs_no_fog = ThermalCameraModel(resolution=(8, 8), fog_density=0.0).step(
            0.0, {"seg": seg, "temperature_map": {0: 100.0}}
        )
        obs_fog = tcam.step(0.0, {"seg": seg, "temperature_map": {0: 100.0}, "depth": depth})
        # Fog should reduce apparent temperature
        assert obs_fog["temperature_c"].mean() < obs_no_fog["temperature_c"].mean()

    def test_reset(self):
        tcam = ThermalCameraModel()
        tcam.step(0.0, {"seg": _make_seg(), "temperature_map": {}})
        tcam.reset()
        assert tcam.is_due(0.0)


# ---------------------------------------------------------------------------
# LidarModel
# ---------------------------------------------------------------------------


class TestLidarModel:
    def test_basic_step(self):
        lidar = LidarModel(n_channels=16, h_resolution=360, update_rate_hz=10.0)
        ri = _make_range_image(16, 360)
        obs = lidar.step(0.0, {"range_image": ri})
        assert "points" in obs
        assert obs["points"].ndim == 2
        assert obs["points"].shape[1] == 4  # x, y, z, intensity

    def test_no_range_image(self):
        lidar = LidarModel()
        obs = lidar.step(0.0, {})
        assert obs["points"].shape == (0, 4)

    def test_dropout(self):
        lidar = LidarModel(n_channels=8, h_resolution=90, dropout_prob=0.5)
        ri = _make_range_image(8, 90)
        obs_full = LidarModel(n_channels=8, h_resolution=90).step(0.0, {"range_image": ri})
        obs_drop = lidar.step(0.0, {"range_image": ri})
        assert obs_drop["points"].shape[0] <= obs_full["points"].shape[0]

    def test_max_range_clip(self):
        lidar = LidarModel(n_channels=4, h_resolution=8, max_range_m=10.0)
        ri = np.full((4, 8), 200.0, dtype=np.float32)  # all beyond max range
        obs = lidar.step(0.0, {"range_image": ri})
        assert obs["points"].shape[0] == 0

    def test_reset(self):
        lidar = LidarModel()
        lidar.step(0.0, {"range_image": _make_range_image()})
        lidar.reset()
        assert lidar.is_due(0.0)


# ---------------------------------------------------------------------------
# GNSSModel
# ---------------------------------------------------------------------------


class TestGNSSModel:
    def test_basic_step(self):
        gnss = GNSSModel(update_rate_hz=10.0)
        obs = gnss.step(0.0, {"pos": np.array([10.0, 20.0, 5.0]), "vel": np.array([1.0, 0.0, 0.0])})
        assert "pos" in obs
        assert "pos_llh" in obs
        assert "fix_quality" in obs
        assert obs["pos"].shape == (3,)

    def test_noise_varies(self):
        gnss = GNSSModel(noise_m=5.0, update_rate_hz=10.0)
        true_pos = np.array([0.0, 0.0, 50.0])
        obs1 = gnss.step(0.0, {"pos": true_pos, "vel": np.zeros(3)})
        obs2 = gnss.step(0.1, {"pos": true_pos, "vel": np.zeros(3)})
        # Two consecutive measurements should differ due to noise
        assert not np.allclose(obs1["pos"], obs2["pos"])

    def test_jammer_zone(self):
        centre = np.array([0.0, 0.0, 0.0])
        gnss = GNSSModel(jammer_zones=[(centre, 100.0)], update_rate_hz=10.0)
        obs = gnss.step(0.0, {"pos": np.array([1.0, 1.0, 1.0]), "vel": np.zeros(3)})
        assert obs["fix_quality"] == GNSSModel.FIX_NO_FIX

    def test_reset(self):
        gnss = GNSSModel()
        gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        gnss.reset()
        assert np.allclose(gnss._bias, 0.0)


# ---------------------------------------------------------------------------
# RadioLinkModel
# ---------------------------------------------------------------------------


class TestRadioLinkModel:
    def test_transmit_and_receive(self):
        radio = RadioLinkModel(update_rate_hz=100.0, base_latency_s=0.01, jitter_sigma_s=0.0)
        src = np.array([0.0, 0.0, 10.0])
        dst = np.array([5.0, 5.0, 10.0])
        radio.transmit({"cmd": "hover"}, src, dst, sim_time=0.0, has_los=True)
        # Before delivery time
        obs = radio.step(0.0, {})
        assert len(obs["delivered"]) == 0
        # After delivery time
        obs = radio.step(1.0, {})
        assert len(obs["delivered"]) == 1

    def test_long_range_drops(self):
        radio = RadioLinkModel(tx_power_dbm=5.0, min_snr_db=20.0, snr_transition_db=1.0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([0.0, 0.0, 5000.0])  # 5 km distance
        results = [radio.transmit({}, src, dst, sim_time=float(i)) for i in range(20)]
        n_dropped = sum(1 for r in results if r is None)
        assert n_dropped > 0

    def test_estimate_link_metrics(self):
        radio = RadioLinkModel()
        metrics = radio.estimate_link_metrics(np.zeros(3), np.array([100.0, 0.0, 0.0]))
        assert "snr_db" in metrics
        assert "packet_error_rate" in metrics
        assert 0.0 <= metrics["packet_error_rate"] <= 1.0

    def test_reset_clears_queue(self):
        radio = RadioLinkModel()
        radio.transmit({}, np.zeros(3), np.ones(3), sim_time=0.0)
        radio.reset()
        assert radio._queue == []

    def test_scheduled_packet(self):
        radio = RadioLinkModel(base_latency_s=0.05, jitter_sigma_s=0.0)
        src = np.array([0.0, 0.0, 0.0])
        dst = np.array([1.0, 0.0, 0.0])
        pkt = radio.transmit({"id": 99}, src, dst, sim_time=0.0)
        if pkt is not None:
            assert isinstance(pkt, ScheduledPacket)
            assert pkt.delivery_time >= 0.05


# ---------------------------------------------------------------------------
# SensorScheduler
# ---------------------------------------------------------------------------


class TestSensorScheduler:
    def test_add_and_update(self):
        s = SensorScheduler()
        cam = CameraModel(name="rgb", update_rate_hz=30.0, resolution=(16, 16))
        s.add(cam, name="rgb")
        rgb = _make_rgb(16, 16)
        obs = s.update(0.0, {"rgb": rgb})
        assert "rgb" in obs

    def test_duplicate_name_raises(self):
        s = SensorScheduler()
        s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")
        with pytest.raises(ValueError):
            s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")

    def test_remove(self):
        s = SensorScheduler()
        s.add(_DummySensor(name="s", update_rate_hz=10.0), name="s")
        s.remove("s")
        assert "s" not in s

    def test_rate_scheduling(self):
        slow = _DummySensor(name="slow", update_rate_hz=1.0)
        fast = _DummySensor(name="fast", update_rate_hz=100.0)
        sched = SensorScheduler([("slow", slow), ("fast", fast)])
        sched.update(0.0, {})
        # At t=0.005, fast is due (>1/100) but slow is not (< 1/1)
        obs = sched.update(0.005, {})
        assert obs["fast"] is not None

    def test_reset_all(self):
        s = SensorScheduler()
        s.add(_DummySensor(name="d", update_rate_hz=10.0))
        s.update(0.0, {})
        s.reset()

    def test_repr(self):
        s = SensorScheduler()
        s.add(_DummySensor(name="d", update_rate_hz=5.0))
        assert "5.0Hz" in repr(s)


# ---------------------------------------------------------------------------
# SensorSuite
# ---------------------------------------------------------------------------


class TestSensorSuite:
    def test_default_factory(self):
        suite = SensorSuite.default()
        assert "rgb" in suite.sensor_names()
        assert "gnss" in suite.sensor_names()
        assert "lidar" in suite.sensor_names()

    def test_step_returns_all_sensors(self):
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

    def test_reset(self):
        suite = SensorSuite.default()
        suite.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        suite.reset()  # should not raise

    def test_disable_sensors(self):
        suite = SensorSuite.default(rgb_rate_hz=0, lidar_rate_hz=0)
        assert "rgb" not in suite.sensor_names()
        assert "lidar" not in suite.sensor_names()

    def test_extra_sensors(self):
        extra = _DummySensor(name="custom", update_rate_hz=5.0)
        suite = SensorSuite(extra_sensors=[("custom", extra)])
        obs = suite.step(0.0, {})
        assert "custom" in obs

    def test_get_sensor(self):
        suite = SensorSuite.default()
        gnss = suite.get_sensor("gnss")
        assert isinstance(gnss, GNSSModel)

    def test_repr(self):
        suite = SensorSuite.default()
        r = repr(suite)
        assert "SensorSuite" in r
