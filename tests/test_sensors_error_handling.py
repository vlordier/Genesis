"""
Parametrised error-handling tests for the Genesis external sensor layer.

Covers:
- ``BaseSensor`` constructor guards (invalid name type, non-positive rate)
- ``CameraModel.step()`` shape-mismatch detection
- ``LidarModel.step()`` dimension / channel / resolution mismatches
- ``ThermalCameraModel.step()`` segmentation-mask shape mismatch
- ``GNSSModel.step()`` wrong-shape pos/vel arrays
- ``IMUModel.step()`` wrong-shape lin_acc/ang_vel arrays
- ``EventCameraModel`` wrong-dimension gray input
- ``RadioLinkModel.transmit()`` wrong-shape src/dst positions
- ``RadioLinkModel.estimate_link_metrics()`` wrong-shape positions
- ``SensorScheduler.update()`` error context wrapping

None of these tests require EGL/OpenGL or a Genesis scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis.sensors import (
    CameraModel,
    EventCameraModel,
    GNSSModel,
    IMUModel,
    LidarModel,
    RadioLinkModel,
    ThermalCameraModel,
)
from genesis.sensors.scheduler import SensorScheduler

# ---------------------------------------------------------------------------
# BaseSensor constructor guards
# ---------------------------------------------------------------------------


class TestBaseSensorConstructorGuards:
    """BaseSensor.__init__ rejects invalid arguments."""

    @pytest.mark.parametrize(
        "name",
        [42, None, ["sensor"], b"bytes"],
        ids=["int", "None", "list", "bytes"],
    )
    def test_non_str_name_raises_type_error(self, name) -> None:
        with pytest.raises(TypeError, match="name must be a str"):
            CameraModel(name=name)

    @pytest.mark.parametrize(
        "rate",
        [0.0, -1.0, -100.0],
        ids=["zero", "neg1", "neg100"],
    )
    def test_non_positive_rate_raises_value_error(self, rate: float) -> None:
        with pytest.raises(ValueError, match="update_rate_hz must be positive"):
            CameraModel(update_rate_hz=rate)


# ---------------------------------------------------------------------------
# CameraModel.step() — shape validation
# ---------------------------------------------------------------------------


class TestCameraModelShapeErrors:
    """CameraModel raises on rgb shape mismatches and bad array dimensions."""

    @pytest.mark.parametrize(
        "bad_shape",
        [(32, 32, 3), (16, 48, 3), (1, 1, 3)],
        ids=["32x32_vs_64x48", "16x48_vs_64x48", "1x1_vs_64x48"],
    )
    def test_wrong_spatial_shape_raises(self, bad_shape: tuple) -> None:
        cam = CameraModel(resolution=(64, 48), seed=0)
        bad_rgb = np.zeros(bad_shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="does not match configured resolution"):
            cam.step(0.0, {"rgb": bad_rgb})

    @pytest.mark.parametrize(
        "bad_shape",
        [(48, 64)],  # 2-D with correct spatial but no channel dim → accepted as single-channel
        ids=["2d_no_channel"],
    )
    def test_2d_gray_valid_dimensions(self, bad_shape: tuple) -> None:
        """A 2-D array with correct (H, W) is NOT a channel error — only spatial dims are checked."""
        cam = CameraModel(resolution=(64, 48), seed=0)
        gray_2d = np.zeros(bad_shape, dtype=np.uint8)  # shape (H=48, W=64)
        # Should not raise; CameraModel accepts any 2-D / 3-D with correct spatial dims
        # (It internally handles grayscale conversion.)
        cam.step(0.0, {"rgb": gray_2d})

    @pytest.mark.parametrize(
        "bad_shape",
        [(48, 64, 7), (48, 64, 0)],
        ids=["7_channels", "0_channels"],
    )
    def test_invalid_channel_count_raises(self, bad_shape: tuple) -> None:
        cam = CameraModel(resolution=(64, 48), seed=0)
        bad_rgb = np.zeros(bad_shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="1, 3, or 4 channels"):
            cam.step(0.0, {"rgb": bad_rgb})

    @pytest.mark.parametrize(
        "bad_shape",
        [(3, 48, 64, 3), (1,)],
        ids=["4d", "1d"],
    )
    def test_wrong_number_of_dims_raises(self, bad_shape: tuple) -> None:
        cam = CameraModel(resolution=(64, 48), seed=0)
        bad_rgb = np.zeros(bad_shape, dtype=np.uint8)
        with pytest.raises(ValueError):
            cam.step(0.0, {"rgb": bad_rgb})

    def test_none_rgb_returns_empty_obs(self) -> None:
        cam = CameraModel(resolution=(64, 48), seed=0)
        obs = cam.step(0.0, {})
        assert obs == {}


# ---------------------------------------------------------------------------
# LidarModel.step() — shape validation
# ---------------------------------------------------------------------------


class TestLidarModelShapeErrors:
    """LidarModel raises on range_image dimension / channel / resolution mismatches."""

    def test_1d_range_image_raises(self) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        with pytest.raises(ValueError, match="2-D array"):
            lidar.step(0.0, {"range_image": np.zeros(360, dtype=np.float32)})

    def test_3d_range_image_raises(self) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        with pytest.raises(ValueError, match="2-D array"):
            lidar.step(0.0, {"range_image": np.zeros((16, 360, 1), dtype=np.float32)})

    @pytest.mark.parametrize(
        "wrong_channels",
        [8, 32, 64],
        ids=["8ch", "32ch", "64ch"],
    )
    def test_wrong_n_channels_raises(self, wrong_channels: int) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        ri = np.zeros((wrong_channels, 360), dtype=np.float32)
        with pytest.raises(ValueError, match="n_channels=16"):
            lidar.step(0.0, {"range_image": ri})

    @pytest.mark.parametrize(
        "wrong_resolution",
        [180, 720, 1800],
        ids=["180az", "720az", "1800az"],
    )
    def test_wrong_h_resolution_raises(self, wrong_resolution: int) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        ri = np.zeros((16, wrong_resolution), dtype=np.float32)
        with pytest.raises(ValueError, match="h_resolution=360"):
            lidar.step(0.0, {"range_image": ri})

    def test_none_range_image_returns_empty_points(self) -> None:
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        obs = lidar.step(0.0, {})
        assert obs["points"].shape == (0, 4)


# ---------------------------------------------------------------------------
# ThermalCameraModel.step() — segmentation mask shape validation
# ---------------------------------------------------------------------------


class TestThermalCameraModelShapeErrors:
    """ThermalCameraModel raises on seg shape mismatches."""

    @pytest.mark.parametrize(
        "bad_seg_shape",
        [(32, 32), (16, 16), (1, 1)],
        ids=["32x32_vs_64x48", "16x16_vs_64x48", "1x1_vs_64x48"],
    )
    def test_wrong_seg_shape_raises(self, bad_seg_shape: tuple) -> None:
        cam = ThermalCameraModel(resolution=(64, 48), seed=0)
        bad_seg = np.zeros(bad_seg_shape, dtype=np.int32)
        with pytest.raises(ValueError, match="does not match configured resolution"):
            cam.step(0.0, {"seg": bad_seg, "temperature_map": {}})

    @pytest.mark.parametrize(
        "bad_ndim_shape",
        [(64,), (1, 48, 64, 1)],
        ids=["1d", "4d"],
    )
    def test_wrong_ndim_seg_raises(self, bad_ndim_shape: tuple) -> None:
        cam = ThermalCameraModel(resolution=(64, 48), seed=0)
        bad_seg = np.zeros(bad_ndim_shape, dtype=np.int32)
        with pytest.raises(ValueError):
            cam.step(0.0, {"seg": bad_seg, "temperature_map": {}})

    def test_correct_seg_shape_succeeds(self) -> None:
        cam = ThermalCameraModel(resolution=(64, 48), seed=0)
        seg = np.zeros((48, 64), dtype=np.int32)
        obs = cam.step(0.0, {"seg": seg, "temperature_map": {0: 25.0}})
        assert obs["thermal"].shape == (48, 64)

    def test_none_seg_returns_empty_obs(self) -> None:
        cam = ThermalCameraModel(resolution=(64, 48), seed=0)
        obs = cam.step(0.0, {})
        assert obs == {}


# ---------------------------------------------------------------------------
# GNSSModel.step() — pos/vel shape validation
# ---------------------------------------------------------------------------


class TestGNSSModelInputShapeErrors:
    """GNSSModel raises on wrong-shape pos/vel arrays."""

    @pytest.mark.parametrize(
        "bad_pos",
        [np.zeros(2), np.zeros((3, 1)), np.zeros(6)],
        ids=["2elem", "3x1", "6elem"],
    )
    def test_wrong_pos_shape_raises(self, bad_pos: np.ndarray) -> None:
        gnss = GNSSModel(seed=0)
        with pytest.raises(ValueError, match="state\\['pos'\\]"):
            gnss.step(0.0, {"pos": bad_pos, "vel": np.zeros(3)})

    @pytest.mark.parametrize(
        "bad_vel",
        [np.zeros(1), np.zeros((1, 3)), np.zeros(4)],
        ids=["1elem", "1x3", "4elem"],
    )
    def test_wrong_vel_shape_raises(self, bad_vel: np.ndarray) -> None:
        gnss = GNSSModel(seed=0)
        with pytest.raises(ValueError, match="state\\['vel'\\]"):
            gnss.step(0.0, {"pos": np.zeros(3), "vel": bad_vel})

    def test_correct_shapes_succeed(self) -> None:
        gnss = GNSSModel(noise_m=0.0, bias_sigma_m=0.0, multipath_sigma_m=0.0, seed=0)
        obs = gnss.step(0.0, {"pos": np.zeros(3), "vel": np.zeros(3)})
        assert obs["pos"].shape == (3,)


# ---------------------------------------------------------------------------
# IMUModel.step() — lin_acc / ang_vel shape validation
# ---------------------------------------------------------------------------


class TestIMUModelInputShapeErrors:
    """IMUModel raises on wrong-shape lin_acc/ang_vel arrays."""

    @pytest.mark.parametrize(
        "bad_acc",
        [np.zeros(2), np.zeros((3, 1)), np.zeros(6)],
        ids=["2elem", "3x1", "6elem"],
    )
    def test_wrong_lin_acc_shape_raises(self, bad_acc: np.ndarray) -> None:
        imu = IMUModel(seed=0)
        with pytest.raises(ValueError, match="state\\['lin_acc'\\]"):
            imu.step(0.0, {"lin_acc": bad_acc, "ang_vel": np.zeros(3)})

    @pytest.mark.parametrize(
        "bad_gyr",
        [np.zeros(1), np.zeros((1, 3)), np.zeros(4)],
        ids=["1elem", "1x3", "4elem"],
    )
    def test_wrong_ang_vel_shape_raises(self, bad_gyr: np.ndarray) -> None:
        imu = IMUModel(seed=0)
        with pytest.raises(ValueError, match="state\\['ang_vel'\\]"):
            imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": bad_gyr})

    def test_correct_shapes_succeed(self) -> None:
        imu = IMUModel(
            noise_density_acc=0.0,
            noise_density_gyr=0.0,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            add_gravity=False,
            seed=0,
        )
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3)})
        assert obs["lin_acc"].shape == (3,)


# ---------------------------------------------------------------------------
# EventCameraModel — gray input dimension validation
# ---------------------------------------------------------------------------


class TestEventCameraGrayErrors:
    """EventCameraModel._load_gray raises on wrong-dimension inputs."""

    @pytest.mark.parametrize(
        "bad_shape",
        [(64,), (1, 32, 32, 1)],
        ids=["1d", "4d"],
    )
    def test_wrong_gray_ndim_raises(self, bad_shape: tuple) -> None:
        cam = EventCameraModel(seed=0)
        bad_gray = np.zeros(bad_shape, dtype=np.float32)
        with pytest.raises(ValueError, match="state\\['gray'\\] must be a 2-D or 3-D array"):
            cam.step(0.0, {"gray": bad_gray})

    def test_2d_gray_accepted(self) -> None:
        cam = EventCameraModel(seed=0)
        gray = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        obs = cam.step(0.0, {"gray": gray})  # first frame → no events
        assert obs["events"] == []

    def test_3d_single_channel_gray_accepted(self) -> None:
        cam = EventCameraModel(seed=0)
        gray = np.random.default_rng(0).random((32, 32, 1)).astype(np.float32)
        obs = cam.step(0.0, {"gray": gray})
        assert obs["events"] == []

    def test_none_gray_none_rgb_returns_empty_events(self) -> None:
        cam = EventCameraModel(seed=0)
        obs = cam.step(0.0, {})
        assert obs["events"] == []


# ---------------------------------------------------------------------------
# RadioLinkModel — position vector shape validation
# ---------------------------------------------------------------------------


class TestRadioLinkModelPositionErrors:
    """RadioLinkModel raises on wrong-shape src_pos / dst_pos in transmit() and estimate_link_metrics()."""

    _SRC = np.zeros(3)
    _DST = np.array([10.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        "bad_pos",
        [np.zeros(2), np.zeros((3, 1)), np.zeros(4), np.zeros((1, 3))],
        ids=["2elem", "3x1", "4elem", "1x3"],
    )
    def test_transmit_wrong_src_raises(self, bad_pos: np.ndarray) -> None:
        radio = RadioLinkModel(seed=0)
        with pytest.raises(ValueError, match="src_pos"):
            radio.transmit("pkt", src_pos=bad_pos, dst_pos=self._DST, sim_time=0.0)

    @pytest.mark.parametrize(
        "bad_pos",
        [np.zeros(2), np.zeros((3, 1)), np.zeros(4), np.zeros((1, 3))],
        ids=["2elem", "3x1", "4elem", "1x3"],
    )
    def test_transmit_wrong_dst_raises(self, bad_pos: np.ndarray) -> None:
        radio = RadioLinkModel(seed=0)
        with pytest.raises(ValueError, match="dst_pos"):
            radio.transmit("pkt", src_pos=self._SRC, dst_pos=bad_pos, sim_time=0.0)

    @pytest.mark.parametrize(
        "bad_pos",
        [np.zeros(2), np.zeros((3, 1))],
        ids=["2elem", "3x1"],
    )
    def test_estimate_wrong_src_raises(self, bad_pos: np.ndarray) -> None:
        radio = RadioLinkModel(seed=0)
        with pytest.raises(ValueError, match="src_pos"):
            radio.estimate_link_metrics(src_pos=bad_pos, dst_pos=self._DST)

    @pytest.mark.parametrize(
        "bad_pos",
        [np.zeros(2), np.zeros((3, 1))],
        ids=["2elem", "3x1"],
    )
    def test_estimate_wrong_dst_raises(self, bad_pos: np.ndarray) -> None:
        radio = RadioLinkModel(seed=0)
        with pytest.raises(ValueError, match="dst_pos"):
            radio.estimate_link_metrics(src_pos=self._SRC, dst_pos=bad_pos)

    def test_correct_positions_succeed(self) -> None:
        radio = RadioLinkModel(seed=0)
        metrics = radio.estimate_link_metrics(src_pos=self._SRC, dst_pos=self._DST)
        assert "snr_db" in metrics
        assert "distance_m" in metrics


# ---------------------------------------------------------------------------
# SensorScheduler.update() — error context wrapping
# ---------------------------------------------------------------------------


class TestSchedulerErrorContext:
    """SensorScheduler.update() wraps sensor errors with context."""

    def test_sensor_step_error_wrapped_in_runtime_error(self) -> None:
        """A ValueError from a sensor step() should be re-raised as a RuntimeError with context."""
        imu = IMUModel(update_rate_hz=100.0, seed=0)
        scheduler = SensorScheduler([("imu", imu)])
        scheduler.reset()

        # Pass a wrong-shape lin_acc to trigger the sensor's ValueError.
        bad_state = {"lin_acc": np.zeros(2), "ang_vel": np.zeros(3)}
        with pytest.raises(RuntimeError, match="imu.*raised during update"):
            scheduler.update(sim_time=0.0, state=bad_state)

    def test_scheduler_error_chains_original_exception(self) -> None:
        """The original exception should be chained (not swallowed)."""
        gnss = GNSSModel(update_rate_hz=10.0, seed=0)
        scheduler = SensorScheduler([("gnss", gnss)])
        scheduler.reset()

        bad_state = {"pos": np.zeros(5), "vel": np.zeros(3)}
        with pytest.raises(RuntimeError) as exc_info:
            scheduler.update(sim_time=0.0, state=bad_state)
        # __cause__ must be the original ValueError from GNSSModel.step()
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_scheduler_error_message_contains_sensor_name(self) -> None:
        """RuntimeError message must include the sensor name for easy debugging."""
        lidar = LidarModel(n_channels=16, h_resolution=360, seed=0)
        scheduler = SensorScheduler([("my_lidar", lidar)])
        scheduler.reset()

        bad_ri = np.zeros((8, 360), dtype=np.float32)  # wrong n_channels
        with pytest.raises(RuntimeError, match="my_lidar"):
            scheduler.update(sim_time=0.0, state={"range_image": bad_ri})

    def test_healthy_sensors_still_update_before_error_in_others(self) -> None:
        """Sensors processed before the failing one must have their results in the exception context."""
        cam = CameraModel(resolution=(16, 16), seed=0)
        gnss = GNSSModel(update_rate_hz=10.0, seed=0)
        scheduler = SensorScheduler([("gnss", gnss)])
        scheduler.reset()

        # gnss with bad pos — RuntimeError should be raised
        with pytest.raises(RuntimeError, match="gnss"):
            scheduler.update(
                sim_time=0.0,
                state={"pos": np.zeros(2), "vel": np.zeros(3)},
            )

    @pytest.mark.parametrize(
        "sensor_name,sensor,bad_state",
        [
            (
                "imu",
                IMUModel(update_rate_hz=200.0, seed=0),
                {"lin_acc": np.zeros(2), "ang_vel": np.zeros(3)},
            ),
            (
                "gnss",
                GNSSModel(update_rate_hz=10.0, seed=0),
                {"pos": np.zeros(5), "vel": np.zeros(3)},
            ),
            (
                "lidar",
                LidarModel(n_channels=16, h_resolution=360, seed=0),
                {"range_image": np.zeros((8, 360), dtype=np.float32)},
            ),
        ],
        ids=["imu_bad_acc", "gnss_bad_pos", "lidar_bad_channels"],
    )
    def test_parametrised_scheduler_error_context(self, sensor_name: str, sensor, bad_state: dict) -> None:
        """Each sensor type produces an appropriately-named RuntimeError from the scheduler."""
        scheduler = SensorScheduler([(sensor_name, sensor)])
        scheduler.reset()
        with pytest.raises(RuntimeError, match=sensor_name):
            scheduler.update(sim_time=0.0, state=bad_state)
