"""
Shared type aliases, TypedDicts, and numpy array type helpers for
the Genesis external sensor realism layer.

These types serve two purposes:

1. **Static type checking** — IDEs and ``mypy`` can infer precise
   types for array shapes and observation dict keys.
2. **Documentation** — TypedDicts make the expected shape of every
   sensor's state input and observation output explicit.

All TypedDicts use ``total=False`` for *state* dicts (callers may omit
any field) and ``total=True`` (the default) for *observation* dicts
(every field is always present in a successful update).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Generic NDArray aliases
# ---------------------------------------------------------------------------

#: A float32 NumPy array of arbitrary shape.
FloatArray = npt.NDArray[np.float32]
#: A float64 NumPy array of arbitrary shape.
Float64Array = npt.NDArray[np.float64]
#: A uint8 NumPy array of arbitrary shape.
UInt8Array = npt.NDArray[np.uint8]
#: A uint16 NumPy array of arbitrary shape.
UInt16Array = npt.NDArray[np.uint16]
#: An int32 NumPy array of arbitrary shape.
Int32Array = npt.NDArray[np.int32]
#: Re-export numpy's ArrayLike for convenience.
ArrayLike = npt.ArrayLike
#: A jammer zone is a (centre_xyz, radius_m) pair.
JammerZone = tuple[ArrayLike, float]

# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------


class Polarity(IntEnum):
    """
    Event polarity for a Dynamic Vision Sensor (DVS).

    ``POSITIVE`` (+1) indicates a log-intensity increase, ``NEGATIVE``
    (-1) indicates a decrease.  The ``IntEnum`` base ensures arithmetic
    with plain integers still works (``Polarity.POSITIVE + 0 == 1``).
    """

    NEGATIVE = -1
    POSITIVE = 1


# ---------------------------------------------------------------------------
# Shared sensor input state
# ---------------------------------------------------------------------------


class SensorState(TypedDict, total=False):
    """
    Combined ideal-state dict consumed by the sensor layer.

    All fields are optional (``total=False``); individual sensors only
    read the keys they need.  Building this dict from Genesis outputs::

        state: SensorState = {
            "rgb":   cam.render(rgb=True),
            "depth": cam.render(depth=True),
            "seg":   cam.render(segmentation=True),
            "normal": cam.render(normal=True),
            "gray":  gray_from_rgb(rgb),
            "pose":  drone.get_pos_quat(),
            "pos":   drone.get_pos().numpy(),
            "vel":   drone.get_vel().numpy(),
            "ang_vel": drone.get_ang_vel().numpy(),
            "range_image": raycaster.read().numpy(),
            "intensity_image": intensity.numpy(),
            "temperature_map": {e.id: e.temp_c for e in scene.entities},
            "obstruction": sky_obstruction_fraction,
            "weather": {"rain_rate_mm_h": 5.0},
        }
    """

    # Visual — shape annotations are in comments; TypedDict fields must be
    # plain types, so we use the closest NDArray alias available.
    rgb: UInt8Array  # shape (H, W, 3) — may also be float32 [0, 1]
    depth: FloatArray  # shape (H, W), metres
    seg: Int32Array  # shape (H, W), entity IDs
    normal: FloatArray  # shape (H, W, 3)
    gray: FloatArray  # shape (H, W), [0, 1]

    # Pose / velocity
    pose: tuple[FloatArray, FloatArray]  # (pos (3,), quat (4,))
    pos: Float64Array  # shape (3,) metres ENU
    vel: Float64Array  # shape (3,) m/s ENU
    ang_vel: FloatArray  # shape (3,) rad/s

    # IMU
    lin_acc: Float64Array  # shape (3,) body-frame linear acceleration (m/s²), no gravity
    gravity_body: Float64Array  # shape (3,) gravity vector in body frame (m/s²)

    # LiDAR
    range_image: FloatArray  # shape (n_channels, h_resolution), metres
    intensity_image: FloatArray  # shape (n_channels, h_resolution), [0, 1]

    # Thermal
    temperature_map: dict[int, float]  # entity_id → temperature °C

    # Environment
    obstruction: float  # 0–1 sky-hemisphere obstruction fraction
    weather: dict[str, float]  # e.g. {"rain_rate_mm_h": 5.0}


# ---------------------------------------------------------------------------
# IMU model
# ---------------------------------------------------------------------------


class ImuObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.IMUModel`."""

    lin_acc: Float64Array  # shape (3,) — noisy specific force (m/s²); includes gravity when add_gravity=True
    ang_vel: Float64Array  # shape (3,) — noisy angular velocity (rad/s)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


class CameraObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.CameraModel`."""

    rgb: UInt8Array  # shape (H, W, 3)


# ---------------------------------------------------------------------------
# Event camera model
# ---------------------------------------------------------------------------


class EventCameraObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.EventCameraModel`."""

    events: list[Any]  # list[Event]; forward-ref avoids circular import


# ---------------------------------------------------------------------------
# Thermal camera model
# ---------------------------------------------------------------------------


class ThermalObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.ThermalCameraModel`."""

    thermal: UInt8Array | UInt16Array  # shape (H, W); dtype depends on bit_depth
    temperature_c: FloatArray  # shape (H, W) — pre-quantisation temperature


# ---------------------------------------------------------------------------
# LiDAR model
# ---------------------------------------------------------------------------


class LidarObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.LidarModel`."""

    points: FloatArray  # shape (N, 4) — x, y, z, intensity
    range_image: FloatArray  # shape (n_channels, h_resolution) — processed ranges


# ---------------------------------------------------------------------------
# GNSS model
# ---------------------------------------------------------------------------


class GnssObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.GNSSModel`."""

    pos: Float64Array  # shape (3,) — noisy world-frame position (m)
    vel: Float64Array  # shape (3,) — noisy world-frame velocity (m/s)
    pos_llh: Float64Array  # shape (3,) — latitude (deg), longitude (deg), altitude (m)
    fix_quality: int  # GnssFixQuality value
    n_satellites: int
    hdop: float


# ---------------------------------------------------------------------------
# Radio link model
# ---------------------------------------------------------------------------


class RadioObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.RadioLinkModel`."""

    delivered: list[Any]  # list[ScheduledPacket]; forward-ref avoids circular import
    queue_depth: int


__all__ = [
    # Array type aliases
    "ArrayLike",
    "Float64Array",
    "FloatArray",
    "Int32Array",
    "JammerZone",
    "Polarity",
    "UInt16Array",
    "UInt8Array",
    # Input state
    "SensorState",
    # Observation TypedDicts
    "CameraObservation",
    "EventCameraObservation",
    "GnssObservation",
    "ImuObservation",
    "LidarObservation",
    "RadioObservation",
    "ThermalObservation",
]
