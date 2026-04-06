"""
External sensor realism layer for Genesis.

This package provides a multi-rate sensor stack that sits **on top** of
Genesis rather than inside it.  Genesis produces ideal, noise-free
measurements; this layer converts them into realistic device outputs.

Public API
----------
.. autosummary::

    BaseSensor
    CameraModel
    EventCameraModel
    ThermalCameraModel
    LidarModel
    GNSSModel
    RadioLinkModel
    SensorScheduler
    SensorSuite
    SensorSuiteConfig

Quick-start (keyword args)
--------------------------
::

    from genesis.sensors import SensorSuite

    suite = SensorSuite.default()
    suite.reset()
    obs = suite.step(scene.cur_t, state)

Quick-start (config-driven)
---------------------------
::

    from genesis.sensors import SensorSuite
    from genesis.sensors.config import CameraConfig, GNSSConfig, SensorSuiteConfig

    cfg = SensorSuiteConfig(
        rgb=CameraConfig(iso=800, jpeg_quality=60),
        gnss=GNSSConfig(noise_m=0.5),
        lidar=None,
    )
    suite = SensorSuite.from_config(cfg)
    print(cfg.model_dump_json(indent=2))  # serialise to JSON
"""

from .base import BaseSensor
from .camera_model import CameraModel
from .config import (
    CameraConfig,
    EventCameraConfig,
    GNSSConfig,
    LidarConfig,
    RadioConfig,
    SensorSuiteConfig,
    ThermalCameraConfig,
)
from .event_camera import Event, EventCameraModel
from .gnss import GNSSModel, GnssFixQuality
from .lidar import LidarModel, LidarPoint
from .radio import RadioLinkModel, ScheduledPacket
from .scheduler import SensorScheduler
from .suite import SensorSuite
from .thermal_camera import ThermalCameraModel
from .types import (
    ArrayLike,
    CameraObservation,
    EventCameraObservation,
    Float64Array,
    FloatArray,
    GnssObservation,
    Int32Array,
    JammerZone,
    LidarObservation,
    Polarity,
    RadioObservation,
    SensorState,
    ThermalObservation,
    UInt16Array,
    UInt8Array,
)

__all__ = [
    # Sensor classes
    "BaseSensor",
    "CameraModel",
    "Event",
    "EventCameraModel",
    "GNSSModel",
    "GnssFixQuality",
    "LidarModel",
    "LidarPoint",
    "RadioLinkModel",
    "ScheduledPacket",
    "SensorScheduler",
    "SensorSuite",
    "ThermalCameraModel",
    # Config classes
    "CameraConfig",
    "EventCameraConfig",
    "GNSSConfig",
    "LidarConfig",
    "RadioConfig",
    "SensorSuiteConfig",
    "ThermalCameraConfig",
    # Type aliases and TypedDicts
    "ArrayLike",
    "CameraObservation",
    "EventCameraObservation",
    "Float64Array",
    "FloatArray",
    "GnssObservation",
    "Int32Array",
    "JammerZone",
    "LidarObservation",
    "Polarity",
    "RadioObservation",
    "SensorState",
    "ThermalObservation",
    "UInt16Array",
    "UInt8Array",
]
