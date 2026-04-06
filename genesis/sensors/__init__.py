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

Quick-start
-----------
::

    from genesis.sensors import SensorSuite

    suite = SensorSuite.default()
    suite.reset()

    # inside the simulation loop
    obs = suite.step(scene.cur_t, state)
"""

from .base import BaseSensor
from .camera_model import CameraModel
from .event_camera import Event, EventCameraModel
from .gnss import GNSSModel
from .lidar import LidarModel, LidarPoint
from .radio import RadioLinkModel, ScheduledPacket
from .scheduler import SensorScheduler
from .suite import SensorSuite
from .thermal_camera import ThermalCameraModel

__all__ = [
    "BaseSensor",
    "CameraModel",
    "Event",
    "EventCameraModel",
    "GNSSModel",
    "LidarModel",
    "LidarPoint",
    "RadioLinkModel",
    "ScheduledPacket",
    "SensorScheduler",
    "SensorSuite",
    "ThermalCameraModel",
]
