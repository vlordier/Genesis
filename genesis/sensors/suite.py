"""
High-level ``SensorSuite`` wrapper.

Bundles all sensor models into a single object that mirrors the structure
described in the problem statement.  Internally it uses a
:class:`~genesis.sensors.scheduler.SensorScheduler` to drive sensors at
their individual rates.

The suite expects an *ideal state* dict that consumers build from Genesis
outputs at every simulation step::

    state = {
        # From scene / entity queries
        "pose":    drone.get_pos_quat(),        # (pos, quat)
        "vel":     drone.get_vel(),             # (3,)
        "ang_vel": drone.get_ang_vel(),         # (3,)

        # From cam.render(...)
        "rgb":     cam.render(rgb=True),
        "depth":   cam.render(depth=True),
        "seg":     cam.render(segmentation=True),
        "normal":  cam.render(normal=True),

        # Domain-specific
        "temperature_map": {entity_id: temp_c, ...},
        "obstruction":     0.3,   # 0–1 sky-obstruction fraction
        "weather":         {"rain_rate_mm_h": 5.0},
    }

    obs = suite.step(sim_time, state)
    print(obs["rgb"]["rgb"])          # corrupted RGB uint8
    print(obs["events"]["events"])    # list of Event objects
    print(obs["gnss"]["pos_llh"])     # lat, lon, alt

Example
-------
::

    import genesis as gs
    from genesis.sensors import SensorSuite

    suite = SensorSuite.default()
    suite.reset()

    # inside the simulation loop
    obs = suite.step(scene.cur_t, state)
"""

from __future__ import annotations

from typing import Any

from .base import BaseSensor
from .camera_model import CameraModel
from .event_camera import EventCameraModel
from .gnss import GNSSModel
from .lidar import LidarModel
from .radio import RadioLinkModel
from .scheduler import SensorScheduler
from .thermal_camera import ThermalCameraModel


class SensorSuite:
    """
    Convenience wrapper that instantiates and drives a full sensor stack.

    Parameters
    ----------
    rgb_camera:
        :class:`~genesis.sensors.camera_model.CameraModel` instance.
    event_camera:
        :class:`~genesis.sensors.event_camera.EventCameraModel` instance.
    thermal_camera:
        :class:`~genesis.sensors.thermal_camera.ThermalCameraModel` instance.
    lidar:
        :class:`~genesis.sensors.lidar.LidarModel` instance.
    gnss:
        :class:`~genesis.sensors.gnss.GNSSModel` instance.
    radio:
        :class:`~genesis.sensors.radio.RadioLinkModel` instance.
    extra_sensors:
        Additional ``(name, BaseSensor)`` pairs to register.
    """

    def __init__(
        self,
        rgb_camera: CameraModel | None = None,
        event_camera: EventCameraModel | None = None,
        thermal_camera: ThermalCameraModel | None = None,
        lidar: LidarModel | None = None,
        gnss: GNSSModel | None = None,
        radio: RadioLinkModel | None = None,
        extra_sensors: list[tuple[str, BaseSensor]] | None = None,
    ) -> None:
        self._scheduler = SensorScheduler()

        if rgb_camera is not None:
            self._scheduler.add(rgb_camera, name="rgb")
        if event_camera is not None:
            self._scheduler.add(event_camera, name="events")
        if thermal_camera is not None:
            self._scheduler.add(thermal_camera, name="thermal")
        if lidar is not None:
            self._scheduler.add(lidar, name="lidar")
        if gnss is not None:
            self._scheduler.add(gnss, name="gnss")
        if radio is not None:
            self._scheduler.add(radio, name="radio")

        for name, sensor in extra_sensors or []:
            self._scheduler.add(sensor, name=name)

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        rgb_rate_hz: float = 30.0,
        event_rate_hz: float = 1000.0,
        thermal_rate_hz: float = 9.0,
        lidar_rate_hz: float = 10.0,
        gnss_rate_hz: float = 10.0,
        radio_rate_hz: float = 100.0,
    ) -> "SensorSuite":
        """
        Create a :class:`SensorSuite` with default sensor configurations.

        All parameters are optional; pass ``0`` (or a negative value) to
        disable a specific sensor.

        Parameters
        ----------
        rgb_rate_hz:
            RGB camera frame rate.
        event_rate_hz:
            Event camera update rate.
        thermal_rate_hz:
            Thermal camera frame rate.
        lidar_rate_hz:
            LiDAR rotation rate.
        gnss_rate_hz:
            GNSS output rate.
        radio_rate_hz:
            Radio link scheduler poll rate.
        """
        return cls(
            rgb_camera=CameraModel(update_rate_hz=rgb_rate_hz) if rgb_rate_hz > 0 else None,
            event_camera=EventCameraModel(update_rate_hz=event_rate_hz) if event_rate_hz > 0 else None,
            thermal_camera=ThermalCameraModel(update_rate_hz=thermal_rate_hz) if thermal_rate_hz > 0 else None,
            lidar=LidarModel(update_rate_hz=lidar_rate_hz) if lidar_rate_hz > 0 else None,
            gnss=GNSSModel(update_rate_hz=gnss_rate_hz) if gnss_rate_hz > 0 else None,
            radio=RadioLinkModel(update_rate_hz=radio_rate_hz) if radio_rate_hz > 0 else None,
        )

    # ------------------------------------------------------------------
    # Sensor lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset all sensor models (call at the start of each episode)."""
        self._scheduler.reset(env_id=env_id)

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Advance all due sensors and return their observations.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds (e.g., ``scene.cur_t``).
        state:
            Ideal measurements from Genesis.  See module docstring for the
            expected layout.

        Returns
        -------
        dict
            Mapping ``sensor_name → observation_dict``.
        """
        return self._scheduler.update(sim_time=sim_time, state=state)

    # ------------------------------------------------------------------
    # Direct sensor access
    # ------------------------------------------------------------------

    def get_sensor(self, name: str) -> BaseSensor:
        """Return a registered sensor by name."""
        return self._scheduler.get_sensor(name)

    def sensor_names(self) -> list[str]:
        """Return all registered sensor names."""
        return self._scheduler.sensor_names()

    @property
    def scheduler(self) -> SensorScheduler:
        """The underlying :class:`SensorScheduler`."""
        return self._scheduler

    def __repr__(self) -> str:
        names = self.sensor_names()
        return f"SensorSuite(sensors={names})"
