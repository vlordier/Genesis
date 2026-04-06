"""
Multi-rate sensor scheduler.

Maintains a collection of :class:`~genesis.sensors.base.BaseSensor`
instances and drives them at their individual update rates.  At each
simulation step the caller passes the current time and an ideal-state
dict; the scheduler determines which sensors are *due* for an update,
calls their ``step()`` methods, and returns only the observations that
were refreshed.

Sensors that are not yet due return their *cached* last observation so
that consumers always have a valid (possibly stale) measurement.

Example
-------
::

    scheduler = SensorScheduler()
    scheduler.add(gnss, name="gnss")
    scheduler.add(imu_model, name="imu")

    while sim.running:
        state = collect_genesis_state(scene)
        obs = scheduler.update(sim_time=scene.cur_t, state=state)
        print(obs["gnss"])   # freshly updated (if due) or cached
"""

from __future__ import annotations

from typing import Any

from .base import BaseSensor


class SensorScheduler:
    """
    Multi-rate sensor scheduler.

    Parameters
    ----------
    sensors:
        Optional initial list of ``(name, sensor)`` pairs.
    """

    def __init__(self, sensors: list[tuple[str, BaseSensor]] | None = None) -> None:
        self._sensors: dict[str, BaseSensor] = {}
        for name, sensor in sensors or []:
            self.add(sensor, name=name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, sensor: BaseSensor, name: str | None = None) -> None:
        """Register a sensor.  *name* defaults to ``sensor.name``."""
        key = name or sensor.name
        if key in self._sensors:
            raise ValueError(f"A sensor named {key!r} is already registered.")
        self._sensors[key] = sensor

    def remove(self, name: str) -> None:
        """Unregister a sensor by name."""
        self._sensors.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    def __len__(self) -> int:
        return len(self._sensors)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset all registered sensors."""
        for sensor in self._sensors.values():
            sensor.reset(env_id=env_id)

    def update(self, sim_time: float, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Advance all due sensors and return a merged observation dict.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds.
        state:
            Ideal Genesis measurements shared across all sensors.

        Returns
        -------
        dict
            Mapping ``sensor_name → observation_dict``.  Sensors that
            were not due for an update contribute their cached
            ``get_observation()`` result.
        """
        observations: dict[str, dict[str, Any]] = {}
        for name, sensor in self._sensors.items():
            if sensor.is_due(sim_time):
                obs = sensor.step(sim_time=sim_time, state=state)
            else:
                obs = sensor.get_observation()
            observations[name] = obs
        return observations

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def sensor_names(self) -> list[str]:
        """Return registered sensor names."""
        return list(self._sensors.keys())

    def get_sensor(self, name: str) -> BaseSensor:
        """Return the sensor registered under *name*."""
        return self._sensors[name]

    def __repr__(self) -> str:
        entries = ", ".join(f"{name}@{s.update_rate_hz}Hz" for name, s in self._sensors.items())
        return f"SensorScheduler([{entries}])"
