"""
Base sensor interface for the Genesis external sensor realism layer.

All sensor models in this package implement this interface so that the
:class:`SensorScheduler` and :class:`SensorSuite` can drive them in a
uniform, rate-aware fashion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

# Half a simulation sub-step at 10 kHz — used to guard against float drift in
# the scheduling comparison.
_SCHEDULE_TOLERANCE_S: Final[float] = 5e-5


class BaseSensor(ABC):
    """
    Abstract base class for every external sensor model.

    Subclasses receive an *ideal* state dict produced by Genesis at every
    scheduled step and return device-like (noisy, delayed, corrupted)
    observations.

    Parameters
    ----------
    name:
        Human-readable identifier used in logs and the :class:`SensorSuite`.
    update_rate_hz:
        Desired sensor update rate in Hz.  The :class:`SensorScheduler` uses
        this to determine when ``step()`` should be called.
    """

    def __init__(self, name: str = "", update_rate_hz: float = 1.0) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__!r}")
        if update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be positive, got {update_rate_hz}")
        self.name = name or type(self).__name__
        self.update_rate_hz = float(update_rate_hz)
        self._last_update_time: float = -1.0

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, env_id: int = 0) -> None:
        """
        Reset all internal state for the given environment index.

        Called at the start of each episode / scene reset.
        """

    @abstractmethod
    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Advance the sensor model by one update tick.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds.
        state:
            Ideal measurements from Genesis.  Which keys are present
            depends on the sensor type; each subclass documents what it
            expects.

        Returns
        -------
        dict
            Sensor observation after applying the device model (noise,
            distortion, timing artefacts, …).
        """

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Return the most recent sensor observation without re-computing it."""

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def is_due(self, sim_time: float) -> bool:
        """
        Return ``True`` when the sensor is ready for the next update.

        Comparison is done with a small tolerance (half a simulation
        sub-step at 10 kHz) to avoid floating-point drift.
        """
        if self._last_update_time < 0:
            return True
        period = 1.0 / self.update_rate_hz
        return (sim_time - self._last_update_time) >= period - _SCHEDULE_TOLERANCE_S

    def _mark_updated(self, sim_time: float) -> None:
        self._last_update_time = sim_time

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, rate={self.update_rate_hz} Hz)"


__all__ = ["BaseSensor"]
