"""
Event camera model.

Simulates a Dynamic Vision Sensor (DVS) from consecutive grayscale frames
rendered by Genesis.  For each pixel the model tracks log-intensity and
fires an event (timestamp + polarity) whenever the change exceeds a
per-pixel threshold.

Optional enhancements enabled by constructor flags:
* **Refractory period** – prevents a pixel from firing twice within a
  minimum dead-time.
* **Threshold variation** – per-pixel Gaussian spread around the nominal
  threshold to simulate manufacturing scatter.
* **Background activity (BA) noise** – Poisson-distributed spontaneous
  events unrelated to scene motion.

Reference
---------
Gallego et al., "Event-based Vision: A Survey", IEEE TPAMI 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import BaseSensor


@dataclass
class Event:
    """A single DVS event."""

    x: int
    y: int
    timestamp: float
    polarity: int  # +1 or -1


def pack_events(pos_yx: np.ndarray, neg_yx: np.ndarray, timestamp: float) -> list[Event]:
    """Convert coordinate arrays to a list of :class:`Event` objects."""
    events: list[Event] = []
    for y, x in pos_yx:
        events.append(Event(x=int(x), y=int(y), timestamp=timestamp, polarity=1))
    for y, x in neg_yx:
        events.append(Event(x=int(x), y=int(y), timestamp=timestamp, polarity=-1))
    return events


class EventCameraModel(BaseSensor):
    """
    Event camera simulator based on log-intensity change detection.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Rate at which grayscale frames are consumed.  Set to the Genesis
        render rate or a multiple of it.
    threshold_pos:
        Positive contrast threshold C+ (log-intensity units).
    threshold_neg:
        Negative contrast threshold C- (log-intensity units).
    refractory_period_s:
        Minimum time between two events at the same pixel (seconds).
        Set to ``0`` to disable.
    threshold_variation:
        Relative Gaussian spread of per-pixel thresholds (σ / nominal).
        Set to ``0`` to disable.
    background_activity_rate_hz:
        Mean rate of spontaneous (noise) events per pixel per second.
        Set to ``0`` to disable.
    """

    def __init__(
        self,
        name: str = "event_camera",
        update_rate_hz: float = 1000.0,
        threshold_pos: float = 0.2,
        threshold_neg: float = 0.2,
        refractory_period_s: float = 0.0,
        threshold_variation: float = 0.0,
        background_activity_rate_hz: float = 0.0,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.threshold_pos = float(threshold_pos)
        self.threshold_neg = float(threshold_neg)
        self.refractory_period_s = float(refractory_period_s)
        self.threshold_variation = float(threshold_variation)
        self.background_activity_rate_hz = float(background_activity_rate_hz)

        self._prev_log: np.ndarray | None = None
        self._last_fire_time: np.ndarray | None = None  # per-pixel last-event timestamp
        self._th_pos_map: np.ndarray | None = None  # per-pixel positive threshold
        self._th_neg_map: np.ndarray | None = None  # per-pixel negative threshold
        self._events: list[Event] = []

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._prev_log = None
        self._last_fire_time = None
        self._th_pos_map = None
        self._th_neg_map = None
        self._events = []
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate events from a new grayscale frame.

        Expected keys in *state*:
        - ``"gray"`` – ``np.ndarray`` shape ``(H, W)`` or ``(H, W, 1)``
          dtype ``uint8`` or ``float32``.  Values are normalised to
          ``[0, 1]`` internally.
        """
        gray = state.get("gray")
        if gray is None:
            # Accept an RGB frame and convert to grayscale
            rgb = state.get("rgb")
            if rgb is None:
                self._events = []
                return {"events": self._events}
            gray = self._rgb_to_gray(rgb)

        gray = np.asarray(gray, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray[..., 0]
        if gray.max() > 1.0:
            gray = gray / 255.0

        h, w = gray.shape

        # Lazy initialisation of per-pixel maps
        if self._prev_log is None:
            self._prev_log = np.log(np.clip(gray, 1e-4, None))
            self._events = []
            self._mark_updated(sim_time)
            return {"events": self._events}

        if self.threshold_variation > 0 and (self._th_pos_map is None or self._th_pos_map.shape != (h, w)):
            rng = np.random.default_rng(seed=0)
            sigma_p = self.threshold_variation * self.threshold_pos
            sigma_n = self.threshold_variation * self.threshold_neg
            self._th_pos_map = np.clip(rng.normal(self.threshold_pos, sigma_p, (h, w)).astype(np.float32), 0.01, None)
            self._th_neg_map = np.clip(rng.normal(self.threshold_neg, sigma_n, (h, w)).astype(np.float32), 0.01, None)

        if self.refractory_period_s > 0 and (self._last_fire_time is None or self._last_fire_time.shape != (h, w)):
            self._last_fire_time = np.full((h, w), -np.inf, dtype=np.float32)

        log_i = np.log(np.clip(gray, 1e-4, None))
        delta = log_i - self._prev_log

        th_p = self._th_pos_map if self._th_pos_map is not None else self.threshold_pos
        th_n = self._th_neg_map if self._th_neg_map is not None else self.threshold_neg

        pos_mask = delta > th_p
        neg_mask = delta < -th_n

        # Refractory period: suppress pixels that fired too recently
        if self.refractory_period_s > 0 and self._last_fire_time is not None:
            active = (sim_time - self._last_fire_time) >= self.refractory_period_s
            pos_mask &= active
            neg_mask &= active

        pos_yx = np.argwhere(pos_mask)
        neg_yx = np.argwhere(neg_mask)

        events = pack_events(pos_yx, neg_yx, sim_time)

        # Update last-fire time for pixels that fired
        if self.refractory_period_s > 0 and self._last_fire_time is not None:
            fire_mask = pos_mask | neg_mask
            self._last_fire_time[fire_mask] = sim_time

        # Background activity noise
        if self.background_activity_rate_hz > 0:
            dt = 1.0 / self.update_rate_hz
            # Expected number of noise events per frame
            mean_noise = self.background_activity_rate_hz * h * w * dt
            n_noise = np.random.poisson(mean_noise)
            if n_noise > 0:
                rng = np.random.default_rng()
                xs = rng.integers(0, w, n_noise)
                ys = rng.integers(0, h, n_noise)
                pols = rng.choice([-1, 1], n_noise)
                for x, y, p in zip(xs, ys, pols):
                    events.append(Event(x=int(x), y=int(y), timestamp=sim_time, polarity=int(p)))

        self._prev_log = log_i
        self._events = events
        self._mark_updated(sim_time)
        return {"events": events}

    def get_observation(self) -> dict[str, Any]:
        return {"events": self._events}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
