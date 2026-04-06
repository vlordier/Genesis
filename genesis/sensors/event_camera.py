"""
Event camera model.

Simulates a Dynamic Vision Sensor (DVS) from consecutive grayscale frames
rendered by Genesis.  For each pixel the model tracks log-intensity and
fires an event (timestamp + polarity) whenever the change exceeds a
per-pixel threshold.

Optional enhancements enabled by constructor flags:

* **Refractory period** -- prevents a pixel from firing twice within a
  minimum dead-time.
* **Threshold variation** -- per-pixel Gaussian spread around the nominal
  threshold to simulate manufacturing scatter.
* **Background activity (BA) noise** -- Poisson-distributed spontaneous
  events unrelated to scene motion.

Reference
---------
Gallego et al., "Event-based Vision: A Survey", IEEE TPAMI 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import EventCameraObservation, FloatArray, Polarity

if TYPE_CHECKING:
    from .config import EventCameraConfig

# Minimum intensity value used when computing log-intensity to avoid log(0).
_LOG_CLIP_MIN: Final[float] = 1e-4
# Minimum admissible per-pixel threshold (prevents division by zero / degenerate maps).
_MIN_PIXEL_THRESHOLD: Final[float] = 0.01
# Number of dimensions for a 3-D image array (H, W, C).
_NDIM_3D: Final[int] = 3


@dataclass(frozen=True)
class Event:
    """
    A single DVS event.

    Frozen to allow use in sets and as dict keys, and to prevent
    accidental mutation of recorded event streams.
    """

    x: int
    y: int
    timestamp: float
    polarity: Polarity  # Polarity.POSITIVE (+1) or Polarity.NEGATIVE (-1)


def _pack_events(
    pos_yx: np.ndarray,
    neg_yx: np.ndarray,
    timestamp: float,
) -> list[Event]:
    """Convert coordinate arrays to a list of :class:`Event` objects."""
    events: list[Event] = []
    for y, x in pos_yx:
        events.append(Event(x=int(x), y=int(y), timestamp=timestamp, polarity=Polarity.POSITIVE))
    for y, x in neg_yx:
        events.append(Event(x=int(x), y=int(y), timestamp=timestamp, polarity=Polarity.NEGATIVE))
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
        Relative Gaussian spread of per-pixel thresholds (sigma / nominal).
        Set to ``0`` to disable.
    background_activity_rate_hz:
        Mean rate of spontaneous (noise) events per pixel per second.
        Set to ``0`` to disable.
    seed:
        Optional seed for the random-number generator (reproducibility).
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
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.threshold_pos = float(threshold_pos)
        self.threshold_neg = float(threshold_neg)
        self.refractory_period_s = float(refractory_period_s)
        self.threshold_variation = float(threshold_variation)
        self.background_activity_rate_hz = float(background_activity_rate_hz)
        self._rng = np.random.default_rng(seed=seed)

        # Pre-compute frame interval (used in background activity rate calculation).
        self._dt: float = 1.0 / self.update_rate_hz

        self._prev_log: FloatArray | None = None
        self._last_fire_time: FloatArray | None = None  # per-pixel last-event timestamp
        self._th_pos_map: FloatArray | None = None  # per-pixel positive threshold
        self._th_neg_map: FloatArray | None = None  # per-pixel negative threshold
        self._events: list[Event] = []
        self._last_obs: dict[str, Any] = {"events": []}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "EventCameraConfig") -> "EventCameraModel":
        """Construct an :class:`EventCameraModel` from an :class:`~genesis.sensors.config.EventCameraConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "EventCameraConfig":
        """Return the current parameters as an :class:`~genesis.sensors.config.EventCameraConfig`."""
        from .config import EventCameraConfig

        return EventCameraConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            threshold_pos=self.threshold_pos,
            threshold_neg=self.threshold_neg,
            refractory_period_s=self.refractory_period_s,
            threshold_variation=self.threshold_variation,
            background_activity_rate_hz=self.background_activity_rate_hz,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """``True`` after the first frame has been consumed."""
        return self._prev_log is not None

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._prev_log = None
        self._last_fire_time = None
        self._th_pos_map = None
        self._th_neg_map = None
        self._events = []
        self._last_obs: dict[str, Any] = {"events": []}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> EventCameraObservation | dict[str, Any]:
        """
        Generate events from a new grayscale frame.

        Expected keys in *state*:
        - ``"gray"`` -- ``np.ndarray`` shape ``(H, W)`` or ``(H, W, 1)``
          dtype ``uint8`` or ``float32``.  Values are normalised to
          ``[0, 1]`` internally.
        - ``"rgb"`` -- accepted as a fallback when ``"gray"`` is absent;
          converted to grayscale via the ITU-R BT.709 luma weights.
        """
        gray = self._load_gray(state)
        if gray is None:
            self._events = []
            self._last_obs = {"events": self._events}
            return self._last_obs

        h, w = gray.shape
        if self._prev_log is None:
            # First frame: initialise log-intensity buffer, no events yet.
            self._prev_log = np.log(np.clip(gray, _LOG_CLIP_MIN, None))
            self._events = []
            self._last_obs = {"events": self._events}
            self._mark_updated(sim_time)
            return self._last_obs

        self._ensure_threshold_maps(h, w)
        self._ensure_refractory_map(h, w)

        log_i = np.log(np.clip(gray, _LOG_CLIP_MIN, None))
        events = self._detect_events(log_i, sim_time)
        events.extend(self._add_background_events(h, w, sim_time))

        self._prev_log = log_i
        self._events = events
        self._last_obs = {"events": events}
        self._mark_updated(sim_time)
        return self._last_obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Private helpers -- frame loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_gray(state: dict[str, Any]) -> FloatArray | None:
        """Extract a normalised float32 grayscale image from *state*.

        Calls ``np.asarray`` only once on the raw input and checks its dtype
        to determine whether to scale by 1/255 (uint8 origin), rather than
        relying on ``img.max() > 1`` which is unreliable for all-dark images.
        """
        raw = state.get("gray")
        if raw is None:
            rgb = state.get("rgb")
            if rgb is None:
                return None
            raw = EventCameraModel._rgb_to_gray(rgb)

        # Single asarray call — reuse the result for both dtype check and value.
        raw_arr = np.asarray(raw)
        is_uint8 = raw_arr.dtype == np.uint8
        gray: FloatArray = raw_arr.astype(np.float32)
        if gray.ndim == _NDIM_3D:
            gray = gray[..., 0]
        if is_uint8:
            gray = gray / 255.0
        return gray

    # ------------------------------------------------------------------
    # Private helpers -- lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_threshold_maps(self, h: int, w: int) -> None:
        """Lazily initialise per-pixel threshold maps when variation > 0."""
        if self.threshold_variation <= 0:
            return
        if self._th_pos_map is not None and self._th_pos_map.shape == (h, w):
            return
        sigma_p = self.threshold_variation * self.threshold_pos
        sigma_n = self.threshold_variation * self.threshold_neg
        self._th_pos_map = np.clip(
            self._rng.normal(self.threshold_pos, sigma_p, (h, w)).astype(np.float32),
            _MIN_PIXEL_THRESHOLD,
            None,
        )
        self._th_neg_map = np.clip(
            self._rng.normal(self.threshold_neg, sigma_n, (h, w)).astype(np.float32),
            _MIN_PIXEL_THRESHOLD,
            None,
        )

    def _ensure_refractory_map(self, h: int, w: int) -> None:
        """Lazily initialise the per-pixel last-fire-time map."""
        if self.refractory_period_s <= 0:
            return
        if self._last_fire_time is not None and self._last_fire_time.shape == (h, w):
            return
        self._last_fire_time = np.full((h, w), -np.inf, dtype=np.float32)

    # ------------------------------------------------------------------
    # Private helpers -- event detection
    # ------------------------------------------------------------------

    def _detect_events(self, log_i: FloatArray, sim_time: float) -> list[Event]:
        """Detect positive and negative events from the log-intensity delta."""
        if self._prev_log is None:
            raise RuntimeError(
                "_detect_events() called before _prev_log was initialised; call step() with at least one frame first."
            )
        delta = log_i - self._prev_log

        th_p: FloatArray | float = self._th_pos_map if self._th_pos_map is not None else self.threshold_pos
        th_n: FloatArray | float = self._th_neg_map if self._th_neg_map is not None else self.threshold_neg

        pos_mask = delta > th_p
        neg_mask = delta < -th_n

        if self.refractory_period_s > 0 and self._last_fire_time is not None:
            active = (sim_time - self._last_fire_time) >= self.refractory_period_s
            pos_mask = pos_mask & active
            neg_mask = neg_mask & active
            fire_mask = pos_mask | neg_mask
            self._last_fire_time[fire_mask] = sim_time

        return _pack_events(np.argwhere(pos_mask), np.argwhere(neg_mask), sim_time)

    def _add_background_events(self, h: int, w: int, sim_time: float) -> list[Event]:
        """Generate spontaneous background-activity noise events."""
        if self.background_activity_rate_hz <= 0:
            return []
        # Use pre-computed _dt instead of 1.0 / self.update_rate_hz each call.
        mean_noise = self.background_activity_rate_hz * h * w * self._dt
        n_noise = int(self._rng.poisson(mean_noise))
        if n_noise == 0:
            return []
        xs = self._rng.integers(0, w, n_noise)
        ys = self._rng.integers(0, h, n_noise)
        raw_pols = self._rng.choice([-1, 1], n_noise)
        return [
            Event(x=int(x), y=int(y), timestamp=sim_time, polarity=Polarity(int(p)))
            for x, y, p in zip(xs, ys, raw_pols, strict=True)
        ]

    @staticmethod
    def _rgb_to_gray(rgb: np.ndarray) -> FloatArray:
        """Convert an RGB image to grayscale using ITU-R BT.709 luma weights.

        Checks dtype to determine scaling rather than using ``arr.max() > 1``.
        """
        arr = np.asarray(rgb)
        is_uint8 = arr.dtype == np.uint8
        arr_f = arr.astype(np.float32)
        if is_uint8:
            arr_f = arr_f / 255.0
        return (0.2126 * arr_f[..., 0] + 0.7152 * arr_f[..., 1] + 0.0722 * arr_f[..., 2]).astype(np.float32)


__all__ = ["Event", "EventCameraModel"]
