"""
Radio / swarm communications link model.

Simulates packet-level radio link behaviour between drones and a ground
control station (GCS).  For each transmitted packet the model:

1. Computes the Euclidean distance and checks line-of-sight (LoS).
2. Estimates path loss using the log-distance model.
3. Adds shadow-fading (log-normal, dB).
4. Derives a received SNR and maps it to packet-drop probability via a
   configurable SNR-to-PER curve.
5. Schedules delivery with realistic latency and jitter, or discards the
   packet.

The result is a time-ordered stream of :class:`ScheduledPacket` objects
that are delivered asynchronously at ``delivery_time``.

Usage
-----
::

    radio = RadioLinkModel(name="swarm_radio", update_rate_hz=100.0)
    maybe_pkt = radio.transmit(
        packet={"cmd": "hover"},
        src_pos=np.array([0, 0, 10]),
        dst_pos=np.array([5, 5, 10]),
        sim_time=1.0,
    )
    obs = radio.step(sim_time=1.1, state={})
    delivered = obs["delivered"]  # list of packets whose delivery_time <= sim_time
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .base import BaseSensor
from .types import ArrayLike, Float64Array, RadioObservation

# Speed of light (m/s)
_SPEED_OF_LIGHT_M_S: Final[float] = 3e8
# Johnson-Nyquist thermal noise floor (dBm/Hz) at 290 K
_THERMAL_NOISE_DBM_PER_HZ: Final[float] = -174.0
# Reference bandwidth (1 MHz) used in the noise floor calculation
_REF_BANDWIDTH_HZ: Final[float] = 1e6
# Typical excess path loss (dB) in non-line-of-sight conditions
_NLOS_EXCESS_LOSS_DB_DEFAULT: Final[float] = 20.0
# Steepness of the sigmoid mapping SNR -> PER
_PER_SIGMOID_STEEPNESS: Final[float] = 5.0
# Minimum distance clamp to avoid log(0) in path-loss calculations
_MIN_DISTANCE_M: Final[float] = 0.1


@dataclass
class ScheduledPacket:
    """A packet scheduled for future delivery."""

    payload: Any
    src_pos: Float64Array
    dst_pos: Float64Array
    send_time: float
    delivery_time: float


class RadioLinkModel(BaseSensor):
    """
    Radio / swarm communications link model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Rate at which the scheduler is polled to deliver pending packets.
    tx_power_dbm:
        Transmit power in dBm.
    frequency_ghz:
        Carrier frequency in GHz (affects free-space path loss).
    noise_figure_db:
        Receiver noise figure in dB.
    path_loss_exponent:
        Log-distance path-loss exponent (2 = free space, 3-4 = urban).
    shadowing_sigma_db:
        Standard deviation of log-normal shadow fading in dB.
    min_snr_db:
        SNR below which packet error rate becomes 1 (complete link failure).
    snr_transition_db:
        SNR range (dB) over which PER transitions from 0 to 1 (sigmoid).
    base_latency_s:
        Minimum latency (processing + propagation) in seconds.
    jitter_sigma_s:
        Standard deviation of latency jitter in seconds.
    nlos_excess_loss_db:
        Additional path loss (dB) applied when ``has_los=False``.
    seed:
        Optional seed for the random-number generator (reproducibility).
    los_required:
        If ``True``, packets sent without line-of-sight are dropped
        immediately.  Must be passed as a keyword argument.
    """

    def __init__(
        self,
        name: str = "radio",
        update_rate_hz: float = 100.0,
        tx_power_dbm: float = 20.0,
        frequency_ghz: float = 2.4,
        noise_figure_db: float = 6.0,
        path_loss_exponent: float = 2.5,
        shadowing_sigma_db: float = 4.0,
        min_snr_db: float = -5.0,
        snr_transition_db: float = 10.0,
        base_latency_s: float = 0.001,
        jitter_sigma_s: float = 0.0005,
        nlos_excess_loss_db: float = _NLOS_EXCESS_LOSS_DB_DEFAULT,
        seed: int | None = None,
        *,
        los_required: bool = False,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.tx_power_dbm = float(tx_power_dbm)
        self.frequency_ghz = float(frequency_ghz)
        self.noise_figure_db = float(noise_figure_db)
        self.path_loss_exponent = float(path_loss_exponent)
        self.shadowing_sigma_db = float(shadowing_sigma_db)
        self.min_snr_db = float(min_snr_db)
        self.snr_transition_db = float(snr_transition_db)
        self.base_latency_s = float(base_latency_s)
        self.jitter_sigma_s = float(jitter_sigma_s)
        self.nlos_excess_loss_db = float(nlos_excess_loss_db)
        self.los_required = los_required
        self._rng = np.random.default_rng(seed=seed)

        # kTB noise floor at 290 K, 1 MHz bandwidth -> ~-114 dBm/MHz
        self._noise_floor_dbm: float = (
            _THERMAL_NOISE_DBM_PER_HZ + 10.0 * math.log10(_REF_BANDWIDTH_HZ) + self.noise_figure_db
        )

        # ------------------------------------------------------------------
        # Pre-computed free-space path loss constant
        # ------------------------------------------------------------------
        # FSPL = 20*log10(4π/λ) + 10*n*log10(d)
        # The first term depends only on frequency (constant) and is cached.
        freq_hz = self.frequency_ghz * 1e9
        lambda_m = _SPEED_OF_LIGHT_M_S / freq_hz
        self._fspl_const_db: float = 20.0 * math.log10(4.0 * math.pi / lambda_m)
        # Pre-compute the PER sigmoid denominator guard
        self._snr_transition_safe: float = max(float(snr_transition_db), 1e-3)

        self._queue: list[ScheduledPacket] = []
        self._last_obs: dict[str, Any] = {"delivered": []}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        """Number of packets currently awaiting delivery."""
        return len(self._queue)

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._queue.clear()
        self._last_obs = {"delivered": []}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> RadioObservation | dict[str, Any]:
        """
        Deliver all packets whose scheduled delivery time has passed.

        The *state* dict is not used by this model but kept for API
        consistency with other sensors.
        """
        # Single-pass partition: avoid two list comprehensions over the queue.
        delivered: list[ScheduledPacket] = []
        remaining: list[ScheduledPacket] = []
        for pkt in self._queue:
            (delivered if pkt.delivery_time <= sim_time else remaining).append(pkt)
        self._queue = remaining
        result: RadioObservation = {"delivered": delivered, "queue_depth": len(remaining)}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Packet transmission
    # ------------------------------------------------------------------

    def transmit(
        self,
        packet: Any,
        src_pos: ArrayLike,
        dst_pos: ArrayLike,
        sim_time: float,
        *,
        has_los: bool = True,
    ) -> ScheduledPacket | None:
        """
        Attempt to transmit *packet* from *src_pos* to *dst_pos*.

        Parameters
        ----------
        packet:
            Arbitrary payload.
        src_pos, dst_pos:
            3-D world-frame positions in metres (any array-like accepted).
        sim_time:
            Current simulation time in seconds.
        has_los:
            Whether there is line-of-sight between the two nodes.  Must be
            passed as a keyword argument.  Set to ``False`` to model NLOS
            (adds ``nlos_excess_loss_db`` extra attenuation).

        Returns
        -------
        ScheduledPacket or None
            The scheduled packet, or ``None`` if it was dropped.
        """
        if self.los_required and not has_los:
            return None

        src: Float64Array = np.asarray(src_pos, dtype=np.float64)
        dst: Float64Array = np.asarray(dst_pos, dtype=np.float64)
        dist_m = max(float(np.linalg.norm(dst - src)), _MIN_DISTANCE_M)

        rx_power_dbm = self._compute_rx_power(dist_m, has_los=has_los)
        snr_db = rx_power_dbm - self._noise_floor_dbm

        per = self._snr_to_per(snr_db)
        if self._rng.random() < per:
            return None  # packet dropped

        # Latency + jitter
        propagation_s = dist_m / _SPEED_OF_LIGHT_M_S
        jitter = abs(float(self._rng.normal(0.0, self.jitter_sigma_s)))
        delivery_time = sim_time + self.base_latency_s + propagation_s + jitter

        pkt = ScheduledPacket(
            payload=packet,
            src_pos=src,
            dst_pos=dst,
            send_time=sim_time,
            delivery_time=delivery_time,
        )
        self._queue.append(pkt)
        return pkt

    # ------------------------------------------------------------------
    # Link budget helpers
    # ------------------------------------------------------------------

    def estimate_link_metrics(
        self,
        src_pos: ArrayLike,
        dst_pos: ArrayLike,
        *,
        has_los: bool = True,
    ) -> dict[str, float]:
        """
        Return estimated link quality metrics without sending a packet.

        Useful for monitoring / visualisation.  ``has_los`` must be passed
        as a keyword argument.
        """
        src: Float64Array = np.asarray(src_pos, dtype=np.float64)
        dst: Float64Array = np.asarray(dst_pos, dtype=np.float64)
        dist_m = max(float(np.linalg.norm(dst - src)), _MIN_DISTANCE_M)

        rx_power_dbm = self._compute_rx_power(dist_m, has_los=has_los)
        snr_db = rx_power_dbm - self._noise_floor_dbm
        per = self._snr_to_per(snr_db)

        return {
            "distance_m": dist_m,
            "rx_power_dbm": rx_power_dbm,
            "snr_db": snr_db,
            "packet_error_rate": per,
            "has_los": float(has_los),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rx_power(self, dist_m: float, *, has_los: bool) -> float:
        """Compute received power in dBm for a given distance and LoS flag.

        Uses pre-computed ``_fspl_const_db`` (depends only on frequency) and
        the standard library ``math.log10`` (faster than numpy for scalars).
        """
        pl_db = self._fspl_const_db + 10.0 * self.path_loss_exponent * math.log10(dist_m)
        if not has_los:
            pl_db += self.nlos_excess_loss_db
        shadow_db = float(self._rng.normal(0.0, self.shadowing_sigma_db))
        return self.tx_power_dbm - pl_db - shadow_db

    def _snr_to_per(self, snr_db: float) -> float:
        """Sigmoid mapping from SNR to packet error rate.

        Uses ``math.exp`` (faster than numpy for scalar inputs).
        """
        # PER -> 0 at high SNR, -> 1 at low SNR
        x = -(snr_db - self.min_snr_db) / self._snr_transition_safe
        per = 1.0 / (1.0 + math.exp(-x * _PER_SIGMOID_STEEPNESS))
        return min(max(per, 0.0), 1.0)
