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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import BaseSensor


@dataclass
class ScheduledPacket:
    """A packet scheduled for future delivery."""

    payload: Any
    src_pos: np.ndarray
    dst_pos: np.ndarray
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
        Log-distance path-loss exponent (2 = free space, 3–4 = urban).
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
    los_required:
        If ``True``, packets sent without line-of-sight are dropped.
    """

    SPEED_OF_LIGHT = 3e8  # m/s

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
        self.los_required = bool(los_required)

        # kTB noise floor at 290 K, 1 MHz bandwidth → ~−114 dBm / MHz
        self._noise_floor_dbm = -174.0 + 10 * np.log10(1e6) + self.noise_figure_db

        self._queue: list[ScheduledPacket] = []
        self._last_obs: dict[str, Any] = {"delivered": []}

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._queue.clear()
        self._last_obs = {"delivered": []}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Deliver all packets whose scheduled delivery time has passed.

        The *state* dict is not used by this model but kept for API
        consistency with other sensors.
        """
        delivered = [p for p in self._queue if p.delivery_time <= sim_time]
        self._queue = [p for p in self._queue if p.delivery_time > sim_time]
        result = {"delivered": delivered, "queue_depth": len(self._queue)}
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
        src_pos: np.ndarray,
        dst_pos: np.ndarray,
        sim_time: float,
        has_los: bool = True,
    ) -> ScheduledPacket | None:
        """
        Attempt to transmit *packet* from *src_pos* to *dst_pos*.

        Parameters
        ----------
        packet:
            Arbitrary payload.
        src_pos, dst_pos:
            3-D world-frame positions in metres.
        sim_time:
            Current simulation time in seconds.
        has_los:
            Whether there is line-of-sight between the two nodes.  Set to
            ``False`` to model NLOS (adds extra attenuation).

        Returns
        -------
        ScheduledPacket or None
            The scheduled packet, or ``None`` if it was dropped.
        """
        if self.los_required and not has_los:
            return None

        src_pos = np.asarray(src_pos, dtype=np.float64)
        dst_pos = np.asarray(dst_pos, dtype=np.float64)
        dist_m = float(np.linalg.norm(dst_pos - src_pos))
        dist_m = max(dist_m, 0.1)  # avoid log(0)

        # Free-space path loss + log-distance model
        freq_hz = self.frequency_ghz * 1e9
        lambda_m = self.SPEED_OF_LIGHT / freq_hz
        fspl_db = 20 * np.log10(4 * np.pi / lambda_m)
        pl_db = fspl_db + 10 * self.path_loss_exponent * np.log10(dist_m)

        # NLOS penalty
        if not has_los:
            pl_db += 20.0  # typical NLOS excess loss

        # Shadow fading
        shadow_db = np.random.normal(0.0, self.shadowing_sigma_db)
        rx_power_dbm = self.tx_power_dbm - pl_db - shadow_db

        # SNR
        snr_db = rx_power_dbm - self._noise_floor_dbm

        # Packet error probability via sigmoid
        per = self._snr_to_per(snr_db)
        if np.random.random() < per:
            return None  # packet dropped

        # Latency + jitter
        propagation_s = dist_m / self.SPEED_OF_LIGHT
        jitter = abs(np.random.normal(0.0, self.jitter_sigma_s))
        delivery_time = sim_time + self.base_latency_s + propagation_s + jitter

        pkt = ScheduledPacket(
            payload=packet,
            src_pos=src_pos,
            dst_pos=dst_pos,
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
        src_pos: np.ndarray,
        dst_pos: np.ndarray,
        has_los: bool = True,
    ) -> dict[str, float]:
        """
        Return estimated link quality metrics without sending a packet.

        Useful for monitoring / visualisation.
        """
        src_pos = np.asarray(src_pos, dtype=np.float64)
        dst_pos = np.asarray(dst_pos, dtype=np.float64)
        dist_m = float(np.linalg.norm(dst_pos - src_pos))
        dist_m = max(dist_m, 0.1)

        freq_hz = self.frequency_ghz * 1e9
        lambda_m = self.SPEED_OF_LIGHT / freq_hz
        fspl_db = 20 * np.log10(4 * np.pi / lambda_m)
        pl_db = fspl_db + 10 * self.path_loss_exponent * np.log10(dist_m)
        if not has_los:
            pl_db += 20.0
        rx_power_dbm = self.tx_power_dbm - pl_db
        snr_db = rx_power_dbm - self._noise_floor_dbm
        per = self._snr_to_per(snr_db)

        return {
            "distance_m": dist_m,
            "rx_power_dbm": rx_power_dbm,
            "snr_db": snr_db,
            "packet_error_rate": per,
            "has_los": has_los,
        }

    def _snr_to_per(self, snr_db: float) -> float:
        """Sigmoid mapping from SNR to packet error rate."""
        # PER → 0 at high SNR, → 1 at low SNR
        x = -(snr_db - self.min_snr_db) / max(self.snr_transition_db, 1e-3)
        per = 1.0 / (1.0 + np.exp(-x * 5))  # steepness factor 5
        return float(np.clip(per, 0.0, 1.0))
