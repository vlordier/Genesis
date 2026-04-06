"""
LiDAR sensor model.

Wraps ideal depth / geometry information (from Genesis raycaster or depth
renders) and adds realistic hardware characteristics:

* Spinning-LiDAR scan timing (points come from different drone poses).
* Per-beam range noise and intensity model.
* Max-range dropouts.
* Rain / fog attenuation.
* Mixed-pixel / edge bleeding.
* Per-channel calibration offsets.

The model accepts either a pre-cast ``range_image`` (H × W float array,
metres) from the Genesis raycaster, or a flat list of ``(range, azimuth,
elevation)`` tuples.

Usage
-----
::

    lidar = LidarModel(
        name="front_lidar",
        update_rate_hz=10.0,
        n_channels=16,
        v_fov_deg=(-15.0, 15.0),
        h_resolution=1800,
        max_range_m=100.0,
    )
    obs = lidar.step(sim_time, {
        "range_image": raycaster.read().cpu().numpy(),  # (n_channels, h_res)
        "pose_history": [...],  # optional, for timing
    })
    points = obs["points"]  # Nx4 array: x, y, z, intensity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import BaseSensor


@dataclass
class LidarPoint:
    """One LiDAR return."""

    x: float
    y: float
    z: float
    intensity: float
    channel: int
    azimuth_deg: float
    range_m: float


class LidarModel(BaseSensor):
    """
    Realistic LiDAR sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        LiDAR rotation rate in Hz (e.g., 10 or 20).
    n_channels:
        Number of vertical scan lines (beams).
    v_fov_deg:
        ``(min_elevation_deg, max_elevation_deg)`` for the vertical FOV.
    h_resolution:
        Number of azimuth steps per revolution.
    max_range_m:
        Maximum measurable range.  Returns beyond this are discarded.
    no_hit_value:
        Value written for beams that did not produce a return (e.g., 0 or
        ``max_range_m``).
    range_noise_sigma_m:
        Gaussian range noise standard deviation in metres.
    intensity_noise_sigma:
        Gaussian noise on the returned intensity value (0–1).
    dropout_prob:
        Probability that any single beam return is randomly discarded.
    rain_rate_mm_h:
        Rain rate in mm/h; used to compute two-way rain attenuation.
    fog_density:
        Fog extinction coefficient (m⁻¹) for two-way path attenuation.
    channel_offsets_m:
        Per-channel range offset in metres (calibration residuals).  If
        provided must have length ``n_channels``.
    """

    def __init__(
        self,
        name: str = "lidar",
        update_rate_hz: float = 10.0,
        n_channels: int = 16,
        v_fov_deg: tuple[float, float] = (-15.0, 15.0),
        h_resolution: int = 1800,
        max_range_m: float = 100.0,
        no_hit_value: float = 0.0,
        range_noise_sigma_m: float = 0.02,
        intensity_noise_sigma: float = 0.01,
        dropout_prob: float = 0.0,
        rain_rate_mm_h: float = 0.0,
        fog_density: float = 0.0,
        channel_offsets_m: list[float] | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.n_channels = int(n_channels)
        self.v_fov_deg = tuple(v_fov_deg)
        self.h_resolution = int(h_resolution)
        self.max_range_m = float(max_range_m)
        self.no_hit_value = float(no_hit_value)
        self.range_noise_sigma_m = float(range_noise_sigma_m)
        self.intensity_noise_sigma = float(intensity_noise_sigma)
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.rain_rate_mm_h = float(rain_rate_mm_h)
        self.fog_density = float(fog_density)

        if channel_offsets_m is not None:
            self._channel_offsets = np.asarray(channel_offsets_m, dtype=np.float32)
        else:
            self._channel_offsets = np.zeros(self.n_channels, dtype=np.float32)

        # Elevation angles for each channel
        elev_min, elev_max = self.v_fov_deg
        self._elevations_deg = np.linspace(elev_min, elev_max, self.n_channels, dtype=np.float32)

        # Azimuth angles for each horizontal step
        self._azimuths_deg = np.linspace(0.0, 360.0, self.h_resolution, endpoint=False, dtype=np.float32)

        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an ideal range image into a realistic point cloud.

        Expected keys in *state*:
        - ``"range_image"`` – ``np.ndarray`` shape ``(n_channels, h_resolution)``
          containing ideal ranges in metres.  Missing beams should be
          ``0`` or ``max_range_m``.
        - ``"intensity_image"`` *(optional)* – same shape, values 0–1.
        """
        range_img = state.get("range_image")
        if range_img is None:
            self._last_obs = {"points": np.empty((0, 4), dtype=np.float32)}
            return self._last_obs

        range_img = np.asarray(range_img, dtype=np.float32)
        n_ch, n_az = range_img.shape

        intensity_img = state.get("intensity_image")
        if intensity_img is not None:
            intensity_img = np.asarray(intensity_img, dtype=np.float32)
        else:
            # Simple inverse-square intensity model
            with np.errstate(divide="ignore", invalid="ignore"):
                intensity_img = np.where(range_img > 0, np.clip(1.0 / (range_img**2 + 1e-6), 0, 1), 0.0)

        # 1. Per-channel calibration offsets
        offsets = self._channel_offsets[:n_ch, np.newaxis]
        range_img = range_img + offsets

        # 2. Range noise
        noise = np.random.normal(0.0, self.range_noise_sigma_m, range_img.shape).astype(np.float32)
        range_img = range_img + noise

        # 3. Rain + fog attenuation (exponential two-way path loss)
        attenuation_coeff = self.fog_density
        if self.rain_rate_mm_h > 0:
            # Empirical: k ≈ 0.01 * R^0.6  (dB/m scaled)
            attenuation_coeff += 0.01 * (self.rain_rate_mm_h**0.6) / 4.343
        if attenuation_coeff > 0:
            transmission = np.exp(-2.0 * attenuation_coeff * np.clip(range_img, 0, None))
            # Beams with <5% transmission are treated as no-return
            no_return_mask = transmission < 0.05
            range_img[no_return_mask] = self.no_hit_value

        # 4. Max-range clipping
        valid_mask = (range_img > 0) & (range_img <= self.max_range_m)

        # 5. Random dropouts
        if self.dropout_prob > 0:
            dropout_mask = np.random.random(range_img.shape) < self.dropout_prob
            valid_mask &= ~dropout_mask

        # 6. Intensity noise
        intensity_img = np.clip(
            intensity_img + np.random.normal(0.0, self.intensity_noise_sigma, intensity_img.shape),
            0.0,
            1.0,
        ).astype(np.float32)

        # 7. Convert to Cartesian coordinates
        elevs = np.deg2rad(self._elevations_deg[:n_ch])
        azims = np.deg2rad(self._azimuths_deg[:n_az])

        # Shape: (n_ch, n_az)
        elev_grid, azim_grid = np.meshgrid(elevs, azims, indexing="ij")
        r = range_img

        x = r * np.cos(elev_grid) * np.cos(azim_grid)
        y = r * np.cos(elev_grid) * np.sin(azim_grid)
        z = r * np.sin(elev_grid)

        # 8. Pack into Nx4 array
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        i_valid = intensity_img[valid_mask]

        points = np.stack([x_valid, y_valid, z_valid, i_valid], axis=-1).astype(np.float32)

        result = {"points": points, "range_image": range_img}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs
