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

The model accepts either a pre-cast ``range_image`` (H x W float array,
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
    })
    points = obs["points"]  # Nx4 array: x, y, z, intensity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import FloatArray, LidarObservation

if TYPE_CHECKING:
    from .config import LidarConfig

# Beams with two-way transmission below this fraction are treated as no-return.
_MIN_TRANSMISSION_FRACTION: Final[float] = 0.05
# Small epsilon used in the default inverse-square intensity model.
_INTENSITY_EPS: Final[float] = 1e-6
# Empirical rain attenuation coefficient factor (Kunkel model approximation).
_RAIN_ATTN_COEFF: Final[float] = 0.01
# Rain rate exponent in the empirical attenuation formula.
_RAIN_ATTN_EXPONENT: Final[float] = 0.6
# Conversion factor from dB/m to Np/m (1 dB = 1/4.343 Np).
_DB_TO_NP: Final[float] = 4.343


@dataclass(frozen=True)
class LidarPoint:
    """
    One LiDAR return (utility dataclass for typed point construction).

    Frozen so that point objects can be stored in sets or used as dict
    keys and cannot be accidentally mutated after recording.
    """

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
        Gaussian noise on the returned intensity value (0-1).
    dropout_prob:
        Probability that any single beam return is randomly discarded.
    rain_rate_mm_h:
        Rain rate in mm/h; used to compute two-way rain attenuation.
    fog_density:
        Fog extinction coefficient (1/m) for two-way path attenuation.
    channel_offsets_m:
        Per-channel range offset in metres (calibration residuals).  If
        provided must have length ``n_channels``.
    beam_divergence_mrad:
        Half-angle beam divergence in milli-radians.  ``0`` = disabled; a
        typical spinning LiDAR has 1.5–3.0 mrad.  Models the mixed-pixel
        effect at surface edges by applying a Gaussian blur to the range
        image before the noise pipeline, with sigma computed from the
        divergence and the sensor's angular resolution.
    seed:
        Optional seed for the random-number generator (reproducibility).
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
        beam_divergence_mrad: float = 0.0,
        seed: int | None = None,
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
        self.beam_divergence_mrad = float(max(0.0, beam_divergence_mrad))
        self._rng = np.random.default_rng(seed=seed)

        if channel_offsets_m is not None:
            self._channel_offsets = np.asarray(channel_offsets_m, dtype=np.float32)
            self._channel_offsets_m_given = True
        else:
            self._channel_offsets = np.zeros(self.n_channels, dtype=np.float32)
            self._channel_offsets_m_given = False

        # Elevation angles for each channel
        elev_min, elev_max = self.v_fov_deg
        self._elevations_deg: FloatArray = np.linspace(elev_min, elev_max, self.n_channels, dtype=np.float32)

        # Azimuth angles for each horizontal step
        self._azimuths_deg: FloatArray = np.linspace(0.0, 360.0, self.h_resolution, endpoint=False, dtype=np.float32)

        # Pre-compute trigonometric values for the fixed beam geometry.
        # These are constant for the lifetime of the sensor; caching avoids
        # repeated deg2rad + trig calls on every step().
        elev_rad = np.deg2rad(self._elevations_deg)
        azim_rad = np.deg2rad(self._azimuths_deg)
        # elev_grid: (n_channels, h_resolution); azim_grid: (n_channels, h_resolution)
        self._elev_grid: FloatArray
        self._azim_grid: FloatArray
        self._elev_grid, self._azim_grid = np.meshgrid(elev_rad, azim_rad, indexing="ij")
        self._cos_elev: FloatArray = np.cos(self._elev_grid).astype(np.float32)
        self._sin_elev: FloatArray = np.sin(self._elev_grid).astype(np.float32)
        self._cos_azim: FloatArray = np.cos(self._azim_grid).astype(np.float32)
        self._sin_azim: FloatArray = np.sin(self._azim_grid).astype(np.float32)

        # ------------------------------------------------------------------
        # Beam divergence: pre-compute the Gaussian blur sigma (pixels) in
        # azimuth and elevation from the divergence and angular resolution.
        # σ_az = beam_divergence_mrad / az_resolution_mrad
        # σ_el = beam_divergence_mrad / el_resolution_mrad
        # A minimum sigma of 0.5 is enforced so even very fine angular
        # resolution still produces some mixed-pixel softening.
        # ------------------------------------------------------------------
        self._bd_sigma_az: float = 0.0
        self._bd_sigma_el: float = 0.0
        if self.beam_divergence_mrad > 0.0 and self.n_channels > 1:
            import math

            az_res_mrad = 360.0 / self.h_resolution * (math.pi / 180.0) * 1000.0
            el_span_mrad = abs(self.v_fov_deg[1] - self.v_fov_deg[0]) * (math.pi / 180.0) * 1000.0
            el_res_mrad = el_span_mrad / max(self.n_channels - 1, 1)
            self._bd_sigma_az = max(self.beam_divergence_mrad / az_res_mrad, 0.5)
            self._bd_sigma_el = max(self.beam_divergence_mrad / el_res_mrad, 0.5)

        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "LidarConfig") -> "LidarModel":
        """Construct a :class:`LidarModel` from a :class:`~genesis.sensors.config.LidarConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "LidarConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.LidarConfig`."""
        from .config import LidarConfig

        return LidarConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_channels=self.n_channels,
            v_fov_deg=self.v_fov_deg,
            h_resolution=self.h_resolution,
            max_range_m=self.max_range_m,
            no_hit_value=self.no_hit_value,
            range_noise_sigma_m=self.range_noise_sigma_m,
            intensity_noise_sigma=self.intensity_noise_sigma,
            dropout_prob=self.dropout_prob,
            rain_rate_mm_h=self.rain_rate_mm_h,
            fog_density=self.fog_density,
            channel_offsets_m=list(self._channel_offsets) if self._channel_offsets_m_given else None,
            beam_divergence_mrad=self.beam_divergence_mrad,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> LidarObservation | dict[str, Any]:
        """
        Convert an ideal range image into a realistic point cloud.

        Expected keys in *state*:
        - ``"range_image"`` -- ``np.ndarray`` shape ``(n_channels, h_resolution)``
          containing ideal ranges in metres.  Missing beams should be
          ``0`` or ``max_range_m``.
        - ``"intensity_image"`` *(optional)* -- same shape, values 0-1.
        """
        range_img = state.get("range_image")
        if range_img is None:
            self._last_obs = {"points": np.empty((0, 4), dtype=np.float32)}
            return self._last_obs

        range_img = np.asarray(range_img, dtype=np.float32)
        n_ch, n_az = range_img.shape

        # 0. Beam divergence: Gaussian blur models mixed-pixel / edge-bleeding
        #    effect caused by finite beam solid angle.  Only applied when
        #    beam_divergence_mrad > 0 and the sigmas were pre-computed.
        if self._bd_sigma_az > 0.0:
            range_img = self._apply_beam_divergence(range_img)

        intensity_img = state.get("intensity_image")
        if intensity_img is not None:
            intensity_img = np.asarray(intensity_img, dtype=np.float32)
        else:
            # Simple inverse-square intensity model
            with np.errstate(divide="ignore", invalid="ignore"):
                intensity_img = np.where(
                    range_img > 0,
                    np.clip(1.0 / (range_img**2 + _INTENSITY_EPS), 0, 1),
                    0.0,
                )

        # 1. Per-channel calibration offsets
        offsets = self._channel_offsets[:n_ch, np.newaxis]
        range_img = range_img + offsets

        # 2. Range noise
        noise = self._rng.normal(0.0, self.range_noise_sigma_m, range_img.shape).astype(np.float32)
        range_img = range_img + noise

        # 3. Rain + fog attenuation (exponential two-way path loss)
        attenuation_coeff = self.fog_density
        if self.rain_rate_mm_h > 0:
            # Empirical approximation (Kunkel 1984): k ~ 0.01 * R^0.6 dB/m
            attenuation_coeff += _RAIN_ATTN_COEFF * (self.rain_rate_mm_h**_RAIN_ATTN_EXPONENT) / _DB_TO_NP
        if attenuation_coeff > 0:
            transmission = np.exp(-2.0 * attenuation_coeff * np.clip(range_img, 0, None))
            # Beams with insufficient transmission are treated as no-return
            range_img[transmission < _MIN_TRANSMISSION_FRACTION] = self.no_hit_value

        # 4. Max-range clipping
        valid_mask = (range_img > 0) & (range_img <= self.max_range_m)

        # 5. Random dropouts
        if self.dropout_prob > 0:
            dropout_mask = self._rng.random(range_img.shape) < self.dropout_prob
            valid_mask &= ~dropout_mask

        # 6. Intensity noise
        intensity_img = np.clip(
            intensity_img + self._rng.normal(0.0, self.intensity_noise_sigma, intensity_img.shape),
            0.0,
            1.0,
        ).astype(np.float32)

        # 7. Convert to Cartesian coordinates using pre-cached trig arrays.
        # Slice to the actual input size in case range_image is smaller than
        # the configured (n_channels, h_resolution).
        cos_elev = self._cos_elev[:n_ch, :n_az]
        sin_elev = self._sin_elev[:n_ch, :n_az]
        cos_azim = self._cos_azim[:n_ch, :n_az]
        sin_azim = self._sin_azim[:n_ch, :n_az]

        r = range_img
        x = r * cos_elev * cos_azim
        y = r * cos_elev * sin_azim
        z = r * sin_elev

        # 8. Pack into Nx4 array
        points: FloatArray = np.stack(
            [x[valid_mask], y[valid_mask], z[valid_mask], intensity_img[valid_mask]],
            axis=-1,
        ).astype(np.float32)

        result: LidarObservation = {"points": points, "range_image": range_img}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_beam_divergence(self, range_img: "FloatArray") -> "FloatArray":
        """Apply a separable Gaussian blur to model beam-divergence mixed pixels.

        The sigma is pre-computed at init from ``beam_divergence_mrad`` and the
        sensor's angular resolution.  Uses scipy when available (exact Gaussian)
        and falls back to numpy separable 1-D box-filter convolutions otherwise.
        """
        # Clamp sigma to at most half the image dimension to avoid edge artefacts.
        n_ch, n_az = range_img.shape
        sigma_el = min(self._bd_sigma_el, (n_ch - 1) * 0.5)
        sigma_az = min(self._bd_sigma_az, (n_az - 1) * 0.5)

        try:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(range_img, sigma=(sigma_el, sigma_az)).astype(np.float32)
        except ImportError:
            # Separable 1-D box-filter approximation.
            def _box1d(arr: "FloatArray", sigma: float, axis: int) -> "FloatArray":
                k = max(1, int(sigma * 2 + 1))
                kernel = np.ones(k, dtype=np.float32) / k
                return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=axis, arr=arr)

            blurred = _box1d(range_img, sigma_el, axis=0)
            blurred = _box1d(blurred, sigma_az, axis=1)
            return blurred.astype(np.float32)
