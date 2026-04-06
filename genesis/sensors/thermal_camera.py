"""
Thermal / IR camera model.

Converts ideal scene data (segmentation mask, entity states) into a
synthetic thermal image.  The model is intentionally approximate so that
it can run without modifying Genesis internals; physical accuracy can be
improved incrementally.

Pipeline
--------
1. Assign a surface temperature to every pixel from entity metadata.
2. Apply a Gaussian PSF to simulate thermal optics blur.
3. Add non-uniformity correction (NUC) defects and Gaussian detector noise.
4. Optionally apply a fog / atmospheric attenuation mask.
5. Quantise to a given bit depth.

The caller must provide a ``temperature_map`` (a per-entity dict mapping
entity ID to temperature in degrees Celsius) together with the segmentation
image rendered by Genesis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import FloatArray, Int32Array, ThermalObservation, UInt16Array, UInt8Array

if TYPE_CHECKING:
    from .config import ThermalCameraConfig

# Number of dimensions for a 3-D image array (H, W, C).
_NDIM_3D: Final[int] = 3
# Typical LWIR sensor range in degrees Celsius.
_DEFAULT_LWIR_TEMP_MIN_C: Final[float] = -20.0
_DEFAULT_LWIR_TEMP_MAX_C: Final[float] = 140.0
# Bit-depth boundary below which uint8 is used for output.
_UINT8_MAX_BIT_DEPTH: Final[int] = 8

# ---------------------------------------------------------------------------
# Optional scipy import -- attempted once at module load to avoid repeated
# try/except blocks in the hot step() path.
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter

    _SCIPY_AVAILABLE: Final[bool] = True
except ImportError:
    _scipy_gaussian_filter = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE: Final[bool] = False  # type: ignore[misc]


class ThermalCameraModel(BaseSensor):
    """
    Synthetic thermal / IR camera sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Frame rate in Hz.
    resolution:
        ``(width, height)`` in pixels.
    temp_ambient_c:
        Default ambient temperature in degrees C assigned to pixels with no
        entity assignment (background).
    temp_sky_c:
        Temperature assigned to sky / open-air background pixels.
    psf_sigma:
        Standard deviation of the Gaussian optics PSF in pixels.
        Set to ``0`` to disable blurring.
    nuc_sigma:
        Standard deviation of the per-pixel gain non-uniformity offset
        (in degrees C).  Applied once at construction; represents sensor NUC
        residual errors.
    noise_sigma:
        Standard deviation of per-frame Gaussian detector noise (in degrees C).
    bit_depth:
        Output bit depth (8 or 14 are typical for thermal cameras).
    fog_density:
        Exponential fog attenuation coefficient (1/m).  0 = no fog.
    temp_range_c:
        ``(t_min, t_max)`` of the quantisation range in degrees C.  Pixels
        outside this range are clipped.  Defaults to the standard LWIR
        operating range (-20, 140).
    seed:
        Optional seed for the random-number generator (reproducibility).
    """

    SKY_ENTITY_ID: Final[int] = -1  # sentinel value for background / sky pixels

    def __init__(
        self,
        name: str = "thermal_camera",
        update_rate_hz: float = 9.0,
        resolution: tuple[int, int] = (320, 240),
        temp_ambient_c: float = 20.0,
        temp_sky_c: float = -30.0,
        psf_sigma: float = 1.0,
        nuc_sigma: float = 0.5,
        noise_sigma: float = 0.05,
        bit_depth: int = 14,
        fog_density: float = 0.0,
        temp_range_c: tuple[float, float] = (_DEFAULT_LWIR_TEMP_MIN_C, _DEFAULT_LWIR_TEMP_MAX_C),
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.resolution = tuple(resolution)
        self.temp_ambient_c = float(temp_ambient_c)
        self.temp_sky_c = float(temp_sky_c)
        self.psf_sigma = float(psf_sigma)
        self.nuc_sigma = float(nuc_sigma)
        self.noise_sigma = float(noise_sigma)
        self.bit_depth = int(bit_depth)
        self.fog_density = float(fog_density)
        self.temp_range_c = (float(temp_range_c[0]), float(temp_range_c[1]))

        # Per-pixel NUC offset -- fixed for the sensor lifetime
        self._rng = np.random.default_rng(seed=seed)
        w, h = self.resolution
        self._nuc_offset: FloatArray = self._rng.normal(0.0, self.nuc_sigma, (h, w)).astype(np.float32)

        # ------------------------------------------------------------------
        # Pre-computed quantisation constants
        # ------------------------------------------------------------------
        t_min, t_max = self.temp_range_c
        self._quantise_levels: int = 2**self.bit_depth
        # Avoid division in _quantise() by pre-computing the reciprocal scale.
        t_range = t_max - t_min
        self._quantise_inv_range: float = 1.0 / t_range if t_range != 0.0 else 1.0
        self._quantise_t_min: float = t_min
        self._quantise_dtype: type[np.uint8] | type[np.uint16] = (
            np.uint8 if self.bit_depth <= _UINT8_MAX_BIT_DEPTH else np.uint16
        )

        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "ThermalCameraConfig") -> "ThermalCameraModel":
        """Construct a :class:`ThermalCameraModel` from a :class:`~genesis.sensors.config.ThermalCameraConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "ThermalCameraConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.ThermalCameraConfig`."""
        from .config import ThermalCameraConfig

        return ThermalCameraConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            resolution=self.resolution,
            temp_ambient_c=self.temp_ambient_c,
            temp_sky_c=self.temp_sky_c,
            psf_sigma=self.psf_sigma,
            nuc_sigma=self.nuc_sigma,
            noise_sigma=self.noise_sigma,
            bit_depth=self.bit_depth,
            fog_density=self.fog_density,
            temp_range_c=self.temp_range_c,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> ThermalObservation | dict[str, Any]:
        """
        Produce a synthetic thermal image.

        Expected keys in *state*:
        - ``"seg"`` -- ``np.ndarray`` shape ``(H, W)`` or ``(H, W, 1)``
          containing integer entity IDs (as rendered by Genesis
          ``cam.render(segmentation=True)``).
        - ``"temperature_map"`` -- ``dict[int, float]`` mapping entity ID
          to surface temperature in degrees C.  Missing entity IDs fall back
          to ``temp_ambient_c``.
        - ``"depth"`` *(optional)* -- ``np.ndarray`` shape ``(H, W)``
          containing per-pixel depth in metres; used for fog attenuation.
        """
        seg_raw = state.get("seg")
        if seg_raw is None:
            self._last_obs = {}
            return self._last_obs

        seg: Int32Array = np.asarray(seg_raw, dtype=np.int32)
        if seg.ndim == _NDIM_3D:
            seg = seg[..., 0]

        temp_map: dict[int, float] = state.get("temperature_map", {})

        # 1. Build temperature image
        temp_img: FloatArray = np.full(seg.shape, self.temp_ambient_c, dtype=np.float32)
        for entity_id, temp in temp_map.items():
            temp_img[seg == entity_id] = float(temp)
        # Sky pixels
        temp_img[seg == self.SKY_ENTITY_ID] = self.temp_sky_c

        # 2. Fog attenuation (hotter objects appear cooler when far away)
        if self.fog_density > 0:
            depth_raw = state.get("depth")
            if depth_raw is not None:
                depth_arr: FloatArray = np.asarray(depth_raw, dtype=np.float32)
                if depth_arr.ndim == _NDIM_3D:
                    depth_arr = depth_arr[..., 0]
                attenuation: FloatArray = np.exp(-self.fog_density * np.clip(depth_arr, 0, None))
                temp_img = temp_img * attenuation + self.temp_ambient_c * (1.0 - attenuation)

        # 3. PSF blur
        if self.psf_sigma > 0:
            temp_img = self._gaussian_blur(temp_img, self.psf_sigma)

        # 4. NUC defects + detector noise
        h, w = temp_img.shape
        nuc = self._nuc_offset[:h, :w]
        noise: FloatArray = self._rng.normal(0.0, self.noise_sigma, (h, w)).astype(np.float32)
        temp_img = temp_img + nuc + noise

        # 5. Quantise
        thermal_raw = self._quantise(temp_img)

        result: ThermalObservation = {"thermal": thermal_raw, "temperature_c": temp_img}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_blur(img: FloatArray, sigma: float) -> FloatArray:
        """Apply a Gaussian blur with standard deviation *sigma* pixels.

        Uses the module-level ``_scipy_gaussian_filter`` import (resolved once
        at import time) to avoid repeated try/except overhead per frame.
        Falls back to a separable box-filter approximation when scipy is
        absent — note that a box filter is **not** a true Gaussian blur;
        it is provided only as a graceful degradation path.
        """
        if _SCIPY_AVAILABLE and _scipy_gaussian_filter is not None:
            return _scipy_gaussian_filter(img, sigma=sigma).astype(np.float32)
        # Box-filter fallback via separable 1-D convolutions (numpy only).
        # Previous code had a dead `if _SCIPY_AVAILABLE` guard here that
        # always evaluated False, causing the unblurred image to be returned.
        k = max(1, int(sigma * 2 + 1))
        box = np.ones(k, dtype=np.float32) / k
        out = np.apply_along_axis(lambda r: np.convolve(r, box, mode="same"), axis=1, arr=img)
        out = np.apply_along_axis(lambda r: np.convolve(r, box, mode="same"), axis=0, arr=out)
        return out.astype(np.float32)

    def _quantise(self, temp_img: FloatArray) -> UInt8Array | UInt16Array:
        """Map temperature to raw sensor counts using a linear scale.

        Uses pre-computed constants (_quantise_t_min, _quantise_inv_range,
        _quantise_levels, _quantise_dtype) to avoid division in the hot path.
        """
        raw = np.clip((temp_img - self._quantise_t_min) * self._quantise_inv_range, 0.0, 1.0) * (
            self._quantise_levels - 1
        )
        return raw.astype(self._quantise_dtype)
