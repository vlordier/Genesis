"""
RGB camera corruption pipeline.

Takes an *ideal* RGB (and optionally depth/normal) image rendered by Genesis
and applies a realistic cascade of sensor artefacts:

1. Lens distortion (radial + tangential)
2. Rolling-shutter smear
3. Motion blur
4. Exposure / gain mapping
5. Vignetting (radial intensity fall-off)
6. Lateral chromatic aberration (channel shift)
7. Shot noise + read noise (Poisson + Gaussian)
8. Dead / hot pixels
9. JPEG-style compression artefacts (optional)

All operations are applied on NumPy ``uint8`` / ``float32`` arrays so that
the model remains backend-agnostic and can run without a GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import CameraObservation, FloatArray, UInt8Array

if TYPE_CHECKING:
    from .config import CameraConfig

# Sensor-level constants
_UINT8_MAX: Final[int] = 255
_FLOAT_CLIP_MAX: Final[float] = 1.0
_FLOAT_CLIP_MIN: Final[float] = 0.0
# Default full-well capacity (electrons) at base ISO
_DEFAULT_FULL_WELL_ELECTRONS: Final[float] = 3500.0


def _bilinear_sample(channel: "FloatArray", xf: "FloatArray", yf: "FloatArray") -> "FloatArray":
    """Bilinear interpolation of a 2-D single-channel image at fractional positions.

    Parameters
    ----------
    channel:
        2-D float32 array of shape ``(H, W)``.
    xf, yf:
        Float arrays of shape ``(H, W)`` holding the fractional sample
        positions in pixel coordinates.  Out-of-bounds positions are
        clamped to the image boundary (replication padding).

    Returns
    -------
    FloatArray
        Interpolated values, same shape as *channel*.
    """
    h, w = channel.shape
    # Clamp to valid range
    xf = np.clip(xf, 0.0, w - 1)
    yf = np.clip(yf, 0.0, h - 1)

    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = (xf - x0).astype(np.float32)  # fractional x-weight
    wy = (yf - y0).astype(np.float32)  # fractional y-weight

    # Bilinear blend (vectorised, no loops)
    top = (1.0 - wx) * channel[y0, x0] + wx * channel[y0, x1]
    bot = (1.0 - wx) * channel[y1, x0] + wx * channel[y1, x1]
    return ((1.0 - wy) * top + wy * bot).astype(np.float32)


class CameraModel(BaseSensor):
    """
    Realistic RGB camera sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Camera frame rate in Hz.
    resolution:
        ``(width, height)`` in pixels.
    distortion_coeffs:
        OpenCV-style distortion coefficients ``(k1, k2, p1, p2[, k3])``.
        Pass an empty tuple or ``None`` to skip distortion.
    rolling_shutter_fraction:
        Fraction of the frame period over which the sensor is actively
        read out (0 - global shutter, 1 - full rolling shutter).
    motion_blur_kernel:
        Half-length of the 1-D motion-blur kernel (0 = disabled).
    base_iso:
        Reference ISO value used for noise scaling.
    iso:
        Effective ISO.  Higher values increase noise.
    read_noise_sigma:
        Standard deviation of Gaussian read noise (in *electron* counts).
    dead_pixel_fraction:
        Fraction of permanently-dead (zero-output) pixels.  Applied once
        at construction time.
    hot_pixel_fraction:
        Fraction of permanently-saturated pixels.  Applied once at
        construction time.
    jpeg_quality:
        If ``> 0`` and ``opencv-python`` is available, apply JPEG
        compression with this quality level (1-100).
    full_well_electrons:
        Full-well capacity at *base_iso* in electron counts.  Scales the
        Poisson shot-noise model.  Default is 3500 e-.
    vignetting_strength:
        Radial vignetting coefficient.  ``0`` = disabled; ``0.5`` = moderate
        (corners ~25% dimmer); ``1.0`` = strong (corners ~50% dimmer).
        Applied as a precomputed ``cos⁴``-like polynomial map.
    chromatic_aberration_px:
        Lateral chromatic aberration: maximum radial channel shift in pixels
        at the image corner (0 = disabled).  Red shifts inward by
        ``chromatic_aberration_px / 2`` pixels and blue shifts outward by the
        same amount, relative to the green reference channel.
    seed:
        Optional seed for the random-number generator (reproducibility).
    """

    def __init__(
        self,
        name: str = "rgb_camera",
        update_rate_hz: float = 30.0,
        resolution: tuple[int, int] = (640, 480),
        distortion_coeffs: tuple[float, ...] | None = None,
        rolling_shutter_fraction: float = 0.0,
        motion_blur_kernel: int = 0,
        base_iso: float = 100.0,
        iso: float = 100.0,
        read_noise_sigma: float = 1.5,
        dead_pixel_fraction: float = 0.0001,
        hot_pixel_fraction: float = 0.00005,
        jpeg_quality: int = 0,
        full_well_electrons: float = _DEFAULT_FULL_WELL_ELECTRONS,
        vignetting_strength: float = 0.0,
        chromatic_aberration_px: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.resolution = tuple(resolution)
        self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float32) if distortion_coeffs else None
        self.rolling_shutter_fraction = float(np.clip(rolling_shutter_fraction, 0.0, 1.0))
        self.motion_blur_kernel = int(max(0, motion_blur_kernel))
        self.base_iso = float(base_iso)
        self.iso = float(iso)
        self.read_noise_sigma = float(read_noise_sigma)
        self.jpeg_quality = int(jpeg_quality)
        self.full_well_electrons = float(full_well_electrons)
        self.vignetting_strength = float(max(0.0, vignetting_strength))
        self.chromatic_aberration_px = float(max(0.0, chromatic_aberration_px))

        self.dead_pixel_fraction = float(dead_pixel_fraction)
        self.hot_pixel_fraction = float(hot_pixel_fraction)

        w, h = self.resolution
        self._rng = np.random.default_rng(seed=seed)
        n_pixels = w * h
        self._dead_mask = self._rng.random(n_pixels) < self.dead_pixel_fraction
        self._hot_mask = self._rng.random(n_pixels) < self.hot_pixel_fraction

        # ------------------------------------------------------------------
        # Pre-computed constants (avoid repeated arithmetic inside step())
        # ------------------------------------------------------------------

        # Exposure gain: multiplied onto the float image every frame.
        self._gain: float = self.iso / self.base_iso
        # Full-well capacity at effective ISO: used in the shot-noise model.
        self._max_electrons: float = self.full_well_electrons * (self.base_iso / self.iso)

        # Rolling-shutter blend weights, shape (H, 1, 1).  Pre-computed for
        # the configured resolution height; recomputed lazily if the actual
        # input height differs.
        self._rs_h: int = h
        self._rs_alphas: FloatArray = np.linspace(0.0, self.rolling_shutter_fraction, h, dtype=np.float32).reshape(
            h, 1, 1
        )

        # Motion-blur 1-D kernel, shape (1, k).  Only materialised when
        # motion_blur_kernel > 0.
        if self.motion_blur_kernel > 0:
            k = self.motion_blur_kernel * 2 + 1
            self._blur_kernel: FloatArray | None = np.ones((1, k), dtype=np.float32) / k
        else:
            self._blur_kernel = None

        # Lens distortion remapping maps (cv2).  Computed once here to avoid
        # the expensive initUndistortRectifyMap call on every frame.
        self._dist_map1: np.ndarray | None = None
        self._dist_map2: np.ndarray | None = None
        if self.distortion_coeffs is not None and len(self.distortion_coeffs) > 0:
            try:
                import cv2

                fw, fh = self.resolution
                focal = float(max(fw, fh))
                cx, cy = fw / 2.0, fh / 2.0
                camera_matrix = np.array(
                    [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                self._dist_map1, self._dist_map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    self.distortion_coeffs,
                    None,
                    camera_matrix,
                    (fw, fh),
                    cv2.CV_32FC1,
                )
            except ImportError:
                pass  # graceful degradation — distortion skipped

        # ------------------------------------------------------------------
        # Vignetting: precompute a (H, W, 1) multiplicative map.
        # Uses a cos⁴-like polynomial model: v(r) = max(0, 1 − strength · r²)
        # where r is the normalised radial distance (0 at centre, 1 at the
        # farthest corner mid-point).  Precomputed at init and skipped when
        # vignetting_strength == 0.
        # ------------------------------------------------------------------
        self._vignette_map: FloatArray | None = None
        if self.vignetting_strength > 0.0:
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
            xg, yg = np.meshgrid(xs, ys)  # (H, W)
            r2: FloatArray = xg**2 + yg**2  # 0 at centre, 2 at corner
            # Normalise so the farthest corner has r2 = 1 (at (1,1)).
            r2_norm = r2 * 0.5
            self._vignette_map = np.clip(1.0 - self.vignetting_strength * r2_norm, 0.0, 1.0).reshape(h, w, 1)

        # ------------------------------------------------------------------
        # Chromatic aberration: precompute normalised radial-direction arrays.
        # During step() the red channel is shifted inward and the blue channel
        # outward by chromatic_aberration_px pixels at the image corner.
        # The green channel is the reference and is not shifted.
        # ------------------------------------------------------------------
        self._ca_dx: FloatArray | None = None
        self._ca_dy: FloatArray | None = None
        self._ca_r_max: float = 1.0
        if self.chromatic_aberration_px > 0.0:
            xs_ca = np.arange(w, dtype=np.float32) - (w - 1) * 0.5
            ys_ca = np.arange(h, dtype=np.float32) - (h - 1) * 0.5
            xg_ca, yg_ca = np.meshgrid(xs_ca, ys_ca)  # (H, W)
            # Normalise so the corner has magnitude 1.
            r_max = float(np.sqrt(((w - 1) * 0.5) ** 2 + ((h - 1) * 0.5) ** 2))
            self._ca_r_max = r_max if r_max > 0 else 1.0
            self._ca_dx = xg_ca / self._ca_r_max  # [-1, 1] approx
            self._ca_dy = yg_ca / self._ca_r_max  # [-1, 1] approx

        self._last_obs: dict[str, Any] = {}
        self._prev_frame: FloatArray | None = None  # for rolling-shutter / blur

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "CameraConfig") -> "CameraModel":
        """Construct a :class:`CameraModel` from a :class:`~genesis.sensors.config.CameraConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "CameraConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.CameraConfig`."""
        from .config import CameraConfig

        return CameraConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            resolution=self.resolution,
            distortion_coeffs=(tuple(self.distortion_coeffs.tolist()) if self.distortion_coeffs is not None else None),
            rolling_shutter_fraction=self.rolling_shutter_fraction,
            motion_blur_kernel=self.motion_blur_kernel,
            base_iso=self.base_iso,
            iso=self.iso,
            read_noise_sigma=self.read_noise_sigma,
            dead_pixel_fraction=self.dead_pixel_fraction,
            hot_pixel_fraction=self.hot_pixel_fraction,
            jpeg_quality=self.jpeg_quality,
            full_well_electrons=self.full_well_electrons,
            vignetting_strength=self.vignetting_strength,
            chromatic_aberration_px=self.chromatic_aberration_px,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._prev_frame = None
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> CameraObservation | dict[str, Any]:
        """
        Apply the full corruption pipeline to ``state["rgb"]``.

        Expected keys in *state*:
        - ``"rgb"`` -- ``np.ndarray`` shape ``(H, W, 3)`` dtype ``uint8`` or
          ``float32`` (will be normalised to ``[0, 1]``).
        - ``"pose_history"`` *(optional)* -- list of recent poses used for
          rolling-shutter simulation.
        """
        rgb = state.get("rgb")
        if rgb is None:
            self._last_obs = {}
            return self._last_obs

        img = self._to_float(rgb)

        # 1. Lens distortion
        img = self._apply_distortion(img)

        # 2. Rolling shutter (smear between prev and current frame)
        if self.rolling_shutter_fraction > 0 and self._prev_frame is not None:
            img = self._apply_rolling_shutter(img, self._prev_frame)

        # 3. Motion blur
        if self._blur_kernel is not None:
            img = self._apply_motion_blur(img)

        # 4. Exposure / gain (use pre-computed scalar)
        img = np.clip(img * self._gain, _FLOAT_CLIP_MIN, _FLOAT_CLIP_MAX)

        # 4a. Vignetting (applied after gain, before noise)
        if self._vignette_map is not None:
            h, w = img.shape[:2]
            vm = self._vignette_map
            if vm.shape[:2] != (h, w):
                # Input resolution differs from configured — skip.
                pass
            else:
                img = img * vm

        # 4b. Lateral chromatic aberration (before noise so CA edges look natural)
        if self._ca_dx is not None and self._ca_dy is not None:
            img = self._apply_chromatic_aberration(img)

        # 5. Noise (Poisson shot + Gaussian read)
        img = self._apply_noise(img)

        # 6. Dead / hot pixels
        img = self._apply_fixed_pattern_noise(img)

        # 7. JPEG artefacts
        img = self._apply_jpeg(img)

        self._prev_frame = img.copy()
        result: CameraObservation = {"rgb": self._to_uint8(img)}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(img: np.ndarray) -> FloatArray:
        """Normalise an image to float32 in [0, 1].

        Uses dtype to decide whether to scale (uint8 → divide by 255) rather
        than ``img.max() > 1``, which is unreliable for all-dark or corrupted
        images.
        """
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / _UINT8_MAX
        return arr.astype(np.float32)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> UInt8Array:
        return (np.clip(img, _FLOAT_CLIP_MIN, _FLOAT_CLIP_MAX) * _UINT8_MAX).astype(np.uint8)

    def _apply_distortion(self, img: FloatArray) -> FloatArray:
        """Apply Brown-Conrady radial + tangential lens distortion.

        Uses pre-computed remapping maps when available (computed at
        construction time).  Falls back gracefully when cv2 is absent or
        ``distortion_coeffs`` was not provided.
        """
        if self._dist_map1 is None or self._dist_map2 is None:
            return img
        try:
            import cv2

            return cv2.remap(img, self._dist_map1, self._dist_map2, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            return img

    def _apply_rolling_shutter(self, img: FloatArray, prev: FloatArray) -> FloatArray:
        """
        Blend previous and current frame row-by-row to simulate rolling
        shutter.

        Uses a vectorised numpy broadcast.  The alpha weights are pre-computed
        for the configured height and recomputed lazily if the actual input
        height differs.
        """
        h = img.shape[0]
        if h != self._rs_h:
            # Input height differs from configured — recompute alphas.
            self._rs_h = h
            self._rs_alphas = np.linspace(0.0, self.rolling_shutter_fraction, h, dtype=np.float32).reshape(h, 1, 1)
        alphas = self._rs_alphas
        return ((1.0 - alphas) * img + alphas * prev).astype(np.float32)

    def _apply_motion_blur(self, img: FloatArray) -> FloatArray:
        """Simple horizontal motion-blur using the pre-built uniform 1-D kernel."""
        assert self._blur_kernel is not None
        try:
            import cv2

            out = cv2.filter2D(img, -1, self._blur_kernel)
        except ImportError:
            from scipy.ndimage import convolve1d

            out = convolve1d(img, self._blur_kernel[0], axis=1, mode="reflect")
        return out.astype(np.float32)

    def _apply_noise(self, img: FloatArray) -> FloatArray:
        """Add Poisson shot noise scaled by ISO and Gaussian read noise."""
        # _max_electrons is pre-computed at init: full_well * (base_iso / iso)
        electrons = img * self._max_electrons
        # Shot noise (Poisson)
        shot = self._rng.poisson(np.clip(electrons, 0, None)).astype(np.float32)
        # Read noise
        read = self._rng.normal(0.0, self.read_noise_sigma, img.shape).astype(np.float32)
        return np.clip((shot + read) / self._max_electrons, _FLOAT_CLIP_MIN, _FLOAT_CLIP_MAX)

    def _apply_fixed_pattern_noise(self, img: FloatArray) -> FloatArray:
        """Apply permanently dead (black) and hot (white) pixels."""
        h, w = img.shape[:2]
        n = h * w
        # The masks were built for the configured (w×h) resolution.  If the
        # input image has a different pixel count, boolean indexing would raise
        # an IndexError (when larger) or silently apply a partial mask (when
        # smaller).  Skip noise application when resolution differs; the caller
        # is responsible for passing correctly-sized images.
        if n != len(self._dead_mask):
            return img
        flat = img.reshape(n, -1)
        flat[self._dead_mask] = _FLOAT_CLIP_MIN
        flat[self._hot_mask] = _FLOAT_CLIP_MAX
        return flat.reshape(img.shape)

    def _apply_jpeg(self, img: FloatArray) -> FloatArray:
        """Encode to JPEG and decode back to simulate compression artefacts."""
        if self.jpeg_quality <= 0:
            return img
        try:
            import cv2

            bgr = cv2.cvtColor(self._to_uint8(img), cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.float32) / _UINT8_MAX
        except ImportError:
            return img

    def _apply_chromatic_aberration(self, img: FloatArray) -> FloatArray:
        """Apply lateral chromatic aberration by shifting R/B channels radially.

        The green channel (index 1) is the reference and is unchanged.
        The red channel is shifted *inward* by ``chromatic_aberration_px / 2``
        pixels at the image corner; the blue channel is shifted *outward* by
        the same amount.

        Uses pure-NumPy bilinear interpolation so that cv2 is not required.
        The precomputed ``_ca_dx``/``_ca_dy`` arrays hold normalised radial
        direction vectors (magnitude 1 at the image corner).
        """
        assert self._ca_dx is not None and self._ca_dy is not None
        h, w = img.shape[:2]
        ca_h, ca_w = self._ca_dx.shape
        if (h, w) != (ca_h, ca_w):
            # Resolution differs from the precomputed maps — skip CA.
            return img

        half_shift = self.chromatic_aberration_px * 0.5

        # Integer pixel grid
        xi = np.arange(w, dtype=np.float32)
        yi = np.arange(h, dtype=np.float32)
        xg, yg = np.meshgrid(xi, yi)  # (H, W)

        out = img.copy()
        # channel 0 = red: shift inward (-half_shift)
        # channel 2 = blue: shift outward (+half_shift)
        for ch_idx, shift in ((0, -half_shift), (2, half_shift)):
            xf = xg + shift * self._ca_dx
            yf = yg + shift * self._ca_dy
            out[..., ch_idx] = _bilinear_sample(img[..., ch_idx], xf, yf)
        return out


__all__ = ["CameraModel"]
