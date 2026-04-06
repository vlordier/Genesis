"""
RGB camera corruption pipeline.

Takes an *ideal* RGB (and optionally depth/normal) image rendered by Genesis
and applies a realistic cascade of sensor artefacts:

1. Lens distortion (radial + tangential)
2. Rolling-shutter smear
3. Motion blur
4. Exposure / gain mapping
5. Shot noise + read noise (Poisson + Gaussian)
6. Dead / hot pixels
7. JPEG-style compression artefacts (optional)

All operations are applied on NumPy ``uint8`` / ``float32`` arrays so that
the model remains backend-agnostic and can run without a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseSensor


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
        read out (0 → global shutter, 1 → full rolling shutter).
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
        compression with this quality level (1–100).
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

        w, h = self.resolution
        rng = np.random.default_rng(seed=seed)
        n_pixels = w * h
        self._dead_mask = rng.random(n_pixels) < dead_pixel_fraction
        self._hot_mask = rng.random(n_pixels) < hot_pixel_fraction

        self._last_obs: dict[str, Any] = {}
        self._prev_frame: np.ndarray | None = None  # for rolling-shutter / blur

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._prev_frame = None
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the full corruption pipeline to ``state["rgb"]``.

        Expected keys in *state*:
        - ``"rgb"`` – ``np.ndarray`` shape ``(H, W, 3)`` dtype ``uint8`` or
          ``float32`` (will be normalised to ``[0, 1]``).
        - ``"pose_history"`` *(optional)* – list of recent poses used for
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
            img = self._apply_rolling_shutter(img)

        # 3. Motion blur
        if self.motion_blur_kernel > 0:
            img = self._apply_motion_blur(img)

        # 4. Exposure / gain
        gain = self.iso / self.base_iso
        img = np.clip(img * gain, 0.0, 1.0)

        # 5. Noise (Poisson shot + Gaussian read)
        img = self._apply_noise(img)

        # 6. Dead / hot pixels
        img = self._apply_fixed_pattern_noise(img)

        # 7. JPEG artefacts
        img = self._apply_jpeg(img)

        self._prev_frame = img.copy()
        result = {"rgb": self._to_uint8(img)}
        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    def _apply_distortion(self, img: np.ndarray) -> np.ndarray:
        """Apply Brown-Conrady radial + tangential lens distortion."""
        if self.distortion_coeffs is None or len(self.distortion_coeffs) == 0:
            return img
        try:
            import cv2  # optional dependency

            h, w = img.shape[:2]
            # Build a simple camera matrix centred in the image
            fx = fy = max(w, h)
            cx, cy = w / 2.0, h / 2.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist = self.distortion_coeffs
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
            return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            return img  # graceful degradation when opencv is not available

    def _apply_rolling_shutter(self, img: np.ndarray) -> np.ndarray:
        """
        Blend previous and current frame row-by-row to simulate rolling
        shutter.  The top row is purely from the current frame; the bottom
        row is ``rolling_shutter_fraction * prev + (1-f) * current``.
        """
        h = img.shape[0]
        alphas = np.linspace(0.0, self.rolling_shutter_fraction, h, dtype=np.float32)
        blended = np.empty_like(img)
        prev = self._prev_frame
        for r in range(h):
            a = alphas[r]
            blended[r] = (1.0 - a) * img[r] + a * prev[r]
        return blended

    def _apply_motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Simple horizontal motion-blur using a uniform 1-D kernel."""
        k = self.motion_blur_kernel * 2 + 1
        kernel = np.ones((1, k), dtype=np.float32) / k
        try:
            import cv2

            out = cv2.filter2D(img, -1, kernel)
        except ImportError:
            from scipy.ndimage import convolve1d

            out = convolve1d(img, kernel[0], axis=1, mode="reflect")
        return out.astype(np.float32)

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Poisson shot noise scaled by ISO and Gaussian read noise."""
        # Photon-count domain: scale to ~[0, 3500] electron counts
        max_electrons = 3500.0 * (self.base_iso / self.iso)
        electrons = img * max_electrons
        # Shot noise (Poisson)
        shot = np.random.poisson(np.clip(electrons, 0, None)).astype(np.float32)
        # Read noise
        read = np.random.normal(0, self.read_noise_sigma, img.shape).astype(np.float32)
        return np.clip((shot + read) / max_electrons, 0.0, 1.0)

    def _apply_fixed_pattern_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply permanently dead (black) and hot (white) pixels."""
        h, w = img.shape[:2]
        flat = img.reshape(h * w, -1)
        flat[self._dead_mask[: h * w]] = 0.0
        flat[self._hot_mask[: h * w]] = 1.0
        return flat.reshape(img.shape)

    def _apply_jpeg(self, img: np.ndarray) -> np.ndarray:
        """Encode to JPEG and decode back to simulate compression artefacts."""
        if self.jpeg_quality <= 0:
            return img
        try:
            import cv2

            bgr = cv2.cvtColor(self._to_uint8(img), cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.float32) / 255.0
        except ImportError:
            return img
