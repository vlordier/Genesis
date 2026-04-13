import inspect
import os
import time

import numpy as np
import quadrants as qd
from PIL import Image

import genesis as gs


def animate(imgs, filename=None, fps=60):
    """
    Create a video from a list of images.

    Images must be uint8 arrays of shape ``(H, W, 3)`` (RGB), ``(H, W, 4)`` (RGBA,
    alpha stripped automatically), or ``(H, W)`` (grayscale). PIL Images are also
    accepted. Float arrays are *not* supported; convert to uint8 before calling.

    Args:
        imgs (list): List of input images (numpy arrays or PIL Images).
        filename (str, optional): Output video path (.mp4). Defaults to
            ``<caller_script>_<timestamp>.mp4`` in the current directory.
        fps (int, optional): Frames per second. Defaults to 60.
    """
    assert isinstance(imgs, list)
    if len(imgs) == 0:
        gs.logger.warning("No image to save.")
        return

    fps = max(1, int(round(fps)))  # PyAV requires an integer time-base denominator

    if filename is None:
        caller_file = inspect.stack()[-1].filename
        # caller file + timestamp + .mp4
        filename = os.path.splitext(os.path.basename(caller_file))[0] + f"_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    os.makedirs(os.path.abspath(os.path.dirname(filename) or "."), exist_ok=True)

    gs.logger.info(f'Saving video to ~<"{filename}">~...')

    _av_ok = False
    try:
        import av

        # libx264 must be compiled into this PyAV build; fall back to moviepy otherwise.
        if "libx264" not in av.codecs_available:
            raise ImportError("PyAV build does not include libx264")

        first = imgs[0]
        if not isinstance(first, np.ndarray):
            first = np.array(first)
        # Strip alpha channel if present; libx264/yuv420p only accepts RGB or grayscale.
        if first.ndim == 3 and first.shape[2] == 4:
            first = first[..., :3]
        height, width = first.shape[:2]
        is_color = first.ndim == 3 and first.shape[2] == 3
        fmt = "rgb24" if is_color else "gray"

        container = av.open(filename, mode="w")
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.options = {"preset": "ultrafast"}

            for img in imgs:
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                img = img.astype(np.uint8)
                # Strip alpha channel for consistency with `first`.
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[..., :3]
                # from_ndarray handles stride/padding internally, avoiding the
                # line_size // channels reshape bug for non-aligned widths.
                frame = av.VideoFrame.from_ndarray(img, format=fmt)
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)
        finally:
            container.close()

        _av_ok = True

    except ImportError as exc:
        gs.logger.warning(
            f"PyAV unavailable ({exc}); falling back to moviepy. "
            "Note: moviepy ≥ 2.x may drop the last frame. Install 'av' for reliable output."
        )

    if not _av_ok:
        from moviepy import ImageSequenceClip

        clip = ImageSequenceClip(imgs, fps=fps)
        clip.write_videofile(filename, fps=fps, logger=None, codec="libx264", preset="ultrafast")

    gs.logger.info("Video saved.")


def save_img_arr(arr, filename="img.png"):
    assert isinstance(arr, np.ndarray)
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    img = Image.fromarray(arr)
    img.save(filename)
    gs.logger.info(f"Image saved to ~<{filename}>~.")


class Timer:
    def __init__(self, skip=False, level=0, qd_sync=False):
        self.accu_log = dict()
        self.skip = skip
        self.level = level
        self.qd_sync = qd_sync
        self.msg_width = 0
        self.reset()

    def reset(self):
        self.just_reset = True
        if self.level == 0 and not self.skip:
            try:
                column, _lines = os.get_terminal_size()
            except OSError:
                column = 80
            print("─" * column)
        if self.qd_sync and not self.skip:
            qd.sync()
        self.prev_time = self.init_time = time.perf_counter()

    def _stamp(self, msg="", _ratio=1.0):
        if self.skip:
            return

        if self.qd_sync:
            qd.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1] / self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2] / self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False

    def stamp(self, msg="", _ratio=1.0):
        return
        if self.skip:
            return

        if self.qd_sync:
            qd.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1] / self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2] / self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False


timers = dict()


def create_timer(name=None, new=False, level=0, qd_sync=False, skip_first_call=False):
    if name is None:
        return Timer()
    else:
        if name in timers and not new:
            timer = timers[name]
            timer.skip = False
            timer.reset()
            return timer
        else:
            timer = Timer(skip=skip_first_call, level=level, qd_sync=qd_sync)
            timers[name] = timer
            return timer


class Rate:
    def __init__(self, rate):
        self.rate = rate
        self.last_time = time.perf_counter()

    def sleep(self):
        current_time = time.perf_counter()
        sleep_duration = 1.0 / self.rate - (current_time - self.last_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.last_time = time.perf_counter()


class FPSTracker:
    def __init__(
        self, n_envs, alpha=0.95, minimum_interval_seconds: float | None = 0.05, outlier_threshold: float = 1.5
    ):
        self.last_time = None
        self.n_envs = n_envs
        self.dt_ema = None
        self.alpha = alpha
        self.minimum_interval_seconds = minimum_interval_seconds
        self.outlier_threshold = outlier_threshold
        self.steps_since_last_print: int = 0
        self.total_fps = 0.0

    def step(self, current_time: float | None = None) -> float | None:
        if not current_time:
            current_time = time.perf_counter()

        if self.last_time:
            dt = current_time - self.last_time
        else:
            self.last_time = current_time
            return None

        self.steps_since_last_print += 1

        # Skip if update is too soon
        if self.minimum_interval_seconds and current_time - self.last_time < self.minimum_interval_seconds:
            return None

        # Outlier rejection
        if self.dt_ema is not None:
            if dt > self.dt_ema * self.outlier_threshold or dt * self.outlier_threshold < self.dt_ema:
                self.dt_ema = dt

        # EMA update
        if self.dt_ema:
            self.dt_ema = self.alpha * self.dt_ema + (1 - self.alpha) * dt
        else:
            self.dt_ema = dt

        fps = 1 / self.dt_ema * self.steps_since_last_print
        if self.n_envs > 0:
            self.total_fps = fps * self.n_envs
            gs.logger.info(
                f"Running at ~<{self.total_fps:,.2f}>~ FPS (~<{fps:.2f}>~ FPS per env, ~<{self.n_envs}>~ envs)."
            )
        else:
            self.total_fps = fps
            gs.logger.info(f"Running at ~<{fps:.2f}>~ FPS.")
        self.last_time = current_time
        self.steps_since_last_print = 0
        return self.total_fps
