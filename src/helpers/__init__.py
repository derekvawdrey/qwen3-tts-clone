"""Audio helpers used by the local streaming examples."""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional

import numpy as np


class _StreamingResampler:
    """Continuous linear resampler for streaming chunks.

    Carries the last input sample and fractional phase across calls so chunk
    boundaries stay click-free. Good enough for speech playback; not a
    high-fidelity anti-aliasing resampler.
    """

    def __init__(self, in_rate: int, out_rate: int):
        self.ratio = in_rate / out_rate  # input samples consumed per output sample
        self._carry: Optional[np.ndarray] = None  # last input sample, shape (1, ch)
        self._frac = 0.0  # position of next output sample, in input samples from buf[0]

    def process(self, x: np.ndarray) -> np.ndarray:
        # x: (n, ch) float32. Prepend the previous chunk's last sample so the
        # interpolation grid is continuous across calls.
        if self._carry is not None:
            buf = np.concatenate([self._carry, x], axis=0)
        else:
            buf = x
        m = buf.shape[0]
        if m < 2:
            self._carry = buf[-1:].copy()
            return np.zeros((0, x.shape[1]), dtype=np.float32)

        start = self._frac if self._carry is not None else 0.0
        last_in = m - 1
        # number of output samples whose source position stays within the buffer
        count = int(np.floor((last_in - start) / self.ratio)) + 1
        if count <= 0:
            self._carry = buf[-1:].copy()
            self._frac = start - last_in
            return np.zeros((0, x.shape[1]), dtype=np.float32)

        positions = start + np.arange(count) * self.ratio
        idx = np.floor(positions).astype(np.int64)
        a = (positions - idx)[:, None].astype(np.float32)
        out = buf[idx] * (1.0 - a) + buf[np.minimum(idx + 1, last_in)] * a

        next_pos = start + count * self.ratio
        self._frac = next_pos - last_in  # measured from next buf[0] (== current last sample)
        self._carry = buf[-1:].copy()
        return out.astype(np.float32)


class StreamPlayer:
    """Play streaming audio chunks through one persistent output stream."""

    def __init__(self, *, channels: int = 1, dtype: str = "float32", max_queue_chunks: int = 0,
                 device=None, ignore_pulse_sink: bool = False):
        self.channels = channels
        self.dtype = dtype
        self.max_queue_chunks = max_queue_chunks
        self.device = device  # None => env AUDIO_OUTPUT_DEVICE, else 'pulse', else system default
        # When True, open the stream with PULSE_SINK temporarily cleared so it binds
        # to the user's real default sink instead of a virtual-mic sink (used for the
        # headphone "monitor" player while TTS itself is routed into the virtual mic).
        self.ignore_pulse_sink = ignore_pulse_sink
        self._device = None

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue_chunks)
        self._pending = np.zeros((0, channels), dtype=np.float32)
        self._stream = None
        self._sample_rate: Optional[int] = None
        self._resampler: Optional[_StreamingResampler] = None
        self._closed = False
        self._drained = threading.Event()

    def _load_sounddevice(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "examples.audio.StreamPlayer requires the optional 'sounddevice' package. "
                "Install it with: pip install sounddevice"
            ) from exc
        return sd

    def _reshape_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio_chunk, dtype=np.float32)
        if arr.ndim == 1:
            if self.channels != 1:
                raise ValueError(f"Expected {self.channels} channels, got mono audio")
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            if arr.shape[1] != self.channels:
                raise ValueError(f"Expected {self.channels} channels, got {arr.shape[1]}")
            return arr
        raise ValueError(f"Expected 1D or 2D audio chunk, got shape {arr.shape}")

    def _callback(self, outdata, frames, _time, status):
        if status:
            pass

        written = 0
        while written < frames:
            if self._pending.shape[0] == 0:
                try:
                    next_chunk = self._queue.get_nowait()
                except queue.Empty:
                    outdata[written:] = 0
                    return

                if next_chunk is None:
                    outdata[written:] = 0
                    self._drained.set()
                    sd = self._load_sounddevice()
                    raise sd.CallbackStop()

                self._pending = next_chunk

            take = min(frames - written, self._pending.shape[0])
            outdata[written:written + take] = self._pending[:take]
            self._pending = self._pending[take:]
            written += take

    def _ensure_stream(self, sample_rate: int):
        if self._stream is not None:
            if sample_rate != self._sample_rate:
                raise ValueError(
                    f"StreamPlayer sample rate changed from {self._sample_rate} to {sample_rate}"
                )
            return

        sd = self._load_sounddevice()
        self._sample_rate = sample_rate
        self._device = self._resolve_device(sd)

        out_rate = self._pick_output_rate(sd, sample_rate)
        if out_rate != sample_rate:
            self._resampler = _StreamingResampler(sample_rate, out_rate)

        # Bind the stream's sink now. For the monitor player, drop PULSE_SINK for
        # the duration of the open so PulseAudio routes it to the real default
        # output (headphones) rather than the virtual-mic sink.
        prev_sink = os.environ.pop("PULSE_SINK", None) if self.ignore_pulse_sink else None
        try:
            self._stream = sd.OutputStream(
                samplerate=out_rate,
                device=self._device,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._callback,
            )
            self._stream.start()
        finally:
            if prev_sink is not None:
                os.environ["PULSE_SINK"] = prev_sink

    def _resolve_device(self, sd):
        """Choose an output device. Prefer an explicit choice / env override,
        then PulseAudio ('pulse', which routes to the user's real default sink
        and resamples), then PortAudio's system default."""
        if self.device is not None:
            return self.device
        env = os.environ.get("AUDIO_OUTPUT_DEVICE")
        if env:
            try:
                return int(env)
            except ValueError:
                return env
        try:
            for d in sd.query_devices():
                if d["max_output_channels"] > 0 and d["name"] == "pulse":
                    return "pulse"
        except Exception:
            pass
        return None  # PortAudio system default

    def _pick_output_rate(self, sd, sample_rate: int) -> int:
        """Return a device-supported output rate, preferring the source rate.

        Many hardware devices (e.g. HDMI) reject 24 kHz; PortAudio/ALSA won't
        resample, so we pick a supported rate and resample in software.
        """
        def supported(rate: int) -> bool:
            try:
                sd.check_output_settings(
                    device=self._device, samplerate=rate,
                    channels=self.channels, dtype=self.dtype,
                )
                return True
            except Exception:
                return False

        if supported(sample_rate):
            return sample_rate

        candidates = []
        try:
            dev = self._device if self._device is not None else sd.default.device[1]
            default_sr = int(sd.query_devices(dev)["default_samplerate"])
            candidates.append(default_sr)
        except Exception:
            pass
        candidates += [48000, 44100, 32000, 22050, 16000]

        for rate in candidates:
            if rate != sample_rate and supported(rate):
                return rate
        # Nothing matched; fall back to the source rate and let it raise loudly.
        return sample_rate

    def __call__(self, audio_chunk: np.ndarray, sample_rate: int):
        if self._closed:
            raise RuntimeError("StreamPlayer is already closed")
        self._ensure_stream(sample_rate)
        chunk = self._reshape_chunk(audio_chunk)
        if self._resampler is not None:
            chunk = self._resampler.process(chunk)
            if chunk.shape[0] == 0:
                return
        self._queue.put(chunk)

    def close(self, *, wait: bool = True, timeout: Optional[float] = None):
        if self._closed:
            return
        self._closed = True

        if self._stream is None:
            return

        self._queue.put(None)
        if wait:
            self._drained.wait(timeout=timeout)

        self._stream.close()
        self._stream = None