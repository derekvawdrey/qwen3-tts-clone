"""Realtime speech-to-speech loop: mic -> STT -> Qwen3-TTS clone -> speaker.

Three threads connected by queues (the "bus"):

    [MicSource]  --audio_q-->  [Transcriber]  --text_q-->  [Speaker]
     InputStream               Silero VAD +                generate_voice_
     @16kHz mono               faster-whisper              clone_streaming
                                                           -> StreamPlayer

Runs **half-duplex**: the mic is gated (`speaking` event) while the TTS is
talking, so the model never transcribes its own output. Press Ctrl+C to stop.

    python -m src.pipeline
"""
from __future__ import annotations

import collections
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.clone_voice import ensure_reference_wav, load_model  # noqa: E402
from src.helpers import StreamPlayer  # noqa: E402

MIC_RATE = 16000       # Silero VAD + Whisper both expect 16 kHz mono
VAD_FRAME = 512        # samples per Silero frame at 16 kHz (~32 ms)
PREROLL_FRAMES = 8     # ~256 ms of audio kept before speech onset (avoid clipping)


def _resolve_input_device(sd):
    """Pick a mic device: explicit env override, else PulseAudio, else default."""
    env = config.MIC_DEVICE
    if env:
        return int(env) if str(env).isdigit() else env
    try:
        for d in sd.query_devices():
            if d["max_input_channels"] > 0 and d["name"] == "pulse":
                return "pulse"
    except Exception:
        pass
    return None  # PortAudio system default


class MicSource(threading.Thread):
    """Capture 16 kHz mono frames and push them onto audio_q (drops on overflow)."""

    def __init__(self, audio_q: queue.Queue, stop_evt: threading.Event):
        super().__init__(name="MicSource", daemon=True)
        self.audio_q = audio_q
        self.stop_evt = stop_evt

    def run(self):
        import sounddevice as sd

        device = _resolve_input_device(sd)
        print(f"[mic] capturing from device={device!r} @ {MIC_RATE} Hz")

        def callback(indata, frames, time_info, status):  # runs in PortAudio thread
            try:
                self.audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # consumer is behind (e.g. loading models) — drop, don't block audio

        with sd.InputStream(
            samplerate=MIC_RATE, blocksize=VAD_FRAME, channels=1,
            dtype="float32", device=device, callback=callback,
        ):
            self.stop_evt.wait()


class Transcriber(threading.Thread):
    """VAD-segment the mic stream and transcribe each utterance to text_q."""

    def __init__(self, audio_q, text_q, speaking_evt, stop_evt):
        super().__init__(name="Transcriber", daemon=True)
        self.audio_q = audio_q
        self.text_q = text_q
        self.speaking_evt = speaking_evt
        self.stop_evt = stop_evt

    def run(self):
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import VADIterator, load_silero_vad

        device, index = _parse_cuda(config.DEVICE)
        print(f"[stt] loading faster-whisper '{config.STT_MODEL}' "
              f"({config.STT_COMPUTE}) on {config.DEVICE} ...")
        asr = WhisperModel(
            config.STT_MODEL, device=device, device_index=index,
            compute_type=config.STT_COMPUTE,
        )
        vad = VADIterator(
            load_silero_vad(), sampling_rate=MIC_RATE,
            min_silence_duration_ms=config.VAD_SILENCE_MS,
        )
        print("[stt] ready")

        preroll: collections.deque = collections.deque(maxlen=PREROLL_FRAMES)
        collecting = False
        buf: list[np.ndarray] = []

        while not self.stop_evt.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Half-duplex: ignore the mic (and reset VAD) while the bot speaks.
            if self.speaking_evt.is_set():
                if collecting or preroll:
                    vad.reset_states()
                    collecting = False
                    buf = []
                    preroll.clear()
                continue

            preroll.append(frame)
            event = vad(torch.from_numpy(frame.astype("float32")))

            if event and "start" in event and not collecting:
                collecting = True
                buf = list(preroll)  # includes the current frame
            elif collecting:
                buf.append(frame)
                if event and "end" in event:
                    audio = np.concatenate(buf).astype("float32")
                    collecting, buf = False, []
                    self._transcribe(asr, audio)

    def _transcribe(self, asr, audio: np.ndarray):
        segments, _ = asr.transcribe(
            audio, language="en", beam_size=1,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            try:
                self.text_q.put_nowait(text)
            except queue.Full:
                pass  # speaker is busy; drop this utterance rather than queue up


class Speaker(threading.Thread):
    """Speak each text from text_q in the cloned voice, gating the mic meanwhile."""

    def __init__(self, text_q, audio_q, speaking_evt, stop_evt):
        super().__init__(name="Speaker", daemon=True)
        self.text_q = text_q
        self.audio_q = audio_q
        self.speaking_evt = speaking_evt
        self.stop_evt = stop_evt

    def run(self):
        print("[tts] loading Qwen3-TTS ...")
        model = load_model()
        ref_audio = str(ensure_reference_wav())
        print("[tts] ready")

        while not self.stop_evt.is_set():
            try:
                text = self.text_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break

            print(f"\n[you] {text}")
            self.speaking_evt.set()
            player = StreamPlayer()
            try:
                for chunk, sr, _timing in model.generate_voice_clone_streaming(
                    text=text,
                    language=config.LANGUAGE,
                    ref_audio=ref_audio,
                    ref_text=config.REF_TEXT,
                    instruct=config.INSTRUCT or None,
                    chunk_size=8,
                ):
                    player(chunk, sr)
            except Exception as exc:  # don't let one bad utterance kill the loop
                print(f"[tts] generation failed: {exc}", file=sys.stderr)
            finally:
                player.close()
                self._flush_mic()
                self.speaking_evt.clear()

    def _flush_mic(self):
        """Discard mic frames captured during playback (residual echo tail)."""
        try:
            while True:
                self.audio_q.get_nowait()
        except queue.Empty:
            pass


def _parse_cuda(device: str):
    """'cuda:0' -> ('cuda', 0); 'cpu' -> ('cpu', 0)."""
    if ":" in device:
        name, idx = device.split(":", 1)
        return name, int(idx)
    return device, 0


def main() -> int:
    audio_q: queue.Queue = queue.Queue(maxsize=200)  # ~6 s buffer @ 32 ms frames
    text_q: queue.Queue = queue.Queue(maxsize=4)
    speaking = threading.Event()
    stop = threading.Event()

    speaker = Speaker(text_q, audio_q, speaking, stop)
    transcriber = Transcriber(audio_q, text_q, speaking, stop)
    mic = MicSource(audio_q, stop)

    # Start the model-loading stages first so the mic isn't capturing into a
    # full queue for long; overflow is dropped anyway.
    speaker.start()
    transcriber.start()
    mic.start()

    print("\nListening — speak into the mic. Ctrl+C to stop.\n")
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        stop.set()
        try:
            text_q.put_nowait(None)
        except queue.Full:
            pass
        for t in (mic, transcriber, speaker):
            t.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
