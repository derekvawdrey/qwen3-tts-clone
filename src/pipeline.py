"""Realtime speech-to-speech loop: mic -> STT -> Qwen3-TTS clone -> speaker.

Three threads connected by queues (the "bus"):

    [MicSource]  --audio_q-->  [Transcriber]  --text_q-->  [Speaker]
     InputStream               Silero VAD +                generate_voice_
     @16kHz mono               faster-whisper              clone_streaming
                                                           -> StreamPlayer

Runs **half-duplex**: the mic is gated (`speaking` event) while the TTS is
talking, so the model never transcribes its own output.

Use the `Pipeline` class for programmatic control (the GUI does this), or run
the module for a console loop:

    python -m src.pipeline
"""
from __future__ import annotations

import collections
import queue
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.clone_voice import (  # noqa: E402
    EXPRESSIVE_TTS_MODEL,
    ensure_reference_wav,
    get_clone_prompt,
    load_model,
    prepare_reference,
    synthesize_stream,
)
from src.helpers import StreamPlayer  # noqa: E402

MIC_RATE = 16000       # Silero VAD + Whisper both expect 16 kHz mono
VAD_FRAME = 512        # samples per Silero frame at 16 kHz (~32 ms)
PREROLL_FRAMES = 8     # ~256 ms of audio kept before speech onset (avoid clipping)


def _emit(events_q, type_, **kw):
    """Push a UI event onto an optional queue (dropped if full/absent)."""
    if events_q is not None:
        try:
            events_q.put_nowait({"type": type_, **kw})
        except queue.Full:
            pass


def _resolve_input_device(sd, device):
    """Pick a mic device: explicit choice, else env, else PulseAudio, else default."""
    if device is not None:
        return device
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

    def __init__(self, audio_q, stop_evt, device=None, events_q=None):
        super().__init__(name="MicSource", daemon=True)
        self.audio_q = audio_q
        self.stop_evt = stop_evt
        self.device = device
        self.events_q = events_q

    def run(self):
        import sounddevice as sd

        device = _resolve_input_device(sd, self.device)

        def callback(indata, frames, time_info, status):  # runs in PortAudio thread
            try:
                self.audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # consumer is behind (e.g. loading models) — drop, don't block

        try:
            with sd.InputStream(
                samplerate=MIC_RATE, blocksize=VAD_FRAME, channels=1,
                dtype="float32", device=device, callback=callback,
            ):
                _emit(self.events_q, "info", text=f"mic: capturing from {device!r}")
                self.stop_evt.wait()
        except Exception as exc:
            _emit(self.events_q, "error", text=f"mic failed: {exc}")


class Transcriber(threading.Thread):
    """VAD-segment the mic stream and transcribe each utterance to text_q."""

    def __init__(self, audio_q, text_q, speaking_evt, stop_evt, events_q=None,
                 half_duplex=True):
        super().__init__(name="Transcriber", daemon=True)
        self.audio_q = audio_q
        self.text_q = text_q
        self.speaking_evt = speaking_evt
        self.stop_evt = stop_evt
        self.events_q = events_q
        self.half_duplex = half_duplex

    def run(self):
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import VADIterator, load_silero_vad

        device, index = _parse_cuda(config.DEVICE)
        _emit(self.events_q, "info",
              text=f"loading STT '{config.STT_MODEL}' ({config.STT_COMPUTE})…")
        try:
            asr = WhisperModel(
                config.STT_MODEL, device=device, device_index=index,
                compute_type=config.STT_COMPUTE,
            )
            vad = VADIterator(
                load_silero_vad(), sampling_rate=MIC_RATE,
                min_silence_duration_ms=config.VAD_SILENCE_MS,
            )
        except Exception as exc:
            _emit(self.events_q, "error", text=f"STT load failed: {exc}")
            return
        _emit(self.events_q, "info", text="STT ready")
        _emit(self.events_q, "status", value="listening")

        preroll = collections.deque(maxlen=PREROLL_FRAMES)
        collecting = False
        buf: list[np.ndarray] = []

        while not self.stop_evt.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Half-duplex: ignore the mic (and reset VAD) while the bot speaks,
            # to avoid transcribing TTS that leaks back in through the speakers.
            # Skipped when full-duplex (virtual mic / headphones) so you can
            # barge in while the cloned voice is still talking.
            if self.half_duplex and self.speaking_evt.is_set():
                if collecting or preroll:
                    vad.reset_states()
                    collecting, buf = False, []
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
            audio, language="en", beam_size=1, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            _emit(self.events_q, "user", text=text)
            try:
                self.text_q.put_nowait(text)
            except queue.Full:
                pass  # speaker is busy; drop rather than build a backlog


class Speaker(threading.Thread):
    """Speak each text from text_q in the cloned voice, gating the mic meanwhile."""

    def __init__(self, text_q, audio_q, speaking_evt, stop_evt,
                 output_device=None, instruct=None, events_q=None, half_duplex=True,
                 ref_audio=None, ref_text=None, language=None, expressive=False,
                 icl=False):
        super().__init__(name="Speaker", daemon=True)
        self.text_q = text_q
        self.audio_q = audio_q
        self.speaking_evt = speaking_evt
        self.stop_evt = stop_evt
        self.output_device = output_device
        # None => fall back to config default; "" => explicitly disabled
        self.instruct = config.INSTRUCT if instruct is None else instruct
        self.events_q = events_q
        self.half_duplex = half_duplex
        self.ref_audio = ref_audio  # path to a reference clip, or None for config default
        self.ref_text = ref_text    # transcript of the clip, or None for config default
        self.language = language or config.LANGUAGE
        self.expressive = expressive  # cloned identity + instruction (1.7B CustomVoice)
        self.icl = icl                # expressive: in-context (ref audio+text) vs x-vector

    def run(self):
        kind = "expressive clone (1.7B CustomVoice)" if self.expressive else "Qwen3-TTS"
        _emit(self.events_q, "info", text=f"loading {kind}…")
        try:
            ref_wav = str(prepare_reference(self.ref_audio) if self.ref_audio
                          else ensure_reference_wav())
            ref_text = config.REF_TEXT if self.ref_text is None else self.ref_text
            # Build the clone prompt BEFORE loading the synthesis model: the Base
            # model loads transiently and frees, keeping the VRAM peak lower.
            if self.expressive:
                use_icl = bool(self.icl and ref_text and ref_text.strip())
                _emit(self.events_q, "info",
                      text=f"building {'ICL' if use_icl else 'x-vector'} clone prompt…")
                get_clone_prompt(ref_wav, ref_text=ref_text, icl=use_icl)
            load_model(EXPRESSIVE_TTS_MODEL if self.expressive else None)
        except Exception as exc:
            _emit(self.events_q, "error", text=f"TTS load failed: {exc}")
            return
        _emit(self.events_q, "info", text="TTS ready")

        while not self.stop_evt.is_set():
            try:
                text = self.text_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break

            self.speaking_evt.set()
            _emit(self.events_q, "status", value="speaking")
            player = StreamPlayer(device=self.output_device)
            try:
                for chunk, sr in synthesize_stream(
                    text, language=self.language, ref_audio=ref_wav,
                    ref_text=self.ref_text, instruct=self.instruct,
                    expressive=self.expressive, icl=self.icl,
                ):
                    player(chunk, sr)
            except Exception as exc:  # one bad utterance shouldn't kill the loop
                _emit(self.events_q, "error", text=f"generation failed: {exc}")
            finally:
                player.close()
                # In half-duplex, drop the echo tail captured during playback.
                # In full-duplex, keep it — it may be the user barging in.
                if self.half_duplex:
                    self._flush_mic()
                self.speaking_evt.clear()
                _emit(self.events_q, "status", value="listening")

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


class Pipeline:
    """Controllable mic->STT->TTS loop. Start/stop and observe via events_q."""

    def __init__(self, input_device=None, output_device=None, instruct=None,
                 events_q=None, half_duplex=True, ref_audio=None, ref_text=None,
                 language=None, expressive=False, icl=False):
        self.input_device = input_device
        self.output_device = output_device
        self.instruct = instruct
        self.events_q = events_q
        self.half_duplex = half_duplex
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.expressive = expressive
        self.icl = icl
        self.audio_q = queue.Queue(maxsize=200)   # ~6 s @ 32 ms frames
        self.text_q = queue.Queue(maxsize=4)
        self.speaking = threading.Event()
        self.stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []
        self._speaker: Speaker | None = None

    @property
    def running(self) -> bool:
        return bool(self._threads)

    def say(self, text: str) -> bool:
        """Queue typed text to be spoken (same path as a transcribed utterance)."""
        text = (text or "").strip()
        if not text or not self._threads:
            return False
        try:
            self.text_q.put_nowait(text)
            return True
        except queue.Full:
            return False

    def set_instruct(self, value):
        """Live-update the style prompt; takes effect on the next utterance."""
        self.instruct = value
        if self._speaker is not None:
            self._speaker.instruct = value

    def set_language(self, value):
        """Live-update the language; takes effect on the next utterance."""
        self.language = value
        if self._speaker is not None:
            self._speaker.language = value or config.LANGUAGE

    def start(self):
        if self._threads:
            return
        _emit(self.events_q, "status", value="loading")
        speaker = Speaker(self.text_q, self.audio_q, self.speaking, self.stop_evt,
                          self.output_device, self.instruct, self.events_q,
                          self.half_duplex, self.ref_audio, self.ref_text,
                          self.language, self.expressive, self.icl)
        transcriber = Transcriber(self.audio_q, self.text_q, self.speaking,
                                  self.stop_evt, self.events_q, self.half_duplex)
        mic = MicSource(self.audio_q, self.stop_evt, self.input_device, self.events_q)
        self._speaker = speaker
        self._threads = [speaker, transcriber, mic]
        # Load-heavy stages first so the mic isn't filling a full queue for long.
        for t in self._threads:
            t.start()

    def stop(self):
        self.stop_evt.set()
        try:
            self.text_q.put_nowait(None)
        except queue.Full:
            pass
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        self._speaker = None
        _emit(self.events_q, "status", value="stopped")


def main() -> int:
    events_q: queue.Queue = queue.Queue()
    pipe = Pipeline(events_q=events_q)
    pipe.start()
    print("\nListening — speak into the mic. Ctrl+C to stop.\n")
    try:
        while True:
            try:
                ev = events_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev["type"] == "user":
                print(f"[you] {ev['text']}")
            elif ev["type"] == "status":
                print(f"[status] {ev['value']}")
            elif ev["type"] == "error":
                print(f"[error] {ev['text']}", file=sys.stderr)
            else:
                print(f"[info] {ev['text']}")
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        pipe.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
