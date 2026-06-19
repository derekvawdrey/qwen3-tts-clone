"""Tkinter control panel for the realtime voice-clone pipeline.

Pick a microphone, output, and voice sample; tweak generation/STT settings;
optionally expose the cloned voice as a virtual microphone; then Start/Stop and
watch the live transcript. Settings persist to .gui_settings.json.

    python -m src.gui
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.clone_voice import load_model  # noqa: E402
from src.pipeline import Pipeline, _parse_cuda  # noqa: E402
from src.virtual_mic import VirtualMic  # noqa: E402

AUTO_LABEL = "Auto (PulseAudio default)"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
SETTINGS_PATH = config.ROOT / ".gui_settings.json"
TTS_MODELS = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
STT_MODELS = ["tiny.en", "base.en", "small.en", "distil-small.en", "medium.en"]
LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Italian",
             "Japanese", "Korean", "Portuguese", "Russian"]

# Transcripts for the bundled reference clips, so they work out of the box.
PRESET_TEXTS = {
    "trump_trade_deficit": "then I did it again but I did it for a lot of others. "
        "You look at the stats, the deficit last month was cut in half.",
    "china-is-going-to-eat-our-lunch-come-on-man-they-can-t-even-figure-out-how-to-"
    "deal-with-the-fact-that-they-have-this-gre":
        "China is going to eat our lunch? Come on, man. They can't even figure out "
        "how to deal with the fact that they have this great division between the "
        "China Sea and the mountains in the east, I mean, in the west.",
    "naomi_voice": "Hello. This is Naomi we are doing a voice recording of my "
        "voice. I don't know what else to say, so hopefully this is a good amount "
        "of time.",
    "bartholomew-we-don-t-have-anymore-time": "Bartholomew, we don't have anymore time!",
    "eight-billion-seven-hundred-and-thirty-seven-million-five-hundred-and-forty-"
    "thousand-dollars": "Eight billion seven hundred and thirty seven million five "
        "hundred and forty thousand dollars",
    "but-they-share-my-unique-face-colonel-watson-s-name-has-chickens-and-they-don-"
    "t-even-have-mustaches": "But they share my unique face! Colonel Watson's name "
        "has chickens, and they don't even have mustaches.",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Qwen3-TTS Voice")
        root.minsize(620, 720)

        self.events_q: queue.Queue = queue.Queue()
        self.pipeline: Pipeline | None = None
        self.vmic = VirtualMic()
        self.settings = self._load_settings()
        self.ref_texts: dict = self.settings.get("ref_texts", {})
        self._voice_paths: dict[str, str] = {}

        self._build_widgets()
        self._refresh_devices()
        self._populate_voices()
        self._apply_settings()
        root.after(100, self._poll_events)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout ----------------------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 6, "pady": 3}

        # --- Devices ---
        dev = ttk.LabelFrame(self.root, text="Devices")
        dev.pack(fill="x", padx=8, pady=4)
        dev.columnconfigure(1, weight=1)
        ttk.Label(dev, text="Microphone:").grid(row=0, column=0, sticky="w", **pad)
        self.mic_combo = ttk.Combobox(dev, state="readonly")
        self.mic_combo.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(dev, text="Output:").grid(row=1, column=0, sticky="w", **pad)
        self.out_combo = ttk.Combobox(dev, state="readonly")
        self.out_combo.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(dev, text="↻ Refresh", command=self._refresh_devices) \
            .grid(row=0, column=2, rowspan=2, sticky="ns", **pad)

        # --- Voice sample ---
        voice = ttk.LabelFrame(self.root, text="Voice sample")
        voice.pack(fill="x", padx=8, pady=4)
        voice.columnconfigure(1, weight=1)
        ttk.Label(voice, text="Clip:").grid(row=0, column=0, sticky="w", **pad)
        self.voice_combo = ttk.Combobox(voice, state="readonly")
        self.voice_combo.grid(row=0, column=1, sticky="ew", **pad)
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)
        ttk.Button(voice, text="Browse…", command=self._browse_voice) \
            .grid(row=0, column=2, **pad)
        ttk.Label(voice, text="Reference text:").grid(row=1, column=0, sticky="nw", **pad)
        self.ref_text = tk.Text(voice, height=3, wrap="word")
        self.ref_text.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(voice, text="Auto-transcribe", command=self._auto_transcribe) \
            .grid(row=1, column=2, sticky="n", **pad)

        # --- Generation ---
        gen = ttk.LabelFrame(self.root, text="Generation")
        gen.pack(fill="x", padx=8, pady=4)
        gen.columnconfigure(1, weight=1)
        ttk.Label(gen, text="TTS model:").grid(row=0, column=0, sticky="w", **pad)
        self.tts_model_var = tk.StringVar(value=config.MODEL_ID)
        ttk.Combobox(gen, textvariable=self.tts_model_var, values=TTS_MODELS) \
            .grid(row=0, column=1, columnspan=2, sticky="ew", **pad)
        ttk.Label(gen, text="Language:").grid(row=1, column=0, sticky="w", **pad)
        self.language_var = tk.StringVar(value=config.LANGUAGE)
        ttk.Combobox(gen, textvariable=self.language_var, values=LANGUAGES) \
            .grid(row=1, column=1, columnspan=2, sticky="ew", **pad)
        ttk.Label(gen, text="Instruct:").grid(row=2, column=0, sticky="w", **pad)
        self.instruct_var = tk.StringVar(value=config.INSTRUCT)
        ttk.Entry(gen, textvariable=self.instruct_var) \
            .grid(row=2, column=1, columnspan=2, sticky="ew", **pad)

        # --- Speech-to-text ---
        stt = ttk.LabelFrame(self.root, text="Speech-to-text")
        stt.pack(fill="x", padx=8, pady=4)
        stt.columnconfigure(1, weight=1)
        ttk.Label(stt, text="STT model:").grid(row=0, column=0, sticky="w", **pad)
        self.stt_model_var = tk.StringVar(value=config.STT_MODEL)
        ttk.Combobox(stt, textvariable=self.stt_model_var, values=STT_MODELS) \
            .grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(stt, text="End-of-speech silence (ms):") \
            .grid(row=1, column=0, sticky="w", **pad)
        self.vad_var = tk.StringVar(value=str(config.VAD_SILENCE_MS))
        ttk.Spinbox(stt, from_=200, to=2000, increment=100, width=8,
                    textvariable=self.vad_var).grid(row=1, column=1, sticky="w", **pad)

        # --- Routing ---
        route = ttk.LabelFrame(self.root, text="Routing")
        route.pack(fill="x", padx=8, pady=4)
        self.vmic_var = tk.BooleanVar(value=VirtualMic.available())
        vmic_chk = ttk.Checkbutton(
            route, text=f"Expose as virtual microphone ('{self.vmic.source_name}')",
            variable=self.vmic_var, command=self._on_vmic_toggle)
        vmic_chk.pack(anchor="w", **pad)
        if not VirtualMic.available():
            vmic_chk.configure(state="disabled")
            self.vmic_var.set(False)
        self.duplex_var = tk.BooleanVar(value=not self.vmic_var.get())
        ttk.Checkbutton(
            route, text="Mute mic while speaking (echo guard — uncheck for barge-in)",
            variable=self.duplex_var).pack(anchor="w", **pad)

        # --- Controls ---
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=8, pady=4)
        self.start_btn = ttk.Button(ctrl, text="▶ Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="■ Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(ctrl, textvariable=self.status_var, foreground="#555") \
            .pack(side="right")

        self.log = tk.Text(self.root, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.tag_configure("you", foreground="#0a6")
        self.log.tag_configure("err", foreground="#c00")
        self.log.tag_configure("dim", foreground="#888")

    # ---- devices / voices ------------------------------------------------
    def _refresh_devices(self):
        import sounddevice as sd

        sd._terminate()  # re-scan so new virtual devices show up
        sd._initialize()
        self.mic_map = {AUTO_LABEL: None}
        self.out_map = {AUTO_LABEL: None}
        for i, d in enumerate(sd.query_devices()):
            label = f"[{i}] {d['name']}"
            if d["max_input_channels"] > 0:
                self.mic_map[label] = i
            if d["max_output_channels"] > 0:
                self.out_map[label] = i
        self.mic_combo["values"] = list(self.mic_map)
        self.out_combo["values"] = list(self.out_map)
        if not self.mic_combo.get():
            self.mic_combo.set(AUTO_LABEL)
        if not self.out_combo.get():
            self.out_combo.set(AUTO_LABEL)

    def _populate_voices(self):
        """List unique audio clips in assets/ (default config clip first)."""
        self._voice_paths.clear()
        seen_stems = set()
        files = sorted(config.ASSETS_DIR.glob("*"))
        for f in files:
            if f.suffix.lower() in AUDIO_EXTS and f.stem not in seen_stems:
                seen_stems.add(f.stem)
                # prefer an existing .wav sibling (already prepped) if present
                wav = f.with_suffix(".wav")
                self._voice_paths[f.stem] = str(wav if wav.exists() else f)
        self.voice_combo["values"] = list(self._voice_paths)
        # default selection: the config reference clip's stem, else first
        default_stem = Path(config.REF_AUDIO_MP3).stem
        if default_stem in self._voice_paths:
            self.voice_combo.set(default_stem)
        elif self._voice_paths:
            self.voice_combo.set(next(iter(self._voice_paths)))
        self._on_voice_selected()

    def _current_voice_path(self) -> str | None:
        return self._voice_paths.get(self.voice_combo.get())

    def _on_voice_selected(self, _event=None):
        path = self._current_voice_path()
        if not path:
            return
        # remembered text > preset > leave as-is
        text = self.ref_texts.get(path) or PRESET_TEXTS.get(Path(path).stem)
        if text is not None:
            self._set_ref_text(text)

    def _browse_voice(self):
        path = filedialog.askopenfilename(
            title="Select a reference voice clip",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.opus"),
                       ("All files", "*.*")])
        if not path:
            return
        label = Path(path).name
        self._voice_paths[label] = path
        self.voice_combo["values"] = list(self._voice_paths)
        self.voice_combo.set(label)
        self._on_voice_selected()

    def _set_ref_text(self, text: str):
        self.ref_text.delete("1.0", "end")
        self.ref_text.insert("1.0", text)

    def _get_ref_text(self) -> str:
        return self.ref_text.get("1.0", "end").strip()

    # ---- auto-transcribe -------------------------------------------------
    def _auto_transcribe(self):
        path = self._current_voice_path()
        if not path:
            return
        self._log(f"transcribing {Path(path).name}…", "dim")
        threading.Thread(target=self._do_transcribe, args=(path,), daemon=True).start()

    def _do_transcribe(self, path: str):
        try:
            from faster_whisper import WhisperModel

            device, index = _parse_cuda(config.DEVICE)
            compute = config.STT_COMPUTE if device == "cuda" else "int8"
            model = WhisperModel(self.stt_model_var.get().strip() or config.STT_MODEL,
                                 device=device, device_index=index, compute_type=compute)
            segments, _ = model.transcribe(str(path), language="en", beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
            self.root.after(0, lambda: (self._set_ref_text(text),
                                        self._log("transcription done", "dim")))
        except Exception as exc:
            self.root.after(0, lambda: self._log(f"transcribe failed: {exc}", "err"))

    # ---- control ---------------------------------------------------------
    @staticmethod
    def _selected(combo, mapping):
        return mapping.get(combo.get(), None)

    def _on_vmic_toggle(self):
        # Virtual mic / headphones have no acoustic echo path → guard off by default.
        self.duplex_var.set(not self.vmic_var.get())

    def _start(self):
        if self.pipeline and self.pipeline.running:
            return
        input_device = self._selected(self.mic_combo, self.mic_map)
        output_device = self._selected(self.out_combo, self.out_map)
        voice_path = self._current_voice_path()
        ref_text = self._get_ref_text()
        if voice_path:
            self.ref_texts[voice_path] = ref_text  # remember per-clip

        # Settings that the pipeline reads from config — apply before start.
        new_model = self.tts_model_var.get().strip()
        if new_model and new_model != config.MODEL_ID:
            config.MODEL_ID = new_model
            load_model.cache_clear()  # force a reload of the new checkpoint
        config.STT_MODEL = self.stt_model_var.get().strip() or config.STT_MODEL
        try:
            config.VAD_SILENCE_MS = int(self.vad_var.get())
        except ValueError:
            pass

        if self.vmic_var.get():
            try:
                sink = self.vmic.create()
            except Exception as exc:
                self._log(f"virtual mic failed: {exc}", "err")
                return
            os.environ["PULSE_SINK"] = sink
            output_device = "pulse"
            self._log(f"virtual mic '{self.vmic.source_name}' active — select it as "
                      "the microphone in your other app", "dim")
        else:
            os.environ.pop("PULSE_SINK", None)

        self._save_settings()
        self.pipeline = Pipeline(
            input_device, output_device, self.instruct_var.get(), self.events_q,
            half_duplex=self.duplex_var.get(), ref_audio=voice_path,
            ref_text=ref_text or None, language=self.language_var.get() or None)
        self.pipeline.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("loading…")

    def _stop(self):
        self.stop_btn.configure(state="disabled")
        self.status_var.set("stopping…")
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        if self.vmic.active:
            self.vmic.destroy()
        os.environ.pop("PULSE_SINK", None)
        self.root.after(0, lambda: self.start_btn.configure(state="normal"))

    # ---- events / log ----------------------------------------------------
    def _poll_events(self):
        try:
            while True:
                ev = self.events_q.get_nowait()
                kind = ev["type"]
                if kind == "status":
                    self.status_var.set(ev["value"])
                elif kind == "user":
                    self._log(f"[you] {ev['text']}", "you")
                elif kind == "error":
                    self._log(ev["text"], "err")
                else:
                    self._log(ev["text"], "dim")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _log(self, text: str, tag: str | None = None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---- settings persistence -------------------------------------------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return {}

    def _save_settings(self):
        data = {
            "mic": self.mic_combo.get(),
            "output": self.out_combo.get(),
            "voice": self.voice_combo.get(),
            "tts_model": self.tts_model_var.get(),
            "language": self.language_var.get(),
            "instruct": self.instruct_var.get(),
            "stt_model": self.stt_model_var.get(),
            "vad_ms": self.vad_var.get(),
            "vmic": self.vmic_var.get(),
            "duplex": self.duplex_var.get(),
            "ref_texts": self.ref_texts,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _apply_settings(self):
        s = self.settings
        if not s:
            return
        if s.get("mic") in self.mic_map:
            self.mic_combo.set(s["mic"])
        if s.get("output") in self.out_map:
            self.out_combo.set(s["output"])
        if s.get("voice") in self._voice_paths:
            self.voice_combo.set(s["voice"])
            self._on_voice_selected()
        for var, key in [(self.tts_model_var, "tts_model"),
                         (self.language_var, "language"),
                         (self.instruct_var, "instruct"),
                         (self.stt_model_var, "stt_model"),
                         (self.vad_var, "vad_ms")]:
            if s.get(key) is not None:
                var.set(s[key])
        if "vmic" in s:
            self.vmic_var.set(s["vmic"])
        if "duplex" in s:
            self.duplex_var.set(s["duplex"])

    def _on_close(self):
        try:
            self._save_settings()
            if self.pipeline:
                self.pipeline.stop()
            if self.vmic.active:
                self.vmic.destroy()
            os.environ.pop("PULSE_SINK", None)
        finally:
            self.root.destroy()


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
