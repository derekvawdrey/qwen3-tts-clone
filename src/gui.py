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

# Discord-ish dark palette.
C_BG = "#313338"        # main content
C_BG_ALT = "#2b2d31"    # buttons / raised
C_BG_INPUT = "#1e1f22"  # inputs / text areas
C_BORDER = "#3f4147"
C_FG = "#dbdee1"        # primary text
C_FG_MUTED = "#949ba4"  # secondary text
C_ACCENT = "#5865f2"    # blurple
C_ACCENT_HI = "#4752c4"
C_DANGER = "#da373c"
C_DANGER_HI = "#a12d2f"
C_GREEN = "#3ba55d"
C_RED = "#f23f43"

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

        self._apply_theme()
        self._build_widgets()
        self._refresh_devices()
        self._populate_voices()
        self._apply_settings()
        root.after(100, self._poll_events)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- theme -----------------------------------------------------------
    def _apply_theme(self):
        """A dark, Discord-ish ttk theme."""
        style = ttk.Style()
        try:
            style.theme_use("clam")  # most styleable built-in theme
        except tk.TclError:
            pass
        self.root.configure(bg=C_BG)
        # dropdown list popups (not ttk-styleable directly)
        self.root.option_add("*TCombobox*Listbox.background", C_BG_INPUT)
        self.root.option_add("*TCombobox*Listbox.foreground", C_FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", C_ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        font = ("Helvetica", 10)
        style.configure(".", background=C_BG, foreground=C_FG,
                        fieldbackground=C_BG_INPUT, bordercolor=C_BORDER,
                        focuscolor=C_BG, font=font)
        style.configure("TFrame", background=C_BG)
        style.configure("TLabel", background=C_BG, foreground=C_FG)
        style.configure("Muted.TLabel", foreground=C_FG_MUTED)
        style.configure("TLabelframe", background=C_BG, bordercolor=C_BORDER,
                        relief="solid")
        style.configure("TLabelframe.Label", background=C_BG, foreground=C_FG_MUTED,
                        font=("Helvetica", 9, "bold"))
        style.configure("TCheckbutton", background=C_BG, foreground=C_FG,
                        indicatorcolor=C_BG_INPUT)
        style.map("TCheckbutton", background=[("active", C_BG)],
                  foreground=[("disabled", C_FG_MUTED)],
                  indicatorcolor=[("selected", C_ACCENT), ("!selected", C_BG_INPUT)])
        style.configure("TButton", background=C_BG_ALT, foreground=C_FG,
                        borderwidth=0, padding=(10, 6))
        style.map("TButton", background=[("active", "#404249"),
                                         ("disabled", C_BG_ALT)],
                  foreground=[("disabled", C_FG_MUTED)])
        style.configure("Accent.TButton", background=C_ACCENT, foreground="#ffffff",
                        padding=(16, 6))
        style.map("Accent.TButton", background=[("active", C_ACCENT_HI),
                                                ("disabled", "#3a3f63")])
        style.configure("Danger.TButton", background=C_BG_ALT, foreground="#f0b6b7")
        style.map("Danger.TButton", background=[("active", C_DANGER_HI),
                                                ("disabled", C_BG_ALT)],
                  foreground=[("active", "#ffffff"), ("disabled", C_FG_MUTED)])
        for widget in ("TCombobox", "TEntry", "TSpinbox"):
            style.configure(widget, fieldbackground=C_BG_INPUT, foreground=C_FG,
                            bordercolor=C_BORDER, arrowcolor=C_FG,
                            insertcolor=C_FG, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", C_BG_INPUT)],
                  foreground=[("readonly", C_FG)],
                  selectbackground=[("readonly", C_BG_INPUT)],
                  selectforeground=[("readonly", C_FG)])
        style.configure("Placeholder.TEntry", fieldbackground=C_BG_INPUT,
                        foreground=C_FG_MUTED, padding=4)

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
        self.ref_text = tk.Text(
            voice, height=3, wrap="word", bg=C_BG_INPUT, fg=C_FG,
            insertbackground=C_FG, relief="flat", highlightthickness=1,
            highlightbackground=C_BORDER, highlightcolor=C_ACCENT, padx=6, pady=4)
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
        self.language_var.trace_add("write", self._on_language_change)
        ttk.Label(gen, text="Instruct:").grid(row=2, column=0, sticky="w", **pad)
        self.instruct_var = tk.StringVar(value=config.INSTRUCT)
        ttk.Entry(gen, textvariable=self.instruct_var) \
            .grid(row=2, column=1, columnspan=2, sticky="ew", **pad)
        # Live-apply to a running pipeline (takes effect on the next utterance).
        self.instruct_var.trace_add("write", self._on_instruct_change)
        self.expressive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            gen, variable=self.expressive_var,
            text="Experimental: expressive clone — follow Instruct "
                 "(1.7B CustomVoice; ignores TTS-model choice)",
        ).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

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
        ctrl.pack(fill="x", padx=8, pady=6)
        self.apply_btn = ttk.Button(ctrl, text="Apply settings", style="Accent.TButton",
                                    command=self._apply)
        self.apply_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="Stop", style="Danger.TButton",
                                   command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status_dot = ttk.Label(ctrl, text="●", foreground=C_FG_MUTED)
        self.status_dot.pack(side="right", padx=(6, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(ctrl, textvariable=self.status_var, style="Muted.TLabel") \
            .pack(side="right")

        self.log = tk.Text(
            self.root, height=10, wrap="word", state="disabled", bg=C_BG_INPUT,
            fg=C_FG, relief="flat", highlightthickness=1, highlightbackground=C_BORDER,
            padx=8, pady=6, insertbackground=C_FG)
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.log.tag_configure("you", foreground=C_GREEN)
        self.log.tag_configure("typed", foreground="#00a8fc")
        self.log.tag_configure("err", foreground=C_RED)
        self.log.tag_configure("dim", foreground=C_FG_MUTED)

        # --- Message box: type text to speak it ---
        msg = ttk.Frame(self.root)
        msg.pack(fill="x", padx=8, pady=(0, 8))
        self.msg_var = tk.StringVar()
        self.msg_entry = ttk.Entry(msg, textvariable=self.msg_var)
        self.msg_entry.pack(side="left", fill="x", expand=True)
        self.msg_entry.bind("<Return>", lambda _e: self._send_text())
        self._init_placeholder()
        ttk.Button(msg, text="Send", style="Accent.TButton",
                   command=self._send_text).pack(side="left", padx=(6, 0))

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

    def _on_instruct_change(self, *_):
        if self.pipeline and self.pipeline.running:
            self.pipeline.set_instruct(self.instruct_var.get())

    # ---- type-to-speak ---------------------------------------------------
    def _init_placeholder(self):
        self._ph_on = False
        self._show_placeholder()
        self.msg_entry.bind("<FocusIn>", self._ph_focus_in)
        self.msg_entry.bind("<FocusOut>", self._ph_focus_out)

    def _show_placeholder(self):
        self.msg_var.set("Type a message to speak…")
        self.msg_entry.configure(style="Placeholder.TEntry")
        self._ph_on = True

    def _ph_focus_in(self, _e):
        if self._ph_on:
            self.msg_var.set("")
            self.msg_entry.configure(style="TEntry")
            self._ph_on = False

    def _ph_focus_out(self, _e):
        if not self.msg_var.get().strip():
            self._show_placeholder()

    def _send_text(self):
        if self._ph_on:
            return
        text = self.msg_var.get().strip()
        if not text:
            return
        if not (self.pipeline and self.pipeline.running):
            self._apply()  # start the pipeline first (loads the model)
        if self.pipeline and self.pipeline.say(text):
            self._log(f"[type] {text}", "typed")
            self.msg_var.set("")
        else:
            self._log("couldn't queue message — press Apply settings first", "dim")

    def _on_language_change(self, *_):
        if self.pipeline and self.pipeline.running:
            self.pipeline.set_language(self.language_var.get())

    def _collect_config(self) -> dict:
        """Snapshot every widget value (must run on the UI thread)."""
        voice_path = self._current_voice_path()
        ref_text = self._get_ref_text()
        if voice_path:
            self.ref_texts[voice_path] = ref_text  # remember per-clip
        return dict(
            input_device=self._selected(self.mic_combo, self.mic_map),
            output_device=self._selected(self.out_combo, self.out_map),
            voice_path=voice_path,
            ref_text=ref_text,
            tts_model=self.tts_model_var.get().strip(),
            stt_model=self.stt_model_var.get().strip(),
            vad=self.vad_var.get(),
            instruct=self.instruct_var.get(),
            language=self.language_var.get(),
            expressive=self.expressive_var.get(),
            vmic=self.vmic_var.get(),
            duplex=self.duplex_var.get(),
        )

    def _apply(self):
        """(Re)start the pipeline with the current settings."""
        cfg = self._collect_config()
        self._save_settings()
        self.apply_btn.configure(state="disabled")

        if not (self.pipeline and self.pipeline.running):
            self.status_var.set("loading…")
            self._launch(cfg)  # nothing running → start now on the UI thread
            return

        # Running → stop in a worker (slow join), then relaunch on the UI thread
        # via the event queue (_poll_events drains it; Tk calls must stay on the
        # main thread).
        self.status_var.set("applying…")

        def worker():
            self._teardown()
            self.events_q.put({"type": "_relaunch", "cfg": cfg})

        threading.Thread(target=worker, daemon=True).start()

    def _launch(self, cfg: dict):
        """Create + start a fresh pipeline from a config snapshot (UI thread)."""
        if cfg["tts_model"] and cfg["tts_model"] != config.MODEL_ID:
            config.MODEL_ID = cfg["tts_model"]
            load_model.cache_clear()
        config.STT_MODEL = cfg["stt_model"] or config.STT_MODEL
        try:
            config.VAD_SILENCE_MS = int(cfg["vad"])
        except ValueError:
            pass

        output_device = cfg["output_device"]
        if cfg["vmic"]:
            try:
                sink = self.vmic.create()
            except Exception as exc:
                self._log(f"virtual mic failed: {exc}", "err")
                self.apply_btn.configure(state="normal")
                self.status_var.set("idle")
                return
            os.environ["PULSE_SINK"] = sink
            output_device = "pulse"
            self._log(f"virtual mic '{self.vmic.source_name}' active — select it as "
                      "the microphone in your other app", "dim")
        else:
            os.environ.pop("PULSE_SINK", None)

        if cfg["expressive"]:
            self._log("expressive clone on — 1.7B CustomVoice; first use extracts "
                      "the voice embedding (may take a few seconds)", "dim")

        self.pipeline = Pipeline(
            cfg["input_device"], output_device, cfg["instruct"], self.events_q,
            half_duplex=cfg["duplex"], ref_audio=cfg["voice_path"],
            ref_text=cfg["ref_text"] or None, language=cfg["language"] or None,
            expressive=cfg["expressive"])
        self.pipeline.start()
        self.apply_btn.configure(state="normal")  # re-apply anytime to restart
        self.stop_btn.configure(state="normal")

    def _stop(self):
        self.stop_btn.configure(state="disabled")
        self.status_var.set("stopping…")

        def worker():
            self._teardown()
            self.events_q.put({"type": "status", "value": "stopped"})

        threading.Thread(target=worker, daemon=True).start()

    def _teardown(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        if self.vmic.active:
            self.vmic.destroy()
        os.environ.pop("PULSE_SINK", None)

    # ---- events / log ----------------------------------------------------
    def _poll_events(self):
        try:
            while True:
                ev = self.events_q.get_nowait()
                kind = ev["type"]
                if kind == "_relaunch":
                    self._launch(ev["cfg"])
                elif kind == "status":
                    self.status_var.set(ev["value"])
                    self.status_dot.configure(foreground={
                        "listening": C_GREEN, "speaking": C_ACCENT,
                        "loading": "#faa61a", "stopped": C_FG_MUTED,
                    }.get(ev["value"], C_FG_MUTED))
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
            "expressive": self.expressive_var.get(),
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
        if "expressive" in s:
            self.expressive_var.set(s["expressive"])

    def _on_close(self):
        try:
            self._save_settings()
            self._teardown()
        finally:
            self.root.destroy()


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
