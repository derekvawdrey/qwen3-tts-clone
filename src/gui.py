"""Modern PySide6 control panel for the realtime voice-clone pipeline.

Pick a microphone, output, and voice; tweak generation/STT settings; optionally
expose the cloned voice as a virtual microphone; then Apply and watch the live
transcript. Type a message to speak it in the cloned voice. Settings persist to
.gui_settings.json.

    python -m src.gui

(The legacy Tkinter version lives in src/gui_tk.py.)
"""
from __future__ import annotations

import html
import json
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import config  # noqa: E402
from src.clone_voice import unload_models  # noqa: E402
from src.personalities import load_personalities  # noqa: E402
from src.pipeline import Pipeline, _parse_cuda  # noqa: E402
from src.virtual_mic import VirtualMic  # noqa: E402

AUTO_LABEL = "Auto (system default)"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
SETTINGS_PATH = config.ROOT / ".gui_settings.json"
TTS_MODELS = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
STT_MODELS = ["tiny.en", "base.en", "small.en", "distil-small.en", "medium.en"]
LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Italian",
             "Japanese", "Korean", "Portuguese", "Russian"]

# --- palette ---------------------------------------------------------------
C_BG = "#1a1b1e"         # window
C_PANEL = "#26282c"      # cards
C_PANEL_HI = "#2f3136"   # hovered / raised
C_INPUT = "#1e1f22"      # inputs
C_BORDER = "#3a3d44"
C_FG = "#e3e5e8"         # primary text
C_FG_MUTED = "#9aa0a8"   # secondary text
C_ACCENT = "#5b6ef5"     # indigo
C_ACCENT_HI = "#4a5be0"
C_DANGER = "#e0484d"
C_DANGER_HI = "#c23a3f"
C_GREEN = "#43b581"
C_AMBER = "#faa61a"
C_BLUE = "#3aa8fc"
C_RED = "#f0494e"

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

STATUS_COLORS = {
    "idle": C_FG_MUTED, "loading": C_AMBER, "applying": C_AMBER,
    "listening": C_GREEN, "speaking": C_ACCENT, "stopping": C_AMBER,
    "stopped": C_FG_MUTED,
}


def _qss() -> str:
    chevron = (config.ASSETS_DIR / "ui" / "chevron-down.svg").as_posix()
    return f"""
    QWidget {{ background: {C_BG}; color: {C_FG}; font-size: 13px; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: {C_BG}; border: 0; }}
    /* Labels/checkboxes must be transparent so they blend with their card
       instead of painting the window background as a dark rectangle. */
    QLabel, QCheckBox {{ background: transparent; }}

    #Header {{ font-size: 19px; font-weight: 700; color: {C_FG}; }}
    #Subtitle {{ color: {C_FG_MUTED}; font-size: 12px; }}

    QFrame#Card {{
        background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 12px;
    }}
    QLabel#CardTitle {{
        color: {C_FG_MUTED}; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
    }}
    QLabel#FieldLabel {{ color: {C_FG_MUTED}; }}
    QLabel#Hint {{ color: {C_FG_MUTED}; font-size: 11px; }}

    QComboBox, QLineEdit, QSpinBox, QPlainTextEdit {{
        background: {C_INPUT}; border: 1px solid {C_BORDER}; border-radius: 8px;
        padding: 7px 9px; color: {C_FG}; selection-background-color: {C_ACCENT};
    }}
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{ border-color: #4b4f57; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QPlainTextEdit:focus {{
        border-color: {C_ACCENT};
    }}
    QComboBox::drop-down {{ border: 0; width: 26px; }}
    QComboBox::down-arrow {{ image: url("{chevron}"); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{
        background: {C_INPUT}; border: 1px solid {C_BORDER}; border-radius: 8px;
        selection-background-color: {C_ACCENT}; selection-color: white;
        outline: 0; padding: 4px;
    }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: 0; }}

    QCheckBox {{ spacing: 9px; color: {C_FG}; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px; border-radius: 5px;
        border: 1px solid {C_BORDER}; background: {C_INPUT};
    }}
    QCheckBox::indicator:hover {{ border-color: {C_ACCENT}; }}
    QCheckBox::indicator:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; }}
    QCheckBox:disabled {{ color: {C_FG_MUTED}; }}

    QPushButton {{
        background: {C_PANEL_HI}; border: 1px solid {C_BORDER}; border-radius: 8px;
        padding: 8px 14px; color: {C_FG};
    }}
    QPushButton:hover {{ background: #393c43; }}
    QPushButton:disabled {{ color: {C_FG_MUTED}; background: {C_PANEL}; }}
    QPushButton#Accent {{
        background: {C_ACCENT}; border: 0; color: white; font-weight: 600; padding: 10px 22px;
    }}
    QPushButton#Accent:hover {{ background: {C_ACCENT_HI}; }}
    QPushButton#Accent:disabled {{ background: #3a4170; color: #b9bfe8; }}
    QPushButton#Danger {{ background: {C_PANEL_HI}; border: 1px solid {C_BORDER}; color: #f0b6b7; }}
    QPushButton#Danger:hover {{ background: {C_DANGER_HI}; color: white; border-color: {C_DANGER_HI}; }}
    QPushButton#Danger:disabled {{ color: {C_FG_MUTED}; }}
    QPushButton#Ghost {{ background: transparent; border: 1px solid {C_BORDER}; }}
    QPushButton#Ghost:hover {{ background: {C_PANEL_HI}; }}

    QTextEdit#Log {{
        background: {C_INPUT}; border: 1px solid {C_BORDER}; border-radius: 10px;
        padding: 10px; font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
        font-size: 12px;
    }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: #43464d; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: #54585f; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    """


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {C_BORDER}; background: {C_BORDER}; max-height: 1px;")
    return f


class Card(QFrame):
    """A titled rounded panel holding a grid of fields."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        lbl = QLabel(title)
        lbl.setObjectName("CardTitle")
        outer.addWidget(lbl)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(9)
        self.grid.setColumnStretch(1, 1)
        outer.addLayout(self.grid)

    def label(self, row: int, text: str):
        lab = QLabel(text)
        lab.setObjectName("FieldLabel")
        self.grid.addWidget(lab, row, 0, Qt.AlignLeft | Qt.AlignVCenter)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS Voice Clone")
        self.setMinimumSize(540, 440)
        self.resize(720, min(960, 880))

        self.events_q: queue.Queue = queue.Queue()
        self.pipeline: Pipeline | None = None
        self.vmic = VirtualMic()
        self.settings = self._load_settings()
        self.ref_texts: dict = self.settings.get("ref_texts", {})
        self._voice_paths: dict[str, str] = {}
        self.personalities = load_personalities()
        self._preset_texts = dict(PRESET_TEXTS)
        for p in self.personalities:
            if p["ref_text"]:
                self._preset_texts[Path(p["audio"]).stem] = p["ref_text"]

        self._build_ui()
        self._refresh_devices()
        self._populate_voices()
        self._apply_settings()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_events)
        self.timer.start(80)

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # Header row: title + status pill
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        h = QLabel("Qwen3-TTS Voice Clone")
        h.setObjectName("Header")
        sub = QLabel("Realtime speech-to-speech · clone any voice")
        sub.setObjectName("Subtitle")
        title_box.addWidget(h)
        title_box.addWidget(sub)
        head.addLayout(title_box)
        head.addStretch(1)
        self.status_pill = QLabel("● idle")
        self.status_pill.setStyleSheet(self._pill_style(C_FG_MUTED))
        head.addWidget(self.status_pill, 0, Qt.AlignVCenter)
        root.addLayout(head)

        # Scrollable settings cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(12)
        self._build_devices(col)
        self._build_voice(col)
        self._build_generation(col)
        self._build_stt(col)
        self._build_routing(col)
        col.addStretch(1)
        scroll.setWidget(inner)
        scroll.setFocusPolicy(Qt.NoFocus)
        root.addWidget(scroll, 1)

        # Keep long combo contents (model ids, "Name — Category") from forcing the
        # card width: size to a short minimum and let the grid stretch handle it.
        for cb in self.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Controls
        ctrl = QHBoxLayout()
        self.apply_btn = QPushButton("Apply settings")
        self.apply_btn.setObjectName("Accent")
        self.apply_btn.clicked.connect(self._apply)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self.apply_btn)
        ctrl.addWidget(self.stop_btn)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        # Transcript / log
        self.log = QTextEdit()
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        # Compact + fixed-policy so it never steals space from the scrolling
        # settings area (which is the flexible region that scrolls when needed).
        self.log.setFixedHeight(110)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self.log)

        # Message box
        msg = QHBoxLayout()
        self.msg_entry = QLineEdit()
        self.msg_entry.setPlaceholderText("Type a message to speak in the cloned voice…")
        self.msg_entry.returnPressed.connect(self._send_text)
        send = QPushButton("Send")
        send.setObjectName("Accent")
        send.clicked.connect(self._send_text)
        msg.addWidget(self.msg_entry, 1)
        msg.addWidget(send)
        root.addLayout(msg)
        self.msg_entry.setFocus()  # avoid a stray focus ring on the scroll area

    def _pill_style(self, color: str) -> str:
        return (f"background: {C_PANEL}; border: 1px solid {C_BORDER}; "
                f"border-radius: 12px; padding: 5px 12px; color: {color}; font-weight: 600;")

    def _build_devices(self, col):
        card = Card("Devices")
        card.label(0, "Microphone")
        self.mic_combo = QComboBox()
        card.grid.addWidget(self.mic_combo, 0, 1)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setObjectName("Ghost")
        self.refresh_btn.setFixedWidth(40)
        self.refresh_btn.setToolTip("Rescan audio devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        card.grid.addWidget(self.refresh_btn, 0, 2)
        card.label(1, "Output")
        self.out_combo = QComboBox()
        card.grid.addWidget(self.out_combo, 1, 1, 1, 2)
        col.addWidget(card)

    def _build_voice(self, col):
        card = Card("Voice")
        card.label(0, "Clip")
        self.voice_combo = QComboBox()
        self.voice_combo.setMaxVisibleItems(18)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_selected)
        card.grid.addWidget(self.voice_combo, 0, 1)
        browse = QPushButton("Browse…")
        browse.setObjectName("Ghost")
        browse.clicked.connect(self._browse_voice)
        card.grid.addWidget(browse, 0, 2)
        card.label(1, "Reference text")
        self.ref_text = QPlainTextEdit()
        self.ref_text.setFixedHeight(70)
        card.grid.addWidget(self.ref_text, 1, 1)
        tx = QPushButton("Auto-\ntranscribe")
        tx.setObjectName("Ghost")
        tx.clicked.connect(self._auto_transcribe)
        card.grid.addWidget(tx, 1, 2)
        col.addWidget(card)

    def _build_generation(self, col):
        card = Card("Generation")
        card.label(0, "TTS model")
        self.tts_combo = QComboBox()
        self.tts_combo.setEditable(True)
        self.tts_combo.addItems(TTS_MODELS)
        self.tts_combo.setCurrentText(config.MODEL_ID)
        card.grid.addWidget(self.tts_combo, 0, 1, 1, 2)
        card.label(1, "Language")
        self.lang_combo = QComboBox()
        self.lang_combo.setEditable(True)
        self.lang_combo.addItems(LANGUAGES)
        self.lang_combo.setCurrentText(config.LANGUAGE)
        self.lang_combo.currentTextChanged.connect(self._on_language_change)
        card.grid.addWidget(self.lang_combo, 1, 1, 1, 2)
        card.label(2, "Instruct")
        self.instruct_edit = QLineEdit(config.INSTRUCT)
        self.instruct_edit.setPlaceholderText("Style/tone prompt (optional)")
        self.instruct_edit.textChanged.connect(self._on_instruct_change)
        card.grid.addWidget(self.instruct_edit, 2, 1, 1, 2)
        self.expressive_chk = QCheckBox(
            "Expressive clone — follow Instruct (1.7B CustomVoice; ignores TTS-model choice)")
        card.grid.addWidget(self.expressive_chk, 3, 0, 1, 3)
        self.icl_chk = QCheckBox(
            "↳ ICL mode — use reference audio + text (stronger; needs accurate reference text)")
        self.icl_chk.setChecked(True)
        card.grid.addWidget(self.icl_chk, 4, 0, 1, 3)
        col.addWidget(card)

    def _build_stt(self, col):
        card = Card("Speech-to-text")
        card.label(0, "STT model")
        self.stt_combo = QComboBox()
        self.stt_combo.setEditable(True)
        self.stt_combo.addItems(STT_MODELS)
        self.stt_combo.setCurrentText(config.STT_MODEL)
        card.grid.addWidget(self.stt_combo, 0, 1)
        card.label(1, "End-of-speech silence (ms)")
        self.vad_spin = QSpinBox()
        self.vad_spin.setRange(200, 2000)
        self.vad_spin.setSingleStep(100)
        self.vad_spin.setValue(int(config.VAD_SILENCE_MS))
        card.grid.addWidget(self.vad_spin, 1, 1, Qt.AlignLeft)
        col.addWidget(card)

    def _build_routing(self, col):
        card = Card("Routing")
        self.vmic_chk = QCheckBox(f"Expose as virtual microphone ('{self.vmic.source_name}')")
        avail = VirtualMic.available()
        self.vmic_chk.setChecked(avail)
        self.vmic_chk.setEnabled(avail)
        self.vmic_chk.toggled.connect(self._on_vmic_toggle)
        card.grid.addWidget(self.vmic_chk, 0, 0, 1, 3)
        self.duplex_chk = QCheckBox("Mute mic while speaking (echo guard — uncheck for barge-in)")
        self.duplex_chk.setChecked(not avail)
        card.grid.addWidget(self.duplex_chk, 1, 0, 1, 3)
        if not avail:
            hint = QLabel("Virtual mic needs PulseAudio/PipeWire (pactl). "
                          "On Windows, use VB-CABLE as the output device.")
            hint.setObjectName("Hint")
            hint.setWordWrap(True)
            card.grid.addWidget(hint, 2, 0, 1, 3)
        col.addWidget(card)

    # ---- devices / voices ------------------------------------------------
    def _refresh_devices(self):
        import sounddevice as sd

        sd._terminate()
        sd._initialize()
        prev_mic, prev_out = self.mic_combo.currentText(), self.out_combo.currentText()
        self.mic_map = {AUTO_LABEL: None}
        self.out_map = {AUTO_LABEL: None}
        for i, d in enumerate(sd.query_devices()):
            label = f"[{i}] {d['name']}"
            if d["max_input_channels"] > 0:
                self.mic_map[label] = i
            if d["max_output_channels"] > 0:
                self.out_map[label] = i
        for combo, mapping, prev in ((self.mic_combo, self.mic_map, prev_mic),
                                     (self.out_combo, self.out_map, prev_out)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(mapping))
            combo.setCurrentText(prev if prev in mapping else AUTO_LABEL)
            combo.blockSignals(False)

    def _populate_voices(self):
        """List clips in assets/ then personalities, grouped with separators."""
        self._voice_paths.clear()
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()

        def add(label: str, path: str):
            self._voice_paths[label] = path
            self.voice_combo.addItem(label)

        seen_stems = set()
        for f in sorted(config.ASSETS_DIR.glob("*")):
            if f.suffix.lower() in AUDIO_EXTS and f.stem not in seen_stems:
                seen_stems.add(f.stem)
                wav = f.with_suffix(".wav")
                add(f.stem, str(wav if wav.exists() else f))

        # Only tag the category when there's more than one (avoids a redundant
        # "— Popular Voices" on every row when the catalog is single-category).
        multi_cat = len({p["category"] for p in self.personalities}) > 1
        last_cat = None
        for p in self.personalities:
            if last_cat != p["category"]:
                if self.voice_combo.count():
                    self.voice_combo.insertSeparator(self.voice_combo.count())
                last_cat = p["category"]
            label = f"{p['name']} — {p['category']}" if multi_cat else p["name"]
            add(label, p["audio"])

        # default selection: config reference clip's stem, else first real item
        default_stem = Path(config.REF_AUDIO_MP3).stem
        self.voice_combo.setCurrentText(
            default_stem if default_stem in self._voice_paths
            else next(iter(self._voice_paths), ""))
        self.voice_combo.blockSignals(False)
        self._on_voice_selected()

    def _current_voice_path(self) -> str | None:
        return self._voice_paths.get(self.voice_combo.currentText())

    def _on_voice_selected(self, *_):
        path = self._current_voice_path()
        if not path:
            return
        text = self.ref_texts.get(path) or self._preset_texts.get(Path(path).stem)
        if text is not None:
            self.ref_text.setPlainText(text)

    def _browse_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a reference voice clip", str(config.ASSETS_DIR),
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.opus);;All files (*)")
        if not path:
            return
        label = Path(path).name
        self._voice_paths[label] = path
        self.voice_combo.blockSignals(True)
        self.voice_combo.addItem(label)
        self.voice_combo.setCurrentText(label)
        self.voice_combo.blockSignals(False)
        self._on_voice_selected()

    def _get_ref_text(self) -> str:
        return self.ref_text.toPlainText().strip()

    # ---- auto-transcribe -------------------------------------------------
    def _auto_transcribe(self):
        path = self._current_voice_path()
        if not path:
            return
        self._log(f"transcribing {Path(path).name}…", C_FG_MUTED)
        threading.Thread(target=self._do_transcribe, args=(path,), daemon=True).start()

    def _do_transcribe(self, path: str):
        try:
            from faster_whisper import WhisperModel

            device, index = _parse_cuda(config.DEVICE)
            compute = config.STT_COMPUTE if device == "cuda" else "int8"
            model = WhisperModel(self.stt_combo.currentText().strip() or config.STT_MODEL,
                                 device=device, device_index=index, compute_type=compute)
            segments, _ = model.transcribe(str(path), language="en", beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                self.events_q.put({"type": "_ref_text", "text": text})
                self.events_q.put({"type": "info", "text": "transcription done"})
            else:
                self.events_q.put({"type": "info", "text": "transcription: no speech detected"})
        except Exception as exc:  # noqa: BLE001
            self.events_q.put({"type": "error", "text": f"transcribe failed: {exc}"})

    # ---- control ---------------------------------------------------------
    @staticmethod
    def _selected(combo: QComboBox, mapping: dict):
        return mapping.get(combo.currentText())

    def _on_vmic_toggle(self, on: bool):
        self.duplex_chk.setChecked(not on)

    def _on_instruct_change(self, *_):
        if self.pipeline and self.pipeline.running:
            self.pipeline.set_instruct(self.instruct_edit.text())

    def _on_language_change(self, *_):
        if self.pipeline and self.pipeline.running:
            self.pipeline.set_language(self.lang_combo.currentText())

    def _send_text(self):
        text = self.msg_entry.text().strip()
        if not text:
            return
        if not (self.pipeline and self.pipeline.running):
            self._apply()
        if self.pipeline and self.pipeline.say(text):
            self._log(f"[type] {text}", C_BLUE)
            self.msg_entry.clear()
        else:
            self._log("couldn't queue message — press Apply settings first", C_FG_MUTED)

    def _collect_config(self) -> dict:
        voice_path = self._current_voice_path()
        ref_text = self._get_ref_text()
        if voice_path:
            self.ref_texts[voice_path] = ref_text
        return dict(
            input_device=self._selected(self.mic_combo, self.mic_map),
            output_device=self._selected(self.out_combo, self.out_map),
            voice_path=voice_path,
            ref_text=ref_text,
            tts_model=self.tts_combo.currentText().strip(),
            stt_model=self.stt_combo.currentText().strip(),
            vad=str(self.vad_spin.value()),
            instruct=self.instruct_edit.text(),
            language=self.lang_combo.currentText(),
            expressive=self.expressive_chk.isChecked(),
            icl=self.icl_chk.isChecked(),
            vmic=self.vmic_chk.isChecked(),
            duplex=self.duplex_chk.isChecked(),
        )

    def _apply(self):
        cfg = self._collect_config()
        self._save_settings()
        self.apply_btn.setEnabled(False)

        if not (self.pipeline and self.pipeline.running):
            self._set_status("loading")
            self._launch(cfg)
            return

        self._set_status("applying")

        def worker():
            # Keep the virtual mic up across the restart so other apps (Discord,
            # Zoom…) don't see the device vanish and fall back to another mic.
            self._teardown(stop_vmic=False)
            self.events_q.put({"type": "_relaunch", "cfg": cfg})

        threading.Thread(target=worker, daemon=True).start()

    def _launch(self, cfg: dict):
        if cfg["tts_model"]:
            config.MODEL_ID = cfg["tts_model"]
        config.STT_MODEL = cfg["stt_model"] or config.STT_MODEL
        try:
            config.VAD_SILENCE_MS = int(cfg["vad"])
        except ValueError:
            pass

        output_device = cfg["output_device"]
        if cfg["vmic"]:
            was_active = self.vmic.active
            try:
                sink = self.vmic.create()  # idempotent: reuses existing modules
            except Exception as exc:  # noqa: BLE001
                self._log(f"virtual mic failed: {exc}", C_RED)
                self.apply_btn.setEnabled(True)
                self._set_status("idle")
                return
            os.environ["PULSE_SINK"] = sink
            output_device = "pulse"
            if not was_active:  # only announce when newly created, not on restart
                self._log(f"virtual mic '{self.vmic.source_name}' active — select it as "
                          "the microphone in your other app", C_FG_MUTED)
        elif self.vmic.active:  # vmic was turned off → tear it down now
            self.vmic.destroy()
            os.environ.pop("PULSE_SINK", None)
        else:
            os.environ.pop("PULSE_SINK", None)

        if cfg["expressive"]:
            self._log("expressive clone on — 1.7B CustomVoice; first use extracts "
                      "the voice embedding (may take a few seconds)", C_FG_MUTED)

        self.pipeline = Pipeline(
            cfg["input_device"], output_device, cfg["instruct"], self.events_q,
            half_duplex=cfg["duplex"], ref_audio=cfg["voice_path"],
            ref_text=cfg["ref_text"] or None, language=cfg["language"] or None,
            expressive=cfg["expressive"], icl=cfg["icl"])
        self.pipeline.start()
        self.apply_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        self.stop_btn.setEnabled(False)
        self._set_status("stopping")

        def worker():
            self._teardown()
            unload_models()
            self.events_q.put({"type": "status", "value": "stopped"})

        threading.Thread(target=worker, daemon=True).start()

    def _teardown(self, stop_vmic: bool = True):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        # On an Apply restart we keep the virtual mic alive (stop_vmic=False) so
        # downstream apps stay bound to it; Stop/close tear it down fully.
        if stop_vmic and self.vmic.active:
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
                elif kind == "_ref_text":
                    self.ref_text.setPlainText(ev["text"])
                elif kind == "status":
                    self._set_status(ev["value"])
                elif kind == "user":
                    self._log(f"[you] {ev['text']}", C_GREEN)
                elif kind == "error":
                    self._log(ev["text"], C_RED)
                else:
                    self._log(ev["text"], C_FG_MUTED)
        except queue.Empty:
            pass

    def _set_status(self, value: str):
        self.status_pill.setText(f"● {value}")
        self.status_pill.setStyleSheet(self._pill_style(STATUS_COLORS.get(value, C_FG_MUTED)))

    def _log(self, text: str, color: str = C_FG):
        safe = html.escape(text)
        self.log.append(f'<span style="color:{color}">{safe}</span>')
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- settings persistence -------------------------------------------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def _save_settings(self):
        data = {
            "mic": self.mic_combo.currentText(),
            "output": self.out_combo.currentText(),
            "voice": self.voice_combo.currentText(),
            "tts_model": self.tts_combo.currentText(),
            "language": self.lang_combo.currentText(),
            "instruct": self.instruct_edit.text(),
            "stt_model": self.stt_combo.currentText(),
            "vad_ms": str(self.vad_spin.value()),
            "vmic": self.vmic_chk.isChecked(),
            "duplex": self.duplex_chk.isChecked(),
            "expressive": self.expressive_chk.isChecked(),
            "icl": self.icl_chk.isChecked(),
            "ref_texts": self.ref_texts,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception:  # noqa: BLE001
            pass

    def _apply_settings(self):
        s = self.settings
        if not s:
            return
        if s.get("mic") in self.mic_map:
            self.mic_combo.setCurrentText(s["mic"])
        if s.get("output") in self.out_map:
            self.out_combo.setCurrentText(s["output"])
        if s.get("voice") in self._voice_paths:
            self.voice_combo.setCurrentText(s["voice"])
            self._on_voice_selected()
        if s.get("tts_model"):
            self.tts_combo.setCurrentText(s["tts_model"])
        if s.get("language"):
            self.lang_combo.setCurrentText(s["language"])
        if s.get("instruct") is not None:
            self.instruct_edit.setText(s["instruct"])
        if s.get("stt_model"):
            self.stt_combo.setCurrentText(s["stt_model"])
        if s.get("vad_ms"):
            try:
                self.vad_spin.setValue(int(s["vad_ms"]))
            except (TypeError, ValueError):
                pass
        if "vmic" in s and self.vmic_chk.isEnabled():
            self.vmic_chk.setChecked(bool(s["vmic"]))
        if "duplex" in s:
            self.duplex_chk.setChecked(bool(s["duplex"]))
        if "expressive" in s:
            self.expressive_chk.setChecked(bool(s["expressive"]))
        if "icl" in s:
            self.icl_chk.setChecked(bool(s["icl"]))

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        try:
            self._save_settings()
            self._teardown()
            unload_models()
        finally:
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Qwen3-TTS Voice Clone")
    app.setFont(QFont(app.font().family(), 10))
    app.setStyleSheet(_qss())
    win = App()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
