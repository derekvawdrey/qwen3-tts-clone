"""Tkinter control panel for the realtime voice-clone pipeline.

Pick a microphone and an output, optionally expose the cloned voice as a
virtual microphone other apps can select, then Start/Stop and watch the live
transcript.

    python -m src.gui
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.virtual_mic import VirtualMic  # noqa: E402

AUTO_LABEL = "Auto (PulseAudio default)"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Qwen3-TTS Voice")
        root.minsize(560, 460)

        self.events_q: queue.Queue = queue.Queue()
        self.pipeline: Pipeline | None = None
        self.vmic = VirtualMic()

        self._build_widgets()
        self._refresh_devices()
        root.after(100, self._poll_events)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout ----------------------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root)
        frm.pack(fill="x")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Microphone:").grid(row=0, column=0, sticky="w", **pad)
        self.mic_combo = ttk.Combobox(frm, state="readonly")
        self.mic_combo.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(frm, text="Output:").grid(row=1, column=0, sticky="w", **pad)
        self.out_combo = ttk.Combobox(frm, state="readonly")
        self.out_combo.grid(row=1, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Button(frm, text="↻ Refresh devices", command=self._refresh_devices) \
            .grid(row=2, column=2, sticky="e", **pad)

        ttk.Label(frm, text="Instruct:").grid(row=3, column=0, sticky="w", **pad)
        self.instruct_var = tk.StringVar(value=config.INSTRUCT)
        ttk.Entry(frm, textvariable=self.instruct_var) \
            .grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        self.vmic_var = tk.BooleanVar(value=VirtualMic.available())
        vmic_chk = ttk.Checkbutton(
            frm,
            text=f"Expose as virtual microphone ('{self.vmic.source_name}')",
            variable=self.vmic_var, command=self._on_vmic_toggle,
        )
        vmic_chk.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        if not VirtualMic.available():
            vmic_chk.configure(state="disabled")
            self.vmic_var.set(False)

        # Half-duplex (echo guard) is only needed when TTS plays through speakers
        # that the mic can hear. With the virtual mic (or headphones) it's not,
        # and turning it off lets you barge in while the voice is talking.
        self.duplex_var = tk.BooleanVar(value=not self.vmic_var.get())
        ttk.Checkbutton(
            frm,
            text="Mute mic while speaking (echo guard — uncheck for barge-in)",
            variable=self.duplex_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", **pad)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x")
        self.start_btn = ttk.Button(ctrl, text="▶ Start", command=self._start)
        self.start_btn.pack(side="left", **pad)
        self.stop_btn = ttk.Button(ctrl, text="■ Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", **pad)
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(ctrl, textvariable=self.status_var, foreground="#555") \
            .pack(side="right", **pad)

        ttk.Separator(self.root).pack(fill="x", padx=8, pady=4)

        self.log = tk.Text(self.root, height=14, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.tag_configure("you", foreground="#0a6")
        self.log.tag_configure("err", foreground="#c00")
        self.log.tag_configure("dim", foreground="#888")

    # ---- devices ---------------------------------------------------------
    def _refresh_devices(self):
        import sounddevice as sd

        sd._terminate()  # force PortAudio to re-scan (picks up new virtual devices)
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

    @staticmethod
    def _selected(combo, mapping):
        return mapping.get(combo.get(), None)

    def _on_vmic_toggle(self):
        # Virtual mic / headphones have no acoustic echo path, so default the
        # echo guard off when the virtual mic is enabled.
        self.duplex_var.set(not self.vmic_var.get())

    # ---- control ---------------------------------------------------------
    def _start(self):
        if self.pipeline and self.pipeline.running:
            return
        input_device = self._selected(self.mic_combo, self.mic_map)
        output_device = self._selected(self.out_combo, self.out_map)
        instruct = self.instruct_var.get()

        if self.vmic_var.get():
            try:
                sink = self.vmic.create()
            except Exception as exc:
                self._log(f"virtual mic failed: {exc}", "err")
                return
            os.environ["PULSE_SINK"] = sink         # route playback into the sink
            output_device = "pulse"
            self._log(f"virtual mic '{self.vmic.source_name}' active — "
                      f"select it as the microphone in your other app", "dim")
        else:
            os.environ.pop("PULSE_SINK", None)

        self.pipeline = Pipeline(input_device, output_device, instruct,
                                 self.events_q, half_duplex=self.duplex_var.get())
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
        # re-enable Start from the UI thread
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
                else:  # info
                    self._log(ev["text"], "dim")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _log(self, text: str, tag: str | None = None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        try:
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
