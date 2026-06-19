"""Manage a PulseAudio/PipeWire virtual microphone via `pactl`.

Creates a null sink plus a remapped source so other apps (Zoom, Discord, OBS,
the browser) see a normal-looking microphone. The TTS is played into the sink
(route playback there with the PULSE_SINK env var or by selecting it as the
output device); whatever lands in the sink is exposed on the source.

    sink:   QwenTTS            (play TTS here)
    source: QwenTTS_Microphone (other apps select this as their mic)
"""
from __future__ import annotations

import shutil
import subprocess

SINK_NAME = "QwenTTS"
SOURCE_NAME = "QwenTTS_Microphone"


class VirtualMic:
    def __init__(self, sink_name: str = SINK_NAME, source_name: str = SOURCE_NAME):
        self.sink_name = sink_name
        self.source_name = source_name
        self._module_ids: list[str] = []

    @staticmethod
    def available() -> bool:
        """True if `pactl` exists (PulseAudio/PipeWire pulse interface)."""
        return shutil.which("pactl") is not None

    @property
    def active(self) -> bool:
        return bool(self._module_ids)

    def _pactl(self, *args: str) -> str:
        return subprocess.run(
            ["pactl", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def _cleanup_existing(self) -> None:
        """Unload any leftover modules from a previous crashed run."""
        try:
            out = self._pactl("list", "short", "modules")
        except Exception:
            return
        for line in out.splitlines():
            if self.sink_name in line or self.source_name in line:
                mid = line.split("\t", 1)[0]
                try:
                    self._pactl("unload-module", mid)
                except Exception:
                    pass

    def create(self) -> str:
        """Create the sink + source. Returns the sink name to play TTS into."""
        if self._module_ids:
            return self.sink_name
        if not self.available():
            raise RuntimeError("pactl not found; cannot create a virtual microphone")

        self._cleanup_existing()
        sink_id = self._pactl(
            "load-module", "module-null-sink",
            f"sink_name={self.sink_name}",
            f"sink_properties=device.description={self.sink_name}",
        )
        self._module_ids.append(sink_id)
        try:
            src_id = self._pactl(
                "load-module", "module-remap-source",
                f"master={self.sink_name}.monitor",
                f"source_name={self.source_name}",
                f"source_properties=device.description={self.source_name}",
            )
            self._module_ids.append(src_id)
        except Exception:
            self.destroy()  # roll back the sink if the source fails
            raise
        return self.sink_name

    def destroy(self) -> None:
        for mid in reversed(self._module_ids):
            try:
                self._pactl("unload-module", mid)
            except Exception:
                pass
        self._module_ids = []
