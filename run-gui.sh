#!/usr/bin/env bash
# Launch the Tkinter speech-to-speech GUI.
# Ensures the `realtime` extra (faster-whisper, silero-vad, sounddevice) is
# installed first — the GUI imports src.pipeline which needs it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run ./setup.sh first (https://docs.astral.sh/uv/)." >&2
  exit 1
fi

echo ">> Ensuring GUI deps (PySide6 + faster-whisper, silero-vad, sounddevice)"
# --inexact: add the gui extra without uninstalling anything outside it
# (e.g. an optionally-installed flash-attn from `make flash-attn`).
uv sync --inexact --extra gui

echo ">> Launching GUI"
exec uv run python -m src.gui
