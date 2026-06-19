#!/usr/bin/env bash
# One-shot setup with uv: create the venv, install Qwen3-TTS + deps, prep the clip.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: https://docs.astral.sh/uv/getting-started/" >&2
  exit 1
fi

echo ">> uv sync (creates .venv, installs deps + dev group)"
uv sync

echo ">> Preparing reference audio (MP3 -> mono WAV)"
uv run python scripts/prep_audio.py

echo
echo "Setup complete. Next steps:"
echo "  uv run pytest                 # fast environment tests"
echo "  uv run pytest --run-clone     # full voice-cloning test (downloads model, uses GPU)"
echo "  uv run python -m src.clone_voice \"Hello from a cloned voice.\" outputs/hello.wav"
