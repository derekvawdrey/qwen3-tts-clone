#!/usr/bin/env python3
"""Optional: auto-fill REF_TEXT by transcribing the reference clip with Whisper.

Prints the transcript and writes it to assets/ref_text.txt. Wire it into config
by setting:  export REF_TEXT="$(cat assets/ref_text.txt)"

Requires faster-whisper (pip install faster-whisper).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> int:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
        return 1

    src = config.REF_AUDIO_WAV if config.REF_AUDIO_WAV.exists() else config.REF_AUDIO_MP3
    model = WhisperModel("base.en", device="cuda" if _cuda() else "cpu", compute_type="float16" if _cuda() else "int8")
    segments, _ = model.transcribe(str(src))
    text = " ".join(s.text.strip() for s in segments).strip()

    out = config.ASSETS_DIR / "ref_text.txt"
    out.write_text(text + "\n")
    print(text)
    print(f"\n(written to {out})", file=sys.stderr)
    return 0


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
