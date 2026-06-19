"""Core voice-cloning helpers around the Qwen3-TTS package.

Usage (CLI):
    python -m src.clone_voice "Text to speak in the cloned voice." out.wav
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from src.helpers import StreamPlayer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def ensure_reference_wav() -> Path:
    """Make sure the prepped mono WAV exists; build it from the MP3 if missing."""
    if not config.REF_AUDIO_WAV.exists():
        from scripts import prep_audio

        if prep_audio.main() != 0:
            raise RuntimeError("Failed to prepare reference WAV from MP3.")
    return config.REF_AUDIO_WAV


@lru_cache(maxsize=1)
def load_model():
    """Load the Qwen3-TTS Base model once (cached for reuse across calls).

    Uses the CUDA-graph-accelerated `faster-qwen3-tts` runtime, a drop-in
    reimplementation of the official package (~5x faster inference).
    """
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    dtype = getattr(torch, _DTYPES.get(config.DTYPE, "bfloat16"))
    model = FasterQwen3TTS.from_pretrained(
        config.MODEL_ID,
        device=config.DEVICE,
        dtype=dtype,
        attn_implementation=config.ATTN_IMPL,
    )
    return model


def clone_to_file(
    text: str,
    out_path: str | Path,
    *,
    language: str | None = None,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
) -> Path:
    """Synthesize `text` in the cloned reference voice and write a WAV.

    `instruct` is a natural-language style prompt steering *how* the text is
    delivered (tone/emotion/pacing). Defaults to ``config.INSTRUCT``; pass an
    empty string to disable.

    Returns the path to the written file.
    """
    import soundfile as sf

    ref_audio = Path(ref_audio) if ref_audio else ensure_reference_wav()
    ref_text = ref_text if ref_text is not None else config.REF_TEXT
    language = language or config.LANGUAGE
    instruct = instruct if instruct is not None else config.INSTRUCT

    model = load_model()
    wavs, sr = model.generate_voice_clone_streaming(
        text=text,
        language=language,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        instruct=instruct or None,
        chunk_size=8
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    return out_path

def clone_to_speaker(
    text: str,
    language: str | None = None,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
) -> None:
    """Synthesize `text` in the cloned reference voice and write a WAV.

    `instruct` is a natural-language style prompt steering *how* the text is
    delivered (tone/emotion/pacing). Defaults to ``config.INSTRUCT``; pass an
    empty string to disable.

    Returns the path to the written file.
    """
    import soundfile as sf

    ref_audio = Path(ref_audio) if ref_audio else ensure_reference_wav()
    ref_text = ref_text if ref_text is not None else config.REF_TEXT
    language = language or config.LANGUAGE
    instruct = instruct if instruct is not None else config.INSTRUCT

    model = load_model()
    play = StreamPlayer()
    try:
        for audio_chunk, sr, _ in model.generate_voice_clone_streaming(
            text=text, language="English",
            ref_audio=ref_audio, ref_text=ref_text,
            chunk_size=8,
        ):
            play(audio_chunk, sr)
    finally:
        play.close()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            'Usage: python -m src.clone_voice "Text to speak" [out.wav]',
            file=sys.stderr,
        )
        return 2
    text = argv[1]
    out = argv[2] if len(argv) > 2 else str(config.OUTPUT_DIR / "clone.wav")
    clone_to_speaker(text, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
