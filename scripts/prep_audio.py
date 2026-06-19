#!/usr/bin/env python3
"""Convert the reference MP3 into a clean mono WAV for voice cloning.

Tries soundfile first (bundled libsndfile >= 1.1 decodes MP3 natively, no system
ffmpeg needed), then falls back to librosa/audioread. Writes a mono WAV at the
configured sample rate that the Qwen3-TTS model loads as `ref_audio`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Return (mono_float32, sample_rate) resampled to target_sr."""
    # 1) soundfile (handles MP3 with modern libsndfile)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = _to_mono(np.asarray(data))
    except Exception as sf_err:  # noqa: BLE001
        # 2) librosa fallback (audioread; may need ffmpeg for some codecs)
        try:
            import librosa

            data, sr = librosa.load(str(path), sr=None, mono=True)
            data = data.astype(np.float32)
        except Exception as lr_err:  # noqa: BLE001
            raise RuntimeError(
                f"Could not decode {path}.\n"
                f"  soundfile error: {sf_err}\n"
                f"  librosa error:   {lr_err}\n"
                "Install a newer soundfile (pip install -U soundfile) or system "
                "ffmpeg (e.g. `sudo apt install ffmpeg`)."
            ) from lr_err

    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr


def main() -> int:
    src = config.REF_AUDIO_MP3
    dst = config.REF_AUDIO_WAV
    if not src.exists():
        print(f"ERROR: reference audio not found: {src}", file=sys.stderr)
        return 1

    data, sr = load_audio(src, config.SAMPLE_RATE)

    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sr, subtype="PCM_16")

    dur = len(data) / sr
    print(f"OK: {src.name} -> {dst}  ({dur:.1f}s, {sr} Hz, mono)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
