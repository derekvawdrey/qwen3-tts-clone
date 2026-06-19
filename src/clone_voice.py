"""Core voice-cloning helpers around the Qwen3-TTS package.

Usage (CLI):
    python -m src.clone_voice "Text to speak in the cloned voice." out.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.helpers import StreamPlayer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

# Experimental "expressive clone": clone identity + instruction style together.
# The base clone checkpoint ignores instructions, so we extract the speaker
# x-vector with the Base model and synthesize on the CustomVoice model (which is
# trained to follow instructions), feeding the embedding as the speaker.
EXPRESSIVE_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
EXPRESSIVE_EMBED_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _torch_dtype():
    import torch
    return getattr(torch, _DTYPES.get(config.DTYPE, "bfloat16"))


def ensure_reference_wav() -> Path:
    """Make sure the prepped mono WAV exists; build it from the MP3 if missing."""
    if not config.REF_AUDIO_WAV.exists():
        from scripts import prep_audio

        if prep_audio.main() != 0:
            raise RuntimeError("Failed to prepare reference WAV from MP3.")
    return config.REF_AUDIO_WAV


def prepare_reference(audio_path: str | Path) -> Path:
    """Decode any audio file to a mono WAV at the model's sample rate.

    Reuses scripts.prep_audio.load_audio (soundfile/librosa, no ffmpeg needed)
    and caches the result under outputs/refs/<stem>.wav.
    """
    from scripts.prep_audio import load_audio
    import soundfile as sf

    audio_path = Path(audio_path)
    out_dir = config.OUTPUT_DIR / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{audio_path.stem}.wav"
    # Reuse the cached WAV unless the source is newer.
    if dst.exists() and dst.stat().st_mtime >= audio_path.stat().st_mtime:
        return dst
    data, sr = load_audio(audio_path, config.SAMPLE_RATE)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return dst


_MODEL_CACHE: dict = {"id": None, "model": None}


def load_model(model_id: str | None = None):
    """Load a Qwen3-TTS model, keeping one resident (defaults to config.MODEL_ID).

    Uses the CUDA-graph-accelerated `faster-qwen3-tts` runtime. Switching
    model_id **frees the previous checkpoint first** (so two never sit in VRAM
    at once), then loads the new one.
    """
    model_id = model_id or config.MODEL_ID
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["id"] == model_id:
        return _MODEL_CACHE["model"]

    unload_models()  # release the previous checkpoint before loading another
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        model_id, device=config.DEVICE, dtype=_torch_dtype(),
        attn_implementation=config.ATTN_IMPL,
    )
    _MODEL_CACHE.update(id=model_id, model=model)
    return model


def unload_models():
    """Drop the cached TTS model and release its GPU memory."""
    import gc

    _MODEL_CACHE["id"] = None
    _MODEL_CACHE["model"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_speaker_embedding(ref_wav: str):
    """Return the speaker x-vector for a prepared reference WAV.

    Cached on disk (<ref_wav>.spk.pt). Extracted with the 1.7B Base model's
    speaker encoder — loaded transiently and freed, so it only costs VRAM the
    first time a voice is seen (before the .pt cache exists). The cache is
    invalidated if the source WAV is newer (e.g. a clip was replaced).
    """
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    ref_path = Path(ref_wav)
    pt_path = ref_path.with_suffix(".spk.pt")
    if pt_path.exists() and pt_path.stat().st_mtime >= ref_path.stat().st_mtime:
        return torch.load(pt_path, map_location="cpu")

    base = FasterQwen3TTS.from_pretrained(
        EXPRESSIVE_EMBED_MODEL, device=config.DEVICE, dtype=_torch_dtype(),
        attn_implementation=config.ATTN_IMPL)
    try:
        items = base.model.create_voice_clone_prompt(
            ref_audio=ref_wav, ref_text="", x_vector_only_mode=True)
        emb = items[0].ref_spk_embedding.detach().to("cpu")
    finally:
        del base
        torch.cuda.empty_cache()
    torch.save(emb, pt_path)
    return emb


def synthesize_stream(text, *, language=None, ref_audio=None, ref_text=None,
                      instruct=None, expressive=False):
    """Yield (audio_chunk, sample_rate) for `text` in the cloned voice.

    expressive=False: standard fast clone on ``config.MODEL_ID`` (instruct is
    accepted but the base model follows it unreliably).
    expressive=True:  clone identity + instruction via the 1.7B CustomVoice
    model, feeding the reference's extracted x-vector as the speaker.
    """
    ref_wav = str(prepare_reference(ref_audio) if ref_audio else ensure_reference_wav())
    ref_text = config.REF_TEXT if ref_text is None else ref_text
    language = language or config.LANGUAGE
    instruct = config.INSTRUCT if instruct is None else instruct

    if expressive:
        emb = get_speaker_embedding(ref_wav).to(config.DEVICE).to(_torch_dtype())
        model = load_model(EXPRESSIVE_TTS_MODEL)
        stream = model.generate_voice_clone_streaming(
            text=text, language=language,
            voice_clone_prompt={"ref_spk_embedding": [emb]},
            instruct=instruct or None, xvec_only=True, chunk_size=8)
    else:
        model = load_model()
        stream = model.generate_voice_clone_streaming(
            text=text, language=language, ref_audio=ref_wav, ref_text=ref_text,
            instruct=instruct or None, chunk_size=8)

    for chunk, sr, _timing in stream:
        yield chunk, sr


def clone_to_file(
    text: str,
    out_path: str | Path,
    *,
    language: str | None = None,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    expressive: bool = False,
) -> Path:
    """Synthesize `text` in the cloned reference voice and write a WAV.

    `instruct` is a natural-language style prompt (tone/emotion/pacing), passed
    through to `generate_voice_clone`. Defaults to ``config.INSTRUCT``; pass an
    empty string to disable.

    NOTE: instruct + voice cloning is experimental and unreliable — the base
    Qwen3-TTS checkpoint is not trained to follow instructions while cloning, so
    the style prompt is often weakly followed or ignored (faster-qwen3-tts warns
    about this). Reliable instruction control lives in the separate VoiceDesign
    (no clone) / CustomVoice (named speakers, 1.7B) checkpoints.

    Set ``expressive=True`` for the experimental cloned-identity + instruction
    mode (1.7B CustomVoice; see ``synthesize_stream``).

    Returns the path to the written file.
    """
    import numpy as np
    import soundfile as sf

    chunks, sr = [], config.SAMPLE_RATE
    for chunk, sr in synthesize_stream(
        text, language=language, ref_audio=ref_audio, ref_text=ref_text,
        instruct=instruct, expressive=expressive,
    ):
        chunks.append(chunk)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.concatenate(chunks), sr)
    return out_path


def clone_to_speaker(
    text: str,
    language: str | None = None,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    expressive: bool = False,
) -> None:
    """Synthesize `text` in the cloned reference voice and stream to the speaker."""
    play = StreamPlayer()
    try:
        for chunk, sr in synthesize_stream(
            text, language=language, ref_audio=ref_audio, ref_text=ref_text,
            instruct=instruct, expressive=expressive,
        ):
            play(chunk, sr)
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
