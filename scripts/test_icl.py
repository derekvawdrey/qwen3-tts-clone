#!/usr/bin/env python3
"""Probe which ICL-mode path works for expressive clone on the CustomVoice model.

Attempt 1: direct ICL on CustomVoice (ref_audio + ref_text, xvec_only=False).
Attempt 2: build the ICL prompt with the Base model, feed it to CustomVoice.

Writes whichever succeed to outputs/ for an ear check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEXT = "We have the best numbers, nobody has ever seen anything like it."
INSTRUCT = "Speak with restrained frustration — slow and tense."


def main() -> int:
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    from src.clone_voice import _torch_dtype, prepare_reference

    ref_wav = str(prepare_reference(config.REF_AUDIO_MP3))
    ref_text = config.REF_TEXT
    print(f"ref: {ref_wav}\nref_text: {ref_text[:60]}…\n")

    cv = FasterQwen3TTS.from_pretrained(
        CUSTOM, device=config.DEVICE, dtype=_torch_dtype(),
        attn_implementation=config.ATTN_IMPL)

    # Attempt 1 — direct ICL on the CustomVoice model.
    try:
        audio, sr = cv.generate_voice_clone(
            text=TEXT, language="English", ref_audio=ref_wav, ref_text=ref_text,
            instruct=INSTRUCT, xvec_only=False)
        sf.write("outputs/icl_direct.wav", audio[0], sr)
        print(f"[1] DIRECT ICL OK -> outputs/icl_direct.wav ({len(audio[0])/sr:.2f}s)")
    except Exception as exc:
        print(f"[1] DIRECT ICL FAILED: {type(exc).__name__}: {exc}")

    # Attempt 2 — build the ICL prompt with Base, then feed CustomVoice.
    try:
        base = FasterQwen3TTS.from_pretrained(
            BASE, device=config.DEVICE, dtype=_torch_dtype(),
            attn_implementation=config.ATTN_IMPL)
        items = base.model.create_voice_clone_prompt(
            ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=False)
        item = items[0]
        dev = cv.model.talker.device
        item.ref_spk_embedding = item.ref_spk_embedding.to(dev).to(_torch_dtype())
        if item.ref_code is not None:
            item.ref_code = item.ref_code.to(dev)
        del base
        torch.cuda.empty_cache()
        audio, sr = cv.generate_voice_clone(
            text=TEXT, language="English", voice_clone_prompt=[item],
            ref_text=ref_text, instruct=INSTRUCT, xvec_only=False)
        sf.write("outputs/icl_via_base.wav", audio[0], sr)
        print(f"[2] BASE-BUILT ICL OK -> outputs/icl_via_base.wav ({len(audio[0])/sr:.2f}s)")
    except Exception as exc:
        print(f"[2] BASE-BUILT ICL FAILED: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
