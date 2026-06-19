#!/usr/bin/env python3
"""Verify the expressive-clone hybrid on the FAST (faster-qwen3-tts) streaming path.

Extract the speaker x-vector with the Base model, then stream-generate on the
CustomVoice model with that embedding + an instruct — all through
faster-qwen3-tts (CUDA graphs + streaming), no official slow path.

Writes outputs/<prefix>fast_A_instruct.wav and <prefix>fast_B_noinstruct.wav.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TEXT = "Nobody has ever seen numbers like these, believe me, they're tremendous."
INSTRUCT = "Speak with restrained frustration — slow, tense, and quiet."


def _load(model_id):
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    dtype = getattr(torch, {"bfloat16": "bfloat16", "float16": "float16",
                            "float32": "float32"}.get(config.DTYPE, "bfloat16"))
    return FasterQwen3TTS.from_pretrained(
        model_id, device=config.DEVICE, dtype=dtype,
        attn_implementation=config.ATTN_IMPL)


def main() -> int:
    import soundfile as sf
    import torch

    from src.clone_voice import prepare_reference

    prefix = os.environ.get("OUT_PREFIX", "")
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_wav = str(prepare_reference(config.REF_AUDIO_MP3))
    print(f"reference: {ref_wav}\n")

    print(f"[1/2] extracting x-vector with {BASE_MODEL} (fast) …")
    base = _load(BASE_MODEL)
    items = base.model.create_voice_clone_prompt(
        ref_audio=ref_wav, ref_text="", x_vector_only_mode=True)
    emb = items[0].ref_spk_embedding.detach().to("cpu")
    print(f"      embedding shape: {tuple(emb.shape)}")
    del base
    torch.cuda.empty_cache()

    print(f"[2/2] streaming on {CUSTOM_MODEL} (fast) …")
    cv = _load(CUSTOM_MODEL)
    emb = emb.to(config.DEVICE).to(
        getattr(torch, {"bfloat16": "bfloat16", "float16": "float16",
                        "float32": "float32"}.get(config.DTYPE, "bfloat16")))
    vcp = {"ref_spk_embedding": [emb]}

    def run(name, instruct):
        chunks, t0, first = [], time.time(), None
        for chunk, sr, _timing in cv.generate_voice_clone_streaming(
                text=TEXT, language="English", voice_clone_prompt=vcp,
                instruct=instruct, xvec_only=True, chunk_size=8):
            if first is None:
                first = time.time() - t0
            chunks.append(chunk)
        wav = np.concatenate(chunks)
        total = time.time() - t0
        dur = len(wav) / sr
        path = out_dir / f"{prefix}{name}"
        sf.write(str(path), wav, sr)
        print(f"      wrote {path}  ({dur:.2f}s audio, gen {total:.2f}s, "
              f"first-chunk {first*1000:.0f}ms, RTF {dur/total:.2f})")

    run("fast_A_instruct.wav", INSTRUCT)
    run("fast_B_noinstruct.wav", None)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
