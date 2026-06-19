#!/usr/bin/env python3
"""POC: cloned voice + instruction style via the CustomVoice model.

The base voice-clone checkpoint does NOT reliably follow instructions. This
script tests the "hybrid" path your note describes, reconstructed on the
official qwen-tts API:

  1. Extract a speaker embedding (x-vector) from the reference clip using the
     1.7B *Base* model (model.extract_speaker_embedding via create_voice_clone_prompt).
  2. Load the 1.7B *CustomVoice* model (trained for instruction following).
  3. Generate with that embedding injected into the speaker slot AND an
     `instruct` — in a single generate() call.

It writes three files to outputs/ so you can judge for yourself:
  A_hybrid_instruct.wav  — cloned embedding + instruct   (the goal)
  B_hybrid_noinstruct.wav — cloned embedding, no instruct (identity baseline)
  C_persona_instruct.wav — a built-in persona + instruct  (does instruct work at all?)

Usage:
  uv run python -m scripts.hybrid_instruct ["text to speak"] ["instruction"]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TEXT = "I cannot believe you did that again. We had one job to do today."
DEFAULT_INSTRUCT = "Speak with restrained frustration — slow, tense, and quiet."


def _load(model_id):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, {"bfloat16": "bfloat16", "float16": "float16",
                            "float32": "float32"}.get(config.DTYPE, "bfloat16"))
    return Qwen3TTSModel.from_pretrained(
        model_id, device_map=config.DEVICE, dtype=dtype,
        attn_implementation=config.ATTN_IMPL,
    )


def main(argv: list[str]) -> int:
    import soundfile as sf
    import torch

    from src.clone_voice import prepare_reference

    import os
    text = argv[1] if len(argv) > 1 else DEFAULT_TEXT
    instruct = argv[2] if len(argv) > 2 else DEFAULT_INSTRUCT
    prefix = os.environ.get("OUT_PREFIX", "")
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_wav = str(prepare_reference(config.REF_AUDIO_MP3))
    print(f"reference: {ref_wav}\ntext: {text!r}\ninstruct: {instruct!r}\n")

    # 1) Extract the speaker embedding with the Base model, then free it.
    print(f"[1/3] extracting speaker embedding with {BASE_MODEL} …")
    base = _load(BASE_MODEL)
    items = base.create_voice_clone_prompt(
        ref_audio=ref_wav, ref_text="", x_vector_only_mode=True)
    spk_emb = items[0].ref_spk_embedding.detach().to("cpu")
    print(f"      embedding shape: {tuple(spk_emb.shape)}")
    del base
    torch.cuda.empty_cache()

    # 2) Load the CustomVoice model.
    print(f"[2/3] loading {CUSTOM_MODEL} …")
    cv = _load(CUSTOM_MODEL)
    speakers = cv.get_supported_speakers()
    print(f"      supported personas: {speakers}")

    # The CustomVoice *wrapper* blocks generate_voice_clone (model-type guard),
    # but the core model.generate() accepts voice_clone_prompt + instruct_ids
    # with no such check. Replicate the wrapper internals directly.
    vcp_dict = dict(
        ref_code=[None],
        ref_spk_embedding=[spk_emb.to(cv.model.talker.device).to(cv.model.talker.dtype)],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )
    instr_tok = cv._tokenize_texts([cv._build_instruct_text(instruct)])[0]

    def clone_generate(with_instruct: bool):
        input_ids = cv._tokenize_texts([cv._build_assistant_text(text)])
        gen_kwargs = cv._merge_generate_kwargs()
        codes_list, _ = cv.model.generate(
            input_ids=input_ids, ref_ids=None, voice_clone_prompt=vcp_dict,
            instruct_ids=[instr_tok] if with_instruct else [None],
            languages=["English"], non_streaming_mode=False, **gen_kwargs)
        wavs_all, fs = cv.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_list])
        wav = wavs_all[0]
        return (wav.cpu().numpy() if hasattr(wav, "cpu") else wav), fs

    # 3) Generate the three comparison clips.
    print("[3/3] generating …")

    def write(name, wav, sr):
        path = out_dir / f"{prefix}{name}"
        sf.write(str(path), wav, sr)
        print(f"      wrote {path}  ({len(wav) / sr:.2f}s)")

    for name, fn in [
        ("A_hybrid_instruct.wav", lambda: clone_generate(True)),
        ("B_hybrid_noinstruct.wav", lambda: clone_generate(False)),
        ("C_persona_instruct.wav",
         (lambda: cv.generate_custom_voice(
             text=text, language="English", speaker=speakers[0], instruct=instruct))
         if speakers else None),
    ]:
        if fn is None:
            continue
        try:
            result = fn()
            wav, sr = (result[0][0], result[1]) if name.startswith("C") else result
            write(name, wav, sr)
        except Exception as exc:
            print(f"      {name} FAILED: {type(exc).__name__}: {exc}")

    print("\nDone. Compare A (goal) vs B (identity) vs C (instruct sanity).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
