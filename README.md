# Qwen3-TTS Voice Cloning Demo

Self-contained repo that installs [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
and clones a voice from a reference clip using zero-shot voice cloning.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the venv + deps)
- Python ≥ 3.10 (uv will fetch one if needed; `accelerate` requires 3.10+)
- NVIDIA GPU with ~6 GB+ free for the `12Hz-1.7B-Base` model (RTX 3090 Ti works)
- No system `ffmpeg` needed — `soundfile`'s bundled libsndfile decodes the MP3.
- Installed virtual cable [VB-audio](https://vb-audio.com/Cable/), or similar.


## Quick start

```bash
./setup.sh                     # uv sync + install qwen-tts/torch/audio + prep the WAV
```

Generate a one-off line in the cloned voice that will be streamed to speaker:

```bash
uv run python -m src.clone_voice "The trade deficit is way down, believe me."
# or:  make demo
```

## FlashAttention 2 (optional, faster inference)

torch is pinned to 2.8.0 (CUDA 12.8) specifically because it's the newest torch
with an official **prebuilt** flash-attn wheel — so this installs in seconds
instead of compiling for ~30 minutes (torch 2.12 / CUDA 13 has no wheel). The
wheel is GPU-arch-agnostic; works on RTX 30xx/40xx/H100 alike.

```bash
make flash-attn                        # installs the prebuilt wheel (seconds)
export QWEN_TTS_ATTN=flash_attention_2 # tell the model to use it
```

Without it the model runs fine on the default `sdpa` attention.

## Configuration

Override anything via env vars (see `config.py`), e.g.:

```bash
export MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-Base"   # smaller/faster
export QWEN_TTS_ATTN="flash_attention_2"           # if flash-attn is installed
export REF_TEXT="...exact transcript of the clip..."
```
