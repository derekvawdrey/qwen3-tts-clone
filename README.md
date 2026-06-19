# Qwen3-TTS Voice Cloning Demo

Self-contained repo that installs [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
and clones a voice from a reference clip using zero-shot voice cloning. Inference
runs on the CUDA-graph-accelerated
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) runtime
(~5× faster, with streaming), and there's an optional realtime
**speech-to-speech** loop (mic → speech-to-text → cloned voice) with a Tkinter
GUI and a Linux **virtual microphone** so the cloned voice can be used as an
input device in other apps (Zoom, Discord, OBS, the browser …).

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the venv + deps)
- Python ≥ 3.10 (uv will fetch one if needed)
- NVIDIA GPU. The default model is `12Hz-0.6B-Base` (small/fast); the
  `12Hz-1.7B-Base` model needs ~6 GB+ free (RTX 3090 Ti works).
- No system `ffmpeg` needed — `soundfile`'s bundled libsndfile decodes the MP3.
- For the realtime loop: a microphone, and (for the virtual mic) PulseAudio or
  PipeWire with the `pactl` command available.

## Quick start

```bash
./setup.sh                     # uv sync + install qwen-tts/torch/audio + prep the WAV
```

Generate a one-off line in the cloned voice, streamed to the speaker:

```bash
uv run python -m src.clone_voice "The trade deficit is way down, believe me."
# or:  make demo
```

## Realtime speech-to-speech

Installs faster-whisper (STT), Silero VAD (utterance detection), and sounddevice:

```bash
uv sync --extra realtime
```

**GUI** (recommended) — pick mic/output, toggle the virtual mic, Start/Stop,
watch the live transcript:

```bash
uv run python -m src.gui
```

**Console** loop (no GUI):

```bash
uv run python -m src.pipeline      # speak; Ctrl+C to stop
```

How it works: three threads connected by queues —
`MicSource` (16 kHz capture) → `Transcriber` (Silero VAD segments speech, then
faster-whisper transcribes) → `Speaker` (`generate_voice_clone_streaming` →
`StreamPlayer`).

**Half-duplex vs barge-in.** The "Mute mic while speaking" option is an *echo
guard*: it's only needed when the TTS plays through **speakers** the mic can
hear, which would otherwise get transcribed as if you said it. With the virtual
mic (or headphones) the output never reaches your mic, so there's no echo — the
GUI auto-disables the guard when the virtual mic is on, which also enables
**barge-in** (you can talk while the cloned voice is still streaming).

### Virtual microphone (Linux / PipeWire / PulseAudio)

Tick **"Expose as virtual microphone"** in the GUI (or it's auto-managed there).
The app runs `pactl` to create a null sink plus a remapped source, so other apps
see a normal mic named **`QwenTTS_Microphone`**; the TTS is routed into it. The
devices are torn down on exit. See `src/virtual_mic.py`.

> When the virtual mic is on, the cloned voice is routed to the virtual device
> (so other apps can hear it) and is **not** played through your speakers.

## Audio device notes

Output and input devices are selectable in the GUI, or via env vars for the
console tools. The bundled `StreamPlayer` resamples in software and defaults to
the PulseAudio device, because some hardware outputs (e.g. HDMI) reject the
model's 24 kHz rate.

```bash
export AUDIO_OUTPUT_DEVICE=pulse   # or a device index from the GUI list
export AUDIO_INPUT_DEVICE=7        # e.g. a specific USB mic
```

## Steering delivery with an instruction prompt

`config.INSTRUCT` is a natural-language style prompt that steers *how* text is
delivered (tone/emotion/pacing), independent of the words. It's passed straight
to `generate_voice_clone`. Set it in the GUI's "Instruct" field, via the env
var, or empty to disable:

```bash
export INSTRUCT="Speak slowly, in a calm, warm tone."
export INSTRUCT=""                 # disable
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
export MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"   # larger/higher quality
export QWEN_TTS_ATTN="flash_attention_2"           # if flash-attn is installed
export REF_TEXT="...exact transcript of the clip..."
export STT_MODEL="base.en"          # faster-whisper model for the realtime loop
export VAD_SILENCE_MS=600           # silence (ms) that ends an utterance
```
