"""Central configuration for the Qwen3-TTS voice-cloning demo.

Every value can be overridden with an environment variable of the same name,
so the test suite and scripts stay flexible without code edits.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths -----------------------------------------------------------------
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "outputs"


# Trump settings
REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "trump_trade_deficit.mp3"))
REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "trump_trade_deficit.wav"))
REF_TEXT = os.environ.get(
    "REF_TEXT",
    "then I did it again but I did it for a lot of others. You look at the stats, the deficit last month was cut in half."
)

# Trump settings 2
# REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "eight-billion-seven-hundred-and-thirty-seven-million-five-hundred-and-forty-thousand-dollars.mp3"))
# REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "eight-billion-seven-hundred-and-thirty-seven-million-five-hundred-and-forty-thousand-dollars.wav"))
# REF_TEXT = os.environ.get(
#     "REF_TEXT",
#     "Eight billion seven hundred and thirty seven million five hundred and forty thousand dollars"
# )


# Naomi settings
# REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "naomi_voice.mp3"))
# REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "naomi_voice.wav"))
# REF_TEXT = os.environ.get(
#     "REF_TEXT",
#     "Hello. This is Naomi we are doing a voice recording of my voice. I don't know what else to say, so hopefully this is a good amount of time."
# )

# REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "bartholomew-we-don-t-have-anymore-time.mp3"))
# REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "bartholomew-we-don-t-have-anymore-time.wav"))
# REF_TEXT = os.environ.get(
#     "REF_TEXT",
#     "Bartholomew, we don't have anymore time!"
# )

# REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "but-they-share-my-unique-face-colonel-watson-s-name-has-chickens-and-they-don-t-even-have-mustaches.mp3"))
# REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "but-they-share-my-unique-face-colonel-watson-s-name-has-chickens-and-they-don-t-even-have-mustaches.wav"))
# REF_TEXT = os.environ.get(
#     "REF_TEXT",
#     "But they share my unique face! Colonel Warson's name has chickens, and they don't even have mustaches."
# )

# Joe Biden
REF_AUDIO_MP3 = Path(os.environ.get("REF_AUDIO_MP3", ASSETS_DIR / "china-is-going-to-eat-our-lunch-come-on-man-they-can-t-even-figure-out-how-to-deal-with-the-fact-that-they-have-this-gre.mp3"))
REF_AUDIO_WAV = Path(os.environ.get("REF_AUDIO_WAV", ASSETS_DIR / "china-is-going-to-eat-our-lunch-come-on-man-they-can-t-even-figure-out-how-to-deal-with-the-fact-that-they-have-this-gre.wav"))
REF_TEXT = os.environ.get(
    "REF_TEXT",
    "China is going to eat our lunch? Come on, man. They can't even figure out how to deal with the fact that they have this great division between the China Sea and the mountains in the east, I mean, in the west. "
)

# --- Model -----------------------------------------------------------------
# The *-Base checkpoint is the one that supports zero-shot voice cloning.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
DEVICE = os.environ.get("QWEN_TTS_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")  # bfloat16 | float16 | float32
# "sdpa" needs no extra deps. Set to "flash_attention_2" only if flash-attn is
# installed (pip install -U flash-attn --no-build-isolation).
ATTN_IMPL = os.environ.get("QWEN_TTS_ATTN", "sdpa")

# --- Generation defaults ---------------------------------------------------
LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")

# Natural-language style instruction that steers *how* the text is delivered
# (tone/emotion/pacing), independent of the words being spoken. Set to "" to
# disable. Override via the INSTRUCT env var.
INSTRUCT = os.environ.get(
    "INSTRUCT",
    "用特别愤怒的语气说"
)
SAMPLE_RATE = int(os.environ.get("REF_SAMPLE_RATE", "24000"))  # target sr for the prepped WAV

# --- Generation watchdog ---------------------------------------------------
# The model occasionally glitches and runs away (loops/repeats), producing far
# more audio — or taking far longer — than the input warrants. The Speaker
# aborts such an utterance and moves to the next queued item. Both budgets scale
# with the estimated speech duration of the input text (length-aware), so short
# inputs get short budgets and long inputs get proportionally longer ones.
SPEECH_WORDS_PER_SEC = float(os.environ.get("SPEECH_WORDS_PER_SEC", "2.6"))  # ~speaking rate
# Abort if produced audio exceeds FLOOR + FACTOR * expected_audio_seconds.
GEN_MAX_AUDIO_FACTOR = float(os.environ.get("GEN_MAX_AUDIO_FACTOR", "3.0"))
GEN_MAX_AUDIO_FLOOR = float(os.environ.get("GEN_MAX_AUDIO_FLOOR", "3.0"))    # seconds
# Wall-clock backstop (e.g. a hard stall): FLOOR + FACTOR * expected_audio_seconds.
GEN_TIMEOUT_FACTOR = float(os.environ.get("GEN_TIMEOUT_FACTOR", "6.0"))
GEN_TIMEOUT_FLOOR = float(os.environ.get("GEN_TIMEOUT_FLOOR", "5.0"))        # seconds

# --- Realtime speech-to-speech pipeline (src/pipeline.py) ------------------
# faster-whisper model + compute type for the STT stage.
STT_MODEL = os.environ.get("STT_MODEL", "base.en")
STT_COMPUTE = os.environ.get("STT_COMPUTE", "float16")  # float16|int8_float16|int8 ...
# Silence (ms) that marks the end of an utterance before transcription fires.
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "600"))
# Mic device: None -> auto (prefer PulseAudio). Override e.g. AUDIO_INPUT_DEVICE=7
MIC_DEVICE = os.environ.get("AUDIO_INPUT_DEVICE")
