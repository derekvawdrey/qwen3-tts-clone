#!/usr/bin/env python3
"""Download voice samples from aiartes.com, trim them, and register them as
selectable "personality" voices for the clone GUI.

For each voice it:
  1. downloads the *original* sample  (https://aiartes.com/records/<slug>_original.mp3)
  2. trims leading/trailing silence, then keeps the first N seconds (default 20)
  3. writes a 24 kHz mono WAV to  assets/personalities/<slug>.wav
  4. (optional) transcribes it with faster-whisper to fill the reference text
  5. records it in  assets/personalities.json  (the manifest the GUI reads)

The script is idempotent: an already-prepped WAV is not re-downloaded, and an
entry that already has reference text is not re-transcribed unless --force.

Usage:
    uv run python scripts/fetch_personalities.py            # all voices, transcribe
    uv run python scripts/fetch_personalities.py --list     # show available voices
    uv run python scripts/fetch_personalities.py --only arnold,gollum
    uv run python scripts/fetch_personalities.py --no-transcribe --seconds 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BASE_URL = "https://aiartes.com"
PAGE_URL = f"{BASE_URL}/voiceai"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

PERS_DIR = config.ASSETS_DIR / "personalities"
CACHE_DIR = config.OUTPUT_DIR / "personalities_cache"
MANIFEST = config.ASSETS_DIR / "personalities.json"


# --- scraping --------------------------------------------------------------
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (trusted host)
        return r.read()


def discover_voices() -> list[dict]:
    """Parse the catalog page into [{slug, name, category}], in page order."""
    html = _get(PAGE_URL).decode("utf-8", "replace")
    # Tokens we care about, in document order: category headers, voice titles,
    # and the "_original.mp3" sources. Pair each original with the last title
    # and last category seen above it.
    token = re.compile(
        r'<h4[^>]*divider header[^>]*>(?P<cat>.*?)</h4>'
        r'|<div class="voice-title">(?P<title>.*?)</div>'
        r'|/records/(?P<slug>[a-z0-9-]+)_original\.mp3',
        re.DOTALL,
    )
    voices: list[dict] = []
    seen: set[str] = set()
    cat, title = "Voices", None
    for m in token.finditer(html):
        if m.group("cat") is not None:
            cat = re.sub(r"<[^>]+>", "", m.group("cat")).strip()
            cat = re.sub(r"\s+", " ", cat)
        elif m.group("title") is not None:
            title = re.sub(r"\s+", " ", m.group("title")).strip()
        elif m.group("slug") is not None:
            slug = m.group("slug")
            if slug in seen:
                continue
            seen.add(slug)
            voices.append({"slug": slug, "name": title or slug, "category": cat})
    return voices


# --- audio prep ------------------------------------------------------------
def _decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to mono float32 (soundfile, then librosa fallback)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = np.asarray(data)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception:  # noqa: BLE001
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=True)
        return data.astype(np.float32), sr


def prep_clip(src_mp3: Path, dst: Path, seconds: float, target_sr: int) -> float:
    """Trim silence, keep the first `seconds`, resample, write MP3. Returns dur.

    Output is a small mono MP3 (~100 KB for 20 s) so the catalog can be bundled
    in the repo cheaply; libsndfile (via soundfile) encodes it without ffmpeg.
    """
    import librosa
    import soundfile as sf

    data, sr = _decode(src_mp3)
    trimmed, _ = librosa.effects.trim(data, top_db=30)  # drop leading/trailing silence
    if trimmed.size:
        data = trimmed
    data = data[: int(seconds * sr)]
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sr, format="MP3")
    return len(data) / sr


# --- transcription ---------------------------------------------------------
class Transcriber:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel

        cuda = self._cuda()
        self.model = WhisperModel(
            model_name,
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )

    @staticmethod
    def _cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            return False

    def __call__(self, wav: Path) -> str:
        segments, _ = self.model.transcribe(str(wav))
        return " ".join(s.text.strip() for s in segments).strip()


# --- manifest --------------------------------------------------------------
def load_manifest() -> dict[str, dict]:
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
        return {e["slug"]: e for e in data.get("voices", [])}
    return {}


def save_manifest(entries: dict[str, dict]) -> None:
    ordered = sorted(entries.values(), key=lambda e: (e.get("category", ""), e["name"]))
    MANIFEST.write_text(json.dumps({"voices": ordered}, indent=2, ensure_ascii=False) + "\n")


# --- main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated slugs to import (default: all)")
    ap.add_argument("--limit", type=int, help="import at most N voices")
    ap.add_argument("--seconds", type=float, default=20.0, help="clip length to keep (default 20)")
    ap.add_argument("--stt-model", default="base.en", help="faster-whisper model (default base.en)")
    ap.add_argument("--no-transcribe", action="store_true", help="skip reference-text transcription")
    ap.add_argument("--force", action="store_true", help="re-download, re-prep, and re-transcribe")
    ap.add_argument("--list", action="store_true", help="list available voices and exit")
    args = ap.parse_args()

    print(f">> Fetching catalog from {PAGE_URL}")
    voices = discover_voices()
    print(f"   found {len(voices)} voices")

    if args.list:
        for v in voices:
            print(f"  {v['slug']:<22} {v['name']}  [{v['category']}]")
        return 0

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        voices = [v for v in voices if v["slug"] in wanted]
        missing = wanted - {v["slug"] for v in voices}
        if missing:
            print(f"   WARNING: unknown slugs ignored: {', '.join(sorted(missing))}", file=sys.stderr)
    if args.limit:
        voices = voices[: args.limit]

    PERS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    transcriber: Transcriber | None = None
    ok = 0

    for i, v in enumerate(voices, 1):
        slug, name = v["slug"], v["name"]
        clip = PERS_DIR / f"{slug}.mp3"
        prefix = f"[{i}/{len(voices)}] {name} ({slug})"
        try:
            # 1-3) download + trim + resample
            if clip.exists() and not args.force:
                print(f"{prefix}: clip exists, skipping prep")
            else:
                src = CACHE_DIR / f"{slug}_original.mp3"
                if not src.exists() or args.force:
                    print(f"{prefix}: downloading")
                    src.write_bytes(_get(f"{BASE_URL}/records/{slug}_original.mp3"))
                dur = prep_clip(src, clip, args.seconds, config.SAMPLE_RATE)
                print(f"{prefix}: prepped {dur:.1f}s -> {clip.name}")

            entry = manifest.get(slug, {})
            entry.update({"slug": slug, "name": name, "category": v["category"],
                          "audio": str(clip.relative_to(config.ROOT)),
                          "language": entry.get("language", config.LANGUAGE),
                          "instruct": entry.get("instruct", "")})

            # 4) transcribe reference text
            if not args.no_transcribe and (args.force or not entry.get("ref_text")):
                if transcriber is None:
                    print(f"   loading faster-whisper '{args.stt_model}' …")
                    transcriber = Transcriber(args.stt_model)
                entry["ref_text"] = transcriber(clip)
                print(f"{prefix}: ref_text = {entry['ref_text'][:70]!r}…")
            entry.setdefault("ref_text", "")

            manifest[slug] = entry
            save_manifest(manifest)  # write incrementally so a crash keeps progress
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"{prefix}: FAILED — {e}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(voices)} voices in {MANIFEST}")
    print("Launch the GUI to pick them:  make gui   (or  uv run python -m src.gui)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
