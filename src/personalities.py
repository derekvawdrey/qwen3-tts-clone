"""Load the "personality" voice catalog produced by scripts/fetch_personalities.py.

The manifest (assets/personalities.json) lists downloadable-voice presets, each
with a display name, a prepped reference WAV, and (optionally) reference text.
The GUI merges these into its voice picker so they're selectable out of the box.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

MANIFEST = config.ASSETS_DIR / "personalities.json"


def load_personalities() -> list[dict]:
    """Return manifest entries whose audio file actually exists on disk.

    Each entry: {slug, name, category, audio (abs path), ref_text, language, instruct}.
    Missing/invalid manifest -> empty list (the GUI just falls back to assets/).
    """
    if not MANIFEST.exists():
        return []
    try:
        data = json.loads(MANIFEST.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    out: list[dict] = []
    for e in data.get("voices", []):
        audio = e.get("audio")
        if not audio:
            continue
        path = Path(audio)
        if not path.is_absolute():
            path = config.ROOT / path
        if not path.exists():
            continue  # not downloaded yet
        out.append({
            "slug": e.get("slug", path.stem),
            "name": e.get("name", path.stem),
            "category": e.get("category", "Voices"),
            "audio": str(path),
            "ref_text": e.get("ref_text", ""),
            "language": e.get("language", config.LANGUAGE),
            "instruct": e.get("instruct", ""),
        })
    return out
