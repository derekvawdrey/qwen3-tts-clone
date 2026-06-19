.PHONY: setup prep test clone-test transcribe demo gui clean

setup:            ## uv sync + install + prep audio
	./setup.sh

prep:             ## Rebuild the mono WAV from the source MP3
	uv run python scripts/prep_audio.py

test:             ## Fast environment tests (no GPU)
	uv run pytest -v

clone-test:       ## Full voice-cloning test (downloads model, uses GPU)
	uv run pytest -v --run-clone

transcribe:       ## Auto-transcribe the reference clip with Whisper (optional)
	uv run --extra transcribe python scripts/transcribe.py

# Install the official *prebuilt* FlashAttention 2 wheel (matches torch 2.8 /
# CUDA 12 / cp311). No compilation — installs in seconds. The wheel URL lives in
# the [flash] extra in pyproject.toml.
flash-attn:       ## Install prebuilt FlashAttention 2 (optional, speeds up inference)
	uv sync --extra flash
	@echo "Done. Enable it with:  export QWEN_TTS_ATTN=flash_attention_2"

demo:             ## Generate a sample line in the cloned voice
	uv run python -m src.clone_voice "The trade deficit is way down, believe me." outputs/demo.wav

gui:              ## Launch the speech-to-speech GUI (syncs realtime deps first)
	./run-gui.sh

clean:            ## Remove generated outputs and the prepped WAV
	rm -rf outputs assets/trump_trade_deficit.wav .pytest_cache
