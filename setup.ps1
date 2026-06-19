<#
.SYNOPSIS
  One-shot Windows setup with uv: create the venv, install Qwen3-TTS + deps,
  prep the reference clip. Windows equivalent of setup.sh.

.NOTES
  Run from a PowerShell prompt:
      powershell -ExecutionPolicy Bypass -File .\setup.ps1

  Requirements:
    - uv          (https://docs.astral.sh/uv/getting-started/)
    - NVIDIA GPU + recent driver (torch is the CUDA 12.8 build)

  Not available on Windows:
    - The virtual microphone (src/virtual_mic.py needs PulseAudio/PipeWire's
      `pactl`). Install VB-CABLE (https://vb-audio.com/Cable/) and select it as
      the output device to feed the cloned voice into other apps instead.
    - The `flash` extra (Linux-only wheel). The default `sdpa` attention works;
      don't pass --extra flash on Windows.
#>
$ErrorActionPreference = "Stop"

# cd to the script's directory (repo root)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it: https://docs.astral.sh/uv/getting-started/`nQuick install:  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}

Write-Host ">> uv sync (creates .venv, installs deps + dev group)"
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">> Preparing reference audio (MP3 -> mono WAV)"
uv run python scripts/prep_audio.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  uv run pytest                 # fast environment tests"
Write-Host "  uv run pytest --run-clone     # full voice-cloning test (downloads model, uses GPU)"
Write-Host "  uv run python -m src.clone_voice `"Hello from a cloned voice.`" outputs/hello.wav"
Write-Host ""
Write-Host "Realtime loop / GUI:  uv sync --extra realtime ;  uv run python -m src.gui"
Write-Host "Virtual mic on Windows: install VB-CABLE and pick it as the output device."
