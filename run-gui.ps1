<#
.SYNOPSIS
  Launch the Tkinter speech-to-speech GUI on Windows.

.DESCRIPTION
  Ensures the `realtime` extra (faster-whisper, silero-vad, sounddevice) is
  installed — the GUI imports src.pipeline which needs it — then starts the GUI.

.NOTES
  Run from a PowerShell prompt:
      powershell -ExecutionPolicy Bypass -File .\run-gui.ps1

  Run setup.ps1 first if you haven't (creates the venv + preps the clip).
#>
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Run setup.ps1 first (https://docs.astral.sh/uv/)."
    exit 1
}

Write-Host ">> Ensuring GUI deps (PySide6 + faster-whisper, silero-vad, sounddevice)"
# --inexact: add the gui extra without uninstalling anything outside it
# (on Windows flash-attn won't be present anyway; harmless here).
uv sync --inexact --extra gui
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">> Launching GUI"
uv run python -m src.gui
exit $LASTEXITCODE
