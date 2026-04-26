Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

Set-Location $projectRoot

Write-Host "Starting unified Coffee QA Web UI..."
Write-Host "URL will be shown by Gradio after startup."

uv run --active web_app.py