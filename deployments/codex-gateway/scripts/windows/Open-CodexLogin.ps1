[CmdletBinding()]
param(
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$url = "http://127.0.0.1:8000/oauth/codex/login"
$deployRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$tokenPath = Join-Path (Join-Path $deployRoot "codex") "auth.json"

Write-Host "Codex login URL:"
Write-Host "  $url"

if (-not $NoBrowser) {
  Start-Process $url
}

Write-Host ""
Write-Host "After the browser flow succeeds, credentials should be saved to:"
Write-Host "  $tokenPath"
Write-Host ""
Write-Host "Check auth status with:"
Write-Host "  docker exec ccproxy-codex-gateway /app/.venv/bin/ccproxy auth status codex"
