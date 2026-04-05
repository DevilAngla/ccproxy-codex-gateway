[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [ValidatePattern("^[A-Za-z0-9._-]+$")]
  [string]$Name,

  [string]$Source = (Join-Path (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\\..")) "codex") "auth.json"),

  [string]$DestinationDirectory = (Join-Path (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\\..")) "config") "credentials"),

  [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Source)) {
  throw "Source credential file not found: $Source"
}

$null = New-Item -ItemType Directory -Path $DestinationDirectory -Force

$json = Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json
if (-not $json.tokens.access_token -or -not $json.tokens.refresh_token) {
  throw "Source file does not look like a valid Codex/OpenAI OAuth credential."
}

$destination = Join-Path $DestinationDirectory "$Name.json"
if ((Test-Path -LiteralPath $destination) -and -not $Force) {
  throw "Target file already exists: $destination. Use -Force to overwrite."
}

Copy-Item -LiteralPath $Source -Destination $destination -Force

Write-Host "Saved credential to:"
Write-Host "  $destination"
