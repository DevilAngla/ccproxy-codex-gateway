[CmdletBinding()]
param(
  [ValidatePattern("^[A-Za-z0-9._-]+$")]
  [string]$Name = "account-01",

  [string]$Container = "ccproxy-codex-gateway"
)

$ErrorActionPreference = "Stop"

$deployRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$hostLogDir = Join-Path $deployRoot "logs"
$hostCredentialDir = Join-Path (Join-Path $deployRoot "config") "credentials"
$hostLogPath = Join-Path $hostLogDir "login-$Name.log"
$containerCredentialPath = "/root/.config/ccproxy/credentials/$Name.json"
$containerLogPath = "/tmp/ccproxy/login-$Name.log"

$null = New-Item -ItemType Directory -Path $hostLogDir -Force
$null = New-Item -ItemType Directory -Path $hostCredentialDir -Force

if (Test-Path -LiteralPath $hostLogPath) {
  Remove-Item -LiteralPath $hostLogPath -Force
}

$command = @"
mkdir -p /root/.config/ccproxy/credentials /tmp/ccproxy &&
/app/.venv/bin/ccproxy auth login codex --no-browser --file $containerCredentialPath > $containerLogPath 2>&1
"@

docker exec -d $Container sh -lc $command | Out-Null

Start-Sleep -Seconds 2

Write-Host "Started Codex CLI login for '$Name'."
Write-Host "Credential target:"
Write-Host "  $(Join-Path $hostCredentialDir "$Name.json")"
Write-Host "Login log:"
Write-Host "  $hostLogPath"
Write-Host ""

if (Test-Path -LiteralPath $hostLogPath) {
  Get-Content -LiteralPath $hostLogPath
} else {
  Write-Host "The login log has not been created yet. Run this to watch it:"
  Write-Host "  Get-Content -Wait `"$hostLogPath`""
}
