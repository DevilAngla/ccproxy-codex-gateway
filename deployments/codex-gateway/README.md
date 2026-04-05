# Codex Gateway Deployment

This deployment bundle runs the patched Codex gateway directly from Docker.
It is designed for the "multiple Codex/ChatGPT accounts behind one local API"
workflow and keeps runtime credentials out of git.

## Included

- `docker-compose.yml`: starts the gateway on `127.0.0.1:8000`
- `config/config.toml`: enables `oauth_codex`, `codex`, `request_tracer`, and `credential_balancer`
- `scripts/windows/*.ps1`: helper scripts for login and credential export
- `.env.example`: points Docker Compose at the published GHCR image

## Secrets And Runtime Files

These paths are intentionally ignored by git and never committed:

- `codex/`
- `logs/`
- `config/credentials/*.json`

## Quick Start

```powershell
cd deployments\codex-gateway
Copy-Item .env.example .env
docker compose pull
docker compose up -d
docker compose ps
```

Service endpoints:

- Gateway: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- Models: `http://127.0.0.1:8000/codex/v1/models`
- Docs: `http://127.0.0.1:8000/docs`

## Login A Codex Account

The simplest Windows flow is the container CLI login, because it writes the
credential file directly to `config/credentials`.

```powershell
cd deployments\codex-gateway
.\scripts\windows\Start-CodexCliLogin.ps1 -Name account-01
Get-Content -Wait .\logs\login-account-01.log
```

After the browser callback succeeds, the credential lands at:

```text
deployments/codex-gateway/config/credentials/account-01.json
```

You can also open the browser login directly:

```powershell
.\scripts\windows\Open-CodexLogin.ps1
```

If you use the browser flow above, export it into the balancer directory:

```powershell
.\scripts\windows\Save-CodexCredential.ps1 -Name account-01
```

## Add More Accounts

1. Run `.\scripts\windows\Start-CodexCliLogin.ps1 -Name account-02`
2. Run it again for `account-03`, `account-04`, and so on
3. Add matching credential entries to `config/config.toml`
4. Restart the stack with `docker compose up -d`

The deployment uses the `credential_balancer` plugin in `failover` mode, so
healthy credentials are used automatically when a prior one fails.

## Use With New API

Use this base URL inside `new-api`:

```text
http://host.docker.internal:8000/codex
```

Recommended model names:

- `gpt-5.4`
- `gpt-5.3-codex`
- `gpt-5.2-codex`

`/v1/responses` is patched here to accept both structured `input` and plain
string `input`, so clients can send either of these:

```json
{"model":"gpt-5.4","input":"Reply with exactly: OK","store":false}
```

```json
{
  "model": "gpt-5.4",
  "store": false,
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Reply with exactly: OK"
        }
      ]
    }
  ]
}
```

## Common Commands

```powershell
docker compose up -d
docker compose logs -f
docker compose ps
docker compose restart
docker compose down
```
