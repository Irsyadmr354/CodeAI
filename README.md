# CodeAI

Harness CLI ringan multi-provider yang menyatukan opencode + oh-my-pi dalam satu REPL.

## Fitur

- Governance `AGENTS.md` 100% — pre-flight block + system prompt precedence Base<System<Global<AGENTS<HOOKS<User.
- Lifecycle `SYSTEM_HOOKS.md` — dispatcher hooks SessionInit, Pre/Post Tool, PreCompact, SessionEnd.
- Kompresi token hemat — compact ≤10 baris + AST skeleton + sliding `fact_cards`.
- Multi-provider + failover — `anthropic/copilot/gemini/openai/ollama/opencode/antigravity`, `failover_order`, custom gateway OpenAI-compatible `/v1/models`.
- Effort inside-setting — `/effort low|medium|high` persist ke `codeai.json`, parsing `base-effort` inline.
- CLI ringan — banner 3-baris (status AGENTS/HOOKS + commands), `/model` fuzzy direct atau selector, Ctrl-C `/steer` redirect.

## Quickstart

```bash
python main.py
python main.py --config codeai.json --verbose
```

```text
/login <provider>      # auth, mis. /login anthropic | /login antigravity
/model                 # fuzzy switch, mis. /model gemini / copilot/gpt-4.1
/effort high           # set effort persist (low|medium|high)
/steer <instruksi>     # redirect subagent aktif (alias /st)
```

Alur tipikal: `python main.py` → `/login` → `/model` → kerja → `/effort` + `/steer` bila perlu.

## Konfigurasi `codeai.json`

```json
{
  "provider": {
    "default": "anthropic",
    "effort": "medium",
    "failover_order": ["copilot", "gemini", "openai", "ollama"]
  },
  "compaction": { "token_budget": 4000 },
  "custom_providers": {},
  "allowance": { "mode": "ask" }
}
```

Custom gateway: tambah entry OpenAI-compatible di `custom_providers`, auto-discovery `/v1/models`. Tanpa secrets di repo — kredensial via `/login`, tersimpan di vault lokal.

## Status

CONVERGED — 31 passed. Detail: `docs/AUDIT_2026-09-04.md` + `docs/EXECUTION_2026-09-05.md`.
