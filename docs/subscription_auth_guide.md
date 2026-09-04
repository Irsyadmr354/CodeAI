# How OpenCode & Oh-My-Pi Connect Providers via Subscriptions (No Pay-Per-Token API Keys)

## 1. Mekanisme Utama

### A. GitHub Copilot Bridge (Flat Subscription)
Baik OpenCode maupun Oh-My-Pi memanfaatkan GitHub Copilot (Individual/Business/Enterprise):
1. **OAuth Device Flow**: User membuka URL `github.com/login/device` dan memasukkan user code 8 digit.
2. **Token Exchange**: Token GitHub ditukar ke endpoint internal Copilot:
   `GET https://api.github.com/copilot_internal/v2/token`
3. **Model Proxy**: Request diteruskan ke `https://api.githubcopilot.com/chat/completions` dengan header:
   - `Authorization: Bearer <copilot_session_token>`
   - `Copilot-Integration-Id: vscode-chat`
   - `Editor-Version: vscode/1.90.0`
4. **Hasil**: Akses ke model unggulan (GPT-4o, Claude 3.5 Sonnet, o1) tanpa biaya token per-request, cukup langganan Copilot flat bulanan ($10-$20/bln).

### B. Credential Auto-Discovery
Oh-My-Pi dan OpenCode memindai file kredensial lokal yang sudah login di komputer user:
- `~/.copilot/config.json` atau token VS Code Copilot extension
- `~/.claude/` (token sesi dari Claude Code CLI)
- `~/.config/gcloud/` (Google Cloud Application Default Credentials)

### C. OpenAI Device Auth (ChatGPT Plus/Pro)
- Menggunakan endpoint OAuth Device Code OpenAI (`/api/accounts/deviceauth/usercode`).
- User verifikasi di browser, token sesi disimpan di credential vault lokal (`auth-broker` / SQLite).
- Request diarahkan ke backend chatgpt subscription via proxy `auth-gateway`.

---

## 2. Cara Mengadopsi ke CodeAI Harness

Untuk menambahkan kapabilitas ini ke CodeAI:
1. **Copilot Adapter (`harness/models/providers/copilot.py`)**:
   - Membaca token dari `~/.copilot/config.json` yang sudah ada di mesin Anda.
   - Mengambil session token dan mengirim request ke endpoint GitHub Copilot.
2. **OAuth Device Code Login Command**:
   - Menambahkan command `/login copilot` di REPL CLI.
3. **Local Session Vault**:
   - Menyimpan token yang sudah terotentikasi di `~/.codeai/auth.json` agar tidak perlu login berulang kali.
