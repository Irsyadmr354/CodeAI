import builtins
import json
import re
import socket
import subprocess
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

from harness.models.auth_vault import AuthVault
from harness.models.base import BaseProvider, ChatMessage, ProviderError, TimeoutError

DEFAULT_ANTIGRAVITY_MODEL = "gemini-3.8-flash"
DEFAULT_ANTIGRAVITY_EFFORT = "medium"

# Base model names (shown in /model picker)
ANTIGRAVITY_BASE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b",
]

# Models that accept effort parameter
EFFORT_MODELS = {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-pro"}
EFFORT_OPTIONS = ["medium", "high", "low"]

# Shared token-budget mapping for non-antigravity providers
# (anthropic thinking budgets; reused as generic effort weight elsewhere).
EFFORT_BUDGET = {"low": 1024, "medium": 4096, "high": 10000}

# ANSI escape codes
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

_NOISE_PATTERNS = [
    re.compile(r'^\s*\[!?\].*?\[!?\]\s*$'),
    re.compile(r'^\s*={2,}\s*Subagent.*?\s*={2,}\s*$', re.IGNORECASE),
    re.compile(r'^\s*-{2,}\s*Subagent.*?\s*-{2,}\s*$', re.IGNORECASE),
    re.compile(r'^\s*#+\s*Subagent.*$', re.IGNORECASE),
    re.compile(r'^\s*\[Subagent(?:\s*:\s*|\s+).*?\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*Subagent\s+.*?(?:started|finished|running|completed).*?$', re.IGNORECASE),
    re.compile(r'^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏].*$'),
]


def _clean(text: str) -> str:
    text = _ANSI_RE.sub('', text)
    lines = [l for l in text.splitlines() if not any(p.match(l) for p in _NOISE_PATTERNS)]
    return "\n".join(lines).strip()


# Auth failure substrings (CLI stderr/stdout, lowercased) → explicit /login hint.
# Deliberately narrow: must NOT match generic failures (e.g. unknown model)
# so non-auth errors still fail over instead of fail-fast.
_AUTH_ERROR_HINTS = (
    "unauthenticated",
    "unauthorized",
    "invalid_grant",
    "invalid credentials",
    "invalid token",
    "expired token",
    "session expired",
    "token expired",
    "not logged in",
    "no auth",
    "login required",
    "re-authenticate",
    "reauthenticate",
    "authenticate",
    "credential",
    "auth failed",
    "auth error",
    "permission denied",
    "sign in",
    "401",
    "403",
)


def _is_auth_output(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _AUTH_ERROR_HINTS)


# Direct Cloud Code endpoint (no API key in URL; bearer via vault).
_CLOUDCODE_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal:generateContent"
_CLOUDCODE_USER_AGENT = "antigravity/1.15.8"
_CLOUDCODE_API_CLIENT = "google-cloud-sdk vscode_cloudshelleditor/0.1"
_CLOUDCODE_CLIENT_METADATA = {"ideType": "ANTIGRAVITY", "platform": "MACOS", "pluginType": "GEMINI"}
_EFFORT_TO_THINKING_LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def _resolve_vault_token(vault: Any) -> Optional[str]:
    """Resolve bearer via vault.get_antigravity_token() or get_token()."""
    if vault is None:
        return None
    getter = getattr(vault, "get_antigravity_token", None)
    if callable(getter):
        try:
            tok = getter()
            if isinstance(tok, str) and tok.strip():
                return tok.strip()
        except Exception:
            pass
    try:
        get_token = getattr(vault, "get_token", None)
        if callable(get_token):
            tok = get_token("antigravity")
            if isinstance(tok, str) and tok.strip():
                return tok.strip()
    except Exception:
        pass
    try:
        discover = getattr(vault, "discover_antigravity_token", None)
        if callable(discover):
            tok = discover()
            if isinstance(tok, str) and tok.strip():
                return tok.strip()
    except Exception:
        pass
    return None


def _resolve_vault_project(vault: Any) -> Optional[str]:
    """Resolve Cloud project from stored antigravity credential."""
    if vault is None:
        return None
    try:
        get_cred = getattr(vault, "get_credential", None)
        if callable(get_cred):
            cred = get_cred("antigravity")
            if isinstance(cred, dict):
                for k in ("projectId", "project_id", "project", "projectID"):
                    v = cred.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    except Exception:
        pass
    return None


def parse_model_effort(model_str: str) -> Tuple[str, Optional[str]]:
    """
    Parse a model string that may contain an effort suffix.
    Supports both `-effort` and `:effort` separators, case-insensitive,
    with surrounding whitespace tolerated.
    Examples:
      "gemini-3.8-flash"           → ("gemini-3.8-flash", None)
      "gemini-3.8-flash-medium"    → ("gemini-3.8-flash", "medium")
      "gemini-3.8-flash:high"      → ("gemini-3.8-flash", "high")
      "gpt-5:low"                  → ("gpt-5", "low")
      "gpt-5-ultra"                → ("gpt-5-ultra", None)  # invalid effort, no crash
    Note: generic split — callers check EFFORT_MODELS for agy applicability.
    The `xhigh` alias is clamped to `high`.
    """
    if not isinstance(model_str, str):
        return model_str, None
    s = model_str.strip()
    if not s:
        return model_str, None
    # Whitespace-tolerant suffix: "base-low", "base - low", "base : HIGH", etc.
    # Model names never contain spaces, so inner-space tolerance is safe.
    m = re.search(r'\s*[-:]\s*(low|medium|high|xhigh)\s*$', s, re.IGNORECASE)
    if m:
        effort = m.group(1).lower()
        if effort == "xhigh":
            effort = "high"
        base = s[: m.start()].strip()
        if base:
            return base, effort
        return s, None
    return s, None


class AntigravityProvider(BaseProvider):
    """
    Provider for Google Antigravity via the `agy` CLI.

    Auth: OAuth session token (managed by `agy` itself).
    Models: gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.1-pro,
            claude-sonnet-4-6, claude-opus-4-6-thinking, gpt-oss-120b
    Effort: low | medium (default) | high  — applies to Gemini models only.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTIGRAVITY_MODEL,
        effort: str = DEFAULT_ANTIGRAVITY_EFFORT,
    ) -> None:
        base, parsed_effort = parse_model_effort(model)
        base_norm = (base or "").strip() if isinstance(base, str) else base
        canonical = next(
            (m for m in ANTIGRAVITY_BASE_MODELS
             if isinstance(base_norm, str) and m.lower() == base_norm.lower()),
            None,
        )
        self.model = canonical if canonical else DEFAULT_ANTIGRAVITY_MODEL
        eff_arg = (effort.strip().lower() if isinstance(effort, str) else "")
        self.effort = parsed_effort or (eff_arg if eff_arg in EFFORT_OPTIONS else DEFAULT_ANTIGRAVITY_EFFORT)
        try:
            self.vault = AuthVault()
        except Exception:
            self.vault = None

    def supports_tools(self) -> bool:
        return False

    def _build_agy_cmd(self, prompt: str) -> List[str]:
        """Build the agy command, splitting model+effort as required by the CLI."""
        cmd = ["agy", "--print", prompt, "--disable-slash-commands"]

        if self.model in EFFORT_MODELS:
            # agy requires --model base_name --effort level
            cmd += ["--model", self.model, "--effort", self.effort]
        else:
            # Non-Gemini models (claude, gpt-oss) use --model only
            cmd += ["--model", self.model]

        return cmd

    def _cloud_model_id(self) -> str:
        """Map base+effort to Cloud model id: gemini gets '-effort' suffix."""
        if self.model in EFFORT_MODELS:
            eff = (self.effort or "").strip().lower() if isinstance(self.effort, str) else ""
            if eff not in EFFORT_OPTIONS:
                eff = DEFAULT_ANTIGRAVITY_EFFORT
            return f"{self.model}-{eff}"
        return self.model

    def _build_cloudcode_contents(
        self, messages: List[ChatMessage]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Map roles: system→systemInstruction, user→user, assistant→model.

        Consecutive same-role turns are merged with newline.
        """
        contents: List[Dict[str, Any]] = []
        system_texts: List[str] = []
        for m in messages or []:
            role = (getattr(m, "role", "") or "").strip().lower()
            text = getattr(m, "content", "") or ""
            if not isinstance(text, str):
                text = str(text)
            if role == "system":
                if text:
                    system_texts.append(text)
                continue
            target = "model" if role in ("assistant", "model") else "user"
            if contents and contents[-1].get("role") == target:
                prev = contents[-1]["parts"][0].get("text", "")
                if prev and text:
                    contents[-1]["parts"][0]["text"] = prev + "\n" + text
                elif text:
                    contents[-1]["parts"][0]["text"] = text
            else:
                contents.append({"role": target, "parts": [{"text": text}]})
        system_instruction = None
        if system_texts:
            system_instruction = {"parts": [{"text": "\n".join(system_texts)}]}
        return contents, system_instruction

    def _cloudcode_headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": _CLOUDCODE_USER_AGENT,
            "X-Goog-Api-Client": _CLOUDCODE_API_CLIENT,
            "Client-Metadata": json.dumps(_CLOUDCODE_CLIENT_METADATA, separators=(",", ":")),
        }

    def _parse_cloudcode_payload(self, payload: Any) -> Dict[str, Any]:
        data = payload
        if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
            data = payload["response"]
        cands = (data.get("candidates") or []) if isinstance(data, dict) else []
        if not cands:
            raise ProviderError("Antigravity: empty response (no candidates).")
        content = cands[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p
        )
        return {"role": "assistant", "content": _clean(text), "tool_calls": None}

    def _post_cloudcode(self, token: str, envelope: Dict[str, Any]) -> Any:
        req = urllib.request.Request(
            _CLOUDCODE_ENDPOINT,
            data=json.dumps(envelope).encode("utf-8"),
            headers=self._cloudcode_headers(token),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _chat_direct(self, token: str, project: str, messages: List[ChatMessage]) -> Dict[str, Any]:
        vault = getattr(self, "vault", None)
        cloud_model = self._cloud_model_id()
        contents, system_instruction = self._build_cloudcode_contents(messages)
        gen_cfg: Dict[str, Any] = {"maxOutputTokens": 8192, "temperature": 1.0}
        if "thinking" in (cloud_model or "").lower():
            eff = (self.effort or "").strip().lower() if isinstance(self.effort, str) else ""
            gen_cfg["thinkingConfig"] = {
                "thinkingLevel": _EFFORT_TO_THINKING_LEVEL.get(eff, "MEDIUM")
            }

        def _envelope() -> Dict[str, Any]:
            inner: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
            if system_instruction is not None:
                inner["systemInstruction"] = system_instruction
            return {
                "project": project,
                "model": cloud_model,
                "request": inner,
                "userAgent": "antigravity",
                "requestId": str(uuid.uuid4()),
            }

        try:
            payload = self._post_cloudcode(token, _envelope())
            return self._parse_cloudcode_payload(payload)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            is_auth = (
                e.code in (401, 403)
                or _is_auth_output(body)
                or "UNAUTHENTICATED" in (body or "").upper()
            )
            if not is_auth:
                raise ProviderError(f"Antigravity HTTP Error {e.code}: {e.reason}")
            # Refresh once via vault then retry 1x.
            new_token = None
            try:
                candidates: List[str] = []
                for _name in ("discover_antigravity_token", "get_antigravity_token"):
                    _fn = getattr(vault, _name, None) if vault is not None else None
                    if callable(_fn):
                        try:
                            _v = _fn() if _name != "discover_antigravity_token" else _fn()
                            if isinstance(_v, str) and _v.strip():
                                candidates.append(_v.strip())
                        except Exception:
                            pass
                try:
                    _gt = getattr(vault, "get_token", None) if vault is not None else None
                    if callable(_gt):
                        _v = _gt("antigravity")
                        if isinstance(_v, str) and _v.strip():
                            candidates.append(_v.strip())
                except Exception:
                    pass
                for _c in candidates:
                    if _c != token:
                        new_token = _c
                        break
                if new_token is None:
                    new_token = _resolve_vault_token(vault)
                    if new_token == token:
                        new_token = None
            except Exception:
                new_token = None
            if new_token and new_token != token:
                try:
                    payload = self._post_cloudcode(new_token, _envelope())
                    return self._parse_cloudcode_payload(payload)
                except urllib.error.HTTPError as e2:
                    body2 = ""
                    try:
                        body2 = e2.read().decode("utf-8", errors="replace")
                    except Exception:
                        body2 = ""
                    detail = (body2 or body or "").strip()
                    suffix = f": {detail}" if detail else ""
                    raise ProviderError(
                        f"Antigravity auth gagal, run /login antigravity{suffix}"
                    )
                except (urllib.error.URLError, TimeoutError) as e2:
                    raise ProviderError(
                        f"Antigravity auth gagal, run /login antigravity: {e2}"
                    )
            detail = (body or "").strip()
            suffix = f": {detail}" if detail else ""
            raise ProviderError(f"Antigravity auth gagal, run /login antigravity{suffix}")
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (socket.timeout, builtins.TimeoutError, TimeoutError)):
                raise TimeoutError(f"Antigravity timed out after 120s: {reason}")
            if isinstance(reason, OSError) and "timed out" in str(reason).lower():
                raise TimeoutError(f"Antigravity timed out after 120s: {reason}")
            raise ProviderError(f"Antigravity Request Error: {reason}")
        except (socket.timeout, builtins.TimeoutError) as e:
            raise TimeoutError(f"Antigravity timed out after 120s: {e}")

    def _chat_via_cli(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        prompt = (
            messages[0].content
            if len(messages) == 1
            else "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        )

        cmd = self._build_agy_cmd(prompt)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            raise ProviderError(
                "Antigravity CLI (`agy`) not found in PATH.\n"
                "Install from https://antigravity.google/docs/cli/install/ "
                "then run `/login antigravity`."
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("Antigravity CLI timed out after 120s.")

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if _is_auth_output(err):
                # Fail-fast auth: explicit /login hint (gateway _is_auth_error catches
                # "auth" + "/login"; never raise as TimeoutError/network).
                detail = f": {err}" if err else ""
                raise ProviderError(
                    f"Antigravity auth gagal, run /login antigravity{detail}"
                )
            raise ProviderError(
                f"Antigravity CLI failed (exit {proc.returncode}): {err}"
            )

        return {
            "role": "assistant",
            "content": _clean(proc.stdout or ""),
            "tool_calls": None,
        }

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        vault = getattr(self, "vault", None)
        if vault is None:
            try:
                vault = AuthVault()
                self.vault = vault
            except Exception:
                vault = None
        token = _resolve_vault_token(vault) if vault is not None else None
        project = _resolve_vault_project(vault) if vault is not None else None
        if token and project:
            return self._chat_direct(token, project, messages)
        return self._chat_via_cli(messages)
