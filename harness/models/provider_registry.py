"""
ProviderRegistry — multi-provider model catalog for CodeAI.

Design inspired by opencode (provider.ts) and omp (registry.ts):

  autoload pattern:
    A provider is "autoloaded" (shown in /model) only when:
      1. It has credentials stored, AND
      2. It has at least one known model in the registry OR can discover models live.

  Model discovery:
    Providers with empty model lists trigger a live /v1/models fetch (OpenAI-compatible).
    Results are cached in-session.

  Credential check precedence (mirrors opencode auth resolution):
    1. Environment variable  (PROVIDER_API_KEY)
    2. Stored vault token    (~/.codeai/auth.json)
    3. Provider-specific discover_* method  (copilot, antigravity)
    4. Ollama local endpoint check
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.models.providers.antigravity import (
    EFFORT_MODELS,
    EFFORT_OPTIONS,
    parse_model_effort,
)

# Bare model strings (no "provider/" prefix) resolve to this provider.
# Unified with gateway/config default ("anthropic"); previously diverged as "openai".
BARE_DEFAULT_PROVIDER = "anthropic"

# oh-my-pi style ":level" suffix aliases canonicalized to antigravity effort.
_LEVEL_ALIASES = {"xhigh": "high"}

_CUSTOM_BASE_URL_KEYS = ("baseURL", "baseUrl", "base_url", "api")


def _local_reachable(api: Any) -> bool:
    """Probe local endpoint reachability (single source of truth).

    True only when the endpoint is actually reachable — never a blind
    True for localhost. Fast TCP connect (0.5s) first, then fallback to
    GET {base}/api/tags (1.0s, ollama semantics).
    """
    import socket
    import urllib.parse

    try:
        s = str(api or "").strip()
        if not s or not s.lower().startswith("http"):
            return False
        base = s.rstrip("/")
        if base.lower().endswith("/v1"):
            base = base[:-3].rstrip("/")
        try:
            parsed = urllib.parse.urlparse(s)
            host = parsed.hostname or ""
            port = parsed.port
            if host and port is None:
                port = 443 if parsed.scheme == "https" else 80
            if host and port:
                try:
                    with socket.create_connection((host, port), timeout=0.5):
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            req = urllib.request.Request(f"{base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return bool(getattr(resp, "status", 200) == 200)
        except Exception:
            return False
    except Exception:
        return False


class ProviderRegistry:
    def __init__(self, custom_providers: Optional[Dict[str, Any]] = None):
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._load_embedded_registry()
        self._load_opencode_registry()
        if custom_providers:
            self.register_custom_providers(custom_providers)

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _load_embedded_registry(self):
        """Built-in provider definitions with known model lists."""
        embedded = {
            "openai": {
                "name": "OpenAI",
                "api": "https://api.openai.com/v1",
                "models": {m: {} for m in ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]},
            },
            "anthropic": {
                "name": "Anthropic",
                "api": "https://api.anthropic.com/v1",
                "models": {m: {} for m in [
                    "claude-3-5-sonnet-20240620",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ]},
            },
            # Gemini REST API (AI Studio API key, AIza... prefix)
            "gemini": {
                "name": "Gemini API",
                "api": "https://generativelanguage.googleapis.com/v1beta",
                "models": {m: {} for m in [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ]},
            },
            # Antigravity — Google Antigravity via `agy` CLI (OAuth, effort-based models)
            # Only base model names here — effort (low/medium/high) is selected interactively.
            # Single source of truth for effort lives in
            # harness.models.providers.antigravity (EFFORT_MODELS/EFFORT_OPTIONS).
            "antigravity": {
                "name": "Antigravity",
                "api": "agy://cli",
                "models": {m: {} for m in [
                    "gemini-3.8-flash",
                    "gemini-3.7-flash",
                    "gemini-3.6-flash",
                    "gemini-3.1-pro",
                    "claude-sonnet-4-6",
                    "claude-opus-4-6-thinking",
                    "gpt-oss-120b",
                ]},
            },
            "copilot": {
                "name": "Copilot",
                "api": "https://api.githubcopilot.com",
                "models": {m: {} for m in ["gpt-4o", "claude-3.5-sonnet", "o1", "o3-mini"]},
            },
            "ollama": {
                "name": "Ollama",
                "api": "http://127.0.0.1:11434/v1",
                "models": {m: {} for m in ["llama3", "qwen2.5-coder", "mistral"]},
            },
            # Providers with no hardcoded model list — discovered live or shown with placeholder
            "openrouter":  {"name": "OpenRouter",  "api": "https://openrouter.ai/api/v1",          "models": {}},
            "deepseek":    {"name": "DeepSeek",     "api": "https://api.deepseek.com/v1",           "models": {}},
            "groq":        {"name": "Groq",         "api": "https://api.groq.com/openai/v1",        "models": {}},
            "mistral":     {"name": "Mistral",      "api": "https://api.mistral.ai/v1",             "models": {}},
            "together":    {"name": "Together",     "api": "https://api.together.xyz/v1",           "models": {}},
            "xai":         {"name": "xAI",          "api": "https://api.x.ai/v1",                  "models": {}},
            "cerebras":    {"name": "Cerebras",     "api": "https://api.cerebras.ai/v1",            "models": {}},
            "fireworks":   {"name": "Fireworks",    "api": "https://api.fireworks.ai/inference/v1", "models": {}},
            "perplexity":  {"name": "Perplexity",   "api": "https://api.perplexity.ai",             "models": {}},
            "sambanova":   {"name": "Sambanova",    "api": "https://api.sambanova.ai/v1",           "models": {}},
            "opencode":    {"name": "OpenCode",     "api": "https://api.opencode.ai/v1",            "models": {}},
            "combo":       {"name": "Combo",        "api": "local",                                 "models": {}},
        }
        self._providers.update(embedded)

    def _load_opencode_registry(self):
        """
        Load provider data from opencode's local models.json cache.

        Special handling for 'opencode' provider:
          - Load ALL models from cache (not capped at 20) — it's our own provider
          - Update the API endpoint to the correct zen URL

        For other known providers:
          - Only enrich API endpoint if missing
          - Add up to 20 extra models beyond our hardcoded list
        """
        opencode_path = Path.home() / ".cache" / "opencode" / "models.json"
        if not opencode_path.exists():
            return
        try:
            with open(opencode_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        for provider_id, provider_info in data.items():
            if not isinstance(provider_info, dict):
                continue
            if provider_id not in self._providers:
                continue

            existing = self._providers[provider_id]

            # Always update API endpoint from cache if it's more specific
            if provider_info.get("api"):
                existing["api"] = provider_info["api"]

            cache_models = provider_info.get("models", {})
            if not isinstance(cache_models, dict):
                continue

            if provider_id == "opencode":
                # Load ALL opencode models — this is the complete Zen model catalog
                existing["models"] = {k: {} for k in cache_models.keys()}
                existing["name"] = provider_info.get("name", "OpenCode Zen")
            elif cache_models and existing.get("models"):
                # For other providers: merge up to 20 extra models
                current = existing["models"]
                additions = {k: v for k, v in cache_models.items() if k not in current}
                for k, v in list(additions.items())[:20]:
                    current[k] = {}

    def get_antigravity_effort_info(self):
        """Return effort-based model info for Antigravity provider.

        Single source of truth: harness.models.providers.antigravity.
        """
        return {
            "effort_models": sorted(EFFORT_MODELS),
            "effort_options": list(EFFORT_OPTIONS),
        }

    # ------------------------------------------------------------------
    # Custom providers (opencode pattern: openai-compatible baseURL)
    # ------------------------------------------------------------------

    def register_custom_providers(self, custom_providers: Dict[str, Any]) -> None:
        """Register user-defined OpenAI-compatible providers.

        Expected shape (mirrors opencode custom provider):
          {"my-provider": {"baseURL": "https://.../v1", "models": [...]}, ...}
        Accepts baseURL/baseUrl/base_url/api key variants. Never crashes on
        malformed entries — they are skipped.
        """
        if not isinstance(custom_providers, dict):
            return
        for pid, cfg in custom_providers.items():
            try:
                if not isinstance(pid, str) or not pid.strip():
                    continue
                pid = pid.strip()
                if not isinstance(cfg, dict):
                    continue
                base_url = ""
                for k in _CUSTOM_BASE_URL_KEYS:
                    v = cfg.get(k)
                    if isinstance(v, str) and v.strip():
                        base_url = v.strip()
                        break
                if not base_url:
                    continue
                name = cfg.get("name") if isinstance(cfg.get("name"), str) else pid
                raw_models = cfg.get("models", {})
                models: Dict[str, Any] = {}
                if isinstance(raw_models, dict):
                    models = {str(k): {} for k in raw_models.keys()}
                elif isinstance(raw_models, list):
                    models = {str(m): {} for m in raw_models if isinstance(m, str)}
                if pid not in self._providers:
                    self._providers[pid] = {"name": name, "api": base_url, "models": models}
                else:
                    # Don't clobber built-ins; only fill missing api endpoint.
                    if not self._providers[pid].get("api"):
                        self._providers[pid]["api"] = base_url
                self._providers[pid]["_custom"] = True if pid not in (
                    "openai", "anthropic", "gemini", "antigravity", "copilot",
                    "ollama", "openrouter", "deepseek", "groq", "mistral",
                    "together", "xai", "cerebras", "fireworks", "perplexity",
                    "sambanova", "opencode", "combo",
                ) else self._providers[pid].get("_custom", False)
            except Exception:
                continue

    def is_custom_provider(self, provider_id: str) -> bool:
        info = self._providers.get(provider_id)
        return bool(info and info.get("_custom"))

    def unregister_custom_provider(self, provider_id: str) -> bool:
        """Remove a previously registered custom provider.

        Normalizes id via strip/lower. Returns False for builtin ids,
        unknown ids, or entries without ``_custom`` True; otherwise
        deletes the entry and returns True. Never raises.
        """
        try:
            if not isinstance(provider_id, str):
                return False
            norm = provider_id.strip().lower()
            if not norm:
                return False
            if norm in (
                "openai", "anthropic", "gemini", "antigravity", "copilot",
                "ollama", "openrouter", "deepseek", "groq", "mistral",
                "together", "xai", "cerebras", "fireworks", "perplexity",
                "sambanova", "opencode", "combo",
            ):
                return False
            target_key = None
            for k in list(self._providers.keys()):
                if isinstance(k, str) and k.strip().lower() == norm:
                    target_key = k
                    break
            if target_key is None:
                return False
            info = self._providers.get(target_key)
            if not isinstance(info, dict) or not info.get("_custom"):
                return False
            del self._providers[target_key]
            return True
        except Exception:
            return False

    def build_custom_provider(self, provider_id: str, api_key: str = "", default_model: str = ""):
        """Passthrough: build a UniversalOpenAIProvider for a custom provider.

        Never raises for unknown ids — raises only if the id has no descriptor
        with an http(s) baseURL. Lazy import keeps registry dependency-light.
        """
        from harness.models.providers.universal_openai import UniversalOpenAIProvider

        descriptor = self._providers.get(provider_id)
        if not descriptor or not str(descriptor.get("api", "")).startswith("http"):
            raise ValueError(f"Custom provider '{provider_id}' has no valid baseURL.")
        return UniversalOpenAIProvider(
            base_url=str(descriptor["api"]),
            api_key=api_key,
            default_model=default_model,
        )

    # ------------------------------------------------------------------
    # Credential detection  (mirrors opencode/omp auth resolution order)
    # ------------------------------------------------------------------

    def _has_credentials(self, provider_id: str, info: Dict[str, Any]) -> bool:
        """
        Return True when a usable credential exists.
        Resolution order:
          1. Environment variable  (PROVIDER_API_KEY)
          2. Stored vault token    (~/.codeai/auth.json)
          3. Provider-specific discover_* method
          4. Ollama local reachability check
        """
        from harness.models.auth_vault import AuthVault

        if os.environ.get(f"{provider_id.upper()}_API_KEY"):
            return True

        vault = AuthVault()

        # Antigravity: needs agy CLI + session token
        if provider_id == "antigravity":
            import shutil
            if not shutil.which("agy"):
                return False
            return bool(vault.get_token("antigravity") or vault.discover_antigravity_token())

        # Standard vault lookup
        if vault.get_token(provider_id):
            return True

        # Provider-specific discover method
        discover = getattr(vault, f"discover_{provider_id}_token", None)
        if callable(discover):
            try:
                if discover():
                    return True
            except Exception:
                pass

        # Step 4: local reachability probe (single source: _local_reachable).
        # True only when actually reachable; never a blind True.
        if provider_id == "ollama":
            return _local_reachable(info.get("api", ""))

        # Custom / local OpenAI-compatible endpoints: probe, don't assume.
        api = str(info.get("api", ""))
        if "127.0.0.1" in api or "localhost" in api:
            return _local_reachable(api)

        return False

    # ------------------------------------------------------------------
    # Live model discovery  (opencode pattern: discover on-demand)
    # ------------------------------------------------------------------

    def _discover_models_live(self, provider_id: str) -> List[str]:
        """
        Fetch /v1/models from an OpenAI-compatible endpoint.
        Caches results into the in-memory registry for this session.
        Returns [] on any failure.
        """
        info = self._providers.get(provider_id, {})
        base_url = info.get("api", "")

        # Skip non-HTTP providers (agy, local, etc.)
        if not base_url.startswith("http"):
            return []

        from harness.models.auth_vault import AuthVault
        import json as _json

        vault = AuthVault()
        api_key = (
            os.environ.get(f"{provider_id.upper()}_API_KEY")
            or vault.get_token(provider_id)
        )
        if not api_key:
            return []

        # Build /v1/models URL
        models_url = base_url.rstrip("/")
        if not models_url.endswith("/models"):
            # Strip trailing /v1 then re-add /v1/models
            clean = models_url
            if clean.endswith("/v1"):
                clean = clean[:-3]
            models_url = clean.rstrip("/") + "/v1/models"

        try:
            req = urllib.request.Request(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read().decode("utf-8"))

            models: List[str] = []
            # OpenAI-compatible: {"data": [{"id": "..."}]}
            if isinstance(data, dict) and "data" in data:
                models = [
                    m["id"] for m in data["data"]
                    if isinstance(m, dict) and isinstance(m.get("id"), str)
                ]
            # Some APIs: {"models": [...]}
            elif isinstance(data, dict) and "models" in data:
                items = data["models"]
                if items and isinstance(items[0], str):
                    models = items
                else:
                    models = [
                        m.get("id") or m.get("name", "")
                        for m in items if isinstance(m, dict)
                    ]
                models = [m for m in models if m]

            if models:
                # Cache into registry for this session
                if "models" not in self._providers[provider_id]:
                    self._providers[provider_id]["models"] = {}
                for m in models:
                    self._providers[provider_id]["models"].setdefault(m, {})

            return models

        except Exception:
            return []

    # ------------------------------------------------------------------
    # Default model fallbacks per provider
    # ------------------------------------------------------------------

    _FALLBACK_DEFAULTS: Dict[str, str] = {
        "openai":     "gpt-4o-mini",
        "anthropic":  "claude-3-5-sonnet-20240620",
        "gemini":     "gemini-2.5-flash",
        "copilot":    "gpt-4o",
        "ollama":     "llama3",
        "opencode":   "mimo-v2.5-free",
        "openrouter": "openai/gpt-4o-mini",
        "deepseek":   "deepseek-chat",
        "groq":       "llama3-8b-8192",
        "xai":        "grok-3",
        "mistral":    "mistral-large-latest",
        "together":   "meta-llama/Llama-3-8b-chat-hf",
    }

    # ------------------------------------------------------------------
    # Effort parsing (antigravity single source + oh-my-pi ":level" suffix)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_antigravity_model(model_str: str) -> Tuple[str, Optional[str]]:
        """Parse antigravity model with `-effort` or `:level` suffix.

        Examples:
          "gemini-3.8-flash"        → ("gemini-3.8-flash", None)
          "gemini-3.8-flash-high"   → ("gemini-3.8-flash", "high")
          "gemini-3.8-flash:high"   → ("gemini-3.8-flash", "high")
          "claude-sonnet-4-6"       → ("claude-sonnet-4-6", None)
        Canonical form is always "base-effort" (dash, never colon).
        """
        s = (model_str or "").strip()
        if ":" in s:
            base_part, level = s.rsplit(":", 1)
            lvl = _LEVEL_ALIASES.get(level.strip().lower(), level.strip().lower())
            if lvl in EFFORT_OPTIONS:
                base_part = base_part.strip()
                # Base may itself carry "-effort"; strip it to get true base.
                true_base, _ = parse_model_effort(base_part)
                if true_base in EFFORT_MODELS:
                    return true_base, lvl
                # Unknown base with valid level — still normalize colon→dash
                # so ids stay canonical, but caller treats effort as None
                # unless base supports it. Keep strict: return as-is base.
                return base_part, None
        return parse_model_effort(s)

    @classmethod
    def canonicalize_antigravity_model(cls, model_str: str) -> str:
        base, effort = cls.parse_antigravity_model(model_str)
        return f"{base}-{effort}" if effort else base

    def _split_model_effort(self, provider_id: str, model: str) -> Tuple[str, Optional[str], str]:
        """Return (canonical_model, base, effort) for a listing entry."""
        if provider_id == "antigravity":
            base, effort = self.parse_antigravity_model(model)
            canonical = f"{base}-{effort}" if effort else base
            return canonical, base, effort
        return model, model, None

    def is_known_model(self, provider: str, model: str) -> bool:
        """Return True when (provider, model) is a known registry entry.

        Checks bundled embedded list + opencode-cache merge + custom
        providers + live-discovered session cache (all stored in
        ``self._providers``). Comparison is case-insensitive and ignores
        trailing effort suffix (``-low``/``-medium``/``-high``/``-xhigh``,
        ``:``-syntax).

        Providers with an empty model catalog (live-discovery providers
        such as openrouter/deepseek before first fetch) allow passthrough
        and return True. Unknown provider ids return False.
        """
        try:
            if not isinstance(provider, str) or not isinstance(model, str):
                return False
            prov = provider.strip()
            mod = model.strip()
            if not prov or not mod:
                return False
            low_prov = prov.lower()
            prov_key = None
            for k in self._providers.keys():
                if isinstance(k, str) and k.lower() == low_prov:
                    prov_key = k
                    break
            if prov_key is None:
                return False
            descriptor = self._providers.get(prov_key, {})
            models_dict = descriptor.get("models", {})
            if not isinstance(models_dict, dict):
                return False
            if len(models_dict) == 0:
                return True
            base = mod
            try:
                if prov_key == "antigravity":
                    b2, _ = self.parse_antigravity_model(mod)
                    base = b2
                else:
                    b, _ = parse_model_effort(mod)
                    base = b
            except Exception:
                base = mod
            base = (base or mod).strip()
            if not base:
                return False
            known = {str(k).strip().lower() for k in models_dict.keys()}
            return base.strip().lower() in known
        except Exception:
            return False

    def validate_model_string(self, full_name: str) -> Tuple[str, str]:
        """Parse and reject unknown provider/model with a clear ValueError.

        Strict backend validator built on :meth:`parse_model_string` +
        :meth:`is_known_model`. Live-discovery providers with an empty
        catalog still pass through; everything else unknown raises.
        """
        s = (full_name or "").strip()
        if not s:
            raise ValueError("Empty model string. Expected 'provider/model' or bare 'model'.")
        provider, model = self.parse_model_string(s)
        # parse_model_string already raises for explicit known-provider
        # unknown-model; re-check for bare/unknown-provider strictness.
        if not self.is_known_model(provider, model):
            raise ValueError(
                f"Unknown model '{s}' (parsed as '{provider}/{model}'). "
                "Run '/providers' to see connected models."
            )
        return provider, model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_providers(self) -> List[Dict[str, Any]]:
        result = []
        for provider_id, info in self._providers.items():
            result.append({
                "id": provider_id,
                "name": info.get("name", provider_id),
                "api": info.get("api"),
                "has_credentials": self._has_credentials(provider_id, info),
            })
        return sorted(result, key=lambda x: x["id"])

    def get_provider_descriptor(self, provider_id: str) -> Optional[Dict[str, Any]]:
        return self._providers.get(provider_id)

    def list_models(self, provider_id: str) -> List[str]:
        """Return known model list for a provider (no live fetch)."""
        provider = self._providers.get(provider_id)
        if not provider or "models" not in provider:
            return []
        models_dict = provider.get("models", {})
        if isinstance(models_dict, dict):
            return list(models_dict.keys())
        return []

    def list_connected_models(self) -> List[Dict[str, Any]]:
        """
        Return models for all connected providers.

        Autoload logic (mirrors opencode):
          - Provider with non-empty model list → show all models directly.
          - Provider with empty model list → attempt live /v1/models discovery.
          - If discovery succeeds → show discovered models.
          - If discovery fails → show one placeholder entry so the provider is
            still reachable (user can type a custom model string in /model).

        Each entry carries canonical id plus split base/effort:
          {"provider", "model" (canonical), "id" (canonical),
           "base", "effort"}
        Antigravity ":level" suffixes are normalized to "base-effort".

        Providers with no credentials are not shown.
        """
        connected = [
            p["id"] for p in self.list_providers() if p["has_credentials"]
        ]
        models: List[Dict[str, Any]] = []

        def _emit(p_id: str, m: str) -> None:
            canonical, base, effort = self._split_model_effort(p_id, m)
            try:
                _known = self.is_known_model(p_id, base)
            except Exception:
                _known = True
            models.append({
                "provider": p_id,
                "model": canonical,
                "id": f"{p_id}/{canonical}",
                "base": base,
                "effort": effort,
                "known": _known,
            })

        for p_id in connected:
            known = self.list_models(p_id)

            if known:
                # autoload: True — known models, show them all
                for m in known:
                    _emit(p_id, m)
            else:
                # autoload: conditional — try live discovery
                live = self._discover_models_live(p_id)
                if live:
                    for m in live:
                        _emit(p_id, m)
                else:
                    # Fallback placeholder — still lets user select this provider
                    default = self._FALLBACK_DEFAULTS.get(p_id, "default")
                    _emit(p_id, default)

        return models

    def parse_model_string(self, full_name: str) -> Tuple[str, str]:
        """Parse "provider/model" or bare "model".

        Bare strings resolve to BARE_DEFAULT_PROVIDER ("anthropic", unified
        with gateway/config). Antigravity ":level" suffixes normalize to
        canonical "base-effort" (canonicalization unchanged).

        Backend validation: explicit "provider/model" where the provider is
        known AND has a non-empty catalog raises ValueError for unknown
        models (checked via is_known_model, case-insensitive, effort
        suffix ignored). Bare strings, unknown providers, and
        live-discovery providers with an empty catalog still pass through
        (custom-provider passthrough via registry descriptors +
        UniversalOpenAIProvider).
        """
        s = (full_name or "").strip()
        if "/" in s:
            provider, model = s.split("/", 1)
            provider, model = provider.strip(), model.strip()
            if provider == "antigravity":
                canonical = self.canonicalize_antigravity_model(model)
                try:
                    _base, _ = self.parse_antigravity_model(model)
                except Exception:
                    _base = model
                if not self.is_known_model(provider, _base):
                    raise ValueError(
                        f"Unknown model '{model}' for provider '{provider}'. "
                        "Run '/providers' to see connected models."
                    )
                return provider, canonical
            if model and not self.is_known_model(provider, model):
                _low = provider.lower()
                _key = None
                for k in self._providers.keys():
                    if isinstance(k, str) and k.lower() == _low:
                        _key = k
                        break
                if _key is not None:
                    _md = self._providers.get(_key, {}).get("models", {})
                    if isinstance(_md, dict) and len(_md) > 0:
                        raise ValueError(
                            f"Unknown model '{model}' for provider '{provider}'. "
                            "Run '/providers' to see connected models."
                        )
            return provider, model
        if not s:
            return BARE_DEFAULT_PROVIDER, s
        # Bare antigravity-style "base:level" or "base-effort" → normalize.
        base, effort = self.parse_antigravity_model(s)
        if effort:
            return BARE_DEFAULT_PROVIDER, f"{base}-{effort}"
        return BARE_DEFAULT_PROVIDER, s
