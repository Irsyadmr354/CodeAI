import builtins
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from harness.config import ProviderConfig
from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError
from harness.models.providers import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider, CopilotProvider
from harness.models.provider_registry import ProviderRegistry
from harness.models.providers.universal_openai import UniversalOpenAIProvider
from harness.models.auth_vault import AuthVault

logger = logging.getLogger(__name__)

# Cooldown for recently-failed providers (opencode-style, seconds).
_FAILOVER_COOLDOWN_S = 300.0


def _is_timeout_error(exc: BaseException) -> bool:
    """True for stdlib timeouts in addition to harness TimeoutError."""
    if isinstance(exc, TimeoutError):  # harness TimeoutError (ProviderError)
        return True
    if isinstance(exc, (socket.timeout, builtins.TimeoutError)):
        return True
    if isinstance(exc, OSError) and "timed out" in str(exc).lower():
        return True
    # urllib wraps timeouts: URLError(reason=TimeoutError/socket.timeout)
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (socket.timeout, builtins.TimeoutError, TimeoutError, OSError)):
        return "timed out" in str(reason).lower() or isinstance(
            reason, (socket.timeout, builtins.TimeoutError, TimeoutError)
        )
    return "timed out" in str(exc).lower() and isinstance(exc, OSError)


def _safe_parse_model_effort(model_str: Any) -> Any:
    """Whitespace-tolerant wrapper around antigravity.parse_model_effort (stdlib only).

    Strips outer whitespace and tolerates inner spaces around the effort
    separator (e.g. "gemini-3.8-flash - low", "gemini-3.8-flash : low",
    " gemini-3.8-flash-low "). Model names never contain spaces, so a
    whitespace-collapsed retry is safe. Returns (base, effort|None).
    """
    from harness.models.providers.antigravity import parse_model_effort as _pme

    if not isinstance(model_str, str):
        return model_str, None
    s = model_str.strip()
    if not s:
        return model_str, None
    try:
        base, effort = _pme(s)
        if isinstance(base, str):
            base = base.strip()
        if isinstance(effort, str):
            effort = effort.strip().lower() or None
        if effort:
            return base, effort
    except Exception:
        pass
    try:
        compact = "".join(s.split())
        if compact and compact != s:
            base2, effort2 = _pme(compact)
            if isinstance(base2, str):
                base2 = base2.strip()
            if isinstance(effort2, str):
                effort2 = effort2.strip().lower() or None
            if effort2:
                return base2, effort2
            return compact, None
    except Exception:
        pass
    return s, None


def _is_auth_error(exc: BaseException) -> bool:
    """True when an error looks like missing/invalid credentials (needs /provider)."""
    try:
        msg = f"{type(exc).__name__}: {exc}".lower()
    except Exception:
        return False
    keys = (
        "credential", "auth", "/provider", "api key", "apikey",
        "unauthorized", "401", "forbidden", "403", "token",
        "agy", "re-authenticate", "authenticate",
    )
    return any(k in msg for k in keys)


def _is_connection_error(exc: BaseException) -> bool:
    """True when an error looks like a network/connection failure."""
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    keys = (
        "connection", "refused", "connect", "unreachable",
        "network", "econn", "socket", "127.0.0.1", "localhost",
        "11434", "ollama", "timed out", "timeout",
    )
    return any(k in msg for k in keys)


def _is_unavailable_error(exc: BaseException) -> bool:
    """True when a provider is unavailable (not installed / no server).

    Covers connection-refused / not-installed / no-server cases:
    Errno 111, FileNotFoundError, ConnectionRefusedError, and message
    markers ("not installed", "not found in PATH", "no server",
    "connection refused", "errno 111", "failed to connect",
    "credentials not found"). Skipped candidates must not become the
    final error when usable candidates exist (see chat() aggregation).
    """
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, ConnectionRefusedError):
        return True
    try:
        if getattr(exc, "errno", None) == 111:
            return True
        reason = getattr(exc, "reason", None)
        if reason is not None:
            if isinstance(reason, (FileNotFoundError, ConnectionRefusedError)):
                return True
            if getattr(reason, "errno", None) == 111:
                return True
            try:
                rmsg = str(reason).lower()
                if "errno 111" in rmsg or "connection refused" in rmsg:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    try:
        msg = f"{type(exc).__name__}: {exc}".lower()
    except Exception:
        return False
    keys = (
        "not installed",
        "not found in path",
        "no server",
        "connection refused",
        "errno 111",
        "failed to connect",
        "credentials not found",
    )
    return any(k in msg for k in keys)


def _parse_local_host_port(base_url: str):
    """Parse (host, port) from a local baseURL (stdlib only)."""
    try:
        from urllib.parse import urlparse as _urlparse

        u = _urlparse(base_url)
        host = u.hostname or "127.0.0.1"
        port = u.port
        if port is None:
            port = 443 if u.scheme == "https" else 80
        return host, port
    except Exception:
        return None


def _local_endpoint_reachable(base_url: str, timeout: float = 0.5) -> bool:
    """Fast socket probe for local endpoints (stdlib only, never raises)."""
    hp = _parse_local_host_port(base_url)
    if hp is None:
        return True
    host, port = hp
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class LLMGateway:
    def __init__(self, config: Any) -> None:
        # Accept either a raw ProviderConfig or a CodeAIConfig wrapper
        if hasattr(config, "provider"):
            self.config = config.provider
        else:
            self.config = config
        self._providers: Dict[str, BaseProvider] = {}
        self.registry = ProviderRegistry()
        # active_model tracks the currently selected model from /model switch
        self.active_model: Optional[str] = getattr(self.config, "active_model", None)
        self._apply_lock = threading.RLock()
        self._cooldown_lock = threading.Lock()
        self._provider_cooldown: Dict[str, float] = {}
        self._initialize_providers()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_providers(self) -> None:
        """Instantiate available built-in providers (lazy — only default + failovers)."""
        from harness.models.providers.antigravity import AntigravityProvider
        builtin = {
            "openai":       OpenAIProvider,
            "anthropic":    AnthropicProvider,
            "gemini":       GeminiProvider,
            "ollama":       OllamaProvider,
            "copilot":      CopilotProvider,
            "antigravity":  AntigravityProvider,
        }

        for name in [self.config.default] + list(self.config.failover_order):
            if name in builtin and name not in self._providers:
                try:
                    self._providers[name] = builtin[name]()
                except Exception as e:
                    logger.debug(f"Could not pre-initialise provider '{name}': {e}")

    # ------------------------------------------------------------------
    # Model injection helpers
    # ------------------------------------------------------------------

    def _config_effort(self) -> Optional[str]:
        for key in ("effort", "antigravity_effort", "default_effort"):
            val = getattr(self.config, key, None)
            if isinstance(val, str) and val.strip().lower() in ("low", "medium", "high"):
                return val.strip().lower()
        cfg = getattr(self.config, "model_dump", None)
        if callable(cfg):
            try:
                data = cfg() or {}
            except Exception:
                data = {}
            for key in ("effort", "antigravity_effort", "default_effort"):
                val = data.get(key)
                if isinstance(val, str) and val.strip().lower() in ("low", "medium", "high"):
                    return val.strip().lower()
        return None

    def _apply_active_model(self, provider: BaseProvider, model: str, _allow_global_effort: bool = True) -> None:
        """
        Inject the selected model into the provider so it is used for this request.
        Normalises `base-effort` / `base:effort` suffixes: the API always
        receives the CLEAN base name; effort is stored on `provider.effort`.
        Thread-safe via _apply_lock (caller should still snapshot/restore).
        When _allow_global_effort is False (combo-member isolated calls),
        global config effort is NOT injected unless the member provider
        explicitly supports effort (Antigravity EFFORT_MODELS); guard never crashes.
        """
        from harness.models.providers.antigravity import AntigravityProvider, EFFORT_MODELS, DEFAULT_ANTIGRAVITY_EFFORT
        with self._apply_lock:
            _m = model.strip() if isinstance(model, str) else model
            # Defensive: if a "provider/model" string leaks in here, use bare part.
            if isinstance(_m, str) and "/" in _m:
                _m = _m.split("/")[-1].strip() or _m.strip()
            base, effort = _safe_parse_model_effort(_m)
            if isinstance(base, str):
                base = base.strip()
            if isinstance(effort, str):
                effort = effort.strip().lower() or None
            cfg_effort = self._config_effort()
            if not _allow_global_effort:
                try:
                    _supports = bool(
                        isinstance(provider, AntigravityProvider)
                        and isinstance(base, str)
                        and base in EFFORT_MODELS
                    )
                except Exception:
                    _supports = False
                if not _supports:
                    cfg_effort = None
            if isinstance(provider, AntigravityProvider):
                provider.model = base
                provider.effort = effort or cfg_effort or (
                    DEFAULT_ANTIGRAVITY_EFFORT if base in EFFORT_MODELS else ""
                )
            elif isinstance(provider, (OpenAIProvider, AnthropicProvider, OllamaProvider, GeminiProvider)):
                provider.model = base
                provider.effort = effort or cfg_effort or getattr(provider, "effort", None) or ""
            elif isinstance(provider, CopilotProvider):
                provider.model = base
                provider.effort = effort or cfg_effort or getattr(provider, "effort", None) or ""
            elif isinstance(provider, UniversalOpenAIProvider):
                provider.default_model = base
                provider.effort = effort or cfg_effort or getattr(provider, "effort", None) or ""

    def _snapshot_provider(self, provider: BaseProvider) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for attr in ("model", "default_model", "effort"):
            if hasattr(provider, attr):
                try:
                    snap[attr] = getattr(provider, attr)
                except Exception:
                    pass
        return snap

    def _restore_provider(self, provider: BaseProvider, snap: Dict[str, Any]) -> None:
        for attr, val in snap.items():
            try:
                setattr(provider, attr, val)
            except Exception:
                pass

    def _build_failover_chain(self, target: str) -> List[str]:
        """failover_order first, then config fallbackChains (+ wildcard), deduped."""
        _target = target.strip() if isinstance(target, str) else target
        chain: List[str] = [_target]
        for p in list(getattr(self.config, "failover_order", []) or []):
            _p = p.strip() if isinstance(p, str) else p
            if _p != _target and _p not in chain:
                chain.append(_p)
        raw = getattr(self.config, "fallbackChains", None)
        if raw is None:
            try:
                raw = (self.config.model_dump() or {}).get("fallbackChains")
            except Exception:
                raw = None
        if isinstance(raw, dict):
            extras: List[str] = []
            extras += list(raw.get(_target, []) or [])
            extras += list(raw.get(f"{_target}/*", []) or [])
            extras += list(raw.get("*", []) or [])
            for p in extras:
                _p = p.strip() if isinstance(p, str) else p
                if isinstance(_p, str) and _p and _p not in chain:
                    chain.append(_p)
        return chain

    def _in_cooldown(self, name: str) -> bool:
        with self._cooldown_lock:
            ts = self._provider_cooldown.get(name)
            return ts is not None and (time.time() - ts) < _FAILOVER_COOLDOWN_S

    def _mark_cooldown(self, name: str) -> None:
        with self._cooldown_lock:
            self._provider_cooldown[name] = time.time()

    def _unavailable_skip_reason(self, name: str, is_target: bool) -> Optional[str]:
        """Pre-attempt skip for clearly-unconnected local providers (stdlib only).

        Failover only crosses CONNECTED providers: a non-target local
        endpoint (e.g. ollama on 127.0.0.1:11434) with no listener is
        skipped without forcing a chat attempt. Returns a short reason
        or None when the candidate should still be attempted. Target is
        never pre-skipped so its original error stays visible.
        """
        if is_target:
            return None
        try:
            desc = self.registry.get_provider_descriptor(name)
            if not desc:
                return None
            api = str(desc.get("api", "") or "")
            if "127.0.0.1" not in api and "localhost" not in api:
                return None
            base = api
            if name == "ollama":
                try:
                    base = os.environ.get("OLLAMA_BASE_URL", api) or api
                except Exception:
                    pass
            if _local_endpoint_reachable(base):
                return None
            return "not installed / no local server running"
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def get_provider(self, name: str) -> BaseProvider:
        """Return a provider instance by name, loading dynamically if necessary."""
        _name = name.strip() if isinstance(name, str) else name
        if _name == "combo":
            raise ProviderError("Combo provider requires a combo name as model (e.g. combo/my_combo).")

        with self._apply_lock:
            if _name in self._providers:
                return self._providers[_name]

            # Try to load dynamically via registry → UniversalOpenAIProvider
            descriptor = self.registry.get_provider_descriptor(_name)
            # Case-insensitive fallback for robustness ("Antigravity " etc.).
            if descriptor is None and isinstance(_name, str) and _name.lower() != _name:
                _low = _name.lower()
                descriptor = self.registry.get_provider_descriptor(_low)
                if descriptor is not None:
                    _name = _low
            if descriptor and descriptor.get("api"):
                vault = AuthVault()
                api_key = (
                    os.environ.get(f"{_name.upper()}_API_KEY")
                    or vault.get_token(_name)
                )
                # Try provider-specific discover method if available
                if not api_key:
                    discover = getattr(vault, f"discover_{_name}_token", None)
                    if callable(discover):
                        api_key = discover()

                base_url = descriptor["api"]
                is_local = "127.0.0.1" in base_url or "localhost" in base_url

                if not api_key and not is_local:
                    raise ProviderError(
                        f"Credentials not found for provider '{_name}'. "
                         f"Run '/provider {_name}' to authenticate."
                    )

                provider = UniversalOpenAIProvider(
                    base_url=base_url,
                    api_key=api_key or "",
                    default_model=getattr(self.config, f"{_name}_model", ""),
                )
                self._providers[_name] = provider
                return provider

            raise ProviderError(
                f"Provider '{_name}' is not configured or unsupported. "
                "Run '/providers' to see available options."
            )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request with automatic failover.
        model may be 'provider/model_name' or bare 'model_name'.
        """
        # ---- Combo routing (whitespace-tolerant: "combo / x" + bare R1 norm) ----
        effective_model = model or self.active_model
        if isinstance(effective_model, str):
            effective_model = effective_model.strip() or None
        _combo_name_to_run: Optional[str] = None
        if isinstance(effective_model, str) and effective_model:
            if "/" in effective_model:
                _c_left, _c_right = (p.strip() for p in effective_model.split("/", 1))
                if _c_left == "combo":
                    _combo_name_to_run = (_c_right.strip() or None)
                    if not _combo_name_to_run:
                        raise ProviderError("Combo provider requires a combo name as model (e.g. combo/my_combo).")
            else:
                _bare = effective_model.strip()
                if _bare == "combo":
                    raise ProviderError("Combo provider requires a combo name as model (e.g. combo/my_combo).")
                elif _bare:
                    try:
                        _def_raw = getattr(self.config, "default", "")
                        _def_norm = _def_raw.strip() if isinstance(_def_raw, str) else _def_raw
                    except Exception:
                        _def_norm = None
                    if _def_norm == "combo" and "/" not in _bare:
                        try:
                            from harness.models.combo import ComboManager as _CM
                            _cand = None
                            try:
                                _cand = _CM().get_combo(_bare)
                            except Exception:
                                _cand = None
                            if _cand is not None:
                                _combo_name_to_run = _bare
                        except Exception:
                            pass
        if _combo_name_to_run is not None:
            from harness.models.combo import ComboManager, ComboProvider
            try:
                _mgr = ComboManager()
                _cdef = _mgr.get_combo(_combo_name_to_run)
            except Exception:
                _cdef = None
            if not _cdef:
                raise ProviderError(f"Combo '{_combo_name_to_run}' not found.")
            _cprov = ComboProvider(_combo_name_to_run, _cdef, gateway_config=self.config)
            return _cprov.chat(messages, tools)

        # ---- Resolve provider + model_name from 'provider/model' string ----
        # Whitespace-tolerant: split "/" then strip each segment.
        _default_raw = getattr(self.config, "default", "anthropic")
        target_provider_name = _default_raw.strip() if isinstance(_default_raw, str) else _default_raw
        target_model_name: Optional[str] = effective_model
        explicit_target = False

        if effective_model and "/" in effective_model:
            _prov_raw, _mod_raw = (p.strip() for p in effective_model.split("/", 1))
            if _prov_raw and _mod_raw:
                # Only treat as provider/model if the left side is a known provider
                descriptor = self.registry.get_provider_descriptor(_prov_raw)
                _canonical = _prov_raw
                if descriptor is None and _prov_raw.lower() != _prov_raw:
                    _alt = self.registry.get_provider_descriptor(_prov_raw.lower())
                    if _alt is not None:
                        descriptor = _alt
                        _canonical = _prov_raw.lower()
                if descriptor is not None and _canonical != "combo":
                    target_provider_name = _canonical
                    target_model_name = _mod_raw
                    explicit_target = True
            elif _prov_raw and not _mod_raw:
                descriptor = self.registry.get_provider_descriptor(_prov_raw)
                _canonical = _prov_raw
                if descriptor is None and isinstance(_prov_raw, str) and _prov_raw.lower() != _prov_raw:
                    _alt = self.registry.get_provider_descriptor(_prov_raw.lower())
                    if _alt is not None:
                        descriptor = _alt
                        _canonical = _prov_raw.lower()
                if descriptor is not None and _canonical != "combo":
                    target_provider_name = _canonical
                    target_model_name = None
                    explicit_target = True
        if isinstance(target_model_name, str):
            target_model_name = target_model_name.strip() or None
        if isinstance(target_provider_name, str):
            target_provider_name = target_provider_name.strip() or target_provider_name

        # ---- Combo-member isolation (R2): member calls use empty chain ----
        _combo_isolated = False
        _combo_active_name: Optional[str] = None
        _combo_member_id: Optional[str] = None
        try:
            _def_cfg = getattr(self.config, "default", "")
            _def_cfg = _def_cfg.strip() if isinstance(_def_cfg, str) else _def_cfg
            if _def_cfg == "combo" and isinstance(model, str) and model.strip():
                _active_raw = getattr(self.config, "active_model", None)
                if _active_raw is None:
                    _active_raw = self.active_model
                _active_combo: Optional[str] = None
                if isinstance(_active_raw, str) and _active_raw.strip():
                    _a = _active_raw.strip()
                    if "/" in _a:
                        try:
                            _al, _ar = (p.strip() for p in _a.split("/", 1))
                        except Exception:
                            _al, _ar = "", ""
                        if _al == "combo" and _ar:
                            _active_combo = _ar
                    else:
                        if _a and _a != "combo":
                            _active_combo = _a
                if _active_combo:
                    try:
                        from harness.models.combo import ComboManager as _CM2
                        try:
                            _cdef2 = _CM2().get_combo(_active_combo)
                        except Exception:
                            _cdef2 = None
                    except Exception:
                        _cdef2 = None
                    if isinstance(_cdef2, dict):
                        try:
                            _members = _cdef2.get("models", []) or []
                        except Exception:
                            _members = []
                        try:
                            _m_norm = model.strip()
                        except Exception:
                            _m_norm = None
                        if isinstance(_m_norm, str) and _m_norm:
                            try:
                                for _m in _members:
                                    if isinstance(_m, str) and _m.strip() == _m_norm:
                                        _combo_isolated = True
                                        _combo_active_name = _active_combo
                                        _combo_member_id = _m_norm
                                        break
                            except Exception:
                                pass
        except Exception:
            _combo_isolated = False

        # Build the failover chain: target provider first, then configured failovers
        # Combo paths never leak to global failover (empty chain, single attempt).
        if _combo_isolated:
            providers_to_try = [target_provider_name]
        else:
            try:
                _targ_norm = target_provider_name.strip() if isinstance(target_provider_name, str) else target_provider_name
            except Exception:
                _targ_norm = target_provider_name
            if _targ_norm == "combo":
                if isinstance(target_model_name, str) and target_model_name.strip():
                    _tn = target_model_name.strip()
                    try:
                        from harness.models.combo import ComboManager as _CM3
                        try:
                            _chk = _CM3().get_combo(_tn)
                        except Exception:
                            _chk = None
                    except Exception:
                        _chk = None
                    if _chk is not None:
                        from harness.models.combo import ComboManager as _CM4, ComboProvider as _CP4
                        try:
                            _cdef4 = _CM4().get_combo(_tn)
                        except Exception:
                            _cdef4 = None
                        if _cdef4:
                            _cprov4 = _CP4(_tn, _cdef4, gateway_config=self.config)
                            return _cprov4.chat(messages, tools)
                    raise ProviderError(f"Combo '{_tn}' not found.")
                raise ProviderError("Combo provider requires a combo name as model (e.g. combo/my_combo).")
            providers_to_try = self._build_failover_chain(target_provider_name)

        last_error: Optional[Exception] = None
        target_error: Optional[Exception] = None
        attempts: List[str] = []
        ordered: List[Any] = []

        for provider_name in providers_to_try:
            # Critical section is per-provider attempt and covers
            # cooldown-check → snapshot → apply → chat → restore → cooldown-mark
            # atomically, so parallel fastest/consensus calls sharing one
            # gateway cannot interleave model/effort mutation. RLock keeps
            # nested get_provider/_apply_active_model safe.
            _pn = provider_name.strip() if isinstance(provider_name, str) else provider_name
            is_target = (_pn == target_provider_name)
            with self._apply_lock:
                if self._in_cooldown(_pn) and not (explicit_target and is_target):
                    logger.debug(
                        f"Skipping provider '{_pn}' (cooldown after recent failure)."
                    )
                    continue
                # Failover only across CONNECTED providers: skip a clearly
                # unconnected local failover candidate (e.g. ollama with no
                # listener) without forcing an attempt. Target is never
                # pre-skipped so its original cause stays visible.
                _skip_reason = self._unavailable_skip_reason(_pn, is_target)
                if _skip_reason:
                    logger.debug(
                        f"Skipping provider '{_pn}' (unavailable: {_skip_reason})."
                    )
                    attempts.append(f"{_pn}: skipped ({_skip_reason})")
                    continue
                snap: Dict[str, Any] = {}
                try:
                    provider = self.get_provider(_pn)
                    snap = self._snapshot_provider(provider)

                    # Inject active model into provider before calling (restored after)
                    if target_model_name:
                        if _combo_isolated:
                            try:
                                self._apply_active_model(provider, target_model_name, _allow_global_effort=False)
                            except TypeError:
                                self._apply_active_model(provider, target_model_name)
                        else:
                            self._apply_active_model(provider, target_model_name)

                    effective_tools = tools
                    try:
                        if tools and not provider.supports_tools():
                            logger.warning(
                                f"Provider '{_pn}' does not support tools — "
                                f"dropping {len(tools)} tool(s) for this call."
                            )
                            effective_tools = None
                    except Exception:
                        pass

                    logger.debug(
                        f"Calling provider '{_pn}' with model '{target_model_name or '<default>'}'"
                    )
                    try:
                        return provider.chat(messages, effective_tools)
                    finally:
                        try:
                            self._restore_provider(provider, snap)
                        except Exception:
                            pass

                except (RateLimitError, TimeoutError) as e:
                    logger.debug(
                        f"Provider '{_pn}' hit transient error ({type(e).__name__}): {e}. "
                        "Trying next failover..."
                    )
                    self._mark_cooldown(_pn)
                    ordered.append((_pn, e))
                    if _is_unavailable_error(e):
                        attempts.append(f"{_pn}: skipped ({e})")
                    else:
                        attempts.append(f"{_pn}: {e}")
                    if is_target and target_error is None:
                        target_error = e
                    last_error = e
                    continue
                except ProviderError as e:
                    ordered.append((_pn, e))
                    if _is_unavailable_error(e):
                        logger.debug(f"Provider '{_pn}' unavailable, skipping: {e}")
                        attempts.append(f"{_pn}: skipped ({e})")
                        if is_target and target_error is None:
                            target_error = e
                        last_error = e
                        continue
                    if is_target and target_error is None:
                        target_error = e
                    last_error = e
                    if _is_timeout_error(e):
                        logger.debug(
                            f"Provider '{_pn}' timeout via ProviderError: {e}. "
                            "Trying next failover..."
                        )
                        self._mark_cooldown(_pn)
                        continue
                    if explicit_target and is_target and _is_auth_error(e):
                        raise ProviderError(
                            f"Provider '{target_provider_name}' failed for model "
                            f"'{target_model_name or '<default>'}': {e} "
                             f"Run '/provider {target_provider_name}' or '/model' to switch."
                        ) from e
                    logger.debug(f"Provider '{_pn}' non-recoverable error: {e}")
                    attempts.append(f"{_pn}: {e}")
                    continue
                except Exception as e:
                    ordered.append((_pn, e))
                    if _is_unavailable_error(e):
                        logger.debug(f"Provider '{_pn}' unavailable, skipping: {e}")
                        attempts.append(f"{_pn}: skipped ({e})")
                        if is_target and target_error is None:
                            target_error = e
                        last_error = e
                        continue
                    if is_target and target_error is None:
                        target_error = e
                    last_error = e
                    if _is_timeout_error(e):
                        logger.debug(
                            f"Provider '{_pn}' hit stdlib timeout "
                            f"({type(e).__name__}): {e}. Trying next failover..."
                        )
                        self._mark_cooldown(_pn)
                        attempts.append(f"{_pn}: {e}")
                        continue
                    if explicit_target and is_target and _is_auth_error(e):
                        raise ProviderError(
                            f"Provider '{target_provider_name}' failed for model "
                            f"'{target_model_name or '<default>'}': {e} "
                             f"Run '/provider {target_provider_name}' or '/model' to switch."
                        ) from e
                    logger.debug(f"Provider '{_pn}' unexpected error: {e}")
                    attempts.append(f"{_pn}: {e}")
                    continue

        if _combo_isolated:
            _usable: Optional[Exception] = None
            if target_error is not None and not _is_unavailable_error(target_error):
                _usable = target_error
            else:
                for _, _e in ordered:
                    if not _is_unavailable_error(_e):
                        _usable = _e
                        break
            _base = _usable if _usable is not None else (
                target_error if target_error is not None else last_error
            )
            if _base is not None and _is_auth_error(_base):
                _hint = f"Run '/provider {target_provider_name}' to re-authenticate, or '/model' to switch."
            elif _base is not None and (_is_connection_error(_base) or _is_timeout_error(_base)):
                _hint = f"Check connection for '{target_provider_name}' or run '/model' to switch provider."
            else:
                _hint = "Run '/model' to switch provider or '/providers' to see options."
            if attempts:
                _chain = " | ".join(attempts)
                if _base is not None:
                    _detail = f"Last error: {_base}. Attempts: {_chain}."
                else:
                    _detail = f"Attempts: {_chain}."
            else:
                _detail = f"Last error: {_base}."
            if "/help" not in _hint:
                _hint = _hint + " Run '/help' for more options."
            raise ProviderError(
                f"Combo '{_combo_active_name}' member '{_combo_member_id}' failed "
                f"(provider '{target_provider_name}' model '{target_model_name or '<default>'}'). "
                f"{_detail} {_hint}"
            ) from _base
        if explicit_target:
            # Prefer the first usable (non-skipped) error so an unavailable
            # failover (e.g. ollama not installed) never masks the original
            # target cause. Skipped entries stay visible in Attempts.
            _usable: Optional[Exception] = None
            if target_error is not None and not _is_unavailable_error(target_error):
                _usable = target_error
            else:
                for _, _e in ordered:
                    if not _is_unavailable_error(_e):
                        _usable = _e
                        break
            _base = _usable if _usable is not None else (
                target_error if target_error is not None else last_error
            )
            if _base is not None and _is_auth_error(_base):
                _hint = f"Run '/provider {target_provider_name}' to re-authenticate, or '/model' to switch."
            elif _base is not None and (_is_connection_error(_base) or _is_timeout_error(_base)):
                _hint = f"Check connection for '{target_provider_name}' or run '/model' to switch provider."
            else:
                _hint = "Run '/model' to switch provider or '/providers' to see options."
            if attempts:
                _chain = " | ".join(attempts)
                if _base is not None:
                    _detail = f"Last error: {_base}. Attempts: {_chain}."
                else:
                    _detail = f"Attempts: {_chain}."
            else:
                _detail = f"Last error: {_base}."
                if last_error is not None and last_error is not _base:
                    _detail += f" (failover last: {last_error})"
            if "/help" not in _hint:
                _hint = _hint + " Run '/help' for more options."
            raise ProviderError(
                f"Provider '{target_provider_name}' failed for model '{target_model_name or '<default>'}'. "
                f"{_detail} {_hint}"
            ) from _base
        if attempts:
            _chain = " | ".join(attempts)
            if last_error is not None:
                raise ProviderError(
                    f"All configured providers failed. Last error: {last_error}. "
                    f"Attempts: {_chain}. "
                    f"Run '/model' to switch provider or '/providers' to see options. "
                    f"Run '/help' for more options."
                ) from last_error
            raise ProviderError(
                f"All configured providers failed. Attempts: {_chain}. "
                f"Run '/model' to switch provider or '/providers' to see options. "
                f"Run '/help' for more options."
            ) from last_error
        raise ProviderError(
            f"All configured providers failed. Last error: {last_error}"
        ) from last_error
