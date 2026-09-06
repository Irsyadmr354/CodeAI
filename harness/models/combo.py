import copy
import json
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.models.base import BaseProvider

_FASTEST_TIMEOUT_S = 30.0
_CONSENSUS_TIMEOUT_S = 60.0


class ComboStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    FASTEST = "fastest"
    CASCADE = "cascade"
    CONSENSUS = "consensus"
    COST_OPTIMIZER = "cost_optimizer"
    WEIGHTED_RANDOM = "weighted_random"
    AB_SPLIT = "ab_split"
    PIPELINE = "pipeline"
    QUALITY_TIER = "quality_tier"
    LOAD_BALANCER = "load_balancer"
    FALLBACK_CHAIN = "fallback_chain"


class ComboManager:
    # Shared round-robin counters (process-wide) so fresh ComboProvider
    # instances per chat still rotate. Guarded by shared lock.
    _shared_rr: Dict[str, int] = {}
    _shared_rr_lock = threading.Lock()

    _NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    _VALID_STRATEGIES = {s.value for s in ComboStrategy}

    def __init__(self):
        self.config_dir = Path.home() / ".codeai"
        self.combo_file = self.config_dir / "combos.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.combos: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        try:
            exists = self.combo_file.exists()
        except OSError:
            self.combos = {}
            return
        if not exists:
            self.combos = {}
            return
        try:
            with open(self.combo_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeError, ValueError):
            # OSError: read/permission; UnicodeError: bad bytes;
            # ValueError: JSON decode (JSONDecodeError subclasses ValueError)
            self.combos = {}
            return
        except Exception:
            self.combos = {}
            return
        if not isinstance(data, dict):
            self.combos = {}
            return
        self.combos = data

    def _save(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.combos, indent=4)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.config_dir), prefix=self.combo_file.name + ".tmp."
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, self.combo_file)
            try:
                os.chmod(self.combo_file, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _reject_nested_combo(models: List[str], combo_name: str = "") -> None:
        for m in models:
            if isinstance(m, str) and m.strip().startswith("combo/"):
                where = f"combo '{combo_name}'" if combo_name else "combo"
                raise ValueError(
                    f"Nested combo not allowed in {where}: model '{m}' "
                    "references another combo (combo/...). "
                    "Use plain provider/model IDs instead."
                )

    @staticmethod
    def _validate_name(name: Any) -> str:
        if not isinstance(name, str):
            raise ValueError(f"Invalid combo name {name!r}: must be a string.")
        cname = name.strip()
        if not cname:
            raise ValueError("Invalid combo name: must be non-empty.")
        if "/" in cname or "\\" in cname or len(cname.split()) != 1:
            raise ValueError(
                f"Invalid combo name {cname!r}: must not contain '/' or whitespace."
            )
        if not ComboManager._NAME_RE.match(cname):
            raise ValueError(
                f"Invalid combo name {cname!r}: use 1-64 chars [A-Za-z0-9_-]."
            )
        return cname

    @staticmethod
    def _validate_strategy(strategy: Any) -> str:
        if isinstance(strategy, ComboStrategy):
            return strategy.value
        if not isinstance(strategy, str):
            raise ValueError(f"Invalid combo strategy {strategy!r}: must be a string.")
        s = strategy.strip()
        if not s:
            raise ValueError("Invalid combo strategy: must be non-empty.")
        # Accept exact value; ComboStrategy is str-Enum so member == value.
        try:
            return ComboStrategy(s).value
        except ValueError:
            pass
        if s not in ComboManager._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid combo strategy {s!r}: must be one of "
                f"{sorted(ComboManager._VALID_STRATEGIES)}."
            )
        return s

    @staticmethod
    def _validate_models(models: Any) -> List[str]:
        if not isinstance(models, (list, tuple)):
            raise ValueError("Invalid combo models: must be a non-empty list.")
        cleaned = list(models)
        if not cleaned:
            raise ValueError("Invalid combo models: must be non-empty.")
        for m in cleaned:
            if not isinstance(m, str) or not m.strip():
                raise ValueError(
                    f"Invalid combo model {m!r}: must be a non-empty string."
                )
        return [m.strip() for m in cleaned]

    def _instance_lock(self) -> Optional[Any]:
        try:
            lk = getattr(self, "_lock", None)
        except Exception:
            return None
        return lk

    def claim_round_robin(self, combo_name: str, num_models: int) -> int:
        """Atomically claim next round-robin slot for combo (shared, locked)."""
        if not isinstance(num_models, int) or num_models <= 0:
            raise ValueError("num_models must be a positive int.")
        key = str(combo_name)
        with ComboManager._shared_rr_lock:
            cur = ComboManager._shared_rr.get(key, 0)
            ComboManager._shared_rr[key] = cur + 1
            return cur % num_models

    def create_combo(self, name: str, strategy: str, models: List[str], params: Optional[Dict[str, Any]] = None, overwrite: bool = False):
        cname = self._validate_name(name)
        sname = self._validate_strategy(strategy)
        cleaned_models = self._validate_models(models)
        if params is None:
            cleaned_params: Dict[str, Any] = {}
        elif not isinstance(params, dict):
            raise ValueError("Invalid combo params: must be a dict.")
        else:
            try:
                cleaned_params = copy.deepcopy(params)
            except Exception:
                cleaned_params = dict(params)
        self._reject_nested_combo(cleaned_models, cname)
        lk = self._instance_lock()
        if lk is not None:
            with lk:
                if cname in self.combos and not overwrite:
                    raise ValueError(
                        f"Combo '{cname}' already exists "
                        "(use overwrite=True to replace)."
                    )
                self.combos[cname] = {
                    "strategy": sname,
                    "models": list(cleaned_models),
                    "params": cleaned_params
                }
        else:
            if cname in self.combos and not overwrite:
                raise ValueError(
                    f"Combo '{cname}' already exists "
                    "(use overwrite=True to replace)."
                )
            self.combos[cname] = {
                "strategy": sname,
                "models": list(cleaned_models),
                "params": cleaned_params
            }
        self._save()

    def get_combo(self, name: str) -> Optional[Dict[str, Any]]:
        lk = self._instance_lock()
        if lk is not None:
            with lk:
                val = self.combos.get(name)
                return copy.deepcopy(val) if val is not None else None
        val = self.combos.get(name)
        return copy.deepcopy(val) if val is not None else None

    def list_combos(self) -> Dict[str, Dict[str, Any]]:
        lk = self._instance_lock()
        if lk is not None:
            with lk:
                return copy.deepcopy(self.combos)
        return copy.deepcopy(self.combos)

    def delete_combo(self, name: str) -> bool:
        lk = self._instance_lock()
        if lk is not None:
            with lk:
                if name in self.combos:
                    del self.combos[name]
                else:
                    return False
        else:
            if name in self.combos:
                del self.combos[name]
            else:
                return False
        try:
            self._save()
        except Exception:
            raise
        return True


class ComboProvider(BaseProvider):
    def __init__(self, combo_name: str, combo_def: Dict[str, Any], gateway_config: Optional[Any] = None, combo_manager: Optional[Any] = None):
        self.combo_name = combo_name
        self.strategy = combo_def["strategy"]
        self.models = combo_def["models"]
        self.params = combo_def.get("params", {})
        ComboManager._reject_nested_combo(self.models, self.combo_name)
        self._round_robin_index = 0
        self._lock = threading.Lock()
        self._load_times: Dict[str, List[float]] = {m: [] for m in self.models}
        # Optional shared manager for round-robin (preserves API: new param optional).
        self._combo_manager = combo_manager
        self._manager = combo_manager
        # Store config so _get_gateway can instantiate a valid LLMGateway
        self._gateway_config = gateway_config

    def _next_round_robin_model(self) -> str:
        """Shared round-robin slot via ComboManager (survives fresh providers)."""
        n = len(self.models)
        idx: Optional[int] = None
        try:
            mgr = getattr(self, "_combo_manager", None)
            if mgr is None:
                try:
                    mgr = getattr(self, "_manager", None)
                except Exception:
                    mgr = None
            if mgr is not None:
                try:
                    claim = getattr(mgr, "claim_round_robin", None)
                    if callable(claim):
                        idx = claim(self.combo_name, n)
                except Exception:
                    idx = None
        except Exception:
            idx = None
        if idx is None:
            try:
                with ComboManager._shared_rr_lock:
                    cur = ComboManager._shared_rr.get(self.combo_name, 0)
                    ComboManager._shared_rr[self.combo_name] = cur + 1
                    idx = cur % n
            except Exception:
                idx = None
        if idx is None:
            try:
                with self._lock:
                    idx = self._round_robin_index % n
                    self._round_robin_index += 1
            except Exception:
                idx = getattr(self, "_round_robin_index", 0) % n
                try:
                    self._round_robin_index = idx + 1
                except Exception:
                    pass
        try:
            return self.models[int(idx) % n]
        except Exception:
            return self.models[0]

    def _get_gateway(self, model_id: Optional[str] = None):
        """Create a fresh LLMGateway using stored config, or build a minimal default.

        Member isolation: the returned gateway is built from a copy of the
        stored config whose failover is emptied (failover_order=[] and
        fallbackChains={}). No-op when the config structure is unsupported
        (returns legacy behaviour instead). Never crashes here.
        """
        from harness.models.gateway import LLMGateway
        try:
            isolated = self._isolated_config()
            if isolated is not None:
                try:
                    return LLMGateway(isolated)
                except Exception:
                    pass
        except Exception:
            pass
        if self._gateway_config is not None:
            try:
                return LLMGateway(self._gateway_config)
            except Exception:
                pass
        # Fallback: build a minimal config so the gateway doesn't crash
        from harness.config import ProviderConfig
        default_config = ProviderConfig()
        return LLMGateway(default_config)

    def _isolated_config(self) -> Optional[Any]:
        """Copy stored config with empty failover (isolation).

        Returns None when the structure is unsupported so the caller can
        fall back to legacy behaviour. Never raises.
        """
        base = self._gateway_config
        try:
            if base is None:
                try:
                    from harness.config import ProviderConfig
                    return ProviderConfig(failover_order=[])
                except Exception:
                    return None
            if hasattr(base, "model_copy"):
                try:
                    iso = base.model_copy(deep=True)
                except Exception:
                    try:
                        iso = copy.deepcopy(base)
                    except Exception:
                        return None
                try:
                    if hasattr(iso, "failover_order"):
                        iso.failover_order = []  # type: ignore[attr-defined]
                except Exception:
                    pass
                for _k in ("fallbackChains", "fallback_chains"):
                    try:
                        if hasattr(iso, _k):
                            setattr(iso, _k, {})
                    except Exception:
                        pass
                try:
                    _extra = getattr(iso, "model_extra", None)
                    if isinstance(_extra, dict):
                        for _k in ("fallbackChains", "fallback_chains"):
                            if _k in _extra:
                                try:
                                    _extra[_k] = {}
                                except Exception:
                                    pass
                except Exception:
                    pass
                try:
                    _prov = getattr(iso, "provider", None)
                    if _prov is not None and hasattr(_prov, "failover_order"):
                        try:
                            _prov.failover_order = []  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    try:
                        _pextra = getattr(_prov, "model_extra", None)
                        if isinstance(_pextra, dict):
                            for _k in ("fallbackChains", "fallback_chains"):
                                if _k in _pextra:
                                    try:
                                        _pextra[_k] = {}
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                except Exception:
                    pass
                return iso
            if isinstance(base, dict):
                try:
                    iso_d = copy.deepcopy(base)
                except Exception:
                    return None
                try:
                    if "failover_order" in iso_d:
                        iso_d["failover_order"] = []
                    for _k in ("fallbackChains", "fallback_chains"):
                        if _k in iso_d:
                            iso_d[_k] = {}
                    _p = iso_d.get("provider")
                    if isinstance(_p, dict):
                        if "failover_order" in _p:
                            _p["failover_order"] = []
                        for _k in ("fallbackChains", "fallback_chains"):
                            if _k in _p:
                                _p[_k] = {}
                except Exception:
                    pass
                return iso_d
            try:
                iso_o = copy.copy(base)
            except Exception:
                return None
            try:
                if hasattr(iso_o, "failover_order"):
                    try:
                        setattr(iso_o, "failover_order", [])
                    except Exception:
                        pass
                for _k in ("fallbackChains", "fallback_chains"):
                    try:
                        if hasattr(iso_o, _k):
                            setattr(iso_o, _k, {})
                    except Exception:
                        pass
                try:
                    _prov2 = getattr(iso_o, "provider", None)
                    if _prov2 is not None and hasattr(_prov2, "failover_order"):
                        try:
                            setattr(_prov2, "failover_order", [])
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                return None
            return iso_o
        except Exception:
            return None

    def _default_provider_name(self) -> str:
        try:
            cfg = self._gateway_config
            if cfg is not None:
                _d = getattr(cfg, "default", None)
                if isinstance(_d, str) and _d.strip():
                    return _d.strip()
                _prov = getattr(cfg, "provider", None)
                if _prov is not None:
                    _d2 = getattr(_prov, "default", None)
                    if isinstance(_d2, str) and _d2.strip():
                        return _d2.strip()
                if isinstance(cfg, dict):
                    _d3 = cfg.get("default")
                    if isinstance(_d3, str) and _d3.strip():
                        return _d3.strip()
                    _p = cfg.get("provider")
                    if isinstance(_p, dict):
                        _d4 = _p.get("default")
                        if isinstance(_d4, str) and _d4.strip():
                            return _d4.strip()
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _split_member_id(model_id: Any, default_provider: str = "unknown") -> Any:
        try:
            _s = str(model_id or "").strip()
        except Exception:
            _s = ""
        _dp = default_provider if isinstance(default_provider, str) and default_provider.strip() else "unknown"
        _dp = _dp.strip()
        if not _s:
            return (_dp, "")
        if "/" in _s:
            _a, _b = _s.split("/", 1)
            _a = _a.strip()
            _b = _b.strip()
            if _a and _b:
                return (_a, _b)
            _bare = _b or _a or _s
            _prov = _a if _a else _dp
            return (_prov if _prov else _dp, _bare)
        return (_dp, _s)

    def _with_serving(self, result: Any, model_id: str) -> Any:
        try:
            if not isinstance(result, dict):
                return result
            try:
                _defprov = self._default_provider_name()
            except Exception:
                _defprov = "unknown"
            try:
                _sp, _sm = self._split_member_id(model_id, _defprov)
            except Exception:
                return result
            if not _sp or not _sm:
                return result
            try:
                tagged = dict(result)
            except Exception:
                return result
            try:
                tagged["serving_provider"] = _sp
                tagged["serving_model"] = _sm
            except Exception:
                pass
            return tagged
        except Exception:
            try:
                return result
            except Exception:
                return result

    def _call_model(
        self,
        gateway: Any,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        res = gateway.chat(messages, model=model_id, tools=tools)
        try:
            return self._with_serving(res, model_id)
        except Exception:
            return res

    def _call_member(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Fresh isolated gateway per member, then tagged call (never crash on gateway build)."""
        try:
            _gw = self._get_gateway(model_id)
        except Exception:
            _gw = None
        if _gw is None:
            try:
                _gw = self._get_gateway()
            except Exception:
                _gw = None
        if _gw is None:
            raise RuntimeError("Unable to build gateway for combo member.")
        return self._call_model(_gw, model_id, messages, tools)

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.models:
            raise ValueError("No models defined in combo.")
        if tools:
            raise ValueError(
                f"Combo '{self.combo_name}' does not support tools "
                f"({len(tools)} tool(s) passed). "
                "Call a tool-capable provider directly instead."
            )

        if self.strategy == ComboStrategy.ROUND_ROBIN:
            model = self._next_round_robin_model()
            return self._call_member(model, messages, tools)

        elif self.strategy == ComboStrategy.RANDOM:
            model = random.choice(self.models)
            return self._call_member(model, messages, tools)

        elif self.strategy == ComboStrategy.FASTEST:
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = {
                    executor.submit(self._call_member, m, messages, tools): m
                    for m in self.models
                }
                try:
                    for future in as_completed(futures, timeout=_FASTEST_TIMEOUT_S):
                        try:
                            result = future.result(timeout=_FASTEST_TIMEOUT_S)
                        except (FuturesTimeoutError, TimeoutError):
                            continue
                        except Exception:
                            continue
                        else:
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            try:
                                _win = futures.get(future)
                                if isinstance(result, dict) and _win:
                                    result = self._with_serving(result, _win)
                            except Exception:
                                pass
                            return result
                except (FuturesTimeoutError, TimeoutError):
                    pass
            raise RuntimeError("All models failed in fastest strategy.")

        elif self.strategy in (ComboStrategy.CASCADE, ComboStrategy.FALLBACK_CHAIN):
            last_error = None
            for model in self.models:
                try:
                    return self._call_member(model, messages, tools)
                except Exception as e:
                    last_error = e
                    continue
            raise RuntimeError(f"All models failed in cascade strategy. Last error: {last_error}")

        elif self.strategy == ComboStrategy.CONSENSUS:
            responses: List[Any] = []
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures_map = {
                    executor.submit(self._call_member, m, messages, tools): m
                    for m in self.models
                }
                try:
                    for future in as_completed(futures_map, timeout=_CONSENSUS_TIMEOUT_S):
                        try:
                            _res = future.result(timeout=_CONSENSUS_TIMEOUT_S)
                        except (FuturesTimeoutError, TimeoutError):
                            pass
                        except Exception:
                            pass
                        else:
                            try:
                                _m = futures_map.get(future, "")
                            except Exception:
                                _m = ""
                            responses.append((_m, _res))
                except (FuturesTimeoutError, TimeoutError):
                    pass
            if not responses:
                raise RuntimeError("All models failed in consensus strategy.")
            # Heuristic: return the longest response as best candidate
            def _clen(t: Any) -> int:
                try:
                    _r = t[1]
                    if isinstance(_r, dict):
                        return len(_r.get("content") or "")
                except Exception:
                    pass
                return 0
            _best_m, _best_r = max(responses, key=_clen)
            try:
                return self._with_serving(_best_r, _best_m)
            except Exception:
                return _best_r

        elif self.strategy == ComboStrategy.COST_OPTIMIZER:
            # Models ordered cheapest-first by convention
            for model in self.models:
                try:
                    return self._call_member(model, messages, tools)
                except Exception:
                    continue
            raise RuntimeError("All models failed in cost_optimizer strategy.")

        elif self.strategy == ComboStrategy.WEIGHTED_RANDOM:
            weights = self.params.get("weights", [1] * len(self.models))
            model = random.choices(self.models, weights=weights, k=1)[0]
            return self._call_member(model, messages, tools)

        elif self.strategy == ComboStrategy.AB_SPLIT:
            model = self.models[0] if random.random() < 0.5 else self.models[1 % len(self.models)]
            return self._call_member(model, messages, tools)

        elif self.strategy == ComboStrategy.PIPELINE:
            if len(self.models) < 2:
                raise ValueError("Pipeline strategy needs at least 2 models.")
            draft = self._call_member(self.models[0], messages, tools)
            refine_messages = list(messages) + [
                {"role": "assistant", "content": draft.get("content", "") if isinstance(draft, dict) else ""},
                {"role": "user", "content": "Refine and improve this response."},
            ]
            return self._call_member(self.models[1], refine_messages, tools)

        elif self.strategy == ComboStrategy.QUALITY_TIER:
            try:
                return self._call_member(self.models[0], messages, tools)
            except Exception:
                if len(self.models) > 1:
                    return self._call_member(self.models[1], messages, tools)
                raise

        elif self.strategy == ComboStrategy.LOAD_BALANCER:
            with self._lock:
                best_model = self.models[0]
                best_avg = float("inf")
                for m in self.models:
                    times = self._load_times.get(m, [])
                    avg = sum(times) / len(times) if times else 0.0
                    if avg < best_avg:
                        best_avg = avg
                        best_model = m

            start = time.time()
            res = self._call_member(best_model, messages, tools)
            duration = time.time() - start
            with self._lock:
                self._load_times.setdefault(best_model, []).append(duration)
                # Keep only last 10 measurements
                if len(self._load_times[best_model]) > 10:
                    self._load_times[best_model].pop(0)
            try:
                return self._with_serving(res, best_model)
            except Exception:
                return res

        else:
            raise ValueError(f"Unknown combo strategy: {self.strategy}")

    def supports_tools(self) -> bool:
        return False
