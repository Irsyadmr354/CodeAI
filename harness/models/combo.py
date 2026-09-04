import json
import os
import random
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
    def __init__(self):
        self.config_dir = Path.home() / ".codeai"
        self.combo_file = self.config_dir / "combos.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.combos: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.combo_file.exists():
            try:
                with open(self.combo_file, "r") as f:
                    self.combos = json.load(f)
            except json.JSONDecodeError:
                self.combos = {}

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

    def create_combo(self, name: str, strategy: str, models: List[str], params: Optional[Dict[str, Any]] = None):
        self._reject_nested_combo(models, name)
        self.combos[name] = {
            "strategy": strategy,
            "models": models,
            "params": params or {}
        }
        self._save()

    def get_combo(self, name: str) -> Optional[Dict[str, Any]]:
        return self.combos.get(name)

    def list_combos(self) -> Dict[str, Dict[str, Any]]:
        return self.combos

    def delete_combo(self, name: str):
        if name in self.combos:
            del self.combos[name]
            self._save()


class ComboProvider(BaseProvider):
    def __init__(self, combo_name: str, combo_def: Dict[str, Any], gateway_config: Optional[Any] = None):
        self.combo_name = combo_name
        self.strategy = combo_def["strategy"]
        self.models = combo_def["models"]
        self.params = combo_def.get("params", {})
        ComboManager._reject_nested_combo(self.models, self.combo_name)
        self._round_robin_index = 0
        self._lock = threading.Lock()
        self._load_times: Dict[str, List[float]] = {m: [] for m in self.models}
        # Store config so _get_gateway can instantiate a valid LLMGateway
        self._gateway_config = gateway_config

    def _get_gateway(self):
        """Create a fresh LLMGateway using stored config, or build a minimal default."""
        from harness.models.gateway import LLMGateway
        if self._gateway_config is not None:
            return LLMGateway(self._gateway_config)
        # Fallback: build a minimal config so the gateway doesn't crash
        from harness.config import ProviderConfig
        default_config = ProviderConfig()
        return LLMGateway(default_config)

    def _call_model(
        self,
        gateway: Any,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gateway.chat(messages, model=model_id, tools=tools)

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

        gateway = self._get_gateway()

        if self.strategy == ComboStrategy.ROUND_ROBIN:
            with self._lock:
                model = self.models[self._round_robin_index % len(self.models)]
                self._round_robin_index += 1
            return self._call_model(gateway, model, messages, tools)

        elif self.strategy == ComboStrategy.RANDOM:
            model = random.choice(self.models)
            return self._call_model(gateway, model, messages, tools)

        elif self.strategy == ComboStrategy.FASTEST:
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = {
                    executor.submit(self._call_model, gateway, m, messages, tools): m
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
                            return result
                except (FuturesTimeoutError, TimeoutError):
                    pass
            raise RuntimeError("All models failed in fastest strategy.")

        elif self.strategy in (ComboStrategy.CASCADE, ComboStrategy.FALLBACK_CHAIN):
            last_error = None
            for model in self.models:
                try:
                    return self._call_model(gateway, model, messages, tools)
                except Exception as e:
                    last_error = e
                    continue
            raise RuntimeError(f"All models failed in cascade strategy. Last error: {last_error}")

        elif self.strategy == ComboStrategy.CONSENSUS:
            responses = []
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = [
                    executor.submit(self._call_model, gateway, m, messages, tools)
                    for m in self.models
                ]
                try:
                    for future in as_completed(futures, timeout=_CONSENSUS_TIMEOUT_S):
                        try:
                            responses.append(future.result(timeout=_CONSENSUS_TIMEOUT_S))
                        except (FuturesTimeoutError, TimeoutError):
                            pass
                        except Exception:
                            pass
                except (FuturesTimeoutError, TimeoutError):
                    pass
            if not responses:
                raise RuntimeError("All models failed in consensus strategy.")
            # Heuristic: return the longest response as best candidate
            return max(responses, key=lambda r: len(r.get("content") or ""))

        elif self.strategy == ComboStrategy.COST_OPTIMIZER:
            # Models ordered cheapest-first by convention
            for model in self.models:
                try:
                    return self._call_model(gateway, model, messages, tools)
                except Exception:
                    continue
            raise RuntimeError("All models failed in cost_optimizer strategy.")

        elif self.strategy == ComboStrategy.WEIGHTED_RANDOM:
            weights = self.params.get("weights", [1] * len(self.models))
            model = random.choices(self.models, weights=weights, k=1)[0]
            return self._call_model(gateway, model, messages, tools)

        elif self.strategy == ComboStrategy.AB_SPLIT:
            model = self.models[0] if random.random() < 0.5 else self.models[1 % len(self.models)]
            return self._call_model(gateway, model, messages, tools)

        elif self.strategy == ComboStrategy.PIPELINE:
            if len(self.models) < 2:
                raise ValueError("Pipeline strategy needs at least 2 models.")
            draft = self._call_model(gateway, self.models[0], messages, tools)
            refine_messages = list(messages) + [
                {"role": "assistant", "content": draft.get("content", "")},
                {"role": "user", "content": "Refine and improve this response."},
            ]
            return self._call_model(gateway, self.models[1], refine_messages, tools)

        elif self.strategy == ComboStrategy.QUALITY_TIER:
            try:
                return self._call_model(gateway, self.models[0], messages, tools)
            except Exception:
                if len(self.models) > 1:
                    return self._call_model(gateway, self.models[1], messages, tools)
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
            res = self._call_model(gateway, best_model, messages, tools)
            duration = time.time() - start
            with self._lock:
                self._load_times.setdefault(best_model, []).append(duration)
                # Keep only last 10 measurements
                if len(self._load_times[best_model]) > 10:
                    self._load_times[best_model].pop(0)
            return res

        else:
            raise ValueError(f"Unknown combo strategy: {self.strategy}")

    def supports_tools(self) -> bool:
        return False
