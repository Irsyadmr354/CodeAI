import json
import logging
import os
import warnings
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Exact sensitive keys triggering chmod 600 (exact key match, case-insensitive).
# NOTE: intentionally exact — e.g. "token_budget" must NOT trigger.
_SENSITIVE_KEYS = frozenset({"token", "api_key", "apikey", "access", "access_token"})


def _contains_sensitive_key(obj: Any) -> bool:
    """Exact key match (case-insensitive) for sensitive data."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                return True
            if _contains_sensitive_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if _contains_sensitive_key(item):
                return True
    return False

DEFAULT_OPENCODE_MODEL = "mimo-v2.5-free"

EFFORT_LEVELS = ("low", "medium", "high")

class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    default: str = Field(default="anthropic", description="Default LLM provider")
    active_model: Optional[str] = None
    effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level: low, medium, high",
    )
    opencode_model: str = Field(
        default=DEFAULT_OPENCODE_MODEL,
        description="Default model for OpenCode provider"
    )
    failover_order: List[str] = Field(
        default_factory=lambda: ["openai", "gemini", "ollama"],
        description="List of fallback providers",
    )

    @field_validator("effort", mode="before")
    @classmethod
    def _normalize_effort(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("effort must be one of low, medium, high")
        s = v.strip().lower()
        if not s:
            return None
        if s not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {list(EFFORT_LEVELS)}")
        return s


class CompactionConfig(BaseModel):
    trigger_limit: int = Field(default=50, description="Message count to trigger compaction")
    token_budget: int = Field(default=8000, description="Token budget for context window")


class AllowanceConfig(BaseModel):
    mode: str = Field(default="ask", description="Execution mode: auto, ask, or block")
    whitelisted_commands: List[str] = Field(
        default_factory=lambda: ["ls", "pwd", "cat", "echo", "git status", "git diff", "grep"]
    )
    blocked_patterns: List[str] = Field(
        default_factory=lambda: ["rm -rf /", "sudo", "chmod -R 777", "chown", "mkfs", "dd if="]
    )


class CodeAIConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    allowance: AllowanceConfig = Field(default_factory=AllowanceConfig)
    custom_providers: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined OpenAI-compatible providers {id: {baseURL, ...}}",
    )
    fallbackChains: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-role fallback chains, e.g. {default: [...]}",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "fallback_chains" in data:
                if "fallbackChains" in data:
                    warnings.warn(
                        "Both 'fallbackChains' and alias 'fallback_chains' present; "
                        "using explicit 'fallbackChains'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    del data["fallback_chains"]
                else:
                    data["fallbackChains"] = data.pop("fallback_chains")
            if "customProviders" in data:
                if "custom_providers" in data:
                    warnings.warn(
                        "Both 'custom_providers' and alias 'customProviders' present; "
                        "using explicit 'custom_providers'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    del data["customProviders"]
                else:
                    data["custom_providers"] = data.pop("customProviders")
        return data

    @classmethod
    def load_from_file(cls, path: str | Path) -> "CodeAIConfig":
        """Loads configuration from a JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            logger.info("Config file not found at %s; using defaults.", config_path)
            return cls()
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            known = set(cls.model_fields.keys()) | {"fallback_chains", "customProviders"}
            unknown = set(data.keys()) - known
            if unknown:
                warnings.warn(
                    f"Unknown config keys {sorted(unknown)} in {config_path}; kept (extra='allow').",
                    UserWarning,
                    stacklevel=2,
                )
        return cls(**data)

    def save_to_file(self, path: str | Path) -> Path:
        """Atomically save configuration as pretty JSON (tmp+rename).

        chmod 600 is applied when the payload contains an exact
        sensitive key (token/api_key/access, case-insensitive). Old codeai.json files without the new fields keep
        loading via pydantic defaults (backward-compat).
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data_dict = self.model_dump()
        content = json.dumps(data_dict, indent=2, ensure_ascii=False) + "\n"
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, target)
        if _contains_sensitive_key(data_dict):
            try:
                os.chmod(target, 0o600)
            except Exception:
                pass
        return target
