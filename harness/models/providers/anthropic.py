import builtins
import json
import logging
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError, ToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        self.version = "2023-06-01"
        self.effort: str = (os.environ.get("ANTHROPIC_EFFORT") or "").strip().lower()

    def _thinking_payload(self) -> Dict[str, Any]:
        """Map effort → Anthropic extended-thinking budget; omit if unsupported."""
        effort = (getattr(self, "effort", "") or "").strip().lower()
        if not effort:
            return {}
        try:
            from harness.models.providers.antigravity import EFFORT_BUDGET
        except Exception:
            EFFORT_BUDGET = {"low": 1024, "medium": 4096, "high": 10000}
        if effort not in EFFORT_BUDGET:
            logger.debug(f"Anthropic: unknown effort '{effort}' — omitting thinking.")
            return {}
        model = (self.model or "").lower()
        # Extended thinking requires claude-3-7+ / claude-4+; 3-5 and older omit.
        # (Match on family markers — a bare "4" would false-positive on dates
        #  like "20240620" in "claude-3-5-sonnet-20240620".)
        if not any(
            h in model
            for h in ("sonnet-4", "opus-4", "haiku-4", "claude-4", "3-7", "3.7")
        ):
            logger.debug(
                f"Anthropic model '{self.model}' does not support thinking — omitting."
            )
            return {}
        return {"thinking": {"type": "enabled", "budget_tokens": EFFORT_BUDGET[effort]}}

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY environment variable is not set")

        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message += msg.content + "\n"
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})
                
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        thinking = self._thinking_payload()
        if thinking:
            payload.update(thinking)
            budget = thinking["thinking"]["budget_tokens"]
            if payload["max_tokens"] <= budget:
                payload["max_tokens"] = budget + 4096
        
        if system_message:
            payload["system"] = system_message.strip()

        if tools:
            # Note: The tool conversion from standard JSON Schema to Anthropic's specific format
            # might require mapping, assuming the provided tools follow basic JSON Schema structure.
            # Assuming OpenAI-like tool definitions are passed in, converting to Anthropic tools.
            anthropic_tools = []
            for t in tools:
                if "function" in t:
                    anthropic_tools.append({
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})
                    })
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }

        req = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                content = ""
                parsed_tool_calls = []
                
                for block in result.get("content", []):
                    if block["type"] == "text":
                        content += block["text"]
                    elif block["type"] == "tool_use":
                        parsed_tool_calls.append(
                            ToolCall(
                                name=block["name"],
                                arguments=block["input"]
                            )
                        )
                            
                return {
                    "role": result.get("role", "assistant"),
                    "content": content,
                    "tool_calls": parsed_tool_calls if parsed_tool_calls else None
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimitError(f"Anthropic Rate Limit: {e.reason}")
            raise ProviderError(f"Anthropic HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError, builtins.TimeoutError)):
                raise TimeoutError(f"Anthropic Timeout: {e.reason}") from e
            if isinstance(e.reason, OSError) and "timed out" in str(e.reason).lower():
                raise TimeoutError(f"Anthropic Timeout: {e.reason}") from e
            raise ProviderError(f"Anthropic Request Error: {e.reason}")
        except Exception as e:
            raise ProviderError(f"Anthropic Unexpected Error: {str(e)}")

    def supports_tools(self) -> bool:
        return True
