import json
import builtins
import logging
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError, ToolCall

logger = logging.getLogger(__name__)

_REASONING_EFFORTS = ("low", "medium", "high")
_REASONING_MODEL_HINTS = ("o1", "o3", "o4", "gpt-5", "reasoning")


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.effort: str = (os.environ.get("OPENAI_EFFORT") or "").strip().lower()

    def _effort_payload(self) -> Dict[str, Any]:
        effort = (getattr(self, "effort", "") or "").strip().lower()
        if not effort or effort not in _REASONING_EFFORTS:
            return {}
        model = (self.model or "").lower()
        if not any(h in model for h in _REASONING_MODEL_HINTS):
            logger.debug(
                f"OpenAI model '{self.model}' does not support reasoning_effort — omitting."
            )
            return {}
        return {"reasoning_effort": effort}

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY environment variable is not set")

        formatted_messages = []
        for msg in messages:
            formatted_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        payload.update(self._effort_payload())
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                choice = result["choices"][0]["message"]
                content = choice.get("content", "")
                
                parsed_tool_calls = []
                if "tool_calls" in choice:
                    for tc in choice["tool_calls"]:
                        if tc["type"] == "function":
                            parsed_tool_calls.append(
                                ToolCall(
                                    name=tc["function"]["name"],
                                    arguments=json.loads(tc["function"]["arguments"])
                                )
                            )
                            
                return {
                    "role": choice.get("role", "assistant"),
                    "content": content,
                    "tool_calls": parsed_tool_calls if parsed_tool_calls else None
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimitError(f"OpenAI Rate Limit: {e.reason}")
            raise ProviderError(f"OpenAI HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError, builtins.TimeoutError)):
                raise TimeoutError(f"OpenAI Timeout: {e.reason}") from e
            if isinstance(e.reason, OSError) and "timed out" in str(e.reason).lower():
                raise TimeoutError(f"OpenAI Timeout: {e.reason}") from e
            raise ProviderError(f"OpenAI Request Error: {e.reason}")
        except Exception as e:
            raise ProviderError(f"OpenAI Unexpected Error: {str(e)}")

    def supports_tools(self) -> bool:
        return True
