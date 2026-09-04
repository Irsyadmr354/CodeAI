import builtins
import json
import logging
import os
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error
import socket

from harness.models.base import BaseProvider, ChatMessage, ToolCall, ProviderError, RateLimitError, TimeoutError

logger = logging.getLogger(__name__)

_ALLOWED_EFFORTS = ("low", "medium", "high")

DEFAULT_OPENCODE_MODEL = "mimo-v2.5-free"

class UniversalOpenAIProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str, default_model: str = "", custom_headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            # Ensure it ends with /v1 if missing and it's a standard openai compatible endpoint
            if not self.base_url.endswith("/chat/completions"):
                pass # Many URLs provide full path, adjust if needed
        self.api_key = api_key
        if not default_model or default_model == "muse-spark-1.3-contributor-free":
            if "opencode" in self.base_url or "zen" in self.base_url:
                default_model = DEFAULT_OPENCODE_MODEL
        self.default_model = default_model
        self.custom_headers = custom_headers or {}
        self.effort: str = (os.environ.get("UNIVERSAL_EFFORT") or "").strip().lower()

    def _effort_extra(self) -> Dict[str, Any]:
        effort = (getattr(self, "effort", "") or "").strip().lower()
        if not effort or effort not in _ALLOWED_EFFORTS:
            return {}
        # OpenAI-compatible passthrough; server may ignore unknown fields.
        return {"reasoning_effort": effort}

    def supports_tools(self) -> bool:
        return True

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        
        endpoint = f"{self.base_url}/chat/completions"
        if "chat/completions" in self.base_url:
            endpoint = self.base_url
            
        formatted_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    } for tc in msg.tool_calls
                ]
            formatted_messages.append(m)

        model_name = self.default_model
        if not model_name or model_name == "muse-spark-1.3-contributor-free":
            if "opencode" in self.base_url or "zen" in self.base_url:
                model_name = DEFAULT_OPENCODE_MODEL

        payload = {
            "model": model_name,
            "messages": formatted_messages,
        }
        _effort_applied = False
        _effort_body = self._effort_extra()
        if _effort_body:
            payload.update(_effort_body)
            _effort_applied = True
        
        if tools:
            payload["tools"] = tools
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "opencode/1.0 (Linux; x64) CodeAI/1.0"
        }
        headers.update(self.custom_headers)
        
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                    
                    choice = response_data["choices"][0]
                    message = choice["message"]
                    
                    content = message.get("content") or ""
                    tool_calls = []
                    
                    if message.get("tool_calls"):
                        for tc in message["tool_calls"]:
                            if tc["type"] == "function":
                                fn = tc["function"]
                                tool_calls.append(
                                    ToolCall(
                                        name=fn["name"],
                                        arguments=json.loads(fn["arguments"])
                                    )
                                )
                    
                    return {
                        "role": message.get("role", "assistant"),
                        "content": content,
                        "tool_calls": tool_calls
                    }
                    
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='replace')
                if _effort_applied and e.code == 400 and "reasoning_effort" in err_body.lower():
                    logger.warning(
                        "Provider does not support 'reasoning_effort' — "
                        "retrying without effort passthrough."
                    )
                    payload.pop("reasoning_effort", None)
                    _effort_applied = False
                    req = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST"
                    )
                    if attempt < max_retries - 1:
                        continue
                    # Last attempt: one final try without effort, then raise below.
                    try:
                        with urllib.request.urlopen(req, timeout=30) as response:
                            response_data = json.loads(response.read().decode("utf-8"))
                            choice = response_data["choices"][0]
                            message = choice["message"]
                            return {
                                "role": message.get("role", "assistant"),
                                "content": message.get("content") or "",
                                "tool_calls": None,
                            }
                    except Exception:
                        pass
                    raise ProviderError(f"HTTP error {e.code}: {err_body}")
                # Retryable status codes: 429, 500, 502, 503, 504
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    import random
                    import time
                    retry_after = e.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(float(retry_after), 10.0)
                    else:
                        delay = min(1.5 * (2 ** attempt) + random.uniform(0.1, 0.5), 8.0)
                    logger.debug(
                        f"Transient HTTP {e.code} error from provider. Auto-retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue

                if e.code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {err_body}")
                raise ProviderError(f"HTTP error {e.code}: {err_body}")
            except urllib.error.URLError as e:
                if isinstance(e.reason, (socket.timeout, TimeoutError, builtins.TimeoutError)) and attempt < max_retries - 1:
                    import time
                    delay = 1.5 * (2 ** attempt)
                    logger.debug(f"Connection timeout. Auto-retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                if isinstance(e.reason, (socket.timeout, TimeoutError, builtins.TimeoutError)):
                    raise TimeoutError("Request timed out") from e
                if isinstance(e.reason, OSError) and "timed out" in str(e.reason).lower():
                    raise TimeoutError("Request timed out") from e
                raise ProviderError(f"URL error: {e.reason}")
            except Exception as e:
                raise ProviderError(f"Unexpected error: {str(e)}")
        raise ProviderError("Unexpected error: retries exhausted")
