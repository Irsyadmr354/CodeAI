import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError, ToolCall


class OllamaProvider(BaseProvider):
    def __init__(self) -> None:
        self.model = os.environ.get("OLLAMA_MODEL", "llama3")
        # Ollama provides an OpenAI compatible API at /v1
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
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
        
        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                choice = result["choices"][0]["message"]
                content = choice.get("content", "")
                
                parsed_tool_calls = []
                if "tool_calls" in choice and choice["tool_calls"]:
                    for tc in choice["tool_calls"]:
                        if tc["type"] == "function":
                            try:
                                args = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                args = {}
                            parsed_tool_calls.append(
                                ToolCall(
                                    name=tc["function"]["name"],
                                    arguments=args
                                )
                            )
                            
                return {
                    "role": choice.get("role", "assistant"),
                    "content": content,
                    "tool_calls": parsed_tool_calls if parsed_tool_calls else None
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimitError(f"Ollama Rate Limit: {e.reason}")
            raise ProviderError(f"Ollama HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise TimeoutError(f"Ollama Timeout: {e.reason}")
            raise ProviderError(f"Ollama Request Error: {e.reason}")
        except Exception as e:
            raise ProviderError(f"Ollama Unexpected Error: {str(e)}")

    def supports_tools(self) -> bool:
        return True
