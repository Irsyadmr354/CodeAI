import builtins
import json
import logging
import os
import re
import socket
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from harness.models.auth_vault import AuthVault
from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError, ToolCall

logger = logging.getLogger(__name__)

_EFFORT_TO_LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key:
            if api_key.startswith("AIza"):
                self.api_key = api_key
                self.access_token = None
            else:
                self.api_key = None
                self.access_token = api_key
        else:
            self.api_key = os.environ.get("GEMINI_API_KEY")
            self.access_token = None
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.vault = AuthVault()
        self.effort: str = (os.environ.get("GEMINI_EFFORT") or "").strip().lower()

    def _thinking_config(self) -> Dict[str, Any]:
        """Map effort → thinkingLevel; omit with debug log if unsupported/empty."""
        effort = (getattr(self, "effort", "") or "").strip().lower()
        if not effort:
            return {}
        if effort not in _EFFORT_TO_LEVEL:
            logger.debug(f"Gemini: unknown effort '{effort}' — omitting thinkingConfig.")
            return {}
        model = (self.model or "").lower()
        if "gemini" not in model and "gemma" not in model:
            logger.debug(f"Gemini model '{self.model}' does not support thinking — omitting.")
            return {}
        return {"thinkingConfig": {"thinkingLevel": _EFFORT_TO_LEVEL[effort]}}

    def _clean_agy_output(self, text: str) -> str:
        # Strip ANSI escape sequences
        text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        
        # Subagent headers, system notices, or status indicators to filter
        subagent_patterns = [
            re.compile(r'^\s*\[!?\].*?\[!?\]\s*$'),
            re.compile(r'^\s*={2,}\s*Subagent.*?\s*={2,}\s*$', re.IGNORECASE),
            re.compile(r'^\s*-{2,}\s*Subagent.*?\s*-{2,}\s*$', re.IGNORECASE),
            re.compile(r'^\s*#+\s*Subagent.*$', re.IGNORECASE),
            re.compile(r'^\s*\[Subagent(?:\s*:\s*|\s+).*?\]\s*$', re.IGNORECASE),
            re.compile(r'^\s*Subagent\s+\[.*?\]\s*:?\s*$', re.IGNORECASE),
            re.compile(r'^\s*Subagent\s+.*?(?:started|finished|running|completed).*?$', re.IGNORECASE),
            re.compile(r'^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏].*$'),
        ]
        
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            if any(p.match(line) for p in subagent_patterns):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        active_api_key = self.api_key
        active_access_token = self.access_token

        if active_api_key and not active_api_key.startswith("AIza"):
            active_access_token = active_api_key
            active_api_key = None

        if not active_api_key and not active_access_token:
            token = self.vault.discover_gemini_token()
            if token:
                if token.startswith("AIza"):
                    active_api_key = token
                else:
                    active_access_token = token

            if not active_api_key and not active_access_token:
                raise ProviderError("GEMINI_API_KEY is not set and no Gemini OAuth token found. Run '/provider gemini'")

        # If active_api_key exists (starts with AIza): Execute via Generative Language REST API
        if active_api_key:
            contents = []
            system_instruction = None

            for msg in messages:
                if msg.role == "system":
                    system_instruction = {
                        "parts": [{"text": msg.content}]
                    }
                else:
                    role = "user" if msg.role == "user" else "model"
                    parts = [{"text": msg.content}]
                    if msg.tool_calls:
                        # Append function call parts
                        for tc in msg.tool_calls:
                            parts.append({
                                "functionCall": {
                                    "name": tc.name,
                                    "args": tc.arguments
                                }
                            })
                    contents.append({
                        "role": role,
                        "parts": parts
                    })
                    
            payload: Dict[str, Any] = {
                "contents": contents,
            }
            thinking = self._thinking_config()
            if thinking:
                payload["generationConfig"] = thinking
            
            if system_instruction:
                payload["systemInstruction"] = system_instruction

            if tools:
                gemini_tools = []
                for t in tools:
                    if "function" in t:
                        gemini_tools.append({
                            "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parameters": t["function"].get("parameters", {})
                        })
                if gemini_tools:
                    payload["tools"] = [{"functionDeclarations": gemini_tools}]

            headers = {"Content-Type": "application/json"}
            url = f"{self.base_url}/models/{self.model}:generateContent?key={active_api_key}"

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    
                    content = ""
                    parsed_tool_calls = []
                    
                    if "candidates" in result and len(result["candidates"]) > 0:
                        parts = result["candidates"][0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                content += part["text"]
                            elif "functionCall" in part:
                                parsed_tool_calls.append(
                                    ToolCall(
                                        name=part["functionCall"]["name"],
                                        arguments=part["functionCall"].get("args", {})
                                    )
                                )
                                
                    return {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": parsed_tool_calls if parsed_tool_calls else None
                    }
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise RateLimitError(f"Gemini Rate Limit: {e.reason}")
                raise ProviderError(f"Gemini HTTP Error {e.code}: {e.reason}")
            except urllib.error.URLError as e:
                if isinstance(e.reason, (socket.timeout, TimeoutError, builtins.TimeoutError)):
                    raise TimeoutError(f"Gemini Timeout: {e.reason}") from e
                if isinstance(e.reason, OSError) and "timed out" in str(e.reason).lower():
                    raise TimeoutError(f"Gemini Timeout: {e.reason}") from e
                raise ProviderError(f"Gemini Request Error: {e.reason}")
            except Exception as e:
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(f"Gemini Unexpected Error: {str(e)}")

        # If active_access_token or Antigravity token exists (OAuth bearer token):
        # Execute using Antigravity's inference engine
        prompt_text = messages[-1].content if messages else ""
        if len(messages) > 1:
            prompt_text = "\n".join([f"{m.role}: {m.content}" for m in messages])

        cmd = ["agy", "--print", prompt_text, "--model", self.model, "--disable-slash-commands"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                err_msg = (proc.stderr or proc.stdout or "").strip()
                if "is not recognized as a known model" in err_msg:
                    raise ProviderError(
                        f"Unknown Gemini model '{self.model}' is not recognized as a known model.\n"
                        "Instructions: Use '/model' to select a valid Gemini model."
                    )

                if proc.returncode != 0:
                    err_msg = (proc.stderr or proc.stdout or "").strip()
                    raise ProviderError(
                        f"Antigravity CLI ('agy') execution failed (exit code {proc.returncode}): {err_msg}\n"
                        "Instructions: Verify Antigravity CLI authentication ('agy') or provide a Google AI Studio API Key "
                        "via '/provider gemini' (Option 1)."
                    )

            raw_output = proc.stdout or ""
            output_text = self._clean_agy_output(raw_output)
            if not output_text and proc.stderr:
                output_text = self._clean_agy_output(proc.stderr)

            return {
                "role": "assistant",
                "content": output_text,
                "tool_calls": None
            }
        except FileNotFoundError:
            raise ProviderError(
                "Antigravity CLI ('agy') was not found in PATH.\n"
                "Instructions: Please install Antigravity CLI ('agy') or provide a Google AI Studio API Key "
                "via '/provider gemini' (Option 1)."
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("Antigravity CLI ('agy') inference timed out after 60s.")
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Antigravity execution error: {e}")

    def supports_tools(self) -> bool:
        return True
