import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from harness.models.auth_vault import AuthVault
from harness.models.base import BaseProvider, ChatMessage, ProviderError, RateLimitError, TimeoutError, ToolCall


class CopilotProvider(BaseProvider):
    def __init__(self) -> None:
        self.model = "gpt-4o"
        self.vault = AuthVault()
        self.session_token = None
        self.session_expires_at = 0

    def _get_oauth_token(self) -> str:
        token = self.vault.discover_copilot_token()
        if not token:
            raise ProviderError("No GitHub Copilot token found. Please run '/provider copilot' to authenticate.")
        return token

    def device_flow_login(self) -> None:
        client_id = "01ab8ac9400c4e429b23" # Standard Copilot client ID
        req = urllib.request.Request(
            "https://github.com/login/device/code",
            data=f"client_id={client_id}&scope=read:user".encode("utf-8"),
            headers={"Accept": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                device_code = data["device_code"]
                user_code = data["user_code"]
                verification_uri = data["verification_uri"]
                print(f"Opening browser to {verification_uri} ...")
                import webbrowser
                try:
                    webbrowser.open(verification_uri)
                except Exception:
                    pass
                print(f"Please enter code on the website: {user_code}")
                
                # Poll for token
                while True:
                    time.sleep(data["interval"])
                    poll_req = urllib.request.Request(
                        "https://github.com/login/oauth/access_token",
                        data=f"client_id={client_id}&device_code={device_code}&grant_type=urn:ietf:params:oauth:grant-type:device_code".encode("utf-8"),
                        headers={"Accept": "application/json"},
                        method="POST"
                    )
                    try:
                        with urllib.request.urlopen(poll_req) as poll_resp:
                            poll_data = json.loads(poll_resp.read().decode("utf-8"))
                            if "access_token" in poll_data:
                                self.vault.store_token("copilot", poll_data["access_token"])
                                print("Successfully authenticated with GitHub Copilot.")
                                return
                            elif poll_data.get("error") != "authorization_pending":
                                raise ProviderError(f"OAuth Error: {poll_data.get('error_description', poll_data.get('error'))}")
                    except urllib.error.HTTPError:
                        pass
        except Exception as e:
            raise ProviderError(f"Device flow login failed: {str(e)}")

    def _get_session_token(self) -> str:
        if self.session_token and time.time() < self.session_expires_at:
            return self.session_token

        oauth_token = self._get_oauth_token()
        req = urllib.request.Request(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {oauth_token}",
                "Accept": "application/json"
            },
            method="GET"
        )
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.session_token = data["token"]
                self.session_expires_at = data["expires_at"] - 300 # Buffer
                return self.session_token
        except Exception as e:
            raise ProviderError(f"Failed to get Copilot session token: {str(e)}")

    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        session_token = self._get_session_token()

        formatted_messages = []
        for msg in messages:
            msg_dict: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                msg_dict["tool_calls"] = []
                for tc in tool_calls:
                    msg_dict["tool_calls"].append({
                        "id": getattr(tc, "id", "call_1"),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                        }
                    })
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id and isinstance(tool_call_id, str):
                msg_dict["tool_call_id"] = tool_call_id
                
            formatted_messages.append(msg_dict)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(
            "https://api.githubcopilot.com/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session_token}",
                "Copilot-Integration-Id": "vscode-chat",
                "Editor-Version": "vscode/1.90.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                choice = result["choices"][0]
                msg_data = choice.get("message", {})
                
                parsed_tool_calls = []
                if "tool_calls" in msg_data:
                    for tc in msg_data["tool_calls"]:
                        if tc.get("type") == "function":
                            args = tc["function"].get("arguments", "{}")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    pass
                            parsed_tool_calls.append(
                                ToolCall(
                                    name=tc["function"]["name"],
                                    arguments=args
                                )
                            )
                
                return {
                    "role": "assistant",
                    "content": msg_data.get("content", ""),
                    "tool_calls": parsed_tool_calls if parsed_tool_calls else None
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimitError(f"Copilot Rate Limit: {e.reason}")
            raise ProviderError(f"Copilot HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise TimeoutError(f"Copilot Timeout: {e.reason}")
            raise ProviderError(f"Copilot Request Error: {e.reason}")
        except Exception as e:
            raise ProviderError(f"Copilot Unexpected Error: {str(e)}")

    def supports_tools(self) -> bool:
        return True
