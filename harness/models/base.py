from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Exception for rate limits (e.g. HTTP 429)."""
    pass


class TimeoutError(ProviderError):
    """Exception for request timeouts."""
    pass


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class BaseProvider(ABC):
    """Abstract BaseProvider class for LLM gateways."""

    @abstractmethod
    def chat(
        self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat request to the provider.
        Must return a dict in a unified format, e.g.,
        {
            "role": "assistant",
            "content": "...",
            "tool_calls": [ToolCall(...), ...]
        }
        """
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """
        Returns True if the provider supports tool/function calling.
        """
        pass
