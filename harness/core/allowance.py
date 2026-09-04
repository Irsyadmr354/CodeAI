import re
import shlex
from enum import Enum
from typing import List

from harness.config import AllowanceConfig


class CommandStatus(Enum):
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    ASK = "ASK"


class AllowanceGuard:
    """Security guard for evaluating shell commands."""

    _CHAIN_TOKENS = frozenset({";", "&&", "||", "|"})

    # Built-in hardening patterns enforced in addition to config.blocked_patterns.
    # Covers bypass variants: rm -rf flags, curl|sh pipes, chmod 777, dd, mkfs,
    # fork bomb. Matched case-insensitively.
    _EXTRA_BLOCKED_PATTERNS = [
        r"rm\s+-[a-z]*r",  # rm -r / -rf / -Rf / -fr variants
        r"rm\s+--recursive",
        r"(curl|wget)\b[^\n|]*\|\s*(sh|bash|zsh|dash)\b",  # curl|sh pipe
        r"\|\s*(sh|bash|zsh|dash)\b",  # generic pipe-to-shell
        r"chmod\b[^\n]*777",  # chmod 777 with or without -R
        r"\bdd\b",  # dd (any invocation)
        r"\bmkfs\b",  # mkfs (any invocation)
        r":\(\)\s*\{\s*:\s*\|\s*:?\s*&\s*\}",  # fork bomb :(){:|:&}
    ]

    def __init__(self, config: AllowanceConfig):
        self.config = config
        self._blocked_regexes = []
        for pattern in list(self.config.blocked_patterns) + self._EXTRA_BLOCKED_PATTERNS:
            try:
                self._blocked_regexes.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                # Invalid regex in config must never crash the guard;
                # fall back to literal match (fail-closed, timeout-safe:
                # stdlib re has no timeout, so any engine error -> ASK).
                self._blocked_regexes.append(
                    re.compile(re.escape(pattern), re.IGNORECASE)
                )

    def _has_chaining(self, cmd_trimmed: str, tokens: List[str]) -> bool:
        """Detect shell chaining/metachars so whitelist prefix can't be abused."""
        if "$(" in cmd_trimmed or "`" in cmd_trimmed:
            return True
        for tok in tokens:
            if tok in self._CHAIN_TOKENS:
                return True
            if "$(" in tok or "`" in tok:
                return True
        # Raw fallback for operators glued without whitespace (e.g. "hi;rm").
        if ";" in cmd_trimmed:
            return True
        if "&&" in cmd_trimmed or "||" in cmd_trimmed:
            return True
        if "|" in cmd_trimmed:
            return True
        return False

    def _matches_whitelist(self, tokens: List[str]) -> bool:
        for safe_cmd in self.config.whitelisted_commands:
            stripped = (safe_cmd or "").strip()
            if not stripped:
                continue
            try:
                safe_tokens = shlex.split(stripped)
            except (ValueError, Exception):
                safe_tokens = stripped.split()
            if not safe_tokens:
                continue
            if tokens[: len(safe_tokens)] == safe_tokens:
                return True
        return False

    def evaluate(self, command: str) -> CommandStatus:
        """
        Evaluates a shell command against the allowance rules.
        """
        try:
            cmd_trimmed = (command or "").strip()
            if not cmd_trimmed:
                return CommandStatus.ASK

            # Check blocked patterns first (case-insensitive, config + built-in).
            for pattern in self._blocked_regexes:
                try:
                    if pattern.search(cmd_trimmed):
                        return CommandStatus.BLOCKED
                except Exception:
                    continue

            # Parse via shlex; unparseable input fails closed to ASK.
            try:
                tokens = shlex.split(cmd_trimmed)
            except (ValueError, Exception):
                return CommandStatus.ASK
            if not tokens:
                return CommandStatus.ASK

            # Chaining bypass guard: whitelist prefix must NOT approve
            # commands containing ; && || | $() backticks.
            if self._has_chaining(cmd_trimmed, tokens):
                return CommandStatus.ASK

            # Check whitelisted commands (token-boundary match, not raw prefix).
            if self._matches_whitelist(tokens):
                return CommandStatus.APPROVED

            # Strict mode validator: unknown/typo modes fail closed to ASK.
            # "auto" approves ONLY full whitelist matches (handled above);
            # anything reaching here in auto mode -> ASK (no fail-open).
            mode = str(getattr(self.config, "mode", "ask") or "ask").strip().lower()
            if mode == "block":
                return CommandStatus.BLOCKED
            return CommandStatus.ASK
        except Exception:
            return CommandStatus.ASK
