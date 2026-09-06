import re
import shlex
from enum import Enum
from pathlib import Path
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

    def __init__(self, config: AllowanceConfig, workspace=None):
        self.config = config
        try:
            self._workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        except Exception:
            self._workspace = Path.cwd()
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

    _SENSITIVE_PREFIXES = (
        "/etc", "/root", "/proc", "/sys", "/dev", "/boot",
        "/var/run", "/run/secrets", "/run/", "/var/secrets",
    )

    _SENSITIVE_NAMES = frozenset({
        ".ssh", ".aws", ".gnupg", ".gnupg2", "id_rsa", "id_ed25519",
        "id_ecdsa", ".env", ".passwd", "secrets", "shadow", "passwd",
    })

    _SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx")

    def _has_unquoted_special(self, cmd: str) -> bool:
        """Detect >, <, $, *, ? outside single/double quotes (fail-closed)."""
        in_single = False
        in_double = False
        escaped = False
        for ch in cmd:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and not in_single:
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if ch in "><$*?":
                return True
        return False

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
        # Redirect, env-var expansion and glob outside quotes -> ASK.
        # Covers >, >>, <, $VAR/${VAR}, *, ? (quote-aware).
        try:
            if self._has_unquoted_special(cmd_trimmed):
                return True
        except Exception:
            # Fail-closed: on scanner error treat as chaining.
            return True
        return False

    def _is_unsafe_path_arg(self, arg: str) -> bool:
        """True if a whitelist extra-arg is a sensitive path or escapes workspace."""
        if not arg or arg in ("-", "--"):
            return False
        if arg.startswith("-"):
            # CLI flag, not a path.
            return False
        low = arg.lower()
        # Home expansion or env-var remainder escapes workspace control.
        if arg.startswith("~"):
            return True
        # Sensitive absolute prefixes (/etc, /proc, ...).
        for prefix in self._SENSITIVE_PREFIXES:
            if low == prefix or low.startswith(prefix.rstrip("/") + "/"):
                return True
        # Sensitive file/dir names and secret suffixes.
        try:
            base = Path(arg).name.lower()
        except Exception:
            base = arg.rsplit("/", 1)[-1].lower()
        if base in self._SENSITIVE_NAMES:
            return True
        for suffix in self._SENSITIVE_SUFFIXES:
            if base.endswith(suffix):
                return True
        # Workspace containment: absolute paths must stay inside workspace;
        # relative paths must not traverse outside via .. .
        try:
            ws = self._workspace
            p = Path(arg)
            if p.is_absolute():
                try:
                    p.resolve().relative_to(ws)
                    return False
                except Exception:
                    return True
            # Relative: join + normalize, must stay inside workspace.
            joined = (ws / p)
            try:
                joined.resolve().relative_to(ws)
                return False
            except Exception:
                return True
        except Exception:
            return True

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
                extra = tokens[len(safe_tokens):]
                if not extra:
                    return True
                # Prefix with args: approve only if no sensitive/out-of-workspace path.
                try:
                    unsafe = any(self._is_unsafe_path_arg(a) for a in extra)
                except Exception:
                    continue
                if unsafe:
                    continue
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

            # Block mode evaluated BEFORE whitelist/chaining (fail-closed):
            # in block mode nothing is APPROVED via whitelist.
            mode = str(getattr(self.config, "mode", "ask") or "ask").strip().lower()
            if mode == "block":
                return CommandStatus.BLOCKED

            # Parse via shlex; unparseable input fails closed to ASK.
            try:
                tokens = shlex.split(cmd_trimmed)
            except (ValueError, Exception):
                return CommandStatus.ASK
            if not tokens:
                return CommandStatus.ASK

            # Chaining bypass guard: whitelist prefix must NOT approve
            # commands containing ; && || | $() backticks, redirects ><> ,
            # env expansion $VAR, globs * ? outside quotes.
            if self._has_chaining(cmd_trimmed, tokens):
                return CommandStatus.ASK

            # Check whitelisted commands (token-boundary match, not raw prefix).
            if self._matches_whitelist(tokens):
                return CommandStatus.APPROVED

            # Strict mode validator: unknown/typo modes fail closed to ASK.
            # "auto" approves ONLY full whitelist matches (handled above);
            # anything reaching here in auto mode -> ASK (no fail-open).
            # Block mode already returned above; re-check defensively.
            if mode == "block":
                return CommandStatus.BLOCKED
            return CommandStatus.ASK
        except Exception:
            return CommandStatus.ASK
