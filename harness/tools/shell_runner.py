import subprocess
import re
from dataclasses import dataclass
from harness.core.allowance import AllowanceGuard, CommandStatus
from harness.context.compactor import trim_log_output

@dataclass
class ShellResult:
    """Result of a shell execution."""
    stdout: str
    stderr: str
    exit_code: int
    command: str

class ShellRunner:
    """Sandboxed shell runner."""
    
    # Simple regex to mask potential secrets (e.g., tokens, api keys)
    SECRET_PATTERN = re.compile(
        r'(?i)(api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{16,}["\']?'
    )

    # Extended patterns: sk-, ghp_/gho_, AKIA, Bearer, --token=, PEM header.
    _EXTRA_SECRET_PATTERNS = (
        re.compile(r'sk-[A-Za-z0-9\-_]{16,}'),
        re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
        re.compile(r'AKIA[0-9A-Z]{16}'),
        re.compile(r'(?i)Bearer\s+[A-Za-z0-9\-._~+/=]{10,}'),
        re.compile(r'(?i)--token[=\s]+[^\s"\'`]+'),
        re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----'),
    )

    def __init__(self, allowance_guard: AllowanceGuard | None, timeout: int = 30):
        self.allowance_guard = allowance_guard
        self.timeout = timeout or 30

    def _mask_secrets(self, text: str) -> str:
        """Mask potential secrets in text."""
        if not text:
            return "" if text is None else text
        masked = self.SECRET_PATTERN.sub(r'\1: ***MASKED***', text)
        for pat in self._EXTRA_SECRET_PATTERNS:
            masked = pat.sub('***MASKED***', masked)
        # Normalize Bearer prefix that was fully masked above.
        return masked

    def run(self, command: str) -> ShellResult:
        """Execute a shell command securely and safely."""
        safe_command = self._mask_secrets(command)
        guard = getattr(self, "allowance_guard", None)
        if guard is not None:
            try:
                status = guard.evaluate(command)
            except Exception:
                status = CommandStatus.ASK
        else:
            status = CommandStatus.APPROVED
        
        if status == CommandStatus.BLOCKED:
            return ShellResult(
                stdout="",
                stderr="Command blocked by AllowanceGuard",
                exit_code=1,
                command=safe_command
            )
        elif status == CommandStatus.ASK:
            # Emulate rejection in unattended mode
            return ShellResult(
                stdout="",
                stderr="Command requires user approval (ASK)",
                exit_code=1,
                command=safe_command
            )
            
        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout or 30
            )
            
            # Apply masking and compaction
            stdout = trim_log_output(self._mask_secrets(process.stdout or ""))
            stderr = trim_log_output(self._mask_secrets(process.stderr or ""))
            
            return ShellResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                command=safe_command
            )
            
        except subprocess.TimeoutExpired:
            return ShellResult(
                stdout="",
                stderr=f"Command timed out after {self.timeout or 30}s",
                exit_code=124,
                command=safe_command
            )
        except Exception as e:
            return ShellResult(
                stdout="",
                stderr=self._mask_secrets(f"Execution error: {e}"),
                exit_code=1,
                command=safe_command
            )
