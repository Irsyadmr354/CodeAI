import shlex
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional
from pathlib import Path

from harness.tools.shell_runner import ShellRunner
from harness.context.compactor import trim_log_output

@dataclass
class VerificationResult:
    """Result of the verification auto-repair loop."""
    passed: bool
    errors: List[str]
    retries_used: int

class VerificationLoop:
    """Runs verifications (linting, tests) with retry mechanism."""
    
    def __init__(self, shell_runner: ShellRunner, max_retries: int = 2):
        self.shell_runner = shell_runner
        self.max_retries = max(0, max_retries)

    def verify(
        self,
        changed_files: List[Path | str],
        on_retry: Optional[Callable[..., None]] = None,
    ) -> VerificationResult:
        """
        Verify changed files with linters and tests.
        """
        if not changed_files:
            return VerificationResult(passed=True, errors=[], retries_used=0)
        retries = 0
        all_errors: List[str] = []
        
        while retries < self.max_retries + 1:
            errors: List[str] = []
            
            # Lint Python files
            for file in changed_files:
                p = Path(file)
                if p.suffix.lower() == '.py':
                    # Single tool, no shell chaining (|| trips AllowanceGuard).
                    if shutil.which("ruff"):
                        lint_cmd = f"ruff check {shlex.quote(str(p))}"
                    elif shutil.which("flake8"):
                        lint_cmd = f"flake8 {shlex.quote(str(p))}"
                    else:
                        continue
                    res = self.shell_runner.run(lint_cmd)
                    if res.exit_code != 0:
                        errors.append(
                            f"Lint error in {p}:\nSTDOUT: {trim_log_output(res.stdout)}\nSTDERR: {trim_log_output(res.stderr)}"
                        )
            
            # Run tests globally
            test_cmd = "pytest"
            res = self.shell_runner.run(test_cmd)
            if res.exit_code != 0:
                errors.append(
                    f"Test failures:\nSTDOUT: {trim_log_output(res.stdout)}\nSTDERR: {trim_log_output(res.stderr)}"
                )
                
            if not errors:
                return VerificationResult(passed=True, errors=[], retries_used=retries)

            all_errors.extend(errors)
            if retries >= self.max_retries:
                break
            if callable(on_retry):
                try:
                    try:
                        on_retry(changed_files, errors)
                    except TypeError:
                        try:
                            on_retry(errors)
                        except TypeError:
                            on_retry()
                except Exception as exc:
                    all_errors.append(f"on_retry gagal: {exc}")
            else:
                all_errors.append("retry tanpa perbaikan: tanpa perbaikan nyata (on_retry tidak tersedia)")
            retries += 1
            
        return VerificationResult(
            passed=False, 
            errors=all_errors, 
            retries_used=min(retries, self.max_retries)
        )
