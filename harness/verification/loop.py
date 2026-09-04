from dataclasses import dataclass
from typing import List
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
        self.max_retries = max_retries

    def verify(self, changed_files: List[Path | str]) -> VerificationResult:
        """
        Verify changed files with linters and tests.
        """
        retries = 0
        
        while retries < self.max_retries + 1:
            errors = []
            
            # Lint Python files
            for file in changed_files:
                p = Path(file)
                if p.suffix == '.py':
                    # Attempt ruff first, fallback to flake8
                    lint_cmd = f"ruff check {p} || flake8 {p}"
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
                
            retries += 1
            
        return VerificationResult(
            passed=False, 
            errors=errors, 
            retries_used=retries
        )
