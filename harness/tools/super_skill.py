import shlex
from pathlib import Path

from harness.tools.diff_editor import DiffEditor, DiffResult
from harness.tools.shell_runner import ShellRunner, ShellResult

class CodeSkeletonizer:
    """Creates skeletons of code files."""

    def skeletonize(self, path: Path | str) -> str:
        """
        Produce a skeleton of the target file.
        Delegates to harness.context.ast_indexer when available;
        falls back to a safe stub on any failure.
        """
        p = Path(path)
        if not p.exists():
            return ""
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            return f"# Skeleton for {p.name}\n"
        try:
            from harness.context.ast_indexer import (
                extract_js_ts_signatures,
                extract_python_signatures,
            )
        except ImportError:
            return f"# Skeleton for {p.name}\n"
        try:
            suffix = p.suffix.lower()
            if suffix == ".py":
                result = extract_python_signatures(text)
            elif suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                result = extract_js_ts_signatures(text)
            else:
                return f"# Skeleton for {p.name}\n"
            if result and result.strip():
                return result
            return f"# Skeleton for {p.name}\n"
        except Exception:
            return f"# Skeleton for {p.name}\n"

class SuperSkill:
    """Unified orchestration for diffs, shells, and code analysis."""
    
    def __init__(self, shell_runner: ShellRunner):
        self.diff_editor = DiffEditor()
        self.shell_runner = shell_runner
        self.skeletonizer = CodeSkeletonizer()

    def apply_diff(self, file_path: Path | str, diff_content: str) -> DiffResult:
        """Apply a diff to a file."""
        return self.diff_editor.apply_diff(file_path, diff_content)
        
    def run_command(self, cmd: str) -> ShellResult:
        """Run a shell command."""
        return self.shell_runner.run(cmd)
        
    def skeletonize_file(self, path: Path | str) -> str:
        """Get the skeleton of a Python file."""
        return self.skeletonizer.skeletonize(path)
        
    def search_symbol(self, pattern: str, directory: Path | str) -> ShellResult:
        """
        Search for a symbol in a directory using ripgrep (or fallback to grep).
        """
        # rg -n is ripgrep with line numbers.
        # If rg is not available, we can fallback to grep.
        dir_path = Path(directory)
        if not pattern:
            return ShellResult(
                stdout="",
                stderr="Empty search pattern",
                exit_code=1,
                command="search_symbol",
            )
        if not dir_path.exists() or not dir_path.is_dir():
            return ShellResult(
                stdout="",
                stderr=f"Invalid search directory: {dir_path}",
                exit_code=1,
                command="search_symbol",
            )
        quoted_pattern = shlex.quote(pattern)
        quoted_dir = shlex.quote(str(dir_path))
        cmd = f"rg -n {quoted_pattern} {quoted_dir} || grep -rn {quoted_pattern} {quoted_dir}"
        return self.shell_runner.run(cmd)
