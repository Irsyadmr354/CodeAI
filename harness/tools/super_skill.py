import logging
import shlex
import shutil
from pathlib import Path

from harness.tools.diff_editor import DiffEditor, DiffResult
from harness.tools.shell_runner import ShellRunner, ShellResult

logger = logging.getLogger(__name__)

class CodeSkeletonizer:
    """Creates skeletons of code files."""

    def skeletonize(self, path: Path | str) -> str | None:
        """
        Produce a skeleton of the target file.
        Delegates to harness.context.ast_indexer when available.

        Returns the skeleton string on success (empty string when the file
        has no extractable signatures but was read successfully).
        Returns None plus a log record on failure (missing file, unreadable,
        indexer unavailable, unsupported suffix, extraction error) so callers
        can distinguish failure from an empty-but-valid result.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("skeletonize: file not found: %s", p)
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as e:
            logger.warning("skeletonize: cannot read %s: %s", p, e)
            return None
        try:
            from harness.context.ast_indexer import (
                extract_js_ts_signatures,
                extract_python_signatures,
            )
        except ImportError as e:
            logger.warning("skeletonize: ast_indexer unavailable: %s", e)
            return None
        try:
            suffix = p.suffix.lower()
            if suffix == ".py":
                result = extract_python_signatures(text)
            elif suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                result = extract_js_ts_signatures(text)
            else:
                logger.info("skeletonize: unsupported suffix for %s", p)
                return None
            if result and result.strip():
                return result
            return ""
        except Exception as e:
            logger.warning("skeletonize failed for %s: %s", p, e)
            return None

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
        
    def skeletonize_file(self, path: Path | str) -> str | None:
        """Get the skeleton of a file; None on failure, empty string when empty."""
        return self.skeletonizer.skeletonize(path)
        
    def search_symbol(self, pattern: str, directory: Path | str) -> ShellResult:
        """
        Search for a symbol in a directory using ripgrep (or fallback to grep).
        Tries rg first when available, then grep, via separate guarded calls.
        """
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
        if shutil.which("rg") is not None:
            rg_cmd = f"rg -n {quoted_pattern} {quoted_dir}"
            rg_res = self.shell_runner.run(rg_cmd)
            hint = (rg_res.stderr or "").lower()
            if "approval" not in hint and "blocked" not in hint:
                if rg_res.exit_code == 0:
                    return rg_res
                if rg_res.stdout and rg_res.stdout.strip():
                    return rg_res
        grep_cmd = f"grep -rn {quoted_pattern} {quoted_dir}"
        return self.shell_runner.run(grep_cmd)
