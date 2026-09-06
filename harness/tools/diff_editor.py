import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class DiffResult:
    """Result of a diff operation."""
    success: bool
    error: Optional[str] = None
    lines_changed: int = 0

class DiffEditor:
    """Surgical diff editor for applying code modifications."""

    BLOCK_PATTERN = re.compile(
        r'<<<<[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n====[ \t]*\r?\n(.*?)>>>>[ \t]*REPLACE',
        re.DOTALL
    )

    def apply_diff(self, file_path: Path | str, diff_content: str) -> DiffResult:
        """
        Parses search-replace blocks and applies them to the file content.

        Block format (CRLF and surrounding spaces around markers tolerated):

            <<<< SEARCH
            <exact snippet, must occur exactly once>
            ====
            <replacement text>
            >>>> REPLACE

        Variants:
          - Pure delete: non-empty search + empty/whitespace-only replace
            removes the snippet. A replace block containing exactly ``DELETE``
            (case-sensitive, surrounding whitespace ignored) is an explicit
            delete marker with identical effect.
          - Pure insert: empty/whitespace-only search + non-empty replace
            appends the replacement at end of file (newline handled).
          - Both blocks empty is rejected as ambiguous.
          - Non-empty search must match exactly once; zero or multiple
            matches are rejected (no silent first-only replace).
        """
        path = Path(file_path)
        try:
            workspace = Path.cwd().resolve()
        except Exception:
            workspace = Path.cwd()
        if path.is_absolute():
            try:
                resolved = path.resolve()
            except Exception:
                return DiffResult(success=False, error=f"Invalid path: {path}")
            try:
                inside = resolved.is_relative_to(workspace)
            except AttributeError:
                try:
                    resolved.relative_to(workspace)
                    inside = True
                except ValueError:
                    inside = False
            if not inside:
                return DiffResult(
                    success=False,
                    error=(
                        f"Refusing absolute path outside workspace: {path} "
                        f"(workspace: {workspace})"
                    ),
                )
        if not path.exists():
            return DiffResult(success=False, error=f"File not found: {path}")
            
        try:
            content = path.read_text(encoding='utf-8')

            normalized = (diff_content or "").replace("\r\n", "\n").replace("\r", "\n")
            blocks = self.BLOCK_PATTERN.findall(normalized)
            if not blocks:
                return DiffResult(success=False, error="No valid search/replace blocks found")
                
            lines_changed = 0
            for search_block, replace_block in blocks:
                search_empty = not search_block.strip()
                replace_empty = not replace_block.strip()
                is_delete_marker = replace_block.strip() == "DELETE"
                if search_empty and replace_empty:
                    return DiffResult(
                        success=False,
                        error="ValueError: search and replace blocks must not both be empty (ambiguous)"
                    )
                if search_empty and not replace_empty:
                    # Pure insert: append replacement at end of file.
                    lines_changed += len(replace_block.splitlines())
                    if content and not content.endswith("\n"):
                        content += "\n"
                    content += replace_block
                    if not content.endswith("\n"):
                        content += "\n"
                    continue
                if not search_empty and (replace_empty or is_delete_marker):
                    # Pure delete: remove the matched snippet.
                    occurrences = content.count(search_block)
                    if occurrences == 0:
                        return DiffResult(
                            success=False,
                            error=f"Search block not found in file:\n{search_block}"
                        )
                    if occurrences > 1:
                        return DiffResult(
                            success=False,
                            error=(
                                f"Ambiguous search block: found {occurrences} occurrences; "
                                "refine search block to be unique (no silent first-only replace)"
                            )
                        )
                    lines_changed += len(search_block.splitlines())
                    content = content.replace(search_block, "", 1)
                    continue
                occurrences = content.count(search_block)
                if occurrences == 0:
                    return DiffResult(
                        success=False, 
                        error=f"Search block not found in file:\n{search_block}"
                    )
                if occurrences > 1:
                    return DiffResult(
                        success=False,
                        error=(
                            f"Ambiguous search block: found {occurrences} occurrences; "
                            "refine search block to be unique (no silent first-only replace)"
                        )
                    )
                
                # Real diff size: added + removed lines.
                lines_changed += len(search_block.splitlines()) + len(replace_block.splitlines())
                content = content.replace(search_block, replace_block, 1)
                
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name + ".tmp."
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
                    tmp.write(content)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return DiffResult(success=True, lines_changed=lines_changed)
            
        except Exception as e:
            return DiffResult(success=False, error=str(e))
