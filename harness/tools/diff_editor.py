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
        r'<<<< SEARCH\n(.*?)\n====\n(.*?)>>>> REPLACE',
        re.DOTALL
    )

    def apply_diff(self, file_path: Path | str, diff_content: str) -> DiffResult:
        """
        Parses search-replace blocks and applies them to the file content.
        """
        path = Path(file_path)
        if not path.exists():
            return DiffResult(success=False, error=f"File not found: {path}")
            
        try:
            content = path.read_text(encoding='utf-8')
            
            blocks = self.BLOCK_PATTERN.findall(diff_content)
            if not blocks:
                return DiffResult(success=False, error="No valid search/replace blocks found")
                
            lines_changed = 0
            for search_block, replace_block in blocks:
                if not search_block.strip():
                    return DiffResult(
                        success=False,
                        error="ValueError: search block must not be empty"
                    )
                if not replace_block.strip():
                    return DiffResult(
                        success=False,
                        error="ValueError: replace block must not be empty"
                    )
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
                
                # Approximate lines changed based on replace block
                lines_changed += len(replace_block.splitlines())
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
