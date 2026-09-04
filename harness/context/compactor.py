import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ANSI escape sequence regex (CSI + single-char) plus OSC sequences (ESC ] ... BEL / ESC \)
ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
OSC_RE = re.compile(r'\x1B\].*?(?:\x07|\x1B\\)')

def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from a string."""
    text = OSC_RE.sub('', text)
    return ANSI_ESCAPE_RE.sub('', text)

def trim_log_output(log_output: str, max_lines: int = 10) -> str:
    """
    Trim log output to max_lines essential lines.
    Keeps first and last lines, and prioritizes ERROR, WARN, INFO, and exit codes.
    """
    clean_log = strip_ansi(log_output)
    lines = clean_log.splitlines()
    if len(lines) <= max_lines:
        return clean_log
        
    essential_patterns = [
        re.compile(r'\bERROR\b', re.IGNORECASE),
        re.compile(r'\bWARN(?:ING)?\b', re.IGNORECASE),
        re.compile(r'\bINFO\b', re.IGNORECASE),
        re.compile(r'\bFATAL\b', re.IGNORECASE),
        re.compile(r'Traceback', re.IGNORECASE),
        re.compile(r'exit[_ ]code', re.IGNORECASE),
        re.compile(r'\bFAILED\b', re.IGNORECASE)
    ]
    
    kept_indices = set()
    if lines:
        kept_indices.add(0)
        kept_indices.add(len(lines) - 1)
    
    # Extract essential lines
    essential_indices = []
    for i, line in enumerate(lines):
        if i in kept_indices:
            continue
        if any(p.search(line) for p in essential_patterns):
            essential_indices.append(i)
            
    # Add essential lines up to max_lines
    for idx in essential_indices:
        if len(kept_indices) < max_lines:
            kept_indices.add(idx)
        else:
            break
            
    # Fill with context lines if we still have room
    current_idx = 1
    while len(kept_indices) < max_lines and current_idx < len(lines) - 1:
        kept_indices.add(current_idx)
        current_idx += 1
        
    sorted_indices = sorted(list(kept_indices))

    # Reserve budget for markers so total output (lines + markers) <= max_lines
    def _count_markers(idxs: list) -> int:
        return sum(1 for a, b in zip(idxs, idxs[1:]) if b > a + 1)

    essential_set = set(essential_indices) & set(sorted_indices)
    while sorted_indices and len(sorted_indices) + _count_markers(sorted_indices) > max_lines and len(sorted_indices) > 2:
        middle = [i for i in sorted_indices if i != sorted_indices[0] and i != sorted_indices[-1]]
        victim = None
        for cand in reversed(middle):
            if cand not in essential_set:
                victim = cand
                break
        if victim is None:
            victim = middle[-1]
        sorted_indices.remove(victim)

    result = []
    last_idx = -1
    for idx in sorted_indices:
        if last_idx != -1 and idx > last_idx + 1:
            result.append("... [truncated] ...")
        result.append(lines[idx])
        last_idx = idx
        
    return "\n".join(result)

@dataclass
class ContextCompactor:
    """Sliding history summarizer and token compactor.

    Session-owned state: ``history`` (exposed as ``active_history``) and
    ``fact_cards`` belong to the chat session, NOT to any model. Switching
    models via ``set_active_model``/``switch_model`` only updates the
    informational ``active_model`` label and MUST NEVER reset history or
    fact cards. Trimming happens solely on token budget via
    ``_compact_if_needed`` (fact cards + keyword summary). Explicit wipe is
    only available through ``clear()`` for the ``/clear`` command.
    """
    token_threshold: int = 4000
    fact_cards: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    active_model: Optional[str] = field(default=None)

    def set_active_model(self, model: Optional[str]) -> None:
        """Track model switch WITHOUT resetting session history.

        Only updates the informational ``active_model`` label. History and
        fact cards are preserved verbatim; no compaction or clearing here.
        """
        self.active_model = model

    def switch_model(self, model: Optional[str]) -> None:
        """Alias for :meth:`set_active_model` (model-picker switch path).

        Preserves ``history``/``fact_cards``; never clears.
        """
        self.set_active_model(model)

    def _do_switch(self, model: Optional[str]) -> None:
        """Compat shim for switch path: preserve history, update label only."""
        self.set_active_model(model)

    def clear(self) -> None:
        """Explicit wipe for the ``/clear`` command only.

        Clears ``history`` and ``fact_cards``. Never called from any model
        switch path. ``active_model`` label is kept (session still on model).
        """
        self.history.clear()
        self.fact_cards.clear()

    def add_message(self, role: str, content: str, model: Optional[str] = None) -> None:
        # Optional model label: track switch without wiping session state.
        if model is not None and model != self.active_model:
            self.set_active_model(model)
        self.history.append({"role": role, "content": content})
        self._compact_if_needed()
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation: ~4 chars per token."""
        return len(text) // 4
        
    def _compact_if_needed(self) -> None:
        total_tokens = sum(self._estimate_tokens(m["content"]) for m in self.history)
        total_tokens += sum(self._estimate_tokens(f) for f in self.fact_cards)
        if total_tokens > self.token_threshold and len(self.history) > 2:
            self._compact_history()

    def _summarize_compacted(self, to_compact: List[Dict[str, str]]) -> str:
        """Single-line keyword summary (not just a count)."""
        text = " ".join(m.get("content", "") for m in to_compact)
        words = re.findall(r'[A-Za-z0-9_]{4,}', text.lower())
        stop = {'that', 'this', 'with', 'from', 'have', 'were', 'been', 'will', 'would', 'should', 'could', 'about', 'into', 'over', 'after', 'before', 'under', 'role', 'content', 'past', 'messages'}
        freq: Dict[str, int] = {}
        order: Dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            if w not in freq:
                freq[w] = 0
                order[w] = len(order)
            freq[w] += 1
        ranked = sorted(freq, key=lambda w: (-freq[w], order[w]))[:5]
        keywords = ", ".join(ranked) if ranked else "no-keywords"
        return f"Summary of {len(to_compact)} msgs: {keywords}"
            
    def _compact_history(self) -> None:
        """Condense old conversation state into fact cards."""
        to_compact = self.history[:-2]
        self.history = self.history[-2:]
        
        summary = self._summarize_compacted(to_compact)
        self.fact_cards.append(summary)
        
    def get_context(self, model: Optional[str] = None) -> Dict[str, Any]:
        # Optional model label: track switch without wiping session state.
        # Never reset history/fact_cards on model change; only token-budget
        # compaction (via add_message) may trim. Return copies so external
        # mutation cannot wipe internal session state.
        if model is not None and model != self.active_model:
            self.set_active_model(model)
        return {
            "fact_cards": list(self.fact_cards),
            "active_history": list(self.history),
            "active_model": self.active_model,
        }
