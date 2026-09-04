import re
from typing import Dict, List, Any
from pathlib import Path

# Allowlist for pre_generation state injection — only these keys may be set via hooks.
_ALLOWED_INJECT_KEYS = frozenset({
    "context",
    "extra_context",
    "system_note",
    "memory_hint",
    "time_context",
    "current_time",
    "locale",
    "timezone",
})

class HooksDispatcher:
    """Lifecycle dispatcher that reads SYSTEM_HOOKS.md and dispatches hooks."""
    def __init__(self, hooks_file: str | Path):
        self.hooks_file = Path(hooks_file)
        self.hooks: Dict[str, List[Dict[str, Any]]] = {
            'before_init': [],
            'pre_tool_call': [],
            'post_tool_call': [],
            'pre_generation': []
        }
        self._load_hooks()
        
    def _parse_simple_yaml(self, content: str) -> Dict[str, Any]:
        """Very basic YAML parser relying only on the standard library."""
        result = {}
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('-'):
                continue
            if ':' not in stripped:
                continue
            key, val = stripped.split(':', 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            # Remove surrounding quotes only when properly paired.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            result[key] = val
        return result

    def _load_hooks(self):
        if not self.hooks_file.exists():
            return
            
        content = self.hooks_file.read_text(encoding='utf-8')
        
        # Find yaml code blocks (case-insensitive yaml|yml, plain fence allowed)
        blocks = re.findall(r'```(?:yaml|yml)?\s*\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
        for block in blocks:
            config = self._parse_simple_yaml(block)
            hook_type = str(config.get('type', '')).strip().lower()
            if hook_type in self.hooks:
                self.hooks[hook_type].append(config)

    def dispatch_before_init(self, system_prompt: str) -> str:
        """Lifecycle hook: before_init modifies system prompt."""
        modified_prompt = system_prompt
        for hook in self.hooks.get('before_init', []):
            if 'append_prompt' in hook:
                modified_prompt += f"\n{hook['append_prompt']}"
        return modified_prompt

    def dispatch_pre_tool_call(self, tool_name: str, args: dict) -> dict:
        """Lifecycle hook: pre_tool_call validates/mutates args."""
        mutated_args = dict(args)
        wanted = str(tool_name or "").strip().lower()
        for hook in self.hooks.get('pre_tool_call', []):
            target = str(hook.get('target_tool', '')).strip().lower()
            if target != wanted:
                continue
            action = str(hook.get('action', '')).strip().lower()
            if action in ('block', 'deny', 'block_tool', 'deny_tool'):
                raise PermissionError(
                    f"pre_tool_call blocked tool '{tool_name}' by hook"
                )
            if 'override_arg' in hook and 'override_value' in hook:
                mutated_args[hook['override_arg']] = hook['override_value']
        return mutated_args

    def dispatch_post_tool_call(self, tool_name: str, output: Any) -> Any:
        """Lifecycle hook: post_tool_call filters output."""
        wanted = str(tool_name or "").strip().lower()
        for hook in self.hooks.get('post_tool_call', []):
            target = str(hook.get('target_tool', '')).strip().lower()
            if target != wanted:
                continue
            raw = hook.get('filter_output')
            if isinstance(raw, bool):
                should_filter = raw is True
            elif isinstance(raw, int):
                should_filter = raw == 1
            elif isinstance(raw, str):
                should_filter = raw.strip().lower() in ('true', '1')
            else:
                should_filter = False
            if should_filter and isinstance(output, str):
                # Example of filtering: truncating output
                try:
                    limit = int(str(hook.get('limit', 100)).strip())
                except (ValueError, TypeError, AttributeError):
                    limit = 100
                if limit < 0:
                    limit = 100
                if len(output) > limit:
                    return output[:limit] + "... [filtered]"
        return output

    def dispatch_pre_generation(self, state: dict) -> dict:
        """Lifecycle hook: pre_generation injects real-time state."""
        mutated_state = dict(state)
        for hook in self.hooks.get('pre_generation', []):
            if 'inject_state_key' in hook and 'inject_state_value' in hook:
                key = hook['inject_state_key']
                if key in _ALLOWED_INJECT_KEYS:
                    mutated_state[key] = hook['inject_state_value']
        return mutated_state
