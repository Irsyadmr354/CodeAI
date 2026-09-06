import logging
import re
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

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

# Allowlist for pre_tool_call override_arg — only these args may be overwritten.
# Includes model/effort/limit plus read_mode for backward compatibility.
_ALLOWED_OVERRIDE_ARGS = frozenset({
    "model",
    "effort",
    "limit",
    "read_mode",
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
            if not stripped or stripped.startswith('#'):
                continue
            # Preserve list/nested items: strip a leading '-' then parse key:value.
            if stripped.startswith('-'):
                stripped = stripped[1:].strip()
                if not stripped:
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
            logger.warning("hooks file not found: %s", self.hooks_file)
            return
            
        try:
            content = self.hooks_file.read_text(encoding='utf-8')
        except OSError as exc:
            logger.warning("failed to read hooks file %s: %s", self.hooks_file, exc)
            return
        
        # Find yaml code blocks (case-insensitive yaml|yml, plain fence allowed,
        # single-line fence supported: newline after opening fence is optional)
        blocks = re.findall(r'```(?:yaml|yml)?[ \t]*\r?\n?(.*?)```', content, re.DOTALL | re.IGNORECASE)
        if not blocks:
            logger.warning("no fenced hook blocks found in %s", self.hooks_file)
        for block in blocks:
            config = self._parse_simple_yaml(block)
            hook_type = str(config.get('type', '')).strip().lower()
            if hook_type in self.hooks:
                self.hooks[hook_type].append(config)
            else:
                logger.warning("ignoring hook with unknown/missing type: %r", hook_type)

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
                raw_arg = hook['override_arg']
                if not isinstance(raw_arg, str):
                    logger.warning("rejecting non-string override_arg: %r", raw_arg)
                    continue
                arg_name = raw_arg.strip()
                if not arg_name:
                    logger.warning("rejecting empty override_arg for tool %r", tool_name)
                    continue
                if arg_name not in _ALLOWED_OVERRIDE_ARGS:
                    logger.warning(
                        "rejecting override_arg %r not in allowlist for tool %r",
                        arg_name, tool_name,
                    )
                    continue
                mutated_args[arg_name] = hook['override_value']
        return mutated_args

    def dispatch_post_tool_call(self, tool_name: str, output: Any) -> Any:
        """Lifecycle hook: post_tool_call filters output."""
        wanted = str(tool_name or "").strip().lower()
        result = output
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
            if should_filter and isinstance(result, str):
                # Example of filtering: truncating output
                try:
                    limit = int(str(hook.get('limit', 100)).strip())
                except (ValueError, TypeError, AttributeError):
                    logger.warning(
                        "invalid limit %r for tool %r, falling back to 100",
                        hook.get('limit', 100), tool_name,
                    )
                    limit = 100
                if limit < 0:
                    logger.warning(
                        "negative limit %r for tool %r, resetting to 100",
                        hook.get('limit'), tool_name,
                    )
                    limit = 100
                if len(result) > limit:
                    result = result[:limit] + "... [filtered]"
        return result

    def dispatch_pre_generation(self, state: dict) -> dict:
        """Lifecycle hook: pre_generation injects real-time state."""
        mutated_state = dict(state)
        for hook in self.hooks.get('pre_generation', []):
            if 'inject_state_key' in hook and 'inject_state_value' in hook:
                key = hook['inject_state_key']
                if key in _ALLOWED_INJECT_KEYS:
                    mutated_state[key] = hook['inject_state_value']
                else:
                    logger.warning("rejecting inject_state_key %r not in allowlist", key)
        return mutated_state
