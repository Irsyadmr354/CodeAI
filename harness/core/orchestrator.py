import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.core.subagents import MessageBus, SubagentSpawner, SubagentRole, SubagentStatus
from harness.core.hooks import HooksDispatcher
from harness.rules.agents_parser import AgentsParser, AgentRules
from harness.context.compactor import ContextCompactor

logger = logging.getLogger(__name__)


CODEAI_IDENTITY = (
    "You are CodeAI, an autonomous multi-provider coding harness. "
    "The active model is {provider_model}, powered by its upstream provider. "
    "When asked who you are ('who are you?', 'siapa kamu?', or any identity question), "
    "answer that you are CodeAI running on {provider_model}. "
    "Never claim Antigravity, Claude, GPT, or DeepSeek as your identity; "
    "those names refer only to the underlying upstream model powering this session, not to you."
)


class AgentsComplianceError(RuntimeError):
    """Raised when a task violates AGENTS.md rules."""
    pass


class Orchestrator:
    def __init__(
        self,
        hooks_dispatcher: HooksDispatcher,
        agents_parser: AgentsParser,
        config: Optional[Any] = None,
        gateway: Optional[Any] = None,
    ):
        self.hooks_dispatcher = hooks_dispatcher
        self.agents_parser = agents_parser
        self.config = config
        self.gateway = gateway
        self.message_bus = MessageBus()
        self.spawner = SubagentSpawner(self.message_bus)
        self.active_subagent_id: Optional[str] = None
        self.workflow_state: str = "IDLE"

        # Load AGENTS.md rules once at init — if file exists, compliance is mandatory
        self._agents_rules: Optional[AgentRules] = None
        self._agents_file_exists: bool = agents_parser.file_path.exists()
        if self._agents_file_exists:
            self._agents_rules = agents_parser.parse()
            logger.debug(
                "AGENTS.md loaded — 100%% compliance enforced. "
                f"Forbidden: {self._agents_rules.forbidden}, "
                f"Boundaries: {self._agents_rules.boundaries}"
            )

        # Sliding context compactor for token savings
        token_budget = 4000
        if config and hasattr(config, "compaction"):
            token_budget = getattr(config.compaction, "token_budget", 4000)
        self.compactor = ContextCompactor(token_threshold=token_budget)

    # ------------------------------------------------------------------
    # AGENTS.md enforcement helpers
    # ------------------------------------------------------------------

    def _preflight_agents_check(self, user_request: str) -> None:
        """
        Pre-flight compliance check against AGENTS.md.
        Forbidden keywords use word-boundary matching (case-insensitive).
        Workspace boundaries use path-aware detection: mere mention of the
        boundary phrase is NOT a violation; real path access (traversal,
        sensitive absolute paths, outside-workspace phrases) IS blocked.
        """
        if not self._agents_file_exists or self._agents_rules is None:
            return  # No AGENTS.md — no enforcement needed

        for forbidden in self._agents_rules.forbidden:
            if self._matches_forbidden(user_request, forbidden):
                raise AgentsComplianceError(
                    f"🚫 AGENTS.md Compliance Violation: request contains forbidden keyword '{forbidden}'.\n"
                    "AGENTS.md is present — 100% compliance is mandatory. Task blocked."
                )

        for boundary in self._agents_rules.boundaries:
            if self._violates_boundary(user_request, boundary):
                raise AgentsComplianceError(
                    f"🚫 AGENTS.md Boundary Violation: request crosses boundary '{boundary}'.\n"
                    "AGENTS.md is present — 100% compliance is mandatory. Task blocked."
                )

    @staticmethod
    def _matches_forbidden(user_request: str, phrase: str) -> bool:
        """Word-boundary, case-insensitive match for forbidden phrases."""
        phrase = (phrase or "").strip()
        if not phrase:
            return False
        esc = re.escape(phrase).replace(r"\ ", r"\s+")
        prefix = r"\b" if (phrase[0].isalnum() or phrase[0] == "_") else ""
        suffix = r"\b" if (phrase[-1].isalnum() or phrase[-1] == "_") else ""
        try:
            return re.search(prefix + esc + suffix, user_request, re.IGNORECASE) is not None
        except re.error:
            return phrase.lower() in user_request.lower()

    def _violates_boundary(self, user_request: str, boundary: str) -> bool:
        """Path-aware boundary check: mention != violation."""
        boundary = (boundary or "").strip()
        if not boundary:
            return False
        b_lower = boundary.lower()
        # Workspace/path-related boundaries: block real path access only.
        if any(k in b_lower for k in ("workspace", "directory", "folder", "path", "repo", "project", "root")):
            if re.search(r"\.\.(?:/|\\|\b)", user_request):
                return True
            if re.search(r"(?<![\w.])~(?:/|\b)", user_request):
                return True
            if re.search(r"[A-Za-z]:[\\/]", user_request):
                return True
            if re.search(r"\\\\", user_request):
                return True
            if re.search(
                r"(?<![\w.])/(?:etc|root|tmp|var|home|usr|bin|opt|private)(?:/|\b)",
                user_request,
                re.IGNORECASE,
            ):
                return True
            if re.search(
                r"\boutside\b[^\n]{0,40}\bworkspace\b"
                r"|\bescape\b[^\n]{0,40}\bworkspace\b"
                r"|\bbypass\b[^\n]{0,40}\bworkspace\b"
                r"|\bignore\b[^\n]{0,40}\bworkspace\b",
                user_request,
                re.IGNORECASE,
            ):
                return True
            # Mere mention of the boundary phrase (e.g. "stay within workspace") is benign.
            return False
        # Generic boundaries: word-boundary match, exempting discussion/mention context.
        if self._matches_forbidden(user_request, boundary):
            if re.search(
                r"\b(explain|what is|describe|discuss|mention|respect|follow|according to|stay within|comply|adhere)\b",
                user_request,
                re.IGNORECASE,
            ):
                return False
            return True
        return False

    def _build_agents_system_preamble(self) -> str:
        """
        Build a system prompt preamble from AGENTS.md rules so the LLM
        is also informed about compliance constraints.
        """
        if not self._agents_file_exists or self._agents_rules is None:
            return ""

        lines = [
            "=== AGENTS.md — GLOBAL RULES (100% MANDATORY COMPLIANCE) ===",
            "These rules are absolute. You MUST NOT violate them under any circumstance.",
        ]

        if self._agents_rules.allowed:
            lines.append("\n[ALLOWED]")
            for item in self._agents_rules.allowed:
                lines.append(f"  ✅ {item}")

        if self._agents_rules.forbidden:
            lines.append("\n[FORBIDDEN — NEVER DO THESE]")
            for item in self._agents_rules.forbidden:
                lines.append(f"  ❌ {item}")

        if self._agents_rules.boundaries:
            lines.append("\n[BOUNDARIES]")
            for item in self._agents_rules.boundaries:
                lines.append(f"  ⚠️  {item}")

        lines.append("=== END AGENTS.md ===\n")
        return "\n".join(lines)

    def _resolve_active_model_label(self) -> str:
        """Resolve 'provider/model' label for identity (handles bare or prefixed models)."""
        try:
            provider = getattr(self.config.provider, "default", None) if self.config else None
            active_model = getattr(self.config.provider, "active_model", None) if self.config else None
        except Exception:
            provider, active_model = None, None
        provider = (provider or "unknown").strip() or "unknown"
        active_model = (active_model or "unknown").strip() or "unknown"
        if active_model == "default":
            return f"{provider}/default"
        if "/" in active_model:
            return active_model
        return f"{provider}/{active_model}"

    def _build_identity_system_prompt(self) -> str:
        """Render CODEAI_IDENTITY with the live provider/model label."""
        try:
            label = self._resolve_active_model_label()
        except Exception:
            label = "unknown/unknown"
        try:
            return CODEAI_IDENTITY.format(provider_model=label)
        except Exception:
            return CODEAI_IDENTITY.replace("{provider_model}", label)

    def _build_precedence_system_prompt(self, user_request: str = "") -> str:
        """
        Build system prompt with strict precedence order:
        Base < System < Global < AGENTS < HOOKS < User.
        Each layer is labelled so ordering is auditable.
        """
        layers = [
            "[BASE]\nCodeAI base system: autonomous coding harness, subagent-first.",
            "[SYSTEM]\nSystem constraints: surgical diffs only, verify before done, no raw rewrites.",
            "[GLOBAL]\nGlobal rules: 100% AGENTS.md compliance mandatory; safety overrides user on conflict.",
        ]
        agents_preamble = self._build_agents_system_preamble()
        if agents_preamble:
            layers.append("[AGENTS]\n" + agents_preamble)
        else:
            layers.append("[AGENTS]\n(none)")
        base_joined = "\n\n".join(layers)
        with_hooks = self.hooks_dispatcher.dispatch_before_init(base_joined)
        if with_hooks != base_joined and with_hooks.startswith(base_joined):
            added = with_hooks[len(base_joined):]
            system_prompt = base_joined + "\n\n[HOOKS]" + added
        elif "[HOOKS]" not in with_hooks:
            system_prompt = with_hooks + "\n\n[HOOKS]\n(none)"
        else:
            system_prompt = with_hooks
        if user_request:
            system_prompt += f"\n\n[USER]\n{user_request}"
        else:
            system_prompt += "\n\n[USER]\n(see user turn)"
        return system_prompt

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def run_pipeline(self, user_request: str) -> str:
        """Execute user request through the active LLM provider."""
        self.workflow_state = "RUNNING"
        # FR-6: ensure a steerable subagent exists while RUNNING so
        # steer_active_subagent / process_user_input are reachable.
        try:
            _track = self.spawner.spawn(SubagentRole.CODER)
            _track.current_task = user_request
            _track.status = SubagentStatus.RUNNING
            self.active_subagent_id = _track.id
        except Exception:
            try:
                _tmp = f"pipeline-{uuid.uuid4()}"
                try:
                    self.message_bus.register_subagent(_tmp)
                except Exception:
                    pass
                self.active_subagent_id = _tmp
            except Exception:
                pass

        # 1. AGENTS.md pre-flight check — blocks request if violation found
        try:
            self._preflight_agents_check(user_request)
        except AgentsComplianceError as e:
            self.workflow_state = "BLOCKED"
            return str(e)

        # 1b. Engine tool-call verification before gateway (FR-1).
        try:
            if self._agents_file_exists and self._agents_rules is not None:
                _ok = self.agents_parser.verify_tool_call(
                    "run_pipeline",
                    {"input": user_request, "content": user_request},
                    self._agents_rules,
                )
                if not _ok:
                    self.workflow_state = "BLOCKED"
                    return (
                        "🚫 AGENTS.md Compliance Violation: tool call blocked by verify_tool_call.\n"
                        "AGENTS.md is present — 100% compliance is mandatory. Task blocked."
                    )
        except AgentsComplianceError as e:
            self.workflow_state = "BLOCKED"
            return str(e)
        except Exception:
            pass

        # 2. Build system prompt with strict precedence: Base<System<Global<AGENTS<HOOKS<User
        system_prompt = self._build_precedence_system_prompt(user_request)

        try:
            if not self.gateway and self.config:
                from harness.models.gateway import LLMGateway
                self.gateway = LLMGateway(self.config)

            active_model = (
                getattr(self.config.provider, "active_model", None) if self.config else None
            )

            from harness.models.base import ChatMessage

            # 3. Compose messages: identity (first) + system + compactor context + current user turn
            messages: List[ChatMessage] = []
            messages.append(ChatMessage(role="system", content=self._build_identity_system_prompt()))
            if system_prompt and system_prompt.strip():
                messages.append(ChatMessage(role="system", content=system_prompt.strip()))

            # Inject fact cards from compactor as a context reminder
            ctx = self.compactor.get_context()
            if ctx["fact_cards"]:
                fact_summary = "Previous context summary:\n" + "\n".join(
                    f"- {fc}" for fc in ctx["fact_cards"]
                )
                messages.append(ChatMessage(role="system", content=fact_summary))

            # Re-inject active history from compactor (already trimmed)
            for hist_msg in ctx["active_history"]:
                messages.append(ChatMessage(role=hist_msg["role"], content=hist_msg["content"]))

            # Current user message
            messages.append(ChatMessage(role="user", content=user_request))

            if self.gateway:
                resp = self.gateway.chat(messages, model=active_model)
                content = resp.get("content", "") or ""

                # 4. Record to compactor for future turns
                self.compactor.add_message("user", user_request)
                self.compactor.add_message("assistant", content)

                self.workflow_state = "DONE"
                try:
                    _sub = self.spawner.get_subagent(self.active_subagent_id) if self.active_subagent_id else None
                    if _sub is not None:
                        _sub.status = SubagentStatus.DONE
                except Exception:
                    pass
                return content
            else:
                self.workflow_state = "DONE"
                try:
                    _sub = self.spawner.get_subagent(self.active_subagent_id) if self.active_subagent_id else None
                    if _sub is not None:
                        _sub.status = SubagentStatus.DONE
                except Exception:
                    pass
                return "No LLM Gateway configured for provider."

        except Exception as e:
            self.workflow_state = "ERROR"
            try:
                _sub = self.spawner.get_subagent(self.active_subagent_id) if self.active_subagent_id else None
                if _sub is not None:
                    _sub.status = SubagentStatus.ERROR
            except Exception:
                pass
            return f"❌ Model/Gateway Error: {str(e)}"

    # ------------------------------------------------------------------
    # Subagent steering
    # ------------------------------------------------------------------

    async def _wait_for_subagent(self, subagent_id: str) -> Dict[str, Any]:
        """Wait for subagent to complete its task via message bus."""
        while True:
            msg = await self.message_bus.receive_from_parent()
            if msg.get("sender_id") == subagent_id and msg.get("type") == "done":
                return msg

    async def steer_active_subagent(self, instruction_delta: str):
        """Injects new instructions to active subagent without resetting context."""
        if not self.active_subagent_id:
            raise ValueError("No active subagent to steer")

        subagent = self.spawner.get_subagent(self.active_subagent_id)
        if subagent and subagent.status == SubagentStatus.RUNNING:
            steer_message = {"type": "steer", "instruction": instruction_delta}
            await self.message_bus.send_to_subagent(self.active_subagent_id, steer_message)

    async def process_user_input(self, input_text: str):
        """For real-time user steering."""
        if self.workflow_state == "RUNNING" and self.active_subagent_id:
            await self.steer_active_subagent(input_text)

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def run_task(self, task: str) -> str:
        """Synchronous wrapper for running a pipeline task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run_pipeline(task))
                    return future.result()
            else:
                return loop.run_until_complete(self.run_pipeline(task))
        except RuntimeError:
            return asyncio.run(self.run_pipeline(task))
