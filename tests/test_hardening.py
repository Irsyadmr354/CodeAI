"""Regression hardening: allowance/shell/hooks/loop/combo/diff/orchestrator (mock-based)."""
import asyncio
import inspect
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.config import AllowanceConfig
from harness.core.allowance import AllowanceGuard, CommandStatus
from harness.core.hooks import HooksDispatcher
from harness.core.orchestrator import Orchestrator
from harness.models.combo import ComboManager
from harness.rules.agents_parser import AgentsParser
from harness.tools.diff_editor import DiffEditor
from harness.tools.shell_runner import ShellRunner
from harness.verification.loop import VerificationLoop


def _isolated_combo_manager(tmpdir: str) -> ComboManager:
    mgr = ComboManager.__new__(ComboManager)
    mgr.config_dir = Path(tmpdir)
    mgr.combo_file = Path(tmpdir) / "combos.json"
    mgr.combos = {}
    return mgr


def _write_hooks(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestAllowanceHardening(unittest.TestCase):
    def test_allowance_block_mode_cannot_be_whitelisted(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AllowanceConfig(mode="block", whitelisted_commands=["echo"], blocked_patterns=[])
            g = AllowanceGuard(cfg, workspace=td)
            self.assertEqual(g.evaluate("echo hello"), CommandStatus.BLOCKED)
            self.assertEqual(g.evaluate("echo"), CommandStatus.BLOCKED)


class TestShellHardening(unittest.TestCase):
    def test_shell_guard_none_maps_to_ask_without_exec(self):
        runner = ShellRunner(allowance_guard=None, timeout=5)
        with patch("harness.tools.shell_runner.subprocess.run") as mrun:
            res = runner.run("echo hi")
            mrun.assert_not_called()
        self.assertEqual(res.exit_code, 1)
        self.assertIn("ASK", res.stderr)


class TestHooksHardening(unittest.TestCase):
    def test_hooks_override_path_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "SYSTEM_HOOKS.md"
            _write_hooks(hp, "```yaml\ntype: pre_tool_call\ntarget_tool: read_file\noverride_arg: path\noverride_value: /etc/passwd\n```\n")
            d = HooksDispatcher(str(hp))
            out = d.dispatch_pre_tool_call("read_file", {"path": "orig.txt"})
            self.assertEqual(out.get("path"), "orig.txt")
            self.assertNotEqual(out.get("path"), "/etc/passwd")

    def test_hooks_two_same_tool_both_apply(self):
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "SYSTEM_HOOKS.md"
            _write_hooks(hp, "```yaml\ntype: pre_tool_call\ntarget_tool: mytool\noverride_arg: model\noverride_value: m1\n```\n```yaml\ntype: pre_tool_call\ntarget_tool: mytool\noverride_arg: limit\noverride_value: 5\n```\n")
            d = HooksDispatcher(str(hp))
            self.assertEqual(len(d.hooks.get("pre_tool_call", [])), 2)
            out = d.dispatch_pre_tool_call("mytool", {"other": "x"})
            self.assertEqual(out.get("model"), "m1")
            self.assertEqual(str(out.get("limit")), "5")
            self.assertEqual(out.get("other"), "x")


class TestLoopHardening(unittest.TestCase):
    def _ok_result(self, cmd=""):
        from harness.tools.shell_runner import ShellResult
        return ShellResult(stdout="", stderr="", exit_code=0, command=cmd)

    def test_loop_without_pipe_chaining_which_based(self):
        from harness.tools.shell_runner import ShellResult
        seen = []

        class FakeRunner:
            def run(self, cmd):
                seen.append(cmd)
                return ShellResult(stdout="", stderr="", exit_code=0, command=cmd)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.py"
            p.write_text("x=1\n", encoding="utf-8")
            with patch("harness.verification.loop.shutil.which") as mwhich:
                mwhich.side_effect = lambda name: "/usr/bin/" + name if name == "ruff" else None
                loop = VerificationLoop(FakeRunner(), max_retries=0)
                res = loop.verify([p])
                self.assertTrue(res.passed)
                self.assertTrue(mwhich.called)
        self.assertGreaterEqual(len(seen), 1)
        for cmd in seen:
            self.assertNotIn("||", cmd)
        lint_cmds = [c for c in seen if c.startswith("ruff check") or c.startswith("flake8")]
        self.assertGreaterEqual(len(lint_cmds), 1)

    def test_loop_quotes_path_with_spaces(self):
        from harness.tools.shell_runner import ShellResult
        seen = []

        class FakeRunner:
            def run(self, cmd):
                seen.append(cmd)
                return ShellResult(stdout="", stderr="", exit_code=0, command=cmd)

        spaced = Path("/tmp/my dir/file with space.py")
        expected = f"ruff check {shlex.quote(str(spaced))}"
        with patch("harness.verification.loop.shutil.which") as mwhich:
            mwhich.side_effect = lambda name: "/usr/bin/ruff" if name == "ruff" else None
            loop = VerificationLoop(FakeRunner(), max_retries=0)
            res = loop.verify([spaced])
            self.assertTrue(res.passed)
        lint = [c for c in seen if "ruff check" in c or "flake8" in c]
        self.assertEqual(len(lint), 1)
        self.assertEqual(lint[0], expected)
        self.assertNotIn("||", lint[0])
        # quoting must protect spaces (quotes or backslash)
        self.assertTrue("'" in lint[0] or '"' in lint[0] or "\\ " in lint[0])


class TestComboHardening(unittest.TestCase):
    def test_combo_create_nested_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = _isolated_combo_manager(td)
            with self.assertRaises(ValueError):
                mgr.create_combo("bad", "round_robin", ["combo/other"])

    def test_combo_overwrite_deterministic_single_key(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = _isolated_combo_manager(td)
            mgr.create_combo("dup", "round_robin", ["m1"])
            with self.assertRaises(ValueError):
                mgr.create_combo("dup", "round_robin", ["m2"])
            self.assertEqual(mgr.combos["dup"]["models"], ["m1"])
            mgr.create_combo("dup", "round_robin", ["m2"], overwrite=True)
            self.assertEqual(len(mgr.combos), 1)
            self.assertEqual(mgr.combos["dup"]["models"], ["m2"])

    def test_combo_get_list_consistent_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = _isolated_combo_manager(td)
            mgr.create_combo("c1", "round_robin", ["m1"])
            got = mgr.get_combo("c1")
            self.assertEqual(got["strategy"], "round_robin")
            self.assertEqual(got["models"], ["m1"])
            lst = mgr.list_combos()
            self.assertIn("c1", lst)
            self.assertEqual(lst["c1"], got)
            self.assertEqual(dict(lst), mgr.combos)

    def test_combo_delete_missing_safe_falsy(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = _isolated_combo_manager(td)
            mgr.create_combo("k", "round_robin", ["m1"])
            before = dict(mgr.combos)
            res = mgr.delete_combo("missing-xyz")
            self.assertFalse(res)
            self.assertEqual(mgr.combos, before)
            mgr.delete_combo("k")
            self.assertNotIn("k", mgr.combos)


class TestDiffHardening(unittest.TestCase):
    def test_diff_rejects_path_outside_workspace(self):
        ed = DiffEditor()
        res = ed.apply_diff("/etc/passwd", "<<<< SEARCH\nx\n====\ny\n>>>> REPLACE")
        self.assertFalse(res.success)
        self.assertIn("outside workspace", (res.error or "").lower())

    def test_diff_pure_delete_succeeds(self):
        ed = DiffEditor()
        tdir = tempfile.TemporaryDirectory(dir=str(Path.cwd().resolve()))
        try:
            p = Path(tdir.name) / "t_del.txt"
            p.write_text("hello\nkeep\nworld\n", encoding="utf-8")
            diff1 = "<<<< SEARCH\nkeep\n====\n\n>>>> REPLACE"
            r1 = ed.apply_diff(str(p), diff1)
            self.assertTrue(r1.success, msg=r1.error)
            self.assertNotIn("keep", p.read_text(encoding="utf-8"))
            # DELETE marker variant
            p.write_text("a\nbye\nc\n", encoding="utf-8")
            diff2 = "<<<< SEARCH\nbye\n====\nDELETE\n>>>> REPLACE"
            r2 = ed.apply_diff(str(p), diff2)
            self.assertTrue(r2.success, msg=r2.error)
            self.assertNotIn("bye", p.read_text(encoding="utf-8"))
        finally:
            tdir.cleanup()


class TestOrchestratorTimeout(unittest.TestCase):
    def _make_orch(self, td: str) -> Orchestrator:
        agents = Path(td) / "AGENTS.md"
        agents.write_text("# Allowed\n- x\n", encoding="utf-8")
        hooks = Path(td) / "SYSTEM_HOOKS.md"
        hooks.write_text("", encoding="utf-8")
        return Orchestrator(HooksDispatcher(str(hooks)), AgentsParser(str(agents)))

    def test_wait_for_subagent_has_timeout_and_fires(self):
        sig = inspect.signature(Orchestrator._wait_for_subagent)
        self.assertIn("timeout", sig.parameters)
        self.assertEqual(sig.parameters["timeout"].default, 120.0)

        async def _run():
            with tempfile.TemporaryDirectory() as td:
                orch = self._make_orch(td)
                with self.assertRaises(asyncio.TimeoutError):
                    await orch._wait_for_subagent("nope", timeout=0.05)

        asyncio.run(_run())

    def test_run_task_has_timeout_param(self):
        sig = inspect.signature(Orchestrator.run_task)
        self.assertIn("timeout", sig.parameters)
        self.assertEqual(sig.parameters["timeout"].default, 300.0)

    def test_subagents_basic_spawn_registered(self):
        from harness.core.subagents import MessageBus, SubagentSpawner, SubagentRole
        bus = MessageBus()
        sp = SubagentSpawner(bus)
        sa = sp.spawn(SubagentRole.CODER)
        self.assertIn(sa.id, sp.subagents)
        self.assertIn(sa.id, bus.queues)


if __name__ == "__main__":
    unittest.main()
