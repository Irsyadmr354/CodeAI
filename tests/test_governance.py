import unittest
import tempfile
from pathlib import Path
from harness.rules.agents_parser import AgentsParser, AgentRules
from harness.core.hooks import HooksDispatcher

class TestGovernance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agents_md = Path(self.temp_dir.name) / "AGENTS.md"
        self.hooks_md = Path(self.temp_dir.name) / "SYSTEM_HOOKS.md"
        
        self.agents_md.write_text(
            "# Allowed\n"
            "- read_file\n"
            "- write_file\n\n"
            "# Forbidden\n"
            "- rm -rf\n"
            "- delete_database\n\n"
            "# Boundaries\n"
            "- Stay within workspace\n"
        )
        
        self.hooks_md.write_text(
            "```yaml\n"
            "type: before_init\n"
            "append_prompt: You are under governance.\n"
            "```\n"
            "```yaml\n"
            "type: pre_tool_call\n"
            "target_tool: read_file\n"
            "override_arg: read_mode\n"
            "override_value: safe\n"
            "```\n"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_agents_parser_extracts_rules(self):
        parser = AgentsParser(self.agents_md)
        rules = parser.parse()
        self.assertIn("read_file", rules.allowed)
        self.assertIn("rm -rf", rules.forbidden)
        self.assertIn("Stay within workspace", rules.boundaries)

    def test_pre_flight_verification_blocks_forbidden(self):
        parser = AgentsParser(self.agents_md)
        rules = parser.parse()
        
        # Safe call
        is_valid = parser.verify_tool_call("run_command", {"cmd": "ls -l"}, rules)
        self.assertTrue(is_valid)
        
        # Forbidden call by tool name
        is_valid = parser.verify_tool_call("delete_database", {}, rules)
        self.assertFalse(is_valid)
        
        # Forbidden call by payload
        is_valid = parser.verify_tool_call("run_command", {"cmd": "rm -rf /"}, rules)
        self.assertFalse(is_valid)

    def test_hooks_dispatcher_lifecycle(self):
        dispatcher = HooksDispatcher(self.hooks_md)
        
        # before_init
        prompt = dispatcher.dispatch_before_init("Initial prompt.")
        self.assertIn("You are under governance.", prompt)
        
        # pre_tool_call
        args = dispatcher.dispatch_pre_tool_call("read_file", {"path": "test.txt"})
        self.assertEqual(args.get("read_mode"), "safe")
        self.assertEqual(args.get("path"), "test.txt")

if __name__ == '__main__':
    unittest.main()
