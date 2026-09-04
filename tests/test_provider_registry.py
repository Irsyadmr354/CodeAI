import unittest
import os
from harness.models.provider_registry import ProviderRegistry
from harness.models.providers.universal_openai import UniversalOpenAIProvider
from harness.models.gateway import LLMGateway
from harness.config import ProviderConfig

class TestProviderRegistry(unittest.TestCase):
    def test_registry_loading(self):
        registry = ProviderRegistry()
        providers = registry.list_providers()
        
        # Embedded providers check
        embedded_ids = [p["id"] for p in providers]
        self.assertIn("openai", embedded_ids)
        self.assertIn("deepseek", embedded_ids)
        
        # Check parse_model_string
        provider, model = registry.parse_model_string("deepseek/deepseek-chat")
        self.assertEqual(provider, "deepseek")
        self.assertEqual(model, "deepseek-chat")
        
        provider, model = registry.parse_model_string("gpt-4o")
        # Bare strings resolve to BARE_DEFAULT_PROVIDER=anthropic (provider_registry.py:36).
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "gpt-4o")

    def test_universal_provider_formatting(self):
        provider = UniversalOpenAIProvider(
            base_url="https://api.deepseek.com/v1",
            api_key="test_key",
            default_model="deepseek-chat"
        )
        
        self.assertEqual(provider.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(provider.api_key, "test_key")
        self.assertEqual(provider.default_model, "deepseek-chat")
        self.assertTrue(provider.supports_tools())

    def test_gateway_dynamic_instantiation(self):
        config = ProviderConfig(default="deepseek", failover_order=[])
        
        # Setup test env variable to bypass credentials check
        os.environ["DEEPSEEK_API_KEY"] = "dummy_key"
        try:
            gateway = LLMGateway(config)
            provider = gateway.get_provider("deepseek")
            
            self.assertIsInstance(provider, UniversalOpenAIProvider)
            self.assertTrue(provider.base_url.startswith("https://api.deepseek.com"))
            self.assertEqual(provider.api_key, "dummy_key")
        finally:
            del os.environ["DEEPSEEK_API_KEY"]

    def test_opencode_default_model_and_fallback(self):
        # When default_model is empty on opencode endpoint
        provider = UniversalOpenAIProvider(
            base_url="https://opencode.ai/zen/v1",
            api_key="opencode_key"
        )
        self.assertEqual(provider.default_model, "mimo-v2.5-free")

        # When default_model is the problematic muse-spark model
        provider_spark = UniversalOpenAIProvider(
            base_url="https://opencode.ai/zen/v1",
            api_key="opencode_key",
            default_model="muse-spark-1.3-contributor-free"
        )
        self.assertEqual(provider_spark.default_model, "mimo-v2.5-free")

    def test_config_opencode_model_field(self):
        config = ProviderConfig()
        self.assertEqual(config.opencode_model, "mimo-v2.5-free")

    def test_cli_get_active_info_custom_model(self):
        from harness.cli import CodeAICLI
        from harness.config import CodeAIConfig, ProviderConfig
        cli = CodeAICLI("test_config.yaml")
        cli.config = CodeAIConfig(provider=ProviderConfig(default="opencode", active_model="claude-3-5-sonnet"))
        provider, model = cli.get_active_info()
        self.assertEqual(provider, "opencode")
        self.assertEqual(model, "claude-3-5-sonnet")

    def test_cli_switch_model_updates_gateway(self):
        from harness.cli import CodeAICLI
        from harness.config import CodeAIConfig, ProviderConfig
        from unittest.mock import MagicMock, patch
        cli = CodeAICLI("test_config.yaml")
        cli.config = CodeAIConfig(provider=ProviderConfig(default="openai"))
        mock_gateway = MagicMock()
        mock_gateway.config = ProviderConfig(default="openai")
        cli.orchestrator = MagicMock()
        cli.orchestrator.gateway = mock_gateway

        with patch("harness.models.provider_registry.ProviderRegistry.list_connected_models", return_value=[{"id": "deepseek/deepseek-chat", "provider": "deepseek", "model": "deepseek-chat"}]):
            cli.switch_model("deepseek/deepseek-chat")
            self.assertEqual(cli.config.provider.default, "deepseek")
            self.assertEqual(cli.config.provider.active_model, "deepseek-chat")
            self.assertEqual(mock_gateway.active_model, "deepseek-chat")
            self.assertEqual(mock_gateway.config.default, "deepseek")
            self.assertEqual(mock_gateway.config.active_model, "deepseek-chat")

    def test_gateway_chat_updates_universal_provider_default_model(self):
        from harness.models.base import ChatMessage
        from unittest.mock import MagicMock, patch
        config = ProviderConfig(default="deepseek", failover_order=[], active_model="deepseek-coder")
        os.environ["DEEPSEEK_API_KEY"] = "dummy_key"
        try:
            gateway = LLMGateway(config)
            provider = gateway.get_provider("deepseek")
            original_default = provider.default_model
            seen_models = []
            def _spy_chat(messages, tools=None):
                # Capture injected model during gateway.chat (before snapshot/restore).
                seen_models.append(provider.default_model)
                return {"role": "assistant", "content": "ok"}
            with patch.object(provider, "chat", side_effect=_spy_chat):
                gateway.chat([ChatMessage(role="user", content="hello")])
                # Injection applied during call, then snapshot/restore reverts state.
                self.assertEqual(seen_models[-1], "deepseek-coder")
                self.assertEqual(provider.default_model, original_default)

                gateway.chat([ChatMessage(role="user", content="hello")], model="deepseek-custom")
                self.assertEqual(seen_models[-1], "deepseek-custom")
                self.assertEqual(provider.default_model, original_default)
        finally:
            del os.environ["DEEPSEEK_API_KEY"]

if __name__ == '__main__':
    unittest.main()
