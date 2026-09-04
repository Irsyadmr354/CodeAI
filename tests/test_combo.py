import unittest
import os
from unittest.mock import patch, MagicMock

from harness.models.provider_registry import ProviderRegistry
from harness.models.combo import ComboProvider, ComboStrategy, ComboManager
from harness.models.base import ChatMessage

class TestCombo(unittest.TestCase):
    def setUp(self):
        self.registry = ProviderRegistry()

    @patch('harness.models.auth_vault.AuthVault.get_token')
    def test_list_connected_models(self, mock_get_token):
        mock_get_token.side_effect = lambda pid: "fake_token" if pid in ("openai", "gemini") else None
        
        connected = self.registry.list_connected_models()
        self.assertTrue(len(connected) > 0)
        providers = {m["provider"] for m in connected}
        self.assertIn("openai", providers)
        self.assertIn("gemini", providers)
        self.assertNotIn("anthropic", providers)
        
        for m in connected:
            self.assertIn("id", m)
            self.assertIn("provider", m)
            self.assertIn("model", m)

    def _mock_gateway(self, side_effects=None):
        gw = MagicMock()
        if side_effects:
            gw.chat.side_effect = side_effects
        else:
            def default_chat(messages, model, tools=None):
                return {"role": "assistant", "content": f"Response from {model}"}
            gw.chat.side_effect = default_chat
        return gw

    @patch('harness.models.combo.ComboProvider._get_gateway')
    def test_round_robin(self, mock_get_gateway):
        gw = self._mock_gateway()
        mock_get_gateway.return_value = gw
        
        combo = ComboProvider("test", {
            "strategy": ComboStrategy.ROUND_ROBIN,
            "models": ["m1", "m2", "m3"]
        })
        
        r1 = combo.chat([])
        r2 = combo.chat([])
        r3 = combo.chat([])
        r4 = combo.chat([])
        
        self.assertEqual(r1["content"], "Response from m1")
        self.assertEqual(r2["content"], "Response from m2")
        self.assertEqual(r3["content"], "Response from m3")
        self.assertEqual(r4["content"], "Response from m1")

    @patch('harness.models.combo.ComboProvider._get_gateway')
    def test_fastest(self, mock_get_gateway):
        import time
        def delayed_chat(messages, model, tools=None):
            if model == "m1":
                time.sleep(0.2)
                return {"role": "assistant", "content": "Slow m1"}
            elif model == "m2":
                time.sleep(0.01)
                return {"role": "assistant", "content": "Fast m2"}
            return {"role": "assistant", "content": "Unknown"}
            
        gw = self._mock_gateway(delayed_chat)
        mock_get_gateway.return_value = gw
        
        combo = ComboProvider("test", {
            "strategy": ComboStrategy.FASTEST,
            "models": ["m1", "m2"]
        })
        
        r = combo.chat([])
        self.assertEqual(r["content"], "Fast m2")

    @patch('harness.models.combo.ComboProvider._get_gateway')
    def test_cascade(self, mock_get_gateway):
        def failing_chat(messages, model, tools=None):
            if model == "m1":
                raise Exception("m1 failed")
            return {"role": "assistant", "content": f"Response from {model}"}
            
        gw = self._mock_gateway(failing_chat)
        mock_get_gateway.return_value = gw
        
        combo = ComboProvider("test", {
            "strategy": ComboStrategy.CASCADE,
            "models": ["m1", "m2"]
        })
        
        r = combo.chat([])
        self.assertEqual(r["content"], "Response from m2")

    @patch('harness.models.combo.ComboProvider._get_gateway')
    def test_consensus(self, mock_get_gateway):
        def length_chat(messages, model, tools=None):
            if model == "m1":
                return {"role": "assistant", "content": "Short"}
            elif model == "m2":
                return {"role": "assistant", "content": "A bit longer response"}
            elif model == "m3":
                return {"role": "assistant", "content": "Med"}
                
        gw = self._mock_gateway(length_chat)
        mock_get_gateway.return_value = gw
        
        combo = ComboProvider("test", {
            "strategy": ComboStrategy.CONSENSUS,
            "models": ["m1", "m2", "m3"]
        })
        
        r = combo.chat([])
        self.assertEqual(r["content"], "A bit longer response")
        
    @patch('harness.models.combo.ComboProvider._get_gateway')
    def test_pipeline(self, mock_get_gateway):
        def pipeline_chat(messages, model, tools=None):
            if model == "m1":
                return {"role": "assistant", "content": "Draft from m1"}
            elif model == "m2":
                last_msg = messages[-2]["content"]
                self.assertEqual(last_msg, "Draft from m1")
                return {"role": "assistant", "content": "Refined by m2"}
                
        gw = self._mock_gateway(pipeline_chat)
        mock_get_gateway.return_value = gw
        
        combo = ComboProvider("test", {
            "strategy": ComboStrategy.PIPELINE,
            "models": ["m1", "m2"]
        })
        
        r = combo.chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(r["content"], "Refined by m2")

if __name__ == '__main__':
    unittest.main()
