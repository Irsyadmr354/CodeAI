import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from harness.models.auth_vault import AuthVault
from harness.models.base import ChatMessage
from harness.models.providers.copilot import CopilotProvider
from harness.models.providers.gemini import GeminiProvider

class TestSubscriptionProviders(unittest.TestCase):

    def test_auth_vault_discovery(self):
        vault = AuthVault()
        with patch.object(vault, 'get_token', return_value="vault_token"):
            self.assertEqual(vault.discover_copilot_token(), "vault_token")

    def test_copilot_provider_headers(self):
        provider = CopilotProvider()
        with patch.object(provider, '_get_session_token', return_value="sess_tok"):
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps({
                    "choices": [{"message": {"content": "Hello"}}]
                }).encode('utf-8')
                mock_urlopen.return_value.__enter__.return_value = mock_response
                
                res = provider.chat([ChatMessage(role="user", content="Hi")])
                self.assertEqual(res["content"], "Hello")
                
                call_args = mock_urlopen.call_args
                req = call_args[0][0]
                self.assertEqual(req.headers.get("Authorization"), "Bearer sess_tok")
                self.assertEqual(req.headers.get("Copilot-integration-id"), "vscode-chat")

    def test_gemini_provider_auth_oauth(self):
        with patch.dict(os.environ, clear=True):
            provider = GeminiProvider()
            with patch.object(provider.vault, 'discover_gemini_token', return_value="ya29.adc_token"):
                with patch('subprocess.run') as mock_run:
                    mock_proc = MagicMock()
                    mock_proc.returncode = 0
                    mock_proc.stdout = "Gemini response via agy"
                    mock_proc.stderr = ""
                    mock_run.return_value = mock_proc
                    
                    res = provider.chat([ChatMessage(role="user", content="Hi")])
                    self.assertEqual(res["content"], "Gemini response via agy")
                    self.assertIsNone(res["tool_calls"])
                    
                    call_args = mock_run.call_args[0][0]
                    self.assertEqual(call_args[0], "agy")
                    self.assertEqual(call_args[1], "--print")
                    self.assertEqual(call_args[2], "Hi")
                    self.assertEqual(call_args[3], "--model")
                    self.assertEqual(call_args[4], "gemini-2.5-flash")
                    self.assertEqual(call_args[5], "--disable-slash-commands")

    def test_gemini_provider_auth_apikey(self):
        with patch.dict(os.environ, clear=True):
            provider = GeminiProvider()
            with patch.object(provider.vault, 'discover_gemini_token', return_value="AIza_test_key"):
                with patch('urllib.request.urlopen') as mock_urlopen:
                    mock_response = MagicMock()
                    mock_response.read.return_value = json.dumps({
                        "candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}]
                    }).encode('utf-8')
                    mock_urlopen.return_value.__enter__.return_value = mock_response
                    
                    res = provider.chat([ChatMessage(role="user", content="Hi")])
                    self.assertEqual(res["content"], "Gemini response")
                    
                    call_args = mock_urlopen.call_args
                    req = call_args[0][0]
                    self.assertIn("?key=AIza_test_key", req.full_url)

    def test_auth_vault_antigravity_discovery(self):
        vault = AuthVault()
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=False), \
             patch('builtins.open', unittest.mock.mock_open(read_data='{"token": "ya29.antigravity"}')):
            self.assertEqual(vault.discover_antigravity_token(), "ya29.antigravity")

    def test_gemini_provider_auth_antigravity(self):
        with patch.dict(os.environ, clear=True):
            provider = GeminiProvider()
            with patch.object(provider.vault, 'discover_gemini_token', return_value="ya29.antigravity"):
                with patch('subprocess.run') as mock_run:
                    mock_proc = MagicMock()
                    mock_proc.returncode = 0
                    mock_proc.stdout = "[Subagent: Coder]\n=== Subagent: Task ===\nAntigravity response"
                    mock_proc.stderr = ""
                    mock_run.return_value = mock_proc
                    
                    res = provider.chat([
                        ChatMessage(role="system", content="System prompt"),
                        ChatMessage(role="user", content="User prompt")
                    ])
                    self.assertEqual(res["content"], "Antigravity response")
                    self.assertIsNone(res["tool_calls"])
                    
                    call_args = mock_run.call_args[0][0]
                    self.assertEqual(call_args[0], "agy")
                    self.assertEqual(call_args[2], "system: System prompt\nuser: User prompt")

    def test_gemini_provider_auth_agy_errors(self):
        from harness.models.base import ProviderError, TimeoutError
        import subprocess
        with patch.dict(os.environ, clear=True):
            provider = GeminiProvider()
            with patch.object(provider.vault, 'discover_gemini_token', return_value="ya29.antigravity"):
                with patch('subprocess.run', side_effect=FileNotFoundError):
                    with self.assertRaises(ProviderError):
                        provider.chat([ChatMessage(role="user", content="Hi")])

                with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="agy", timeout=60)):
                    with self.assertRaises(TimeoutError):
                        provider.chat([ChatMessage(role="user", content="Hi")])

                mock_proc = MagicMock()
                mock_proc.returncode = 1
                mock_proc.stderr = "Command failed"
                mock_proc.stdout = ""
                with patch('subprocess.run', return_value=mock_proc):
                    with self.assertRaises(ProviderError):
                        provider.chat([ChatMessage(role="user", content="Hi")])

    def test_google_oauth_constants(self):
        from harness.models.google_oauth import CLIENT_ID, CLIENT_SECRET, SCOPES, _resolve_client_id, _resolve_client_secret
        TEST_ONLY_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "TEST-ONLY-CLIENT-ID")
        # CLIENT_ID is env-overridable; import-time snapshot reflects env (empty when unset).
        expected_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self.assertEqual(CLIENT_ID, expected_id)
        self.assertEqual(_resolve_client_id(), expected_id)
        # Resolver is env-overridable to TEST-ONLY value (no real secret).
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": TEST_ONLY_CLIENT_ID}):
            self.assertEqual(_resolve_client_id(), TEST_ONLY_CLIENT_ID)
        # CLIENT_SECRET is env/vault-only (never hardcoded literal); import-time snapshot reflects env.
        self.assertEqual(CLIENT_SECRET, os.environ.get("GOOGLE_CLIENT_SECRET", ""))
        self.assertNotEqual(CLIENT_SECRET, "REDACTED-TEST-SECRET")
        self.assertNotIn("REDACTED-", CLIENT_SECRET)
        # Resolver is env-overridable.
        with patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "dummy-test-secret"}):
            self.assertEqual(_resolve_client_secret(), "dummy-test-secret")
        # SCOPES per spec publik Antigravity Cloud Code: 4 scope lama + cclog + experimentsandconfigs.
        self.assertIn("https://www.googleapis.com/auth/cloud-platform", SCOPES)
        self.assertIn("https://www.googleapis.com/auth/userinfo.email", SCOPES)
        self.assertIn("openid", SCOPES)
        self.assertIn("profile", SCOPES)
        self.assertIn("https://www.googleapis.com/auth/cclog", SCOPES)
        self.assertIn("https://www.googleapis.com/auth/experimentsandconfigs", SCOPES)

    def test_google_oauth_exchange_code_payload(self):
        import urllib.parse
        import urllib.error
        from harness.models.google_oauth import OAuthCallbackHandler, _resolve_client_id, _resolve_client_secret
        handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
        OAuthCallbackHandler.code_verifier = "test_verifier"
        OAuthCallbackHandler.redirect_uri = "http://localhost:8085/oauth/callback"

        TEST_ONLY_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "TEST-ONLY-CLIENT-ID")
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": TEST_ONLY_CLIENT_ID, "GOOGLE_CLIENT_SECRET": "dummy-test-secret"}):
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps({"access_token": "ya29.new_token"}).encode("utf-8")
                mock_urlopen.return_value.__enter__.return_value = mock_resp

                result = handler._exchange_code_for_token("auth_test_code")
                self.assertEqual(result, {"access_token": "ya29.new_token"})

                req = mock_urlopen.call_args[0][0]
                self.assertEqual(req.full_url, "https://oauth2.googleapis.com/token")
                parsed_data = urllib.parse.parse_qs(req.data.decode("utf-8"))
                self.assertEqual(parsed_data["client_id"][0], _resolve_client_id())
                self.assertEqual(parsed_data["client_secret"][0], "dummy-test-secret")
                self.assertEqual(parsed_data["client_secret"][0], _resolve_client_secret())
                self.assertEqual(parsed_data["code"][0], "auth_test_code")
                self.assertEqual(parsed_data["grant_type"][0], "authorization_code")

            # Test HTTPError handling cleanly
            with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError("https://oauth2.googleapis.com/token", 400, "Bad Request", {}, io.BytesIO(b'{"error": "invalid_grant"}'))):
                err_result = handler._exchange_code_for_token("bad_code")
                self.assertIsNone(err_result)

    def test_auth_vault_antigravity_token_refresh(self):
        import urllib.parse
        vault = AuthVault()
        expired_token_data = {
            "token": {
                "access_token": "ya29.old_expired",
                "refresh_token": "1//refresh_token_xyz",
                "expiry": "2020-01-01T00:00:00+00:00"
            }
        }
        mock_file = unittest.mock.mock_open(read_data=json.dumps(expired_token_data))

        TEST_ONLY_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "TEST-ONLY-CLIENT-ID")
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": TEST_ONLY_CLIENT_ID, "GOOGLE_CLIENT_SECRET": "dummy-test-secret"}), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=False), \
             patch('builtins.open', mock_file), \
             patch('urllib.request.urlopen') as mock_urlopen:

            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "access_token": "ya29.refreshed_new_token",
                "expires_in": 3600
            }).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            token = vault.discover_antigravity_token()
            self.assertEqual(token, "ya29.refreshed_new_token")
            # After token refresh, discover returns the fresh token.
            # Vault.credentials only reflects explicitly stored tokens,
            # not runtime-discovered ones (storage happens via /login).
            # What matters is the returned token and that the file was rewritten.

            # Check that request payload had client_id, client_secret, refresh_token, grant_type="refresh_token"
            req = mock_urlopen.call_args[0][0]
            parsed_payload = urllib.parse.parse_qs(req.data.decode("utf-8"))
            self.assertEqual(parsed_payload["client_id"][0], TEST_ONLY_CLIENT_ID)
            self.assertEqual(parsed_payload["client_secret"][0], "dummy-test-secret")
            self.assertEqual(parsed_payload["refresh_token"][0], "1//refresh_token_xyz")
            self.assertEqual(parsed_payload["grant_type"][0], "refresh_token")

            # Check that file was written back
            mock_file().write.assert_called()

    def test_handle_login_gemini_apikey_only(self):
        """Gemini login now only accepts an AI Studio API key (no more OAuth option)."""
        from harness.cli import CodeAICLI
        from unittest.mock import patch, MagicMock
        cli = CodeAICLI("dummy.yaml")
        mock_gateway = MagicMock()
        mock_gateway.config = MagicMock()
        cli.orchestrator = MagicMock()
        cli.orchestrator.gateway = mock_gateway

        with patch("webbrowser.open"):
            # "AIza_valid_key" = the key, "y" = confirm switch
            with patch("harness.cli.Prompt.ask", side_effect=["AIza_valid_key", "y"]):
                with patch("harness.models.auth_vault.AuthVault.store_token") as mock_store:
                    cli.handle_login("gemini")
                    mock_store.assert_called_once_with("gemini", "AIza_valid_key")
                    self.assertEqual(cli.config.provider.default, "gemini")
                    self.assertEqual(cli.config.provider.active_model, "gemini-2.5-flash")
                    self.assertEqual(mock_gateway.config.default, "gemini")
                    self.assertEqual(mock_gateway.active_model, "gemini-2.5-flash")

    def test_handle_login_antigravity_detected_session(self):
        """Antigravity login detects existing agy session and registers it."""
        from harness.cli import CodeAICLI
        from unittest.mock import patch, MagicMock
        from harness.models.providers.antigravity import DEFAULT_ANTIGRAVITY_MODEL
        cli = CodeAICLI("dummy.yaml")

        with patch("shutil.which", return_value="/usr/bin/agy"):
            with patch("harness.models.auth_vault.AuthVault.discover_antigravity_token", return_value="ya29.test_token"):
                with patch("harness.models.auth_vault.AuthVault.store_token") as mock_store:
                    with patch("harness.cli.Prompt.ask", return_value="y"):
                        cli.handle_login("antigravity")
                        mock_store.assert_called_once_with("antigravity", "ya29.test_token")
                        self.assertEqual(cli.config.provider.default, "antigravity")
                        self.assertEqual(cli.config.provider.active_model, DEFAULT_ANTIGRAVITY_MODEL)

    def test_handle_login_google_oauth_tokeninfo_validation(self):
        """Legacy: kept for compatibility — now tests antigravity store_token flow."""
        from harness.cli import CodeAICLI
        from unittest.mock import patch, MagicMock
        from harness.models.providers.antigravity import DEFAULT_ANTIGRAVITY_MODEL
        cli = CodeAICLI("dummy.yaml")

        with patch("shutil.which", return_value="/usr/bin/agy"):
            with patch("harness.models.auth_vault.AuthVault.discover_antigravity_token", return_value="ya29.test_oauth_token"):
                with patch("harness.models.auth_vault.AuthVault.store_token") as mock_store:
                    with patch("harness.cli.Prompt.ask", return_value="y"):
                        cli.handle_login("antigravity")
                        mock_store.assert_called_once_with("antigravity", "ya29.test_oauth_token")
                        self.assertEqual(cli.config.provider.default, "antigravity")
                        self.assertEqual(cli.config.provider.active_model, DEFAULT_ANTIGRAVITY_MODEL)

    def test_handle_login_gemini_option1(self):
        """Gemini API key login updates gateway active_model correctly."""
        from harness.cli import CodeAICLI
        from unittest.mock import patch, MagicMock
        cli = CodeAICLI("dummy.yaml")
        mock_gateway = MagicMock()
        mock_gateway.config = MagicMock()
        cli.orchestrator = MagicMock()
        cli.orchestrator.gateway = mock_gateway

        with patch("webbrowser.open"):
            # Prompt receives key then switch confirmation
            with patch("harness.cli.Prompt.ask", side_effect=["AIza_valid_key", "y"]):
                with patch("harness.models.auth_vault.AuthVault.store_token") as mock_store:
                    cli.handle_login("gemini")
                    mock_store.assert_called_once_with("gemini", "AIza_valid_key")
                    self.assertEqual(cli.config.provider.default, "gemini")
                    self.assertEqual(cli.config.provider.active_model, "gemini-2.5-flash")
                    self.assertEqual(mock_gateway.config.default, "gemini")
                    self.assertEqual(mock_gateway.active_model, "gemini-2.5-flash")

if __name__ == '__main__':
    unittest.main()
