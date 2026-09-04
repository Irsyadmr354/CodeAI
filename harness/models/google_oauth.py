import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform"
    " https://www.googleapis.com/auth/userinfo.email"
    " openid profile"
    " https://www.googleapis.com/auth/cclog"
    " https://www.googleapis.com/auth/experimentsandconfigs"
)


def _resolve_client_id() -> str:
    """Env-overridable client id (runtime); env -> vault -> ''."""
    env_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    if env_id:
        return env_id
    try:
        from harness.models.auth_vault import AuthVault

        try:
            vault = AuthVault()
            try:
                tok = vault.get_token("google_client_id")
                if tok:
                    return tok
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass
    return CLIENT_ID or ""


def _resolve_client_secret() -> str:
    """Resolve secret: env -> vault -> import-time snapshot. Never raises KeyError."""
    env_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if env_secret:
        return env_secret
    try:
        from harness.models.auth_vault import AuthVault

        try:
            vault = AuthVault()
            try:
                val = vault._get_google_client_secret()
                if val:
                    return val
            except Exception:
                pass
            try:
                tok = vault.get_token("google_client_secret")
                if tok:
                    return tok
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass
    return CLIENT_SECRET or ""

def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (RFC 7636)."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return code_verifier, code_challenge

def _extract_email_from_id_token(id_token: str) -> Optional[str]:
    """Decode JWT payload (base64url) to extract email claim; stdlib only."""
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        email = payload.get("email")
        return email if isinstance(email, str) else None
    except Exception:
        return None


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    code_verifier: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    email: Optional[str] = None
    expected_state: Optional[str] = None
    redirect_uri: str = "http://127.0.0.1:8085/oauth/callback"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/oauth/callback"):
            params = urllib.parse.parse_qs(parsed.query)

            if "error" in params:
                error_msg = params["error"][0]
                self._send_html_response(400, f"<h2>❌ Login Gagal: {error_msg}</h2>")
                return

            # State anti-CSRF: reject mismatch when expected_state is set.
            if OAuthCallbackHandler.expected_state is not None:
                recv_state = params.get("state", [None])[0]
                if recv_state != OAuthCallbackHandler.expected_state:
                    logger.error("OAuth state mismatch: rejecting callback (expected state does not match).")
                    self._send_html_response(400, "<h2>❌ State tidak valid (kemungkinan CSRF).</h2>")
                    return

            if "code" in params:
                auth_code = params["code"][0]
                token_result = self._exchange_code_for_token(auth_code)
                if token_result:
                    OAuthCallbackHandler.access_token = token_result.get("access_token")
                    OAuthCallbackHandler.refresh_token = token_result.get("refresh_token")
                    try:
                        OAuthCallbackHandler.expires_in = token_result.get("expires_in")
                    except Exception:
                        OAuthCallbackHandler.expires_in = None
                    email_val = token_result.get("email")
                    if not email_val and token_result.get("id_token"):
                        email_val = _extract_email_from_id_token(token_result.get("id_token") or "")
                    OAuthCallbackHandler.email = email_val
                    self._send_html_response(
                        200,
                        """<div style="font-family:sans-serif; text-align:center; padding-top:60px; background:#121212; color:#fff; height:100vh;">
                        <h1 style="color:#4CAF50;">✅ Login Berhasil!</h1>
                        <p style="font-size:18px; color:#ddd;">CodeAI telah berhasil meng-intercept OAuth Token Google Antigravity Anda.</p>
                        <p style="color:#888;">Anda dapat menutup jendela browser ini dan kembali ke terminal CodeAI.</p>
                        </div>"""
                    )
                else:
                    self._send_html_response(500, "<h2>❌ Gagal menukar kode otorisasi dengan token.</h2>")
            else:
                self._send_html_response(400, "<h2>❌ Parameter kode otorisasi tidak ditemukan.</h2>")
        else:
            self.send_response(404)
            self.end_headers()

    def _exchange_code_for_token(self, code: str) -> Optional[dict]:
        token_url = "https://oauth2.googleapis.com/token"
        client_secret = _resolve_client_secret()
        if not client_secret:
            logger.error(
                "GOOGLE_CLIENT_SECRET is not set (env GOOGLE_CLIENT_SECRET "
                "or vault key 'google_client_secret'); proceeding with empty "
                "secret, token exchange will likely fail."
            )
        payload = {
            "client_id": _resolve_client_id(),
            "client_secret": client_secret,
            "code": code,
            "code_verifier": OAuthCallbackHandler.code_verifier or "",
            "grant_type": "authorization_code",
            "redirect_uri": OAuthCallbackHandler.redirect_uri
        }
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(token_url, data=data, method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            logger.error(f"Failed to exchange code (HTTP {e.code}): {err_body}")
            return None
        except Exception as e:
            logger.error(f"Failed to exchange code: {e}")
            return None

    def _send_html_response(self, code: int, body: str):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass

class GoogleOAuthInterceptor:
    def __init__(self, port: int = 8085, fallback_port: int = 8086):
        self.port = port
        self.fallback_port = fallback_port

    def get_auth_url(self, port: int, state: Optional[str] = None) -> Tuple[str, str]:
        code_verifier, code_challenge = generate_pkce()
        if state is None:
            state = secrets.token_urlsafe(32)
        redirect_uri = f"http://127.0.0.1:{port}/oauth/callback"
        params = {
            "client_id": _resolve_client_id(),
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
        OAuthCallbackHandler.expected_state = state
        return url, code_verifier

    def get_auth_url_with_state(self, port: int) -> Tuple[str, str, str]:
        """Explicit (url, code_verifier, state) variant; get_auth_url stays 2-tuple compat."""
        code_verifier, code_challenge = generate_pkce()
        state = secrets.token_urlsafe(32)
        redirect_uri = f"http://127.0.0.1:{port}/oauth/callback"
        params = {
            "client_id": _resolve_client_id(),
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
        OAuthCallbackHandler.expected_state = state
        return url, code_verifier, state

    def intercept(self, timeout: int = 120) -> Optional[Dict[str, Any]]:
        server = None
        used_port = self.port
        try:
            # Bind loopback only.
            server = socketserver.TCPServer(("127.0.0.1", self.port), OAuthCallbackHandler)
        except socket.error:
            used_port = self.fallback_port
            try:
                # Bind loopback only.
                server = socketserver.TCPServer(("127.0.0.1", self.fallback_port), OAuthCallbackHandler)
            except socket.error:
                return None

        # Reset previous flow state.
        OAuthCallbackHandler.access_token = None
        OAuthCallbackHandler.refresh_token = None
        OAuthCallbackHandler.expires_in = None
        OAuthCallbackHandler.email = None
        OAuthCallbackHandler.code_verifier = None
        OAuthCallbackHandler.expected_state = None

        oauth_url, code_verifier = self.get_auth_url(used_port)
        OAuthCallbackHandler.code_verifier = code_verifier
        OAuthCallbackHandler.redirect_uri = f"http://127.0.0.1:{used_port}/oauth/callback"

        print(f"\n🌐 Membuka Google Login di browser...")
        try:
            webbrowser.open(oauth_url)
        except Exception:
            pass

        print(f"Jika browser tidak otomatis terbuka, buka tautan ini:\n{oauth_url}\n")
        print(f"Menunggu intercept token di http://127.0.0.1:{used_port} (timeout {timeout}s)...")

        def run_server():
            while OAuthCallbackHandler.access_token is None:
                server.handle_request()

        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        try:
            if OAuthCallbackHandler.access_token is None:
                return None
            result: Dict[str, Any] = {
                "access_token": OAuthCallbackHandler.access_token,
                "refresh_token": OAuthCallbackHandler.refresh_token,
                "expires_in": OAuthCallbackHandler.expires_in,
            }
            if OAuthCallbackHandler.email:
                result["email"] = OAuthCallbackHandler.email
            return result
        finally:
            try:
                server.server_close()
            except Exception:
                pass

def intercept_google_oauth(timeout: int = 120) -> Optional[str]:
    """Legacy compat wrapper: returns access_token string only."""
    result = GoogleOAuthInterceptor().intercept(timeout)
    if isinstance(result, dict):
        return result.get("access_token")
    return result
