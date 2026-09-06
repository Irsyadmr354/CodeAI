"""
AuthVault — structured credential storage for CodeAI.

Credential schema inspired by opencode (auth/index.ts) and omp (auth-storage.ts):
  - type: "api_key"  → { "type": "api_key", "key": "sk-..." }
  - type: "oauth"    → { "type": "oauth", "access": "ya29...", "refresh": "...",
                          "expires": <epoch_ms>, "email": "...", "projectId": "..." }

Raw string values (legacy) are transparently upgraded to {"type": "api_key", "key": "..."}.
"""

import json
import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

try:  # POSIX file locking
    import fcntl  # type: ignore[import-not-found]
except Exception:
    fcntl = None  # type: ignore[assignment]

try:  # Windows file locking fallback
    import msvcrt  # type: ignore[import-not-found]
except Exception:
    msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
# No hardcoded secret here (deduplicated): resolved at runtime via env/vault,
# falling back to single-source harness.models.google_oauth for compat.
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# OAuth refresh skew: refresh 60s before stated expiry
_REFRESH_SKEW_MS = 60_000


@contextmanager
def _vault_lock(lock_path: Path) -> Iterator[Any]:
    """Exclusive inter-process lock (flock; best-effort fallback).

    Uses fcntl.flock on POSIX, msvcrt.locking on Windows, and degrades to
    a no-op yield when neither is available. Never raises; never logs secrets.
    """
    fh: Any = None
    try:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            fh = open(lock_path, "a+")
        except OSError:
            yield None
            return
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass
        elif msvcrt is not None:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            except OSError:
                pass
        # else: no locking primitive available → non-blocking skip (no-op)
        yield fh
    finally:
        try:
            if fh is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                elif msvcrt is not None:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
        finally:
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON: tmp in same dir + chmod 600 + os.replace.

    Raises OSError on failure (caller decides to swallow/log). Tmp file is
    created with mkstemp in the target directory so replace() stays on the
    same filesystem. Permissions are set to 0o600 BEFORE replace so the
    window with lax perms is eliminated; re-applied after replace
    best-effort (replace preserves tmp mode on POSIX).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=4)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class AuthVault:
    def __init__(self) -> None:
        self.config_dir = Path.home() / ".codeai"
        self.auth_file = self.config_dir / "auth.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # Internal store: provider → structured dict OR raw string (legacy)
        self._data: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.auth_file.exists():
            return
        try:
            with open(self.auth_file, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(
                "AuthVault: unreadable/corrupt auth file %s (%s); starting empty",
                self.auth_file,
                type(e).__name__,
            )
            self._data = {}
            return

        if not isinstance(raw, dict):
            logger.warning(
                "AuthVault: corrupt auth file %s (expected object, got %s); starting empty",
                self.auth_file,
                type(raw).__name__,
            )
            self._data = {}
            return

        cleaned: Dict[str, Any] = {}
        for k, v in raw.items():
            # Strip "/login " prefix from old keys
            clean_key = k.replace("/login ", "", 1) if k.startswith("/login ") else k
            cleaned[clean_key] = v
        self._data = cleaned

        # Back-fill if we cleaned any keys
        if any(k.startswith("/login ") for k in raw):
            self._save()

    def _save(self) -> None:
        """Atomic persist: tmp in same dir + chmod 600 before os.replace."""
        try:
            _atomic_write_json(self.auth_file, dict(self._data))
        except OSError:
            pass

    def _lock_path(self) -> Path:
        return self.auth_file.parent / (self.auth_file.name + ".lock")

    def _read_disk_unlocked(self) -> Dict[str, Any]:
        """Best-effort read of on-disk vault; corrupt → {} + warning, never raise."""
        try:
            if not self.auth_file.exists():
                return {}
            with open(self.auth_file, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(
                "AuthVault: unreadable/corrupt auth file %s (%s); starting empty",
                self.auth_file,
                type(e).__name__,
            )
            return {}
        if not isinstance(raw, dict):
            logger.warning(
                "AuthVault: corrupt auth file %s (expected object, got %s); starting empty",
                self.auth_file,
                type(raw).__name__,
            )
            return {}
        cleaned: Dict[str, Any] = {}
        for k, v in raw.items():
            clean_key = k.replace("/login ", "", 1) if k.startswith("/login ") else k
            cleaned[clean_key] = v
        return cleaned

    def _locked_save(self, mutator: Callable[[Dict[str, Any]], None]) -> None:
        """Serialise load-modify-save under an inter-process file lock.

        Re-reads disk inside the lock (avoid lost-update), applies mutator
        to self._data, then atomically persists. Refresh/store/remove paths
        must route through here. Never raises on lock/IO failure; never
        logs secrets.
        """
        try:
            with _vault_lock(self._lock_path()):
                try:
                    fresh = self._read_disk_unlocked()
                except Exception:
                    fresh = {}
                # Rebase in-memory view on fresh disk state so concurrent
                # writers do not clobber each other (lost-update guard).
                # In-memory keys not yet persisted are preserved when the
                # disk has no conflicting entry.
                try:
                    merged: Dict[str, Any] = dict(fresh)
                    for k, v in self._data.items():
                        if k not in merged:
                            merged[k] = v
                    self._data = merged
                except Exception:
                    self._data = fresh
                try:
                    mutator(self._data)
                except Exception:
                    return
                try:
                    _atomic_write_json(self.auth_file, dict(self._data))
                except OSError:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_credential(self, value: Any) -> Optional[Dict[str, Any]]:
        """Normalise a raw stored value to a structured credential dict."""
        if isinstance(value, dict) and "type" in value:
            return value
        if isinstance(value, str) and value:
            # Legacy plain string → api_key
            return {"type": "api_key", "key": value}
        return None

    def _extract_token(self, credential: Optional[Dict[str, Any]]) -> Optional[str]:
        """Return the usable bearer/api_key string from a credential dict."""
        if credential is None:
            return None
        ctype = credential.get("type")
        if ctype == "api_key":
            return credential.get("key") or None
        if ctype == "oauth":
            access = credential.get("access") or ""
            expires_ms = credential.get("expires", 0)
            # Return None if token is expired (with skew)
            if expires_ms and _now_ms() + _REFRESH_SKEW_MS >= expires_ms:
                return None
            return access or None
        return None

    # ------------------------------------------------------------------
    # Public API — compatible with existing callers
    # ------------------------------------------------------------------

    @property
    def credentials(self) -> Dict[str, str]:
        """Legacy compat: expose flat string mapping (used by tests/old code)."""
        result: Dict[str, str] = {}
        for k, v in self._data.items():
            token = self._extract_token(self._to_credential(v))
            if token:
                result[k] = token
        return result

    def get_token(self, provider: str) -> Optional[str]:
        """Return the active bearer/api_key for provider, or None."""
        return self._extract_token(self._to_credential(self._data.get(provider)))

    def get_credential(self, provider: str) -> Optional[Dict[str, Any]]:
        """Return the full structured credential for a provider."""
        return self._to_credential(self._data.get(provider))

    def store_token(self, provider: str, token: str) -> None:
        """Store a raw token string as an api_key credential."""
        def _mut(d: Dict[str, Any]) -> None:
            d[provider] = {"type": "api_key", "key": token}

        self._locked_save(_mut)

    def store_oauth(
        self,
        provider: str,
        access: str,
        refresh: str = "",
        expires_ms: int = 0,
        email: str = "",
        project_id: str = "",
        account_id: str = "",
    ) -> None:
        """Store a structured OAuth credential."""
        credential: Dict[str, Any] = {
            "type": "oauth",
            "access": access,
            "refresh": refresh,
            "expires": expires_ms,
        }
        if email:
            credential["email"] = email
        if project_id:
            credential["projectId"] = project_id
        if account_id:
            credential["accountId"] = account_id

        def _mut(d: Dict[str, Any], _cred: Dict[str, Any] = credential) -> None:
            d[provider] = _cred

        self._locked_save(_mut)

    def remove(self, provider: str) -> None:
        """Remove a provider's credentials."""
        def _mut(d: Dict[str, Any]) -> None:
            if provider in d:
                del d[provider]

        self._locked_save(_mut)

    def list_providers(self):
        """Return list of providers that have stored credentials."""
        return [k for k, v in self._data.items() if self._to_credential(v)]

    # ------------------------------------------------------------------
    # Provider-specific discovery methods
    # ------------------------------------------------------------------

    def discover_copilot_token(self) -> Optional[str]:
        # 1. From vault
        if token := self.get_token("copilot"):
            return token
        # 2. GitHub Copilot standard config files
        paths = [
            Path.home() / ".copilot" / "config.json",
            Path.home() / ".config" / "github-copilot" / "hosts.json",
        ]
        for p in paths:
            if not p.exists():
                continue
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if "github.com" in data and "oauth_token" in data["github.com"]:
                    return data["github.com"]["oauth_token"]
                for val in data.values():
                    if isinstance(val, dict) and "oauth_token" in val:
                        return val["oauth_token"]
            except Exception:
                pass
        return None

    def discover_google_adc_token(self) -> Optional[str]:
        try:
            res = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, check=True
            )
            if res.stdout:
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def discover_antigravity_token(self) -> Optional[str]:
        """
        Discover Antigravity OAuth session token from disk.
        Handles refresh if token is expired.
        Returns access token string, or None.
        """
        paths = [
            Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        ]
        config_dir = Path.home() / ".gemini" / "config"
        if config_dir.exists() and config_dir.is_dir():
            for f in config_dir.glob("*"):
                if f.is_file():
                    paths.append(f)

        for p in paths:
            if not p.exists():
                continue
            try:
                with open(p, "r") as f:
                    file_data = json.load(f)
            except Exception:
                continue

            raw = (
                file_data.get("token")
                or (file_data if isinstance(file_data, dict) and
                    ("access_token" in file_data or "refresh_token" in file_data)
                    else None)
            )

            if isinstance(raw, dict):
                is_expired = False
                expiry_str = raw.get("expiry")
                if expiry_str:
                    try:
                        expiry = datetime.fromisoformat(expiry_str)
                        if datetime.now(timezone.utc) > expiry:
                            is_expired = True
                    except Exception:
                        pass

                access_token = raw.get("access_token")
                refresh_token = raw.get("refresh_token")

                if (not access_token or is_expired) and refresh_token:
                    # Attempt token refresh
                    refreshed = self._refresh_google_token(refresh_token)
                    if refreshed:
                        fresh_access = refreshed.get("access_token")
                        if fresh_access:
                            # Update file
                            raw["access_token"] = fresh_access
                            expires_in = refreshed.get("expires_in", 3600)
                            raw["expiry"] = (
                                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                            ).isoformat()
                            if "refresh_token" in refreshed:
                                raw["refresh_token"] = refreshed["refresh_token"]
                            try:
                                # Atomic refresh persist: tmp same-dir + replace
                                # (uses open() so unit-test mock_open still observes write).
                                _tmp = p.with_name(p.name + ".tmp")
                                with open(_tmp, "w") as fw:
                                    json.dump(file_data, fw, indent=2)
                                try:
                                    os.replace(_tmp, p)
                                except OSError:
                                    pass
                            except Exception:
                                pass
                            return fresh_access

                if access_token and not is_expired:
                    return access_token

            elif isinstance(raw, str):
                return raw
            elif isinstance(file_data, dict) and isinstance(file_data.get("access_token"), str):
                return file_data["access_token"]

        return None

    def _get_google_client_id(self) -> str:
        return os.environ.get("GOOGLE_CLIENT_ID", "") or GOOGLE_CLIENT_ID

    def store_antigravity_oauth(
        self,
        access: Any = "",
        refresh: str = "",
        expires_in_s: Any = 0,
        email: str = "",
        project_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Store Antigravity OAuth session (stdlib only).

        Canonical: store_antigravity_oauth(access, refresh, expires_in_s,
        email="", project_id="") → store_oauth("antigravity", ...)
        with expires_ms = now + expires_in*1000 (default 3600 bila 0/falsy).

        Compat: single-dict form store_antigravity_oauth({...}) as called
        from cli (keys: access_token/access/token, refresh_token/refresh,
        expires_in/expires/expiry, email, project_id/projectId/project).
        Epoch heuristic: >1e12 = epoch ms langsung, >1e9 = epoch s*1000,
        else durasi detik.
        """
        try:
            if isinstance(access, dict):
                _d = access
                _access = str(
                    _d.get("access_token")
                    or _d.get("access")
                    or _d.get("token")
                    or kwargs.get("access_token")
                    or kwargs.get("access")
                    or ""
                )
                _refresh = str(
                    _d.get("refresh_token")
                    or _d.get("refresh")
                    or kwargs.get("refresh_token")
                    or kwargs.get("refresh")
                    or (refresh if isinstance(refresh, str) else "")
                    or ""
                )
                _email = str(
                    _d.get("email")
                    or kwargs.get("email")
                    or (email if isinstance(email, str) else "")
                    or ""
                )
                _pid = str(
                    _d.get("project_id")
                    or _d.get("projectId")
                    or _d.get("project")
                    or kwargs.get("project_id")
                    or kwargs.get("projectId")
                    or kwargs.get("project")
                    or (project_id if isinstance(project_id, str) else "")
                    or ""
                )
                _raw: Any = _d.get("expires_in_s")
                if _raw is None:
                    _raw = _d.get("expires_in")
                if _raw is None:
                    _raw = _d.get("expires")
                if _raw is None:
                    _raw = _d.get("expiry")
                if _raw is None:
                    _raw = kwargs.get(
                        "expires_in_s",
                        kwargs.get("expires_in", kwargs.get("expires", kwargs.get("expiry", 0))),
                    )
                if (not _raw) and expires_in_s:
                    _raw = expires_in_s
                access, refresh, email, project_id = _access, _refresh, _email, _pid
                expires_in_s = _raw if _raw is not None else 0
            else:
                if not access:
                    access = str(
                        kwargs.get("access_token")
                        or kwargs.get("access")
                        or kwargs.get("token")
                        or ""
                    )
                if not refresh:
                    refresh = str(
                        kwargs.get("refresh_token") or kwargs.get("refresh") or ""
                    )
                if not email:
                    email = str(kwargs.get("email") or "")
                if not project_id:
                    project_id = str(
                        kwargs.get("project_id")
                        or kwargs.get("projectId")
                        or kwargs.get("project")
                        or ""
                    )
                if not expires_in_s:
                    _raw2 = kwargs.get(
                        "expires_in_s",
                        kwargs.get(
                            "expires_in",
                            kwargs.get("expires", kwargs.get("expiry", 0)),
                        ),
                    )
                    if _raw2:
                        expires_in_s = _raw2
            access_s = str(access or "")
            refresh_s = str(refresh or "")
            email_s = str(email or "")
            pid_s = str(project_id or "")
            try:
                _f = float(expires_in_s)  # type: ignore[arg-type]
            except Exception:
                _f = 0
            if _f > 1e12:
                expires_ms = int(_f)
            elif _f > 1e9:
                expires_ms = int(_f * 1000)
            else:
                _dur = int(_f) if _f and _f > 0 else 3600
                if _dur <= 0:
                    _dur = 3600
                expires_ms = _now_ms() + _dur * 1000
            self.store_oauth("antigravity", access_s, refresh_s, expires_ms, email_s, pid_s)
        except Exception:
            # Never crash login on store; fallback best-effort via store_oauth.
            try:
                self.store_oauth(
                    "antigravity",
                    str(access) if not isinstance(access, dict) else "",
                    str(refresh or ""),
                    _now_ms() + 3600 * 1000,
                    str(email or ""),
                    str(project_id or ""),
                )
            except Exception:
                pass

    def get_antigravity_token(self, auto_refresh: bool = True) -> Optional[str]:
        """Return valid Antigravity access STRING, or None (never raise).

        oauth valid → access; expired + refresh → _refresh_google_token →
        update store → fresh; gagal → None.
        """
        try:
            cred = self.get_credential("antigravity")
            if cred is None:
                return None
            if cred.get("type") == "api_key":
                return cred.get("key") or None
            if cred.get("type") != "oauth":
                return None
            access = str(cred.get("access") or "")
            try:
                expires_ms = int(cred.get("expires", 0) or 0)
            except Exception:
                expires_ms = 0
            refresh_tok = str(cred.get("refresh") or "")
            is_expired = bool(expires_ms and _now_ms() + _REFRESH_SKEW_MS >= expires_ms)
            if not is_expired:
                return access or None
            if not auto_refresh:
                return None
            if not refresh_tok:
                return None
            try:
                refreshed = self._refresh_google_token(refresh_tok)
            except Exception:
                return None
            if not isinstance(refreshed, dict):
                return None
            fresh_access = str(
                refreshed.get("access_token") or refreshed.get("access") or ""
            ).strip()
            if not fresh_access:
                return None
            try:
                _ei_raw = refreshed.get("expires_in", 3600)
                _ei = int(float(_ei_raw))  # type: ignore[arg-type]
            except Exception:
                _ei = 3600
            if _ei <= 0:
                _ei = 3600
            new_refresh = str(
                refreshed.get("refresh_token") or refreshed.get("refresh") or refresh_tok
            )
            old_email = str(cred.get("email") or "")
            if not old_email and isinstance(refreshed.get("email"), str):
                old_email = refreshed.get("email", "")
            old_pid = str(
                cred.get("projectId")
                or cred.get("project_id")
                or cred.get("project")
                or ""
            )
            if not old_pid:
                for _k in ("projectId", "project_id", "project", "projectID"):
                    _v = refreshed.get(_k)
                    if isinstance(_v, str) and _v.strip():
                        old_pid = _v.strip()
                        break
            new_expires_ms = _now_ms() + _ei * 1000
            try:
                self.store_oauth(
                    "antigravity", fresh_access, new_refresh, new_expires_ms, old_email, old_pid
                )
            except Exception:
                pass
            return fresh_access
        except Exception:
            return None

    def discover_antigravity_project(
        self, access_token: Any = None, timeout: Any = 15
    ) -> Optional[str]:
        """Discover Cloud project id via loadCodeAssist (stdlib only).

        POST https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist
        Bearer + antigravity headers. Parse project id (kunci
        project/projectId/name, rekursif) → update kredensial vault
        (project_id + email bila ada, bukan full respons). Gagal → None.
        Tanpa secret di log.
        """
        try:
            if isinstance(access_token, bool):
                access_token = None
            elif isinstance(access_token, (int, float)):
                try:
                    timeout = int(access_token)
                except Exception:
                    pass
                access_token = None
            if isinstance(access_token, dict):
                _dd = access_token
                access_token = str(
                    _dd.get("access_token") or _dd.get("access") or _dd.get("token") or ""
                )
            if not access_token:
                try:
                    _getter = getattr(self, "get_antigravity_token", None)
                    if callable(_getter):
                        try:
                            access_token = _getter()
                        except Exception:
                            access_token = None
                except Exception:
                    access_token = None
                if not access_token:
                    try:
                        access_token = self.get_token("antigravity")
                    except Exception:
                        access_token = None
            if not isinstance(access_token, str):
                return None
            access_token = access_token.strip()
            if not access_token:
                return None
            try:
                timeout_s = int(timeout)  # type: ignore[arg-type]
            except Exception:
                timeout_s = 15
            if timeout_s <= 0:
                timeout_s = 15
            url = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
            body = json.dumps({}).encode("utf-8")
            headers = {
                "Authorization": "Bearer " + access_token,
                "Content-Type": "application/json",
                "User-Agent": "antigravity/1.15.8",
                "X-Goog-Api-Client": "google-cloud-sdk",
                "Client-Metadata": "ANTIGRAVITY",
            }
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except Exception:
                return None
            try:
                data = json.loads(raw)
            except Exception:
                return None

            def _find_project(obj: Any, depth: int = 0) -> Optional[str]:
                if depth > 6 or obj is None:
                    return None
                if isinstance(obj, dict):
                    for _k in (
                        "projectId",
                        "project_id",
                        "project",
                        "projectID",
                        "cloudaicompanionProject",
                    ):
                        _v = obj.get(_k)
                        if isinstance(_v, str) and _v.strip():
                            return _v.strip()
                        if isinstance(_v, dict):
                            for _sk in ("id", "projectId", "name"):
                                _sv = _v.get(_sk)
                                if isinstance(_sv, str) and _sv.strip():
                                    return _sv.strip()
                    _nm = obj.get("name")
                    if isinstance(_nm, str) and _nm.strip():
                        _ns = _nm.strip()
                        if _ns.startswith("projects/"):
                            _tail = _ns.split("/")[-1].strip()
                            if _tail:
                                return _tail
                    for _v2 in obj.values():
                        if isinstance(_v2, (dict, list)):
                            _found = _find_project(_v2, depth + 1)
                            if _found:
                                return _found
                    return None
                if isinstance(obj, list):
                    for _it in obj:
                        _found2 = _find_project(_it, depth + 1)
                        if _found2:
                            return _found2
                    return None
                return None

            def _find_email(obj: Any, depth: int = 0) -> Optional[str]:
                if depth > 6 or obj is None:
                    return None
                if isinstance(obj, dict):
                    for _ek in ("email", "userEmail", "user_email"):
                        _ev = obj.get(_ek)
                        if isinstance(_ev, str) and _ev.strip():
                            return _ev.strip()
                    for _v3 in obj.values():
                        if isinstance(_v3, (dict, list)):
                            _fe = _find_email(_v3, depth + 1)
                            if _fe:
                                return _fe
                    return None
                if isinstance(obj, list):
                    for _it2 in obj:
                        _fe2 = _find_email(_it2, depth + 1)
                        if _fe2:
                            return _fe2
                    return None
                return None

            pid = _find_project(data)
            if not pid or not isinstance(pid, str):
                return None
            pid = pid.strip()
            if pid.startswith("projects/"):
                pid = pid.split("/")[-1].strip()
            if not pid:
                return None
            found_email = _find_email(data)
            try:
                cred = self.get_credential("antigravity")
                if isinstance(cred, dict) and cred.get("type") == "oauth":
                    cur_access = str(cred.get("access") or access_token)
                    cur_refresh = str(cred.get("refresh") or "")
                    try:
                        cur_exp = int(cred.get("expires", 0) or 0)
                    except Exception:
                        cur_exp = 0
                    cur_email = str(cred.get("email") or "")
                    if isinstance(found_email, str) and found_email.strip():
                        cur_email = found_email.strip()
                    self.store_oauth(
                        "antigravity", cur_access, cur_refresh, cur_exp, cur_email, pid
                    )
                elif isinstance(cred, dict) and cred.get("type") == "api_key":
                    _em = found_email.strip() if isinstance(found_email, str) else ""
                    self.store_oauth("antigravity", access_token, "", 0, _em, pid)
                else:
                    _em2 = found_email.strip() if isinstance(found_email, str) else ""
                    if cred is None:
                        self.store_oauth("antigravity", access_token, "", 0, _em2, pid)
                    else:
                        try:
                            cur2_access = str(cred.get("access") or access_token)
                            cur2_refresh = str(cred.get("refresh") or "")
                            try:
                                cur2_exp = int(cred.get("expires", 0) or 0)
                            except Exception:
                                cur2_exp = 0
                            cur2_email = str(cred.get("email") or "")
                            if isinstance(found_email, str) and found_email.strip():
                                cur2_email = found_email.strip()
                            self.store_oauth(
                                "antigravity",
                                cur2_access,
                                cur2_refresh,
                                cur2_exp,
                                cur2_email,
                                pid,
                            )
                        except Exception:
                            self.store_oauth(
                                "antigravity", access_token, "", 0, _em2, pid
                            )
            except Exception:
                pass
            return pid
        except Exception:
            return None

    def _get_google_client_secret(self) -> str:
        """Resolve client secret without hardcoding: env -> vault -> single-source fallback."""
        env_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        if env_secret:
            return env_secret
        try:
            stored = self._data.get("google_client_secret")
            if isinstance(stored, dict):
                val = stored.get("key") or stored.get("value") or ""
                if val:
                    return val
            elif isinstance(stored, str) and stored:
                return stored
        except Exception:
            pass
        try:
            from harness.models.google_oauth import CLIENT_SECRET as _single_source

            if _single_source:
                return _single_source
        except Exception:
            pass
        return GOOGLE_CLIENT_SECRET

    def _refresh_google_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh a Google OAuth token using the stored refresh token."""
        try:
            payload = urllib.parse.urlencode({
                "client_id": self._get_google_client_id(),
                "client_secret": self._get_google_client_secret(),
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def discover_gemini_token(self) -> Optional[str]:
        """Discover Gemini token: vault → Antigravity session → ADC."""
        if token := self.get_token("gemini"):
            return token
        if token := self.discover_antigravity_token():
            return token
        return self.discover_google_adc_token()
