import sys
import os
import time
import asyncio
import difflib
import traceback
import logging
import threading
from typing import Tuple, Optional, List, Callable, Dict, Any

# Silence all loggers to ERROR level
logging.basicConfig(level=logging.ERROR)
for _log_name in ["harness", "urllib3", "markdown_it", "asyncio"]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Optional rich imports with graceful fallback
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.status import Status
    from rich.rule import Rule
    from rich.text import Text
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    Live = None
    Spinner = None

from harness.config import CodeAIConfig
from harness.core.orchestrator import Orchestrator
from harness.core.hooks import HooksDispatcher
from harness.rules.agents_parser import AgentsParser
from harness.core.allowance import AllowanceGuard

PROVIDER_DEFAULT_MODELS = {
    "anthropic": lambda: os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    "openai":    lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini":    lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "copilot":   lambda: os.environ.get("COPILOT_MODEL", "gpt-4o"),
    "ollama":    lambda: os.environ.get("OLLAMA_MODEL", "llama3"),
    "opencode":  lambda: "mimo-v2.5-free",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg, style="", end="\n"):
    if RICH_AVAILABLE:
        console.print(msg, style=style, end=end)
    else:
        print(msg, end=end)

def _rule(title="", style="dim"):
    if RICH_AVAILABLE:
        console.print(Rule(title, style=style))
    else:
        print(f"─── {title} " + "─" * (40 - len(title)))


# ---------------------------------------------------------------------------
# TUI picker fullscreen (STDLIB ONLY: termios/tty/select/sys)
# ---------------------------------------------------------------------------

def _tui_filter(items, query, show=None):
    """Logika murni (tanpa TTY): filter substring case-insensitive.

    Return list[(orig_idx, item)] agar mapping ke `items` utuh.
    Query kosong → semua item.
    """
    _show = show if callable(show) else (lambda x: x)  # noqa: E731

    def _label(it):
        try:
            return str(_show(it))
        except Exception:
            return str(it)

    q = (query or "").strip().lower()
    if not q:
        return list(enumerate(items or []))
    out = []
    for i, it in enumerate(items or []):
        try:
            if q in _label(it).lower():
                out.append((i, it))
        except Exception:
            continue
    return out


def _tui_move(idx, n, key):
    """Logika murni (tanpa TTY): gerak highlight wrap-around.

    key: 'up'/'down' (panah), 'j'/'k' (vim), 'home'/'end'. Lainnya → tetap.
    """
    if n <= 0:
        return 0
    try:
        idx = int(idx) % n
    except Exception:
        idx = 0
    k = (key or "").strip().lower()
    if k == "home":
        return 0
    if k == "end":
        return n - 1
    if k in ("up", "k", "ctrl-p"):
        return (idx - 1) % n
    if k in ("down", "j", "ctrl-n"):
        return (idx + 1) % n
    return idx


def _tui_setraw(fd, when=None):
    """Raw-mode seperti tty.setraw TAPI pertahankan OPOST|ONLCR.

    Menyalin attrs, mematikan input/canonical/echo/ISIG persis setraw,
    namun oflag dipaksa ber-OPOST|ONLCR agar '\\n' tetap jadi CRLF di pty
    (stdin+stdout satu device) dan tidak diagonal.
    """
    import termios
    if when is None:
        when = termios.TCSAFLUSH
    attrs = termios.tcgetattr(fd)
    new = [attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[5], list(attrs[6])]
    new[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
    new[1] |= (termios.OPOST | termios.ONLCR)
    new[2] &= ~(termios.CSIZE | termios.PARENB)
    new[2] |= termios.CS8
    new[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, when, new)


class _TUIPicker:
    """Picker fullscreen interaktif (raw termios) + simulasi key-sequence.

    Interaktif: panah ↑↓ / j/k pindah, huruf filter live, Spasi toggle ✓
    (multi), Enter pilih, Esc/q/Ctrl-C batal (None, JANGAN switch). Tiap
    refresh clear screen ANSI (`\\x1b[2J\\x1b[H`); termios SELALU direstore
    di finally (bahkan saat exception). Non-TTY → pemanggil fallback ke
    _popup_pick line-based. Simulasi: pick(keys=[...]) tanpa TTY — token
    'up'/'down'/'enter'/'esc'/'q'/'backspace'/'space'/'ctrl-c'/'j'/'k',
    1 huruf biasa (j/k/q = navigasi/batal), atau string >1 huruf = ketik
    harfiah per huruf (termasuk j/k/q sebagai query).
    """

    VIEWPORT = 15
    FOOTER = "Cara pakai: ketik untuk mencari · tombol atas bawah untuk pindah · Enter untuk pilih · Esc untuk batal"

    def __init__(self, title="", items=None, show=None, initial="", multi=False):
        self.title = title or "Pilih"
        self.items = list(items) if items else []
        self.show = show if callable(show) else (lambda x: x)  # noqa: E731
        self.query = (initial or "").strip()
        self.multi = bool(multi)
        self.idx = 0
        self.selected: List[int] = []
        self._scroll = 0

    def label(self, it) -> str:
        try:
            return str(self.show(it))
        except Exception:
            return str(it)

    def filtered(self):
        return _tui_filter(self.items, self.query, self.show)

    @staticmethod
    def available() -> bool:
        try:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                return False
            import termios  # noqa: F401
            import tty  # noqa: F401
            import select  # noqa: F401
            return True
        except Exception:
            return False

    def _viewport(self, total: int) -> int:
        vs = self.VIEWPORT
        if total <= 0:
            self._scroll = 0
            return 0
        if self.idx >= total:
            self.idx = total - 1
        if self.idx < 0:
            self.idx = 0
        sc = self._scroll
        if self.idx < sc:
            sc = self.idx
        elif self.idx >= sc + vs:
            sc = self.idx - vs + 1
        sc = max(0, min(sc, max(0, total - vs)))
        self._scroll = sc
        return sc

    def _lines(self):
        # VERTIKAL murni: 1 item = 1 baris, TANPA Columns/Table/wrap.
        # baris1 judul+posisi, baris2 query, lalu viewport, lalu footer.
        import shutil
        try:
            _tw = int(shutil.get_terminal_size(fallback=(80, 24)).columns)
        except Exception:
            _tw = 80
        if _tw < 20:
            _tw = 80
        _max = max(20, min(_tw, 100) - 4)

        def _trunc(s: str, budget: int) -> str:
            t = str(s).replace("\r", " ").replace("\n", " ")
            if len(t) <= budget:
                return t
            if budget <= 1:
                return t[:budget]
            return t[: budget - 1] + "…"

        filt = self.filtered()
        total = len(filt)
        sc = self._viewport(total)
        pos = f"▶ {self.idx + 1}/{total}" if total else "▶ 0/0"
        lines = [_trunc(f"{self.title}  {pos}", _max), _trunc(f"Cari: {self.query}▊", _max)]
        for r in range(sc, min(sc + self.VIEWPORT, total)):
            oi, it = filt[r]
            raw = str(self.label(it)).replace("\r", " ").replace("\n", " ")
            mark = ("✓ " if oi in self.selected else "  ") if self.multi else ""
            pre = "▶ " if r == self.idx else "  "
            num = f"{r + 1:2}. "
            budget = _max - len(pre) - len(num) - len(mark)
            if budget < 1:
                budget = 1
            lab = _trunc(raw, budget)
            if r == self.idx:
                lines.append(f"\x1b[7m{pre}{num}{mark}{lab}\x1b[0m")
            else:
                lines.append(f"{pre}{num}{mark}{lab}")
        if total == 0:
            lines.append(_trunc(f"  ✗ tidak cocok: '{self.query}' — coba kata lain, contoh: gemini", _max))
        if self.multi:
            lines.append(_trunc(f"  [terpilih {len(self.selected)} terhitung · Spasi=Tandai · Enter=Selesai (min 1)]", _max))
        # Footer Cara pakai JANGAN dipotong agar frasa baku utuh di semua lebar terminal.
        lines.append(self.FOOTER if not self.multi else self.FOOTER + " · Spasi=Tandai")
        return lines

    def _render(self) -> None:
        # Fullscreen refresh: clear + hide kursor, SATU write per baris via join+\n.
        try:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write("\x1b[?25l")
            sys.stdout.write("\n".join(self._lines()))
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

    @staticmethod
    def parse_key_sequence(seq: str) -> str:
        """Pure parser sekuens-key -> token (unit-testable, tanpa TTY).

        seq: raw string (mis. "\x1b", "\x1b[A", "\x1b[B", "\x1b[C",
        "\x1b[D", "\x1b[H", "\x1b[F", "\x1bOA", "\r", "j", ...).
        Return: up/down/left/right/home/end/enter/backspace/ctrl-c/space/
        esc/j/k/q/single-char/unknown. CSI tak dikenal -> unknown
        (JANGAN esc agar tak batal).
        """
        if not seq:
            return "esc"
        if seq in ("\r", "\n"):
            return "enter"
        if seq in ("\x7f", "\x08"):
            return "backspace"
        if seq in ("\x03", "\x04"):
            return "ctrl-c"
        if seq == " ":
            return "space"
        if seq == "\x1b":
            return "esc"
        if seq.startswith("\x1b[") or seq.startswith("\x1bO"):
            if len(seq) < 3:
                return "unknown"
            final = seq[-1]
            if final == "A":
                return "up"
            if final == "B":
                return "down"
            if final == "C":
                return "right"
            if final == "D":
                return "left"
            if final in ("H",):
                return "home"
            if final in ("F",):
                return "end"
            if final == "~":
                try:
                    _num = seq[2:-1].split(";")[0].strip()
                    if _num in ("1", "7"):
                        return "home"
                    if _num in ("4", "8"):
                        return "end"
                except Exception:
                    pass
                return "unknown"
            return "unknown"
        if len(seq) == 1 and seq.lower() in ("j", "k", "q"):
            return seq.lower()
        if len(seq) == 1 and (seq.isprintable() or ord(seq) > 127):
            return seq
        return "unknown"

    def _read_key(self) -> str:
        import select
        ch = sys.stdin.read(1)
        if not ch:
            return "esc"
        if ch != "\x1b":
            if ch in ("\r", "\n"):
                return "enter"
            if ch in ("\x7f", "\x08"):
                return "backspace"
            if ch in ("\x03", "\x04"):
                return "ctrl-c"
            if ch == " ":
                return "space"
            if len(ch) == 1 and ch.lower() in ("j", "k", "q"):
                return ch.lower()
            if len(ch) == 1 and (ch.isprintable() or ord(ch) > 127):
                return ch
            return "unknown"
        r, _, _ = select.select([sys.stdin], [], [], 0.08)
        if not r:
            return "esc"
        ch2 = sys.stdin.read(1)
        if not ch2:
            return "esc"
        if ch2 not in ("[", "O"):
            return "esc"
        seq = ch + ch2
        for _ in range(8):
            r2, _, _ = select.select([sys.stdin], [], [], 0.08)
            if not r2:
                break
            chN = sys.stdin.read(1)
            if not chN:
                break
            seq += chN
            if chN.isalpha() or chN == "~":
                break
        return self.parse_key_sequence(seq)

    def _handle_key(self, key):
        """Return 'select'/'cancel'/None. Mutasi query/idx/selected."""
        filt = self.filtered()
        total = len(filt)
        if key in ("up", "down", "home", "end"):
            if total:
                self.idx = _tui_move(self.idx, total, key)
            return None
        if key in ("left", "right", "unknown"):
            return None
        if isinstance(key, str) and key in ("j", "k"):
            if total:
                self.idx = _tui_move(self.idx, total, key)
            return None
        if key in ("esc", "q", "ctrl-c"):
            return "cancel"
        if key == "backspace":
            if self.query:
                self.query = self.query[:-1]
                self.idx = 0
                self._scroll = 0
            return None
        if key == "enter":
            if not total:
                return "cancel"
            if self.multi:
                if self.selected:
                    return "select"
                # Minimal 1: highlight ikut tersimpan agar Enter selalu ≥1.
                self.selected = [filt[min(self.idx, total - 1)][0]]
                return "select"
            return "select"
        if key == "space":
            if self.multi:
                if total:
                    oi = filt[min(self.idx, total - 1)][0]
                    if oi in self.selected:
                        self.selected.remove(oi)
                    else:
                        self.selected.append(oi)
                return None
            self.query += " "
            self.idx = 0
            self._scroll = 0
            return None
        if isinstance(key, str) and len(key) == 1 and (key.isprintable() or ord(key) > 127):
            self.query += key
            self.idx = 0
            self._scroll = 0
            return None
        return None

    def _result(self):
        filt = self.filtered()
        if self.multi:
            return list(self.selected) if self.selected else None
        if not filt:
            return None
        return filt[min(self.idx, len(filt) - 1)][0]

    def _run_keys(self, keys):
        for tok in keys or []:
            if tok is None:
                continue
            t = tok if isinstance(tok, str) else str(tok)
            tl = t.lower()
            if tl in ("up", "down", "enter", "esc", "backspace", "space", "ctrl-c"):
                act = self._handle_key(tl)
            elif len(t) == 1:
                if t == " ":
                    act = self._handle_key("space")
                elif tl in ("j", "k", "q"):
                    act = self._handle_key(tl)
                else:
                    self.query += t
                    self.idx = 0
                    self._scroll = 0
                    continue
            else:
                # Token >1 huruf = ketik harfiah per huruf (j/k/q = query).
                for ch in t:
                    self.query += ch
                self.idx = 0
                self._scroll = 0
                continue
            if act == "select":
                return self._result()
            if act == "cancel":
                return None
        return None

    def pick(self, keys=None):
        """Interaktif raw-TTY, atau simulasi bila `keys` diisi (tanpa TTY)."""
        if keys is not None:
            return self._run_keys(list(keys))
        import termios
        import tty
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            old = None
        try:
            if old is not None:
                _tui_setraw(fd)
            try:
                sys.stdout.write("\x1b[?25l")
                sys.stdout.flush()
            except Exception:
                pass
            while True:
                self._render()
                key = self._read_key()
                act = self._handle_key(key)
                if act == "select":
                    return self._result()
                if act == "cancel":
                    return None
        finally:
            if old is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except Exception:
                    pass
            try:
                sys.stdout.write("\x1b[?25h\x1b[0m\n")
                sys.stdout.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI class
# ---------------------------------------------------------------------------

class CodeAICLI:
    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.orchestrator: Optional[Orchestrator] = None
        self._last_elapsed: Optional[float] = None
        # Non-blocking discovery cache (lazy + TTL, stdlib only).
        self._reg_singleton = None
        self._providers_cache = None
        self._providers_ts: float = 0.0
        self._models_cache = None
        self._models_ts: float = 0.0
        self._cache_lock = threading.Lock()
        self._spin_lock = threading.Lock()
        self._chat_lock = threading.Lock()
        try:
            self.config = CodeAIConfig.load_from_file(self.config_path)
        except Exception:
            self.config = CodeAIConfig()

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def get_active_info(self) -> Tuple[str, str]:
        if not self.config or not self.config.provider:
            return "anthropic", PROVIDER_DEFAULT_MODELS["anthropic"]()
        provider = self.config.provider.default or "anthropic"
        model = getattr(self.config.provider, "active_model", None)
        if not model or model == "default":
            model = PROVIDER_DEFAULT_MODELS.get(provider, lambda: "default")()
        return provider, model

    def _registry(self):
        from harness.models.provider_registry import ProviderRegistry
        if getattr(self, "_reg_singleton", None) is not None:
            return self._reg_singleton
        try:
            _lk = self._cache_lock
        except AttributeError:
            _lk = self._cache_lock = threading.Lock()
        with _lk:
            if getattr(self, "_reg_singleton", None) is not None:
                return self._reg_singleton
            cp = getattr(self.config, "custom_providers", None) if self.config else None
            self._reg_singleton = ProviderRegistry(custom_providers=cp or None)
            return self._reg_singleton

    def _run_with_timeout(self, fn, timeout: float = 1.5, fallback=None):
        """Jalankan fn() blocking di thread daemon + join(timeout); timeout → fallback."""
        import threading
        box: dict = {}
        def _t():
            try:
                box["v"] = fn()
            except Exception as e:
                box["e"] = e
        th = threading.Thread(target=_t, daemon=True)
        th.start()
        th.join(timeout=timeout)
        try:
            if th.is_alive():
                return fallback
        except Exception:
            pass
        if "v" in box:
            return box["v"]
        return fallback

    def _list_providers_fast(self, timeout: float = 1.5) -> list:
        """Cache-dulu + lazy: cache segar (<30s) langsung; miss → fetch timeout, gagal → stale/[] non-blocking."""
        now = time.monotonic()
        try:
            _lk = self._cache_lock
        except AttributeError:
            import threading as _th0
            _lk = self._cache_lock = _th0.Lock()
        with _lk:
            try:
                ttl_ok = self._providers_cache is not None and (now - float(self._providers_ts)) < 30.0
            except Exception:
                ttl_ok = False
            if ttl_ok:
                return self._providers_cache
            stale = self._providers_cache if self._providers_cache is not None else []
        reg = self._registry()
        res = self._run_with_timeout(reg.list_providers, timeout=timeout, fallback=None)
        if res is None:
            # Lazy refresh di background agar panggilan berikut segar tanpa block UI kini.
            try:
                import threading as _th
                def _bg():
                    try:
                        _v = reg.list_providers()
                        with _lk:
                            self._providers_cache = _v
                            self._providers_ts = time.monotonic()
                    except Exception:
                        pass
                _th.Thread(target=_bg, daemon=True).start()
            except Exception:
                pass
            return stale
        with _lk:
            self._providers_cache = res
            self._providers_ts = now
        return res

    def _list_connected_fast(self, timeout: float = 2.0) -> list:
        """Cache-dulu untuk list_connected_models (hindari block 8s discovery); timeout → stale/[] + refresh lazy."""
        now = time.monotonic()
        try:
            _lk2 = self._cache_lock
        except AttributeError:
            import threading as _th0b
            _lk2 = self._cache_lock = _th0b.Lock()
        with _lk2:
            try:
                ttl_ok = self._models_cache is not None and (now - float(self._models_ts)) < 30.0
            except Exception:
                ttl_ok = False
            if ttl_ok:
                return self._models_cache
            stale = self._models_cache if self._models_cache is not None else []
        reg = self._registry()
        res = self._run_with_timeout(reg.list_connected_models, timeout=timeout, fallback=None)
        if res is None:
            try:
                import threading as _th2
                def _bg2():
                    try:
                        _v2 = reg.list_connected_models()
                        with _lk2:
                            self._models_cache = _v2
                            self._models_ts = time.monotonic()
                    except Exception:
                        pass
                _th2.Thread(target=_bg2, daemon=True).start()
            except Exception:
                pass
            return stale
        with _lk2:
            self._models_cache = res
            self._models_ts = now
        return res

    @staticmethod
    def _short(s: str, n: int = 22) -> str:
        # Tail-only truncation (never mid-truncate to avoid `--` mangling).
        # NOTE: hanya untuk model; label provider TIDAK boleh lewat sini
        # agar nama penuh (mis. "antigravity") tak terpotong jadi "AI".
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    # --- Error-chain helpers (stdlib only, cli-side; gateway tak disentuh) ---
    @staticmethod
    def _is_auth_like(text: str) -> bool:
        try:
            t = str(text or "").lower()
        except Exception:
            return False
        keys = ("credential", "auth", "/provider", "api key", "apikey",
                "unauthorized", "401", "forbidden", "403", "token",
                "agy", "re-authenticate", "authenticate")
        return any(k in t for k in keys)

    @staticmethod
    def _is_conn_like(text: str) -> bool:
        try:
            t = str(text or "").lower()
        except Exception:
            return False
        keys = ("connection", "refused", "connect", "unreachable",
                "network", "econn", "socket", "127.0.0.1", "localhost",
                "11434", "ollama", "timed out", "timeout")
        return any(k in t for k in keys)

    @staticmethod
    def _ollama_installed() -> bool:
        try:
            import shutil as _sh
            return bool(_sh.which("ollama"))
        except Exception:
            return False

    @staticmethod
    def _ollama_running(timeout: float = 1.0) -> bool:
        # Local probe stdlib-only; tak pernah raise; timeout singkat.
        try:
            import urllib.request as _ur
            _req = _ur.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with _ur.urlopen(_req, timeout=timeout) as _r:
                return int(getattr(_r, "status", 200)) == 200
        except Exception:
            pass
        try:
            import socket as _so
            _s = _so.create_connection(("127.0.0.1", 11434), timeout=timeout)
            try:
                _s.close()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _connected_ids_for_suggest(self) -> list:
        # Hanya CONNECTED (punya kredensial) + local running; ollama
        # tak terinstall/tak running SELALU disaring (jangan sarankan).
        try:
            provs = self._list_providers_fast()
        except Exception:
            return []
        out = []
        try:
            _oll_inst = self._ollama_installed()
        except Exception:
            _oll_inst = False
        _oll_run = None
        for p in provs or []:
            try:
                _pid = str(p.get("id", "")).strip()
                if not _pid or _pid == "combo":
                    continue
                if str(p.get("api", "")) == "local":
                    continue
                if not p.get("has_credentials"):
                    continue
                if _pid == "ollama":
                    if not _oll_inst:
                        continue
                    if _oll_run is None:
                        try:
                            _oll_run = self._ollama_running(timeout=1.0)
                        except Exception:
                            _oll_run = False
                    if not _oll_run:
                        continue
                out.append(_pid)
            except Exception:
                continue
        return out

    def _failover_chain_for(self, provider: str, model: str) -> list:
        # Rekonstruksi rantai gateway tanpa menyentuh gateway/combo/compactor.
        try:
            _gw = getattr(self.orchestrator, "gateway", None) if self.orchestrator else None
            if _gw is not None and hasattr(_gw, "_build_failover_chain"):
                _ch = _gw._build_failover_chain(provider)
                if isinstance(_ch, list) and _ch:
                    _seen, _out = set(), []
                    for _x in _ch:
                        _s = str(_x).strip() if isinstance(_x, str) else str(_x)
                        if _s and _s not in _seen:
                            _seen.add(_s)
                            _out.append(_s)
                    if _out:
                        return _out
        except Exception:
            pass
        try:
            _base = [str(provider).strip()]
            _fo = getattr(getattr(self.orchestrator.gateway, "config", None), "failover_order", []) if self.orchestrator and getattr(self.orchestrator, "gateway", None) else []
            for _x in list(_fo or []):
                _s = str(_x).strip()
                if _s and _s not in _base:
                    _base.append(_s)
            return [x for x in _base if x]
        except Exception:
            pass
        try:
            return [str(provider).strip() or "anthropic"]
        except Exception:
            return ["anthropic"]

    def _format_task_error(self, provider: str, model: str, err_text: str) -> str:
        # Render rantai penuh per baris (ringkas), saran hanya CONNECTED.
        try:
            raw = str(err_text or "")
        except Exception:
            raw = ""
        try:
            low = raw.lower()
        except Exception:
            low = ""
        # Ekstrak sebab terakhir (last error) satu baris ≤120 char.
        try:
            _cause = raw
            if "last error:" in low:
                _cause = raw[low.rfind("last error:") + len("last error:"):].strip().rstrip(".").strip()
                # Buang hint trailing "run '/provider...'" bila menempel.
                _ll = _cause.lower()
                _cut = _ll.find("run '/provider")
                if _cut > 20:
                    _cause = _cause[:_cut].strip().rstrip(".").strip()
                else:
                    _cut2 = _ll.find(" run '/model")
                    if _cut2 > 20:
                        _cause = _cause[:_cut2].strip().rstrip(".").strip()
            elif "failed for model" in low:
                # "Provider 'x' failed for model 'y': <sebab> Run '...'"
                _idx = raw.find(": ", raw.lower().find("failed for model"))
                if _idx != -1:
                    _tail = raw[_idx + 2:].strip()
                    _lt = _tail.lower()
                    _c = _lt.find("run '/provider")
                    if _c > 0:
                        _tail = _tail[:_c].strip().rstrip(".").strip()
                    else:
                        _c2 = _lt.find("run '/model")
                        if _c2 > 0:
                            _tail = _tail[:_c2].strip().rstrip(".").strip()
                    if _tail:
                        _cause = _tail
            _cause = " ".join(str(_cause).split())
            if len(_cause) > 120:
                _cause = _cause[:119] + "…"
            if not _cause:
                _cause = "gagal"
        except Exception:
            _cause = "gagal"
        try:
            chain = self._failover_chain_for(provider, model)
        except Exception:
            chain = [provider]
        if not chain:
            chain = [provider]
        try:
            connected = self._connected_ids_for_suggest()
        except Exception:
            connected = []
        try:
            conn_set = set(connected or [])
        except Exception:
            conn_set = set()
        # Pemilik sebab: explicit-target → provider; "all configured..." → akhir rantai.
        try:
            _is_all = "all configured providers failed" in low
            _owner = chain[-1] if _is_all and chain else provider
        except Exception:
            _owner = provider
        try:
            _oll_inst = self._ollama_installed()
        except Exception:
            _oll_inst = False
        _oll_run = None
        if _oll_inst:
            try:
                _oll_run = self._ollama_running(timeout=1.0)
            except Exception:
                _oll_run = False
        else:
            _oll_run = False
        lines: list = []
        lines.append("Gagal menjawab")
        try:
            _sebab = str(_cause or "").strip() or "tidak ada rincian"
        except Exception:
            _sebab = "tidak ada rincian"
        lines.append(f"Sebab: {_sebab}")
        for _pid in chain:
            try:
                _ps = str(_pid).strip() or "?"
            except Exception:
                _ps = "?"
            if _ps == "ollama":
                if not _oll_inst:
                    lines.append("• ollama: dilewati (belum dipasang)")
                elif not _oll_run:
                    lines.append("• ollama: dilewati (belum jalan — jalankan `ollama serve`)")
                elif _ps == _owner:
                    if self._is_auth_like(_cause):
                        lines.append(f"• ollama: {_cause} → saran /provider ollama")
                    elif self._is_conn_like(_cause):
                        lines.append(f"• ollama: {_cause} → periksa sambungan atau /model")
                    else:
                        lines.append(f"• ollama: {_cause}")
                else:
                    lines.append("• ollama: dicoba (cadangan)")
                continue
            if _ps not in conn_set:
                lines.append(f"• {_ps}: belum terhubung → saran /provider {_ps}")
            elif _ps == _owner:
                if self._is_auth_like(_cause):
                    lines.append(f"• {_ps}: {_cause} → saran /provider {_ps}")
                elif self._is_conn_like(_cause):
                    lines.append(f"• {_ps}: {_cause} → periksa sambungan atau /model")
                else:
                    lines.append(f"• {_ps}: {_cause}")
            else:
                lines.append(f"• {_ps}: dicoba (cadangan)")
        if connected:
            try:
                _show = ", ".join(list(connected)[:5])
                lines.append(f"Saran: /model {_show} (hanya yang Terhubung) · /providers (/daftar) untuk daftar")
            except Exception:
                pass
        else:
            lines.append("Saran: /provider untuk menyambung (lihat /providers atau /daftar)")
        lines.append("Langkah 1: ketik /provider untuk menyambung, contoh: /provider gemini")
        lines.append("Langkah 2: ketik /model untuk memilih model yang Terhubung, contoh: /model gemini")
        lines.append("Langkah 3: ulangi pertanyaan Anda, contoh: halo")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Popup + search bar (generik, stdlib only, rich-optional)
    # ------------------------------------------------------------------

    def _popup_render(self, title: str, query: str, shown: list, total: int) -> None:
        """Render satu ronde popup — SATU item SATU baris vertikal, TANPA Table/Columns/wrap."""
        import shutil
        try:
            _tw = int(shutil.get_terminal_size(fallback=(80, 24)).columns)
        except Exception:
            _tw = 80
        if _tw < 20:
            _tw = 80
        _max = max(20, min(_tw, 100) - 4)

        def _trunc(s: str, budget: int) -> str:
            t = str(s).replace("\r", " ").replace("\n", " ")
            if len(t) <= budget:
                return t
            if budget <= 1:
                return t[:budget]
            return t[: budget - 1] + "…"

        footer = "Cara pakai: ketik untuk mencari · tombol atas bawah untuk pindah · Enter untuk pilih · Esc untuk batal"
        count_line = f"… +{total - len(shown)} cocok (ketik lagi untuk menyaring, contoh: gemini)" if total > len(shown) else ""
        # Plain vertical writes (no Rich Table/Panel/Columns → anti-menyamping).
        try:
            sys.stdout.write(_trunc(f"── {title}", _max) + "\n")
            sys.stdout.write(_trunc(f"Cari: {query if query else '—'}", _max) + "\n")
            for i, label in enumerate(shown, 1):
                pre = f"{i:2}. "
                budget = _max - len(pre)
                if budget < 1:
                    budget = 1
                sys.stdout.write(pre + _trunc(label, budget) + "\n")
            if count_line:
                sys.stdout.write(_trunc(count_line, _max) + "\n")
            # Footer Cara pakai JANGAN dipotong agar frasa baku utuh.
            sys.stdout.write(footer + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def _popup_input(self, hint: str = "[Cari:] nomor/teks (Enter=batal, q=batal)") -> Optional[str]:
        # stdlib input() untuk kedua mode (rich hanya untuk render) agar
        # mudah di-mock via builtins.input pada pengujian.
        try:
            return input(f"{hint}: ")
        except (EOFError, KeyboardInterrupt):
            return None

    def _popup_pick(self, title: str, items: list, show: Optional[Callable] = None, initial: str = "") -> Optional[int]:
        """Popup generik + search bar. Return index ke `items` atau None (batal).

        LOOP refine: teks → filter substring case-insensitive → render ulang;
        nomor valid → return; Enter kosong → batal (None, JANGAN switch); q → batal.
        """
        if show is None:
            show = lambda x: x  # noqa: E731
        if not items:
            _print("[dim]Tidak ada pilihan.[/dim]")
            return None

        def _labels(it):
            try:
                return str(show(it))
            except Exception:
                return str(it)

        def _filtered(q: str):
            q = (q or "").strip()
            if not q:
                return list(enumerate(items))
            ql = q.lower()
            return [(i, it) for i, it in enumerate(items) if ql in _labels(it).lower()]

        query = (initial or "").strip()
        filt = _filtered(query)
        while True:
            total = len(filt)
            visible_pairs = filt[:15]
            shown = [_labels(it) for _, it in visible_pairs]
            if total == 0:
                # Tetap render popup + pesan tolak (verbatim untuk simulasi).
                self._popup_render(title, query, [], 0)
                _print(f"[red]✗ tidak cocok: '{query}'[/red] [dim]coba kata kunci lain, contoh: gemini / q=batal[/dim]")
            else:
                self._popup_render(title, query, shown, total)
            raw = self._popup_input()
            if raw is None:
                return None
            s = (raw or "").strip()
            if not s:
                return None
            if s.lower() == "q":
                return None
            if s.isdigit():
                n = int(s)
                if 1 <= n <= total:
                    return visible_pairs[n - 1][0] if n <= len(visible_pairs) else filt[n - 1][0]
                _print(f"[red]Nomor di luar 1..{total}[/red] [dim](Enter=batal)[/dim]")
                continue
            # teks → filter ulang
            query = s
            filt = _filtered(query)
            continue

    def _popup_pick_multi(self, title: str, items: list, show: Optional[Callable] = None, initial: str = "") -> Optional[List[int]]:
        """Varian multi-pilih: `1,3` tambah, substring=filter, `done` selesai."""
        if show is None:
            show = lambda x: x  # noqa: E731
        if not items:
            return None

        def _labels(it):
            try:
                return str(show(it))
            except Exception:
                return str(it)

        def _filtered(q: str):
            q = (q or "").strip()
            if not q:
                return list(enumerate(items))
            ql = q.lower()
            return [(i, it) for i, it in enumerate(items) if ql in _labels(it).lower()]

        def _show_with_mark(it, oi, selected):
            mark = "✓ " if oi in selected else "  "
            return f"{mark}{_labels(it)}"

        selected: List[int] = []
        query = (initial or "").strip()
        filt = _filtered(query)
        hint = "[Cari:] 1,3/selesai/teks (Enter=Selesai/batal, q=batal)"
        while True:
            total = len(filt)
            visible_pairs = filt[:15]
            shown = [_show_with_mark(it, oi, selected) for oi, it in visible_pairs]
            sel_info = f"terpilih {len(selected)}" if selected else "belum ada yang dipilih"
            self._popup_render(f"{title} · {sel_info}", query, shown, total)
            raw = self._popup_input(hint)
            if raw is None:
                return selected if selected else None
            s = (raw or "").strip()
            if not s:
                return selected if selected else None
            if s.lower() == "q":
                return None
            if s.lower() in ("done", "selesai"):
                return selected if selected else None
            parts = [p.strip() for p in s.split(",")]
            if parts and all(p.isdigit() for p in parts):
                ok = True
                for p in parts:
                    n = int(p)
                    if 1 <= n <= total:
                        oi = (visible_pairs[n - 1][0] if n <= len(visible_pairs) else filt[n - 1][0])
                        if oi not in selected:
                            selected.append(oi)
                    else:
                        _print(f"[red]Nomor di luar 1..{total}: {p}[/red]")
                        ok = False
                        break
                _print(f"[dim]terpilih {len(selected)}[/dim]")
                continue
            query = s
            filt = _filtered(query)
            continue

    # ------------------------------------------------------------------
    # TUI pick (fullscreen bila TTY, fallback popup line-based)
    # ------------------------------------------------------------------

    def _tui_pick(self, title: str, items: list, show: Optional[Callable] = None,
                  initial: str = "", multi: bool = False, _keys=None):
        """Pilih via _TUIPicker fullscreen; fallback _popup_pick bila non-TTY.

        single → Optional[int] (index `items`); multi=True → Optional[List[int]].
        Esc/q/Enter-kosong → None (JANGAN switch). `_keys` hanya injeksi uji.
        """
        if show is None:
            show = lambda x: x  # noqa: E731
        if not items:
            _print("[dim]Tidak ada pilihan.[/dim]")
            return None
        if _keys is not None:
            return _TUIPicker(title, items, show=show, initial=initial, multi=multi).pick(keys=_keys)
        if _TUIPicker.available():
            try:
                return _TUIPicker(title, items, show=show, initial=initial, multi=multi).pick()
            except Exception as e:
                logging.getLogger(__name__).warning(f"TUI picker gagal, fallback popup: {e}")
        if multi:
            return self._popup_pick_multi(title, items, show=show, initial=initial)
        return self._popup_pick(title, items, show=show, initial=initial)

    def _tui_pick_multi(self, title: str, items: list, show: Optional[Callable] = None,
                        initial: str = "", _keys=None) -> Optional[List[int]]:
        """Multi-pilih: Spasi toggle ✓, Enter selesai (min 1), Esc/q batal."""
        return self._tui_pick(title, items, show=show, initial=initial, multi=True, _keys=_keys)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self):
        # Single silence point (ERROR only; never crash on logging setup)
        try:
            logging.getLogger().setLevel(logging.ERROR)
            for name in list(logging.root.manager.loggerDict.keys()):
                logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass

        try:
            self.config = CodeAIConfig.load_from_file(self.config_path)
        except Exception:
            self.config = CodeAIConfig()

        hooks = HooksDispatcher("SYSTEM_HOOKS.md")
        agents = AgentsParser("AGENTS.md")

        try:
            self.orchestrator = Orchestrator(hooks, agents, config=self.config)
        except Exception as e:
            _print(f"[bold red]Gagal menyiapkan orchestrator: {e}[/bold red]")
            if self.verbose:
                traceback.print_exc()
            sys.exit(1)

    def _effort_id(self, eff: str = "") -> str:
        try:
            _m = {"low": "rendah", "medium": "sedang", "high": "tinggi"}
            return _m.get(str(eff or "").strip().lower(), str(eff or "—"))
        except Exception:
            return str(eff or "—")

    def _status_baku(self) -> Tuple[str, str, str]:
        try:
            _prov, _mod = self.get_active_info()
        except Exception:
            _prov, _mod = "", ""
        try:
            _prov = str(_prov or "").strip()
            _mod = str(_mod or "").strip()
        except Exception:
            _prov, _mod = "", ""
        try:
            _conn = self._list_connected_fast()
        except Exception:
            _conn = []
        try:
            _total = len(list(_conn or []))
        except Exception:
            _total = 0
        try:
            _provs = self._list_providers_fast()
            _n_conn = sum(1 for p in (_provs or []) if p.get("has_credentials"))
        except Exception:
            _n_conn = _total
        _ada = bool(_total > 0 or _n_conn > 0)
        if not _ada or not _prov or not _mod:
            return ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        try:
            _ids = [str(m.get("id", "")) for m in (_conn or [])]
        except Exception:
            _ids = []
        try:
            _aktif = f"{_prov}/{_mod}" if "/" not in str(_mod) else str(_mod)
        except Exception:
            _aktif = f"{_prov}/{_mod}"
        _pos = 0
        try:
            if _aktif in _ids:
                _pos = _ids.index(_aktif) + 1
            else:
                for _i, _m in enumerate(_conn or []):
                    try:
                        if str(_m.get("provider", "")).strip() == _prov:
                            _pos = _i + 1
                            break
                    except Exception:
                        continue
                if _pos == 0:
                    _pos = 1
        except Exception:
            _pos = 1
        try:
            _tot = _total if _total else max(1, _n_conn)
        except Exception:
            _tot = 1
        return (f"● Terhubung", f"{_prov}/{self._short(_mod)}", f"Model {_pos} dari {_tot}")

    def display_banner(self):
        # Banner 2 baris informatif: status DULU, lalu label model, lalu posisi.
        provider, model = self.get_active_info()
        agents_exists = os.path.exists("AGENTS.md")
        hooks_exists = os.path.exists("SYSTEM_HOOKS.md")
        try:
            _st, _lbl, _pos = self._status_baku()
        except Exception:
            _st, _lbl, _pos = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        ag = "ON" if agents_exists else "off"
        hk = "ON" if hooks_exists else "off"
        try:
            _eff_raw = getattr(getattr(self.config, "provider", None), "effort", None) or "—"
        except Exception:
            _eff_raw = "—"
        eff = self._effort_id(_eff_raw) if _eff_raw != "—" else "—"
        el = f"{self._last_elapsed:.1f}s" if getattr(self, "_last_elapsed", None) else "—"
        if RICH_AVAILABLE:
            console.print(f"[bold blue]CodeAI Harness[/bold blue] [dim]v0.1.0[/dim]  {_st} — {_lbl} — {_pos}")
            console.print(f"  [dim]AGENTS:{ag} HOOKS:{hk} · Kekuatan pikir:{eff} · Waktu:{el}  │  /bantuan /model /provider /combo /keluar (Ctrl-C alih)[/dim]")
        else:
            print(f"CodeAI Harness v0.1.0  |  {_st} — {_lbl} — {_pos}")
            print(f"AGENTS:{ag} HOOKS:{hk} · Kekuatan pikir:{eff} · Waktu:{el}  |  /bantuan /model /provider /combo /keluar (Ctrl-C alih)")

    def run(self):
        self.startup()
        self.display_banner()
        self.repl_loop()

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _ask_main(self) -> str:
        # Kotak pesan: status DULU, lalu label model, lalu posisi.
        try:
            _prov, _mod = self.get_active_info()
        except Exception:
            _prov, _mod = "?", "?"
        try:
            _prov = str(_prov).strip() or "?"
        except Exception:
            _prov = "?"
        try:
            _eff_raw = getattr(getattr(self.config, "provider", None), "effort", None) or "—"
            _eff = self._effort_id(_eff_raw) if _eff_raw != "—" else "—"
        except Exception:
            _eff = "—"
        try:
            _short = self._short(_mod)
        except Exception:
            _short = str(_mod)
        try:
            _st, _lbl, _pos = self._status_baku()
        except Exception:
            _st, _lbl, _pos = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        header = f"╭─ ❯ {_st} — {_lbl} — {_pos} ─"
        _petunjuk = "Ketik di sini lalu tekan Enter: pesan atau /perintah · contoh: /model gemini"
        if RICH_AVAILABLE:
            console.print("─── Tulis pesan ───")
            console.print(f"[bold cyan]{header}[/bold cyan] [dim]{_petunjuk}[/dim]")
            try:
                val = input("│ ❯ ")
            except (EOFError, KeyboardInterrupt):
                raise
            try:
                console.print("[dim]╰─[/dim]")
            except Exception:
                pass
            return val
        print("─── Tulis pesan ───")
        print(f"{header} ─ {_petunjuk}")
        try:
            val = input("│ ❯ ")
        except (EOFError, KeyboardInterrupt):
            raise
        print("╰─")
        return val

    def repl_loop(self):
        while True:
            try:
                user_input = self._ask_main()

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self.handle_command(user_input)
                else:
                    self.dispatch_task(user_input)

            except KeyboardInterrupt:
                _print("\n[dim]Terhenti (Ctrl-C). Ketik /quit (/keluar) untuk keluar atau /steer (/alih) <teks> untuk mengarahkan.[/dim]")
            except EOFError:
                break
            except Exception as e:
                _print(f"[bold red]Gagal menjawab: {e}[/bold red] [dim](ketik /bantuan)[/dim]")
                if self.verbose:
                    traceback.print_exc()

    # ------------------------------------------------------------------
    # Command dispatcher
    # ------------------------------------------------------------------

    def handle_command(self, cmd_line: str):
        parts = cmd_line.split(" ", 1)
        cmd  = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        dispatch = {
            "/quit":      lambda: self._cmd_quit(),
            "/exit":      lambda: self._cmd_quit(),
            "/q":         lambda: self._cmd_quit(),
            "/keluar":    lambda: self._cmd_quit(),
            "/status":    lambda: self.show_status(),
            "/s":         lambda: self.show_status(),
            "/keadaan":   lambda: self.show_status(),
            "/config":    lambda: self.show_config(),
            "/pengaturan": lambda: self.show_config(),
            "/steer":     lambda: self.steer_orchestrator(args) if args else _print("[yellow]Cara pakai: /steer (/alih) <perintah>[/yellow] [dim]Contoh: /steer lanjutkan[/dim]"),
            "/st":        lambda: self.steer_orchestrator(args) if args else _print("[yellow]Cara pakai: /steer (/alih) <perintah>[/yellow] [dim]Contoh: /steer lanjutkan[/dim]"),
            "/alih":      lambda: self.steer_orchestrator(args) if args else _print("[yellow]Cara pakai: /alih (/steer) <perintah>[/yellow] [dim]Contoh: /alih lanjutkan[/dim]"),
            "/providers": lambda: self.show_providers(args),
            "/p":         lambda: self.show_providers(args),
            "/daftar":    lambda: self.show_providers(args),
            "/provider":  lambda: self.handle_provider(args),
            "/models":    lambda: self.show_models(args) if args else _print("[yellow]Cara pakai: /models <penyedia>[/yellow] [dim]Contoh: /models gemini[/dim]"),
            "/model":     lambda: self.switch_model(args),
            "/m":         lambda: self.switch_model(args),
            "/effort":    lambda: self.handle_effort(args),
            "/e":         lambda: self.handle_effort(args),
            "/combo":     lambda: self.handle_combo(args),
            "/c":         lambda: self.handle_combo(args),
            "/history":   lambda: self.show_history(),
            "/riwayat":   lambda: self.show_history(),
            "/help":      lambda: self.show_help(),
            "/h":         lambda: self.show_help(),
            "/bantuan":   lambda: self.show_help(),
        }

        handler = dispatch.get(cmd)
        if handler:
            handler()
        else:
            try:
                _cands = list(dispatch.keys())
                _m = difflib.get_close_matches(cmd, _cands, n=1, cutoff=0.6)
                _hint = f"  [dim]Mungkin maksud Anda {_m[0]}?[/dim]" if _m else "  [dim](ketik /bantuan)[/dim]"
            except Exception:
                _hint = "  [dim](ketik /bantuan)[/dim]"
            _print(f"[red]Perintah tidak dikenal: {cmd}[/red]{_hint} [dim]Contoh: /bantuan[/dim]")

    def _cmd_quit(self):
        _print("[dim]Sampai jumpa.[/dim]")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _word_chunks(text: str) -> List[str]:
        """Pecah teks jadi chunk per-kata (whitespace dipertahankan) untuk flush inkremental.

        Stdlib only (re). Return [] untuk teks kosong.
        """
        import re as _re
        t = text if isinstance(text, str) else str(text)
        if not t:
            return []
        try:
            parts = _re.findall(r"\S+\s*", t)
        except Exception:
            parts = []
        return parts or [t]

    def dispatch_task(self, task: str):
        import threading
        provider, model = self.get_active_info()
        short = self._short(model)
        t0 = time.monotonic()
        # Pipeline token thread-safe: _on_token dipanggil dari worker saat
        # chunk tiba (stream asli bila gateway/provider support) atau dari
        # fallback progressive-reveal; renderer membaca snapshot tiap 0.1s.
        # Flush per chunk via sys.stdout (stdlib only, tanpa dep baru).
        _tok_lock = threading.Lock()
        _tokens: List[str] = []
        _streamed = {"n": 0}
        _done = threading.Event()
        try:
            _spin_lock = self._spin_lock
        except AttributeError:
            _spin_lock = self._spin_lock = threading.Lock()
        try:
            _chat_lock = self._chat_lock
        except AttributeError:
            _chat_lock = self._chat_lock = threading.Lock()

        def _on_token(tok) -> None:
            try:
                if _done.is_set():
                    return
            except Exception:
                pass
            try:
                t = tok if isinstance(tok, str) else str(tok)
            except Exception:
                return
            if not t:
                return
            with _tok_lock:
                _tokens.append(t)
                _streamed["n"] += 1
            try:
                sys.stdout.flush()
            except Exception:
                pass

        def _snapshot(max_chars: int = 1200) -> str:
            with _tok_lock:
                s = "".join(_tokens)
            if len(s) > max_chars:
                s = "… " + s[-max_chars:]
            return s

        # Stream probe: dukung `stream`/`on_token`/`on_chunk`/`callback`?
        # Gateway kini chat(messages,tools,model)->dict (non-stream) → False;
        # worker tetap pasang on_token best-effort (TypeError → plain call).
        try:
            import inspect as _insp
            _gw0 = getattr(self.orchestrator, "gateway", None) if self.orchestrator else None
            _pnames: set = set()
            if _gw0 is not None and hasattr(_gw0, "chat"):
                try:
                    _pnames |= set(_insp.signature(_gw0.chat).parameters)
                except Exception:
                    pass
                try:
                    _pv0 = _gw0.get_provider(_gw0.config.default) if hasattr(_gw0, "get_provider") else None
                    if _pv0 is not None and hasattr(_pv0, "chat"):
                        _pnames |= set(_insp.signature(_pv0.chat).parameters)
                except Exception:
                    pass
            _stream_supported = bool(_pnames & {"stream", "on_token", "on_chunk", "callback"})
        except Exception:
            _stream_supported = False

        _box: dict = {}

        def _run():
            _orig = None
            _did_patch = False
            _gw2 = getattr(self.orchestrator, "gateway", None) if self.orchestrator else None
            try:
                # Teruskan on_token ke gateway bila didukung; TypeError → plain.
                if _gw2 is not None and _stream_supported and hasattr(_gw2, "chat"):
                    try:
                        import functools as _ft
                        try:
                            _already = bool(getattr(_gw2.chat, "_codeai_patched", False))
                        except Exception:
                            _already = False
                        if _already:
                            _orig = None
                        else:
                            try:
                                _got_lock = _chat_lock.acquire(timeout=1.0)
                            except Exception:
                                _got_lock = False
                            try:
                                try:
                                    _recheck = bool(getattr(_gw2.chat, "_codeai_patched", False))
                                except Exception:
                                    _recheck = False
                                if _recheck:
                                    _orig = None
                                else:
                                    _orig = _gw2.chat

                                    @_ft.wraps(_orig)
                                    def _patched(messages, tools=None, model=None, **kw):
                                        kw.setdefault("on_token", _on_token)
                                        try:
                                            return _orig(messages, tools, model, **kw)
                                        except TypeError:
                                            kw.pop("on_token", None)
                                            try:
                                                return _orig(messages, tools, model, **kw)
                                            except TypeError:
                                                if tools is not None:
                                                    return _orig(messages, tools)
                                                return _orig(messages)
                                    try:
                                        _patched._codeai_patched = True
                                    except Exception:
                                        pass
                                    _gw2.chat = _patched
                                    _did_patch = True
                            finally:
                                try:
                                    if _got_lock:
                                        _chat_lock.release()
                                except Exception:
                                    pass
                    except Exception:
                        _orig = None
                        _did_patch = False
                try:
                    if _done.is_set():
                        return
                except Exception:
                    pass
                _res = self.orchestrator.run_task(task)
                try:
                    if _done.is_set():
                        return
                except Exception:
                    pass
                _box["r"] = _res
            except Exception as e:
                try:
                    if _done.is_set():
                        return
                except Exception:
                    pass
                try:
                    _box["e"] = f"❌ Gagal menjawab: {e}"
                except Exception:
                    pass
            finally:
                try:
                    if _gw2 is not None and _orig is not None and _did_patch:
                        # Join worker selesai sebelum restore; kembalikan patch asli.
                        try:
                            if getattr(getattr(_gw2, "chat", None), "_codeai_patched", False):
                                _gw2.chat = _orig
                        except Exception:
                            try:
                                _gw2.chat = _orig
                            except Exception:
                                pass
                except Exception:
                    pass

        _th = threading.Thread(target=_run, daemon=True)
        _th.start()
        if RICH_AVAILABLE and Live is not None and Spinner is not None:
            from rich.console import Group as _Group
            from rich.text import Text as _Text
            # SATU Spinner dipertahankan (jangan recreate per tick — itu yang
            # membekukan animasi dots); hanya .text dimutasi (elapsed hidup).
            # Label provider PENUH (bukan "AI") + model short.
            _spin = Spinner("dots", text=f" orchestrator berpikir · {provider}/{short} · 0.0s · Ctrl-C alih", style="cyan")

            def _renderable():
                _dt = time.monotonic() - t0
                try:
                    with _spin_lock:
                        _spin.text = f" orchestrator berpikir · {provider}/{short} · {_dt:.1f}s · Ctrl-C alih"
                except Exception:
                    pass
                _body = _snapshot()
                if _body:
                    return _Group(_spin, _Text(_body))
                return _Group(_spin, _Text("menunggu balasan…", style="dim"))

            # Header jawaban: provider PENUH (mis. antigravity) + model short.
            # JANGAN "AI ◆" generik dan JANGAN potong provider via _short.
            _print(f"[dim]─ [/dim][bold green]{provider} ◆[/bold green][dim] · {short} ─[/dim]")
            try:
                with Live(_renderable(), console=console, refresh_per_second=10, transient=True) as _live:
                    while _th.is_alive():
                        try:
                            _live.update(_renderable())
                            time.sleep(0.1)
                        except KeyboardInterrupt:
                            # Ctrl-C = steer, JANGAN bunuh render/worker.
                            try:
                                _live.stop()
                            except Exception:
                                pass
                            _print("\n[dim]Alih (tugas tetap jalan — Enter kosong = lanjut)[/dim]")
                            try:
                                _s = input("alih ❯ ")
                            except (EOFError, KeyboardInterrupt):
                                _s = ""
                            if (_s or "").strip():
                                try:
                                    self.steer_orchestrator(_s.strip())
                                except Exception as _se:
                                    _print(f"[red]Gagal alih: {_se}[/red]")
                            try:
                                _live.start()
                            except Exception:
                                pass
                    # Fallback progressive-reveal: gateway non-streaming →
                    # alirkan hasil final kata-per-kata (flush per chunk) agar
                    # tetap terlihat mengalir. Live transient → Markdown final
                    # di bawah adalah satu-satunya salinan permanen.
                    _res0 = _box.get("r")
                    if not _streamed["n"] and isinstance(_res0, str) and _res0.strip() \
                            and not str(_res0).startswith(("❌", "🚫", "Task failed")):
                        try:
                            _chunks = self._word_chunks(_res0)
                            _n = len(_chunks)
                            _per = min(0.01, 0.6 / max(1, _n // 10)) if _n > 40 else 0.0
                            for _i, _ch in enumerate(_chunks):
                                _on_token(_ch)
                                if _n > 40 and (_i % 10 == 0):
                                    try:
                                        _live.update(_renderable())
                                    except Exception:
                                        pass
                                    if _per:
                                        time.sleep(_per)
                            try:
                                _live.update(_renderable())
                            except Exception:
                                pass
                        except (EOFError, KeyboardInterrupt):
                            pass
                        except Exception:
                            pass
            except Exception:
                pass
            _th.join(timeout=5)
            try:
                _alive = _th.is_alive()
            except Exception:
                _alive = False
            try:
                with _tok_lock:
                    result = _box.get("r", _box.get("e", "_Tidak ada jawaban._"))
            except Exception:
                try:
                    result = _box.get("r", _box.get("e", "_Tidak ada jawaban._"))
                except Exception:
                    result = "_Tidak ada jawaban._"
            try:
                _done.set()
            except Exception:
                pass
            result = result or "_Tidak ada jawaban._"
            dt = time.monotonic() - t0
            self._last_elapsed = dt
            try:
                _chk = result.get("content", "") if isinstance(result, dict) else str(result)
            except Exception:
                _chk = str(result)
            failed = str(_chk).startswith(("❌", "🚫", "Task failed"))
            if failed:
                try:
                    _detail = self._format_task_error(provider, model, str(_chk))
                except Exception:
                    _detail = ""
                if _detail:
                    _print(f"{_chk}\n{_detail}  [dim](ketik /bantuan)[/dim]", style="red")
                else:
                    _print(f"{_chk}  [dim](jalankan /provider atau /model — ketik /bantuan)[/dim]", style="red")
            else:
                try:
                    _content = result.get("content", "") if isinstance(result, dict) else str(result)
                except Exception:
                    _content = str(result)
                _print(f"[dim]· {provider}/{short} · {dt:.1f}s ─[/dim]")
                try:
                    console.print(Markdown(str(_content)))
                except Exception:
                    console.print(str(_content))
                try:
                    if str(provider or "").strip().lower() == "combo":
                        self._combo_footer(model, result)
                except Exception:
                    pass
        elif RICH_AVAILABLE:
            # Rich ada tapi Live/Spinner tak tersedia → Status statis (legacy).
            # Worker sudah jalan di atas; tinggal tunggu (jangan run ulang).
            with Status(f"[dim]orchestrator berpikir · {provider}/{short} · Ctrl-C alih[/dim]", spinner="dots", spinner_style="cyan"):
                _th.join(timeout=5)
            try:
                _alive2 = _th.is_alive()
            except Exception:
                _alive2 = False
            try:
                with _tok_lock:
                    result = _box.get("r", _box.get("e", "_Tidak ada jawaban._"))
            except Exception:
                try:
                    result = _box.get("r", _box.get("e", "_Tidak ada jawaban._"))
                except Exception:
                    result = "_Tidak ada jawaban._"
            try:
                _done.set()
            except Exception:
                pass
            result = result or "_Tidak ada jawaban._"
            dt = time.monotonic() - t0
            self._last_elapsed = dt
            try:
                _chk2 = result.get("content", "") if isinstance(result, dict) else str(result)
            except Exception:
                _chk2 = str(result)
            failed = str(_chk2).startswith(("❌", "🚫", "Task failed"))
            if failed:
                try:
                    _detail2 = self._format_task_error(provider, model, str(_chk2))
                except Exception:
                    _detail2 = ""
                if _detail2:
                    _print(f"{_chk2}\n{_detail2}  [dim](ketik /bantuan)[/dim]", style="red")
                else:
                    _print(f"{_chk2}  [dim](jalankan /provider atau /model — ketik /bantuan)[/dim]", style="red")
            else:
                try:
                    _content2 = result.get("content", "") if isinstance(result, dict) else str(result)
                except Exception:
                    _content2 = str(result)
                _print(f"[dim]─ [/dim][bold green]{provider} ◆[/bold green][dim] · {short} · {dt:.1f}s ─[/dim]")
                try:
                    console.print(Markdown(str(_content2)))
                except Exception:
                    console.print(str(_content2))
                try:
                    if str(provider or "").strip().lower() == "combo":
                        self._combo_footer(model, result)
                except Exception:
                    pass
        else:
            # Non-rich: ticker elapsed via \r + token mengalir inline (flush).
            # Label provider PENUH + model short.
            print(f"orchestrator berpikir · {provider}/{short} · Ctrl-C untuk alih …")
            _shown = {"n": 0}
            _flowing = {"on": False}
            try:
                while _th.is_alive():
                    try:
                        with _tok_lock:
                            _new = "".join(_tokens[_shown["n"]:])
                            _shown["n"] = len(_tokens)
                        if _new:
                            if not _flowing["on"]:
                                sys.stdout.write("\n")
                                _flowing["on"] = True
                            sys.stdout.write(_new)
                            sys.stdout.flush()
                        else:
                            # Setelah token mengalir, ticker \r dimatikan agar
                            # tidak menimpa baris token (elapsed ada di footer).
                            if not _flowing["on"]:
                                _dt2 = time.monotonic() - t0
                                sys.stdout.write(f"\r  · {_dt2:.1f}s …")
                                sys.stdout.flush()
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        # Ctrl-C = steer, jangan bunuh render/worker.
                        try:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        except Exception:
                            pass
                        print("Alih (tugas tetap jalan — Enter kosong = lanjut)")
                        try:
                            _s2 = input("alih ❯ ")
                        except (EOFError, KeyboardInterrupt):
                            _s2 = ""
                        if (_s2 or "").strip():
                            try:
                                self.steer_orchestrator(_s2.strip())
                            except Exception as _se2:
                                print(f"Gagal alih: {_se2}")
            except Exception:
                pass
            _th.join(timeout=5)
            try:
                _alive3 = _th.is_alive()
            except Exception:
                _alive3 = False
            # Ekor token yang tiba setelah tick terakhir → alirkan dulu.
            try:
                with _tok_lock:
                    _tail = "".join(_tokens[_shown["n"]:])
                    _shown["n"] = len(_tokens)
                if _tail:
                    if not _flowing["on"]:
                        sys.stdout.write("\n")
                        _flowing["on"] = True
                    sys.stdout.write(_tail)
                    sys.stdout.flush()
                elif not _flowing["on"]:
                    sys.stdout.write("\r" + " " * 24 + "\r")
            except Exception:
                pass
            try:
                with _tok_lock:
                    result = _box.get("r", _box.get("e", "Tidak ada jawaban."))
            except Exception:
                try:
                    result = _box.get("r", _box.get("e", "Tidak ada jawaban."))
                except Exception:
                    result = "Tidak ada jawaban."
            try:
                _done.set()
            except Exception:
                pass
            dt = time.monotonic() - t0
            self._last_elapsed = dt
            try:
                _chk3 = result.get("content", "") if isinstance(result, dict) else str(result)
            except Exception:
                _chk3 = str(result)
            if str(_chk3).startswith(("❌", "🚫", "Task failed")):
                try:
                    _detail3 = self._format_task_error(provider, model, str(_chk3))
                except Exception:
                    _detail3 = ""
                if _detail3:
                    print(f"\n─── Gagal menjawab ───\n{_chk3}\n{_detail3}\n[{dt:.1f}s]")
                else:
                    print(f"\n─── Gagal menjawab ───\n{_chk3}  (jalankan /provider atau /model — ketik /bantuan)\n[{dt:.1f}s]")
            elif _shown["n"] > 0:
                # Jawaban sudah mengalir live → cukup footer elapsed.
                print(f"\n─ {provider} ◆ · {short} · {dt:.1f}s ─ (selesai)")
                try:
                    if str(provider or "").strip().lower() == "combo":
                        self._combo_footer(model, result)
                except Exception:
                    pass
            else:
                try:
                    _c3b = result.get("content", "") if isinstance(result, dict) else str(result)
                except Exception:
                    _c3b = str(_chk3)
                # Fallback inkremental: flush kata-per-kata (stdlib only).
                for _ch2 in self._word_chunks(str(_c3b)):
                    try:
                        sys.stdout.write(_ch2)
                    except Exception:
                        break
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                print(f"\n─ {provider} ◆ · {short} · {dt:.1f}s ─")
                try:
                    if str(provider or "").strip().lower() == "combo":
                        self._combo_footer(model, result)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # _auth_provider (internal auth, via /provider)
    # ------------------------------------------------------------------

    def _auth_provider(self, provider_name: str = ""):
        provider_name = (provider_name or "").lower().strip()
        if not provider_name:
            # Tanpa arg → popup daftar provider ASLI + status connected.
            # Filter: kecualikan pseudo-provider `combo` (api==local) dari login.
            try:
                provs = [p for p in self._list_providers_fast()
                         if p.get("id") != "combo" and p.get("api") != "local"]
            except Exception:
                provs = []
            if not provs:
                options = [
                    ("antigravity", "Google Antigravity — OAuth session"),
                    ("gemini", "Google Gemini API — AI Studio API Key"),
                    ("copilot", "GitHub Copilot — OAuth Device Flow"),
                    ("openrouter", "OpenRouter — Multi-model gateway"),
                    ("deepseek", "DeepSeek — API Key"),
                    ("groq", "Groq — Fast Inference API Key"),
                    ("openai", "OpenAI — API Key"),
                    ("anthropic", "Anthropic — API Key"),
                    ("ollama", "Ollama — Local LLM"),
                ]
                items = [{"id": k, "label": f"{k} — {d}", "has_credentials": False} for k, d in options]
            else:
                items = [{"id": p["id"], "label": f"{'● Terhubung' if p.get('has_credentials') else '○ Belum terhubung'} — {p['id']} — {p.get('name', '')}", "has_credentials": bool(p.get("has_credentials"))} for p in provs]
            _pick = self._tui_pick("Penyedia — pilih", items, show=lambda x: x["label"], initial="")
            if _pick is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            provider_name = items[_pick]["id"]

        if not provider_name:
            return

        # Direct-arg guard (filter :955): tolak pseudo-provider combo (api==local).
        _pn = (provider_name or "").lower().strip()
        if _pn == "combo" or _pn.startswith("combo/"):
            _print("[red]Itu gabungan bukan penyedia[/red] [dim](gabungan dipakai via /model combo/<nama>, contoh: /model combo/andalan)[/dim]")
            return
        try:
            _desc = self._registry().get_provider_descriptor(_pn)
            if _desc is not None and str(_desc.get("api", "")) == "local":
                _print("[red]Itu gabungan bukan penyedia[/red] [dim](gabungan dipakai via /model combo/<nama>, contoh: /model combo/andalan)[/dim]")
                return
        except Exception:
            pass

        _print(f"\n[cyan]Menyambung ke {provider_name}…[/cyan]")

        # ---- Antigravity (agy CLI, OAuth session) ----
        if provider_name == "antigravity":
            self._login_antigravity()

        # ---- Copilot ----
        elif provider_name == "copilot":
            try:
                from harness.models.providers.copilot import CopilotProvider
                prov = CopilotProvider()
                prov.device_flow_login()
                _print("[bold green]✅ Copilot Terhubung.[/bold green]")
                self._ask_switch_provider("copilot", PROVIDER_DEFAULT_MODELS.get("copilot", lambda: "gpt-4o")())
            except Exception as e:
                _print(f"[bold red]Gagal masuk: {e}[/bold red] [dim]Contoh: /provider copilot[/dim]")

        # ---- Gemini API (AI Studio API key) ----
        elif provider_name == "gemini":
            self._login_gemini()

        # ---- Generic API key providers ----
        else:
            vault_token = Prompt.ask(f"Masukkan kunci API {provider_name}") if RICH_AVAILABLE else input(f"Masukkan kunci API {provider_name}: ")
            vault_token = vault_token.strip()
            if vault_token:
                from harness.models.auth_vault import AuthVault
                AuthVault().store_token(provider_name, vault_token)
                default_model = PROVIDER_DEFAULT_MODELS.get(provider_name, lambda: "default")()
                _print(f"[bold green]✅ {provider_name} tersimpan.[/bold green]")
                self._ask_switch_provider(provider_name, default_model)

    def _login_antigravity(self):
        """
        Connect Google Antigravity via Google OAuth intercept, with vault
        session fast-path and agy-file fallback. Never prints tokens
        (only email/status).
        """
        import os
        from harness.models.auth_vault import AuthVault
        from harness.models.providers.antigravity import DEFAULT_ANTIGRAVITY_MODEL

        vault = AuthVault()

        def _safe_email() -> str:
            try:
                cred = vault.get_credential("antigravity")
                if isinstance(cred, dict):
                    em = str(cred.get("email") or "").strip()
                    return em
            except Exception:
                pass
            return ""

        # (a) sesi lama via vault — masih valid → connected langsung + tawar switch.
        # Fast-path file session (kompat lama): discover → store → connected.
        # Dianggap bagian dari "sesi lama via vault object" agar /provider tetap
        # instan bila sesi agy sudah ada (tanpa membuka browser).
        try:
            _stored = vault.get_token("antigravity")
        except Exception:
            _stored = None
        try:
            _existing_file = vault.discover_antigravity_token()
        except Exception:
            _existing_file = None
        if _existing_file:
            try:
                vault.store_token("antigravity", _existing_file)
            except Exception:
                pass
            _em0 = _safe_email()
            if _em0:
                _print(f"[green]✅ Sesi Antigravity lama ditemukan ({_em0}).[/green]")
            else:
                _print("[green]✅ Sesi Antigravity lama ditemukan.[/green]")
            _print(f"[bold green]✅ Antigravity Terhubung. Model bawaan: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
            try:
                self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
            except Exception:
                pass
            return
        if _stored:
            _em0b = _safe_email()
            if _em0b:
                _print(f"[green]✅ Sesi Antigravity lama ditemukan ({_em0b}).[/green]")
            else:
                _print("[green]✅ Sesi Antigravity lama ditemukan.[/green]")
            _print(f"[bold green]✅ Antigravity Terhubung. Model bawaan: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
            try:
                self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
            except Exception:
                pass
            return

        # (d) pre-check client creds kosong → panduan, JANGAN crash; tetap coba OAuth.
        try:
            _cid = (os.environ.get("GOOGLE_CLIENT_ID") or "").strip()
            _csec = (os.environ.get("GOOGLE_CLIENT_SECRET") or "").strip()
            if not _cid or not _csec:
                try:
                    from harness.models.google_oauth import _resolve_client_id as _rcid
                    from harness.models.google_oauth import _resolve_client_secret as _rsec
                    if not _cid:
                        try:
                            _cid = str(_rcid() or "").strip()
                        except Exception:
                            pass
                    if not _csec:
                        try:
                            _csec = str(_rsec() or "").strip()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            _cid, _csec = "", ""
        _missing_creds = (not _cid or not _csec)
        if _missing_creds:
            _print("[yellow]GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET belum diset — pertukaran OAuth kemungkinan Gagal.[/yellow]")
            _print("[dim]Isi env GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau masuk via `agy` sekali lalu ulangi /provider antigravity.[/dim]")

        # (b) Google OAuth intercept → dict/str creds → store → project.
        _creds = None
        _intercept_error = ""
        try:
            from harness.models.google_oauth import GoogleOAuthInterceptor
            try:
                _creds = GoogleOAuthInterceptor().intercept()
            except TypeError:
                _creds = GoogleOAuthInterceptor().intercept(120)
        except Exception as e:
            # (d) exchange/intercept gagal → instruksi, JANGAN crash → lanjut fallback (e).
            try:
                _intercept_error = str(e or "").strip()
            except Exception:
                _intercept_error = ""
            _print("[red]Gagal intersep OAuth.[/red]")
            if _missing_creds or "client" in _intercept_error.lower() or "secret" in _intercept_error.lower() or "exchange" in _intercept_error.lower():
                _print("[dim]Isi GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau masuk via `agy` sekali lalu ulangi /provider antigravity.[/dim]")
            _creds = None

        # (c) intercept None (user batal/timeout) → pesan batal jelas, lalu fallback (e).
        if _creds is None:
            if not _intercept_error:
                _print("[yellow]Masuk Dibatalkan (waktu habis/pengguna membatalkan di peramban).[/yellow]")
            # (e) fallback terakhir: baca sesi agy (mungkin dibuat selama window OAuth).
            try:
                _fb = vault.discover_antigravity_token()
            except Exception:
                _fb = None
            if _fb:
                try:
                    vault.store_token("antigravity", _fb)
                except Exception:
                    pass
                _print(f"[bold green]✅ Antigravity Terhubung. Model bawaan: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
                try:
                    self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
                except Exception:
                    pass
                return
            if _intercept_error:
                _print("[dim]Tidak ada sesi agy cadangan. Jalankan `agy` untuk masuk lalu ulangi /provider antigravity, atau ulangi OAuth setelah isi client creds.[/dim]")
            else:
                _print("[dim]Tidak ada sesi agy cadangan. Jalankan `agy` untuk masuk lalu ulangi /provider antigravity, atau ulangi OAuth.[/dim]")
            return

        # Normalisasi creds: dict (baru) atau str (lama). Masking: tak pernah print token.
        _access = ""
        _refresh = ""
        _expires_ms = 0
        _email = ""
        _project = ""
        try:
            if isinstance(_creds, dict):
                _access = str(_creds.get("access_token") or _creds.get("access") or _creds.get("token") or "").strip()
                _refresh = str(_creds.get("refresh_token") or _creds.get("refresh") or "").strip()
                _email = str(_creds.get("email") or "").strip()
                _project = str(_creds.get("project_id") or _creds.get("projectId") or _creds.get("project") or "").strip()
                try:
                    _raw_exp = _creds.get("expires") if _creds.get("expires") is not None else _creds.get("expires_in")
                    if isinstance(_raw_exp, (int, float)) and _raw_exp > 0:
                        _f = float(_raw_exp)
                        # Heuristik: >1e12 = epoch ms, >1e9 = epoch s, else durasi detik.
                        if _f > 1e12:
                            _expires_ms = int(_f)
                        elif _f > 1e9:
                            _expires_ms = int(_f * 1000)
                        else:
                            import time as _tm
                            _expires_ms = int(_tm.time() * 1000) + int(_f * 1000)
                except Exception:
                    _expires_ms = 0
            elif isinstance(_creds, str):
                _access = _creds.strip()
                # Refresh pendamping (bila interceptor menyimpannya di handler).
                try:
                    from harness.models.google_oauth import OAuthCallbackHandler as _H
                    _refresh = str(getattr(_H, "refresh_token", "") or "").strip()
                except Exception:
                    _refresh = ""
            else:
                _access = ""
        except Exception:
            _access = ""
        if not _access:
            # (d) exchange gagal/creds kosong → instruksi, JANGAN crash → fallback (e).
            _print("[red]Gagal pertukaran OAuth (kredensial kosong).[/red]")
            _print("[dim]Isi GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau masuk via `agy` sekali lalu ulangi /provider antigravity.[/dim]")
            try:
                _fb2 = vault.discover_antigravity_token()
            except Exception:
                _fb2 = None
            if _fb2:
                try:
                    vault.store_token("antigravity", _fb2)
                except Exception:
                    pass
                _print(f"[bold green]✅ Antigravity Terhubung. Model bawaan: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
                try:
                    self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
                except Exception:
                    pass
                return
            _print("[dim]Tidak ada sesi agy cadangan. Jalankan `agy` untuk masuk lalu ulangi /provider antigravity.[/dim]")
            return

        # Simpan OAuth: prefer store_antigravity_oauth bila ada, else store_oauth.
        try:
            _store_agy = getattr(vault, "store_antigravity_oauth", None)
            if callable(_store_agy):
                try:
                    if isinstance(_creds, dict):
                        _store_agy(_creds)
                    else:
                        _store_agy({"access_token": _access, "refresh_token": _refresh, "expires": _expires_ms, "email": _email})
                except TypeError:
                    try:
                        vault.store_oauth("antigravity", _access, _refresh, _expires_ms, _email, _project)
                    except Exception:
                        vault.store_token("antigravity", _access)
            else:
                try:
                    vault.store_oauth("antigravity", _access, _refresh, _expires_ms, _email, _project)
                except Exception:
                    vault.store_token("antigravity", _access)
        except Exception:
            try:
                vault.store_token("antigravity", _access)
            except Exception:
                pass

        # Discover+store project bila didukung; never crash, never print token.
        try:
            _discover_proj = getattr(vault, "discover_antigravity_project", None)
            if callable(_discover_proj):
                _proj = _discover_proj()
                _pid = ""
                try:
                    if isinstance(_proj, dict):
                        _pid = str(_proj.get("projectId") or _proj.get("project_id") or _proj.get("project") or "").strip()
                    elif isinstance(_proj, str):
                        _pid = _proj.strip()
                except Exception:
                    _pid = ""
                if _pid:
                    try:
                        _cur = vault.get_credential("antigravity")
                        _cur_access = _access
                        _cur_refresh = _refresh
                        _cur_exp = _expires_ms
                        _cur_email = _email or (str(_cur.get("email") or "") if isinstance(_cur, dict) else "")
                        vault.store_oauth("antigravity", _cur_access, _cur_refresh, _cur_exp, _cur_email, _pid)
                    except Exception:
                        pass
        except Exception:
            pass

        # Sukses → pesan verbatim + tawar switch. Hanya email/status, tanpa token.
        _shown_email = _email or _safe_email()
        if _shown_email:
            _print(f"[bold green]✅ Antigravity Terhubung via Google OAuth ({_shown_email})[/bold green]")
        else:
            _print("[bold green]✅ Antigravity Terhubung via Google OAuth[/bold green]")
        try:
            self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
        except Exception:
            pass

    def _login_gemini(self):
        """Connect Gemini via Google AI Studio API Key (for the REST generative language API)."""
        import webbrowser
        from harness.models.auth_vault import AuthVault
        vault = AuthVault()

        _print("[dim]Gemini API (Google AI Studio) — untuk gemini-2.5-flash, gemini-2.5-pro, dan lain-lain.[/dim]")
        _print("[dim]Untuk model Antigravity (gemini-3.8, claude, gpt-oss) pakai /provider antigravity.[/dim]")
        _print("[dim]Membuka https://aistudio.google.com/app/apikey …[/dim]")

        try:
            webbrowser.open("https://aistudio.google.com/app/apikey")
        except Exception:
            pass

        token = Prompt.ask("Tempel kunci API") if RICH_AVAILABLE else input("Kunci API: ")
        token = token.strip()
        if token:
            vault.store_token("gemini", token)
            _print("[bold green]✅ Kunci API Gemini tersimpan.[/bold green]")
            self._ask_switch_provider("gemini", "gemini-2.5-flash")
        else:
            _print("[yellow]Tidak ada kunci — Dibatalkan.[/yellow]")

    def _persist_config(self) -> None:
        """Persist provider.default+active_model+effort to config_path (atomic, never crash)."""
        try:
            # Prefer native saver if a future CodeAIConfig provides one.
            save_fn = getattr(self.config, "save_to_file", None)
            if callable(save_fn):
                save_fn(self.config_path)
                return
            import json
            import tempfile
            data = self.config.model_dump() if hasattr(self.config, "model_dump") else dict(self.config.__dict__)
            parent = os.path.dirname(os.path.abspath(self.config_path))
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            target_dir = parent or "."
            fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".codeai-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp, self.config_path)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                raise
        except Exception as e:
            logging.getLogger(__name__).warning(f"persist config to {self.config_path} failed: {e}")

    @staticmethod
    def _normalize_effort_syntax(raw: str) -> str:
        """Normalize `base:effort` -> `base-effort` (and `provider/base:effort`), lowercase effort."""
        s = (raw or "").strip()
        if not s:
            return s
        # Split optional provider/ prefix so colon handling applies to model part only.
        prefix = ""
        rest = s
        if "/" in s:
            prefix, rest = s.split("/", 1)
            prefix = prefix.strip()
        # Colon form takes precedence: base:effort (last colon separates effort).
        if ":" in rest:
            base_part, eff_part = rest.rsplit(":", 1)
            base_part, eff_part = base_part.strip(), eff_part.strip().lower()
            try:
                from harness.models.providers.antigravity import EFFORT_OPTIONS
                valid = set(EFFORT_OPTIONS)
            except Exception:
                valid = {"low", "medium", "high"}
            if eff_part in valid and base_part:
                rest = f"{base_part}-{eff_part}"
            else:
                # Not an effort suffix — keep as-is (colon may belong to model name).
                rest = f"{base_part}:{eff_part}" if base_part else rest
        else:
            # Dash form: lowercase trailing effort if it matches known options.
            try:
                from harness.models.providers.antigravity import EFFORT_OPTIONS
                valid = set(EFFORT_OPTIONS)
            except Exception:
                valid = {"low", "medium", "high"}
            for eff in valid:
                if rest.lower().endswith(f"-{eff}") and len(rest) > len(eff) + 1:
                    rest = rest[: -(len(eff) + 1)] + f"-{eff}"
                    break
        return f"{prefix}/{rest}" if prefix else rest

    def _set_active_provider(self, provider: str, model: str):
        """Update config and live orchestrator/gateway with new provider+model (persisted)."""
        if not self.config:
            self.config = CodeAIConfig()
        self.config.provider.default = provider
        self.config.provider.active_model = model
        # Keep provider.effort in sync when model carries an effort suffix.
        try:
            from harness.models.providers.antigravity import parse_model_effort
            _base, _eff = parse_model_effort(model)
            if _eff:
                try:
                    setattr(self.config.provider, "effort", _eff)
                except Exception:
                    pass
        except Exception:
            pass
        if self.orchestrator and getattr(self.orchestrator, "gateway", None):
            try:
                if hasattr(self.orchestrator.gateway, "config"):
                    self.orchestrator.gateway.config.default = provider
                    self.orchestrator.gateway.config.active_model = model
                    try:
                        _b2, _e2 = parse_model_effort(model)
                        if _e2:
                            setattr(self.orchestrator.gateway.config, "effort", _e2)
                    except Exception:
                        pass
                self.orchestrator.gateway.active_model = model
            except Exception as e:
                logging.getLogger(__name__).warning(f"mirror active model to gateway failed: {e}")
        try:
            self.orchestrator.compactor.set_active_model(f"{provider}/{model}")
        except Exception:
            pass
        self._persist_config()

    def _ask_switch_provider(self, provider: str, model: str) -> None:
        """
        After a successful login, ask whether the user wants to switch to
        the newly authenticated provider immediately.
        Credentials are already stored at this point regardless of the answer.
        """
        current_provider, _ = self.get_active_info()
        if current_provider == provider:
            # Already using this provider — just update the model silently
            self._set_active_provider(provider, model)
            return

        if RICH_AVAILABLE:
            sw = Prompt.ask(
                f"Pindah ke penyedia aktif [cyan]{provider}[/cyan] / [yellow]{model}[/yellow]? Contoh: y",
                choices=["y", "n"],
                default="n",
            )
        else:
            sw = input(f"Pindah ke {provider}/{model}? Contoh y [y/N]: ").strip().lower()

        if sw == "y":
            self._set_active_provider(provider, model)
            _print(f"[green]Kini memakai [bold]{provider}[/bold] / [bold]{model}[/bold][/green]")
        else:
            _print(f"[dim]Tersimpan. Masih memakai {current_provider}. Jalankan /model untuk pindah kapan saja. Contoh: /model gemini[/dim]")

    # ------------------------------------------------------------------
    # /providers
    # ------------------------------------------------------------------

    def show_providers(self, filter_q: str = ""):
        providers = self._list_providers_fast()
        q = (filter_q or "").strip().lower()
        if q:
            providers = [p for p in providers if q in p["id"].lower() or q in str(p.get("name", "")).lower()]
        try:
            _st0, _lbl0, _pos0 = self._status_baku()
        except Exception:
            _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")

        if RICH_AVAILABLE:
            table = Table(title=f"Penyedia Cari: {filter_q.strip() if filter_q.strip() else '—'} — {_st0} — {_lbl0} — {_pos0}", show_header=True, header_style="bold blue")
            table.add_column("Status", style="green", no_wrap=True)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Nama")
            for p in providers:
                status = "[green]● Terhubung[/green]" if p["has_credentials"] else "[dim]○ Belum terhubung[/dim]"
                table.add_row(status, p["id"], p.get("name", p["id"]))
            console.print()
            console.print(table)
            console.print("[dim]Cara pakai: nomor=pilih · teks=Saring · /providers (/daftar) teks untuk menyaring · contoh: /providers gemini[/dim]")
            console.print()
        else:
            print(f"\n╭─ Penyedia Cari: {filter_q.strip() if filter_q.strip() else '—'} — {_st0} — {_lbl0} — {_pos0} ─" + "─" * 20 + "╮")
            for p in providers:
                s = "● Terhubung" if p["has_credentials"] else "○ Belum terhubung"
                print(f"│   {s} — {p['id']} ({p.get('name', '')})")
            print("│ Cara pakai: nomor=pilih · teks=Saring · /providers (/daftar) teks untuk menyaring · contoh: /providers gemini")
            print("╰" + "─" * 40 + "╯\n")

    # ------------------------------------------------------------------
    # /models
    # ------------------------------------------------------------------

    def show_models(self, provider_id: str):
        registry = self._registry()
        models = registry.list_models(provider_id)
        try:
            _st0, _lbl0, _pos0 = self._status_baku()
        except Exception:
            _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        if not models:
            _print(f"[yellow]Tidak ada model untuk '{provider_id}'.[/yellow] [dim]{_st0} — {_lbl0} — {_pos0} · Contoh: /model gemini[/dim]")
            return
        _print(f"\n[bold]{_st0} — {_lbl0} — {_pos0}[/bold]")
        _print(f"[bold]Model untuk {provider_id}:[/bold] [dim]Cari: ketik kata kunci · contoh: gemini[/dim]")
        for i, m in enumerate(models, 1):
            _print(f"  [cyan]{i}. {m}[/cyan]")
        _print(f"[dim]Cara pakai: ketik untuk mencari · tombol atas bawah untuk pindah · Enter untuk pilih · Esc untuk batal · contoh: /model {provider_id}[/dim]")

    # ------------------------------------------------------------------
    # /provider (auth unified builtin+custom, stdlib only)
    # ------------------------------------------------------------------

    def handle_provider(self, args: str = ""):
        """Auth/switch provider (builtin+custom): /provider [id|add|list|remove]. Tanpa arg → picker."""
        parts = (args or "").strip().split()
        sub = parts[0].lower() if parts else ""
        rest = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        if sub == "add":
            self._provider_add()
            return
        if sub == "list":
            self._provider_list()
            return
        if sub == "remove":
            self._provider_remove(rest)
            return
        if not sub:
            try:
                provs = [p for p in self._list_providers_fast()
                         if p.get("id") != "combo" and p.get("api") != "local"]
            except Exception:
                provs = []
            if not provs:
                self._provider_list()
                return
            items = [{"id": p["id"], "label": f"{'● Terhubung' if p.get('has_credentials') else '○ Belum terhubung'} — {p['id']} — {p.get('name', '')}", "has_credentials": bool(p.get("has_credentials"))} for p in provs]
            _pick = self._tui_pick("Penyedia — pilih", items, show=lambda x: x["label"], initial="")
            if _pick is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            _pid = items[_pick]["id"]
            _has = bool(items[_pick].get("has_credentials"))
            if _has:
                try:
                    _conn = self._list_connected_fast()
                except Exception:
                    _conn = []
                _mine = [m for m in (_conn or []) if str(m.get("provider", "")).lower() == str(_pid).lower()]
                if _mine:
                    try:
                        self.switch_model(_mine[0]["id"])
                    except Exception:
                        try:
                            self._set_active_provider(_pid, _mine[0].get("model", "default"))
                        except Exception:
                            pass
                    return
                self._auth_provider(_pid)
                return
            self._auth_provider(_pid)
            return
        _want = (parts[0] or "").strip()
        _low = _want.lower()
        _actual = None
        try:
            _provs2 = self._list_providers_fast()
        except Exception:
            _provs2 = []
        for _p in _provs2 or []:
            try:
                if str(_p.get("id", "")).strip().lower() == _low:
                    _actual = str(_p.get("id", "")).strip()
                    break
            except Exception:
                continue
        if _actual is None:
            try:
                _reg = self._registry()
                _keys = list(getattr(_reg, "_providers", {}).keys())
                for _k in _keys:
                    if isinstance(_k, str) and _k.strip().lower() == _low:
                        _actual = _k
                        break
            except Exception:
                pass
        if _actual is not None:
            self._auth_provider(_actual)
            return
        _print(f"[red]Penyedia tidak dikenal '{_want}'.[/red] [dim](lihat /provider list, contoh: /provider gemini)[/dim]")
        self._provider_list()
        return

    def _provider_prompt_secret(self, prompt_text: str = "Kunci API (opsional, Enter=kosong): ") -> str:
        """Minta api key tersembunyi (tak pernah echo/log). Stdlib getpass dulu."""
        try:
            import getpass as _gp
            try:
                _v = _gp.getpass(prompt_text)
                return (_v or "").strip()
            except (EOFError, KeyboardInterrupt):
                return ""
            except Exception:
                pass
        except Exception:
            pass
        try:
            if RICH_AVAILABLE:
                try:
                    _v2 = Prompt.ask(prompt_text.strip(), password=True, default="")
                    return (_v2 or "").strip()
                except TypeError:
                    pass
                except (EOFError, KeyboardInterrupt):
                    return ""
                except Exception:
                    pass
            _v3 = input(prompt_text)
            return (_v3 or "").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        except Exception:
            return ""

    def _provider_fetch_models(self, base_url: str, api_key: str = "", timeout: float = 10.0):
        """GET {base}/v1/models Bearer timeout 10, parse data[].id. Return (models, err)."""
        try:
            import json as _json
            import urllib.request as _ur
            base = (base_url or "").strip().rstrip("/")
            if not base:
                return [], "empty baseURL"
            models_url = base
            if not models_url.endswith("/models"):
                _clean = models_url[:-3] if models_url.endswith("/v1") else models_url
                models_url = _clean.rstrip("/") + "/v1/models"
            _headers = {"Accept": "application/json"}
            _key = (api_key or "").strip()
            if _key:
                _headers["Authorization"] = f"Bearer {_key}"
            _req = _ur.Request(models_url, headers=_headers, method="GET")
            with _ur.urlopen(_req, timeout=timeout) as _resp:
                _raw = _resp.read().decode("utf-8", errors="replace")
            _data = _json.loads(_raw) if (_raw or "").strip() else {}
            _models: list = []
            if isinstance(_data, dict) and isinstance(_data.get("data"), list):
                for _m in _data["data"]:
                    if isinstance(_m, dict) and isinstance(_m.get("id"), str) and _m["id"].strip():
                        _models.append(_m["id"].strip())
            elif isinstance(_data, dict) and isinstance(_data.get("models"), list):
                for _m in _data["models"]:
                    if isinstance(_m, str) and _m.strip():
                        _models.append(_m.strip())
                    elif isinstance(_m, dict):
                        _mid = _m.get("id") or _m.get("name") or ""
                        if isinstance(_mid, str) and _mid.strip():
                            _models.append(_mid.strip())
            return _models, ""
        except Exception as _e:
            try:
                return [], str(_e or "fetch failed")
            except Exception:
                return [], "fetch failed"

    def _provider_test_connection(self, base_url: str, api_key: str = "", timeout: float = 10.0):
        """Tes koneksi: HTTP respons apa pun (200/401/403/404) = reachable; hanya network error = gagal."""
        try:
            import urllib.request as _ur
            import urllib.error as _ue
            base = (base_url or "").strip().rstrip("/")
            if not base:
                return False, "empty baseURL"
            models_url = base
            if not models_url.endswith("/models"):
                _clean = models_url[:-3] if models_url.endswith("/v1") else models_url
                models_url = _clean.rstrip("/") + "/v1/models"
            _headers = {"Accept": "application/json"}
            _key = (api_key or "").strip()
            if _key:
                _headers["Authorization"] = f"Bearer {_key}"
            try:
                _req = _ur.Request(models_url, headers=_headers, method="GET")
                with _ur.urlopen(_req, timeout=timeout):
                    return True, ""
            except _ue.HTTPError:
                return True, ""
            except Exception as _e1:
                try:
                    _req2 = _ur.Request(base, headers={"Accept": "*/*"}, method="GET")
                    with _ur.urlopen(_req2, timeout=timeout):
                        return True, ""
                except _ue.HTTPError:
                    return True, ""
                except Exception as _e2:
                    try:
                        _msg = str(_e2 or _e1 or "unreachable")
                    except Exception:
                        _msg = "unreachable"
                    return False, _msg
        except Exception as _e:
            try:
                return False, str(_e or "unreachable")
            except Exception:
                return False, "unreachable"

    def _provider_add(self) -> None:
        import re as _re
        _BUILTIN = {"openai", "anthropic", "gemini", "antigravity", "copilot",
                    "ollama", "openrouter", "deepseek", "groq", "mistral",
                    "together", "xai", "cerebras", "fireworks", "perplexity",
                    "sambanova", "opencode", "combo"}
        # --- id ---
        try:
            _raw_pid = (Prompt.ask("ID penyedia [a-z0-9-], contoh: my-provider") if RICH_AVAILABLE else input("ID penyedia [a-z0-9-], contoh: my-provider: "))
        except (EOFError, KeyboardInterrupt):
            _print("[dim]Dibatalkan.[/dim]")
            return
        _pid = (_raw_pid or "").strip()
        if not _pid or not _re.match(r"^[a-z0-9-]+$", _pid):
            _print("[red]ID tidak valid.[/red] [dim]Gunakan format [a-z0-9-] huruf-kecil/angka/strip, contoh: my-provider[/dim]")
            return
        if _pid in _BUILTIN:
            _print(f"[red]ID '{_pid}' adalah penyedia bawaan.[/red] [dim]Pilih id lain (bawaan tak bisa ditimpa).[/dim]")
            return
        try:
            _reg0 = self._registry()
            _existing = _reg0.get_provider_descriptor(_pid)
            if _existing is not None:
                _print(f"[red]ID '{_pid}' sudah terdaftar.[/red] [dim]Pilih id lain atau /provider remove dulu.[/dim]")
                return
            # case-insensitive duplikat
            try:
                _all_ids = list(getattr(_reg0, "_providers", {}).keys())
            except Exception:
                _all_ids = []
            for _k in _all_ids:
                if isinstance(_k, str) and _k.strip().lower() == _pid.lower() and _k != _pid:
                    _print(f"[red]ID '{_pid}' duplikat (ada '{_k}').[/red]")
                    return
        except Exception:
            pass
        try:
            _cp0 = getattr(self.config, "custom_providers", None) if self.config else None
            if isinstance(_cp0, dict):
                for _k in list(_cp0.keys()):
                    if isinstance(_k, str) and _k.strip().lower() == _pid.lower():
                        _print(f"[red]ID '{_pid}' sudah ada di config.[/red]")
                        return
        except Exception:
            pass
        # --- baseURL ---
        try:
            _raw_base = (Prompt.ask("Alamat dasar (https://…/v1), contoh: https://api.example.com/v1") if RICH_AVAILABLE else input("Alamat dasar (https://…/v1), contoh: https://api.example.com/v1: "))
        except (EOFError, KeyboardInterrupt):
            _print("[dim]Dibatalkan.[/dim]")
            return
        _base = (_raw_base or "").strip().rstrip("/")
        try:
            import urllib.parse as _up
            _ok_scheme = _base.startswith("http://") or _base.startswith("https://")
            _parsed = _up.urlparse(_base) if _ok_scheme else None
            _ok_url = bool(_ok_scheme and _parsed is not None and _parsed.netloc)
        except Exception:
            _ok_url = False
        if not _ok_url:
            _print("[red]Alamat dasar tidak valid.[/red] [dim]Harus http(s)://host… contoh: https://api.example.com/v1[/dim]")
            return
        # --- api key opsional (tersembunyi) ---
        _key = self._provider_prompt_secret("Kunci API (opsional, Enter=kosong): ")
        # --- models: auto discovery → manual fallback ---
        _models: list = []
        try:
            _fetched, _ferr = self._provider_fetch_models(_base, _key, timeout=10.0)
        except Exception:
            _fetched, _ferr = [], "fetch failed"
        if _fetched:
            _models = list(_fetched)
            _print(f"[green]Ditemukan {len(_models)} model. Contoh: {_models[0]}[/green]")
        else:
            if _ferr:
                _print(f"[dim]Cari otomatis Gagal ({_ferr}) — isi manual atau kosongkan. Contoh: model-a,model-b[/dim]")
            try:
                _raw_m = (Prompt.ask("Model koma (kosong=menyusul), contoh: model-a,model-b", default="") if RICH_AVAILABLE else input("Model koma (kosong=menyusul), contoh: model-a,model-b: "))
            except (EOFError, KeyboardInterrupt):
                _print("[dim]Dibatalkan.[/dim]")
                return
            _mtxt = (_raw_m or "").strip()
            if _mtxt:
                _models = [m.strip() for m in _mtxt.split(",") if m.strip()]
            else:
                _models = []
        # --- validasi penuh: URL + tes koneksi; gagal → batal, jangan simpan buta ---
        try:
            _ok_conn, _cerr = self._provider_test_connection(_base, _key, timeout=10.0)
        except Exception:
            _ok_conn, _cerr = False, "tes sambungan Gagal"
        if not _ok_conn:
            _print(f"[red]Tes sambungan Gagal: {_cerr}[/red] [dim]Periksa alamat dasar/jaringan. Dibatalkan, tidak disimpan.[/dim]")
            return
        # --- sukses → simpan ---
        try:
            if not self.config:
                self.config = CodeAIConfig()
            _cp = getattr(self.config, "custom_providers", None)
            if not isinstance(_cp, dict):
                _cp = {}
                try:
                    self.config.custom_providers = _cp
                except Exception:
                    pass
            _entry = {"baseURL": _base, "models": list(_models), "name": _pid}
            _cp[_pid] = _entry
            self._persist_config()
            try:
                self._registry().register_custom_providers({_pid: _entry})
            except Exception:
                pass
            if (_key or "").strip():
                try:
                    from harness.models.auth_vault import AuthVault
                    AuthVault().store_token(_pid, (_key or "").strip())
                except Exception:
                    pass
            try:
                try:
                    _lk_inv = self._cache_lock
                except AttributeError:
                    _lk_inv = self._cache_lock = threading.Lock()
                with _lk_inv:
                    self._providers_cache = None
                    self._providers_ts = 0.0
                    self._models_cache = None
                    self._models_ts = 0.0
            except Exception:
                pass
        except Exception as _e:
            _print(f"[red]Gagal menyimpan penyedia: {_e}[/red]")
            return
        _print(f"[bold green]✅ Penyedia '{_pid}' tersimpan.[/bold green] [dim]({len(_models)} model)[/dim]")
        try:
            if _models:
                self._set_active_provider(_pid, _models[0])
                _print(f"✅ {_pid}/{_models[0]} aktif — langsung bisa chat. Contoh: halo")
            else:
                self._set_active_provider(_pid, "")
                _print(f"✅ {_pid}/ aktif — langsung bisa chat. [dim]Jalankan /model {_pid}/<nama> setelah discovery, atau /provider list untuk cek. Contoh: /model {_pid}/[/dim]")
        except Exception:
            _print(f"[dim]Tersimpan. Gunakan /model {_pid}/<nama> untuk pindah. Contoh: /model {_pid}/[/dim]")

    def _provider_list(self) -> None:
        try:
            _provs = self._list_providers_fast()
        except Exception:
            _provs = []
        try:
            _reg = self._registry()
        except Exception:
            _reg = None
        _rows: list = []
        for _p in _provs or []:
            try:
                _pid = str(_p.get("id", "")).strip()
                if not _pid:
                    continue
                _name = str(_p.get("name", _pid))
                _has = bool(_p.get("has_credentials"))
                _status = "● Terhubung" if _has else "○ Belum terhubung"
                _api = ""
                _n = 0
                try:
                    _desc = _reg.get_provider_descriptor(_pid) if _reg is not None else None
                except Exception:
                    _desc = None
                if isinstance(_desc, dict):
                    _api = str(_desc.get("api", "") or "")
                    _md = _desc.get("models", {})
                    if isinstance(_md, dict):
                        _n = len(_md)
                    elif isinstance(_md, list):
                        _n = len(_md)
                try:
                    _reg_models = _reg.list_models(_pid) if _reg is not None and hasattr(_reg, "list_models") else []
                    if _reg_models:
                        _n = len(list(_reg_models))
                except Exception:
                    pass
                _short_api = _api if len(_api) <= 34 else (_api[:31] + "…")
                _rows.append((_pid, _name, _short_api, _n, _status))
            except Exception:
                continue
        if RICH_AVAILABLE:
            table = Table(title="Penyedia (kustom via /provider add)", show_header=True, header_style="bold blue")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Nama")
            table.add_column("BaseURL", no_wrap=True)
            table.add_column("Model", justify="right")
            table.add_column("Status")
            for _pid, _name, _api_s, _n, _st in _rows:
                table.add_row(_pid, _name, _api_s or "—", str(_n), f"[green]{_st}[/green]" if "●" in _st else f"[dim]{_st}[/dim]")
            console.print()
            console.print(table)
            console.print("[dim]Cara pakai: /provider add · /provider remove <id> · /model untuk pindah · contoh: /provider list[/dim]")
            console.print()
        else:
            print("\n── Penyedia (kustom via /provider add) ──")
            for _pid, _name, _api_s, _n, _st in _rows:
                _mark = "● Terhubung" if "●" in _st else "○ Belum terhubung"
                print(f"  {_mark} — {_pid} ({_name}) [{_n} model] {_api_s or ''}")
            print("  Cara pakai: /provider add · /provider remove <id> · /model untuk pindah · contoh: /provider list\n")

    def _provider_remove(self, target: str = "") -> None:
        _pid_raw = (target or "").strip()
        if not _pid_raw:
            try:
                _pid_raw = (Prompt.ask("ID penyedia yang dihapus, contoh: my-provider") if RICH_AVAILABLE else input("ID penyedia yang dihapus, contoh: my-provider: "))
            except (EOFError, KeyboardInterrupt):
                _print("[dim]Dibatalkan.[/dim]")
                return
            _pid_raw = (_pid_raw or "").strip()
        if not _pid_raw:
            _print("[dim]Dibatalkan.[/dim]")
            return
        _want = _pid_raw.strip()
        try:
            _reg = self._registry()
        except Exception:
            _reg = None
        _actual = None
        try:
            _keys = list(getattr(_reg, "_providers", {}).keys()) if _reg is not None else []
            for _k in _keys:
                if isinstance(_k, str) and _k.strip().lower() == _want.lower():
                    _actual = _k
                    break
        except Exception:
            _actual = None
        _check = _actual or _want
        try:
            _is_custom = bool(_reg.is_custom_provider(_check)) if _reg is not None else False
        except Exception:
            _is_custom = False
        if not _is_custom:
            try:
                _desc = _reg.get_provider_descriptor(_check) if _reg is not None else None
            except Exception:
                _desc = None
            if _desc is None:
                _print(f"[red]Penyedia '{_want}' tidak ditemukan atau bukan kustom.[/red] [dim](lihat /provider list, contoh: /provider list)[/dim]")
            else:
                _print(f"[red]Tidak bisa hapus bawaan '{_check}'.[/red] [dim]Hanya penyedia kustom yang bisa dihapus.[/dim]")
            return
        try:
            _conf = (Prompt.ask(f"Hapus penyedia kustom '{_check}'? Contoh y", choices=["y", "n"], default="n") if RICH_AVAILABLE else input(f"Hapus '{_check}'? Contoh y [y/N]: "))
        except (EOFError, KeyboardInterrupt):
            _print("[dim]Dibatalkan.[/dim]")
            return
        if (_conf or "").strip().lower() != "y":
            _print("[dim]Dibatalkan.[/dim]")
            return
        try:
            _cp = getattr(self.config, "custom_providers", None) if self.config else None
            if isinstance(_cp, dict):
                for _k in list(_cp.keys()):
                    if isinstance(_k, str) and _k.strip().lower() == _check.strip().lower():
                        del _cp[_k]
        except Exception:
            pass
        try:
            if _reg is not None:
                _reg.unregister_custom_provider(_check)
        except Exception:
            pass
        try:
            from harness.models.auth_vault import AuthVault
            AuthVault().remove(_check)
        except Exception:
            pass
        try:
            self._persist_config()
        except Exception:
            pass
        try:
            try:
                _lk_inv2 = self._cache_lock
            except AttributeError:
                _lk_inv2 = self._cache_lock = threading.Lock()
            with _lk_inv2:
                self._providers_cache = None
                self._providers_ts = 0.0
                self._models_cache = None
                self._models_ts = 0.0
        except Exception:
            pass
        _print(f"[bold green]✅ Penyedia '{_check}' dihapus.[/bold green]")

    # ------------------------------------------------------------------
    # /effort
    # ------------------------------------------------------------------

    def handle_effort(self, args: str = ""):
        """Show or set Antigravity effort from inside the REPL (`/effort [low|medium|high]`, alias `/e`)."""
        from harness.models.providers.antigravity import EFFORT_MODELS, EFFORT_OPTIONS, DEFAULT_ANTIGRAVITY_EFFORT, parse_model_effort
        want = (args or "").strip().lower().split()[0] if (args or "").strip() else ""
        provider, model = self.get_active_info()
        # active_model may carry provider/ prefix in some flows — parse base robustly.
        bare = model.split("/", 1)[1] if "/" in str(model) else str(model)
        base, cur_eff = parse_model_effort(bare)
        stored = getattr(self.config.provider, "effort", None) if self.config and self.config.provider else None

        if not want:
            shown = cur_eff or (stored if isinstance(stored, str) and stored in EFFORT_OPTIONS else DEFAULT_ANTIGRAVITY_EFFORT)
            # Tanpa arg → popup low/medium/high + tandai aktif ★ (search bar).
            items = list(EFFORT_OPTIONS)
            try:
                _order = {"low": 0, "medium": 1, "high": 2}
                items = sorted(items, key=lambda x: _order.get(x, 9))
            except Exception:
                pass
            def _eff_label(x):
                try:
                    _id = self._effort_id(x)
                    _num = {"low": "1", "medium": "2", "high": "3"}.get(str(x).strip().lower(), "")
                    _t = f"{_num} {_id}" if _num else f"{_id}"
                    return f"{_t} ★ aktif" if x == shown else _t
                except Exception:
                    return str(x)
            _pick = self._tui_pick("Kekuatan pikir — pilih (1 rendah · 2 sedang · 3 tinggi)", items, show=_eff_label, initial="")
            if _pick is None:
                try:
                    _shown_id = self._effort_id(shown)
                except Exception:
                    _shown_id = str(shown)
                if base in EFFORT_MODELS:
                    _print(f"[cyan]Kekuatan pikir:[/cyan] [bold]{_shown_id}[/bold] [dim](model {base}-{shown})[/dim] [dim]Contoh: /effort 2[/dim]")
                else:
                    _print(f"[cyan]Kekuatan pikir:[/cyan] [bold]{_shown_id}[/bold] [dim](model '{bare}' tanpa pilihan kekuatan; bawaan untuk Antigravity)[/dim] [dim]Contoh: /effort 2[/dim]")
                return
            want = items[_pick]

        # Terima angka 1/2/3 sebagai alias Indonesia (1 rendah 2 sedang 3 tinggi).
        try:
            _w0 = str(want or "").strip().lower()
            if _w0 in ("1", "rendah"):
                want = "low"
            elif _w0 in ("2", "sedang"):
                want = "medium"
            elif _w0 in ("3", "tinggi"):
                want = "high"
        except Exception:
            pass
        if want not in EFFORT_OPTIONS:
            _print(f"[red]Kekuatan pikir tidak dikenal '{want}'.[/red] [dim]Cara pakai: /effort 1 rendah|2 sedang|3 tinggi · contoh: /effort 2[/dim]")
            return

        if base in EFFORT_MODELS:
            new_model = f"{base}-{want}"
            if not self.config:
                self.config = CodeAIConfig()
            try:
                setattr(self.config.provider, "effort", want)
            except Exception:
                pass
            self._set_active_provider(provider, new_model)
            try:
                _want_id = self._effort_id(want)
            except Exception:
                _want_id = str(want)
            _print(f"[bold green]✓ Kekuatan pikir menjadi {_want_id}[/bold green] [dim]({provider}/{new_model})[/dim]")
        else:
            # Current model has no effort knob — remember as default for next Antigravity switch.
            if not self.config:
                self.config = CodeAIConfig()
            try:
                setattr(self.config.provider, "effort", want)
            except Exception:
                pass
            self._persist_config()
            if self.orchestrator and getattr(self.orchestrator, "gateway", None):
                try:
                    setattr(self.orchestrator.gateway.config, "effort", want)
                except Exception:
                    pass
            try:
                _want_id2 = self._effort_id(want)
            except Exception:
                _want_id2 = str(want)
            _print(f"[bold green]✓ Kekuatan pikir bawaan tersimpan: {_want_id2}[/bold green] [dim](model '{bare}' tanpa pilihan kekuatan)[/dim]")

    # ------------------------------------------------------------------
    # /model
    # ------------------------------------------------------------------

    def switch_model(self, model_str: str):
        registry = self._registry()
        connected = self._list_connected_fast()
        _show_fn = lambda m: m["id"]  # noqa: E731

        def _generic_split(s: str):
            low = s.lower()
            for _eff in ("medium", "high", "low"):
                if low.endswith(f"-{_eff}") and len(s) > len(_eff) + 1:
                    return s[: -(len(_eff) + 1)], _eff
            return s, None

        def _resolve_bare_effort(norm: str) -> str:
            """Map bare `base` / `base-effort` (incl. shorthand `gemini-3.8-low`) to full id."""
            try:
                from harness.models.providers.antigravity import parse_model_effort
            except Exception:
                return norm
            if "/" in norm:
                _prov, _m = norm.split("/", 1)
                _b, _e = parse_model_effort(_m)
                if _e:
                    if any(m["id"] == f"{_prov}/{_b}" for m in connected):
                        return norm
                    _sh0 = [m for m in connected if m["provider"] == _prov and m.get("model", "").startswith(_b)]
                    if len(_sh0) == 1:
                        return f"{_prov}/{_sh0[0].get('model')}-{_e}"
                    return norm
                _gb, _ge = _generic_split(_m)
                if _ge:
                    for m in connected:
                        if m["id"] == f"{_prov}/{_gb}":
                            return norm
                    _sh = [m for m in connected if m["provider"] == _prov and m.get("model", "").startswith(_gb)]
                    if len(_sh) == 1:
                        return f"{_prov}/{_sh[0].get('model')}-{_ge}"
                return norm
            _b, _e = parse_model_effort(norm)
            for m in connected:
                if m.get("model") == norm or (_e and m.get("model") == _b):
                    return f"{m['provider']}/{norm}"
            _gb, _ge = _generic_split(norm)
            if _ge:
                _sh = [m for m in connected if m.get("model", "").startswith(_gb)]
                if len(_sh) == 1:
                    return f"{_sh[0]['provider']}/{_sh[0].get('model')}-{_ge}"
                # STRICT: unknown custom base -> keep as-is so caller rejects (never invent provider).
            return norm

        def _ids() -> list:
            return [m["id"] for m in connected]

        def _suggest(q: str, n: int = 5) -> list:
            try:
                _all = _ids()
                _m = difflib.get_close_matches(q, _all, n=n, cutoff=0.3)
                if not _m:
                    _m = _all[:n]
                return _m
            except Exception:
                return _ids()[:n]

        def _reject(q: str) -> None:
            _print(f"[red]✗ tidak cocok: '{q}'[/red] [dim]coba /model <kata kunci>, contoh: /model gemini[/dim]")
            for _c in _suggest(q, 5):
                _print(f"  [cyan]{_c}[/cyan]")

        def _is_known(tid: str) -> bool:
            if any(m["id"] == tid for m in connected):
                return True
            # Allow antigravity base-effort variant when base is connected.
            try:
                from harness.models.providers.antigravity import EFFORT_MODELS as _EM2, parse_model_effort as _pme2
                if "/" in tid:
                    _pv, _md = tid.split("/", 1)
                    if _pv == "antigravity":
                        _b, _e = _pme2(_md)
                        if _e and _b in _EM2 and any(m["id"] == f"{_pv}/{_b}" for m in connected):
                            return True
                else:
                    _b, _e = _pme2(tid)
                    if _e and any(m.get("model") == _b for m in connected):
                        return True
            except Exception:
                pass
            return False

        def _do_switch(tid: str) -> None:
            prov, mod = registry.parse_model_string(tid)
            try:
                from harness.models.providers.antigravity import EFFORT_MODELS as _EM, parse_model_effort as _pme
                _b0, _e0 = _pme(mod)
                if prov != "antigravity" and _b0 in _EM:
                    prov = "antigravity"
            except Exception:
                pass
            if prov == "antigravity":
                from harness.models.providers.antigravity import EFFORT_MODELS, EFFORT_OPTIONS, DEFAULT_ANTIGRAVITY_EFFORT, parse_model_effort
                base_model, existing_effort = parse_model_effort(mod)
                if base_model in EFFORT_MODELS and not existing_effort:
                    _stored = getattr(self.config.provider, "effort", None) if self.config and self.config.provider else None
                    _def = _stored if isinstance(_stored, str) and _stored in EFFORT_OPTIONS else DEFAULT_ANTIGRAVITY_EFFORT
                    if RICH_AVAILABLE:
                        effort_choice = Prompt.ask(f"pilih kekuatan pikir 1 rendah 2 sedang 3 tinggi [bawaan {_def}], contoh: 2", default=_def)
                    else:
                        effort_choice = input(f"pilih kekuatan pikir 1 rendah 2 sedang 3 tinggi [bawaan {_def}], contoh: 2: ").strip() or _def
                    effort_choice = (effort_choice or "").strip().lower()
                    try:
                        _ec = str(effort_choice or "").strip().lower()
                        if _ec in ("1", "rendah"):
                            effort_choice = "low"
                        elif _ec in ("2", "sedang"):
                            effort_choice = "medium"
                        elif _ec in ("3", "tinggi"):
                            effort_choice = "high"
                    except Exception:
                        pass
                    if effort_choice.isdigit() and 1 <= int(effort_choice) <= len(EFFORT_OPTIONS):
                        effort = EFFORT_OPTIONS[int(effort_choice) - 1]
                    elif effort_choice in EFFORT_OPTIONS:
                        effort = effort_choice
                    else:
                        effort = _def
                    mod = f"{base_model}-{effort}"
            self._set_active_provider(prov, mod)
            try:
                self.orchestrator.compactor.set_active_model(f"{prov}/{mod}")
            except Exception:
                pass
            _print(f"\n[green]Pindah ke [bold]{prov}[/bold] / [bold]{mod}[/bold][/green] [dim]Contoh: halo[/dim]\n")
            try:
                _ctx = self.orchestrator.compactor.get_context()
                _hist = _ctx.get("active_history", []) or []
                _facts = _ctx.get("fact_cards", []) or []
                _print(f"[dim]Sesi dipertahankan: {len(_hist)} pesan + {len(_facts)} ringkasan[/dim]")
                for _msg in _hist[-5:]:
                    _role = _msg.get("role", "unknown")
                    _body = str(_msg.get("content", ""))
                    if len(_body) > 300:
                        _body = _body[:300] + "…"
                    _print(f"[dim]{_role}: {_body}[/dim]")
            except Exception:
                pass

        # (a) /model <arg>: unik langsung switch; 0/banyak → popup prefilled.
        # TOLAK unknown tetap (tidak pernah `Switched to anthropic/<mentah>`).
        if model_str:
            raw_q = model_str.strip()
            norm = self._normalize_effort_syntax(raw_q)
            if norm.startswith("combo/"):
                try:
                    from harness.models.combo import ComboManager as _CM
                    _cdef = _CM().get_combo(norm.split("/", 1)[1])
                except Exception:
                    _cdef = None
                if _cdef:
                    _do_switch(norm)
                else:
                    _reject(raw_q)
                return
            if norm.isdigit():
                try:
                    if 1 <= int(norm) <= len(connected):
                        _do_switch(connected[int(norm) - 1]["id"])
                        return
                except Exception:
                    pass
            if any(m["id"] == norm for m in connected):
                _do_switch(norm)
                return
            _low = norm.lower()
            _exact = [m for m in connected if m["id"].lower() == _low]
            if len(_exact) == 1:
                _do_switch(_exact[0]["id"])
                return
            try:
                _res = _resolve_bare_effort(norm)
                if _res != norm and _is_known(_res):
                    _do_switch(_res)
                    return
                if _is_known(norm):
                    _do_switch(norm)
                    return
            except Exception:
                pass
            _hits = [m for m in connected if _low in m["id"].lower()]
            if len(_hits) == 1:
                _do_switch(_hits[0]["id"])
                return
            # 0 atau banyak → TUI prefilled arg (refine live di dalam).
            _pool = connected
            _pick = self._tui_pick("Model — pilih Cari: ketik kata kunci", _pool, show=_show_fn, initial=raw_q)
            if _pick is None:
                # Batal: bila 0 cocok tampilkan REJECT verbatim, bila banyak cukup batal.
                if len(_hits) == 0:
                    _reject(raw_q)
                else:
                    _print("[dim]Dibatalkan.[/dim]")
                return
            _do_switch(_pool[_pick]["id"])
            return

        if not connected:
            _print("[yellow]○ Belum terhubung — belum ada model — Langkah 1: ketik /provider[/yellow] [dim]Contoh: /provider gemini[/dim]")
            return
        # Tanpa arg → TUI SEMUA connected (live filter, viewport 15 ikut highlight).
        _pool0 = connected
        _pick0 = self._tui_pick("Model — pilih Cari: ketik kata kunci", _pool0, show=_show_fn, initial="")
        if _pick0 is None:
            _print("[dim]Dibatalkan.[/dim]")
            return
        _do_switch(_pool0[_pick0]["id"])
        return

    # ------------------------------------------------------------------
    # /combo
    # ------------------------------------------------------------------

    def handle_combo(self, args: str):
        from harness.models.combo import ComboManager, ComboStrategy
        manager = ComboManager()

        args = (args or "").strip()
        _parts = args.split(None, 1)
        _cmd = _parts[0].lower() if _parts else ""
        _rest = _parts[1].strip() if len(_parts) > 1 else ""

        if _cmd == "list":
            combos = manager.list_combos()
            if not combos:
                _print("[yellow]Belum ada gabungan tersimpan.[/yellow] [dim]Contoh: /combo create[/dim]")
                return
            try:
                _ap, _am = self.get_active_info()
            except Exception:
                _ap, _am = "", ""
            try:
                _st0, _lbl0, _pos0 = self._status_baku()
            except Exception:
                _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
            if RICH_AVAILABLE:
                table = Table(title=f"Gabungan — {_st0} — {_lbl0} — {_pos0}", show_header=True, header_style="bold blue")
                table.add_column("Nama", style="cyan")
                table.add_column("Strategi")
                table.add_column("Model")
                for name, data in combos.items():
                    _mark = " ★ aktif" if (_ap == "combo" and _am == name) else ""
                    table.add_row(f"{name}{_mark}", data["strategy"], ", ".join(data["models"]))
                console.print()
                console.print(table)
                console.print("[dim]Cara pakai: ketik untuk mencari · tombol atas bawah untuk pindah · Enter untuk pilih · Esc untuk batal · contoh: /combo use andalan[/dim]")
                console.print()
            else:
                print(f"\nGabungan — {_st0} — {_lbl0} — {_pos0}:")
                for name, data in combos.items():
                    _mark = " ★ aktif" if (_ap == "combo" and _am == name) else ""
                    print(f"  {name}{_mark} [{data['strategy']}]: {', '.join(data['models'])}")
                print("[dim]Contoh: /combo use andalan[/dim]\n")

        elif _cmd == "use":
            self._combo_use(_rest)
            return
        elif _cmd == "edit":
            self._combo_edit(_rest)
            return
        elif _cmd == "remove":
            _name = _rest.strip()
            if _name.lower().startswith("combo/"):
                _name = _name.split("/", 1)[1].strip()
            if not _name:
                _print("[yellow]Cara pakai: /combo remove <nama>[/yellow] [dim]Contoh: /combo remove andalan[/dim]")
                return
            try:
                _existing = manager.get_combo(_name)
            except Exception:
                _existing = None
            if not _existing:
                _print(f"[red]Gabungan '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list, contoh: /combo list)[/dim]")
                return
            try:
                _conf = (Prompt.ask(f"Hapus gabungan '{_name}'? Contoh y", choices=["y", "n"], default="n") if RICH_AVAILABLE else input(f"Hapus gabungan '{_name}'? Contoh y [y/N]: "))
            except (EOFError, KeyboardInterrupt):
                _print("[dim]Dibatalkan.[/dim]")
                return
            if (_conf or "").strip().lower() != "y":
                _print("[dim]Dibatalkan.[/dim]")
                return
            try:
                manager.delete_combo(_name)
            except Exception as _e:
                _print(f"[red]Gagal hapus gabungan: {_e}[/red] [dim]Contoh: /combo list[/dim]")
                return
            _print(f"[bold green]✅ Gabungan '{_name}' dihapus.[/bold green]")
            return
        elif not _cmd or _cmd == "create":
            connected = self._list_connected_fast()
            if not connected:
                _print("[yellow]○ Belum terhubung — belum ada model — Langkah 1: ketik /provider[/yellow] [dim]Contoh: /provider gemini[/dim]")
                return

            # Alur: nama → strategi (12 semua) → multi-pilih model → simpan.
            name = (Prompt.ask("Nama gabungan (kosong=kembali), contoh: andalan") if RICH_AVAILABLE else input("Nama gabungan (kosong=kembali), contoh: andalan: ")).strip()
            if not name or name.lower() in ("back", "q"):
                _print("[dim]Dibatalkan.[/dim]")
                return

            strategies = [s.value for s in ComboStrategy]
            # 12 strategi SEMUA tampil (viewport TUI 15) — Enter pilih, Esc batal.
            _spick = self._tui_pick("Gabungan — strategi Cari: ketik kata kunci", strategies, show=lambda x: x, initial="")
            if _spick is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            strategy = strategies[_spick]

            # Multi-pilih model: Spasi toggle ✓, Enter selesai (min 1), Esc/q batal.
            _mpicks = self._tui_pick_multi("Gabungan — model Cari: ketik kata kunci", connected, show=lambda m: m["id"], initial="")
            if not _mpicks:
                _print("[dim]Dibatalkan.[/dim]")
                return
            models = [connected[i]["id"] for i in _mpicks]

            manager.create_combo(name, strategy, models)
            _print(f"[bold green]✅ Gabungan '{name}' tersimpan.[/bold green] [dim]({strategy} · {len(models)} model · pakai via /model combo/{name}, contoh: /model combo/{name})[/dim]")

            sw = Prompt.ask("Pindah ke gabungan ini? Contoh y", choices=["y", "n"], default="y") if RICH_AVAILABLE else input("Pindah ke gabungan ini? Contoh y [y/N]: ")
            if sw.strip().lower() == "y":
                self.switch_model(f"combo/{name}")
        else:
            _print("[yellow]Cara pakai: /combo [list|create|use|edit|remove][/yellow] [dim]/combo use <nama> · /combo edit <nama> · /combo remove <nama> (ketik /bantuan, contoh: /combo list)[/dim]")

    def _combo_use(self, name_arg: str = "") -> None:
        """Aktifkan combo via jalur switch yang sudah ada (setara /model combo/<nama>)."""
        from harness.models.combo import ComboManager
        manager = ComboManager()
        try:
            combos = manager.list_combos() or {}
        except Exception:
            combos = {}
        _name = (name_arg or "").strip()
        if _name.lower().startswith("combo/"):
            _name = _name.split("/", 1)[1].strip()
        if not _name:
            if not combos:
                _print("[yellow]Belum ada gabungan tersimpan.[/yellow] [dim](buat via /combo create, contoh: /combo create)[/dim]")
                return
            items = sorted(combos.keys())
            def _show(n):
                try:
                    _d = combos.get(n) or {}
                    return f"{n} [{_d.get('strategy', '?')}] — {', '.join(_d.get('models', []) or [])}"
                except Exception:
                    return str(n)
            _pick = self._tui_pick("Gabungan — pakai Cari: ketik kata kunci", items, show=_show, initial="")
            if _pick is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            _name = items[_pick]
        try:
            _def = manager.get_combo(_name)
        except Exception:
            _def = None
        if not _def:
            try:
                for _k in list(combos.keys()):
                    if isinstance(_k, str) and _k.strip().lower() == _name.strip().lower():
                        _name = _k
                        _def = manager.get_combo(_k)
                        break
            except Exception:
                pass
        if not _def:
            _print(f"[red]Gabungan '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list, contoh: /combo list)[/dim]")
            return
        try:
            self.switch_model(f"combo/{_name}")
        except Exception as _e:
            _print(f"[red]Gagal pakai gabungan '{_name}': {_e}[/red] [dim]Contoh: /combo list[/dim]")

    def _combo_edit(self, name_arg: str = "") -> None:
        """Ubah strategi + tambah/buang models, simpan overwrite (nama tetap)."""
        from harness.models.combo import ComboManager, ComboStrategy
        manager = ComboManager()
        try:
            combos = manager.list_combos() or {}
        except Exception:
            combos = {}
        _name = (name_arg or "").strip()
        if _name.lower().startswith("combo/"):
            _name = _name.split("/", 1)[1].strip()
        if not _name:
            if not combos:
                _print("[yellow]Belum ada gabungan tersimpan.[/yellow] [dim](buat via /combo create, contoh: /combo create)[/dim]")
                return
            items = sorted(combos.keys())
            def _show2(n):
                try:
                    _d = combos.get(n) or {}
                    return f"{n} [{_d.get('strategy', '?')}] — {', '.join(_d.get('models', []) or [])}"
                except Exception:
                    return str(n)
            _pick0 = self._tui_pick("Gabungan — ubah Cari: ketik kata kunci", items, show=_show2, initial="")
            if _pick0 is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            _name = items[_pick0]
        try:
            _cur = manager.get_combo(_name)
        except Exception:
            _cur = None
        _canon = _name
        if not _cur:
            try:
                for _k in list(combos.keys()):
                    if isinstance(_k, str) and _k.strip().lower() == _name.strip().lower():
                        _canon = _k
                        _cur = manager.get_combo(_k)
                        break
            except Exception:
                pass
        if not isinstance(_cur, dict):
            _print(f"[red]Gabungan '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list, contoh: /combo list)[/dim]")
            return
        try:
            _cur_strategy = str(_cur.get("strategy", "") or "")
        except Exception:
            _cur_strategy = ""
        try:
            _cur_models = list(_cur.get("models", []) or [])
        except Exception:
            _cur_models = []
        try:
            _cur_params = dict(_cur.get("params", {}) or {})
        except Exception:
            _cur_params = {}
        strategies = [s.value for s in ComboStrategy]
        def _sshow(s):
            try:
                return f"{s} ★ saat ini" if s == _cur_strategy else s
            except Exception:
                return s
        _spick = self._tui_pick("Gabungan — strategi baru Cari: ketik kata kunci", strategies, show=_sshow, initial=_cur_strategy or "")
        if _spick is None:
            _print("[dim]Dibatalkan.[/dim]")
            return
        new_strategy = strategies[_spick]
        connected = self._list_connected_fast()
        if not connected:
            _print("[yellow]○ Belum terhubung — belum ada model — Langkah 1: ketik /provider[/yellow] [dim]Contoh: /provider gemini[/dim]")
            return
        try:
            _cur_txt = ", ".join(_cur_models) if _cur_models else "—"
        except Exception:
            _cur_txt = "—"
        _mpicks = self._tui_pick_multi(f"Gabungan — model (saat ini: {_cur_txt}) Cari: ketik kata kunci", connected, show=lambda m: m["id"], initial="")
        if not _mpicks:
            _print("[dim]Dibatalkan.[/dim]")
            return
        new_models = [connected[i]["id"] for i in _mpicks]
        try:
            manager.create_combo(_canon, new_strategy, new_models, params=_cur_params, overwrite=True)
        except TypeError:
            try:
                manager.create_combo(_canon, new_strategy, new_models)
            except Exception as _e:
                _print(f"[red]Gagal simpan gabungan: {_e}[/red] [dim]Contoh: /combo list[/dim]")
                return
        except Exception as _e:
            _print(f"[red]Gagal simpan gabungan: {_e}[/red] [dim]Contoh: /combo list[/dim]")
            return
        _print(f"[bold green]✅ Gabungan '{_canon}' diperbarui.[/bold green] [dim]({new_strategy} · {len(new_models)} model · pakai via /combo use {_canon}, contoh: /combo use {_canon})[/dim]")

    @staticmethod
    def _combo_extract_serving(result):
        """Cari info member penyaji di response dict (varian keys). Return (prov, model) atau (None, None)."""
        try:
            if not isinstance(result, dict):
                return None, None
            def _g(*keys):
                for _k in keys:
                    try:
                        if _k in result and result[_k]:
                            _v = result[_k]
                            if isinstance(_v, str) and _v.strip():
                                return _v.strip()
                            elif isinstance(_v, dict):
                                return _v
                            elif _v:
                                return _v
                    except Exception:
                        continue
                try:
                    _low = {str(k).lower(): v for k, v in result.items()}
                    for _k in keys:
                        _lk = str(_k).lower()
                        if _lk in _low and _low[_lk]:
                            _v = _low[_lk]
                            if isinstance(_v, str) and _v.strip():
                                return _v.strip()
                            elif isinstance(_v, dict):
                                return _v
                            elif _v:
                                return _v
                except Exception:
                    pass
                return None
            _sp = _g("serving_provider", "served_provider", "member_provider")
            _sm = _g("serving_model", "served_model", "member_model")
            if isinstance(_sp, str) and isinstance(_sm, str) and _sp.strip() and _sm.strip():
                return _sp.strip(), _sm.strip()
            _srv = _g("serving", "served_by", "serving_member")
            if isinstance(_srv, dict):
                try:
                    _p = _srv.get("provider") or _srv.get("serving_provider") or _srv.get("member_provider")
                    _m = _srv.get("model") or _srv.get("serving_model") or _srv.get("member") or _srv.get("member_model")
                    if _p and _m:
                        return str(_p).strip(), str(_m).strip()
                    if _m and "/" in str(_m):
                        _a, _b = str(_m).split("/", 1)
                        if _a.strip() and _b.strip():
                            return _a.strip(), _b.strip()
                except Exception:
                    pass
            elif isinstance(_srv, str) and _srv.strip():
                _s = _srv.strip()
                if "/" in _s:
                    _a, _b = _s.split("/", 1)
                    if _a.strip() and _b.strip():
                        return _a.strip(), _b.strip()
            _mem = _g("member", "member_id")
            if isinstance(_mem, str) and _mem.strip():
                _s = _mem.strip()
                if "/" in _s:
                    _a, _b = _s.split("/", 1)
                    if _a.strip() and _b.strip():
                        return _a.strip(), _b.strip()
                _p2 = _g("provider", "member_provider", "serving_provider")
                if isinstance(_p2, str) and _p2.strip() and _s:
                    return _p2.strip(), _s
            elif isinstance(_mem, dict):
                try:
                    _p = _mem.get("provider") or _mem.get("serving_provider")
                    _m = _mem.get("model") or _mem.get("serving_model") or _mem.get("id")
                    if _p and _m:
                        return str(_p).strip(), str(_m).strip()
                except Exception:
                    pass
            try:
                if "provider" in result and "model" in result:
                    _p = str(result.get("provider") or "").strip()
                    _m = str(result.get("model") or "").strip()
                    if _p and _m and "/" not in _p:
                        return _p, _m
            except Exception:
                pass
            return None, None
        except Exception:
            return None, None

    def _combo_footer(self, combo_name: str, result) -> None:
        """Footer tiap respons combo. Fallback strategi+member bila serving tak diekspos (combo.py tak disentuh)."""
        try:
            _cn = str(combo_name or "").strip()
        except Exception:
            _cn = str(combo_name)
        try:
            if "/" in _cn:
                _cn = _cn.split("/", 1)[1].strip() or _cn
        except Exception:
            pass
        if not _cn:
            return
        try:
            _sp, _sm = self._combo_extract_serving(result)
        except Exception:
            _sp, _sm = None, None
        if _sp and _sm:
            _print(f"[dim]Dijawab oleh gabungan {_cn} memakai {_sp}/{_sm}[/dim]")
            return
        try:
            from harness.models.combo import ComboManager as _CM2
            _def = _CM2().get_combo(_cn)
        except Exception:
            _def = None
        if isinstance(_def, dict):
            try:
                _strat = str(_def.get("strategy", "?") or "?")
            except Exception:
                _strat = "?"
            try:
                _mems = list(_def.get("models", []) or [])
            except Exception:
                _mems = []
            _mem_txt = ", ".join(_mems) if _mems else "—"
            _print(f"[dim]Dijawab oleh gabungan {_cn} memakai {_mem_txt} [{_strat}][/dim]")
        else:
            _print(f"[dim]Dijawab oleh gabungan {_cn} memakai — (detail tidak ditemukan)[/dim]")
        try:
            logging.getLogger(__name__).warning("Handoff next-wave: ComboProvider tak mengekspos serving member (butuh serving_provider/serving_model di return dict) — lihat harness/models/combo.py:ComboProvider.chat")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # /history
    # ------------------------------------------------------------------

    def show_history(self):
        if not self.orchestrator:
            _print("[yellow]Orchestrator belum siap.[/yellow] [dim]Contoh: ketik halo untuk mulai[/dim]")
            return

        ctx = self.orchestrator.compactor.get_context()
        fact_cards = ctx.get("fact_cards", [])
        history    = ctx.get("active_history", [])

        try:
            _st0, _lbl0, _pos0 = self._status_baku()
        except Exception:
            _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        if not fact_cards and not history:
            _print(f"[dim]{_st0} — {_lbl0} — {_pos0}[/dim]")
            _print("[dim]Belum ada riwayat percakapan.[/dim] [dim]Contoh: ketik halo untuk mulai[/dim]")
            return

        if RICH_AVAILABLE:
            console.print(f"[dim]{_st0} — {_lbl0} — {_pos0}[/dim]")
            console.print()
            if fact_cards:
                _rule("Konteks Ringkas (Kartu Fakta)", style="dim")
                for fc in fact_cards:
                    console.print(f"  [dim]{fc}[/dim]")
                console.print()

            if history:
                _rule("Riwayat Aktif", style="dim")
                for msg in history:
                    role  = msg.get("role", "tidak dikenal")
                    body  = msg.get("content", "")
                    color = "cyan" if role == "user" else "green"
                    label = "Anda ❯" if role == "user" else "AI ◆"
                    console.print(f"[bold {color}]{label}[/bold {color}]")
                    # Truncate very long entries for readability
                    if len(body) > 500:
                        body = body[:500] + "… [dim](dipotong)[/dim]"
                    console.print(f"  {body}")
                    console.print()
        else:
            print(f"\n{_st0} — {_lbl0} — {_pos0}")
            if fact_cards:
                print("\n── Kartu Fakta ──")
                for fc in fact_cards:
                    print(f"  {fc}")
            if history:
                print("\n── Riwayat ──")
                for msg in history:
                    print(f"[{'Anda ❯' if msg.get('role') == 'user' else 'AI ◆'}]: {msg.get('content','')[:300]}")
            print()

    # ------------------------------------------------------------------
    # /status, /config, /steer, /help
    # ------------------------------------------------------------------

    def show_status(self):
        provider, model = self.get_active_info()
        agents_exists = os.path.exists("AGENTS.md")
        hooks_exists  = os.path.exists("SYSTEM_HOOKS.md")
        state = self.orchestrator.workflow_state if self.orchestrator else "N/A"
        try:
            _st, _lbl, _pos = self._status_baku()
        except Exception:
            _st, _lbl, _pos = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        try:
            _eff_raw = getattr(getattr(self.config, "provider", None), "effort", None) or "—"
            _eff = self._effort_id(_eff_raw) if _eff_raw != "—" else "—"
        except Exception:
            _eff = "—"
        _keadaan = str(state or "—")

        if RICH_AVAILABLE:
            console.print(f"[bold]{_st} — {_lbl} — {_pos}[/bold]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_row("[dim]Status[/dim]",  f"[green]{_st}[/green]")
            table.add_row("[dim]Penyedia[/dim]",  f"[cyan]{provider}[/cyan]")
            table.add_row("[dim]Model[/dim]",     f"[yellow]{model}[/yellow] [dim]({_lbl})[/dim]")
            table.add_row("[dim]Posisi[/dim]",     f"{_pos}")
            table.add_row("[dim]Kekuatan pikir[/dim]", f"{_eff} [dim](1 rendah · 2 sedang · 3 tinggi)[/dim]")
            table.add_row("[dim]Keadaan[/dim]",     f"{_keadaan}")
            table.add_row("[dim]AGENTS.md[/dim]", "[green]ada[/green]" if agents_exists else "[dim]tidak ada[/dim]")
            table.add_row("[dim]HOOKS[/dim]",     "[green]ada[/green]" if hooks_exists  else "[dim]tidak ada[/dim]")
            console.print()
            console.print(table)
            console.print("[dim]Contoh: /model gemini · /provider gemini · ketik /bantuan[/dim]")
            console.print()
        else:
            print(f"\n{_st} — {_lbl} — {_pos}")
            print(f"Status: {_st}\nPenyedia: {provider}\nModel: {model} ({_lbl})\nPosisi: {_pos}\nKekuatan pikir: {_eff} (1 rendah · 2 sedang · 3 tinggi)\nKeadaan: {_keadaan}")
            print(f"AGENTS.md: {'ada' if agents_exists else 'tidak ada'}")
            print(f"HOOKS: {'ada' if hooks_exists else 'tidak ada'}")
            print("Contoh: /model gemini · /provider gemini · ketik /bantuan\n")

    def show_config(self):
        try:
            _st0, _lbl0, _pos0 = self._status_baku()
        except Exception:
            _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        if RICH_AVAILABLE:
            import json
            from pydantic import BaseModel
            data = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config.__dict__
            console.print(f"[dim]{_st0} — {_lbl0} — {_pos0}[/dim]")
            console.print()
            console.print_json(json.dumps(data, default=str))
            console.print("[dim]Contoh: /pengaturan untuk lihat, /keadaan untuk ringkas[/dim]")
            console.print()
        else:
            print(f"{_st0} — {_lbl0} — {_pos0}")
            print(self.config.__dict__)
            print("Contoh: /pengaturan untuk lihat, /keadaan untuk ringkas")

    def steer_orchestrator(self, instruction: str):
        instruction = (instruction or "").strip()
        if not instruction:
            _print("[yellow]Cara pakai: /steer (/alih) <perintah>[/yellow] [dim]Contoh: /steer lanjutkan · /alih lanjutkan[/dim]")
            return
        if not self.orchestrator:
            _print("[yellow]Tidak ada tugas berjalan[/yellow] [dim](jalankan tugas dulu, lalu /steer atau /alih, contoh: /alih lanjutkan)[/dim]")
            return
        # Steer jujur: hanya RUNNING boleh ✓; selain itu tolak tanpa ✓ palsu.
        try:
            _state = getattr(self.orchestrator, "workflow_state", None)
            _active = getattr(self.orchestrator, "active_subagent_id", None)
        except Exception:
            _state, _active = None, None
        if _state != "RUNNING" or not _active:
            _print("[yellow]Tidak ada tugas berjalan[/yellow] [dim](jalankan tugas dulu, lalu /steer atau /alih, contoh: /alih lanjutkan)[/dim]")
            return
        _print(f"[blue]◉ mengarahkan → {instruction}[/blue]")
        try:
            async def _do():
                try:
                    _r = await self.orchestrator.steer_active_subagent(instruction)
                    if _r is False:
                        try:
                            _r2 = await self.orchestrator.process_user_input(instruction)
                        except Exception:
                            _r2 = False
                        if isinstance(_r2, bool):
                            return _r2
                        return True
                    if isinstance(_r, bool):
                        return _r
                    return True
                except Exception:
                    _r3 = await self.orchestrator.process_user_input(instruction)
                    if isinstance(_r3, bool):
                        return _r3
                    return True
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    _steered = pool.submit(asyncio.run, _do()).result(timeout=30)
            else:
                try:
                    _steered = asyncio.run(asyncio.wait_for(_do(), timeout=30))
                except AttributeError:
                    _steered = asyncio.run(_do())
            try:
                _ok_flag = bool(_steered) if isinstance(_steered, bool) else True
            except Exception:
                _ok_flag = True
            if not _ok_flag:
                _print("[yellow]Tidak ada tugas berjalan[/yellow] [dim](jalankan tugas dulu, lalu /steer atau /alih, contoh: /alih lanjutkan)[/dim]")
                return
            _print("[green]✓ sudah dialihkan[/green]")
        except ValueError as e:
            _print(f"[yellow]Tidak ada tugas berjalan: {e}[/yellow] [dim](jalankan tugas dulu, contoh: halo)[/dim]")
        except Exception as e:
            _print(f"[red]Gagal alih: {e}[/red] [dim](ketik /bantuan)[/dim]")

    def show_help(self):
        try:
            _st0, _lbl0, _pos0 = self._status_baku()
        except Exception:
            _st0, _lbl0, _pos0 = ("○ Belum terhubung", "belum ada model", "Langkah 1: ketik /provider")
        groups = [
            ("MODEL", [("/model (/m)", "Pindah model — ketik kata kunci atau pilih; contoh: /model gemini"),
                       ("/effort (/e)", "Lihat/atur kekuatan pikir — 1 rendah · 2 sedang · 3 tinggi; contoh: /effort 2"),
                       ("/combo (/c)", "Kelola gabungan [list|create|use|edit|remove]; contoh: /combo list"),
                       ("/models <penyedia>", "Daftar model penyedia; contoh: /models gemini")]),
            ("SAMBUNGAN", [("/provider [id]", "Masuk/pindah penyedia; contoh: /provider gemini"),
                      ("/providers (/p) (/daftar)", "Daftar penyedia + status Terhubung; contoh: /daftar"),
                      ("/provider add|list|remove", "Kelola penyedia kustom; contoh: /provider list"),
                      ("/daftar", "Sama dengan /providers — daftar penyedia")]),
            ("SESI", [("/steer (/st) (/alih)", "Alihkan tugas berjalan; contoh: /alih lanjutkan"),
                         ("/history (/riwayat)", "Lihat riwayat + kartu fakta; contoh: /riwayat"),
                         ("/status (/s) (/keadaan)", "Lihat penyedia, model, status, posisi; contoh: /keadaan"),
                         ("/config (/pengaturan)", "Lihat pengaturan berjalan; contoh: /pengaturan")]),
            ("BANTUAN", [("/help (/h) (/bantuan)", "Bantuan ini; contoh: /bantuan"), ("/quit (/q) (/keluar)", "Keluar; contoh: /keluar")]),
        ]
        if RICH_AVAILABLE:
            console.print(f"[dim]{_st0} — {_lbl0} — {_pos0}[/dim]")
            for g, rows in groups:
                console.print(f"[bold blue]{g}[/bold blue]")
                table = Table(show_header=False, box=None, padding=(0, 2))
                for cmd, desc in rows:
                    table.add_row(f"[cyan]{cmd}[/cyan]", f"[dim]{desc}[/dim]")
                console.print(table)
            console.print("[dim]Cara pakai: Ctrl-C berhenti, /steer (/alih) mengarahkan · contoh: /model gemini · ketik /bantuan[/dim]")
        else:
            print(f"\n{_st0} — {_lbl0} — {_pos0}")
            for g, rows in groups:
                print(f"\n[{g}]")
                for cmd, desc in rows:
                    print(f"  {cmd:<30} {desc}")
            print("\nCara pakai: Ctrl-C berhenti, /steer (/alih) mengarahkan · contoh: /model gemini · ketik /bantuan")
