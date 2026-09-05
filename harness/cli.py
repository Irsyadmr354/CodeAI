import sys
import os
import time
import asyncio
import difflib
import traceback
import logging
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
    FOOTER = "↑↓/jk=pindah · huruf=cari live · Enter=pilih · Esc=batal"

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
        lines = [_trunc(f"{self.title}  {pos}", _max), _trunc(f"🔍 {self.query}▊", _max)]
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
            lines.append(_trunc(f"  ✗ tidak cocok: '{self.query}'", _max))
        if self.multi:
            lines.append(_trunc(f"  [terpilih {len(self.selected)} · Spasi=toggle · Enter=selesai (min 1)]", _max))
        lines.append(_trunc(self.FOOTER if not self.multi else self.FOOTER + " · Spasi=toggle", _max))
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
        if "v" in box:
            return box["v"]
        return fallback

    def _list_providers_fast(self, timeout: float = 1.5) -> list:
        """Cache-dulu + lazy: cache segar (<30s) langsung; miss → fetch timeout, gagal → stale/[] non-blocking."""
        now = time.monotonic()
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
                        self._providers_cache = _v
                        self._providers_ts = time.monotonic()
                    except Exception:
                        pass
                _th.Thread(target=_bg, daemon=True).start()
            except Exception:
                pass
            return stale
        self._providers_cache = res
        self._providers_ts = now
        return res

    def _list_connected_fast(self, timeout: float = 2.0) -> list:
        """Cache-dulu untuk list_connected_models (hindari block 8s discovery); timeout → stale/[] + refresh lazy."""
        now = time.monotonic()
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
                        self._models_cache = _v2
                        self._models_ts = time.monotonic()
                    except Exception:
                        pass
                _th2.Thread(target=_bg2, daemon=True).start()
            except Exception:
                pass
            return stale
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
        for _pid in chain:
            try:
                _ps = str(_pid).strip() or "?"
            except Exception:
                _ps = "?"
            if _ps == "ollama":
                if not _oll_inst:
                    lines.append("• ollama: skipped (not installed)")
                elif not _oll_run:
                    lines.append("• ollama: skipped (not running — jalankan `ollama serve`)")
                elif _ps == _owner:
                    if self._is_auth_like(_cause):
                        lines.append(f"• ollama: {_cause} → saran /provider ollama")
                    elif self._is_conn_like(_cause):
                        lines.append(f"• ollama: {_cause} → cek koneksi atau /model")
                    else:
                        lines.append(f"• ollama: {_cause}")
                else:
                    lines.append("• ollama: dicoba (failover)")
                continue
            if _ps not in conn_set:
                lines.append(f"• {_ps}: not connected → saran /provider {_ps}")
            elif _ps == _owner:
                if self._is_auth_like(_cause):
                    lines.append(f"• {_ps}: {_cause} → saran /provider {_ps}")
                elif self._is_conn_like(_cause):
                    lines.append(f"• {_ps}: {_cause} → cek koneksi atau /model")
                else:
                    lines.append(f"• {_ps}: {_cause}")
            else:
                lines.append(f"• {_ps}: dicoba (failover)")
        if connected:
            try:
                _show = ", ".join(list(connected)[:5])
                lines.append(f"Saran: /model {_show} (hanya yang CONNECTED) · /providers untuk daftar")
            except Exception:
                pass
        else:
            lines.append("Saran: /provider <provider> untuk menghubungkan (lihat /providers)")
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

        footer = "nomor=pilih · teks=filter · Enter=kembali/batal · q=batal"
        count_line = f"… +{total - len(shown)} cocok (ketik lagi untuk filter)" if total > len(shown) else ""
        # Plain vertical writes (no Rich Table/Panel/Columns → anti-menyamping).
        try:
            sys.stdout.write(_trunc(f"── {title}", _max) + "\n")
            sys.stdout.write(_trunc(f"🔍 {query if query else '—'}", _max) + "\n")
            for i, label in enumerate(shown, 1):
                pre = f"{i:2}. "
                budget = _max - len(pre)
                if budget < 1:
                    budget = 1
                sys.stdout.write(pre + _trunc(label, budget) + "\n")
            if count_line:
                sys.stdout.write(_trunc(count_line, _max) + "\n")
            sys.stdout.write(_trunc(footer, _max) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def _popup_input(self, hint: str = "[cari?] nomor/teks (Enter=batal, q=batal)") -> Optional[str]:
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
                _print(f"[red]✗ tidak cocok: '{query}'[/red] [dim]coba substring lain / q=batal[/dim]")
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
        hint = "[cari?] 1,3/done/teks (Enter=selesai/batal, q=batal)"
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
            if s.lower() == "done":
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
            _print(f"[bold red]Failed to initialise orchestrator: {e}[/bold red]")
            if self.verbose:
                traceback.print_exc()
            sys.exit(1)

    def display_banner(self):
        # Slim 2-line banner (max 2 rows).
        provider, model = self.get_active_info()
        agents_exists = os.path.exists("AGENTS.md")
        hooks_exists = os.path.exists("SYSTEM_HOOKS.md")
        try:
            provs = self._list_providers_fast()
            n_conn = sum(1 for p in provs if p.get("has_credentials"))
        except Exception:
            n_conn = 0
        ag = "ON" if agents_exists else "off"
        hk = "ON" if hooks_exists else "off"
        eff = getattr(getattr(self.config, "provider", None), "effort", None) or "—"
        el = f"{self._last_elapsed:.1f}s" if getattr(self, "_last_elapsed", None) else "—"
        short = self._short(model)
        if RICH_AVAILABLE:
            console.print(f"[bold blue]CodeAI Harness[/bold blue] [dim]v0.1.0[/dim]  [cyan]{provider}[/cyan]/[yellow]{short}[/yellow]  [dim]{n_conn} connected · {eff} · {el}[/dim]")
            console.print(f"  [dim]AGENTS:{ag} HOOKS:{hk}  │  /help /model /provider /combo /quit  (Ctrl-C steer)[/dim]")
        else:
            print(f"CodeAI Harness v0.1.0  |  {provider}/{short}  |  {n_conn} connected · {eff} · {el}")
            print(f"AGENTS:{ag} HOOKS:{hk}  |  /help /model /provider /combo /quit  (Ctrl-C steer)")

    def run(self):
        self.startup()
        self.display_banner()
        self.repl_loop()

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _ask_main(self) -> str:
        # SATU box bersih: header ╭─ ❯ <provider-penuh> · <short> · <effort> ─.
        # Provider TAMPIL PENUH (jangan via _short/AI) agar "antigravity"
        # tak terpotong; _short hanya untuk model. Tutup ╰─. TANPA Prompt
        # rich ganda dan TANPA hint cari (milik picker saja).
        try:
            _prov, _mod = self.get_active_info()
        except Exception:
            _prov, _mod = "?", "?"
        try:
            _prov = str(_prov).strip() or "?"
        except Exception:
            _prov = "?"
        try:
            _eff = getattr(getattr(self.config, "provider", None), "effort", None) or "—"
        except Exception:
            _eff = "—"
        try:
            _short = self._short(_mod)
        except Exception:
            _short = str(_mod)
        header = f"╭─ ❯ {_prov} · {_short} · {_eff} ─"
        if RICH_AVAILABLE:
            console.print(f"[bold cyan]{header}[/bold cyan] [dim]pesan atau /perintah · cth: /model gemini[/dim]")
            try:
                val = input("│ ❯ ")
            except (EOFError, KeyboardInterrupt):
                raise
            try:
                console.print("[dim]╰─[/dim]")
            except Exception:
                pass
            return val
        print(f"{header} ─ ketik pesan atau /perintah (cth: /model gemini)")
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
                _print("\n[dim]Interrupted (Ctrl-C). Type /quit to exit or /steer <text> to redirect.[/dim]")
            except EOFError:
                break
            except Exception as e:
                _print(f"[bold red]Error: {e}[/bold red] [dim](type /help)[/dim]")
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
            "/status":    lambda: self.show_status(),
            "/s":         lambda: self.show_status(),
            "/config":    lambda: self.show_config(),
            "/steer":     lambda: self.steer_orchestrator(args) if args else _print("[yellow]Usage: /steer <instruction>[/yellow]"),
            "/st":        lambda: self.steer_orchestrator(args) if args else _print("[yellow]Usage: /steer <instruction>[/yellow]"),
            "/providers": lambda: self.show_providers(args),
            "/p":         lambda: self.show_providers(args),
            "/provider":  lambda: self.handle_provider(args),
            "/models":    lambda: self.show_models(args) if args else _print("[yellow]Usage: /models <provider>[/yellow]"),
            "/model":     lambda: self.switch_model(args),
            "/m":         lambda: self.switch_model(args),
            "/effort":    lambda: self.handle_effort(args),
            "/e":         lambda: self.handle_effort(args),
            "/combo":     lambda: self.handle_combo(args),
            "/c":         lambda: self.handle_combo(args),
            "/history":   lambda: self.show_history(),
            "/help":      lambda: self.show_help(),
            "/h":         lambda: self.show_help(),
        }

        handler = dispatch.get(cmd)
        if handler:
            handler()
        else:
            try:
                _cands = list(dispatch.keys())
                _m = difflib.get_close_matches(cmd, _cands, n=1, cutoff=0.6)
                _hint = f"  [dim]Did you mean {_m[0]}?[/dim]" if _m else "  [dim](type /help)[/dim]"
            except Exception:
                _hint = "  [dim](type /help)[/dim]"
            _print(f"[red]Unknown command: {cmd}[/red]{_hint}")

    def _cmd_quit(self):
        _print("[dim]Goodbye.[/dim]")
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

        def _on_token(tok) -> None:
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
            _gw2 = getattr(self.orchestrator, "gateway", None) if self.orchestrator else None
            try:
                # Teruskan on_token ke gateway bila didukung; TypeError → plain.
                if _gw2 is not None and _stream_supported and hasattr(_gw2, "chat"):
                    try:
                        import functools as _ft
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
                        _gw2.chat = _patched
                    except Exception:
                        _orig = None
                _box["r"] = self.orchestrator.run_task(task)
            except Exception as e:
                _box["e"] = f"❌ Task failed: {e}"
            finally:
                try:
                    if _gw2 is not None and _orig is not None:
                        _gw2.chat = _orig
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
            _spin = Spinner("dots", text=f" orchestrator thinking · {provider}/{short} · 0.0s · Ctrl-C steer", style="cyan")

            def _renderable():
                _dt = time.monotonic() - t0
                try:
                    _spin.text = f" orchestrator thinking · {provider}/{short} · {_dt:.1f}s · Ctrl-C steer"
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
                            _print("\n[dim]Steer (tugas tetap jalan — Enter kosong = lanjut)[/dim]")
                            try:
                                _s = input("steer ❯ ")
                            except (EOFError, KeyboardInterrupt):
                                _s = ""
                            if (_s or "").strip():
                                try:
                                    self.steer_orchestrator(_s.strip())
                                except Exception as _se:
                                    _print(f"[red]Steer failed: {_se}[/red]")
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
            result = _box.get("r", _box.get("e", "_No response._"))
            result = result or "_No response._"
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
                    _print(f"{_chk}\n{_detail}  [dim](type /help)[/dim]", style="red")
                else:
                    _print(f"{_chk}  [dim](run /provider or /model — type /help)[/dim]", style="red")
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
            with Status(f"[dim]orchestrator thinking · {provider}/{short} · Ctrl-C steer[/dim]", spinner="dots", spinner_style="cyan"):
                _th.join()
            result = _box.get("r", _box.get("e", "_No response._"))
            result = result or "_No response._"
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
                    _print(f"{_chk2}\n{_detail2}  [dim](type /help)[/dim]", style="red")
                else:
                    _print(f"{_chk2}  [dim](run /provider or /model — type /help)[/dim]", style="red")
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
            print(f"orchestrator thinking · {provider}/{short} · Ctrl-C to steer …")
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
                        print("Steer (tugas tetap jalan — Enter kosong = lanjut)")
                        try:
                            _s2 = input("steer ❯ ")
                        except (EOFError, KeyboardInterrupt):
                            _s2 = ""
                        if (_s2 or "").strip():
                            try:
                                self.steer_orchestrator(_s2.strip())
                            except Exception as _se2:
                                print(f"Steer failed: {_se2}")
            except Exception:
                pass
            _th.join(timeout=5)
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
            result = _box.get("r", _box.get("e", "No response."))
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
                    print(f"\n─── Error ───\n{_chk3}\n{_detail3}\n[{dt:.1f}s]")
                else:
                    print(f"\n─── Error ───\n{_chk3}  (run /provider or /model — type /help)\n[{dt:.1f}s]")
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
                items = [{"id": p["id"], "label": f"{p['id']} {'✅ connected' if p.get('has_credentials') else '○ not connected'} — {p.get('name', '')}", "has_credentials": bool(p.get("has_credentials"))} for p in provs]
            _pick = self._tui_pick("Login — pilih provider", items, show=lambda x: x["label"], initial="")
            if _pick is None:
                _print("[dim]Dibatalkan.[/dim]")
                return
            provider_name = items[_pick]["id"]

        if not provider_name:
            return

        # Direct-arg guard (filter :955): tolak pseudo-provider combo (api==local).
        _pn = (provider_name or "").lower().strip()
        if _pn == "combo" or _pn.startswith("combo/"):
            _print("[red]Pilih provider asli, bukan combo[/red] [dim](combo dipakai via /model combo/<nama>)[/dim]")
            return
        try:
            _desc = self._registry().get_provider_descriptor(_pn)
            if _desc is not None and str(_desc.get("api", "")) == "local":
                _print("[red]Pilih provider asli, bukan combo[/red] [dim](combo dipakai via /model combo/<nama>)[/dim]")
                return
        except Exception:
            pass

        _print(f"\n[cyan]Connecting to {provider_name}…[/cyan]")

        # ---- Antigravity (agy CLI, OAuth session) ----
        if provider_name == "antigravity":
            self._login_antigravity()

        # ---- Copilot ----
        elif provider_name == "copilot":
            try:
                from harness.models.providers.copilot import CopilotProvider
                prov = CopilotProvider()
                prov.device_flow_login()
                _print("[bold green]✅ Copilot authenticated.[/bold green]")
                self._ask_switch_provider("copilot", PROVIDER_DEFAULT_MODELS.get("copilot", lambda: "gpt-4o")())
            except Exception as e:
                _print(f"[bold red]Login failed: {e}[/bold red]")

        # ---- Gemini API (AI Studio API key) ----
        elif provider_name == "gemini":
            self._login_gemini()

        # ---- Generic API key providers ----
        else:
            vault_token = Prompt.ask(f"Enter {provider_name} API key") if RICH_AVAILABLE else input(f"{provider_name} API key: ")
            vault_token = vault_token.strip()
            if vault_token:
                from harness.models.auth_vault import AuthVault
                AuthVault().store_token(provider_name, vault_token)
                default_model = PROVIDER_DEFAULT_MODELS.get(provider_name, lambda: "default")()
                _print(f"[bold green]✅ {provider_name} credentials saved.[/bold green]")
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
                _print(f"[green]✅ Existing Antigravity session detected ({_em0}).[/green]")
            else:
                _print("[green]✅ Existing Antigravity session detected.[/green]")
            _print(f"[bold green]✅ Antigravity connected. Default model: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
            try:
                self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
            except Exception:
                pass
            return
        if _stored:
            _em0b = _safe_email()
            if _em0b:
                _print(f"[green]✅ Existing Antigravity session detected ({_em0b}).[/green]")
            else:
                _print("[green]✅ Existing Antigravity session detected.[/green]")
            _print(f"[bold green]✅ Antigravity connected. Default model: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
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
            _print("[yellow]GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET belum diset — OAuth exchange kemungkinan gagal.[/yellow]")
            _print("[dim]Set env GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau login via `agy` sekali lalu ulangi /provider antigravity.[/dim]")

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
            _print("[red]OAuth intercept gagal.[/red]")
            if _missing_creds or "client" in _intercept_error.lower() or "secret" in _intercept_error.lower() or "exchange" in _intercept_error.lower():
                _print("[dim]Set GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau login via `agy` sekali lalu ulangi /provider antigravity.[/dim]")
            _creds = None

        # (c) intercept None (user batal/timeout) → pesan batal jelas, lalu fallback (e).
        if _creds is None:
            if not _intercept_error:
                _print("[yellow]Login dibatalkan (timeout/pengguna membatalkan di browser).[/yellow]")
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
                _print(f"[bold green]✅ Antigravity connected. Default model: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
                try:
                    self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
                except Exception:
                    pass
                return
            if _intercept_error:
                _print("[dim]Tidak ada sesi agy fallback. Jalankan `agy` untuk login lalu ulangi /provider antigravity, atau ulangi OAuth setelah set client creds.[/dim]")
            else:
                _print("[dim]Tidak ada sesi agy fallback. Jalankan `agy` untuk login lalu ulangi /provider antigravity, atau ulangi OAuth.[/dim]")
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
            _print("[red]OAuth exchange gagal (creds kosong).[/red]")
            _print("[dim]Set GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, atau login via `agy` sekali lalu ulangi /provider antigravity.[/dim]")
            try:
                _fb2 = vault.discover_antigravity_token()
            except Exception:
                _fb2 = None
            if _fb2:
                try:
                    vault.store_token("antigravity", _fb2)
                except Exception:
                    pass
                _print(f"[bold green]✅ Antigravity connected. Default model: {DEFAULT_ANTIGRAVITY_MODEL}[/bold green]")
                try:
                    self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
                except Exception:
                    pass
                return
            _print("[dim]Tidak ada sesi agy fallback. Jalankan `agy` untuk login lalu ulangi /provider antigravity.[/dim]")
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
            _print(f"[bold green]✅ Antigravity connected via Google OAuth ({_shown_email})[/bold green]")
        else:
            _print("[bold green]✅ Antigravity connected via Google OAuth[/bold green]")
        try:
            self._ask_switch_provider("antigravity", DEFAULT_ANTIGRAVITY_MODEL)
        except Exception:
            pass

    def _login_gemini(self):
        """Connect Gemini via Google AI Studio API Key (for the REST generative language API)."""
        import webbrowser
        from harness.models.auth_vault import AuthVault
        vault = AuthVault()

        _print("[dim]Gemini API (Google AI Studio) — for gemini-2.5-flash, gemini-2.5-pro, etc.[/dim]")
        _print("[dim]For Antigravity models (gemini-3.8, claude, gpt-oss) use /provider antigravity instead.[/dim]")
        _print("[dim]Opening https://aistudio.google.com/app/apikey …[/dim]")

        try:
            webbrowser.open("https://aistudio.google.com/app/apikey")
        except Exception:
            pass

        token = Prompt.ask("Paste API key") if RICH_AVAILABLE else input("API key: ")
        token = token.strip()
        if token:
            vault.store_token("gemini", token)
            _print("[bold green]✅ Gemini API Key saved.[/bold green]")
            self._ask_switch_provider("gemini", "gemini-2.5-flash")
        else:
            _print("[yellow]No key entered — cancelled.[/yellow]")

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
                f"Switch active provider to [cyan]{provider}[/cyan] / [yellow]{model}[/yellow]?",
                choices=["y", "n"],
                default="n",
            )
        else:
            sw = input(f"Switch to {provider}/{model}? [y/N]: ").strip().lower()

        if sw == "y":
            self._set_active_provider(provider, model)
            _print(f"[green]Now using [bold]{provider}[/bold] / [bold]{model}[/bold][/green]")
        else:
            _print(f"[dim]Credentials saved. Still using {current_provider}. Run /model to switch anytime.[/dim]")

    # ------------------------------------------------------------------
    # /providers
    # ------------------------------------------------------------------

    def show_providers(self, filter_q: str = ""):
        providers = self._list_providers_fast()
        q = (filter_q or "").strip().lower()
        if q:
            providers = [p for p in providers if q in p["id"].lower() or q in str(p.get("name", "")).lower()]

        if RICH_AVAILABLE:
            table = Table(title=f"Providers 🔍 {filter_q.strip() if filter_q.strip() else '—'}", show_header=True, header_style="bold blue")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name")
            table.add_column("Status")
            for p in providers:
                status = "[green]✅ Connected[/green]" if p["has_credentials"] else "[dim]○ Not connected[/dim]"
                table.add_row(p["id"], p.get("name", p["id"]), status)
            console.print()
            console.print(table)
            console.print("[dim]nomor=pilih - teks=filter - /providers teks untuk filter[/dim]")
            console.print()
        else:
            print(f"\n╭─ Providers 🔍 {filter_q.strip() if filter_q.strip() else '—'} ─" + "─" * 20 + "╮")
            for p in providers:
                s = "✅" if p["has_credentials"] else "○"
                print(f"│   {s} {p['id']} ({p.get('name', '')})")
            print("│ nomor=pilih - teks=filter - /providers teks untuk filter")
            print("╰" + "─" * 40 + "╯\n")

    # ------------------------------------------------------------------
    # /models
    # ------------------------------------------------------------------

    def show_models(self, provider_id: str):
        registry = self._registry()
        models = registry.list_models(provider_id)
        if not models:
            _print(f"[yellow]No models found for '{provider_id}'.[/yellow]")
            return
        _print(f"\n[bold]Models for {provider_id}:[/bold]")
        for m in models:
            _print(f"  [cyan]{m}[/cyan]")
        _print()

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
            items = [{"id": p["id"], "label": f"{p['id']} {'✅ connected' if p.get('has_credentials') else '○ not connected'} — {p.get('name', '')}", "has_credentials": bool(p.get("has_credentials"))} for p in provs]
            _pick = self._tui_pick("Provider — pilih", items, show=lambda x: x["label"], initial="")
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
        _print(f"[red]Unknown provider '{_want}'.[/red] [dim](lihat /provider list)[/dim]")
        self._provider_list()
        return

    def _provider_prompt_secret(self, prompt_text: str = "API key (opsional, Enter=kosong): ") -> str:
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
            _raw_pid = (Prompt.ask("Provider id [a-z0-9-]") if RICH_AVAILABLE else input("Provider id [a-z0-9-]: "))
        except (EOFError, KeyboardInterrupt):
            _print("[dim]Dibatalkan.[/dim]")
            return
        _pid = (_raw_pid or "").strip()
        if not _pid or not _re.match(r"^[a-z0-9-]+$", _pid):
            _print("[red]ID tidak valid.[/red] [dim]Gunakan format [a-z0-9-] huruf-kecil/angka/strip, cth: my-provider[/dim]")
            return
        if _pid in _BUILTIN:
            _print(f"[red]ID '{_pid}' adalah provider bawaan.[/red] [dim]Pilih id lain (builtin tak bisa ditimpa).[/dim]")
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
            _raw_base = (Prompt.ask("Base URL (https://…/v1)") if RICH_AVAILABLE else input("Base URL (https://…/v1): "))
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
            _print("[red]Base URL tidak valid.[/red] [dim]Harus http(s)://host… cth: https://api.example.com/v1[/dim]")
            return
        # --- api key opsional (tersembunyi) ---
        _key = self._provider_prompt_secret("API key (opsional, Enter=kosong): ")
        # --- models: auto discovery → manual fallback ---
        _models: list = []
        try:
            _fetched, _ferr = self._provider_fetch_models(_base, _key, timeout=10.0)
        except Exception:
            _fetched, _ferr = [], "fetch failed"
        if _fetched:
            _models = list(_fetched)
            _print(f"[green]Discovered {len(_models)} models.[/green]")
        else:
            if _ferr:
                _print(f"[dim]Auto-discovery gagal ({_ferr}) — isi manual atau kosongkan.[/dim]")
            try:
                _raw_m = (Prompt.ask("Models koma (kosong=discovery menyusul)", default="") if RICH_AVAILABLE else input("Models koma (kosong=discovery menyusul): "))
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
            _ok_conn, _cerr = False, "connection test failed"
        if not _ok_conn:
            _print(f"[red]Tes koneksi gagal: {_cerr}[/red] [dim]Periksa Base URL/jaringan. Dibatalkan, tidak disimpan.[/dim]")
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
                self._providers_cache = None
                self._providers_ts = 0.0
                self._models_cache = None
                self._models_ts = 0.0
            except Exception:
                pass
        except Exception as _e:
            _print(f"[red]Gagal menyimpan provider: {_e}[/red]")
            return
        _print(f"[bold green]✅ Provider '{_pid}' tersimpan.[/bold green] [dim]({len(_models)} models)[/dim]")
        try:
            if _models:
                self._set_active_provider(_pid, _models[0])
                _print(f"✅ {_pid}/{_models[0]} aktif — langsung bisa chat.")
            else:
                self._set_active_provider(_pid, "")
                _print(f"✅ {_pid}/ aktif — langsung bisa chat. [dim]Run /model {_pid}/<nama> setelah discovery, atau /provider list untuk cek.[/dim]")
        except Exception:
            _print(f"[dim]Tersimpan. Gunakan /model {_pid}/<nama> untuk switch.[/dim]")

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
                _status = "✅ Connected" if _has else "○ Not connected"
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
            table = Table(title="Providers (custom via /provider add)", show_header=True, header_style="bold blue")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name")
            table.add_column("BaseURL", no_wrap=True)
            table.add_column("Models", justify="right")
            table.add_column("Status")
            for _pid, _name, _api_s, _n, _st in _rows:
                table.add_row(_pid, _name, _api_s or "—", str(_n), f"[green]{_st}[/green]" if "✅" in _st else f"[dim]{_st}[/dim]")
            console.print()
            console.print(table)
            console.print("[dim]/provider add · /provider remove <id> · /model untuk switch[/dim]")
            console.print()
        else:
            print("\n── Providers (custom via /provider add) ──")
            for _pid, _name, _api_s, _n, _st in _rows:
                _mark = "✅" if "✅" in _st else "○"
                print(f"  {_mark} {_pid} ({_name}) [{_n} models] {_api_s or ''}")
            print("  /provider add · /provider remove <id> · /model untuk switch\n")

    def _provider_remove(self, target: str = "") -> None:
        _pid_raw = (target or "").strip()
        if not _pid_raw:
            try:
                _pid_raw = (Prompt.ask("Provider id to remove") if RICH_AVAILABLE else input("Provider id to remove: "))
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
                _print(f"[red]Provider '{_want}' tidak ditemukan atau bukan custom.[/red] [dim](lihat /provider list)[/dim]")
            else:
                _print(f"[red]Tidak bisa hapus builtin '{_check}'.[/red] [dim]Hanya custom provider yang bisa dihapus.[/dim]")
            return
        try:
            _conf = (Prompt.ask(f"Hapus custom provider '{_check}'?", choices=["y", "n"], default="n") if RICH_AVAILABLE else input(f"Hapus '{_check}'? [y/N]: "))
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
            self._providers_cache = None
            self._providers_ts = 0.0
            self._models_cache = None
            self._models_ts = 0.0
        except Exception:
            pass
        _print(f"[bold green]✅ Provider '{_check}' dihapus.[/bold green]")

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
            _pick = self._tui_pick("Effort — pilih", items, show=lambda x: f"{x} ★ aktif" if x == shown else x, initial="")
            if _pick is None:
                if base in EFFORT_MODELS:
                    _print(f"[cyan]Effort:[/cyan] [bold]{shown}[/bold] [dim](model {base}-{shown})[/dim]")
                else:
                    _print(f"[cyan]Effort:[/cyan] [bold]{shown}[/bold] [dim](current model '{bare}' has no effort; stored default for Antigravity)[/dim]")
                return
            want = items[_pick]

        if want not in EFFORT_OPTIONS:
            _print(f"[red]Invalid effort '{want}'.[/red] [dim]Use: /effort low|medium|high[/dim]")
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
            _print(f"[bold green]✓ Effort set to {want}[/bold green] [dim]({provider}/{new_model})[/dim]")
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
            _print(f"[bold green]✓ Effort default saved: {want}[/bold green] [dim](current model '{bare}' has no effort knob)[/dim]")

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
            _print(f"[red]✗ tidak cocok: '{q}'[/red] [dim]coba /model <substring>[/dim]")
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
                        effort_choice = Prompt.ask(f"effort [low/medium/high, default {_def}]", default=_def)
                    else:
                        effort_choice = input(f"effort [low/medium/high, default {_def}]: ").strip() or _def
                    effort_choice = (effort_choice or "").strip().lower()
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
            _print(f"\n[green]Switched to [bold]{prov}[/bold] / [bold]{mod}[/bold][/green]\n")
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
            _pick = self._tui_pick("Model — pilih", _pool, show=_show_fn, initial=raw_q)
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
            _print("[yellow]No connected providers. Run /provider to connect.[/yellow]")
            return
        # Tanpa arg → TUI SEMUA connected (live filter, viewport 15 ikut highlight).
        _pool0 = connected
        _pick0 = self._tui_pick("Model — pilih", _pool0, show=_show_fn, initial="")
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
                _print("[yellow]No combos saved.[/yellow]")
                return
            try:
                _ap, _am = self.get_active_info()
            except Exception:
                _ap, _am = "", ""
            if RICH_AVAILABLE:
                table = Table(title="Combos", show_header=True, header_style="bold blue")
                table.add_column("Name", style="cyan")
                table.add_column("Strategy")
                table.add_column("Models")
                for name, data in combos.items():
                    _mark = " ★ aktif" if (_ap == "combo" and _am == name) else ""
                    table.add_row(f"{name}{_mark}", data["strategy"], ", ".join(data["models"]))
                console.print()
                console.print(table)
                console.print()
            else:
                print("\nCombos:")
                for name, data in combos.items():
                    _mark = " ★ aktif" if (_ap == "combo" and _am == name) else ""
                    print(f"  {name}{_mark} [{data['strategy']}]: {', '.join(data['models'])}")
                print()

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
                _print("[yellow]Usage: /combo remove <nama>[/yellow]")
                return
            try:
                _existing = manager.get_combo(_name)
            except Exception:
                _existing = None
            if not _existing:
                _print(f"[red]Combo '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list)[/dim]")
                return
            try:
                _conf = (Prompt.ask(f"Hapus combo '{_name}'?", choices=["y", "n"], default="n") if RICH_AVAILABLE else input(f"Hapus combo '{_name}'? [y/N]: "))
            except (EOFError, KeyboardInterrupt):
                _print("[dim]Dibatalkan.[/dim]")
                return
            if (_conf or "").strip().lower() != "y":
                _print("[dim]Dibatalkan.[/dim]")
                return
            try:
                manager.delete_combo(_name)
            except Exception as _e:
                _print(f"[red]Gagal hapus combo: {_e}[/red]")
                return
            _print(f"[bold green]✅ Combo '{_name}' dihapus.[/bold green]")
            return
        elif not _cmd or _cmd == "create":
            connected = self._list_connected_fast()
            if not connected:
                _print("[yellow]No connected providers. Run /provider first.[/yellow]")
                return

            # Alur: nama → strategi (12 semua) → multi-pilih model → simpan.
            name = (Prompt.ask("Combo name (empty back)") if RICH_AVAILABLE else input("Name (empty back): ")).strip()
            if not name or name.lower() in ("back", "q"):
                _print("[dim]Cancelled.[/dim]")
                return

            strategies = [s.value for s in ComboStrategy]
            # 12 strategi SEMUA tampil (viewport TUI 15) — Enter pilih, Esc batal.
            _spick = self._tui_pick("Combo — strategi", strategies, show=lambda x: x, initial="")
            if _spick is None:
                _print("[dim]Cancelled.[/dim]")
                return
            strategy = strategies[_spick]

            # Multi-pilih model: Spasi toggle ✓, Enter selesai (min 1), Esc/q batal.
            _mpicks = self._tui_pick_multi("Combo — models", connected, show=lambda m: m["id"], initial="")
            if not _mpicks:
                _print("[dim]Cancelled.[/dim]")
                return
            models = [connected[i]["id"] for i in _mpicks]

            manager.create_combo(name, strategy, models)
            _print(f"[bold green]✅ Combo '{name}' created.[/bold green] [dim]({strategy} · {len(models)} models · pakai via /model combo/{name})[/dim]")

            sw = Prompt.ask("Switch to this combo?", choices=["y", "n"], default="y") if RICH_AVAILABLE else input("Switch? [y/N]: ")
            if sw.strip().lower() == "y":
                self.switch_model(f"combo/{name}")
        else:
            _print("[yellow]Usage: /combo [list|create|use|edit|remove][/yellow] [dim]/combo use <nama> · /combo edit <nama> · /combo remove <nama> (type /help)[/dim]")

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
                _print("[yellow]No combos saved.[/yellow] [dim](buat via /combo create)[/dim]")
                return
            items = sorted(combos.keys())
            def _show(n):
                try:
                    _d = combos.get(n) or {}
                    return f"{n} [{_d.get('strategy', '?')}] — {', '.join(_d.get('models', []) or [])}"
                except Exception:
                    return str(n)
            _pick = self._tui_pick("Combo — pakai", items, show=_show, initial="")
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
            _print(f"[red]Combo '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list)[/dim]")
            return
        try:
            self.switch_model(f"combo/{_name}")
        except Exception as _e:
            _print(f"[red]Gagal pakai combo '{_name}': {_e}[/red]")

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
                _print("[yellow]No combos saved.[/yellow] [dim](buat via /combo create)[/dim]")
                return
            items = sorted(combos.keys())
            def _show2(n):
                try:
                    _d = combos.get(n) or {}
                    return f"{n} [{_d.get('strategy', '?')}] — {', '.join(_d.get('models', []) or [])}"
                except Exception:
                    return str(n)
            _pick0 = self._tui_pick("Combo — edit", items, show=_show2, initial="")
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
            _print(f"[red]Combo '{_name}' tidak ditemukan.[/red] [dim](lihat /combo list)[/dim]")
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
        _spick = self._tui_pick("Combo — strategi (baru)", strategies, show=_sshow, initial=_cur_strategy or "")
        if _spick is None:
            _print("[dim]Dibatalkan.[/dim]")
            return
        new_strategy = strategies[_spick]
        connected = self._list_connected_fast()
        if not connected:
            _print("[yellow]No connected providers. Run /provider first.[/yellow]")
            return
        try:
            _cur_txt = ", ".join(_cur_models) if _cur_models else "—"
        except Exception:
            _cur_txt = "—"
        _mpicks = self._tui_pick_multi(f"Combo — models (saat ini: {_cur_txt})", connected, show=lambda m: m["id"], initial="")
        if not _mpicks:
            _print("[dim]Dibatalkan.[/dim]")
            return
        new_models = [connected[i]["id"] for i in _mpicks]
        try:
            manager.create_combo(_canon, new_strategy, new_models, params=_cur_params)
        except TypeError:
            try:
                manager.create_combo(_canon, new_strategy, new_models)
            except Exception as _e:
                _print(f"[red]Gagal simpan combo: {_e}[/red]")
                return
        except Exception as _e:
            _print(f"[red]Gagal simpan combo: {_e}[/red]")
            return
        _print(f"[bold green]✅ Combo '{_canon}' diperbarui.[/bold green] [dim]({new_strategy} · {len(new_models)} models · pakai via /combo use {_canon})[/dim]")

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
            _print(f"[dim]dilayani oleh combo {_cn} → {_sp}/{_sm} member aktual[/dim]")
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
            _print(f"[dim]dilayani oleh combo {_cn} [{_strat}] · member: {_mem_txt} (serving member tak diekspos provider)[/dim]")
        else:
            _print(f"[dim]dilayani oleh combo {_cn} (detail combo tak ditemukan)[/dim]")
        try:
            logging.getLogger(__name__).warning("Handoff next-wave: ComboProvider tak mengekspos serving member (butuh serving_provider/serving_model di return dict) — lihat harness/models/combo.py:ComboProvider.chat")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # /history
    # ------------------------------------------------------------------

    def show_history(self):
        if not self.orchestrator:
            _print("[yellow]Orchestrator not initialised.[/yellow]")
            return

        ctx = self.orchestrator.compactor.get_context()
        fact_cards = ctx.get("fact_cards", [])
        history    = ctx.get("active_history", [])

        if not fact_cards and not history:
            _print("[dim]No conversation history yet.[/dim]")
            return

        if RICH_AVAILABLE:
            console.print()
            if fact_cards:
                _rule("Compressed Context (Fact Cards)", style="dim")
                for fc in fact_cards:
                    console.print(f"  [dim]{fc}[/dim]")
                console.print()

            if history:
                _rule("Active History", style="dim")
                for msg in history:
                    role  = msg.get("role", "unknown")
                    body  = msg.get("content", "")
                    color = "cyan" if role == "user" else "green"
                    label = "You ❯" if role == "user" else "AI ◆"
                    console.print(f"[bold {color}]{label}[/bold {color}]")
                    # Truncate very long entries for readability
                    if len(body) > 500:
                        body = body[:500] + "… [dim](truncated)[/dim]"
                    console.print(f"  {body}")
                    console.print()
        else:
            if fact_cards:
                print("\n── Fact Cards ──")
                for fc in fact_cards:
                    print(f"  {fc}")
            if history:
                print("\n── History ──")
                for msg in history:
                    print(f"[{'You ❯' if msg.get('role') == 'user' else 'AI ◆'}]: {msg.get('content','')[:300]}")
            print()

    # ------------------------------------------------------------------
    # /status, /config, /steer, /help
    # ------------------------------------------------------------------

    def show_status(self):
        provider, model = self.get_active_info()
        agents_exists = os.path.exists("AGENTS.md")
        hooks_exists  = os.path.exists("SYSTEM_HOOKS.md")
        state = self.orchestrator.workflow_state if self.orchestrator else "N/A"

        if RICH_AVAILABLE:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_row("[dim]Provider[/dim]",  f"[cyan]{provider}[/cyan]")
            table.add_row("[dim]Model[/dim]",     f"[yellow]{model}[/yellow]")
            table.add_row("[dim]State[/dim]",     state)
            table.add_row("[dim]AGENTS.md[/dim]", "[green]loaded[/green]" if agents_exists else "[dim]not found[/dim]")
            table.add_row("[dim]HOOKS[/dim]",     "[green]loaded[/green]" if hooks_exists  else "[dim]not found[/dim]")
            console.print()
            console.print(table)
            console.print()
        else:
            print(f"\nProvider: {provider}\nModel: {model}\nState: {state}")
            print(f"AGENTS.md: {'yes' if agents_exists else 'no'}")
            print(f"HOOKS: {'yes' if hooks_exists else 'no'}\n")

    def show_config(self):
        if RICH_AVAILABLE:
            import json
            from pydantic import BaseModel
            data = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config.__dict__
            console.print()
            console.print_json(json.dumps(data, default=str))
            console.print()
        else:
            print(self.config.__dict__)

    def steer_orchestrator(self, instruction: str):
        instruction = (instruction or "").strip()
        if not instruction:
            _print("[yellow]Usage: /steer <instruction>[/yellow]")
            return
        if not self.orchestrator:
            _print("[yellow]Tidak ada tugas berjalan[/yellow] [dim](jalankan tugas dulu, lalu /steer)[/dim]")
            return
        # Steer jujur: hanya RUNNING boleh ✓; selain itu tolak tanpa ✓ palsu.
        try:
            _state = getattr(self.orchestrator, "workflow_state", None)
            _active = getattr(self.orchestrator, "active_subagent_id", None)
        except Exception:
            _state, _active = None, None
        if _state != "RUNNING" or not _active:
            _print("[yellow]Tidak ada tugas berjalan[/yellow] [dim](jalankan tugas dulu, lalu /steer)[/dim]")
            return
        _print(f"[blue]◉ steering → {instruction}[/blue]")
        try:
            async def _do():
                try:
                    await self.orchestrator.steer_active_subagent(instruction)
                except Exception:
                    await self.orchestrator.process_user_input(instruction)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, _do()).result()
            else:
                asyncio.run(_do())
            _print("[green]✓ steered[/green]")
        except ValueError as e:
            _print(f"[yellow]Nothing to steer: {e}[/yellow] [dim](run a task first)[/dim]")
        except Exception as e:
            _print(f"[red]Steer failed: {e}[/red] [dim](type /help)[/dim]")

    def show_help(self):
        groups = [
            ("Model", [("/model (/m)", "Switch — fuzzy query direct or selector; base-effort inline"),
                       ("/effort (/e)", "Show/set Antigravity effort (persisted)"),
                       ("/combo (/c)", "Manage combos [list|create|use|edit|remove]")]),
            ("Auth", [("/provider [id]", "Authenticate/switch provider"),
                      ("/providers (/p)", "List providers + status"),
                      ("/provider add|list|remove", "Manage custom providers"),
                      ("/models <prov>", "List models for provider")]),
            ("Session", [("/steer (/st)", "Steer active subagent"),
                         ("/history", "Show history + fact cards"),
                         ("/status (/s)", "Provider, model, state"),
                         ("/config", "Show runtime config (JSON)")]),
            ("Help", [("/help (/h)", "This help"), ("/quit (/q)", "Exit")]),
        ]
        if RICH_AVAILABLE:
            for g, rows in groups:
                console.print(f"[bold blue]{g}[/bold blue]")
                table = Table(show_header=False, box=None, padding=(0, 2))
                for cmd, desc in rows:
                    table.add_row(f"[cyan]{cmd}[/cyan]", f"[dim]{desc}[/dim]")
                console.print(table)
            console.print("[dim]Tip: Ctrl-C interrupts, /steer redirects · prefix works; unknown → did-you-mean.[/dim]")
        else:
            for g, rows in groups:
                print(f"\n[{g}]")
                for cmd, desc in rows:
                    print(f"  {cmd:<18} {desc}")
            print("\nTip: Ctrl-C interrupts, /steer redirects · prefix works; unknown → did-you-mean.")
