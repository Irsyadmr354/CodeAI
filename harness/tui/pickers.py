"""Picker TUI: parser CSI penuh + pilih tunggal/ganda (stdlib saja).

Pola ditiru dari `_TUIPicker` di `harness/cli.py` (hanya dibaca, tak diubah):
panah/home/end, j/k, Enter, Esc/q/Ctrl-C, Spasi, live-search.

Pemisahan testable:
- murni (tanpa TTY): parse_key, filter_items, move_index, render_lines,
  _Picker.handle, _Picker.render, pick_*(keys=[...]).
- impure (butuh TTY): read_key, loop interaktif di pick_* tanpa `keys`.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .components import footer_petunjuk

__all__ = [
    "VIEWPORT",
    "parse_key",
    "parse_key_sequence",
    "filter_items",
    "move_index",
    "render_lines",
    "available",
    "read_key",
    "pick_single",
    "pick_multi",
]

VIEWPORT = 10


def parse_key(seq: str) -> str:
    """Parser murni sekuens-key menjadi token (tanpa TTY).

    Token: up/down/left/right/home/end/enter/backspace/ctrl-c/space/
    esc/j/k/q, 1 huruf live-search, atau unknown. CSI tak dikenal -> unknown
    (jangan esc agar tak batal mendadak).
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
        if final == "H":
            return "home"
        if final == "F":
            return "end"
        if final == "~":
            try:
                num = seq[2:-1].split(";")[0].strip()
                if num in ("1", "7"):
                    return "home"
                if num in ("4", "8"):
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


def parse_key_sequence(seq: str) -> str:
    """Alias kompatibel untuk `parse_key`."""
    return parse_key(seq)


def filter_items(
    items: Sequence[Any] | None,
    query: str | None,
    show: Callable[[Any], str] | None = None,
) -> List[Tuple[int, Any]]:
    """Filter substring case-insensitive; kembali [(indeks_asli, item)]."""
    tampil = show if callable(show) else (lambda x: x)
    q = (query or "").strip().lower()
    if not q:
        return list(enumerate(list(items or [])))

    def _label(it: Any) -> str:
        try:
            return str(tampil(it))
        except Exception:
            return str(it)

    keluar: List[Tuple[int, Any]] = []
    for i, it in enumerate(list(items or [])):
        try:
            if q in _label(it).lower():
                keluar.append((i, it))
        except Exception:
            continue
    return keluar


def move_index(idx: int, n: int, key: str) -> int:
    """Gerak highlight wrap-around: up/down/j/k/home/end, lain tetap."""
    if n <= 0:
        return 0
    try:
        pos = int(idx) % n
    except Exception:
        pos = 0
    k = (key or "").strip().lower()
    if k == "home":
        return 0
    if k == "end":
        return n - 1
    if k in ("up", "k"):
        return (pos - 1) % n
    if k in ("down", "j"):
        return (pos + 1) % n
    return pos


def _potong_akhir(s: str, budget: int) -> str:
    t = str(s).replace("\r", " ").replace("\n", " ")
    if len(t) <= budget:
        return t
    if budget <= 1:
        return t[:budget]
    return t[: budget - 1] + "…"


def render_lines(
    judul: str,
    filtered: Sequence[Tuple[int, Any]],
    idx: int,
    query: str,
    multi: bool = False,
    selected: Sequence[int] | None = (),
    viewport: int = VIEWPORT,
    show: Callable[[Any], str] | None = None,
) -> List[str]:
    """Render murni: 1 item = 1 baris vertikal (tanpa I/O, tanpa clear)."""
    import shutil

    try:
        lebar = int(shutil.get_terminal_size(fallback=(80, 24)).columns)
    except Exception:
        lebar = 80
    if lebar < 20:
        lebar = 80
    maks = max(20, min(lebar, 100) - 4)
    tampil = show if callable(show) else (lambda x: x)

    def _label(it: Any) -> str:
        try:
            return str(tampil(it))
        except Exception:
            return str(it)

    daftar = list(filtered or [])
    total = len(daftar)
    try:
        pos = int(idx)
    except Exception:
        pos = 0
    if total <= 0:
        pos = 0
    else:
        pos = max(0, min(pos, total - 1))
    try:
        vs = int(viewport)
    except Exception:
        vs = VIEWPORT
    if vs <= 0:
        vs = VIEWPORT
    awal = 0
    if total > vs:
        awal = max(0, min(pos - vs + 1, total - vs))
        # Jaga highlight terlihat: geser minimal agar pos dalam jendela.
        if pos < awal:
            awal = pos
    tandai = set()
    try:
        for v in list(selected or []):
            tandai.add(int(v))
    except Exception:
        pass
    baris = [
        _potong_akhir(f"{str(judul or 'Pilih').strip() or 'Pilih'}  ▶ {pos + 1 if total else 0}/{total}", maks),
        _potong_akhir(f"Cari: {(query or '')}▊", maks),
    ]
    for r in range(awal, min(awal + vs, total)):
        oi, it = daftar[r]
        mentah = _label(it).replace("\r", " ").replace("\n", " ")
        mark = ("✓ " if oi in tandai else "  ") if multi else ""
        pre = "▶ " if r == pos else "  "
        num = f"{r + 1:2}. "
        budget = maks - len(pre) - len(num) - len(mark)
        if budget < 1:
            budget = 1
        lab = _potong_akhir(mentah, budget)
        teks = f"{pre}{num}{mark}{lab}"
        if r == pos:
            baris.append(f"\x1b[7m{teks}\x1b[0m")
        else:
            baris.append(teks)
    if total == 0:
        baris.append(_potong_akhir(f"  ✗ tidak cocok: '{(query or '').strip()}' — coba kata lain", maks))
    if multi:
        baris.append(_potong_akhir(f"  [terpilih {len(tandai)} · Spasi=Tandai · Enter=Selesai (min 1)]", maks))
    baris.append(footer_petunjuk(multi=bool(multi)))
    return baris


def available() -> bool:
    """True bila stdin/stdout TTY dan modul raw tersedia."""
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
        import termios  # noqa: F401
        import tty  # noqa: F401
        import select  # noqa: F401
        return True
    except Exception:
        return False


def read_key() -> str:
    """Baca satu tombol mentah lalu petakan via `parse_key` (butuh TTY)."""
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
        chn = sys.stdin.read(1)
        if not chn:
            break
        seq += chn
        if chn.isalpha() or chn == "~":
            break
    return parse_key(seq)


def _bersih_awal(items: Sequence[Any] | None, awal: Sequence[int] | None) -> List[int]:
    n = len(list(items or []))
    keluar: List[int] = []
    tampak = set()
    for v in list(awal or []):
        try:
            oi = int(v)
        except Exception:
            continue
        if 0 <= oi < n and oi not in tampak:
            tampak.add(oi)
            keluar.append(oi)
    return keluar


class _Picker:
    """State picker testable: render murni + handle murni, tanpa TTY."""

    def __init__(
        self,
        judul: str = "",
        items: Sequence[Any] | None = None,
        show: Callable[[Any], str] | None = None,
        query: str = "",
        multi: bool = False,
        initial_selected: Sequence[int] | None = None,
    ):
        self.judul = str(judul or "").strip() or "Pilih"
        self.items = list(items) if items else []
        self.show = show if callable(show) else (lambda x: x)
        self.query = (query or "").strip()
        self.multi = bool(multi)
        self.idx = 0
        self.selected: List[int] = _bersih_awal(self.items, initial_selected) if self.multi else []

    def filtered(self) -> List[Tuple[int, Any]]:
        return filter_items(self.items, self.query, self.show)

    def render(self) -> List[str]:
        return render_lines(
            self.judul,
            self.filtered(),
            self.idx,
            self.query,
            multi=self.multi,
            selected=self.selected,
            viewport=VIEWPORT,
            show=self.show,
        )

    def handle(self, key: str) -> Optional[str]:
        """Mutasi query/idx/selected; kembali 'select'/'cancel'/None."""
        filt = self.filtered()
        total = len(filt)
        if key in ("up", "down", "home", "end"):
            if total:
                self.idx = move_index(self.idx, total, key)
            return None
        if key in ("left", "right", "unknown"):
            return None
        if isinstance(key, str) and key in ("j", "k"):
            if total:
                self.idx = move_index(self.idx, total, key)
            return None
        if key in ("esc", "q", "ctrl-c"):
            return "cancel"
        if key == "backspace":
            if self.query:
                self.query = self.query[:-1]
                self.idx = 0
            return None
        if key == "enter":
            if not total:
                return "cancel"
            if self.multi:
                if self.selected:
                    return "select"
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
            return None
        if isinstance(key, str) and len(key) == 1 and (key.isprintable() or ord(key) > 127):
            self.query += key
            self.idx = 0
            return None
        return None

    def result_single(self) -> Optional[int]:
        filt = self.filtered()
        if not filt:
            return None
        return filt[min(self.idx, len(filt) - 1)][0]

    def result_multi(self) -> List[int]:
        return list(self.selected)

    def run_keys(self, keys: Sequence[str] | None) -> Any:
        """Simulasi non-TTY dari daftar token (testable, tanpa TTY)."""
        for tok in list(keys or []):
            if tok is None:
                continue
            t = tok if isinstance(tok, str) else str(tok)
            tl = t.lower()
            if tl in ("up", "down", "home", "end", "left", "right", "enter", "esc", "backspace", "space", "ctrl-c", "unknown"):
                aksi = self.handle(tl)
            elif len(t) == 1:
                if t == " ":
                    aksi = self.handle("space")
                elif tl in ("j", "k", "q"):
                    aksi = self.handle(tl)
                else:
                    self.query += t
                    self.idx = 0
                    continue
            else:
                # Token >1 huruf = ketik harfiah per huruf (j/k/q = query).
                for ch in t:
                    self.query += ch
                self.idx = 0
                continue
            if aksi == "select":
                return self.result_multi() if self.multi else self.result_single()
            if aksi == "cancel":
                return [] if self.multi else None
        return [] if self.multi else None


def _tampil() -> None:
    try:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
    except Exception:
        pass


def _pulih(old: Any) -> None:
    try:
        if old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    except Exception:
        pass
    try:
        sys.stdout.write("\x1b[?25h\x1b[0m\n")
        sys.stdout.flush()
    except Exception:
        pass


def pick_single(
    judul: str,
    items: Sequence[Any] | None,
    show: Callable[[Any], str] | None = None,
    keys: Sequence[str] | None = None,
) -> Optional[int]:
    """Pilih satu: kembali indeks asli atau None bila batal/kosong."""
    if not list(items or []):
        return None
    pk = _Picker(judul=judul, items=items, show=show, multi=False)
    if keys is not None:
        return pk.run_keys(list(keys))
    if not available():
        return None
    import termios

    try:
        old = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        old = None
    try:
        if old is not None:
            from harness.cli import _tui_setraw as _raw  # pakai ulang, tanpa duplikasi
            _raw(sys.stdin.fileno())
    except Exception:
        try:
            import tty

            tty.setraw(sys.stdin.fileno())
        except Exception:
            pass
    try:
        _tampil()
        while True:
            try:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write("\x1b[?25l")
                sys.stdout.write("\n".join(pk.render()))
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            aksi = pk.handle(read_key())
            if aksi == "select":
                return pk.result_single()
            if aksi == "cancel":
                return None
    finally:
        _pulih(old)


def pick_multi(
    judul: str,
    items: Sequence[Any] | None,
    show: Callable[[Any], str] | None = None,
    initial_selected: Sequence[int] | None = None,
    keys: Sequence[str] | None = None,
) -> List[int]:
    """Pilih banyak: kembali daftar indeks (pre-tandai via initial_selected)."""
    if not list(items or []):
        return []
    pk = _Picker(judul=judul, items=items, show=show, multi=True, initial_selected=initial_selected)
    if keys is not None:
        hasil = pk.run_keys(list(keys))
        return list(hasil) if hasil is not None else []
    if not available():
        return list(pk.selected)
    import termios

    try:
        old = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        old = None
    try:
        if old is not None:
            from harness.cli import _tui_setraw as _raw  # pakai ulang, tanpa duplikasi
            _raw(sys.stdin.fileno())
    except Exception:
        try:
            import tty

            tty.setraw(sys.stdin.fileno())
        except Exception:
            pass
    try:
        _tampil()
        while True:
            try:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write("\x1b[?25l")
                sys.stdout.write("\n".join(pk.render()))
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            aksi = pk.handle(read_key())
            if aksi == "select":
                return pk.result_multi()
            if aksi == "cancel":
                return []
    finally:
        _pulih(old)
