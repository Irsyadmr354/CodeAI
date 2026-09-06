"""Komponen TUI murni (tanpa I/O, stdlib saja).

Aturan keras:
- Tidak ada dump lebih dari 5 ID dalam satu baris.
- Saran perintah hanya yang valid: /model /provider /combo /bantuan.
- Satu informasi hanya muncul di satu tempat.
"""

from __future__ import annotations

from typing import List, Sequence


VALID_COMMANDS = ("/model", "/provider", "/combo", "/bantuan")

__all__ = [
    "VALID_COMMANDS",
    "status_baku",
    "header_pos",
    "short_id",
    "model_lines",
    "error_box",
    "confirm_lines",
    "footer_petunjuk",
]


def status_baku(connected: bool) -> str:
    """Status koneksi baku, satu baris."""
    return "● Terhubung" if connected else "○ Belum terhubung"


def header_pos(label: str, pos: int, total: int, connected: bool) -> str:
    """Kepala baris: label + posisi + status (satu baris, tanpa newline)."""
    nama = str(label if label is not None else "").strip() or "Pilih"
    try:
        total_i = int(total)
    except Exception:
        total_i = 0
    try:
        pos_i = int(pos)
    except Exception:
        pos_i = 0
    if total_i < 0:
        total_i = 0
    if total_i == 0:
        pos_i = 0
    else:
        if pos_i < 1:
            pos_i = 1
        if pos_i > total_i:
            pos_i = total_i
    return f"{nama}  ▶ {pos_i}/{total_i}  {status_baku(bool(connected))}"


def short_id(text: object, width: int = 40) -> str:
    """Potong ID panjang di tengah: 'abc…xyz' agar hemat tempat."""
    s = str(text if text is not None else "")
    s = s.replace("\r", " ").replace("\n", " ")
    try:
        w = int(width)
    except Exception:
        w = 40
    if w <= 0:
        return ""
    if len(s) <= w:
        return s
    if w < 4:
        return s[:w]
    keep = w - 1  # 1 untuk "…"
    head = (keep + 1) // 2
    tail = keep - head
    if tail <= 0:
        return s[:head] + "…"
    return s[:head] + "…" + s[-tail:]


def model_lines(models: Sequence[object] | None, limit: int = 5) -> List[str]:
    """Daftar model vertikal: 1 model = 1 baris, maks `limit` + overflow.

    Overflow selalu berbentuk '… +N lainnya' (tidak pernah sebaris >5 ID).
    """
    if not models:
        return ["(belum ada model)"]
    try:
        lim = int(limit)
    except Exception:
        lim = 5
    if lim <= 0:
        lim = 5
    daftar = list(models)
    keluar: List[str] = []
    for i, m in enumerate(daftar[:lim]):
        keluar.append(f"{i + 1}. {short_id(str(m))}")
    sisa = len(daftar) - lim
    if sisa > 0:
        keluar.append(f"… +{sisa} lainnya")
    return keluar


def error_box(judul: str, sebab: str, langkah: Sequence[str] | None) -> str:
    """Kotak galat ringkas: Judul / Sebab / Langkah bernomor, tanpa redundansi."""
    j = str(judul if judul is not None else "").strip() or "Terjadi kesalahan"
    s = str(sebab if sebab is not None else "").strip()
    mentah = list(langkah) if langkah else []
    bersih: List[str] = []
    terlihat = set()
    for item in mentah:
        t = str(item if item is not None else "").strip()
        if not t:
            continue
        kunci = t.lower()
        if kunci in terlihat:
            continue
        if kunci == j.lower():
            continue
        if s and kunci == s.lower():
            continue
        terlihat.add(kunci)
        bersih.append(t)
    baris = [f"✗ {j}"]
    if s and s.lower() != j.lower():
        baris.append(f"Sebab: {s}")
    if bersih:
        baris.append("Langkah:")
        for n, t in enumerate(bersih, 1):
            baris.append(f"  {n}. {t}")
    return "\n".join(baris)


def confirm_lines(pertanyaan: str) -> str:
    """Baris konfirmasi tunggal dengan isyarat (y/n)."""
    q = str(pertanyaan if pertanyaan is not None else "").strip()
    q = q.rstrip(":").strip() or "Lanjutkan?"
    return f"{q} (y/n): "


def footer_petunjuk(multi: bool = False) -> str:
    """Satu baris petunjuk; hanya menyebut perintah valid (/bantuan)."""
    dasar = "Ketik cari · ↑↓/j/k pindah · Enter pilih · Esc batal"
    if bool(multi):
        dasar += " · Spasi tandai"
    return dasar + " · /bantuan"
