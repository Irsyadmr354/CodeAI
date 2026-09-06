"""Layar TUI model: baca / pakai (use/edit) / hapus + posisi.

Logika disalin dari ``harness/cli.py`` (switch_model, _status_baku,
_list_connected_fast) — hanya disalin, tidak mengimpor modul berat agar
tetap stdlib dan tidak menyentuh berkas lain.

Single source: langsung dari harness.tui.components/pickers (import gagal
= error nyata, jangan sembunyikan).
Bahasa: Indonesia. Stdlib saja.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Single source: langsung dari komponen/picker asli (tanpa fallback lokal)
# ---------------------------------------------------------------------------

HANDOFF = (
    "Single source: harness/tui/components.py (status_baku, header_pos, "
    "short_id, model_lines, error_box, footer_petunjuk) dan "
    "harness/tui/pickers.py (pick_single, pick_multi) dipakai langsung; "
    "import gagal = error nyata."
)

from harness.tui.components import (
    status_baku,
    header_pos,
    short_id,
    model_lines,
    error_box,
    footer_petunjuk,
)
from harness.tui.pickers import pick_single, pick_multi


# ---------------------------------------------------------------------------
# Konstanta + bantuan murni
# ---------------------------------------------------------------------------

BATAS_BARIS = 8
MAKS_ID = 5
SARAN_MAKS = 5

__all__ = [
    "HANDOFF",
    "BATAS_BARIS",
    "MAKS_ID",
    "SARAN_MAKS",
    "render_model_list",
    "render_model_detail",
    "suggest_models",
    "run_model_read",
    "run_model_use",
    "run_model_edit",
    "run_model_remove",
    "read_models",
    "baca_daftar",
    "read_model_detail",
    "baca_detail",
    "use_model",
    "edit_model",
    "remove_model",
]


def _model_id(item: Any) -> str:
    """Ambil ID model dari dict {'id': ...} atau teks biasa."""
    if isinstance(item, dict):
        return str(item.get("id", "") or "").strip()
    return str(item or "").strip()


def _model_provider(item: Any) -> str:
    """Ambil penyedia: dict['provider'] atau potong 'prov/model'."""
    if isinstance(item, dict):
        prov = str(item.get("provider", "") or "").strip()
        if prov:
            return prov
    mid = _model_id(item)
    if "/" in mid:
        kiri = mid.split("/", 1)[0].strip()
        if kiri:
            return kiri
    return "—"


def _model_terhubung(item: Any) -> bool:
    """Status terhubung: dukung connected/has_credentials/status."""
    try:
        if isinstance(item, dict):
            if isinstance(item.get("connected"), bool):
                return bool(item.get("connected"))
            if isinstance(item.get("has_credentials"), bool):
                return bool(item.get("has_credentials"))
            st = item.get("status")
            if isinstance(st, str) and st.strip():
                if "●" in st:
                    return True
                if "○" in st:
                    return False
    except Exception:
        pass
    # Daftar yang diberikan adalah model terhubung (mirror _list_connected_fast).
    return True


def _status_teks(item: Any) -> str:
    """Status satu baris via kontrak status_baku."""
    return str(status_baku(bool(_model_terhubung(item))))


def _ringkas(mid: str, lebar: int = 40) -> str:
    """ID ringkas via kontrak short_id."""
    return str(short_id(mid, lebar))


def _kepala(judul: str, pos: int, total: int, connected: bool) -> str:
    """Kepala via kontrak header_pos."""
    return str(header_pos(judul, int(pos), int(total), bool(connected)))


def _kaki(multi: bool = False) -> str:
    """Kaki via kontrak footer_petunjuk."""
    return str(footer_petunjuk(bool(multi)))


def _galat(judul: str, sebab: str = "", langkah: Any = None) -> str:
    """Galat via kontrak error_box (saran hanya perintah valid)."""
    if langkah is None:
        langkah = ["/model", "/provider list", "/bantuan"]
    return str(error_box(judul, sebab, list(langkah)))


def _tulis(out: Any, teks: str) -> None:
    """Tulis via out() atau print (tak pernah crash)."""
    try:
        if callable(out):
            out(teks)
        else:
            print(teks)
    except Exception:
        try:
            print(teks)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Baca (Read): daftar Compact + detail satu model
# ---------------------------------------------------------------------------

def render_model_list(
    models: Sequence[Any] | None,
    active_id: str = "",
    offset: int = 0,
    limit: int = BATAS_BARIS,
) -> str:
    """Daftar Compact: '1. id — ● Terhubung', maks 8 baris + posisi.

    Footer posisi selalu ada; aktif ditandai ' ★ aktif'.
    """
    daftar = list(models or [])
    total = len(daftar)
    aktif = str(active_id or "").strip()
    if total == 0:
        baris = [
            f"{_status_teks({})} — belum ada model — Langkah 1: ketik /provider",
            "Contoh: /provider gemini",
        ]
        return "\n".join(baris)
    try:
        lim = int(limit)
    except Exception:
        lim = BATAS_BARIS
    if lim <= 0:
        lim = BATAS_BARIS
    if lim > BATAS_BARIS:
        lim = BATAS_BARIS
    try:
        off = int(offset)
    except Exception:
        off = 0
    if off < 0:
        off = 0
    if off >= total:
        off = max(0, total - lim)
    potongan = daftar[off: off + lim]
    pos_awal = off + 1
    pos_akhir = off + len(potongan)
    kepala = _kepala("Model", pos_awal if total else 0, total, True)
    # Posisi eksplisit per spec ("Model X dari N" + rentang jendela).
    if total and len(potongan) != total:
        posisi = f"Model {pos_awal}-{pos_akhir} dari {total}"
    else:
        posisi = f"Model {pos_akhir} dari {total}" if total else "—"
    baris = [kepala]
    for nomor, item in enumerate(potongan, start=pos_awal):
        mid = _model_id(item) or "—"
        st = _status_teks(item)
        tanda = " ★ aktif" if (aktif and mid == aktif) else ""
        baris.append(f"{nomor}. {_ringkas(mid)} — {st}{tanda}")
    baris.append(posisi)
    baris.append(_kaki(False))
    return "\n".join(baris)


def render_model_detail(
    model: Any,
    pos: int = 0,
    total: int = 0,
    active_id: str = "",
) -> str:
    """Detail satu model: label + status + posisi + provider."""
    mid = _model_id(model) or "—"
    prov = _model_provider(model)
    st = _status_teks(model)
    aktif = str(active_id or "").strip()
    label = mid + (" ★ aktif" if (aktif and mid == aktif) else "")
    try:
        pos_i = int(pos)
    except Exception:
        pos_i = 0
    try:
        total_i = int(total)
    except Exception:
        total_i = 0
    if total_i <= 0:
        posisi = "—"
    else:
        if pos_i < 1:
            pos_i = 1
        if pos_i > total_i:
            pos_i = total_i
        posisi = f"Model {pos_i} dari {total_i}"
    return (
        f"Label: {_ringkas(label)}\n"
        f"Penyedia: {prov}\n"
        f"Status: {st}\n"
        f"Posisi: {posisi}"
    )


def suggest_models(kueri: Any, models: Sequence[Any] | None, n: int = SARAN_MAKS) -> List[str]:
    """Saran valid saja: maks n ID yang ada (mirror cli._suggest)."""
    try:
        maks = int(n)
    except Exception:
        maks = SARAN_MAKS
    if maks <= 0:
        maks = SARAN_MAKS
    if maks > MAKS_ID:
        maks = MAKS_ID
    ids = [_model_id(m) for m in (models or [])]
    ids = [i for i in ids if i]
    if not ids:
        return []
    q = str(kueri or "").strip()
    if not q:
        return ids[:maks]
    try:
        mirip = difflib.get_close_matches(q, ids, n=maks, cutoff=0.3)
    except Exception:
        mirip = []
    if mirip:
        return mirip[:maks]
    # Fallback substring dulu agar relevan, lalu awal daftar.
    try:
        rendah = q.lower()
        hits = [i for i in ids if rendah in i.lower()][:maks]
        if hits:
            return hits
    except Exception:
        pass
    return ids[:maks]


# Alias baca (Read)
read_models = render_model_list
baca_daftar = render_model_list
read_model_detail = render_model_detail
baca_detail = render_model_detail


# ---------------------------------------------------------------------------
# Interaktif tipis (run_*): picker + alih tipis
# ---------------------------------------------------------------------------

def _panggil_pick_single(
    fn: Any,
    judul: str,
    items: Sequence[Any],
    show: Callable[[Any], str],
    awal: str = "",
) -> Optional[int]:
    """Panggil pick_single langsung (single source, tanpa gaya lama)."""
    return fn(judul, items, show=show)


def run_model_read(
    models: Sequence[Any] | None,
    active_id: str = "",
    offset: int = 0,
    out: Callable[[str], None] | None = None,
) -> str:
    """Cetak + kembalikan daftar (tipis, tanpa logika)."""
    teks = render_model_list(models, active_id=active_id, offset=offset)
    _tulis(out or print, teks)
    return teks


def run_model_use(
    models: Sequence[Any] | None,
    active_id: str = "",
    kueri: str = "",
    pick_fn: Callable[..., Optional[int]] | None = None,
    switch_fn: Callable[[str], None] | None = None,
    out: Callable[[str], None] | None = None,
) -> Optional[str]:
    """Pilih model aktif (use): cocok unik langsung alih, else picker.

    Mirror switch_model: digit 1-based, persis, persis-ci, substringunik
    langsung alih; 0/banyak → picker prefilled (live-search di dalam).
    Batal → 'Dibatalkan.'; 0 cocok → tolak + saran valid (maks 5).
    Return ID terpilih atau None.
    """
    tulis: Callable[[str], None] = out or (lambda s: print(s))
    daftar = list(models or [])
    if not daftar:
        msg = _galat(
            "Belum terhubung — belum ada model.",
            "Langkah 1: ketik /provider",
            ["/provider", "/provider gemini", "/bantuan"],
        )
        _tulis(tulis, msg)
        return None
    ids = [_model_id(m) for m in daftar]

    def _alih(mid: str) -> str:
        if callable(switch_fn):
            try:
                switch_fn(mid)
            except Exception:
                pass
        else:
            try:
                if "/" in mid:
                    prov, mod = mid.split("/", 1)
                else:
                    prov, mod = "—", mid
                _tulis(tulis, f"Pindah ke {prov.strip() or '—'} / {mod.strip() or '—'}")
            except Exception:
                pass
        return mid

    q = str(kueri or "").strip()
    if q:
        # Digit 1-based (mirror cli).
        if q.isdigit():
            try:
                idx = int(q) - 1
                if 0 <= idx < len(daftar):
                    return _alih(ids[idx])
            except Exception:
                pass
        # Persis.
        for i, mid in enumerate(ids):
            if mid == q:
                return _alih(mid)
        # Persis case-insensitive unik.
        try:
            rendah = q.lower()
            sama = [i for i, mid in enumerate(ids) if mid.lower() == rendah]
            if len(sama) == 1:
                return _alih(ids[sama[0]])
        except Exception:
            sama = []
        # Substring unik langsung alih.
        try:
            hits = [i for i, mid in enumerate(ids) if q.lower() in mid.lower()]
        except Exception:
            hits = []
        if len(hits) == 1:
            return _alih(ids[hits[0]])
    else:
        hits = []

    # Picker (live-search di dalam; prefilled = q).
    pool = daftar
    show = lambda m: _model_id(m)  # noqa: E731
    pilih = pick_fn or pick_single
    try:
        idx = _panggil_pick_single(pilih, "Model — pilih", pool, show, awal=q)
    except Exception:
        idx = None
    if idx is None:
        if q and len(hits) == 0:
            saran = suggest_models(q, daftar, SARAN_MAKS)
            saran_teks = "\n".join(model_lines(saran, SARAN_MAKS))
            msg = _galat(
                f"Tidak cocok: '{q}'",
                "coba /model <kata kunci>",
                ["/model gemini", "/model", "/bantuan"],
            )
            _tulis(tulis, msg + (f"\n{saran_teks}" if saran else ""))
        else:
            _tulis(tulis, "Dibatalkan.")
        return None
    try:
        i = int(idx)
    except Exception:
        _tulis(tulis, "Dibatalkan.")
        return None
    if not (0 <= i < len(pool)):
        _tulis(tulis, "Dibatalkan.")
        return None
    return _alih(_model_id(pool[i]))


def run_model_edit(
    models: Sequence[Any] | None,
    active_id: str = "",
    kueri: str = "",
    pick_fn: Callable[..., Optional[int]] | None = None,
    switch_fn: Callable[[str], None] | None = None,
    out: Callable[[str], None] | None = None,
) -> Optional[str]:
    """Edit = pilih model aktif (use) via pick_single."""
    return run_model_use(
        models, active_id=active_id, kueri=kueri,
        pick_fn=pick_fn, switch_fn=switch_fn, out=out,
    )


def run_model_remove(
    model_id: str = "",
    out: Callable[[str], None] | None = None,
) -> str:
    """Tolak hapus model: bawaan tak bisa dihapus; kustom ikut providernya."""
    tulis: Callable[[str], None] = out or (lambda s: print(s))
    mid = str(model_id or "").strip()
    if not mid:
        msg = _galat(
            "Model bawaan tidak bisa dihapus.",
            "model kustom ikut penyedianya",
            ["/provider list", "/provider remove <id>", "/bantuan"],
        )
        _tulis(tulis, msg)
        return msg
    msg = _galat(
        f"Tidak bisa hapus model '{mid}'.",
        "model bawaan tidak bisa dihapus; model kustom ikut penyedianya",
        ["/provider list", "/provider remove <id>", "/bantuan"],
    )
    _tulis(tulis, msg)
    return msg


# Alias pakai/hapus (Use/Remove)
use_model = run_model_use
pakai_model = run_model_use
edit_model = run_model_edit
ubah_model = run_model_edit
remove_model = run_model_remove
hapus_model = run_model_remove
