"""Layar TUI gabungan (combo): baca / buat / ubah / hapus / pakai + footer.

Logika disalin dari ``harness/cli.py`` (handle_combo, _combo_use,
_combo_edit, _combo_extract_serving, _combo_footer) dan
``harness/models/combo.py`` (ComboManager: validasi nama/strategi/models,
tolak combo bersarang) — hanya disalin, tidak mengimpor modul berat agar
tetap stdlib dan tidak menyentuh berkas lain.

Single source: langsung dari harness.tui.components/pickers (import gagal
= error nyata, jangan sembunyikan).
Bahasa: Indonesia. Stdlib saja.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Single source: langsung dari komponen/picker asli (tanpa fallback lokal)
# ---------------------------------------------------------------------------

HANDOFF = (
    "Single source: harness/tui/components.py "
    "(header_pos, model_lines, error_box, footer_petunjuk) dan "
    "harness/tui/pickers.py (pick_single, pick_multi) dipakai langsung; "
    "import gagal = error nyata."
)

from harness.tui.components import (
    header_pos,
    model_lines,
    error_box,
    footer_petunjuk,
)
from harness.tui.pickers import pick_single, pick_multi


# ---------------------------------------------------------------------------
# Konstanta: 12 strategi + deskripsi 1 baris + aturan tampil
# ---------------------------------------------------------------------------

MAKS_ID = 5

STRATEGI_DESKRIPSI: Dict[str, str] = {
    "round_robin": "Giliran bergantian tiap permintaan.",
    "random": "Pilih acak satu model tiap permintaan.",
    "fastest": "Fan-out semua, pakai jawaban tercepat.",
    "cascade": "Coba berurutan sampai berhasil.",
    "consensus": "Fan-out semua, pilih jawaban terbaik.",
    "cost_optimizer": "Coba dari yang termurah dulu.",
    "weighted_random": "Acak berbobot sesuai params weights.",
    "ab_split": "Bagi acak 50/50 dua model pertama.",
    "pipeline": "Draf model-1 lalu disempurnakan model-2.",
    "quality_tier": "Utama model-1, cadangan model-2.",
    "load_balancer": "Pilih rerata latensi tercepat.",
    "fallback_chain": "Rantai cadangan berurutan bila gagal.",
}

STRATEGI_LIST: List[str] = [
    "round_robin",
    "random",
    "fastest",
    "cascade",
    "consensus",
    "cost_optimizer",
    "weighted_random",
    "ab_split",
    "pipeline",
    "quality_tier",
    "load_balancer",
    "fallback_chain",
]

STRATEGI_FANOUT = {"fastest", "consensus"}

_NAMA_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Bantuan murni (tanpa I/O): validasi + potong + saran
# ---------------------------------------------------------------------------

def _strip_prefix_combo(nama: Any) -> str:
    """Buang awalan 'combo/' bila ada (logika cli.py)."""
    teks = str(nama or "").strip()
    if teks.lower().startswith("combo/"):
        teks = teks.split("/", 1)[1].strip()
    return teks


def validasi_nama(nama: Any) -> str:
    """Mirror ComboManager._validate_name; raise ValueError bila tak valid."""
    if not isinstance(nama, str):
        raise ValueError(f"Nama gabungan {nama!r} harus berupa teks. Contoh: /combo create")
    cname = _strip_prefix_combo(nama)
    if not cname:
        raise ValueError("Nama gabungan wajib diisi. Contoh: /combo create")
    if "/" in cname or "\\" in cname or len(cname.split()) != 1:
        raise ValueError(
            f"Nama gabungan {cname!r} tak boleh berisi '/' atau spasi. Contoh: andalan"
        )
    if not _NAMA_RE.match(cname):
        raise ValueError(
            f"Nama gabungan {cname!r} memakai 1-64 karakter [A-Za-z0-9_-]. Contoh: andalan"
        )
    return cname


def validasi_strategi(strategi: Any) -> str:
    """Mirror ComboManager._validate_strategy; raise ValueError bila tak valid."""
    if not isinstance(strategi, str) or not strategi.strip():
        raise ValueError("Strategi wajib diisi. Contoh: fastest")
    s = strategi.strip()
    if s in STRATEGI_DESKRIPSI:
        return s
    raise ValueError(
        f"Strategi {s!r} tidak dikenal; pilih satu dari: "
        f"{', '.join(STRATEGI_LIST)}. Contoh: fastest"
    )


def validasi_models(models: Any) -> List[str]:
    """Mirror ComboManager._validate_models + tolak combo bersarang."""
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("Model gabungan wajib minimal 1. Contoh: /model")
    bersih: List[str] = []
    for m in models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"Model {m!r} harus teks tak kosong. Contoh: /model")
        mid = m.strip()
        if mid.lower().startswith("combo/"):
            raise ValueError(
                f"Model '{mid}' tak boleh combo bersarang (combo/...). "
                "Pakai ID prov/model biasa. Contoh: /model"
            )
        bersih.append(mid)
    return bersih


def _ambil_id(item: Any) -> str:
    """Ambil ID dari dict {'id': ...} atau teks biasa."""
    if isinstance(item, dict):
        return str(item.get("id", "") or "").strip()
    return str(item or "").strip()


def _potong_ids(ids: Sequence[str], maks: int = MAKS_ID) -> str:
    """Gabung ID, MAKS `maks` lalu '+N lagi' (aturan ≤5 ID)."""
    bersih = [str(i).strip() for i in (ids or []) if str(i or "").strip()]
    if not bersih:
        return "—"
    tampil = bersih[:maks]
    teks = ", ".join(tampil)
    if len(bersih) > maks:
        teks += f" +{len(bersih) - maks} lagi"
    return teks


def saran_use(nama: str) -> str:
    """Saran valid untuk memakai gabungan."""
    return f"/combo use {nama}"


def saran_model(nama: str) -> str:
    """Saran valid untuk mengaktifkan via /model."""
    return f"/model combo/{nama}"


def _cari_kanonik(nama: str, combos: Dict[str, Any]) -> Optional[str]:
    """Cari kunci kanonik (persis dulu, lalu case-insensitive)."""
    if nama in (combos or {}):
        return nama
    rendah = nama.strip().lower()
    for kunci in list((combos or {}).keys()):
        if isinstance(kunci, str) and kunci.strip().lower() == rendah:
            return kunci
    return None


# ---------------------------------------------------------------------------
# Baca: daftar + detail
# ---------------------------------------------------------------------------

def daftar_baris(nama: str, combo: Dict[str, Any], aktif: bool = False) -> str:
    """Satu baris daftar: 'nama — strategi — N model — ★ aktif'.

    Penanda bintang hanya bila aktif; tanpa newline.
    """
    try:
        strategi = str((combo or {}).get("strategi", "?") or "?")
    except Exception:
        strategi = "?"
    try:
        models = list((combo or {}).get("models", []) or [])
    except Exception:
        models = []
    tanda = " — ★ aktif" if aktif else ""
    return f"{nama} — {strategi} — {len(models)} model{tanda}"


def render_daftar(
    combos: Dict[str, Any],
    aktif: Optional[str] = None,
    posisi: str = "",
) -> str:
    """Render daftar gabungan; kosong → pesan + saran buat baru."""
    combos = combos or {}
    aktif_nama = _strip_prefix_combo(aktif or "")
    if not combos:
        return "Belum ada gabungan tersimpan. Contoh: /combo create"
    total = len(combos)
    base = header_pos("Gabungan", total, total, False)
    pos = str(posisi or "").strip()
    kepala = f"{base} — {pos}" if pos else str(base).strip()
    baris = [str(kepala).strip()]
    for nama in sorted(combos.keys()):
        try:
            is_aktif = bool(aktif_nama and nama == aktif_nama)
        except Exception:
            is_aktif = False
        try:
            baris.append(daftar_baris(nama, combos.get(nama) or {}, aktif=is_aktif))
        except Exception:
            baris.append(f"{nama} — ? — 0 model")
    baris.append(footer_petunjuk(True))
    return "\n".join(baris)


def render_detail(nama: str, combo: Dict[str, Any], posisi: str = "") -> str:
    """Render detail: strategi + model_lines Compact + posisi.

    Model dipadatkan via model_lines (MAKS 5 ID).
    """
    cname = _strip_prefix_combo(nama)
    combo = combo or {}
    try:
        strategi = str(combo.get("strategi", "?") or "?")
    except Exception:
        strategi = "?"
    try:
        models = list(combo.get("models", []) or [])
    except Exception:
        models = []
    model_teks = model_lines(models, MAKS_ID)
    # Pastikan tetap ≤5 ID walau komponen luar longgar.
    if model_teks.count(",") >= MAKS_ID:
        model_teks = _potong_ids([_ambil_id(m) for m in models])  # type: ignore
    base = header_pos(f"Gabungan {cname}", 1, 1, False)
    pos = str(posisi or "").strip()
    kepala = f"{base} — {pos}" if pos else str(base).strip()
    return f"{kepala}\nStrategi: {strategi}\nModel: {model_teks}"


def compact_dukung() -> bool:
    """Selalu False: model_lines asli tak punya argumen compact (single source)."""
    return False


# Alias baca (Read)
read_combos = render_daftar
baca_daftar = render_daftar
read_combo_detail = render_detail
baca_detail = render_detail


# ---------------------------------------------------------------------------
# Footer: SATU baris Compact
# ---------------------------------------------------------------------------

def footer_combo(
    nama: Any,
    serving_prov: Any = None,
    serving_model: Any = None,
    strategi: Any = "",
    dicoba: Any = None,
) -> str:
    """Footer SATU baris: 'Dijawab gabungan X memakai prov/model'.

    Bila strategi fan-out (fastest/consensus) dan `dicoba` diketahui,
    tambah ' (dicoba N)'. JANGAN daftar semua member — hanya penyaji.
    Tanpa newline; serving kosong → 'memakai —'.
    """
    cname = _strip_prefix_combo(nama) or "—"
    prov = str(serving_prov or "").strip()
    smodel = str(serving_model or "").strip()
    strat = str(strategi or "").strip()
    if prov and smodel:
        pakai = f"{prov}/{smodel}"
    elif smodel:
        pakai = smodel
    else:
        pakai = "—"
    teks = f"Dijawab gabungan {cname} memakai {pakai}"
    try:
        n = int(dicoba) if dicoba is not None else None
    except (TypeError, ValueError):
        n = None
    if strat in STRATEGI_FANOUT and n is not None and n > 0:
        teks += f" (dicoba {n})"
    return teks.replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Buat (Create): nama → strategi picker 12 → multi models
# ---------------------------------------------------------------------------

def opsi_strategi() -> List[Tuple[str, str]]:
    """12 opsi (nilai, 'nilai — deskripsi 1 baris')."""
    return [(s, f"{s} — {STRATEGI_DESKRIPSI[s]}") for s in STRATEGI_LIST]


def flow_create(
    connected: Sequence[Any],
    tanya_nama: Optional[Callable[[str], str]] = None,
    pilih_strategi: Optional[Callable[..., Optional[int]]] = None,
    pilih_models: Optional[Callable[..., Optional[List[int]]]] = None,
    tampil: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Alur buat: nama → strategi (12 + deskripsi) → multi model.

    Return dict {'name','strategy','models'} atau None bila batal.
    Galat validasi dikembalikan sebagai pesan via `tampil`, lalu None
    (pemanggil boleh mengulang). Saran selalu valid (/combo use, /model).
    """
    keluar: Callable[[str], None] = tampil or (lambda s: print(s))
    connected = list(connected or [])
    if not connected:
        keluar(error_box("Belum terhubung — belum ada model.", "", ["ketik /provider"]))
        return None

    # 1) Nama.
    minta: Callable[[str], str] = tanya_nama or (lambda p: input(p))
    try:
        mentah = minta("Nama gabungan (kosong=kembali), contoh: andalan: ")
    except (EOFError, KeyboardInterrupt):
        keluar("Dibatalkan.")
        return None
    if mentah is None or not str(mentah).strip() or str(mentah).strip().lower() in ("back", "q"):
        keluar("Dibatalkan.")
        return None
    try:
        cname = validasi_nama(mentah)
    except ValueError as e:
        keluar(error_box(str(e), "", ["contoh: andalan"]))
        return None

    # 2) Strategi: 12 semua + deskripsi 1 baris.
    opsi = opsi_strategi()
    nilai_opsi = [v for v, _label in opsi]
    label_opsi = [label for _v, label in opsi]
    pick1: Callable[..., Optional[int]] = pilih_strategi or pick_single
    try:
        idx = pick1("Gabungan — strategi", label_opsi, show=lambda x: x)
    except Exception:
        idx = None
    if idx is None or not (0 <= int(idx) < len(nilai_opsi)):
        keluar("Dibatalkan.")
        return None
    strategi = nilai_opsi[int(idx)]

    # 3) Multi model (min 1, tolak combo bersarang).
    ids = [_ambil_id(c) for c in connected]
    pickm: Callable[..., Optional[List[int]]] = pilih_models or pick_multi
    try:
        picks = pickm("Gabungan — model (Spasi pilih, Enter selesai)", connected, show=_ambil_id)
    except Exception:
        picks = None
    if not picks:
        keluar("Dibatalkan.")
        return None
    try:
        models = validasi_models([ids[i] for i in picks if 0 <= int(i) < len(ids)])
    except ValueError as e:
        keluar(error_box(str(e), "", ["contoh: /model"]))
        return None
    return {"name": cname, "strategy": strategi, "models": models}


# Alias buat (Create)
create_combo = flow_create
buat_combo = flow_create


# ---------------------------------------------------------------------------
# Ubah (Edit): pre-tandai + sorot strategi + diff MAKS 5/baris kelompok
# ---------------------------------------------------------------------------

def _indeks_awal(connected_ids: List[str], cur_models: List[str]) -> List[int]:
    """Indeks pre-tandai model existing di daftar connected."""
    try:
        him = set(str(m) for m in (cur_models or []))
    except Exception:
        him = set()
    return [i for i, cid in enumerate(connected_ids) if cid in him]


def diff_ringkas(
    lama_models: Sequence[str],
    baru_models: Sequence[str],
    lama_strategi: str = "",
    baru_strategi: str = "",
) -> List[str]:
    """Diff ringkas, MAKS 5 baris per kelompok (tambah/buang).

    Kelompok: Strategi (1 baris), Ditambah (≤5), Dibuang (≤5).
    Lebihnya dipadatkan '+N lagi'.
    """
    try:
        lama = [str(m).strip() for m in (lama_models or []) if str(m or "").strip()]
    except Exception:
        lama = []
    try:
        baru = [str(m).strip() for m in (baru_models or []) if str(m or "").strip()]
    except Exception:
        baru = []
    try:
        lama_set, baru_set = set(lama), set(baru)
    except Exception:
        lama_set, baru_set = set(), set()
    tambah = [m for m in baru if m not in lama_set]
    buang = [m for m in lama if m not in baru_set]

    baris: List[str] = []
    ls, bs = str(lama_strategi or "—"), str(baru_strategi or "—")
    baris.append(f"Strategi: {ls} → {bs}" if ls != bs else f"Strategi: {bs} (tetap)")
    baris.append(f"Ditambah ({len(tambah)}): {_potong_ids(tambah)}")
    # Rincian tambah MAKS 5 baris.
    for m in tambah[:MAKS_ID]:
        baris.append(f"  + {m}")
    if len(tambah) > MAKS_ID:
        baris.append(f"  +{len(tambah) - MAKS_ID} lagi")
    baris.append(f"Dibuang ({len(buang)}): {_potong_ids(buang)}")
    for m in buang[:MAKS_ID]:
        baris.append(f"  - {m}")
    if len(buang) > MAKS_ID:
        baris.append(f"  -{len(buang) - MAKS_ID} lagi")
    return baris


def flow_edit(
    nama: str,
    saat_ini: Dict[str, Any],
    connected: Sequence[Any],
    pilih_strategi: Optional[Callable[..., Optional[int]]] = None,
    pilih_models: Optional[Callable[..., Optional[List[int]]]] = None,
    konfirmasi: Optional[Callable[[str], str]] = None,
    tampil: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Alur ubah: strategi tersorot + pre-tandai + diff + konfirmasi.

    Return dict baru {'name','strategy','models'} atau None
    (batal / tak valid). Nama tetap (kanonik dari `nama`).
    """
    keluar: Callable[[str], None] = tampil or (lambda s: print(s))
    try:
        cname = validasi_nama(nama)
    except ValueError as e:
        keluar(error_box(str(e), "", ["lihat /combo list"]))
        return None
    if not isinstance(saat_ini, dict):
        keluar(error_box(f"Gabungan '{cname}' tidak ditemukan.", "", ["lihat /combo list"]))
        return None
    try:
        cur_strat = str(saat_ini.get("strategy", "") or "")
    except Exception:
        cur_strat = ""
    try:
        cur_models = list(saat_ini.get("models", []) or [])
    except Exception:
        cur_models = []

    opsi = opsi_strategi()
    label_opsi = [label for _v, label in opsi]
    pick1: Callable[..., Optional[int]] = pilih_strategi or pick_single
    try:
        s_idx = pick1(
            "Gabungan — strategi baru",
            label_opsi,
            show=lambda x: x,
        )
    except Exception:
        s_idx = None
    if s_idx is None:
        keluar("Dibatalkan.")
        return None
    try:
        strategi_baru = STRATEGI_LIST[int(s_idx)]
    except (ValueError, IndexError, TypeError):
        keluar("Dibatalkan.")
        return None

    # Model pre-tandai existing.
    connected = list(connected or [])
    if not connected:
        keluar("Belum terhubung — belum ada model. Contoh: /provider")
        return None
    ids = [_ambil_id(c) for c in connected]
    pre = _indeks_awal(ids, [str(m) for m in cur_models])
    pickm: Callable[..., Optional[List[int]]] = pilih_models or pick_multi
    try:
        picks = pickm(
            f"Gabungan — model (saat ini: {_potong_ids([str(m) for m in cur_models])})",
            connected,
            show=_ambil_id,
            initial_selected=pre,
        )
    except Exception:
        picks = None
    if picks is None or not picks:
        keluar("Dibatalkan.")
        return None
    try:
        models_baru = validasi_models([ids[int(i)] for i in picks if 0 <= int(i) < len(ids)])
    except ValueError as e:
        keluar(error_box(str(e), "", ["contoh: /model"]))
        return None

    for baris in diff_ringkas(cur_models, models_baru, cur_strat, strategi_baru):
        keluar(baris)
    tanya: Callable[[str], str] = konfirmasi or (lambda p: input(p))
    try:
        jawab = tanya(f"Simpan perubahan gabungan '{cname}'? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        keluar("Dibatalkan.")
        return None
    if str(jawab or "").strip().lower() != "y":
        keluar("Dibatalkan.")
        return None
    return {"name": cname, "strategy": strategi_baru, "models": models_baru}


# Alias ubah (Edit)
edit_combo = flow_edit
ubah_combo = flow_edit


# ---------------------------------------------------------------------------
# Hapus (Remove): konfirmasi, tolak unknown
# ---------------------------------------------------------------------------

def flow_remove(
    nama: str,
    combos: Dict[str, Any],
    konfirmasi: Optional[Callable[[str], str]] = None,
    tampil: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Hapus gabungan; tolak unknown + minta konfirmasi y.

    Return (ok, pesan). Pesan galat selalu bawa saran '/combo list'.
    """
    keluar: Callable[[str], None] = tampil or (lambda s: print(s))
    combos = combos or {}
    try:
        cname = validasi_nama(nama)
    except ValueError as e:
        msg = str(e)
        keluar(msg)
        return False, msg
    kanon = _cari_kanonik(cname, combos)
    if kanon is None:
        msg = error_box(f"Gabungan '{cname}' tidak ditemukan.", "", ["lihat /combo list"])
        keluar(msg)
        return False, msg
    tanya: Callable[[str], str] = konfirmasi or (lambda p: input(p))
    try:
        jawab = tanya(f"Hapus gabungan '{kanon}'? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        keluar("Dibatalkan.")
        return False, "Dibatalkan."
    if str(jawab or "").strip().lower() != "y":
        keluar("Dibatalkan.")
        return False, "Dibatalkan."
    return True, f"Gabungan '{kanon}' dihapus."


# Alias hapus (Remove)
remove_combo = flow_remove
hapus_combo = flow_remove


# ---------------------------------------------------------------------------
# Pakai (Use): aktifkan
# ---------------------------------------------------------------------------

def flow_use(
    nama: str,
    combos: Dict[str, Any],
    pilih: Optional[Callable[..., Optional[int]]] = None,
    tampil: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], str]:
    """Aktifkan gabungan; return ('combo/<nama>', pesan) atau (None, pesan).

    Nama kosong → picker bila `pilih` tersedia; unknown ditolak dengan
    saran valid '/combo list'. Sukses bawa saran '/model combo/<nama>'.
    """
    keluar: Callable[[str], None] = tampil or (lambda s: print(s))
    combos = combos or {}
    cname = _strip_prefix_combo(nama or "")
    if not cname:
        if not combos:
            msg = "Belum ada gabungan tersimpan. Contoh: /combo create"
            keluar(msg)
            return None, msg
        if pilih is None:
            msg = "Nama gabungan wajib diisi. Contoh: /combo use andalan"
            keluar(msg)
            return None, msg
        items = sorted(combos.keys())
        try:
            idx = pilih("Gabungan — pakai", items, show=lambda n: n)
        except Exception:
            idx = None
        if idx is None:
            keluar("Dibatalkan.")
            return None, "Dibatalkan."
        try:
            cname = items[int(idx)]
        except (ValueError, IndexError, TypeError):
            keluar("Dibatalkan.")
            return None, "Dibatalkan."
    kanon = _cari_kanonik(cname, combos)
    if kanon is None:
        msg = error_box(f"Gabungan '{cname}' tidak ditemukan.", "", ["lihat /combo list"])
        keluar(msg)
        return None, msg
    target = f"combo/{kanon}"
    keluar(f"Gabungan '{kanon}' aktif. Contoh: {saran_model(kanon)}")
    return target, f"Gabungan '{kanon}' aktif. Contoh: {saran_model(kanon)}"


# Alias pakai (Use)
use_combo = flow_use
pakai_combo = flow_use


__all__ = [
    "HANDOFF",
    "MAKS_ID",
    "STRATEGI_DESKRIPSI",
    "STRATEGI_LIST",
    "STRATEGI_FANOUT",
    "footer_combo",
    "daftar_baris",
    "render_daftar",
    "render_detail",
    "compact_dukung",
    "opsi_strategi",
    "diff_ringkas",
    "validasi_nama",
    "validasi_strategi",
    "validasi_models",
    "saran_use",
    "saran_model",
    "flow_create",
    "flow_edit",
    "flow_remove",
    "flow_use",
    "read_combos",
    "baca_daftar",
    "read_combo_detail",
    "baca_detail",
    "create_combo",
    "buat_combo",
    "edit_combo",
    "ubah_combo",
    "remove_combo",
    "hapus_combo",
    "use_combo",
    "pakai_combo",
]
