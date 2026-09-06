"""Layar TUI penyedia: baca / ubah / hapus + tes koneksi.

Logika disalin dari ``harness/cli.py`` (handle_provider, _provider_add,
_provider_list, _provider_remove, _provider_test_connection,
_provider_fetch_models) — hanya disalin, tidak mengimpor modul berat agar
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
BASE_RINGKAS_LEBAR = 34

# Mirror cli._provider_add (bawaan tak bisa ditimpa/dihapus).
BUILTIN_IDS = frozenset({
    "openai", "anthropic", "gemini", "antigravity", "copilot",
    "ollama", "openrouter", "deepseek", "groq", "mistral",
    "together", "xai", "cerebras", "fireworks", "perplexity",
    "sambanova", "opencode", "combo",
})

__all__ = [
    "HANDOFF",
    "BATAS_BARIS",
    "MAKS_ID",
    "SARAN_MAKS",
    "BUILTIN_IDS",
    "render_provider_list",
    "render_provider_detail",
    "suggest_providers",
    "validate_base_url",
    "test_koneksi",
    "run_provider_read",
    "run_provider_edit",
    "run_provider_remove",
    "read_providers",
    "baca_daftar",
    "read_provider_detail",
    "baca_detail",
    "edit_provider",
    "ubah_provider",
    "remove_provider",
    "hapus_provider",
]


def _prov_id(item: Any) -> str:
    """Ambil ID penyedia dari dict atau teks."""
    if isinstance(item, dict):
        for kunci in ("id", "provider", "pid"):
            try:
                val = item.get(kunci)
            except Exception:
                val = None
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    return str(item or "").strip()


def _prov_name(item: Any) -> str:
    """Nama tampil (default = id)."""
    pid = _prov_id(item)
    if isinstance(item, dict):
        try:
            nama = item.get("name", "")
        except Exception:
            nama = ""
        if isinstance(nama, str) and nama.strip():
            return nama.strip()
    return pid or "—"


def _prov_base(item: Any) -> str:
    """Alamat dasar penuh (dukung baseURL/base_url/api/base)."""
    if isinstance(item, dict):
        for kunci in ("baseURL", "base_url", "baseUrl", "api", "base", "url"):
            try:
                val = item.get(kunci)
            except Exception:
                val = None
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _ringkas(teks: str, lebar: int = 40) -> str:
    """Ringkas via kontrak short_id."""
    return str(short_id(teks, lebar))


def _base_ringkas(item: Any) -> str:
    """BaseURL ringkas (maks 34, kosong → '—')."""
    base = _prov_base(item)
    if not base:
        return "—"
    return _ringkas(base, BASE_RINGKAS_LEBAR)


def _prov_n(item: Any) -> int:
    """Jumlah model: dukung list/dict/int di beberapa kunci."""
    if isinstance(item, dict):
        for kunci in ("models", "n_models", "model_count", "count", "n"):
            try:
                val = item.get(kunci)
            except Exception:
                continue
            if isinstance(val, dict):
                return len(val)
            if isinstance(val, (list, tuple)):
                return len(val)
            if isinstance(val, int) and val >= 0:
                return val
    return 0


def _prov_terhubung(item: Any) -> bool:
    """Terhubung: dukung has_credentials/connected/status."""
    try:
        if isinstance(item, dict):
            if isinstance(item.get("has_credentials"), bool):
                return bool(item.get("has_credentials"))
            if isinstance(item.get("connected"), bool):
                return bool(item.get("connected"))
            st = item.get("status")
            if isinstance(st, str) and st.strip():
                if "●" in st:
                    return True
                if "○" in st:
                    return False
    except Exception:
        pass
    return False


def _status_teks(item: Any) -> str:
    """Status satu baris via kontrak status_baku."""
    return str(status_baku(bool(_prov_terhubung(item))))


def _kepala(judul: str, pos: int, total: int, connected: bool) -> str:
    """Kepala via kontrak header_pos."""
    return str(header_pos(judul, int(pos), int(total), bool(connected)))


def _kaki(multi: bool = False) -> str:
    """Kaki via kontrak footer_petunjuk."""
    return str(footer_petunjuk(bool(multi)))


def _galat(judul: str, sebab: str = "", langkah: Any = None) -> str:
    """Galat via kontrak error_box (saran hanya perintah valid)."""
    if langkah is None:
        langkah = ["/provider list", "/provider add", "/bantuan"]
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
# Baca (Read): daftar + detail
# ---------------------------------------------------------------------------

def render_provider_list(
    providers: Sequence[Any] | None,
    offset: int = 0,
    limit: int = BATAS_BARIS,
) -> str:
    """Daftar penyedia: nama, baseURL ringkas, models N, status + posisi."""
    daftar = list(providers or [])
    total = len(daftar)
    if total == 0:
        return (
            "○ Belum terhubung — belum ada penyedia.\n"
            "Contoh: /provider add"
        )
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
    try:
        ada = any(_prov_terhubung(p) for p in daftar)
    except Exception:
        ada = False
    kepala = _kepala("Penyedia", pos_awal if total else 0, total, ada)
    if total and len(potongan) != total:
        posisi = f"Penyedia {pos_awal}-{pos_akhir} dari {total}"
    else:
        posisi = f"Penyedia {pos_akhir} dari {total}" if total else "—"
    baris = [kepala]
    for nomor, item in enumerate(potongan, start=pos_awal):
        pid = _prov_id(item) or "—"
        nama = _prov_name(item)
        st = _status_teks(item)
        n = _prov_n(item)
        base = _base_ringkas(item)
        if nama and nama != pid:
            baris.append(f"{nomor}. {pid} ({nama}) — {st} — {n} model — {base}")
        else:
            baris.append(f"{nomor}. {pid} — {st} — {n} model — {base}")
    baris.append(posisi)
    baris.append(_kaki(False))
    return "\n".join(baris)


def render_provider_detail(provider: Any) -> str:
    """Detail satu penyedia: nama, baseURL ringkas, models N, status."""
    pid = _prov_id(provider) or "—"
    nama = _prov_name(provider)
    base = _base_ringkas(provider)
    n = _prov_n(provider)
    st = _status_teks(provider)
    if nama and nama != pid:
        nama_baris = f"Nama: {nama} ({pid})"
    else:
        nama_baris = f"Nama: {pid}"
    return (
        f"{nama_baris}\n"
        f"BaseURL: {base}\n"
        f"Models: {n} model\n"
        f"Status: {st}"
    )


def suggest_providers(kueri: Any, providers: Sequence[Any] | None, n: int = SARAN_MAKS) -> List[str]:
    """Saran valid saja: maks n ID penyedia yang ada."""
    try:
        maks = int(n)
    except Exception:
        maks = SARAN_MAKS
    if maks <= 0:
        maks = SARAN_MAKS
    if maks > MAKS_ID:
        maks = MAKS_ID
    ids = [_prov_id(p) for p in (providers or [])]
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
    try:
        rendah = q.lower()
        hits = [i for i in ids if rendah in i.lower()][:maks]
        if hits:
            return hits
    except Exception:
        pass
    return ids[:maks]


# Alias baca (Read)
read_providers = render_provider_list
baca_daftar = render_provider_list
read_provider_detail = render_provider_detail
baca_detail = render_provider_detail


# ---------------------------------------------------------------------------
# Validasi + tes koneksi (mirror cli._provider_add / _test_connection)
# ---------------------------------------------------------------------------

def validate_base_url(base: Any) -> Tuple[bool, str]:
    """Validasi alamat dasar http(s)://host… (mirror cli)."""
    teks = str(base or "").strip().rstrip("/")
    if not teks:
        return False, "Alamat dasar tidak valid. Harus http(s)://host… contoh: https://api.example.com/v1"
    try:
        import urllib.parse as _up

        ok_skim = teks.startswith("http://") or teks.startswith("https://")
        parsed = _up.urlparse(teks) if ok_skim else None
        ok = bool(ok_skim and parsed is not None and parsed.netloc)
    except Exception:
        ok = False
    if not ok:
        return False, "Alamat dasar tidak valid. Harus http(s)://host… contoh: https://api.example.com/v1"
    return True, ""


def _models_url(base: str) -> str:
    """Bangun {base}/v1/models (mirror cli)."""
    b = str(base or "").strip().rstrip("/")
    if b.endswith("/models"):
        return b
    bersih = b[:-3] if b.endswith("/v1") else b
    return bersih.rstrip("/") + "/v1/models"


def test_koneksi(base_url: Any, kunci: str = "", timeout: float = 10.0) -> Tuple[bool, str]:
    """Tes koneksi: respons HTTP apa pun = reachable; hanya network = gagal."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return False, "empty baseURL"
    try:
        t = float(timeout)
    except Exception:
        t = 10.0
    try:
        import urllib.error as _ue
        import urllib.request as _ur

        target = _models_url(base)
        headers: Dict[str, str] = {"Accept": "application/json"}
        key = str(kunci or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            req = _ur.Request(target, headers=headers, method="GET")
            with _ur.urlopen(req, timeout=t):
                return True, ""
        except _ue.HTTPError:
            return True, ""
        except Exception as e1:
            try:
                req2 = _ur.Request(base, headers={"Accept": "*/*"}, method="GET")
                with _ur.urlopen(req2, timeout=t):
                    return True, ""
            except _ue.HTTPError:
                return True, ""
            except Exception as e2:
                try:
                    msg = str(e2 or e1 or "unreachable")
                except Exception:
                    msg = "unreachable"
                return False, msg
    except Exception as e:
        try:
            return False, str(e or "unreachable")
        except Exception:
            return False, "unreachable"


# ---------------------------------------------------------------------------
# Interaktif tipis (run_*)
# ---------------------------------------------------------------------------

def run_provider_read(
    providers: Sequence[Any] | None,
    offset: int = 0,
    out: Callable[[str], None] | None = None,
) -> str:
    """Cetak + kembalikan daftar penyedia (tipis)."""
    teks = render_provider_list(providers, offset=offset)
    _tulis(out or print, teks)
    return teks


def run_provider_edit(
    provider: Any,
    input_fn: Callable[[str], str] | None = None,
    test_fn: Callable[..., Tuple[bool, str]] | None = None,
    save_fn: Callable[[str, Dict[str, Any]], None] | None = None,
    out: Callable[[str], None] | None = None,
    api_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Ubah nama/baseURL + tes koneksi; gagal → batal (tak disimpan).

    Return dict baru {'id','name','baseURL',...} atau None bila batal.
    Kosong = pertahankan nilai lama; tanpa perubahan → kembalikan asli.
    """
    tulis: Callable[[str], None] = out or (lambda s: print(s))
    if not isinstance(provider, dict) or not _prov_id(provider):
        msg = _galat(
            "Penyedia tidak dikenal.",
            "lihat /provider list",
            ["/provider list", "/bantuan"],
        )
        _tulis(tulis, msg)
        return None
    pid = _prov_id(provider)
    nama_lama = _prov_name(provider)
    if nama_lama == "—":
        nama_lama = pid
    base_lama = _prov_base(provider)
    tanya: Callable[[str], str] = input_fn or (lambda p: input(p))

    # 1) BaseURL baru (kosong = pertahankan).
    try:
        raw_base = tanya(
            f"Alamat dasar baru [{base_lama or '—'}], contoh: https://api.example.com/v1 (kosong=pertahankan): "
        )
    except (EOFError, KeyboardInterrupt):
        _tulis(tulis, "Dibatalkan.")
        return None
    except Exception:
        _tulis(tulis, "Dibatalkan.")
        return None
    teks_base = str(raw_base or "").strip().rstrip("/")
    base_baru = base_lama if not teks_base else teks_base

    # 2) Nama baru (kosong = pertahankan).
    try:
        raw_nama = tanya(f"Nama baru [{nama_lama}] (kosong=pertahankan): ")
    except (EOFError, KeyboardInterrupt):
        _tulis(tulis, "Dibatalkan.")
        return None
    except Exception:
        _tulis(tulis, "Dibatalkan.")
        return None
    teks_nama = str(raw_nama or "").strip()
    nama_baru = nama_lama if not teks_nama else teks_nama

    if base_baru == base_lama and nama_baru == nama_lama:
        _tulis(tulis, "Tidak ada perubahan. Dibatalkan.")
        return dict(provider)

    # 3) Validasi URL penuh (mirror _provider_add).
    ok_url, pesan_url = validate_base_url(base_baru)
    if not ok_url:
        _tulis(tulis, _galat(pesan_url, "periksa alamat dasar", ["/provider list", "/bantuan"]))
        return None

    # 4) Tes koneksi; gagal → batal, jangan simpan.
    uji = test_fn or test_koneksi
    try:
        try:
            ok_conn, cerr = uji(base_baru, api_key)
        except TypeError:
            ok_conn, cerr = uji(base_baru)
    except Exception as e:
        ok_conn, cerr = False, str(e or "tes sambungan gagal")
    if not ok_conn:
        _tulis(
            tulis,
            _galat(
                f"Tes sambungan gagal: {cerr or 'unreachable'}",
                "periksa alamat dasar/jaringan. Dibatalkan, tidak disimpan",
                ["/provider list", "/bantuan"],
            ),
        )
        return None

    baru: Dict[str, Any] = dict(provider)
    baru["id"] = pid
    baru["name"] = nama_baru
    baru["baseURL"] = base_baru
    # Selaraskan kunci umum lain bila ada.
    if "base_url" in baru:
        baru["base_url"] = base_baru
    if "api" in baru and isinstance(baru.get("api"), str):
        baru["api"] = base_baru
    if callable(save_fn):
        try:
            save_fn(pid, baru)
        except Exception as e:
            _tulis(tulis, _galat(f"Gagal menyimpan penyedia: {e}", "", ["/provider list", "/bantuan"]))
            return None
    _tulis(tulis, f"✅ Penyedia '{pid}' diperbarui.")
    return baru


def _cari_kanonik(target: str, providers: Sequence[Any] | None) -> Optional[str]:
    """Cari ID kanonik (persis dulu, lalu case-insensitive)."""
    want = str(target or "").strip()
    if not want:
        return None
    ids = [_prov_id(p) for p in (providers or []) if _prov_id(p)]
    if want in ids:
        return want
    rendah = want.lower()
    for pid in ids:
        if pid.strip().lower() == rendah:
            return pid
    return None


def run_provider_remove(
    target: Any = "",
    providers: Sequence[Any] | None = None,
    is_custom_fn: Callable[[str], bool] | None = None,
    input_fn: Callable[[str], str] | None = None,
    delete_fn: Callable[[str], None] | None = None,
    out: Callable[[str], None] | None = None,
    pick_fn: Callable[..., Optional[int]] | None = None,
) -> Tuple[bool, str]:
    """Hapus khusus kustom + konfirmasi; tolak bawaan/unknown.

    Return (ok, pesan). Tanpa providers: nilai custom via BUILTIN_IDS.
    """
    tulis: Callable[[str], None] = out or (lambda s: print(s))
    want = str(target or "").strip()

    # Target kosong → picker bila daftar ada, else tanya.
    if not want and providers:
        pilih = pick_fn or pick_single
        ids = [_prov_id(p) for p in providers if _prov_id(p)]
        if not ids:
            msg = "Dibatalkan."
            _tulis(tulis, msg)
            return False, msg
        try:
            idx = pilih("Penyedia — hapus", list(providers), show=_prov_id)
        except Exception:
            idx = None
        if idx is None:
            _tulis(tulis, "Dibatalkan.")
            return False, "Dibatalkan."
        try:
            want = _prov_id(list(providers)[int(idx)])
        except Exception:
            _tulis(tulis, "Dibatalkan.")
            return False, "Dibatalkan."
    if not want:
        tanya0: Callable[[str], str] = input_fn or (lambda p: input(p))
        try:
            want = str(tanya0("ID penyedia yang dihapus, contoh: my-provider: ") or "").strip()
        except (EOFError, KeyboardInterrupt):
            _tulis(tulis, "Dibatalkan.")
            return False, "Dibatalkan."
        except Exception:
            _tulis(tulis, "Dibatalkan.")
            return False, "Dibatalkan."
    if not want:
        _tulis(tulis, "Dibatalkan.")
        return False, "Dibatalkan."

    kanon = _cari_kanonik(want, providers) if providers else None
    cek = kanon or want

    # Tentukan kustom (mirror _provider_remove: hanya kustom bisa dihapus).
    is_custom: Optional[bool] = None
    if callable(is_custom_fn):
        try:
            is_custom = bool(is_custom_fn(cek))
        except Exception:
            is_custom = None
    if is_custom is None and providers:
        try:
            for p in providers:
                if _prov_id(p) == cek or _prov_id(p).lower() == cek.lower():
                    if isinstance(p, dict):
                        if isinstance(p.get("is_custom"), bool):
                            is_custom = bool(p.get("is_custom"))
                            break
                        if isinstance(p.get("custom"), bool):
                            is_custom = bool(p.get("custom"))
                            break
                    break
        except Exception:
            pass
    if is_custom is None:
        try:
            rendah = cek.strip().lower()
            builtin_rendah = {b.lower() for b in BUILTIN_IDS}
            if rendah in builtin_rendah:
                is_custom = False
            elif providers is not None:
                dikenal = any(
                    _prov_id(p).lower() == rendah for p in (providers or []) if _prov_id(p)
                )
                # Dikenal tapi bukan bawaan → anggap kustom.
                is_custom = bool(dikenal)
            else:
                is_custom = False
        except Exception:
            is_custom = False

    if not is_custom:
        if providers is not None:
            try:
                dikenal = any(
                    _prov_id(p).strip().lower() == cek.strip().lower()
                    for p in (providers or []) if _prov_id(p)
                )
            except Exception:
                dikenal = False
            if not dikenal:
                msg = _galat(
                    f"Penyedia '{want}' tidak ditemukan atau bukan kustom.",
                    "lihat /provider list",
                    ["/provider list", "/bantuan"],
                )
                _tulis(tulis, msg)
                return False, msg
        msg = _galat(
            f"Tidak bisa hapus bawaan '{cek}'.",
            "hanya penyedia kustom yang bisa dihapus",
            ["/provider list", "/bantuan"],
        )
        _tulis(tulis, msg)
        return False, msg

    # Konfirmasi y (mirror cli).
    tanya: Callable[[str], str] = input_fn or (lambda p: input(p))
    try:
        jawab = tanya(f"Hapus penyedia kustom '{cek}'? [y/N] Contoh y: ")
    except (EOFError, KeyboardInterrupt):
        _tulis(tulis, "Dibatalkan.")
        return False, "Dibatalkan."
    except Exception:
        _tulis(tulis, "Dibatalkan.")
        return False, "Dibatalkan."
    if str(jawab or "").strip().lower() != "y":
        _tulis(tulis, "Dibatalkan.")
        return False, "Dibatalkan."
    if callable(delete_fn):
        try:
            delete_fn(cek)
        except Exception as e:
            msg = _galat(f"Gagal hapus penyedia: {e}", "", ["/provider list", "/bantuan"])
            _tulis(tulis, msg)
            return False, msg
    msg_ok = f"✅ Penyedia '{cek}' dihapus."
    _tulis(tulis, msg_ok)
    return True, msg_ok


# Alias ubah/hapus (Edit/Remove)
edit_provider = run_provider_edit
ubah_provider = run_provider_edit
remove_provider = run_provider_remove
hapus_provider = run_provider_remove
