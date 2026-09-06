"""Test TUI baru — deterministik, fungsi murni saja (mock input via keys).

Cakupan: status_baku/header_pos, short_id tengah, model_lines maks5+overflow,
error_box Judul/Sebab/Langkah tanpa redundansi, footer_combo satu baris tanpa
daftar member, key parser (panah/jk/Enter/Esc/Spasi), render daftar
model/provider/combo Compact (<=8 baris, posisi ada), saran invalid tak pernah
muncul. Tanpa interaksi TTY.
"""
import re
import unittest

from harness.tui.components import (
    error_box,
    footer_petunjuk,
    header_pos,
    model_lines,
    short_id,
    status_baku,
)
from harness.tui.pickers import (
    filter_items,
    move_index,
    parse_key,
    parse_key_sequence,
    pick_multi,
    pick_single,
    render_lines,
)
from harness.tui.screens_combo import (
    daftar_baris,
    footer_combo,
    render_daftar,
    render_detail,
)
from harness.tui.screens_model import render_model_detail, render_model_list, suggest_models
from harness.tui.screens_provider import render_provider_list, suggest_providers


class TestStatusHeader(unittest.TestCase):
    def test_status_baku_nilai_baku(self):
        self.assertEqual(status_baku(True), "● Terhubung")
        self.assertEqual(status_baku(False), "○ Belum terhubung")
        self.assertNotIn("\n", status_baku(True))
        self.assertNotIn("\n", status_baku(False))

    def test_header_pos_satu_baris_dan_isi(self):
        h = header_pos("Model", 2, 5, True)
        self.assertNotIn("\n", h)
        self.assertIn("Model", h)
        self.assertIn("2/5", h)
        self.assertIn("● Terhubung", h)
        self.assertIn("▶", h)
        h2 = header_pos("Penyedia", 1, 3, False)
        self.assertNotIn("\n", h2)
        self.assertIn("○ Belum terhubung", h2)
        self.assertIn("1/3", h2)

    def test_header_pos_clamp_dan_kosong(self):
        self.assertIn("5/5", header_pos("Pilih", 99, 5, False))
        self.assertIn("1/3", header_pos("Pilih", 0, 3, True))
        self.assertIn("0/0", header_pos("Pilih", 5, 0, False))
        self.assertIn("Pilih", header_pos("", 1, 2, True))
        self.assertNotIn("\n", header_pos("A", 1, 2, True))


class TestShortId(unittest.TestCase):
    def test_short_id_tengah_memotong(self):
        s = "A" * 20 + "B" * 20 + "C" * 20
        r = short_id(s, 40)
        self.assertEqual(len(r), 40)
        self.assertIn("…", r)
        self.assertTrue(r.startswith(s[:20]))
        self.assertTrue(r.endswith(s[-19:]))
        self.assertNotIn("\n", r)
        # newline disanitasi
        self.assertNotIn("\n", short_id("a\nb", 40))
        self.assertNotIn("\r", short_id("a\rb", 40))

    def test_short_id_pendek_dan_batas(self):
        self.assertEqual(short_id("pendek", 40), "pendek")
        self.assertEqual(short_id("abc", 0), "")
        self.assertEqual(short_id("abcdef", 3), "abc")
        self.assertEqual(len(short_id("x" * 100, 10)), 10)
        self.assertEqual(short_id("", 40), "")


class TestModelLines(unittest.TestCase):
    def test_model_lines_maks5_overflow(self):
        models = ["m%d" % i for i in range(7)]
        lines = model_lines(models, 5)
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], "1. m0")
        self.assertEqual(lines[4], "5. m4")
        self.assertEqual(lines[5], "… +2 lainnya")
        # tepat 5 tanpa overflow
        self.assertEqual(len(model_lines(["a", "b", "c", "d", "e"], 5)), 5)
        # 6 -> +1 lainnya
        lines6 = model_lines(["a", "b", "c", "d", "e", "f"], 5)
        self.assertEqual(lines6[-1], "… +1 lainnya")
        # tiap baris model hanya satu nomor
        for ln in lines[:5]:
            self.assertRegex(ln, r"^\d+\. ")

    def test_model_lines_kosong(self):
        self.assertEqual(model_lines([]), ["(belum ada model)"])
        self.assertEqual(model_lines(None), ["(belum ada model)"])


class TestErrorBox(unittest.TestCase):
    def test_error_box_struktur_tanpa_redundansi(self):
        e = error_box("Judul X", "Sebab Y", ["Langkah A", "Judul X", "Sebab Y", "Langkah A", "Langkah B", "  "])
        self.assertIn("✗ Judul X", e)
        self.assertIn("Sebab: Sebab Y", e)
        self.assertIn("Langkah:", e)
        self.assertEqual(e.count("Langkah A"), 1)
        self.assertEqual(e.count("Langkah B"), 1)
        # judul/sebab tidak diulang sebagai langkah
        langkah_bagian = e.split("Langkah:")[-1]
        self.assertNotIn("Judul X", langkah_bagian)
        self.assertNotIn("Sebab Y", langkah_bagian)
        # judul==sebab -> Sebab dihilangkan, tanpa redundansi
        e2 = error_box("Sama", "Sama", ["Sama"])
        self.assertEqual(e2.count("Sama"), 1)
        self.assertNotIn("Langkah:", e2)
        e3 = error_box("Judul", "Sebab", [])
        self.assertIn("✗ Judul", e3)
        self.assertIn("Sebab: Sebab", e3)


class TestFooterCombo(unittest.TestCase):
    def test_footer_combo_satu_baris_tanpa_member(self):
        f = footer_combo("andalan", "provA", "modelX", "round_robin", None)
        self.assertNotIn("\n", f)
        self.assertNotIn("\r", f)
        self.assertIn("andalan", f)
        self.assertIn("provA/modelX", f)
        self.assertIn("Dijawab gabungan", f)
        self.assertEqual(f.count("memakai"), 1)
        self.assertNotIn("modelLAIN_XYZ", f)
        f2 = footer_combo("andalan", "", "", "", None)
        self.assertIn("memakai —", f2)
        self.assertNotIn("\n", f2)

    def test_footer_combo_fanout_dicoba(self):
        f_fast = footer_combo("andalan", "p", "m", "fastest", 3)
        self.assertIn("(dicoba 3)", f_fast)
        self.assertNotIn("\n", f_fast)
        f_cons = footer_combo("andalan", "p", "m", "consensus", 2)
        self.assertIn("(dicoba 2)", f_cons)
        f_rr = footer_combo("andalan", "p", "m", "round_robin", 3)
        self.assertNotIn("(dicoba", f_rr)


class TestParseKey(unittest.TestCase):
    def test_parse_key_panah(self):
        self.assertEqual(parse_key("\x1b[A"), "up")
        self.assertEqual(parse_key("\x1b[B"), "down")
        self.assertEqual(parse_key("\x1b[C"), "right")
        self.assertEqual(parse_key("\x1b[D"), "left")
        self.assertEqual(parse_key("\x1b[H"), "home")
        self.assertEqual(parse_key("\x1b[F"), "end")
        self.assertEqual(parse_key("\x1b[Z"), "unknown")
        self.assertNotEqual(parse_key("\x1b[Z"), "esc")
        self.assertEqual(parse_key_sequence("\x1b[A"), "up")

    def test_parse_key_jk_enter_esc_spasi(self):
        self.assertEqual(parse_key("j"), "j")
        self.assertEqual(parse_key("k"), "k")
        self.assertEqual(parse_key("J"), "j")
        self.assertEqual(parse_key("K"), "k")
        self.assertEqual(parse_key("\r"), "enter")
        self.assertEqual(parse_key("\n"), "enter")
        self.assertEqual(parse_key("\x1b"), "esc")
        self.assertEqual(parse_key(""), "esc")
        self.assertEqual(parse_key(" "), "space")
        self.assertEqual(parse_key("q"), "q")
        self.assertEqual(parse_key("\x03"), "ctrl-c")
        self.assertEqual(move_index(0, 3, "up"), 2)
        self.assertEqual(move_index(2, 3, "down"), 0)
        self.assertEqual(move_index(0, 3, "j"), 1)
        self.assertEqual(move_index(1, 3, "k"), 0)


class TestRenderCompact(unittest.TestCase):
    def test_render_model_compact_posisi(self):
        small = render_model_list(["a", "b", "c"], active_id="b")
        lines = small.splitlines()
        self.assertLessEqual(len(lines), 8)
        self.assertTrue(any("Model" in ln and "dari" in ln for ln in lines))
        self.assertIn("★ aktif", small)
        big = render_model_list(["m%d" % i for i in range(20)])
        big_lines = big.splitlines()
        numbered = [ln for ln in big_lines if re.match(r"^\d+\.", ln.strip())]
        self.assertEqual(len(numbered), 8)
        self.assertTrue(any("dari" in ln for ln in big_lines))
        self.assertIn("Model", big_lines[0])
        det = render_model_detail("m1", pos=2, total=5, active_id="m1")
        self.assertIn("Posisi:", det)
        self.assertIn("Model 2 dari 5", det)

    def test_render_provider_compact_posisi(self):
        small = render_provider_list(
            [{"id": "p1", "baseURL": "https://a.example.com/v1", "models": ["x"]}, {"id": "p2"}]
        )
        lines = small.splitlines()
        self.assertLessEqual(len(lines), 8)
        self.assertTrue(any("Penyedia" in ln and "dari" in ln for ln in lines))
        self.assertIn("p1", small)
        self.assertIn("p2", small)
        big = render_provider_list([{"id": "p%d" % i} for i in range(20)])
        big_lines = big.splitlines()
        numbered = [ln for ln in big_lines if re.match(r"^\d+\.", ln.strip())]
        self.assertEqual(len(numbered), 8)
        self.assertTrue(any("dari" in ln for ln in big_lines))

    def test_render_combo_compact_posisi(self):
        combos = {
            "b": {"strategi": "fastest", "models": ["m1"]},
            "a": {"strategi": "random", "models": ["m1", "m2"]},
        }
        txt = render_daftar(combos)
        lines = txt.splitlines()
        self.assertLessEqual(len(lines), 8)
        self.assertIn("Gabungan", lines[0])
        self.assertIn("a — random — 2 model", txt)
        self.assertIn("b — fastest — 1 model", txt)
        self.assertLess(txt.index("a —"), txt.index("b —"))
        det = render_detail("andalan", {"strategi": "fastest", "models": ["m1", "m2"]})
        self.assertIn("Strategi: fastest", det)
        self.assertIn("Model:", det)
        self.assertIn("Gabungan andalan", det)
        det7 = render_detail("andalan", {"strategi": "fastest", "models": ["m%d" % i for i in range(7)]})
        self.assertIn("+2 lainnya", det7)

    def test_render_lines_compact_viewport(self):
        filt = [(i, "item%d" % i) for i in range(3)]
        rl = render_lines("Judul", filt, 0, "", multi=False)
        self.assertLessEqual(len(rl), 8)
        self.assertIn("▶", rl[0])
        self.assertIn("Cari:", rl[1])
        self.assertIn("/bantuan", rl[-1])
        filt20 = [(i, "item%d" % i) for i in range(20)]
        rl20 = render_lines("Judul", filt20, 0, "", multi=False)
        self.assertEqual(len(rl20), 13)
        rl_multi = render_lines("Judul", filt, 0, "", multi=True)
        self.assertTrue(any("Spasi" in ln for ln in rl_multi))


class TestSaranInvalid(unittest.TestCase):
    def test_saran_invalid_tak_pernah_muncul(self):
        invalid = "/provider combo"
        renders = [
            render_model_list(["a", "b"]),
            render_provider_list([{"id": "p1"}]),
            render_daftar({"a": {"strategi": "random", "models": ["m1"]}}),
            render_detail("a", {"strategi": "random", "models": ["m1"]}),
            footer_combo("andalan", "p", "m", "round_robin"),
            error_box("Judul", "Sebab", ["Langkah A"]),
            footer_petunjuk(False),
            footer_petunjuk(True),
            "\n".join(model_lines(["m1", "m2"])),
            "\n".join(render_lines("J", [(0, "a")], 0, "")),
            "\n".join(suggest_models("a", ["a1", "a2"])),
            "\n".join(suggest_providers("p", [{"id": "p1"}])),
            daftar_baris("a", {"strategi": "random", "models": ["m1"]}),
        ]
        for r in renders:
            self.assertNotIn(invalid, r)
        self.assertTrue(any("/bantuan" in r for r in renders))


class TestPickerMurni(unittest.TestCase):
    def test_pick_single_multi_keys_murni(self):
        self.assertEqual(pick_single("J", ["a", "b"], show=lambda x: x, keys=["enter"]), 0)
        self.assertIsNone(pick_single("J", ["a", "b"], show=lambda x: x, keys=["esc"]))
        self.assertEqual(pick_single("J", ["a", "b"], show=lambda x: x, keys=["down", "enter"]), 1)
        self.assertEqual(pick_single("J", ["a", "b", "c"], show=lambda x: x, keys=["j", "enter"]), 1)
        self.assertEqual(
            pick_single("J", ["a", "b", "c"], show=lambda x: x, keys=["down", "k", "enter"]), 0
        )
        self.assertEqual(pick_multi("J", ["a", "b"], show=lambda x: x, keys=["space", "enter"]), [0])
        self.assertEqual(pick_multi("J", ["a", "b"], show=lambda x: x, keys=["esc"]), [])
        self.assertEqual(filter_items(["apel", "jeruk", "Anggur"], "ap"), [(0, "apel")])


if __name__ == "__main__":
    unittest.main()
