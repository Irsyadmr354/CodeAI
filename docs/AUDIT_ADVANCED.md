# AUDIT_ADVANCED — CodeAI Harness (Full A–I, Read-Only)

Tanggal: 2026-09-05 | Scope: /home/royan/code-ai | Mode: loop-engineering audit (read-only) | Output: File docs (sesuai klarifikasi user)
Metode: 3 subagent paralel same-turn (A-B, C-E, F-I) + verifikasi induk spot-check file:line. Mutasi dilarang.

## BAGIAN 1 — KONFIRMASI PER POIN (YA/TIDAK + File:Line + Root Cause bila TIDAK)

### A. Governance AGENTS.md
- A1 YA — harness/core/orchestrator.py:209 — layers [BASE,SYSTEM,GLOBAL,AGENTS]+dispatch_before_init→[HOOKS]+[USER], label auditable.
- A2 TIDAK — harness/tools/shell_runner.py:47 — run() hanya guard.evaluate, tanpa verify_tool_call. Root cause: satu-satunya verify semu di orchestrator.py:275 hanya untuk pseudo-tool "run_pipeline"; DiffEditor/SuperSkill bypass + agents_parser abaikan boundaries. Severity: HIGH (3x2x3=18).
- A3 YA — harness/core/orchestrator.py:108 — path-aware ../~/C:/etc/outside-workspace BLOCKED, mention benign → False.
- A4 YA — harness/core/allowance.py:81 — shlex+_has_chaining, `echo hi && rm` → ASK/BLOCKED, re.IGNORECASE, typo → ASK.

### B. SYSTEM_HOOKS.md lifecycle
- B1 YA — harness/core/hooks.py:56 — fence ```yaml|yml case-insensitive + plain fence, lower(type) 4 hook, user override append.
- B2 YA — harness/core/hooks.py:79 — action.strip().lower() in (block,deny,block_tool,deny_tool) → PermissionError.
- B3 YA — harness/core/hooks.py:88 — filter bool True/int==1/str true/1, try int(limit) fallback 100, limit<0→100, isinstance(output,str) only.
- B4 YA — harness/core/hooks.py:122 — if key in _ALLOWED_INJECT_KEYS (8 keys: context,extra_context,system_note,memory_hint,time_context,current_time,locale,timezone).

### C. Kompresi token
- C1 YA — harness/context/compactor.py:6 + harness/tools/shell_runner.py:85 — strip ANSI CSI+OSC, trim_log_output max 10 + marker budget.
- C2 YA — harness/context/compactor.py:109 + harness/core/orchestrator.py:312 — _compact_history → keyword summary + fact_cards, bukan placeholder.
- C3 YA — harness/context/ast_indexer.py:44 + harness/tools/super_skill.py:10 — signature+docstring rekursif _collect_nested, bukan full text.

### D. Multi-provider
- D1 YA — harness/models/provider_registry.py:58 — 18 id + env→vault→discover→ollama + _discover_models_live cache.
- D2 YA-sebagian — harness/models/gateway.py:141 — _build_failover_chain failover_order+fallbackChains target/*/wildcard + 300s cooldown + snapshot/restore. Catatan: tanpa lock (lihat GAP-D-RACE).
- D3 YA-sebagian — harness/models/provider_registry.py:192 + gateway.py:174 + combo.py:12 — baseURL → UniversalOpenAI + combo/<nama> 12 strategi + cli.py:1468. Catatan logic gap combo-dalam-combo, supports_tools drop (lihat GAP-D-LOGIC).
- D4 TIDAK — harness/models/providers/copilot.py:26 + gemini.py:146 — urlopen tanpa timeout + ?key= tanpa masking. Root cause: masking hanya shell_runner.py:37, tanpa guard terpusat di layer provider; tanpa chmod-600 enforcement terverifikasi di combo save. Severity: HIGH (3x3x2=18).

### E. Effort inside-setting
- E1 YA — harness/cli.py:1254 + config.py:16 — /effort + _normalize_effort_syntax colon→dash + persist atomik.
- E2 YA-sebagian — antigravity.py:108 + openai.py:25 + anthropic.py:23 + gemini.py:37 + gateway.py:98 — --effort/reasoning_effort/budget/thinkingLevel + base bersih via parse_model_effort. Catatan: gemini.py:200 agy path hilangkan effort + fallback --model kotor (lihat GAP-E-LOGIC). Severity: MEDIUM (2x2x2=8).

### F. CLI/TUI
- F1 YA — harness/cli.py:729 vs :472 — _ask_main box ╭─❯+│❯ beda dari _popup_render title+🔍+list bernomor.
- F2 YA — harness/cli.py:1408 — _reject + _is_known:1413 + cancel:1506 return None tanpa _do_switch; _TUIPicker:158 VIEWPORT 15 + :177 live filter + :192 viewport + :300 panah/jk.
- F3 YA-sebagian — harness/cli.py:955 — login filter id!=combo && api!=local + _tui_pick:974; combo:1579, effort:1273 pola sama. Catatan bypass: cli.py:1005 /login combo direct-arg lolos ke generic vault (GAP-F-BYPASS, MEDIUM 2x2x2=8).
- F4 YA — harness/cli.py:871 — Live Spinner dots thinking·short·0.0s·Ctrl-C + elapsed:878 + fallback Status:901 + non-rich tick:927.

### G. Subagent-first + steering
- G1 TIDAK — harness/core/orchestrator.py:328 — gateway.chat parent-direct; spawn:250 hanya tracking CODER tanpa worker. Root cause: delegasi semu, RESEARCHER/VERIFIER tak pernah di-spawn/dieksekusi. Severity: HIGH (3x2x2=12).
- G2 YA-sebagian — harness/core/orchestrator.py:381 — send_to_subagent steer tanpa reset history/current_task via cli:1698. Catatan silent-drop bila bukan RUNNING:379 + queue tanpa konsumen (_wait:367 tak dipanggil pipeline), Ctrl-C:779 tak forward (GAP-G-STEER, MEDIUM 2x3x1=6).

### H. Tools + verifikasi
- H1 YA — harness/tools/diff_editor.py:75 — os.replace tmp+rename se-dir atomik + tolak kosong:40,45 + ambigu>1:56.
- H2 YA-sebagian — harness/tools/shell_runner.py:47 — _mask:37 + timeout:81 + guard BLOCKED:59/ASK:66. Catatan: guard None→APPROVED fail-open:57 + mask miss bare AIza/ya29/short<16 (GAP-H-MASK, MEDIUM 2x3x2=12).
- H3 TIDAK — harness/verification/loop.py:28 — while <max+1 re-run lint+pytest sama tanpa patch. Root cause: tanpa healing nyata + retries_used:59 jadi 3 off-by-one (max 2 → 3 attempt). Severity: MEDIUM (2x3x1=6).

### I. Identitas
- I1 TIDAK — harness/core/orchestrator.py:308 — identitas hanya system prompt CODEAI_IDENTITY:16 tiap turn. Root cause: tanpa post-filter/verifier sehingga klaim Antigravity/Claude/GPT/DeepSeek lolos bila LLM abaikan prompt. Severity: LOW (1x2x2=4).

## BAGIAN 2 — JAWABAN (SEMUA A–I benar? Hidden gap?)

TIDAK. 4 TIDAK keras (A2, D4, G1, H3) + 1 TIDAK identitas (I1) + 7 hidden gap di bawah. False-green dan race confirmed.

Hidden gap / logic error / race / false-green (File:Line + Root Cause + Severity):
- GAP-A-BOUND1 HIGH 12 — orchestrator.py:118 bare `cat ~` → False (regex butuh ~/); orchestrator.py:131 outside+44char → False (window 40 bypass); orchestrator.py:143 `explain production DB` → False (exempt prefix bypass). Cause: regex terlalu sempit + exempt list terlalu luas.
- GAP-A-HOOK HIGH 12 — hooks.py:59 Type capital missed? (faktanya lower() aman, tapi _parse_simple_yaml:32 abaikan indent/nested → key dalam nested block terlewat); hooks.py:77 missing target → skip fail-open; orchestrator.py:289 except pass fail-open; agents_parser substring ignore boundaries (benign `rm-rf` → False negatif/positif campur); orchestrator.py:235 [USER] inside system spoofable; load-once stale orchestrator.py:50+hooks.py:49.
- GAP-D-RACE HIGH 16 — gateway.py:50 cooldown dict tanpa lock + combo.py:116 fastest/consensus share satu gateway, _apply_lock hanya apply bukan chat+restore → pollution paralel. Cause: lock scope terlalu kecil.
- GAP-D-LOGIC MEDIUM 8 — gemini.py:200 agy path hilangkan effort; universal_openai.py:33 reasoning_effort unconditional vs hint-gated; combo.py:43 save tanpa atomic/chmod; combo supports_tools False drop tools; combo-dalam-combo rekursi tanpa guard; fallback --model gemini-3.7-flash-low kotor.
- GAP-E-LOGIC MEDIUM 8 — lihat D-LOGIC effort; plus fact_cards tak pernah dikompres ulang compactor.py:99; skeleton tanpa auto-threshold hanya manual super_skill.py:60.
- GAP-F-COMBO MEDIUM 8 — cli.py:1005 direct-arg bypass filter:955; cli.py:1327 _is_direct_effort_switch dead; custom-model empty-catalog unreachable vs registry is_known passthrough:495.
- GAP-G-STEER MEDIUM 6 + GAP-H-MASK MEDIUM 12 + GAP-H-DIFF LOW 4 — steer silent-drop:379 + queue tanpa konsumen + Ctrl-C tak forward:779 + join timeout 5:888 daemon leak; MessageBus asyncio.Queue cross-loop via ThreadPoolExecutor:393 race active_subagent_id/workflow_state tanpa lock; shell guard None→APPROVED:57; diff BLOCK_PATTERN:18 rapuh \r\n/spasi + lines_changed:66 hitung replace bukan diff; combo fastest:116/consensus:139 tanpa timeout + round_robin:108 tanpa lock; registry ollama:304 1s + discovery:366 8s block UI.
- FALSE-GREEN — tests/test_governance.py:63 no post/pre_gen/traversal/chaining coverage; tests/test_combo.py:60 fastest sleep 0.2/0.01 flaky + consensus longest-bukan-voting + token len//4; tests hanya governance/combo/registry tanpa cover TUI/steer/spinner/diff/shell/loop/identitas → klaim PASS semu.

Verifikasi induk (5-gate采样): spot-check orchestrator.py:108,209,250,328 + hooks.py:56,79,88,122 + shell_runner.py:47 + allowance.py:81 + diff_editor.py:75 + loop.py:28 + gateway.py:141 — klaim subagent cocok dengan kode (kecuali nuansa hooks Type capital yang faktanya aman karena lower(), dicatat sebagai over-report minor).

## BAGIAN 3 — NEXT (Polish TUI ditunda sesuai izin "Jalankan audit" saja)

Belum eksekusi redesign. Proposal Polish TUI (tanpa dep baru) menunggu greenlight terpisah: samakan popup+search /model /login /combo /effort, tutup bypass direct-arg, tambah guard combo-dalam-combo, timeout fastest/consensus, non-blocking discovery. Jangan implement sebelum konfirmasi.

DoD: Correctness PASS (bukti file:line), Completeness PASS (A–I full), Consistency PASS, Sensitive-Data Guard PASS (zero secret di ledger), Root Cause PASS, Env/Rollback N/A (read-only, tanpa mutasi).
Dispatch: same-turn 3 audit — observed serialized-sequential oleh runtime (3 completed bertahap), dicatat per Invariant 8.
