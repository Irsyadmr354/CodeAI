# Polish TUI Ledger — 2026-09-05 (branch loop-eng/polish-tui-20260905)

Baseline: e60ae19 + dirty user work (preserved). Tooling: pytest 31 passed baseline; lint/build NOT DETECTED (gap). Rollback: git OK, isolated branch.
Dispatch: 2 wave (Wave1 gateway OK; Wave2 CLI+Combo re-dispatch after explicit auth). Same-turn issued, runtime serialized — disclosed.

Changes (surgical, stdlib only, API preserved):
- harness/cli.py: /login direct-arg tolak combo ("Pilih provider asli, bukan combo"); hapus dead _is_direct_effort_switch; picker numbered + unified /model /login /combo /effort; steer jujur "Tidak ada tugas berjalan"; discovery fast-path cache TTL30s + background refresh (hindari block 8s/1s); prompt box/spinner dipertahankan.
- harness/models/combo.py (+88/-27): guard nested combo/ ValueError; fastest 30s / consensus 60s; atomic save tmp+replace + chmod 600; Lock round_robin/load_times; tools truthy → explicit ValueError.
- harness/models/gateway.py: RLock + _cooldown_lock; get_provider + chat failover atomik snapshot→apply→chat→restore; failover_order/300s/chains untouched.

Verification independen: py_compile OK; pytest 31 passed (baseline 31 → now 31, no regresi); secret sweep SWEEP_DONE bersih; diff-size 1307+/372- (mayoritas prior user work + 3 fix surgical).
DoD 6-gate: 1 Correctness PASS, 2 Completeness PASS (CONVERGED, 0 unverified), 3 Consistency PASS, 4 Guard PASS, 5 RootCause PASS, 6 Env/Rollback PASS (isolated branch).
Next: review `git diff` di branch lalu merge/PR ke main + hapus __pycache__.
