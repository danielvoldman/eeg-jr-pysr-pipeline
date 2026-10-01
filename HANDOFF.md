# HANDOFF.md (2026-10-01, written from memory at the end of a very long chat; verify against git and the tests)

Source of truth for the science: PREREGISTRATION.md (planning doc v0.6, tag prereg-v0.6). Standing rules: CLAUDE.md. Build order: PLAN.md. Every deviation (DEV-xxx) and implementation clarification (IMP-xxx): DEVIATIONS.md. Old entries are never edited except their Commit column; corrections go in new entries.

## Project
Coupled two-node Jansen-Rit model fit to two bipolar resting-state EEG channels (P3-PO3, P4-PO4, OpenNeuro ds003775 v1.2.1) with a UKF; PySR refines only a residual on the coupling term. Owner: Daniel Voldman. Windows, VS Code, PowerShell, Python 3.12 in .venv. ALWAYS use .venv\Scripts\python.exe (system Python has a stray `tests` package; tests/ has no __init__.py by design, IMP-051). Repo: C:\Users\Alpha\Videos\JansenRit_PYSR\JansenRit_Code\eeg-jr-pysr-pipeline.

## How we work
- Claude (chat) writes prompts; the owner pastes them into Claude Code; the owner pastes plans/results back for checking. Plan-first prompts for anything where the document leaves a value open; implement-directly prompts otherwise.
- Tests use expected values independent of the code under test; mutation checks (break, show failing test, restore, prove with cmp).
- Independent-reviewer prompt (fresh /clear) only after risky steps (caught real bugs after C2).
- Owner writes in capitals, wants short answers, gets frustrated by slowness; be honest about mistakes; never change a LOCKED value without a deviation entry; stop and ask where the document leaves a value open. A proof of concept with an honest "detectable, not exact" result is an accepted outcome.
- Real-data development uses ONLY the 12 pilot subjects (training side in all 5 seeds). Never read test subjects.
- Do not re-litigate: Wilson-Cowan, single-channel E/I split, open-form PySR, UDE, 4-phase layout, section 16.5.3 not-adopted list, Jansen-Rit choice.

## State (verify: git log --oneline -15; git status; pytest -rs -q)
HEAD c24c937 "DEV-005 adopted: filter A". Full suite: 761 passed, 1 skipped (C1 writer test), 3 deselected (pysr marker; pytest.ini excludes them; run `pytest -m pysr` before releases).
Recent commits (newest first): c24c937 DEV-005 adopted; 312f453/267339c E8; c710d66 E7; 58dc0ca E6; 335b505 E5; 6274038 E4; addf374 E3; fdeeb7c E2; 7f75020 E1; 999286c; c6c7f0b D2; 41c341f D1; 987cdd5 gate ordering; 9b5d9c4 Stage C closed; 4b134fd C8; d5b32f6 C7; f40fd8d C6; b93d488 C5; c1d2176 C3; a655db1/1c94e5b/06d84da C2.. ; Stage A/B earlier.
Done: Stages 0, A, B, C (closed), D (D1 machinery, D2 PySR 2.6.0 / SymbolicRegression.jl 2.5.1 / Julia 1.11.9 under .venv, turbo stays false), E1-E4 and E5 UKF-only stage, E6-E8 diagnostics.

## Key decisions
- DEV-005 ADOPTED: filter A = per-channel slow OU coloured observation-noise state (21-D pass 1, 14-D windows); 19-D dropped; B not adopted; divergence rule unchanged. q = 1e-2 pre-declared fixed (ukf.process_noise.q_fixed, locked). g0.filter = A; ukf.numba.enabled = true (DEV-004: smoothed-covariance tolerance rtol 1e-6, atol 1e-8; everything else 1e-8/1e-10).
- DEV-003: 0.5 s start-up exemption for the state-SD divergence flag. DEV-001/002 earlier.
- Why: real EEG has a 1/f background the model lacks; 19-D parameters run away (p ~22 prior SD) and gains are lost. A/B fix divergence and nulls; gain error still ~20-40% at best on easy synthetic data (4-21% with planted residual zero); matched real-like operating points (p=120/138, all limit-cycle) fail; A absorbs part of alpha (8-11% at q=1e-2, 25-32% at 1e-3); NIS 0.5-0.9 so the section 7.5 rule refuses and lands on the grid edge (reported alongside).
- Known risk: A's nulls failed at matched points in E5, so full G0 may HARD STOP under section 20. Preregistered behaviour; decide with numbers and a deviation entry, never by quietly loosening the gate.
- L-vs-M (matched weak form) and derivative estimator choice deferred to Stage I on filtered states. turbo off. TV derivative is pilot-set only.
- Seeds: split 42 (extra 43-46); G0 blocks 91000 pilot, 92000 full, 93000 tuning, 94000 preprocessing gate, +10000 per fresh-seed round.

## Next
1. Docs: fill c24c937 into DEV-005's Commit line (in the next commit).
2. WIRING GAP (do first in Stage F): nothing reads q_fixed; passes.run_pass1 still defaults to 19-D. Make real-data runs use A at q=1e-2 unless overridden, with a test.
3. Stage F: src/baseline.py, M0 VAR (order 1-32 by 5-fold subject-wise CV, pooled frozen coefficients) and M0b (refit per test recording), scored on exactly the same samples as M1-M3 (section 10.2). Plan-first prompt.
4. Stage G (robustness/scoring: ablation M0-M3, bootstrap, ICC, wPLI, AAFT, C4 free-run, section 5.2 sensitivity), Stage H (figures), then pilot (--pilot; E5 PySR stage with A never run yet; estimated 8-12 h, run in checkpointed stages, time one series first, stop if >12 h), lock placeholders (Stage J: any pilot change gets a deviation; add limitations note: G0 covers only limit-cycle regime), full G0 (Stage I) and full run (Stage K), requirements.txt/README/LICENSE (Stage L; pin joblib 1.6.0, specparam 2.0.0rc7, pysr etc.). Check PLAN.md for exact stage letters.
5. Open from section 21.2: noise-adaptive R (largely moot), OS affinity mask/SMT pairing (Windows, FF mask honoured by Julia), worker count 4/6/8 benchmark, Numba validation (passed in C5).

## Lessons
- Claude Code sometimes ends its turn without printing a report; ask it to check git log/status and print results from files. Background jobs keep running.
- Disable sleep during long runs; results are saved only at the end of some drivers.
- Check line counts/first lines after copying files (earlier mix-up).
- Do not tune any value after seeing test-subject results.

## Verify first
git log --oneline --decorate -15 ; git status ; .\.venv\Scripts\python.exe -m pytest -rs -q ; git tag
Expect clean tree, 761 passed / 1 skipped / 3 deselected, tag prereg-v0.6.