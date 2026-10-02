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
HEAD = c151f4c (docs: IMP-080 to IMP-082 commit columns) on top of a4b5b90 (G2 stage). Full suite: 929 passed, 1 skipped (tests/test_state_space.py:476, the write_initial_covariance writer test: neural_variance is already written; it is NOT a C1 scoring writer), 3 deselected (pysr marker; pytest.ini excludes them; run `pytest -m pysr` before releases).
Recent commits (newest first): c151f4c docs (IMP-080 to IMP-082 commit columns); a4b5b90 G2/G4/G6 (Holm family, C3 ICC, wPLI, AAFT; IMP-080 to IMP-082); 85290c9, a318ae7 IMP-079 docs; fa2c0d9 G1 (C1 scoring); 824c8aa F1 (baseline.py, scoring_mask, IMP-075/076); dc48b4f F0 (filter A and q_fixed wired, DEV-006, IMP-074); 2f61362 golden numbers; c25886b; c24c937 DEV-005 adopted; 312f453/267339c E8; c710d66 E7; 58dc0ca E6; 335b505 E5; 6274038 E4; addf374 E3; fdeeb7c E2; 7f75020 E1; 999286c; c6c7f0b D2; 41c341f D1; 987cdd5 gate ordering; 9b5d9c4 Stage C closed; 4b134fd C8; d5b32f6 C7; f40fd8d C6; b93d488 C5; c1d2176 C3; a655db1/1c94e5b/06d84da C2.. ; Stage A/B earlier.
Done: Stages 0, A, B, C (closed), D (D1 machinery, D2 PySR 2.6.0 / SymbolicRegression.jl 2.5.1 / Julia 1.11.9 under .venv, turbo stays false), E1-E4 and E5 UKF-only stage, E6-E8 diagnostics, F0 (dc48b4f) and F1 (824c8aa): Stage F closed. G0.5 (c3682ba: M3 residual hook, src/ukf_resid.py NumPy filter, frozen-equation writer/loader, IMP-077) and G1 (C1 scoring, bootstrap, holm(p, m), guard, src/robustness.py, IMP-078) done on pilot subjects only; G2 partly (Holm family and adjusted values open). results/pilot/c1_42.json exists (mechanics only, M3 absent: no frozen equation yet). G2 stage (a4b5b90), on pilot subjects only: Holm family = 11 members (IMP-080: 2 C1 steps + 3 free-run windows + 2 test-partition ICC directions + 4 non-primary seeds; statistics.holm_family_size = 11; no adjusted value until all 11 raw p exist, the adjustment runs once in G8); G4 (C3 ICC) marked [~]: machinery built (ICC(3,1) per direction, F-based and cluster-bootstrap CIs, vigilance-adjusted ICC, pair rules, pilot-only guard) but no frozen equation exists, so the pilot file uses the M2 stand-in and makes no verdict (IMP-081); G6 done on pilot data (wPLI, imaginary coherency, AAFT; aaft.seed = 42, placeholder; IMP-082). Pilot outputs: results/pilot/c3_42.json, results/pilot/diagnostics_42.json (mechanics only). G2 (Holm) box stays open in PLAN.

## Known risks (G2 stage)
- Real M2 diverged at recording level on 6 of the 12 pilot ses-t1 recordings (the DEV-005 risk again); the pilot AAFT therefore ran on 6 recordings, not 10 (replacement draws ran out). A full G0 may still HARD STOP under section 20; decide with numbers and a deviation entry.
- The AAFT test cannot separate coupling from lagged common input (section 14); only Null B in G0 addresses that. Same for wPLI and imaginary coherency.
- Pilot C3 stays at n = 2 usable pairs (sub-051 ses-t2 has too little clean data, sub-074 ses-t1 diverged): the pilot ICC and its CIs are mechanics only and carry no information.

## Key decisions
- DEV-005 ADOPTED: filter A = per-channel slow OU coloured observation-noise state (21-D pass 1, 14-D windows); 19-D dropped; B not adopted; divergence rule unchanged. q = 1e-2 pre-declared fixed (ukf.process_noise.q_fixed, locked). g0.filter = A; ukf.numba.enabled = true (DEV-004: smoothed-covariance tolerance rtol 1e-6, atol 1e-8; everything else 1e-8/1e-10).
- DEV-003: 0.5 s start-up exemption for the state-SD divergence flag. DEV-001/002 earlier.
- Why: real EEG has a 1/f background the model lacks; 19-D parameters run away (p ~22 prior SD) and gains are lost. A/B fix divergence and nulls; gain error still ~20-40% at best on easy synthetic data (4-21% with planted residual zero); matched real-like operating points (p=120/138, all limit-cycle) fail; A absorbs part of alpha (8-11% at q=1e-2, 25-32% at 1e-3); NIS 0.5-0.9 so the section 7.5 rule refuses and lands on the grid edge (reported alongside).
- Known risk: A's nulls failed at matched points in E5, so full G0 may HARD STOP under section 20. Preregistered behaviour; decide with numbers and a deviation entry, never by quietly loosening the gate.
- L-vs-M (matched weak form) and derivative estimator choice deferred to Stage I on filtered states. turbo off. TV derivative is pilot-set only.
- Seeds: split 42 (extra 43-46); G0 blocks 91000 pilot, 92000 full, 93000 tuning, 94000 preprocessing gate, +10000 per fresh-seed round.

## Next
Stage letters follow PLAN.md (G3 = C2 aggregation, G5 = C4 free-run, G7 = section 5.2 sensitivity, G8 = summary files).
1. G3 (C2 aggregation: signature recurrence and C1 in at least 4 of 5 seeds; supplies the 4 c1_seed Holm members), G5 (C4 free-run; supplies the 3 free-run Holm members, contrast and raw p to be fixed in its plan), G7 (section 5.2 sensitivity), G8 (summary.json, equations.tex, and the Holm adjustment once all 11 raw p exist).
2. Stage H (figures), then pilot (--pilot; E5 PySR stage with A never run yet; estimated 8-12 h, run in checkpointed stages, time one series first, stop if > 12 h; it also produces the first frozen equation, which G4 and G1 need), Stage J (lock placeholders, limitations note incl. G0 covers only the limit-cycle regime), Stage I (full G0), Stage K (full run), Stage L (requirements.txt, README, LICENSE; pin joblib 1.6.0, specparam 2.0.0rc7, pysr etc.). Check PLAN.md for exact stage letters.
3. Open from section 21.2: noise-adaptive R (largely moot), OS affinity mask/SMT pairing (Windows, FF mask honoured by Julia), worker count 4/6/8 benchmark, Numba validation (passed in C5).

## Lessons
- Claude Code sometimes ends its turn without printing a report; ask it to check git log/status and print results from files. Background jobs keep running.
- Disable sleep during long runs; results are saved only at the end of some drivers.
- Check line counts/first lines after copying files (earlier mix-up).
- Do not tune any value after seeing test-subject results.

## Verify first
git log --oneline --decorate -15 ; git status ; .\.venv\Scripts\python.exe -m pytest -rs -q ; git tag
Expect clean tree, 929 passed / 1 skipped / 3 deselected, tag prereg-v0.6.