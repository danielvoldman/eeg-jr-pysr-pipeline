# HANDOFF.md

Written 2026-09-29 at the end of a very long chat, to start a fresh one. It is a summary from memory, so **verify it against the repo before trusting it**: run the commands in the last section and let git and the test suite override anything written here.

`PREREGISTRATION.md` (planning document v0.6, frozen at tag `prereg-v0.6`) is the source of truth for the science. `CLAUDE.md` holds the standing rules for Claude Code. `PLAN.md` holds the build order and notes. `DEVIATIONS.md` holds every deviation (§1) and implementation clarification (§2).

## Project

A research pipeline: a coupled two-node Jansen-Rit model fit to two bipolar resting-state EEG channels (P3-PO3, P4-PO4, OpenNeuro ds003775 v1.2.1) with an unscented Kalman filter, with PySR refining only a residual on the coupling term. Four phases on a home CPU. Owner: Daniel Voldman. Environment: Windows, VS Code, PowerShell, Python 3.12.10 in `.venv` (not conda base). Repo folder: `C:\Users\Alpha\Videos\JansenRit_PYSR\JansenRit_Code\eeg-jr-pysr-pipeline`. The repo is not inside OneDrive.

## How we work

- Claude (in this chat) writes prompts; the owner pastes them into Claude Code in VS Code; the owner pastes Claude Code's plan or results back here for checking.
- Prompt pattern: read CLAUDE.md, PLAN.md and named preregistration sections; plan first; wait for approval where the document leaves something open; implement; run tests and show the full output; commit with the step name and section numbers; STOP at named points.
- Independent reviewer prompt (fresh `/clear` session) after risky steps. It has caught real problems in A1, A2, A4 and B3-B6. Skip it for plumbing.
- Tests must use expected values that are independent of the code under test, and mutation checks (break the code, show the test fails, restore, prove with `cmp`).
- Batching several small steps into one prompt is agreed, with a stop only where the document leaves something open.
- The owner writes in capitals and sometimes asks for one-sentence answers; honour that. Be honest about my own mistakes, briefly.
- Do not re-litigate: Wilson-Cowan, single-channel E/I splitting, open-form PySR discovery, UDE escalation, the §18 repo layout and 4-phase design, the §16.5.3 "not adopted" list.
- Old DEVIATIONS entries are never edited, except their Commit column. Corrections go in a new entry.
- Real-data development uses ONLY the 12 pilot subjects (training side in all five seeds). Never read test subjects while developing.

## State at the end of this chat

Commits (newest first, from memory; verify with git): `c96fc57` DEV-002 paperwork, `67a08e8` B5-B6 cleanup (IMP-013), `fb651ed` B5-B6, `b4c377f` and `3fc4e4f` (DEV-001), `146c21f` and `0643fd9` (B3-B4), `eed3a5f` and `0cee46e` and `3b8b419` (B2), `038ce82` and `7252c34` (B1), `b7cdcf0` to `2ebdfcc` (A4), `a0be7be` and `280131d` (A3), `6fe743f` and `e356efb` (A2), `aee998b` and `d5b6d18` (A1), `a035a06` (Stage 0, tag `prereg-v0.6`). 360 tests passing at `67a08e8`.

**DEV-002 is committed** (`c96fc57`): the corrected-time exclusion rule "excessive_blink_correction" (10% of the trimmed recording, after the bad-electrode rule and before the clean-data minimum) is in DEVIATIONS.md §1, and IMP-013's Commit column holds `67a08e8`. Nothing is pending.

**Done (PLAN.md ticks):** Stage 0; A1 main.py skeleton; A2 Jansen-Rit simulator; A3 zero-phase band-pass and FFT downsampling; A4 reference constants; B1 download (632 files, manifest recorded); B2 split; B3-B4 loading, units check, notch, band-pass, bipolar pairs; B5-B6 blink correction, segment rejection, rescaling, exclusions.

**Key values:** `mu_ref = 7.572449764282774`, `sigma_ref = 1.131413384422008` mV (seed 42, computed at 7191d10, stored in config.yml with provenance; a guard test recomputes them). Primary split seed 42; extra seeds 43-46; pilot draw seed 42. 78 train / 33 test. Pilot subjects: sub-005, sub-008, sub-019, sub-051, sub-056, sub-063, sub-074, sub-081, sub-082, sub-086, sub-088, sub-107. Split-file SHA-256s and per-seed test-side t2 counts (8, 16, 11, 12, 9) are in the B2 note in PLAN.md. Copy `outputs\split_*.json` to a folder outside the repo as a backup (outputs/ is git-ignored).

**Deviations so far:** DEV-001 (electrode RMS lower bound 5 -> 1 uV; also the segment floor). DEV-002 (corrected-time exclusion rule). IMP-001 to IMP-013 (see DEVIATIONS.md §2).

**Settled choices worth remembering:** provenance tags in config.yml are literature, locked, placeholder, computed, fitted, unset (`unset` = the document gives no value: stop and ask). Config leaves have the form `{value, prov, ref}`. Pilot flags live under results/pilot/. Edge trims 5.0 s (0.5 Hz) and 25.0 s (0.1 Hz variant). Blink correction: undecimated wavelet (sym4, level 10, levels 7-10, clip at 3 robust SD, +/-0.25 s splice). Filter: 4th-order Butterworth forward-backward (nominal cutoffs are the -6 dB points). Simulator: Heun at 2048 Hz, per-step uniform(120,320) input, 20-step delay.

**Deferred to the phase-1 runner (noted in PLAN.md B4 and B6):** the 5% units-check halt that actually stops, calling write_exclusions and per-variant exclusion file naming, config_sha256 and manifest_summary extraction, saving the band-pass impulse response for 0.5 and 0.1 Hz under outputs/, restricting `allow_all` to the phase-1 runner with a test. Also: `requirements.txt` (pin at Stage L, including PySR, filterpy, openneuro-py 2026.9.1, pywt, numba, scipy, mne), README, LICENSE.

**Open items from preregistration §21.2:** (1) hard rejection vs noise-adaptive observation covariance (pilot); (2) OS is Windows, affinity mask FF still to verify (SMT pairing); (3) worker count 4/6/8 (pilot benchmark); (4) Numba UKF validation vs filterpy.

## Next: Stage C (the coupled state-space model and UKF)

Planned order (my proposal, not yet started):
- **C1** augmented 19-state model (12 neural + p1, p2, rho1, rho2, g12, g21, m), priors, E/I reparameterization, delay ring buffer; one predict step, no filter.
- **C2** reference UKF with filterpy plus the unscented RTS smoother, tested on simulated two-node data with known truth (including the negative-eigenvalue / NaN stability check; fallback is dimension reduction, then square-root UKF, §7.4, §7.5).
- **C3** two-pass parameter handling (recording-level parameters held fixed in 2-s windows).
- **C4** Q/R tuning rule (NIS closest to 2; separate G0 tuning set).
- **C5** Numba port validated against filterpy (rtol 1e-8, atol 1e-10, plus unchanged downstream statistics).
Then Stage D (PySR regression machinery), E (synthetic gate G0), F (baseline), G (robustness and scoring), H (figures), pilot, lock placeholders, full run, release files.

Preregistration sections to read for C1-C2: §7.1-§7.6, §8.1, §9.3, §10.1. Document facts that bite: the scaled unscented transform at n = 19 with alpha 1, beta 2, kappa 0 has mean weights 0 (centre) and 1/38, covariance centre weight 2; the delay buffer is held outside the sigma points and the state stays 19-D; the filter uses 4 Heun sub-steps per 256 Hz observation with process noise added once per observation step.

## Lessons from this chat (avoid repeating)

- Two file mix-ups happened (CLAUDE.md content ended up in PREREGISTRATION.md). Always check line counts and first lines after copying.
- My specs contained three errors caught by Claude Code stopping: the EDF-header saturation rule, the sine-only edge-transient test, and the wavelet test without the band-pass. Prefer measuring before fixing thresholds, and keep the STOP rules.
- Do not tune any value after seeing test-subject results. The 1 uV floor and the trims were decided on pilot data only.

## Verify before continuing (run in PowerShell, venv active)

```
git log --oneline --decorate -15
git status
.\.venv\Scripts\python.exe -m pytest tests -q
Select-String -Path config.yml -Pattern "mu_ref|sigma_ref" | Select-Object -First 4
git tag
```

Expect a clean tree, all tests passing, both constants filled in, and the tag `prereg-v0.6`.
