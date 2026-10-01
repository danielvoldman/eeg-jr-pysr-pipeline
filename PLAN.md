# PLAN.md: build order and progress

Work on one step per session, and run `/clear` between steps. Each step lists the preregistration sections to read, the files it touches, and what "done" means. When a step is finished, tick its box and add a dated one-line note under it.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

This is the **build** order. The **run** order is different and is fixed by §16.5.1 and §18.1.

---

## Stage 0: Manual setup (done by me, not Claude Code)

- [ ] Repo folder, `git init`, and `.gitignore` copied from §18.3
- [ ] `PREREGISTRATION.md`: v0.6 converted from .docx, with the §7.6 and §8.2 tables checked
- [ ] `CLAUDE.md`, `PLAN.md` and `DEVIATIONS.md` in place
- [ ] `config.yml` seeded with the §7.3 and §7.6 values and their provenance tags
- [ ] `.venv` created and the Python version chosen
- [ ] First commit

---

## Stage A: Foundations and reference constants

### [x] A1. main.py skeleton
Read: §16.2, §18, §18.1.
Files: `main.py`, plus a small config-reading helper. §18 lists no separate config module, so ask before adding one.
Done when:
- `python main.py --phase N [--pilot]` parses its arguments.
- The thread variables are set before NumPy is imported. A test shows 1 BLAS thread inside a spawned worker, using `threadpoolctl`.
- Phase N refuses to start without `outputs/phase{N-1}.done`.
- Pilot output goes under `results/pilot/`.
- Run-time folders are created by the code and never committed.

Note (2026-09-28): done; main.py, src/config.py, tests/ (32 pass). Runners are stubs that exit 1 and never write a flag. IMP-001 and IMP-002 logged.

### [x] A2. Jansen–Rit simulator (model.py)
Read: §7.1, §7.3 (including the sigmoid), §7.2, §7.5 (delay), §7.6 (integrator), §8.1, §9.1 (generation rate).
Files: `src/model.py`.
Done when tests show:
- The sigmoid behaves as specified: S(v₀) = e₀ = 2.5 s⁻¹, it approaches 2e₀ = 5 s⁻¹, and S(0) ≠ 0 (no offset, per §7.3).
- The six-state right-hand side matches the §7.1 equations term by term on a hand-computed example.
- The coupling enters the y₄ bracket alongside p (§8.1).
- With g₁₂ = g₂₁ = 0 and no mixing, the two-node output equals two independent single nodes run with the same noise draws.
- The delayed coupling reads exactly d sub-steps back (impulse test).
- The same seed gives bit-identical output, and different seeds give different output.
- A 600-s run produces no NaN or Inf.
- Sanity report (not a pass rule): the spectral peak of the noise-driven output at p = 220. Classic Jansen–Rit gives alpha around 10 Hz. If it doesn't, stop and tell me.

Note (2026-09-28): done; src/model.py (Heun, numba kernel + py_func fallback), tests/test_model.py. Sanity report: single-node p=220 y1-y2 Welch peak 10.75 Hz. IMP-003 logged; numba and scipy installed (approved). The 600-s duration vs 10-s burn-in question is left to A4.

### [x] A3. Filtering and downsampling functions (preprocess.py)
Read: §5.1 (band-pass step and downsample step), §10.1.
Files: `src/preprocess.py` (filter and resample functions only).
Done when tests show:
- The filter is zero-phase: a 10 Hz sine shows no lag.
- Passband gain is about 1 at 10 Hz, and there is clear attenuation at 0.1 Hz and 60 Hz.
- The anti-alias and downsample path works from both 1,024 Hz and 2,048 Hz to 256 Hz.
- bandpass_impulse_response returns the response; saving and documenting it is done in B4.

Note (2026-09-28): done; src/preprocess.py (bandpass, bandpass_sos, bandpass_impulse_response, downsample), tests/test_preprocess.py. Butterworth SOS via sosfiltfilt, order 4 per pass, so the nominal 0.5 and 45 Hz cutoffs are the -6 dB points; FFT downsampling with a 2 s odd reflect-pad (IMP-004). bandpass() is for whole continuous recordings before segmentation, never for short segments; inputs of padlen samples or fewer raise by design. Trimming or masking the filtered edges is decided in B4 using the printed edge-transient numbers; nothing is trimmed in A3. The impulse response is returned by a function only; saving it is deferred to phase 1 (see the B4 bullet). Note (2026-09-28, after review of a0be7be): the impulse-response "Done when" item was clarified to say the function returns the response and saving and documenting it is done in B4.

### [x] A4. Reference constants μ_ref and σ_ref
Read: §5.1 (rescaling paragraph), §7.3, §7.6 ("Rescaling constants" row).
Files: `src/model.py` (`compute_reference_constants`), which may import A3's functions from `preprocess.py`; `config.yml`.
**Open questions. Ask me before implementing:**
1. Is the reference simulation a single node, or the two-node model at g = 0? If two-node, is mixing on or off?
2. Is the uniform(120, 320) input redrawn at every 2,048-Hz integration step, or held over some interval? Per-step noise has an effect that depends on the step size.
3. Which integrator: Heun (§7.6) or Euler–Maruyama / stochastic Heun (§9.1)?
4. How much initial transient is discarded as burn-in before computing the constants?
5. Is μ_ref, "the mean of the unfiltered output", taken before or after the anti-alias downsampling to 256 Hz?
6. Which seed value?

Done when:
- `mu_ref` and `sigma_ref` are written to `config.yml` with provenance `computed`, together with the seed, duration, sample rates and git commit.
- Rerunning with the same seed reproduces identical values.
- A test confirms that `preprocess.py` does not import `model.py`.
- I get a short written summary of the numbers.

Note (2026-09-28): done; mu_ref = 7.572449764282774, sigma_ref = 1.131413384422008, seed 42, computed at commit 7191d10 (mV, single node, 600 s kept after 10 s burn-in, 5 s trim per end, ddof 1). src/model.py, tests/test_reference_constants.py (111 tests pass). Seed spread across 42-47: 0.03% for mu_ref, 2.8% for sigma_ref (IMP-005). B5's per-recording rescaling must use the same SD convention (ddof 1).

---

## Stage B: Data and preprocessing (runs in Phase 1)

### [x] B1. download.py
Read: §4.1, §18.
Done when:
- ds003775 v1.2.1 is in `data/`, and 111 subjects and 153 EDF files are found.
- A SHA-256 checksum is recorded for every file.

Note (2026-09-28): done; real download of ds003775 v1.2.1 completed and passed `download.py --verify`. Manifest summary line: "# files=632 manifest_sha256=ba7785725f5f9fb3d74345a5db82276b8550de286dd044fd5896cd821ffa64b5 dataset=ds003775 version=1.2.1 openneuro_py=2026.9.1". Longest absolute path in data/ is 149 characters. One file hit a transient DNS error during the download and was retried automatically by the tool; all size and hash checks passed afterwards. Code in commit 7252c34 (IMP-007).

### [x] B2. Split
Read: §11.1, §17.
Files: `main.py` writes `outputs/split_<seed>.json`.
Note: the seeds are decided (IMP-008): primary 42; extra 43, 44, 45, 46; pilot draw seed 42.
Done when tests show:
- The split is made on all 111 IDs before exclusions, and the result is deterministic.
- There are 78 training and 33 test subjects.
- Each subject's t2 recording is on the same side as their t1.
- 12 pilot subjects (4 of them with t2) are drawn from the training side.
- The pilot subjects are on the training side in all 5 seeds.

Note (2026-09-29): done; real split written with `main.py --make-split` (code commit 3b8b419, IMP-008). A second run reported "unchanged" for all five seeds and rewrote nothing (modification times identical). outputs/ is git-ignored, so the hashes below are the record. The split files must never be redrawn (§11.1).
- SHA-256: split_42.json 90072cd0a1cb5ee276d9320827f8e2eb48104e41584922255607cb71cbd406e4; split_43.json 33536cddb78fa2f048e44de0ce1ae8fc1c9c5ddb71c6fc64dc6156aaf7260382; split_44.json ee6e7144168233b24248621b292f195d90bda8025b0bafa31be0ae80e2ec45ea; split_45.json 7d75b781130cb74dbb03fd267262207f7f43d09d68403bc355ccf7b11a3a1035; split_46.json e9452bd22e64688248af8565caf6aab473c1963a885b86a0d217da49b8e5a435.
- numpy version stored in the files: 2.5.3. Manifest: ds003775 v1.2.1, manifest_sha256=ba7785725f5f9fb3d74345a5db82276b8550de286dd044fd5896cd821ffa64b5.
- Per seed: 78 train, 33 test, 12 pilot; pilot within train, disjoint from test, train and test disjoint. t2 subjects on the test side: seed 42 = 8, 43 = 16, 44 = 11, 45 = 12, 46 = 9 (of 42 t2 subjects in total).
- The 12 pilot IDs, identical in all five seeds: sub-005, sub-008, sub-019, sub-051, sub-056, sub-063, sub-074, sub-081, sub-082, sub-086, sub-088, sub-107.

### [x] B3. Loading and units check
Read: §4.2, §5.1 steps 1–2.
Done when:
- Files are loaded with `units="uV"` and their SHA-256 checksum is recorded.
- The units check applies a 0.5 Hz high-pass to a copy and requires the median SD to fall within 2–200 µV.
- The pipeline halts if more than 5% of recordings fail.
- For sub-010 ses-t1, the EEG is read and only its scans.tsv is skipped.
- Every exclusion is logged.

Note (2026-09-29): done; verified by the pilot report (`python -m src.preprocess --pilot-report`): 16 of 16 SHA-256 checksums OK, 0 of 16 units-check failures (halt: no), median SD after the 0.5 Hz high-pass 4.4 to 11.3 uV. Code commit 0643fd9, fixes 146c21f (IMP-009, IMP-010). Excluding recordings is B6; here only flagged.

### [x] B4. Channel pipeline, part 1
Read: §5.1 steps 3–6, §6.
Done when:
- The P3, PO3, P4 and PO4 labels are verified against `channels.tsv`.
- The bad-channel bounds (5–150 µV RMS, flat-line, saturation) are applied to those four electrodes only.
- The 50 Hz notch filter and its harmonics are applied.
- The zero-phase band-pass from A3 is applied.
- The bipolar pairs P3–PO3 and P4–PO4 are formed.
- Variant flags for §5.2 exist: 0.1 Hz high-pass, and strict rejection.
- Phase 1 saves the band-pass impulse response for both high-pass variants (0.5 and 0.1 Hz) under outputs/ (§5.1 disclosure).

Note (2026-09-29): done, with one item deferred: bandpass_impulse_response exists (A3), but saving it under outputs/ happens when phase 1 is wired, which has not been done. Verified by the pilot report and the noise-like edge report over ten seeds (0.5 Hz, trim 5.0 s: worst deviation after trim 0.13% of SD) and twenty seeds (0.1 Hz, trim 25.0 s: 0.20%), IMP-010, commit 146c21f. The electrode RMS lower bound was changed from 5 to 1 uV by DEV-001 (hash in DEVIATIONS.md); the bounds are now 1-150 uV. Bad electrodes are only flagged here; exclusion is B6 (§12). A print-only 1-s bipolar segment RMS diagnostic for B5 was added (IMP-011).

Note (2026-09-29, cleanup): Phase-1 wiring still to do: the 5% halt that raises or stops, calling write_exclusions, per-variant exclusion file naming, config_sha256 and manifest_summary extraction, saving the band-pass impulse response for 0.5 and 0.1 Hz under outputs/, and restricting allow_all to the phase-1 runner with a test. The lower electrode RMS bound is 1 uV (DEV-001); the 5 uV in the bullet above is superseded (see IMP-013).

### [x] B5. Channel pipeline, part 2
Read: §5.1 steps 7–10, §10.2.
Done when:
- Blinks are detected (0.5–5 Hz copy above 6 robust SD) and corrected with the wavelet method, ±0.25 s around each.
- Gross rejection runs on 1-s segments.
- The EMG screen uses 70–110 Hz power computed **before** the low-pass, flagging segments above 5× the median.
- Every rejection is padded by 0.5 s, and clean stretches under 5 s are dropped.
- The vigilance proxy (alpha/theta ratio per 2-s epoch) is recorded.
- The data are downsampled to 256 Hz.
- Continuous clean segments are output.
- Rescaling to μ_ref and σ_ref uses clean samples only.
- The pipeline runs on 2 pilot subjects, with per-subject caching and a per-step log.

Note (2026-09-29): done. `python -m src.preprocess --segment-report` on all 12 pilot subjects (16 recordings): 0 of 16 units-check failures; blinks corrected in 8 of 16 recordings (largest corrected fraction 0.0474 of the trimmed recording); the rescaled clean samples equal mu_ref and sigma_ref (ddof 1) for every kept recording. Design and choices in IMP-012, code commit fb651ed; cleanup after the independent review in IMP-013 (wavelet window measured and kept, housekeeping, code hash in the cache keys). Wavelet blink correction is only a safeguard against large blinks: at 30 uV the 6 robust SD detector misses most blinks on 1/f (IMP-012). Phase-1 wiring still to do: the 5% halt that raises or stops, calling write_exclusions, per-variant exclusion file naming, config_sha256 and manifest_summary extraction, saving the band-pass impulse response for 0.5 and 0.1 Hz under outputs/, and restricting allow_all to the phase-1 runner with a test.

### [x] B6. Exclusions
Read: §12.
Done when:
- All rules are applied by code, and each exclusion is judged per recording.
- An exclusion log is written.
- The split is never redrawn.

Note (2026-09-29): done. Rules applied per recording in this order: units check, bad electrode, excessive blink correction (new, IMP-013: corrected time of either channel above 10% of the trimmed recording), clean data below 60 s. Pilot result: sub-051 ses-t2 excluded for insufficient clean data (20.0 s); no other pilot recording excluded; the corrected-time guard excludes none of the 16 (subject sub-051 stays in C1, out of C3). The exclusion structure and file writer exist (`apply_exclusions`, `write_exclusions`, tested with tmp_path) but nothing writes outputs/exclusions.json yet. Phase-1 wiring still to do: the 5% halt that raises or stops, calling write_exclusions, per-variant exclusion file naming, config_sha256 and manifest_summary extraction, saving the band-pass impulse response for 0.5 and 0.1 Hz under outputs/, and restricting allow_all to the phase-1 runner with a test.

---

## Stage C: State-space model and UKF (model.py) [x] CLOSED 2026-09-30

Stage C is closed. C1 to C5 are build steps. C4b, C4c, C4d, C6, C7 and C8 are read-only DIAGNOSTICS (no src or config change, nothing adopted): C4b d48d906, C4c e3db3fc, C4d c959251, C6 f40fd8d, C7 d5b32f6, C8 4b134fd. Their combined finding is DEV-005 (status OPEN, not adopted).

**GATE: DEV-005 decision.** DEV-005 is decided after the --pilot comparison (19-D vs candidates A/B on the 12 pilot subjects and synthetic series, section 17 pilot exception), before the full G0 and before the confirmatory run. Pilot-mode runs are not blocked. Stages D to H (build steps) are not blocked.

Note: Before any phase >= 2 gets a real runner: check_prerequisite must also read outputs/gate.json and carry any low_confidence flag forward (CLAUDE.md rule 6, §18.1). Add tqdm progress bars when phases have real work (§18).

### [x] C1. Augmented state-space model
Read: §7.2, §7.4, §7.6.
Done when:
- The 19-D state is 12 neural states plus p₁, p₂, ρ₁, ρ₂, g₁₂, g₂₁ and m.
- The E/I reparameterization holds A·B fixed.
- The mixing observation model has unit diagonal.
- The priors match §7.6, with m truncated to [0, 0.5].
- The M1 variant has 17 dimensions (gains removed).
- The §7.4 reduction order is available as config switches.

Note (2026-09-29): done; src/state_space.py, tests/test_state_space.py (model.py untouched), IMP-014 to IMP-018. Layouts 19 (M2), 17 (M1), and the §7.4 switches compose through the same code path. Observation model B: y_obs = mu_ref + M (y - mu_ref) (IMP-015). m is a raw Gaussian in the state and is clipped to [0, 0.5] only inside observe() (IMP-016). Code commit 6d138d5; the per-state initial variance (seed 42, 600 s, dry run and write at 6d138d5) is stored in config by the second C1 commit (IMP-017). No filter, smoother, Q/R or sigma-point code.
Note for Stage E (E2): the G0 generator must mix the deviations from mu_ref with M(m) and then rescale, to match the fitted observation model (IMP-015).
Note for C3: clipping of the REPORTED recording-level m to [0, 0.5] is decided in C3 (IMP-016).

### [x] C2. Reference UKF and smoother (filterpy)
Read: §7.5, §7.6, §9.3.
Done when tests show:
- The scaled unscented transform uses α = 1, β = 2, κ = 0. At n = 19, the mean weights are 0 at the centre and 1/38 for every other point, and the covariance centre weight is 2.
- There are 4 Heun sub-steps per observation.
- The delay ring buffer is held outside the sigma points, following the filtered/predicted-mean rule, and is filled with the steady-state value after re-initialization.
- The unscented RTS smoother works.
- The divergence definition is implemented.
- Stability monitors catch negative eigenvalues and NaN/Inf.
- The filter tracks the known states and parameters on A2 synthetic data.

Note (2026-09-29): done; src/ukf.py (own NumPy, filterpy 1.4.5 algorithm reproduced exactly), tests/test_ukf.py, IMP-019 to IMP-023. filterpy==1.4.5 is installed as a test oracle only; forward filter and smoother agree with it to rtol 1e-10. Measured on simulated data: mean NIS 1.82 at q = 1e-2; pure-NumPy timing for one 60,000-step recording (234 s): forward 68.8 s (1.15 ms/step), forward with covariances 69.6 s plus smoother 62.5 s = 132 s; min covariance eigenvalue 3.8e-6, NIS 1.84.
Note for C4: the filter replicates filterpy, so NIS uses an S without Q (IMP-019); q acts on NIS only through P (the next step's sigma spread). The rule still picks the q whose NIS is closest to 2.

### [x] C2b. Reviewer fixes to the UKF
Read: §7.5, §9.3; the independent review of C1/C2 (commits 6d138d5, 0ca0de0, a655db1).
Done when:
- The divergence handler catches only numpy.linalg.LinAlgError; config errors and other exceptions propagate.
- The §9.3 monitor includes the offending step's minimum eigenvalue.
- The covariance jitter (1e-9 I) is used for the sigma points when the plain Cholesky fails; the filterpy oracle still agrees at rtol 1e-10.
- delay_substeps >= substeps_per_observation is checked at construction.
- New tests: coupled one-directional data, the buffer protocol against an independent history, the divergence reference point, per-point drift parameters, buffer entries after replace_latest, every reduction-switch combination, and the tightened seed-mean, y1 and gain bounds.

Note (2026-09-29): done; IMP-024 to IMP-028, commit hashes of IMP-019 to IMP-023 filled in (a655db1). Mutation checks: restoring the broad catch, dropping the offending step from the monitor, and measuring the divergence from zero each make the new tests fail; the restored src/ukf.py is byte-identical (cmp). Measured: coupled test filtered g12 8.03 for a truth of 12 (g21 -0.15); the first attempt with p = (220, 260) tripped the 10 SD divergence rule (true node-2 mean y1 at 10.5 SD) and the asymmetry was reduced to (220, 235), band unchanged (IMP-026). Open finding for the pilot and G0: the 10 SD divergence rule is scaled by the small y1 SD, so nodes with a moderately shifted p or a moderate coupling drive are already flagged; look at this before real-data exclusions (§7.5, §12).

### [x] C2c. Divergence reference at the current parameters
Read: §7.5; IMP-021, IMP-026.
Done when:
- The reference of the state-beyond-10-SD rule is the deterministic (coupled) fixed point at the current posterior-mean parameters, cached and refreshed after a 5% of prior SD move, with a counted fallback (IMP-029).
- The 10x multiple and the SD scale are unchanged.
- The state flag alone is suppressed for the first 0.5 s after each (re)initialization (DEV-003).
- The C2b coupled test runs with its original p = (220, 260), band and truth unchanged.

Note (2026-09-29): done; IMP-029 and DEV-003. Coupled test with p = (220, 260): filtered g12 8.02, g21 -0.28, smoothed 8.76 and -0.80. False-flag measurement on 20 healthy synthetic series (p in [150, 300], gains in {0, 5, 12, 27}): 3 flagged (15, 18, 19) with the reference change alone, 2 flagged (15, 19) with the start-up exemption; series 15 and 19 (g = 27 both ways) have TRUE trajectories 13.8 and 36.1 SD from their own fixed point, so the rule flags strong two-way coupling (DEV-003 open risk, no regime exempted). Mutation checks: reference reverted to the prior parameters, reference measured from zero and exemption length 0 each make the new tests fail; restored files are byte-identical (cmp).

### [x] C3. Two-pass parameter handling
Read: §7.5 (parameter handling), §8.3, §10.2.
Done when:
- Recording-level parameters are the mean of the smoothed trajectory after burn-in.
- The windowed pass uses 2-s windows, re-initializes the 12 neural states, holds the parameters fixed, and discards 0.5 s of burn-in.

Note (2026-09-29): done; src/passes.py (new), tests/test_passes.py, tests/sim_data.py, small changes to state_space.py (Layout.fixed_params, make_fixed_layout) and ukf.py (DivergenceReference at fixed parameters), IMP-030 to IMP-036; commit hashes of IMP-024 to IMP-029 and DEV-003 filled in. Pass 1 carries the parameter mean and covariance across segments with gap inflation (IMP-030); recording-level m is the mean of clipped values (IMP-031); estimator burn-in measured at 6 s (IMP-033); windows start from the literal section 7.6 state, deviation at 0.5 s at most 1.17 SD (IMP-034); recording_diverged = OR of both passes (IMP-035). Measured recovery (bands fixed first): g12 10.47 [12], g21 -1.42 [0], p1 232.7 [220], p2 259.8 [260], m 0.328 [0.35], log rho -1.93 and -1.89 [-1.91]; on 8 segments of 7500 samples g12 12.11 [12], g21 4.73 [5]. Runtime for 60,000 samples on one core: pass 1 105.6 s, pass 2 86.8 s (a 12-D window costs 1.51 ms per sample, more than expected). Mutation checks (parameters left in the window state, windows crossing a gap, burn-in not discarded in either pass, carry replaced by restart) each make the tests fail; restored files byte-identical (cmp). Leakage control: gains re-estimated inside the windows give a mean g12 of 10.0 (range 5.75 to 13.54) against 10.47 fixed.
Deferred (IMP-036): the section 11.3 frozen-model gain estimator (base + frozen PySR residual) needs a residual hook in the predict step and belongs to G4. The pass-1 filtered-gain mean is exposed as the section 9.2 null-gate estimator.

### [x] C4. Q/R rule
Read: §7.5, §7.6.
Done when:
- R = 0.25·σ_ref²·I₂.
- q is chosen from an 8-value log grid spanning 1e-4 to 1e-1 by NIS closest to 2.
- The real-data mode uses 20 random training subjects per split seed.
- The G0 mode uses the 20-series synthetic tuning set.

Note (2026-09-29): done; src/tuning.py (new), tests/test_tuning.py, small edits to ukf.py (FilterResult.P_last) and passes.py (run_pass1 forward_only, per-segment nis and nis_keep), config ukf.qr_rule additions, IMP-037 to IMP-043; commit hashes of IMP-030 to IMP-036 filled in (c1d2176). tune_q(recordings, cfg, cache_dir, n_jobs) is the core for both modes; run_real does the gate check, the training-only draw (offset 1000 + split seed) and writes outputs/qr_<seed>.json (pilot: results/pilot/); it is not wired into main.py. Measured forward-only pass 1 on 60,000 samples: 60.8 s on one core (full pass 1 134.6 s in the same script), about 9.8 h serial and 2.4 h on 4 workers for the 576 distinct (recording, q) runs. Mutation checks (test subjects in the draw, grid off by one in two ways, NIS with Q inside S, burn-in not skipped, divergent recordings dropped per q) each make the tests fail; restored files byte-identical (cmp). Smoke test on 3 pilot recordings (sub-074, sub-019, sub-081; not a result, nothing committed): every recording diverged (10 SD state rule, DEV-003 open risk) at nearly every q, 0 of 3 recordings usable at all 8 q, so the rule refused to select. Where a recording did not diverge or only some segments were dropped, the mean NIS over kept samples was 3.7 to 7.0, above 2. Real-data tuning is therefore blocked until the divergence rule on real data is resolved (pilot, G0); this is a finding, nothing was changed.

### [x] C4b. Read-only diagnostic of the divergence rule on real data
Read: §7.5; IMP-021, IMP-026, IMP-029, DEV-003, IMP-037 to IMP-043.
Done when: tools/c4b_divergence_diagnostic.py runs on the 12 pilot recordings (ses-t1) at q = 1e-4, 1e-3, 1e-2, 1e-1 with the state-SD flag disabled inside the script only, and prints the per-state exceedance, parameter trajectory, NIS and failure tables next to a synthetic comparison.

Note (2026-09-30): done; script only, no src or config change, no rule or threshold changed, nothing written under outputs/, results not committed as findings (the tables were reported in the session). The flag is disabled by replacing ukf._first_divergence in the script's own worker processes (NaN/Inf and PD checks stay on). Headline (pilot only, 12 recordings): no NaN/Inf/PD failure in 72 runs; only the y1 states exceed 10 SD (up to 39.5 SD at q = 1e-2, against 15 on synthetic); mean NIS at q = 1e-2 is 1.39 to 2.84 with the flag off; filtered p, log rho and gains run 9 to 26 prior SD from the prior at q >= 1e-3, against 0.8 to 3.5 on synthetic. No option (a) to (d) was implemented; the decision is open.

### [x] C4c. Read-only diagnostic: parameter freedom versus the y1 excursions
Read: §7.5; DEV-003, IMP-021, IMP-026, IMP-029, IMP-037 to IMP-043; tools/c4b_divergence_diagnostic.py.
Done when: tools/c4c_parameter_freedom_diagnostic.py runs the 12 pilot recordings (ses-t1) at q = 1e-2 and 1e-3 under V0 baseline, V1 parameters frozen at the prior, V2 random-walk factor x0.1, V3 x10 (all applied inside the script's own processes), with the state-SD flag disabled, and prints the y1 exceedance, parameter excursion, NIS, y1 posterior-SD and spectrum tables.

Note (2026-09-30): done; script only, no src, config or rule change, nothing under outputs/, results not committed as findings. Headline (pilot only, 12 recordings): freezing the parameters (V1) removes most of the >10 SD exceedance at q = 1e-2 (median 0.021 to 0.003) but the y1 excursions reach 62 SD and the median NIS rises from 2.06 to 3.26; the slower walk (V2) changes little and the faster walk (V3) lowers the exceedance (0.008) while parameters move further (up to 56 prior SD); the y1 posterior SD is about the prior SD (0.9 to 1.3 x) and a 10 posterior-SD test would still flag 5212 of 1,363,200 (sample, node) pairs (rule: 25,895); at the median filtered parameters the model output has a flatter aperiodic slope (-0.8 to -1.0) than the observed channel (-1.6 to -1.7). No rule change was implemented; the decision is open.

### [x] C4d. Read-only diagnostic on synthetic data: does 1/f background reproduce the real-data behaviour?
Read: §7.5; DEV-003, IMP-026, IMP-029; tools/c4b_*, tools/c4c_*, tests/sim_data.py.
Done when: tools/c4d_aperiodic_diagnostic.py runs the six C4b synthetic series as S0 plain, S1 + 1/f (slope -1.6, 50% of the 1-45 Hz variance) and S2 (80%), the noise added before the band-pass and the rescaling, at q = 1e-2 and 1e-3 with the state-SD flag disabled, and prints the exceedance, parameter excursion, NIS and spectrum tables plus the y1/y2 correlation of 3 pilot recordings.

Note (2026-09-30): done; script only, no src, config or rule change, nothing under outputs/, results not committed as findings. Headline (6 synthetic series, pilot values from C4b/C4c): S1 reproduces the real-data pattern at q = 1e-2 (y1 beyond 10 SD 0.038 against 0.021 real, max 25.5 SD against 27, p excursion 17 prior SD against 21, NIS 2.56 against 2.06); S0 does not (0.000, 5.8 SD, 0.9 prior SD, 1.82); the spectra match the real slope (S1 -1.58, real -1.58) with a higher alpha share (0.47 against 0.39 real median); at q = 1e-3 the real recordings lie between S1 and S2. Real filtered y1 and y2 are positively correlated (0.6 to 0.73) but there is no synthetic baseline for that correlation. No rule change was implemented; the decision is open.

### [x] C5. Numba port and validation (§21.2 #4)
Note (C2c, c): time the divergence-reference refresh in the C5 benchmark. It runs on about 11 to 14 percent of steps (3 to 8 ms per fixed-point solve, pure Python around model.steady_state); it is not part of the Numba port yet.
Read: §7.5, §16.5.2.
Done when:
- The filterpy Cholesky convention (upper-triangular) is confirmed from the installed source.
- Means and covariances match filterpy at rtol 1e-8 and atol 1e-10.
- Downstream statistics are unchanged.
- A timing comparison is reported.
- The fallback path to filterpy exists.

Note (2026-09-30): done; src/ukf_numba.py (new), tests/test_ukf_numba.py (37 tests), backend argument in ukf.run_filter/run_smoother, config ukf.numba.enabled false (placeholder, default stays NumPy) and rtol/atol_smoothed_cov, IMP-044 to IMP-050, DEV-004 (smoothed covariances only: rtol 1e-6, atol 1e-8; everything else and all downstream statistics stay at rtol 1e-8, atol 1e-10). filterpy's Cholesky convention (scipy.linalg.cholesky default, upper, 'U'*U = (n + kappa) P', rows of U) confirmed from the installed 1.4.5 source. Speed-up on 60,000 samples, one core: forward 3.02 s against 68.8 s (23x), smoother 2.77 s against 62.5 s (23x), pass 1 forward-only 3.1 s against 60.8 s (20x), full pass 1 5.96 s against 134.6 s (23x), pass 2 3.6 s against 86.8 s (24x). The reference tracker (fixed-point solve) was ported too: 361 refreshes at 0.85 ms = 9.6% of the run (Python would be 58%). cache=True works on Windows (fresh-process warm-up 1.4 s; first compile about 8.6 s + 2.4 s); stale-entry caveat when only model.py changes (IMP-048). Mutation checks (Cholesky orientation, dropped covariance centre weight, wrong buffer entry replaced, fastmath=True) each fail a test; restored file byte-identical (cmp). The pilot still decides whether ukf.numba.enabled becomes true (§21.2 #4).

### [x] C6. Read-only diagnostic on synthetic data: gain and m recovery with 1/f background
Read: §7.5, §9.1 to §9.3; DEV-003, DEV-004, IMP-026, IMP-029, IMP-033; tools/c4d_*, tests/sim_data.py.
Done when: tools/c6_gain_recovery_diagnostic.py runs 6 series per cell (4 coupling levels x C2 and a null; 234 s, 4 segments of 70/50/40/60 s) as S0, S1, S2 at q = 1e-2 and S1 at q = 1e-3 with the Numba backend, the state-SD flag disabled (config copy inside the script), and prints gain recovery (smoothed and filtered-gain estimators), level and null criteria, m recovery and contraction, divergence with the standard rule ON, and runtime.

Note (2026-09-30): done; script only, no src, config or rule change, nothing under outputs/, results not committed as findings (raw JSON in the scratchpad). Seeds: sim 30000 + 100 x cell + series, 1/f 40000 + 10 x sim seed + channel, m ~ U(0.1, 0.4) from 50000 + sim seed. Headline (6 series per cell): S0 recovers gains within 1 to 3% (levels 2 to 4) and the nulls stay below 0.3; with 1/f at q = 1e-2 the gain is lost (S1 estimates of about 55 to 130 whatever the truth, relative error 370 to 2700%, null |g| median 51 against delta 1.08; S2 alike; 0 of 6 series ok in every cell) and the standard rule drops every segment (24 of 24); at q = 1e-3 (S1) the gains are attenuated to about half at levels 2 to 4 (error 38 to 53%, level 1 median 14% with a range to 760%), the nulls sit at 0.7 to 0.9 (1 of 6 series below delta), m is biased up by 0.15 to 0.27 and its contraction falls to 0.40 and 0.36 at levels 3 and 4; contraction of the gains stays 0.83 to 0.94 even where the estimate is wrong by a factor 10, so it does not flag the failure. No rule change was implemented; the decision (DEV-004b) is open. Runtime per series with Numba (full pass 1, 4 workers in parallel): 5.5 to 6 s (S0), 7.4 to 8.8 s (1/f).

### [x] C7. Compare candidates for the 1/f aperiodic background on synthetic data (A: OU coloured observation noise, B: random-walk offset, C: adaptive R)
Read: §5.1, §7.2, §7.4, §7.5, §9.1, §9.3, §20, §21.2; DEV-003, DEV-004, IMP-026, IMP-029; tools/c4d_*, tools/c6_*, src/state_space.py, src/ukf.py, src/ukf_numba.py.
Done when: the candidates run through the unchanged src/passes.py on fresh-seed synthetic series (S0, S1, S2 x L1-L4 + null x q = 1e-2 and 1e-3, 6 series per cell, plus a same-seed 19-D control), every cell is reported with both gain estimators, the stability monitors, mean NIS, the divergence counts with the standard rule ON and the runtime, and the fixed criteria of the plan are evaluated (c1 median gain error <= 15% at L2-L4, c2 nulls within delta = 1.08, c3 m contraction >= 0.5, c4 >= 5 of 6 series undiverged with the rule ON, c5 no NaN/PD failure; S0 not worse than the control by more than 2 percentage points). No src, config or rule change; no real recording.

Note (2026-09-30): done; tools/c7/ext_ukf.py (edited copies of the Numba kernels; extra states appended after the parameters), tools/c7/aperiodic_prior.py, tools/c7_candidate_comparison.py, tests/test_c7_experimental.py (12 tests: the copy equals src.ukf_numba with no extra state; A and B equal an independent NumPy filter and smoother at rtol 1e-8; two mutation checks fail as they should, restored file byte-identical). IMP-051 added. Nothing under src/ or config.yml changed; results not committed as findings (raw JSON and tables in the scratchpad). Result: NO candidate is viable. A and B remove all divergence (0 of 720 segments dropped in every S1/S2 cell at both q, against 120 of 720 for the control at q = 1e-2) and keep the S1 nulls inside delta at both q, and they cut the level-median gain error over L2-L4 (S1 and S2, both q, both estimators) from 0.36-16.2 (control) to 0.16-0.76 (A) and 0.08-0.40 (B), but in every (variant, q) at least one of L2-L4 stays above 15% (best: B, S2, q = 1e-3, 14 / 14 / 22% at L2 / L3 / L4; smoothed estimator); m contraction falls below 0.5 at L4 for q = 1e-3; A and B fail c2 in S2 at q = 1e-2; B is worse than the control on S0 at q = 1e-2 (L1 error 7-8% against 3-6%). C (adaptive R) does not help at q = 1e-2 (identical to the control) and only partly at q = 1e-3 (level-median error 23-65% in S1/S2, 8 of 30 S1 and 13 of 30 S2 recordings diverged with the rule ON), and its NIS is about 2 by construction. Stability at 21-D: over all 720 runs no NaN/Inf, no covariance-not-PD failure, 0 negative-eigenvalue steps, 0 jitter fallbacks, smallest eigenvalue 2.6e-7 (q = 1e-3) to 4.7e-6. Mean NIS of A and B in S1/S2 is 0.44 to 0.62 (control 2.1 to 5.8), so the section 7.5 NIS rule would be out of band for them.
Changes from the plan: (1) the estimator fit range is 1-40 Hz, not the planned 1-45 Hz (to stay clear of the 45 Hz low-pass transition). (2) tau sits at the 0.32 s upper bound in 90% of the S1 fits and 95% of the S2 fits and the fitted s2 is 1.0 to 1.7 times the channel variance (in-band share of the fit 0.48 for S1, 0.70 for S2, planted 0.5 and 0.8), so tau must NOT be read as a measured time constant; the bound was not widened and the estimator was not reparametrised after seeing its output (decision of the reviewer). (3) The C6 tables were re-run in .venv first (identical, IMP-051). (4) tests/__init__.py was not added (it breaks collection; IMP-051). (5) The control arm is the extended kernel with no extra state, tested equal to src.ukf_numba to 1e-12. The decision on DEV-005 is open.

### [x] C8. Read-only diagnostic on synthetic data: is the q = 1e-3 gain attenuation caused by the gain prior?
Read: §7.3, §7.5, §7.6, §9.1 to §9.3; DEV-004, IMP-051; tools/c7/*, tools/c7_candidate_comparison.py.
Done when: tools/c8_gain_prior_diagnostic.py runs arms control, A_ou, B_walk x S0, S1, S2 x L1-L4 + null (6 series per cell, C7 seeds) at q = 1e-3 with the gain prior SD x1, x3, x10 (variant K: config copy, all four places; variant P: prior only, Q and refresh threshold unchanged) and evaluates c1 to c5, printing gain error, estimated/true ratio, posterior SD of g, m contraction, divergence, NIS and stability. No src or config change.

Note (2026-09-30): done; tools/c8_gain_prior_diagnostic.py, tests/test_c8_gain_prior.py (6 tests, two mutation checks), IMP-052; IMP-051 commit filled (d5b32f6). Nothing under src/, config.yml or outputs/; results not committed as findings (raw JSON in the scratchpad). x1 reproduces C7's q = 1e-3 rows exactly (270 runs, difference 0.0). Result (6 series per cell): no (arm, S1|S2, setting) passes c1; the first failing criterion is c1 everywhere. c1 never passes, so the case "c1 passes only where c2 fails" does not occur. Widening helps only a little and only under K: A_ou and B_walk level-median error at L2-L4 (S1, smoothed) 18/30/31 and 19/40/20 at x1 against 13/27/29 and 10/33/11 at K10; variant P (prior only) is almost identical to x1 (18/30/30 and 18/39/10 at P10). The control's S1 nulls degrade with K (5/6 at x1, 3/6 K3, 1/6 K10) and its L1 error rises (13% to 73%); A and B keep 6/6 nulls at every setting. Stability: no NaN/Inf/PD failure, no negative eigenvalue or jitter in any of 1350 runs (smallest eigenvalue 2.6e-7). Wall time 4122 s, 4 workers. No prior change is proposed; adoption would be a separate DEVIATIONS step.

---

## Stage D: Regression machinery (regression.py)

Session split (2026-09-30): the session's 'D1' commit holds PLAN steps D1, D2 and D4 and the bookkeeping part of D5 (no Julia); the session's 'D2' (PySR install, smoke checks, determinism and turbo check, PySR recovery test, pilot 3-refit mechanics on synthetic rows) holds PLAN step D3 and the fitting part of D5.

### [x] D1. Derivative estimators
Read: §8.3.
Done when:
- The weak-form estimator (Gaussian-window test function with compact support; 60, 100 and 160 ms options) is implemented.
- The TV baseline with its 5-value grid is implemented.
- Both are tested against analytic signals.

Note (2026-09-30): done; src/regression.py (new), tests/test_regression.py, tests/planted_sim.py (test-only simulator with the planted product), tools/d1_estimator_report.py, config residual.* leaves, IMP-053, IMP-054. Weak form: sigma = h/3, phi = g - g(h), K = 7 / 12 / 20 taps at 60 / 100 / 160 ms; matched and literal subtraction both implemented (config residual.subtraction_form, default matched by the user's decision); TV by lagged diffusivity on the approved 5-value grid. OPEN CHOICE for Stage I (user, 2026-09-30): matched (M, default) versus literal (L) subtraction is decided on FILTERED states at the pilot, not on the D1 numbers (noise-free test simulator: M shrinks the recovered coefficient, c_hat / c_true 0.89 / 0.69 / 0.36 at 60 / 100 / 160 ms, L 1.07 / 1.11 / 1.14; these are mechanics, not findings; IMP-053). M with smoothed inputs is not to be implemented. TV is for the pilot set only (about 0.9 s per window).

### [x] D2. Residual target and rows
Read: §8.2 (inputs, z-scoring), §8.3, §8.4.
Done when:
- The residual is dy₄/dt minus the base-model prediction at the recording-level gains.
- Rows come from the 2-s windows after burn-in, with both directions pooled.
- z-scoring uses the fit fold's mean and SD, and those constants are stored.

Note (2026-09-30): done in the same commit; window_rows / recording_rows / fit_zscore / design, IMP-055. Rows lose K samples at each end of a window (360 per window and node at 100 ms, not 384); only x_smooth[:, :12] is read, so the DEV-005 layout decision does not matter here.

### [x] D3. PySR wrapper
Read: §8.2, §16.5.
Done when:
- The PySR version is pinned, and operators, inputs, parsimony and timeouts come from config.
- Real fits run in parallel. A serial fixed-iteration mode exists for determinism checks.
- One warm session is reused for all fits.
- Each fit gets a 50,000-row subsample with equal rows per subject.
- PySR's batching option is not used.
- turbo stays off until verified.

Note (2026-09-30): done; PySR 2.6.0, SymbolicRegression.jl 2.5.1, Julia 1.11.9 (installed by juliapkg under .venv/julia_env, depot .venv/julia_depot; pinned in config pysr.version), IMP-059 to IMP-062. src/regression.py: load_pysr, model_kwargs, fit_model, run_fit, front_entries, evaluate_equation, fronts_identical / fronts_equivalent; tools/d2_smoke.py, tools/d2_pysr_reports.py; tests/test_regression_pysr.py (marker pysr, excluded by default in pytest.ini: run with -m pysr). Checks: 8 Julia threads, both import orders, 30 s timeout overshoot 0.87 s on 5,000 rows (7 to 20 s at 50,000 rows), affinity masks honoured, smoke fit finds the product at complexity 3, float64 costs 1.6x float32, serial turbo-off run bitwise repeatable, turbo on/off NOT equivalent on 3 of 3 seeds (1.5 to 1.8x faster): turbo stays false.

### [x] D4. Pareto selection and term signatures
Read: §8.2, §8.4.
Done when:
- The split is 80/20 by subject. For G0 it is the last 20% of windows.
- The simplest equation within 5% of the minimum validation loss is selected and frozen.
- A bare constant counts as "no term".
- The sympy signature tests pass: tanh(1.02·x + 0.01) → tanh(x), and denominators are kept literally.

Note (2026-09-30): done in the same commit (selection on stubbed fronts; PySR itself not yet run); IMP-056 to IMP-058. sympy 1.14.0 installed in .venv (pin at Stage L).

### [x] D5. Refit ensemble
Read: §8.2, §15 (C2).
Done when 25 half-sample refits at 600 s each run with the same selection rule.

Note (2026-09-30): done in two parts. D1 commit: subject draws, seeds and the recurrence count (draw_ensemble, signature_recurrence, IMP-056, IMP-058). D2 commit: run_fit and run_ensemble with real PySR fits, mechanics checked on synthetic rows only (pilot: 3 refits at 600 s; pytest marker test with 25 s timeouts); the 25-refit ensemble itself (4.2 h) has not been run and needs real rows (Stage K). IMP-062.

---

## Stage E: Synthetic gate (synthetic_gate.py)

Note (2026-09-30, from D2): the section 9.1 planted basis (potential / SD, NOT centred) has a mean of about 24 and an SD of about 12 in the test simulator, so after z-scoring the inputs it is mostly linear terms plus a product. When G0 recovery is scored, check that it is not trivially a linear term (the D2 marker test requires the product signature on an uncentred basis). No change to section 9.1.

**GATE: DEV-005 decision.** DEV-005 is decided after the --pilot comparison (19-D vs candidates A/B on the 12 pilot subjects and synthetic series, section 17 pilot exception), before the full G0 and before the confirmatory run. Pilot-mode runs are not blocked.

### [x] E1. Operating-regime grid
Read: §9.1.
Done when:
- The 12 × 8 × 5 grid over (p, input-noise SD, observation noise) is built.
- Matching uses scale-free features only (alpha peak, relative alpha power, specparam slope), taken from training subjects only.
- The regime (noise-driven or limit cycle) is reported.

Note (2026-09-30): done; src/synthetic_gate.py (grid, features, matching, regime label, E1 report: python -m src.synthetic_gate --grid-report --pilot), tests/test_synthetic_gate_grid.py (20 tests, 6 mutation checks), config g0.seeds / filter / filter_options / regime / features, specparam 2.0.0rc7 installed, IMP-063. Full suite 638 passed, 1 skipped (C1 writer test), 3 deselected. Findings: all 480 grid points are limit-cycle; the 15 pilot recordings all match the grid's edges (p 120 or 138, input-SD factor >= 1.10, noise share <= 0.5); axes unchanged, decision with the user. Plan additions agreed with the user: src/ukf_ext.py (E4), E5 pilot driver and DEV-005 comparison report (new step after E4).

### [x] E2. Series generator
Read: §9.1, §9.2.
Done when:
- Series are generated at 2,048 Hz.
- There are 4 coupling levels, each with a planted product residual at 50% RMS.
- Null A and Null B are generated. Null B has a 0–20 ms lag and a 30–50% input-variance share.
- 1/f plus white noise is added, and m ~ U(0.1, 0.4).
- Each series is scaled to µV, run through the real preprocessing, and rescaled.

Note (2026-09-30): done; src/synthetic_gate.py (planted kernel, run_planted, null_b_input, generate_series, tuning_set, tune_g0_q for 19D, time_one_series), tests/test_synthetic_gate_series.py (21 tests, 8 mutation checks), config g0.seeds.arm_codes / substreams, series_duration_s, generation_burn_in_s, backend, tuning_level_index, n_tuning_series, IMP-064 and IMP-065. One series (19-D, Numba) takes about 1 s to generate and 12.9 s for both passes with the state-SD flag off; with the standard rule ON that series diverged at step 187 and would count as failed. Realised planted RMS ratio 0.46 to 0.53 at L1 to L3 but 0.76 to 0.84 at L4. Run `python -m src.synthetic_gate --time-series --pilot`.

### [x] E3. Preprocessing-gate artifacts
Read: §5.2.
Done when:
- Blinks, EMG bursts and drift are injected at the stated settings.
- The artifact-only null uses bilateral, near-zero-lag artifacts.

Note (2026-10-01): done; src/synthetic_gate.py (artifact_signal, blink_wave, emg_carrier, generate_series artifacts / arm artifact_null, preprocessing_gate_set, null_pass, preprocessing_bias_verdict, time_gate_series), tests/test_synthetic_gate_artifacts.py (20 tests, 9 mutation checks), config g0.preprocessing_gate.* leaves, IMP-066 (E2 review decisions) and IMP-067. Run `python -m src.synthetic_gate --time-gate-series --pilot`. One gate series takes about 1.5 to 1.8 s to generate and about 15 s with both passes (flag off); the 30%-of-SD EMG bursts are mostly not removed by the 5x screen (7 and 8 of 240 segments rejected). The PySR part of the bias verdict (NRMSE) and the series' estimates come in E4 / E5.

### [x] E4. Gate scoring and gate.json
Read: §9.1–§9.4, §5.2, §18.1, §20.
Note (C2b): the §9.3 identifiability check must report the posterior contraction of m. In observe() m is clipped to [0, 0.5] (IMP-016) and the prior spread (0.2 +- 0.66 at the sigma points) puts the two m-direction sigma points on the clip boundaries, so the filter sees less information about m; a poor contraction for m is a finding to log, not a reason to change the clip silently.
Note (C2c, a): in the C2b coupled test (12 s, g12 = 12, g21 = 0, p = (220, 260)) the filter recovered g12 = 8.0 (smoothed 8.8), about 30 percent low, probably prior shrinkage toward 0 (prior SD 10.8, posterior SD 1.6). That is a likely risk for the G0 +/-15 percent gain-error criterion; look at it on the positive-control series before trusting that criterion.
Note (C2c, b): the state-SD divergence rule flags strong two-way coupling (g near 27 both ways) because the SD scale comes from the uncoupled simulation (DEV-003 open risk). Check at G0 on synthetic data whether the strongest coupling level triggers it and decide then; no change was made.
Done when:
- The positive-control rule is implemented: median NRMSE ≤ 0.25 and median gain error ≤ 15%, from the second-weakest level up.
- The null δ-band rule is implemented, with δ = half the weakest level.
- Contraction must be ≥ 50% (median).
- Stability flags are set.
- Null or stability failure means a hard stop. Positive-control or contraction failure sets `low_confidence`. Pilot mode never stops.

Note (2026-10-01): done; src/ukf_ext.py (promoted copy of the C7 A and B filters, not adopted), src/synthetic_gate.py (gate_filters, filter_context, evaluate_series, tune_g0_q_option, positive / null / contraction / stability verdicts, gate_flags, gain_profile, choose_parsimony, linear_floor_nrmse, equation_nrmse, option_report, build_gate_document, write_gate), tests/test_ukf_ext.py (12) and tests/test_synthetic_gate_scoring.py (26), config ukf.aperiodic.*, g0.pass.from_level_index, g0.gate_schema_version, IMP-068 and IMP-069. Pass 2 for A and B is 14-D (no kernel change). 16 mutation checks, restored byte-identical. No PySR was run. The E5 driver supplies the PySR results (has_term, nrmse) and assembles the per-option documents.

### [ ] E5. Pilot driver and DEV-005 comparison report
Read: §9.4, §17; DEV-005, IMP-063 onward.
Done when: `python -m src.synthetic_gate --pilot` runs 20 positive, 20 Null A and 20 Null B per filter option (19D, A, B) on the same cached series, UKF-only first (gains, contraction, stability, divergence, null delta-band, NIS, runtime), then STOPS for review before any PySR stage; the PySR stage (default parsimony 0.01) and the parsimony / derivative-estimator measurements follow only on approval. One series per option is timed end to end first; an all-options estimate above about 10 h stops the run. The pilot never writes outputs/gate.json.
Note (2026-10-01): UKF-only stage done, PySR stage NOT started and DEV-005 NOT decided; the checkbox stays open until the PySR stage is approved and run. src/synthetic_gate.py (run_pilot, time_option, estimate_runtime, option_summary, option_verdicts, check_pairing, format_comparison, write_pilot_json), tests/test_pilot_driver.py (14 tests, 12 mutation checks), config g0.pilot.*, IMP-070. Run `python -m src.synthetic_gate --pilot [--estimate-only]`; results in results/pilot/ (g0_filter_comparison.json, g0_pilot_timing.json, gate_<option>.json). Timing: conservative all-options estimate 0.41 h (limit 10 h); the real run took about 20 min per option on 4 workers. Finding: the section 7.5 rule refused for all three options on the pilot tuning set under the standard divergence rule (0 of 20 usable), so the q of the comparison is a DIAGNOSTIC q from the same rule with the state-SD flag disabled (IMP-070). Open: the first-run tuning times (5.3, 3.8, 3.7 min) are not in the final JSON (the tuning came from the cache).

### [x] E6. Operating-point diagnostic (synthetic only, diagnostic)
Question: is the E5 failure caused by the matched operating points or by the pipeline? tools/e6_operating_point_diagnostic.py (no src or config change; results/pilot/e6_operating_point.json), tests/test_e6_operating_point.py (12 tests, 13 mutation checks), IMP-071. Arms: (a) E5 series reused, (b) fixed C7-style operating point with a light chain, (c) the same with the real chain. No PySR, no gate.json, no DEV-005 decision.

### [x] E7. Zero-residual diagnostic (synthetic only, diagnostic)
Question: does the planted product residual explain the remaining gain error? tools/e7_zero_residual_diagnostic.py (filters A and B only, 19D dropped; residual c = 0 inside the script; arms (i) the E6 fixed point and (ii) the E5 matched points; results/pilot/e7_zero_residual.json), tests/test_e7_zero_residual.py (8 tests, 9 mutation checks), IMP-072. No PySR, no gate.json, no DEV-005 decision.

---

## Stage F: Baseline (baseline.py)

### [ ] F1. VAR baselines
Read: §13, §10.2.
Done when:
- M0 has its order (1–32) chosen by 5-fold subject-wise cross-validation, with coefficients from pooled least squares, then frozen.
- M0b is refitted on each test recording.
- Both are scored on exactly the same samples as the UKF models.

---

## Stage G: Robustness and scoring (robustness.py)

### [ ] G1. C1 scoring
Read: §10.2, §12, §13, §15.
Done when:
- M1–M3 are scored with a continuous forward filter, excluding burn-in.
- Matched sets are used.
- Divergence counts are reported per variant.
- The sensitivity analysis counts M3-only divergences as M3 losses.

### [ ] G2. Bootstrap and Holm correction
Read: §11.4, §14, §15.2.
Done when:
- There are 10,000 subject resamples from a pre-drawn index matrix, vectorized.
- Resampling is clustered when t1 and t2 appear together.
- The Holm family is the four bullets of §15.2.

### [ ] G3. C2 aggregation
Read: §15.
Done when the ≥ 70% signature recurrence and the "C1 passes in ≥ 4 of 5 seeds" rule are both computed.

### [ ] G4. C3 ICC
Read: §11.3, §15.
Note (C3): the section 11.3 estimator (frozen base + PySR residual model) is deferred here: it needs a residual hook in ukf/state_space predict. Until then passes.Pass1Result.gain_estimate gives the base-model filtered-gain mean (the section 9.2 null-gate estimator).
Done when:
- ICC(3,1) is computed per direction, with an F-based confidence interval.
- It is computed for all pairs and for test-only pairs.
- The vigilance-adjusted version is reported alongside.

### [ ] G5. C4 free-run
Read: §10.2, §15.
Done when:
- The free-run is stochastic, with 20 realizations.
- Welch spectra use 2-s Hann segments on a 0.5-Hz grid over 1–45 Hz.
- Errors are computed per segment.
- The upper CI bound of the ratio must be ≤ 1.20.
- The scoring condition is checked.

### [ ] G6. Diagnostics
Read: §14.
Done when wPLI, imaginary coherency and the AAFT test (10 recordings × 50 surrogates) are computed.

### [ ] G7. §5.2 sensitivity analysis
Read: §5.2.
Done when the M2 UKF has been run on 20 training subjects under each of the 4 preprocessing variants.

### [ ] G8. summary.json and equations.tex
Read: §18.2.
Done when both files are written, including when the residual is absent or `low_confidence` is set.

---

## Stage H: Figures (figures.py)

### [ ] H1. Nine figures
Read: §19.
Done when:
- All 9 figures are PNGs at 300 DPI.
- A failed stage produces a labelled placeholder figure.
- The script reads only `results/` and `outputs/gate.json`.

---

## Stage I: Pilot (§17)

**GATE: DEV-005 decision.** DEV-005 is decided after the --pilot comparison (19-D vs candidates A/B on the 12 pilot subjects and synthetic series, section 17 pilot exception), before the full G0 and before the confirmatory run. Pilot-mode runs are not blocked.

- [ ] Before the pilot, compute the exact observed ICC that corresponds to a CI lower bound of 0.40 at n = 42 (§15, C3).
- [ ] Before the pilot, run the binomial check for any "X% of subjects" criterion at n = 33 (§11.5).
- [ ] Run `--pilot` through phases 1–4.
- [ ] Resolve the §21.2 items and log each one in `DEVIATIONS.md` §4:
  - hard rejection versus noise-adaptive R;
  - the affinity mask;
  - the worker count (4, 6 or 8);
  - Numba validation.
- [ ] Make the pilot choices:
  - derivative estimator;
  - sub-steps (1, 2 or 4);
  - 256 versus 512 Hz;
  - parsimony penalty from {0.003, 0.01, 0.03};
  - δ;
  - turbo;
  - burn-in convergence.
- [ ] Record per-stage timings and rescale the §16.3 budget.

## Stage J: Lock placeholders

- [ ] Finalize every PLACEHOLDER in `config.yml`, and log each one in `DEVIATIONS.md` §3. This must be done **before** any full-cohort result is inspected.

## Stage K: Full run

- [ ] Phase 1: preprocessing, then G0 (can run alone over one or two nights)
- [ ] Phase 2
- [ ] Phase 3
- [ ] Phase 4

## Stage L: Release files

- [ ] `requirements.txt`, pinned with `pip freeze`, including the exact PySR and filterpy versions (§18.4)
- [ ] `README.md` (§18.4)
- [ ] `LICENSE` (optional)
