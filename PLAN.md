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
- The impulse response is saved and documented (the §5.1 disclosure).

Note (2026-09-28): done; src/preprocess.py (bandpass, bandpass_sos, bandpass_impulse_response, downsample), tests/test_preprocess.py. Butterworth SOS via sosfiltfilt, order 4 per pass, so the nominal 0.5 and 45 Hz cutoffs are the -6 dB points; FFT downsampling with a 2 s odd reflect-pad (IMP-004). bandpass() is for whole continuous recordings before segmentation, never for short segments; inputs of padlen samples or fewer raise by design. Trimming or masking the filtered edges is decided in B4 using the printed edge-transient numbers; nothing is trimmed in A3. The impulse response is returned by a function only; saving it is deferred to phase 1 (see the B4 bullet).

### [ ] A4. Reference constants μ_ref and σ_ref
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

---

## Stage B: Data and preprocessing (runs in Phase 1)

### [ ] B1. download.py
Read: §4.1, §18.
Done when:
- ds003775 v1.2.1 is in `data/`, and 111 subjects and 153 EDF files are found.
- A SHA-256 checksum is recorded for every file.

### [ ] B2. Split
Read: §11.1, §17.
Files: `main.py` writes `outputs/split_<seed>.json`.
Note: the 4 extra split seed values are not specified in the document. Ask me.
Done when tests show:
- The split is made on all 111 IDs before exclusions, and the result is deterministic.
- There are 78 training and 33 test subjects.
- Each subject's t2 recording is on the same side as their t1.
- 12 pilot subjects (4 of them with t2) are drawn from the training side.
- The pilot subjects are on the training side in all 5 seeds.

### [ ] B3. Loading and units check
Read: §4.2, §5.1 steps 1–2.
Done when:
- Files are loaded with `units="uV"` and their SHA-256 checksum is recorded.
- The units check applies a 0.5 Hz high-pass to a copy and requires the median SD to fall within 2–200 µV.
- The pipeline halts if more than 5% of recordings fail.
- For sub-010 ses-t1, the EEG is read and only its scans.tsv is skipped.
- Every exclusion is logged.

### [ ] B4. Channel pipeline, part 1
Read: §5.1 steps 3–6, §6.
Done when:
- The P3, PO3, P4 and PO4 labels are verified against `channels.tsv`.
- The bad-channel bounds (5–150 µV RMS, flat-line, saturation) are applied to those four electrodes only.
- The 50 Hz notch filter and its harmonics are applied.
- The zero-phase band-pass from A3 is applied.
- The bipolar pairs P3–PO3 and P4–PO4 are formed.
- Variant flags for §5.2 exist: 0.1 Hz high-pass, and strict rejection.
- Phase 1 saves the band-pass impulse response for both high-pass variants (0.5 and 0.1 Hz) under outputs/ (§5.1 disclosure).

### [ ] B5. Channel pipeline, part 2
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

### [ ] B6. Exclusions
Read: §12.
Done when:
- All rules are applied by code, and each exclusion is judged per recording.
- An exclusion log is written.
- The split is never redrawn.

---

## Stage C: State-space model and UKF (model.py)

Note: Before any phase >= 2 gets a real runner: check_prerequisite must also read outputs/gate.json and carry any low_confidence flag forward (CLAUDE.md rule 6, §18.1). Add tqdm progress bars when phases have real work (§18).

### [ ] C1. Augmented state-space model
Read: §7.2, §7.4, §7.6.
Done when:
- The 19-D state is 12 neural states plus p₁, p₂, ρ₁, ρ₂, g₁₂, g₂₁ and m.
- The E/I reparameterization holds A·B fixed.
- The mixing observation model has unit diagonal.
- The priors match §7.6, with m truncated to [0, 0.5].
- The M1 variant has 17 dimensions (gains removed).
- The §7.4 reduction order is available as config switches.

### [ ] C2. Reference UKF and smoother (filterpy)
Read: §7.5, §7.6, §9.3.
Done when tests show:
- The scaled unscented transform uses α = 1, β = 2, κ = 0. At n = 19, the mean weights are 0 at the centre and 1/38 for every other point, and the covariance centre weight is 2.
- There are 4 Heun sub-steps per observation.
- The delay ring buffer is held outside the sigma points, following the filtered/predicted-mean rule, and is filled with the steady-state value after re-initialization.
- The unscented RTS smoother works.
- The divergence definition is implemented.
- Stability monitors catch negative eigenvalues and NaN/Inf.
- The filter tracks the known states and parameters on A2 synthetic data.

### [ ] C3. Two-pass parameter handling
Read: §7.5 (parameter handling), §8.3, §10.2.
Done when:
- Recording-level parameters are the mean of the smoothed trajectory after burn-in.
- The windowed pass uses 2-s windows, re-initializes the 12 neural states, holds the parameters fixed, and discards 0.5 s of burn-in.

### [ ] C4. Q/R rule
Read: §7.5, §7.6.
Done when:
- R = 0.25·σ_ref²·I₂.
- q is chosen from an 8-value log grid spanning 1e-4 to 1e-1 by NIS closest to 2.
- The real-data mode uses 20 random training subjects per split seed.
- The G0 mode uses the 20-series synthetic tuning set.

### [ ] C5. Numba port and validation (§21.2 #4)
Read: §7.5, §16.5.2.
Done when:
- The filterpy Cholesky convention (upper-triangular) is confirmed from the installed source.
- Means and covariances match filterpy at rtol 1e-8 and atol 1e-10.
- Downstream statistics are unchanged.
- A timing comparison is reported.
- The fallback path to filterpy exists.

---

## Stage D: Regression machinery (regression.py)

### [ ] D1. Derivative estimators
Read: §8.3.
Done when:
- The weak-form estimator (Gaussian-window test function with compact support; 60, 100 and 160 ms options) is implemented.
- The TV baseline with its 5-value grid is implemented.
- Both are tested against analytic signals.

### [ ] D2. Residual target and rows
Read: §8.2 (inputs, z-scoring), §8.3, §8.4.
Done when:
- The residual is dy₄/dt minus the base-model prediction at the recording-level gains.
- Rows come from the 2-s windows after burn-in, with both directions pooled.
- z-scoring uses the fit fold's mean and SD, and those constants are stored.

### [ ] D3. PySR wrapper
Read: §8.2, §16.5.
Done when:
- The PySR version is pinned, and operators, inputs, parsimony and timeouts come from config.
- Real fits run in parallel. A serial fixed-iteration mode exists for determinism checks.
- One warm session is reused for all fits.
- Each fit gets a 50,000-row subsample with equal rows per subject.
- PySR's batching option is not used.
- turbo stays off until verified.

### [ ] D4. Pareto selection and term signatures
Read: §8.2, §8.4.
Done when:
- The split is 80/20 by subject. For G0 it is the last 20% of windows.
- The simplest equation within 5% of the minimum validation loss is selected and frozen.
- A bare constant counts as "no term".
- The sympy signature tests pass: tanh(1.02·x + 0.01) → tanh(x), and denominators are kept literally.

### [ ] D5. Refit ensemble
Read: §8.2, §15 (C2).
Done when 25 half-sample refits at 600 s each run with the same selection rule.

---

## Stage E: Synthetic gate (synthetic_gate.py)

### [ ] E1. Operating-regime grid
Read: §9.1.
Done when:
- The 12 × 8 × 5 grid over (p, input-noise SD, observation noise) is built.
- Matching uses scale-free features only (alpha peak, relative alpha power, specparam slope), taken from training subjects only.
- The regime (noise-driven or limit cycle) is reported.

### [ ] E2. Series generator
Read: §9.1, §9.2.
Done when:
- Series are generated at 2,048 Hz.
- There are 4 coupling levels, each with a planted product residual at 50% RMS.
- Null A and Null B are generated. Null B has a 0–20 ms lag and a 30–50% input-variance share.
- 1/f plus white noise is added, and m ~ U(0.1, 0.4).
- Each series is scaled to µV, run through the real preprocessing, and rescaled.

### [ ] E3. Preprocessing-gate artifacts
Read: §5.2.
Done when:
- Blinks, EMG bursts and drift are injected at the stated settings.
- The artifact-only null uses bilateral, near-zero-lag artifacts.

### [ ] E4. Gate scoring and gate.json
Read: §9.1–§9.4, §5.2, §18.1, §20.
Done when:
- The positive-control rule is implemented: median NRMSE ≤ 0.25 and median gain error ≤ 15%, from the second-weakest level up.
- The null δ-band rule is implemented, with δ = half the weakest level.
- Contraction must be ≥ 50% (median).
- Stability flags are set.
- Null or stability failure means a hard stop. Positive-control or contraction failure sets `low_confidence`. Pilot mode never stops.

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
