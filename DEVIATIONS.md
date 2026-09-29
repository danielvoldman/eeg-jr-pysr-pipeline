# DEVIATIONS.md

This is the dated record of every change to, or clarification of, the preregistered design in `PREREGISTRATION.md` (v0.6, frozen 2026-09-27).

**Rules**
- Add an entry before, or together with, the commit that makes the change.
- Never edit or delete an old entry. If a decision changes, add a new entry that supersedes it and names the old ID.
- The one exception: the Commit column of an entry may be filled in after the fact (for example when the hash was not known at commit time). Nothing else in an old entry may be edited.
- Entries in §1 are reported in the paper's Limitations section (preregistration §0). Entries in §2 are disclosed in Methods.

---

## 1. Deviations from the preregistration

This section covers any change to a LOCKED item at any time, and any change to anything after real data have been loaded.

| ID | Date | Section | Tag of changed item | Summary | Real data loaded? |
|---|---|---|---|---|---|
| (none yet) | | | | | |

**Entry template**

```
### DEV-001: short title
- Date:
- Preregistration section(s):
- Item and its tag (LOCKED / PLACEHOLDER):
- Before:
- After:
- Why:
- Real data loaded at the time? (no / yes, and what had been seen)
- Could this affect G0 or a C1–C4 verdict? How?
- Approved by:
- Commit:
```

---

## 2. Implementation clarifications (before real data)

This section covers places where the document was silent or ambiguous and a concrete choice had to be made while writing code. These are not deviations from a preregistered decision, but they are disclosed.

| ID | Date | Section | Question | Choice | config.yml key | Approved by | Commit |
|---|---|---|---|---|---|---|---|
| IMP-001 | 2026-09-28 | §18 | §18 lists no config reader, tests or pytest config | config helper src/config.py, tests/ and pytest.ini added; §18 lists none of these. Also: pilot runs log to logs/phase{N}_pilot.log | none | Daniel Voldman | d5b6d18 |
| IMP-002 | 2026-09-28 | §18.1 | §18.1 puts pilot output under results/pilot/ but does not say where pilot phase flags live | Pilot flags at results/pilot/phase{n}.done. A pilot run reads and writes only pilot flags; a full run only outputs/ flags, so a pilot phase can never unlock a full phase | paths.pilot_phase_flag_pattern (placeholder) | Daniel Voldman | d5b6d18 |
| IMP-003 | 2026-09-28 | §5.1, §7.5, §7.6, §9.1 | The document leaves the reference-simulation structure, integrator, input redraw interval, burn-in and seed, and the simulator delay step count, open | Reference simulation: single node, standard §7.3 parameters, no mixing, no coupling; Heun with the input drawn once per 2,048-Hz step and held within the step; fresh uniform draw every step, centred on p with half-width (320-120)/2; burn-in 10 s; seed 42. Simulator delay 20 steps at 2,048 Hz (9.765625 ms, equal to the UKF's 10 sub-steps at 1,024 Hz). Simulator numerics: steady-state scan of 2001 points and 100 bisection iterations; Welch segment 8 s for the A2 sanity report Further choices, disclosed in the A2 fix round: (a) the second Heun stage reads the delayed coupling one step later than the first (stage 1 reads S at step k-d, stage 2 at step k+1-d); (b) the start is the uncoupled steady state and the history buffer is filled with the uncoupled steady-state S, so a coupled system does not start at rest and burn-in absorbs the transient; (c) the coupling argument is S(y1-y2) of the source node; (d) g12 runs from node 1 to node 2 and g21 from node 2 to node 1; (e) steady_state raises when the residual has more than one root. Consequence: the input noise is drawn per integration step with no sqrt(dt) scaling, so its strength depends on the step size, which is fixed at 2048 Hz by §5.1 and §9.1; mu_ref, sigma_ref and G0 depend on that. | rescaling.reference_simulation.{seed, burn_in_s, node_structure, integrator, noise_redraw_interval}; coupling.sim_delay_steps; simulator.{steady_state_scan_points, steady_state_bisection_iterations, sanity_welch_segment_s} (all placeholder) | Daniel Voldman | e356efb, fixed in the commit made by the A2 fix round |
| IMP-004 | 2026-09-28 | §5.1, §5.2, §10.1, §16.5.3 | The document allows forward-backward or linear-phase FIR filtering and gives no filter family, order, edge handling or resampling pad | Butterworth in second-order-sections form, high-pass and low-pass designed separately at order 4 per pass, applied forward and backward with scipy.signal.sosfiltfilt (zero phase; no causal filter). Forward-backward filtering squares the magnitude response, so the nominal 0.5 Hz and 45 Hz cutoffs are the -6 dB points of the combined filter, not the -3 dB points; the cutoffs are not adjusted to compensate. The high-pass frequency is an argument, defaulting to preprocessing.bandpass.highpass_hz; callers pass sensitivity_highpass_hz for the §5.2 variant. Edge handling: sosfiltfilt padlen = round(10 s * fs) with odd extension. bandpass() is for whole continuous recordings before segmentation, never for short segments; inputs of padlen samples or fewer raise by design; trimming or masking the filtered edges is decided in B4 using the edge-transient numbers printed by tests/test_preprocess.py, and nothing is trimmed in A3. Downsampling: FFT resampling (scipy.signal.resample with an odd reflect-pad; MNE's default resampling is also FFT-based but pads with reflect_limited, so the edge handling differs; no polyphase, no decimate), integer ratios only (1024->256, 2048->256), length must be divisible by the ratio, 2 s odd reflect-pad (rounded up to a multiple of the ratio) before the transform and trimmed after; no second anti-alias filter. Order in the real pipeline: band-pass at the native rate, then downsample. Inputs must be float64 (TypeError otherwise). Impulse response: bandpass_impulse_response() returns the response of the band-pass code path (120 s, centred); saving it for both high-pass variants is done in phase 1 (PLAN.md B4). | preprocessing.bandpass.{filter_family, filter_order, edge_pad_s, edge_padtype, impulse_response_duration_s}; preprocessing.resample.{pad_s, reflect_type} (all placeholder) | Daniel Voldman | a0be7be |
| IMP-005 | 2026-09-28 | §5.1, §7.6 | The document says the reference simulation runs "for 600 s" without saying whether burn-in counts; gives no trim, SD convention, diagnostics, write procedure or entry point for mu_ref and sigma_ref | Units: mu_ref and sigma_ref are in mV (model output units). Simulation: single node at the standard §7.3 parameters, p = 220, uniform(120, 320) input redrawn every step (IMP-003), Heun, sim_fs_hz 2048, seed 42; burn_in_s (10 s) is simulated IN ADDITION to duration_s (600 s), discarded, and exactly 600 s is kept (post-step values; the initial state row is dropped). mu_ref = mean of the UNFILTERED y1-y2 over the kept 600 s at 2048 Hz (before any downsampling). sigma_ref = SD after preprocess.bandpass at 2048 Hz (hp 0.5, lp 45) and preprocess.downsample to 256 Hz, then edge_trim_s = 5 s trimmed from EACH end (590 s x 256 Hz = 151040 samples kept), sample SD with ddof = 1; B5's per-recording rescaling must use the same SD convention (ddof = 1, same trim question decided in B5). Written only by targeted text replacement of the three `value: null` fields (mu_ref, sigma_ref, computed_provenance), floats via repr(), never yaml.dump; a real write refuses unless `git status --porcelain` is empty and HEAD equals --expect-commit (the hash of the approved dry run), and refuses to overwrite non-null values unless --force (such an overwrite must then be logged in §1). computed_provenance stores seed, kept duration, burn-in, sim and observation rates, trim, ddof, sample counts, units, git hash and dirty flag, and numpy, scipy and numba versions. Entry point: `python -m src.model --compute-reference [--dry-run] [--force] [--expect-commit HASH]`; phase 1 in main.py is not wired to it. The dry run prints, and never stores, diagnostics: SD unfiltered vs filtered, mean at 2048 Hz vs 256 Hz, the alpha peak (highest periodogram bin above DC, 8 s Hann segments, numpy only), and mu and sigma for seed 42 and diagnostic seeds 43-47 with spread and relative spread. The stored constants use seed 42 only. Diagnostic result (dry run at df29a2a): across seeds 42-47 the relative spread was 0.03% for mu_ref and 2.8% for sigma_ref, so sigma_ref carries about 1% seed-to-seed uncertainty; the stored values use seed 42 only, chosen before the numbers were seen. | rescaling.reference_simulation.{edge_trim_s, ddof, diagnostic_seeds} (placeholder); rescaling.computed_provenance (computed); rescaling.{mu_ref, sigma_ref} (computed, written after approval of the dry run) | Daniel Voldman | df29a2a |
| IMP-006 | 2026-09-28 | §5.1, §7.6 | Review of the A4 commits (df29a2a to 2ebdfcc) found process slips and two open points on how mu_ref and sigma_ref are used later | (a) Commit 7191d10 added a sentence to the body of IMP-005 and commit 280131d changed wording in IMP-004, both against the rule that only the Commit column of an old entry may change; disclosed here, not reverted. (b) The write at 7191d10 reproduced the approved dry-run values digit for digit, so no second dry run was needed. (c) B5 rule: per-recording rescaling uses the same SD convention (ddof = 1) on the band-passed 256 Hz clean samples; how the filtered edges are trimmed is decided in B4/B5 from the A3 edge-transient numbers. The ddof and edge_trim_s under rescaling.reference_simulation belong to the reference simulation and must not be read as real-data settings. (d) One sigma_ref, computed with the 0.5 Hz high-pass, is used for every filter variant including the 0.1 Hz sensitivity variant (§5.1 "computed once"). (e) The A4 fix round refactors compute_reference_constants to call reference_statistics (one implementation of the mean and SD) without changing the constants, proved by an exact-equality test (==) against the stored values; main() also repeats the clean-tree and expected-commit check immediately before writing. | none (no config value changed; only the note on simulator.sanity_welch_segment_s) | Daniel Voldman | A4 fix round (hash to be filled in later) |

**Entry template**

```
### IMP-001: short title
- Date:
- Preregistration section(s):
- Question the document left open:
- Choice made:
- Why:
- Stored in config.yml as (key, provenance tag):
- Approved by:
- Commit:
```

---

## 3. Placeholder finalization (after the pilot, before full-cohort results)

These are expected calibrations, not deviations. They are recorded here so that the final values and the evidence behind them are in one place.

| Parameter | Section | Default | Final | Pilot evidence | Date |
|---|---|---|---|---|---|
| (filled in at Stage J) | | | | | |

---

## 4. Resolution of the open items in §21.2

| # | Item | Resolution | Evidence | Date |
|---|---|---|---|---|
| 1 | Hard rejection vs noise-adaptive observation covariance | Open: compared in the pilot; hard rejection is the default | | |
| 2 | Operating system and CPU affinity mask | OS: Windows (VS Code, PowerShell). Mask: `FF` (logical CPUs 0–7) assumed; SMT sibling pairing still to be verified with Task Manager or Coreinfo | User decision | 2026-09-28 (OS only) |
| 3 | Worker count for the joblib pool | Open: pilot benchmark of 4 vs 6 vs 8 | | |
| 4 | Numba UKF validation against filterpy | Open: pilot (rtol 1e-8, atol 1e-10, plus downstream statistics) | | |
