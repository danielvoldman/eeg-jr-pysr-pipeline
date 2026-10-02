# G0 hard stop: a preregistered negative result

Project: two-node Jansen–Rit model, unscented Kalman filter and PySR residual on two bipolar resting-state EEG channels (OpenNeuro ds003775 v1.2.1). Specification: `PREREGISTRATION.md` v0.6 (tag `prereg-v0.6`). Decision record: `DEVIATIONS.md`, entries DEV-001 to DEV-007 and IMP-001 to IMP-099.

**How to read the numbers.** `results/` and `outputs/` are not committed (CLAUDE.md rule 4), so this file is the committed record. A number carries an invisible HTML-comment marker: either `src` (a field of a result file, hashed in the table of §8) or `doc` (text that must occur in a tracked document). `tests/test_closing_report.py` checks every marker. Numbers marked `doc` come from `DEVIATIONS.md` or `PLAN.md` as of commit `c96d2be`.

## 1. Outcome in one paragraph

The gate G0 (§9, §15.1) failed at full size. Under the preregistered rule a failed null arm is a hard stop (§9.2, §18.1, §20). In the formal run of round 1 (seed block 102000, 60 series per arm, filter A, q = 0.01, δ = 1.08), 51<!--src results/g0_null_arms_r1.json arms.null_A.false_positives=51--> of 60 Null A series and 55<!--src results/g0_null_arms_r1.json arms.null_B.false_positives=55--> of 60 Null B series were false positives on the gain half of the criterion. The criterion is 0 of 60 (§15.1). In plain terms: on these synthetic series, with no true coupling, the estimator cannot separate coupling from zero at δ. Nothing was tuned, loosened or rerun. No real-data model fit, PySR run or C1–C4 verdict was made, so none exists. This report is the preregistered response: report it, never loosen it (§20).

## 2. Claim-by-claim verdicts

| Claim | Verdict | Why |
|---|---|---|
| **G0** | **FAILED, hard stop** | Null A 51/60 and Null B 55/60 gain-half false positives (criterion 0/60, §15.1). One-sided 95% upper bounds 0.9195<!--src results/g0_null_arms_r1.json arms.null_A.upper_bound_95=0.9195--> and 0.9666<!--src results/g0_null_arms_r1.json arms.null_B.upper_bound_95=0.9666-->; 0 of 60 would give 0.0487<!--doc DEVIATIONS.md::0 of 60 passes (0.0487)-->. The positive control, posterior contraction, numerical stability and the preprocessing-bias check were **not run** on the formal arms (`not_run`). The artifact-only null is **cited from the pilot**, not a verdict of this run: 20<!--src results/g0_null_arms_r1.json cited_artifact_only_null.n_fail=20--> of 20 failed. |
| **C1** (refinement beats base coupling and VAR) | **Not attempted** | Phase 2 is refused by the hard stop, and no frozen equation exists. `results/pilot/c1_42.json` is mechanics only, with M3 absent. |
| **C2** (refit and seed stability) | **Not attempted** | No primary fit and no ensemble, so no signature recurrence. |
| **C3** (session reliability) | **Not attempted** | The pilot has 2<!--doc PLAN.md::2 usable pairs--> usable pairs and an M2 stand-in, so the pilot ICC carries no information. |
| **C4** (short-window generalization) | **Not attempted** | The gate blocks it. Independently, on the pilot the stability condition was met by 0 of 12<!--doc PLAN.md::stability condition is 0 of 12--> recordings against the required 80%. |

"Not attempted" means the question was not asked. It is neither a failed claim nor a null result. The §20 row "PySR returns bare constants on real data → genuine null" is not triggered, because PySR was never run on real data.

The gate has two halves for each null arm (§15.1): the gain half and "a selected residual term". The second half needs PySR and was not evaluated. The series with no gain failure (9<!--src results/g0_null_arms_r1.json arms.null_A.n_pending=9--> in Null A, 5<!--src results/g0_null_arms_r1.json arms.null_B.n_pending=5--> in Null B) are **pending, not passed**.

## 3. The formal null arms (K0, round 1)

| | Null A | Null B |
|---|---|---|
| Series | 60<!--src results/g0_null_arms_r1.json arms.null_A.n=60--> | 60<!--src results/g0_null_arms_r1.json arms.null_B.n=60--> |
| Dropped by the legacy divergence rule (counted as failures, §9.2) | 39<!--src results/g0_null_arms_r1.json arms.null_A.rule_on.dropped=39--> | 39<!--src results/g0_null_arms_r1.json arms.null_B.rule_on.dropped=39--> |
| Not dropped, filtered gain outside δ | 12<!--src results/g0_null_arms_r1.json arms.null_A.rule_on.outside_delta=12--> | 16<!--src results/g0_null_arms_r1.json arms.null_B.rule_on.outside_delta=16--> |
| Not dropped, inside δ (gain half only, pending) | 9<!--src results/g0_null_arms_r1.json arms.null_A.rule_on.inside_delta=9--> | 5<!--src results/g0_null_arms_r1.json arms.null_B.rule_on.inside_delta=5--> |
| False positives (gain half) | 51 | 55 |
| Diagnostic only, state-SD flag off: outside δ | 33<!--src results/g0_null_arms_r1.json arms.null_A.flag_off_diagnostic.counts.outside_delta=33--> | 43<!--src results/g0_null_arms_r1.json arms.null_B.flag_off_diagnostic.counts.outside_delta=43--> |

The flag-off row is never a verdict. It shows that removing the divergence rule would not rescue either arm. The 12 and 16 outside-δ series stay outside without the rule, and 21 of the 39 dropped Null A series would also be outside. Stability information is clean (no NaN/Inf, no negative eigenvalue, no failed Cholesky; information only).

Setup (fixed in `IMP-094` before the run, unchanged afterwards): δ = 1.08<!--src results/g0_null_arms_r1.json delta=1.08--> (units 1/s); q = 0.01<!--src results/g0_null_arms_r1.json q=0.01-->; filter A; 78 training subjects<!--src results/g0_null_arms_r1.json inputs.n_training_subjects=78-->; 108<!--src results/g0_null_arms_r1.json inputs.n_recordings=108--> usable recordings; 1/f exponent 1.394<!--src results/g0_null_arms_r1.json inputs.exponent=1.394-->; 480<!--src results/g0_null_arms_r1.json inputs.n_valid_grid_points=480--> grid points. The evaluation took 516<!--src results/g0_null_arms_r1.json timing_s.evaluate=516--> s on 4 workers.

Gate state (`outputs/gate.json`): `hard_stop` true<!--src outputs/gate.json hard_stop=True-->, `complete` false<!--src outputs/gate.json complete=False-->, `low_confidence` false<!--src outputs/gate.json low_confidence=False--> (it is not set from checks that were not run). `phase1.done` was not written, and phase 2 refuses to start.

## 4. Diagnostics that were run

| What | Result | Evidence |
|---|---|---|
| C4b–C4d (real pilot, then synthetic with 1/f background) | The 19-D filter breaks on a 1/f background; adding 1/f to synthetic data reproduces the real-data behaviour | DEV-005 |
| C6–C8 (synthetic) | With 1/f the 19-D gain error is 370<!--doc DEVIATIONS.md::relative error 370 to 2700%--> to 2700% at q = 0.01. The gain prior is not the cause | DEV-005 items 4 and 6 |
| DEV-005 (filter A adopted, 2026-10-01) | An OU coloured observation-noise state removes the divergence on the diagnostics, but the gain error stays about 20 to 40%<!--doc DEVIATIONS.md::Gain error is about 20 to 40% at best--> at best | DEV-005 |
| E5/E6 (operating points) | At matched real-like operating points every positive series is dropped and the nulls fail; at the easier default point A and B keep 19 of 20 nulls inside δ | DEV-005, IMP-070, IMP-071 |
| E7 (planted residual set to zero) | Gain error falls to 4, 18, 21 and 34% at L1–L4 at the easy point, so the unmodelled residual explains part of the error there but not at the matched points | IMP-072 |
| E8 (12 real pilot recordings) | 19-D reproduces C4b–C4d; A removes the parameter runaway and drops a median 17.7% of samples | IMP-073, DEV-005 |
| D1/D2 oracle PySR on synthetic rows | The smoke fit finds the planted product at complexity 3. Turbo on and off give different fronts (3 of 3 seeds, speed-up 1.5 to 1.8), so turbo stays off | IMP-060 |
| DEV-007 and its selection test | The dwell-plus-parameter divergence rule failed its pre-declared test: sensitivity 0.883<!--doc DEVIATIONS.md::sensitivity 0.883--> (needs 0.9), false-positive rate 0.425<!--doc DEVIATIONS.md::false-positive rate 0.425--> (needs 0.05). The runaway labels were a design error. The rule was **not adopted** and the legacy rule ships | IMP-091 to IMP-093 |
| One-step metric (pilot, 3 subjects) | On this oversampled, band-limited signal the one-step error favours a linear VAR over the Jansen–Rit UKF by about three orders of magnitude<!--doc PLAN.md::about three orders of magnitude-->. This is a property of the metric and of R, so M3 < M0 was expected to fail | IMP-079 |
| Planted-ratio note (E2) | The realised planted-to-base RMS ratio is 0.46 to 0.53 at L1–L3 but 0.76 to 0.84<!--doc PLAN.md::0.76 to 0.84--> at L4, against the 0.50 target | PLAN E2, IMP-065 |
| Filtered-parameter bias | The filtered p is 260 to 440<!--doc DEVIATIONS.md::p_est 260 to 440 against a true 120 to 138--> where the synthetic truth is 120 to 138: the filter sits in the noise-driven regime while the data are limit-cycle | DEV-007 |
| K0 round 0 (block 92000) | Killed by an operator timeout during evaluation. No result was seen or recorded, and the block stays burnt | IMP-096, IMP-097 |

## 5. Limitations

- **Regime coverage.** G0 covers only the limit-cycle regime: all grid points are limit-cycle (E1). Whether the null failure extends to the noise-driven regime was never tested.
- **Biased filtered p.** See §4. The cause was not investigated beyond the stated observation.
- **Filter A absorbs alpha.** It absorbs 8 to 11%<!--doc DEVIATIONS.md::8 to 11% of the observed alpha-band power--> of the observed alpha-band power at q = 0.01. Mean NIS is 0.5 to 0.9<!--doc DEVIATIONS.md::Mean NIS is 0.5 to 0.9-->, so the §7.5 NIS rule refuses, and q = 0.01 is fixed by declaration (DEV-005, DEV-006).
- **Divergence rates.** Formal arms: 39 of 60 dropped in each (§3). G0 pilot: 88 of 100<!--doc DEVIATIONS.md::dropped 88 of 100 series--> series dropped. Real pilot recordings: 6 of 12<!--doc PLAN.md::6 recordings have a diverged pass 1--> diverge in pass 1 and 9 of 12<!--doc PLAN.md::9 of 12 at 0.1 Hz--> under the 0.1 Hz variant.
- **Divergence-rule risk (DEV-003).** The rule flags strong two-way coupling even where the true trajectory is healthy. The legacy rule was kept (IMP-093) and the dwell code is dormant.
- **Real data.** Only the 12 pilot subjects (training side) were ever used for development, and no test-subject data were used<!--doc DEVIATIONS.md::no test-subject data-->. Every real-data file under `results/pilot/` is mechanics only.
- **tau.** The fitted OU time constant sits at its bound in most fits, so it must not be read as a measured time constant (C7, DEV-005 limit 7).
- **Seed blocks.** Burnt or consumed: 92000 (round 0, lost, no result seen), 94000 (artifact-only null, pilot), 96000 (rule selection), 99000 (throwaway) and 99500 (smoke). Block 102000 was used once and is the formal result. See `results/g0_seed_ledger.json`.
- **Deviations.** DEV-001 (electrode RMS floor 5→1 µV), DEV-002 (excessive blink-correction exclusion), DEV-003 (start-up exemption), DEV-004 (smoothed-covariance tolerance), DEV-005 (filter A), DEV-006 (state counts and fixed q), DEV-007 (not adopted). Per `DEVIATIONS.md` all must be reported with any paper.
- **Single round.** One formal round of 60 series per arm. No second round was run, by decision (IMP-099).
- **Not covered.** The preprocessing-bias check, the positive control and PySR were not run at formal size, so nothing here says how the pipeline would have scored a true coupling.

## 6. §1.4 and what this result does and does not say

§1.4 states the pre-registered leading explanation if **C1 or C2 fails**: resting eyes-closed EEG may be too uninformative to pin down a coupling function (near-linear regimes, one condition, little state transition). That did not happen. C1 and C2 were not attempted, so §1.4 is neither supported nor refuted by this result. It stays a hypothesis about real data that this study did not reach.

What was found is more proximate and sits at the gate: on synthetic data with a 1/f background and no true coupling, the estimator cannot separate coupling from zero at δ = 1.08. This does not show that the pipeline produces physical coupling, and it does not show that real EEG contains none. The two statements (§1.4 and the gate failure) are separate and are not merged here.

## 7. Findings and unknowns

**Findings (measured, with source above)**
1. Both formal null arms fail the gain half of G0 by a wide margin (51/60 and 55/60 against 0/60).
2. The failure does not come from the divergence rule alone: with the rule off, 33 and 43 of 60 series are still outside δ (diagnostic).
3. With a 1/f background the 19-D filter loses the gain, and filter A reduces but does not remove the gain error (20 to 40% at best).
4. The pre-declared divergence-rule fix failed its own selection test and was not adopted.
5. On the pilot's real recordings, divergence under the legacy rule is common (6 of 12; 9 of 12 at 0.1 Hz).
6. The pilot's one-step metric favours a linear VAR by about three orders of magnitude.

**Unknown (never tested here)**
1. Whether any other observation model, q, δ or divergence rule would pass the null arms.
2. Whether the failure holds in the noise-driven regime.
3. How the positive control, contraction and the preprocessing-bias check score at formal size.
4. Whether the PySR half of the null criterion would be met (PySR was not run at formal size).
5. Why the filtered p is biased high (only the observation is recorded).
6. Whether real EEG of this kind contains identifiable inter-node coupling, and whether M3 beats M2 or M0. No such test was made.
7. Anything about the 33 test subjects. They were never read.

## 8. Source files

| File | sha256 |
|---|---|
| `results/g0_null_arms_r1.json` | `70124794b8b2ff913f8a57d8fbc4520a5feebad2d0342f76d9d953e622b11972` |
| `outputs/gate.json` | `4c2dcd59d4d149ecf1360cd765dace5e4d15bea484de543e663d793417ae7825` |
| `results/pilot/gate_A.json` (cited artifact-only null, pilot) | `826f458e5721d226a44db0f896a2c42b95b8922309a4e05728e95a41a100c60c` |
| `results/g0_seed_ledger.json` | `61f9ed16ea811fae694ef113b48d29e2a0b7d84a8324b23d62700952f8dce59b` |

Run provenance: code commit `712365ae7eebedf08c0e51589a4ea196993190e9`, config sha256 `c670dd21dafc0923b4d49da1528304ab2d982a9a83c632b5480a2e4a947bfc11` (field `provenance` of the result file). Tracked sources: `DEVIATIONS.md`, `PLAN.md`, `HANDOFF.md`, `PREREGISTRATION.md` at the tag `g0-hard-stop-v1`.
