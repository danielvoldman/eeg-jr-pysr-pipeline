<!--
FROZEN PREREGISTRATION. Do not edit; record changes in DEVIATIONS.md.
Converted verbatim from: EEG_JansenRit_PySR_Planning_Document.docx (planning document v0.6, 2026-09-27)
SHA-256 of the project copy used for conversion (a text export saved under the .docx name, not the original Word file): 9efc4e6730998c88cc04696031db5a0b5afe1a76ec3473e08153174a423b77ac
Converted: 2026-09-28
-->

Coupled Jansen–Rit + PySR on EEG · Planning Document v0.6

**PLANNING DOCUMENT & PREREGISTRATION**

**Time-Domain Symbolic Refinement of a Coupled Jansen–Rit Model on Two-Channel Resting-State EEG**

**Author:** Daniel Voldman

**Dataset:** OpenNeuro ds003775 v1.2.1 (SRM resting-state EEG)

**Document status:** Draft v0.6 (implementer dry-run fixes and Jansen–Rit verification), written before any real data is loaded

**Date:** September 27, 2026

**Predecessor project:** A Mixed Verdict on Neural State Postulates in C. elegans (celegans-nsd-pipeline)

| **The one-sentence version** Use a two-node Jansen–Rit neural-mass model as a fixed, physiologically grounded skeleton, fit it to two independent bipolar EEG channels by estimating hidden states with an unscented Kalman filter, let PySR refine only a residual on the coupling term between the two nodes, and test honestly whether that refinement adds anything over the skeleton and a linear baseline, and whether the coupling estimate holds up across subjects, sessions and window lengths. |
| --- |

**Contents**

# 0. How to Read This Document

This document plays the same role PREREGISTRATION.md played in the C. elegans project. Everything in it is committed **before** any real EEG is loaded, so that thresholds, exclusion rules and fallbacks are decided at a moment when there are no results to be tempted by. Every decision carries one of three status tags:

| **Tag** | **Meaning** | **What you do with it** |
| --- | --- | --- |
| LOCKED | Decided and backed by the research audits or by internal consistency. | Do not change without writing a dated deviation note. |
| PLACEHOLDER | A reasonable starting value that the pilot run is allowed to recalibrate. | Finalize after the pilot, before the full-cohort run is inspected. |
| DECIDE BEFORE PILOT | Still open. Must be resolved before any code touches real data. | Resolve and move to LOCKED or PLACEHOLDER. |

The rule that makes this work: **any change after real data is loaded gets written down as a deviation, with the date and the reason**, and reported in the paper's Limitations section. Last time, this is what turned things like the VB01 anomaly and the ensemble-definition gaps into honest disclosures instead of hidden problems.

**Evidence labels.** Where this document relies on something an audit labelled as the audit author's own inference rather than a published finding, it says so (“reasoned, not tested”). Those points are design choices, not citations.

## 0.1 Revision log: v0.1 → v0.6

v0.2 incorporated an independent stress test of v0.1 against all six evidence audits then available. It found 3 fatal and 11 moderate issues (all internal design contradictions or under-specifications, none requiring new literature) plus 10 places where the document dropped, softened or overstated an audit's caveats. v0.3 folds in a seventh, dedicated performance audit and resolves the two remaining open engineering decisions it identified (UKF implementation; derivative estimator). v0.4 closes out six of the ten remaining open decisions in §21 with the simplest, safest available option for each, leaving only the four genuinely pilot-dependent items open. v0.5 applies a final, gentler read-through review: 6 must-fix text contradictions (PySR execution settings, Q/R versus G0 ordering, the null-gain interval, the partial-G0 rule), 12 definitions that needed pinning down, and a set of small consistency notes. None reopened a LOCKED design decision. v0.6 applies a third, implementer-style dry-run review (4 rules that could not work as written, 7 missing values and about 20 smaller definitions) and a verification of the Jansen–Rit constants and equations against independent sources. Four of its findings changed LOCKED pass rules (null gate, C4) or LOCKED procedures (parameter handling, rescaling); the arithmetic or simulation behind each is shown in the text. Every change is listed here.

| **ID** | **Problem in v0.1** | **Fix in v0.2** | **Where** |
| --- | --- | --- | --- |
| I-3 (fatal) | Unclear whether the coupling gain was estimated; the G0 null only inspected PySR's residual, so a UKF gain could silently absorb mixing. | Estimated parameter list fixed; coupling gains are estimated; G0 null now requires both no residual AND a gain interval containing zero. | §7.4, §9.2 |
| I-4 (fatal) | C2 measured term recurrence “across bootstrap fits”, which is 100% by construction under the no-refit rule. | C2 now uses a small, budgeted refit ensemble (25 half-sample refits) plus 5 split seeds. | §8.2, §14, §15 |
| I-10 (fatal) | C3 “structure stable across sessions” cannot fail when one frozen equation is used for both sessions. | C3 restated as reliability of the per-session coupling-gain estimate, with a defined estimator and a CI-based threshold. | §1.2, §11.3, §15 |
| I-1 | Units check on raw, unfiltered, per-channel SD could exclude recordings for electrode quality or drift. | Units check uses a robust median SD after a 0.5 Hz high-pass; electrode quality left to the bad-channel rule. | §4.2, §5.1, §12 |
| I-2 | Preprocessing gate tested parameters that are never estimated and had no failure path. | Gate targets the actually estimated quantities plus an artifact-only null; failure row added. | §5.2, §20 |
| I-5 | No rule for picking an equation from the Pareto front. | Selection rule preregistered. | §8.2 |
| I-6 | UKF noise settings, filter vs smoother, and the residual target were undefined. | All three defined, with a preregistered tuning rule. | §7.5, §8.3 |
| I-7 | Null control omitted common input. | Second null arm with correlated, lagged common input. | §9.2 |
| I-8 | Null sample size could not support a 5% false-positive claim. | ≥ 60 null series per arm, pass stated as an upper confidence bound, fresh seeds after any fix. | §9.4, §20 |
| I-9 | Synthetic regime, coupling strength and generation rate unspecified (risk of passing by construction). | Generated at 2,048 Hz with a fine integrator, coupling-strength grid including weak values, PSD matched to training subjects. | §9.1 |
| I-11 | No ablation, so C1 could pass with the PySR term contributing nothing. | Four-step nested comparison. | §13, §15 |
| I-12 | C1 had 3 epoch lengths × 2 metrics with no primary. | One primary metric; everything else secondary with Holm correction. | §15 |
| I-13 | C4 meaningless until epoch handling in the UKF was defined. | Training windows and free-run spectral scoring defined. | §10, §15 |
| I-14 | Pilot subjects could later enter the confirmatory test set. | Pilot drawn only from the training partition of the primary split. | §11.1, §17 |
| P-1 to P-10 | Caveats dropped or findings overstated relative to the audits. | Relabelled, restored or added (Kuhlmann ARMA result, Escuain-Poole, regime limits, Stankovski and Delabays, Acharya scope, Yokoyama scope, EMG, vigilance, compute unit, verification). | Throughout |
| v0.3 | Two engineering decisions (§21) still needed resolving: UKF implementation, and the derivative estimator. | Resolved first by two targeted lookups, then extended by the performance audit: filterpy's numerical-stability risk at 19-D confirmed as real (not a style choice), with a Numba rewrite added as the validated-not-assumed performance path; weak/integral-form derivative estimation adopted as primary over point-wise methods. A dedicated seventh audit, on performance only, then found additional structural (risk-free) and numerical (gate-checked) speedups across every stage. | §7.5, §8.2, §16.5, §17, §21 |
| v0.4 | Ten decisions were still open in §21. | Six locked with the simplest, safest option: montage P3–PO3 / P4–PO4, keep sub-010 EEG, exclude subjects with a bad pair electrode, standard notch filter, no common-input term in the fitted model, split seed 42. Four pilot-dependent items remain. | §4.2, §5.1, §6, §7.2, §11.1, §12, §21 |
| v0.5 (must fix) | Two sections disagreed about the same setting in six places (review items 1–6). | PySR: real fits run in parallel, serial only for small pilot checks; populations left at default; fixed-size subsample instead of batching. Q/R: G0 tunes its own on a separate synthetic set; real-data Q/R tuned only after G0 passes, per split seed. Null test gain interval defined (block-mean estimator; the interval rule itself was superseded in v0.6 by the δ-band rule). One gate rule: a null failure stops the run, a positive-control failure continues flagged low confidence. | §7.5, §8.2, §9, §18, §19, §20 |
| v0.5 (definitions) | Twelve points where code written from the text could go two ways (review items 7–18). | Delay implementation, the state the residual is defined on, one shared residual equation, Pareto fit-then-freeze, C2 term-matching rule, derivative estimator chosen on the pilot synthetic set, ±15% gain criterion, rescaling reference, C1 burn-in and error definition, C4 stochastic free-run and spectral grid, C3 primary version, per-recording exclusion rules, per-variant divergence reporting. | §5.1, §7.5, §8.2–8.4, §9.1, §10.2, §11.3, §12, §15 |
| v0.5 (notes) | Small inconsistencies (count of synthetic series, stale stage numbers, leftover 5-s wording, tolerance choice). | Synthetic-series count corrected (about 220, not 380) and the total budget recomputed; Numba tolerance loosened with the Cholesky convention specified; extra split seeds constrained so pilot subjects stay in training. | §11.1, §16, §17, §18 |
| v0.6 (rules) | Four rules could not work as written (implementer dry-run). | (1) Null gate: under the interval rule a correct pipeline passes 0-of-60 only about 0.2% of the time, so the rule is now │estimated gain│ < δ (half the weakest planted coupling). (2) C4: the long/short ratio of averaged spectra was about 0.38 even for a perfect model, so every window is scored per 2-s segment (ratio about 1.00 for a non-drifting model). (3) Slow parameters are estimated once per recording and held fixed inside 2-s training windows, so shrunk gains cannot leak the base coupling into PySR's target. (4) The rescaling reference is fixed once from literature settings, which removes the circular dependency with G0; p is excluded from the ±15% bias check. | §5.1, §5.2, §7.5, §8.3, §9.1–9.2, §10.2, §15 |
| v0.6 (values) | Values missing inside LOCKED designs. | New §7.6 table of PLACEHOLDER defaults (Q/R family, priors, sigma points, integrator); planted residual, null strengths and G0 aggregation defined in §9; weak-form derivative defined; PySR inputs and z-scoring specified; preprocessing detector defaults; VAR coefficient handling; AAFT test size; divergence and delay-buffer definitions. | §5.1, §7.5–7.6, §8, §9, §13, §14 |
| v0.6 (verification) | Jansen–Rit constants and equations were written from memory. | Checked against independent sources (Escuain-Poole 2018, Grimbert & Faugeras 2006 via secondary sources, reference code): all constants and the six-state equations confirmed. Added the explicit sigmoid formula, a notation note (output y₁−y₂), the p wording, Escuain-Poole as a published precedent for the base coupling form, and a coupling grid and prior re-centred on the published medium coupling (k = 5–10). | §7.1, §7.3, §8.1, §9.1 |
| v0.6 (plumbing, budget) | Ordering, ownership and budget gaps found in the data-flow check. | Split made once before exclusions; reference constants stored in config.yml so preprocess.py no longer depends on model.py; gate.json carries stability and contraction flags; pilot mode never hard-stops; the §16.5.1 order now includes G0; four unbudgeted UKF workloads added; total re-estimated to about 26–62 h. | §11.1, §16, §18, §20 |

# 1. Project Summary and Claims Being Tested

## 1.1 What the project does

The C. elegans project fit a two-unit FitzHugh–Nagumo system to ensemble-averaged calcium traces and asked PySR to discover the derivative laws. PySR returned bare constants and the AVA→AVB coupling was never recovered. This project keeps the core idea (a known dynamical skeleton plus symbolic regression) but fixes the weak points: the skeleton is now a model built specifically to generate EEG, the two signals are two real recorded channels instead of constructed averages, the hidden excitatory and inhibitory states are estimated by model inversion rather than invented, and PySR is asked a smaller, answerable question.

## 1.2 The claims, each scored separately

As with A1, A4 and A5 last time, each claim gets its own verdict. There is no single “the project worked” result.

| **ID** | **Claim** | **Tested on** | **Status** |
| --- | --- | --- | --- |
| G0 | **Gate, not a claim.** The pipeline recovers known coupling from synthetic Jansen–Rit data, and does NOT report coupling from two kinds of zero-coupling null data (mixing only; common input plus mixing). | Simulated data only | LOCKED |
| C1 | **Refinement adds value.** Adding the PySR residual to the base coupling improves held-out prediction over the base-coupling Jansen–Rit model, and the full model beats a tuned linear VAR baseline. | 33 held-out subjects (t1) | LOCKED |
| C2 | **Refinement is stable.** The same coupling residual terms recur across independent refits on different subject subsets, and C1 holds across split seeds. | 25-refit ensemble + 5 split seeds | LOCKED |
| C3 | **Coupling estimate is reliable within a person.** The per-session estimate of the coupling gain agrees between sessions 2–3 months apart. | ~42 subjects with both sessions | LOCKED |
| C4 | **Short-window training generalizes to longer windows.** A model trained only on 2-s windows reproduces the spectral structure of 5-s and 10-s windows about as well as it reproduces 2-s windows. | Held-out subjects | LOCKED (condition in §10.2) |

## 1.3 What is explicitly NOT being claimed

- **Not a universal law of the brain.** The “time-invariant” idea is scoped down to C2–C4: stability across refits, sessions and window lengths, within eyes-closed resting-state EEG in healthy adults. It does not generalize to eyes-open, tasks, sleep, clinical populations or other species.

- **Not a claim that the coupling equation's form is stable across sessions.** One equation is fit once and frozen; what C3 tests is whether the coupling **coefficient** re-estimated per session is reliable.

- **Not literal recovery of excitatory and inhibitory populations.** The E and I states are latent variables of a model, estimated from a mixed signal. They are model quantities, not measurements.

- **Not a first application of equation discovery to EEG.** See §2.

## 1.4 The leading explanation for a null result, stated in advance

Three separate audits point the same way: resting, eyes-closed EEG may simply be too uninformative to pin down a coupling function. Near-linear resting regimes make coupling forms inseparable (Delabays et al. 2025, SR audit); eyes-open/eyes-closed contrasts help break parameter degeneracies but this dataset is eyes-closed only (NMM audit); and the coupling audit recommends richer dynamics (state transitions) for coupling identification. **If C1 or C2 fails, this is the pre-stated leading explanation**, not a post-hoc one.

# 2. The Research Gap and How to Frame It

Six separate evidence audits searched for prior work (four on the core novelty, one on dataset/compute feasibility, one on preprocessing); a seventh, later audit covers performance optimization only and does not bear on the novelty claim. The precise gap that survived is:

| **The gap** No published study has performed **time-domain** symbolic regression on **real EEG** using a **neural-mass skeleton**, with a **refinement of the coupling term** between two regions as the discoverable element, and with the coupling estimate **formally tested for reliability** across refits, window lengths and recording sessions. |
| --- |

## 2.1 The piece-by-piece evidence

| **Component** | **Closest existing work** | **What remains open** |
| --- | --- | --- |
| Equation discovery on EEG | Omejc et al. 2026 (PLOS Comput Biol): grammar-based discovery from neural-mass building blocks, fit to EEG **power spectra**. | Time-domain discovery. PySR has no EEG/MEG/iEEG/LFP precedent. |
| Data-driven coupling on resting EEG | Stankovski et al. 2017: coupling functions reconstructed from real resting EEG in the **phase domain**. Delabays et al. 2025: SINDy-based coupling (hypergraph) reconstruction on resting EEG from 109 subjects, as polynomials, not neural-mass terms. | A neural-mass-structured coupling refined in the time domain on scalp EEG. |
| Skeleton choice | Jansen–Rit fit to real scalp EEG by UKF (Kuhlmann et al. 2016); ancestor model fit by maximum likelihood (Valdes et al. 1999). | No symbolic refinement of a fitted Jansen–Rit skeleton. |
| Coupling term | DCM fixes the coupling form and only compares pre-specified options. | No two-node Jansen–Rit fit to two real scalp channels with coupling identifiability tested; no symbolic refinement of neural-mass coupling from EEG. |
| Reliability / invariance | Acharya et al. 2024: data-driven models of stimulation-evoked iEEG (linear and black-box) had to be retrained per session, though they did generalize across stimulation frequencies. Whether symbolic structure is more or less stable than black-box weights is untested (SR audit). | No test of whether a symbolically refined coupling estimate is stable across windows or sessions. |

## 2.2 Framing rules for the paper

- Cite Omejc et al. (2026) prominently and describe this work as a **time-domain, coupling-focused extension** of it. Cite Stankovski 2017 and Delabays 2025 as the closest data-driven coupling work, and say how this differs. Reviewers will raise all three.

- State in the abstract that the expected honest outcome is **a map of which coupling features are identifiable from two channels**, not a definitive coupling law. The coupling audit said this directly.

- The strongest standalone novelty is the reliability testing (C2–C4), because it is unoccupied. Describe the Acharya precedent accurately: a different signal (evoked iEEG) and different model families, so it is a warning, not a prediction.

# 3. What Changed From the Original Idea, and Why

The original idea was single-channel resting EEG with a Wilson–Cowan skeleton and PySR, 2-s epochs with 1-s overlap, and blind tests on 5-s and 10-s windows. The research changed most of that. Recording these changes is part of the honest record, the same way the Hydra → C. elegans pivot was recorded last time.

| **Original plan** | **Revised plan** | **Why** |
| --- | --- | --- |
| Wilson–Cowan skeleton | **Two-node Jansen–Rit** | Wilson–Cowan has essentially no precedent for direct fitting to scalp EEG and no native EEG observation model. Jansen–Rit was built to generate EEG and has real time-series fitting precedent. |
| One channel split into E and I channels (band, Hilbert or delay embedding) | **E and I are hidden states estimated by UKF model inversion** | No precedent for any split; E and I currents overlap across the whole spectrum of a field signal (Mazzoni 2015; Teleńczuk 2017), so any split would be circular. |
| Two-channel wearable epilepsy montage | **Two independent bipolar derivations from the 64-channel dataset** | Wearables often share an electrode or reference (e.g., UNEEG SubQ middle electrode), which manufactures fake zero-lag coupling (Bastos & Schoffelen 2016). |
| PySR discovers the model | **PySR refines a residual on a fixed coupling form** | Neural-mass parameters are “sloppy”: of 22 Liley parameters fit to 82 subjects, only 1 was individually identifiable, and about 5 parameter combinations were (Hartoyo 2019). Open-form coupling discovery would multiply that problem. |
| UDE escalation as fallback | **PySR only, straight Python; no escalation** | Removes the Python/Julia interop crash. Consequence: a null PySR result is a genuine null, decided in advance (§20). |
| 2-s epochs with 1-s overlap (clinical convention) | **2-s training windows; 2/5/10-s scoring windows** | The binding constraint for derivative-based fitting is points per cycle, not the clinical epoch convention; window length is now a C4 test variable. |
| Deterministic ODE fit | **Stochastic treatment (noise term / likelihood loss)** | Resting EEG is noise-driven; a deterministic fit absorbs noise into spurious terms (de Vries et al. 2026). |

# 4. Dataset: OpenNeuro ds003775

## 4.1 Facts confirmed by the audit

| **Property** | **Value** | **Source** |
| --- | --- | --- |
| Dataset | SRM resting-state EEG, v1.2.1, DOI 10.18112/openneuro.ds003775.v1.2.1 | OpenNeuro / NEMAR |
| Subjects | 111 healthy adults; 42 have a second session (t2) 2–3 months later | Hatlestad-Hall et al. 2022, Data in Brief |
| Recordings | 153 total (111 t1 + 42 t2) | Derived |
| Hardware | 64-channel BioSemi ActiveTwo, ActiView 7.07 (DC-coupled, no online high-pass) | Hatlestad-Hall et al. 2022 |
| Condition | Eyes closed, 4 minutes continuous | Hatlestad-Hall et al. 2022 |
| Sampling rate | **1024 Hz**, no online filters except hardware anti-aliasing | Hatlestad-Hall et al. 2022 (one secondary paper claims 512 Hz; rejected) |
| Samples per channel | 245,760 per 4-minute file | Derived |
| Raw file state | Average-referenced, otherwise unprocessed EDF | Dataset README |
| Cognitive scores | Collected at t1 only | Hatlestad-Hall et al. 2022 |
| Line noise | 50 Hz (Norway) | Dataset paper |

## 4.2 Known problems and how each is handled

| **Issue** | **Handling** | **Status** |
| --- | --- | --- |
| **µV/V units bug.** Raw EDFs are labelled µV but read as V (confirmed by MNE lead developer, mne-bids issue #1044; still unpatched). | Always read with units="uV". Then run a **units check**: high-pass a copy at 0.5 Hz, take the **median** SD across all channels, and require it to lie in **2–200 µV**. This window is wide enough that drift or a few bad electrodes cannot trip it, but a 10⁶ scaling error is still unmistakable. Electrode quality is judged separately by the bad-channel rule in §5.1. A failing recording is excluded and logged; if more than 5% of recordings fail, the pipeline halts, since that indicates a systematic reading problem. | LOCKED |
| **Derivative files are unsuitable.** Pre-cut into 4-s epochs, 1 Hz high-pass already applied. | Ignore the derivative. Preprocess from raw EDF (§5). | LOCKED |
| **sub-010 ses-t1 scans.tsv** has a documented OpenNeuro storage problem. | **LOCKED: keep the EEG data, read the file directly, and skip only the scans.tsv for this subject.** Simplest safe option — the underlying EEG is probably unaffected by a metadata-file bug, and dropping a whole subject over one file is the more conservative loss without being necessary. | LOCKED |
| **Heavy interpolation.** In the official derivative ~12.4% of recordings lost more than a quarter of channels. | Irrelevant to the units check now (median-based). Our own bad-channel detection runs on the four bipolar-pair electrodes only; that per-subject result is tracked. | LOCKED (tracking); policy in §12 |
| **Derivative sampling rate and bad-channel thresholds** could not be read remotely. | Not needed since we preprocess from raw, but open s2_preprocess.m locally for reference. | LOCKED |
| **Downstream exclusions.** LEAD preprint dropped sub-029 and sub-104 (cannot read; not a data-quality reason). Microstate preprint lost 1 subject to QC. | Do not inherit the literacy exclusions. Expect ~1–2% QC loss under our own rules; §20 covers a much higher rate. | LOCKED |

# 5. Preprocessing Pipeline (From Raw EDF)

Because the provided derivative cannot be used, preprocessing is our own and must be documented step by step. A dedicated evidence audit (the preprocessing audit) searched for how preprocessing choices interact with neural-mass/Kalman-filter fitting. The headline finding: **no published study validates any standard EEG artifact-rejection default against Jansen–Rit or state-space parameter recovery.** The choices below are reasoned from the closest indirect evidence, and the pipeline compensates with its own sensitivity gate (§5.2).

## 5.1 Steps

- **Load** each raw EDF with MNE, forcing units="uV". Record the file's SHA-256 checksum.

- **Units check** as defined in §4.2 (median SD after a 0.5 Hz high-pass, 2–200 µV). This is a units test only.

- **Bad-channel detection** on the four electrodes that form the two bipolar pairs only, using gross physiological bounds: RMS amplitude after the §5.1 band-pass below ~5 µV or above ~150 µV, plus flat-line and saturation detection, following Kuhlmann et al. (2016). **Do not** use normality, kurtosis or tight z-score thresholds (see step 8).

- **Line-noise removal** at 50 Hz and harmonics. **LOCKED: standard notch filter.** Simplest, most standard option; nothing in this design specifically needs the spectral preservation around 50 Hz that ZapLine-style methods offer, so there is no reason to take on the extra complexity. The audits do not bear on this choice either way.

- **Band-pass filter: default 0.5 Hz high-pass (PLACEHOLDER; matches Kuhlmann et al. 2016 and the §4.2 units check) to 45 Hz low-pass, zero-phase** (forward-backward or linear-phase FIR). The §5.2 sensitivity check compares this against a 0.1 Hz high-pass, the lower end of the range the audit supports. **Do not use a 1 Hz or 1–2 Hz high-pass**: that cutoff is recommended to improve ICA decomposition (Klug & Gramann 2021), not to preserve the signal being fit. Avoid causal IIR filters, which distort phase and waveform shape (Widmann et al. 2015; Tanner et al. 2015). Because the high-pass removes the model output's nonzero mean, each recording's channels are **rescaled, after bipolar derivation and gross rejection and using that recording's clean samples only, so that their SD equals σ_ref and their mean equals μ_ref**. These two **reference constants are computed once and never tuned to G0 or to data**: simulate the model output y₁−y₂ at the standard §7.3 parameters with p = 220 s⁻¹ and Jansen–Rit's own noise input (uniform 120–320 s⁻¹) for 600 s at 2,048 Hz, then downsample. μ_ref is the mean of the unfiltered output; σ_ref is the SD of the output after the same 0.5–45 Hz band-pass and 256-Hz downsampling as real data. They are stored in config.yml (provenance: computed), so preprocess.py does not depend on model.py at run time. G0's synthetic data go through the identical rescaling. **Consequence, disclosed:** because the observation gain is fixed at 1 and this reference anchors the mean, the estimated p is only interpretable relative to that reference; the E/I terms and coupling gains are the quantities of interest. An observation-offset state is not used, since it would add states beyond the locked 19. **Reasoned, not tested:** the preprocessing audit's own inference is that filtering adds a moving-average component that conflicts with the UKF's white-observation-noise assumption (it cites Barnett & Seth 2011 for the general filtering effect). Document the filter's impulse response for this reason.

- **Build bipolar derivations** from the four raw electrodes (§6). Because each bipolar channel is a difference of two electrodes, the common average reference cancels within each channel.

- **Ocular/muscle artifact correction: no ICLabel-style whole-component ICA rejection.** With only two bipolar channels, ICA is close to unusable because decomposition quality degrades sharply with few channels (Liu & Zhang 2022), and deleting whole components discards genuine neural activity that leaks into components labelled as artifactual (Castellanos & Makarov 2006; Issa & Juhasz 2019). Instead, use **segment-localized correction** (EMD- or wavelet-based correction applied only to detected blink/saccade windows) or EOG regression if an EOG channel exists. **Defaults (PLACEHOLDER):** flag blink-like windows where a 0.5–5 Hz copy exceeds 6 robust SDs (MAD-based) and correct ±0.25 s around each with the wavelet method (the pilot may compare EMD). P3–PO3 and P4–PO4 are far from the eyes, so this step is mostly a safeguard. This follows the approach of Yokoyama & Kitajo (2023), which is the nearest published use of a bipolar scalp channel with Kalman-type assimilation but differs substantially from this design (single Pz–Oz channel, sleep data, ensemble Kalman filter, 0.6–20 Hz band); the E/I audit also listed it as surfaced but not vetted. Treat it as a pointer, not a validated precedent.

- **Artifact/segment rejection: reject only gross, non-physiological events** (saturation, flat-lines, electrode pops, RMS far outside 5–150 µV, tested on 1-s segments) plus a **gross EMG screen**: flag segments whose 70–110 Hz power (computed before the low-pass) exceeds a fixed multiple of the recording's median (default 5× on 1-s segments, PLACEHOLDER), as Kuhlmann et al. (2016) screened 70–110 Hz power. Every rejected event is padded by 0.5 s on each side, and clean stretches shorter than 5 s afterwards are discarded (PLACEHOLDER). EMG (~20–300 Hz) overlaps high-frequency neural activity (Muthukumaraswamy 2013), so the low-pass alone is not relied on. **Do not reject on normality, kurtosis or Lilliefors-style tests.** Reasoned, not tested: the preprocessing audit's own inference is that such rejection preferentially removes non-Gaussian, high-amplitude epochs, which is where a sigmoidal model differs most from a linear one. Record a simple vigilance proxy (alpha 8–13 Hz over theta 4–8 Hz power, per 2-s epoch) as a covariate, used in C3 (§11.3), not for rejection (Østergaard et al. 2024).

- **Anti-alias and downsample** from 1024 Hz to **256 Hz** for the observation stream (see §10.1 for the numerical-accuracy handling inside the UKF).

- **Segment handling.** Keep each clean stretch between rejected segments as a continuous segment. The UKF is (re)initialized at the start of each clean segment and never run across a rejected gap (Kuhlmann et al. 2016). Windowing for training and scoring is defined in §10.2.

## 5.2 Mandatory preprocessing-sensitivity gate (runs with G0 in §9)

Because no published protocol is validated for this task, the defaults are tested rather than trusted. The gate targets **the quantities this pipeline actually estimates** (§7.4), not parameters it holds fixed.

- **Positive arm.** Simulate the two-node model (per §9.1) with known p, E/I terms and coupling gains; add artifacts at these PLACEHOLDER settings (in µV, after scaling the synthetic channel to the training subjects' median bipolar SD): blink-like transients (0.3 s, 30 µV, about 1 per 15 s), EMG bursts (20–150 Hz, 1–3 s, RMS 30% of the channel SD, about 3 per minute) and slow drift (0.05 Hz, 50 µV); run the exact real preprocessing. **Pass:** the bias of the coupling gains and the E/I terms relative to truth is within ±15% (PLACEHOLDER; p is excluded because it is only meaningful relative to the fixed rescaling reference, §5.1), and the PySR residual still meets the §9.1 NRMSE tolerance.

- **Artifact-only null arm.** Zero coupling, mixing present, plus **bilateral, near-zero-lag** artifacts (blinks and EMG hitting both channels). **Pass:** same criterion as the §9.2 null (no residual selected and both estimated gains within δ of zero). This tests whether artifacts surviving correction can create coupling, which neither the v0.1 gate nor the v0.1 null did.

- On real data, report parameter sensitivity across **two filter settings and two rejection strictnesses**, stating which estimates are robust to these choices. **Variants:** high-pass 0.5 vs 0.1 Hz; rejection default (EMG multiple 5×, RMS upper bound 150 µV) vs strict (3×, 100 µV). **What is refit:** the continuous M2 UKF only (no PySR), on 20 training subjects under each of the 4 variants. It is owned by robustness.py, which calls preprocess.py with variant flags.

- **Default: hard rejection (LOCKED as the starting point).** Simpler to implement and reason about than the noise-adaptive alternative. A **noise-adaptive observation covariance** in the UKF (the Yokoyama & Kitajo idea) is compared against it in the pilot; this is the one remaining item that is genuinely pilot-dependent rather than decidable in advance, since it requires seeing how each behaves on real data.

- Failure path: §20.

# 6. Channel and Montage Selection

| **The rule** Two **bipolar derivations**, **bilaterally separated**, with **no shared electrode** and **no shared reference**. One pair over each hemisphere at homologous sites. |
| --- |

This follows Bastos and Schoffelen (2016), who show that bipolar derivations largely remove the common-reference problem, and the coupling audit's finding that shared-reference or closely spaced unilateral pairs build in spurious zero-lag coupling.

| **Candidate** | **Left pair** | **Right pair** | **Notes** |
| --- | --- | --- | --- |
| Option A: temporal | T7 – TP7 | T8 – TP8 | Homologous temporal cortex; plausible callosal coupling; watch for temporalis EMG. |
| Option B: central | C3 – CP3 | C4 – CP4 | Lower EMG risk; strong resting alpha/mu content. |
| Option C: parieto-occipital | P3 – PO3 | P4 – PO4 | Strongest eyes-closed alpha, which helps Jansen–Rit fitting. |

**Status: LOCKED. Option C (parieto-occipital: P3–PO3, P4–PO4).** Chosen for the simplest, safest reason available: eyes-closed resting EEG has its strongest, cleanest alpha signal at parieto-occipital sites, which gives the model the easiest possible signal to fit and the least risk of a boring, avoidable failure. The audit found **no peer-reviewed literature tying any specific electrode pair to an expected physiological coupling**, so the paper must say this plainly: this is a pragmatic signal-quality choice, not a claim about where coupling is expected to be strongest. Check the exact labels against the dataset's channels.tsv before use.

# 7. Model Skeleton: Two-Node Jansen–Rit

## 7.1 Structure

Each node is a standard Jansen–Rit cortical column: three populations (pyramidal cells, excitatory interneurons, inhibitory interneurons), linear second-order synaptic kernels, and one static sigmoid. That gives 6 state variables per node, 12 for the two-node system. The model output per node, the pyramidal postsynaptic potential difference y₁ − y₂, is the EEG proxy.

This structure gives the natural split the project needs: **the linear synaptic kernels stay fixed and physiologically anchored, and only the coupling nonlinearity is open to refinement.**

**Notation and equations (verified in v0.6).** The six-state form used throughout is the Grimbert & Faugeras (2006) form, also written out with delayed coupling by Escuain-Poole et al. (2018, eqs. 4–6): ẏ₀ = y₃; ẏ₃ = A·a·S(y₁−y₂) − 2a·y₃ − a²·y₀; ẏ₁ = y₄; ẏ₄ = A·a·[p + C₂·S(C₁·y₀)] − 2a·y₄ − a²·y₁; ẏ₂ = y₅; ẏ₅ = B·b·C₄·S(C₃·y₀) − 2b·y₅ − b²·y₂. Here y₀ is the PSP evoked on the interneuron populations by pyramidal firing, y₁ and y₂ are the excitatory and inhibitory PSPs on the pyramidal population, and the EEG proxy is **y₁ − y₂**. One secondary source labels y₀ as the output; three others, including Escuain-Poole, use y₁ − y₂, which is what this document uses.

## 7.2 Observation model

- Each bipolar channel observes its node's y₁ − y₂ after per-channel mean/SD rescaling (§5.1); the observation gain is therefore **fixed at 1**, not estimated.

- An explicit **2×2 instantaneous mixing matrix** absorbs residual volume conduction between channels. It is symmetric with unit diagonal and one off-diagonal leakage parameter **m**, which is estimated (§7.4). Without it, zero-lag shared variance is misread as coupling.

- **LOCKED: no shared common-input term in the fitted model.** Simplest, safest choice given the 19-D state is already at the edge of numerical stability (§7.5) — adding a further parameter would only increase that risk. Regardless of this choice, the synthetic null in §9.2 always includes a common-input arm, so the confound is still tested for even though the fitted model doesn't carry a dedicated term for it. If real-data residuals later look structured in a way this can't explain, this is the first place to revisit, as a recorded deviation.

- Additive observation noise and stochastic input drive (§3, stochastic treatment).

## 7.3 Starting parameter values

Standard values from Jansen and Rit (1995). **Verified in v0.6** against independent sources that reproduce the original (the Escuain-Poole et al. 2018 parameter table, Grimbert & Faugeras 2006 via secondary sources, a 2024 parameter-inference paper, and reference simulation code); the 1995 paper itself was not read. Record provenance per parameter in config.yml (literature / placeholder / fitted).

| **Parameter** | **Meaning** | **Standard value** | **Provenance** |
| --- | --- | --- | --- |
| A | Max excitatory PSP amplitude | 3.25 mV | Jansen & Rit 1995 |
| B | Max inhibitory PSP amplitude | 22 mV | Jansen & Rit 1995 |
| a | Excitatory rate constant | 100 s⁻¹ | Jansen & Rit 1995 |
| b | Inhibitory rate constant | 50 s⁻¹ | Jansen & Rit 1995 |
| C | Connectivity constant | 135 | Jansen & Rit 1995 |
| C₁, C₂, C₃, C₄ | Intra-column connectivities | C, 0.8C, 0.25C, 0.25C | Jansen & Rit 1995 |
| e₀ | Half of max firing rate | 2.5 s⁻¹ | Jansen & Rit 1995 |
| v₀ | Sigmoid midpoint potential | 6 mV | Jansen & Rit 1995 |
| r | Sigmoid steepness | 0.56 mV⁻¹ | Jansen & Rit 1995 |
| p | Mean external input | 220 s⁻¹ (mean of Jansen–Rit's uniform 120–320 input noise); estimated | Jansen & Rit 1995 via Grimbert & Faugeras 2006 |
| Inter-node delay d | Conduction delay, both directions | 10 ms, fixed | PLACEHOLDER |
| Inter-node gains g₁₂, g₂₁ | Base coupling strength | Prior N(0, (0.1·C₂)²), SD ≈ 10.8 s⁻¹; estimated | Fitted |

**Sigmoid (LOCKED, verified):** S(v) = 2e₀ / (1 + exp(r·(v₀ − v))) with e₀ = 2.5 s⁻¹, so the maximum firing rate is 2e₀ = 5 s⁻¹ (the form in Escuain-Poole et al. 2018, eq. 3). Some published versions subtract a constant so that S(0) = 0; this project does not, and the choice is fixed here so that code, G0 and the C₂ scale in §9.1 all agree. **Input noise:** Jansen and Rit drove p with uniform noise between 120 and 320 pulses per second (mean 220). The estimated p in §7.4 is the mean input; the stochastic drive is the process noise (§7.6).

## 7.4 Which quantities are estimated (LOCKED)

v0.1 left this open, which made the G0 null ambiguous. It is now fixed:

| **Quantity** | **Estimated?** | **How** |
| --- | --- | --- |
| 12 Jansen–Rit states | Yes | UKF state |
| p₁, p₂ (input mean per node) | Yes | Augmented state, slow random walk |
| E/I balance term per node (A/B reparameterized as ratio × fixed scale) | Yes | Augmented state, slow random walk |
| g₁₂, g₂₁ (base coupling gains) | Yes | Augmented state; **this is the coupling coefficient used in C3** |
| m (mixing leakage) | Yes | Augmented state |
| Delay d | No | Fixed (PLACEHOLDER value, §7.3) |
| Observation gain | No | Fixed at 1 after rescaling |
| All other §7.3 parameters | No | Fixed at literature values |

That is a **19-dimensional augmented state** (12 + 7). The preprocessing audit notes that Liang et al. (2023) found four-parameter UKF tracking less accurate than three-parameter, so if G0's identifiability check (§9.3) shows the set is not recoverable, reduce it in this preregistered order, and record the change: (1) fix the E/I terms at literature values; (2) tie p₁ = p₂; (3) tie g₁₂ = g₂₁. The coupling gains and m are never dropped, because G0's null logic depends on them.

Reparameterizing A and B as an E/I ratio plus a scale follows arXiv 2406.05002 (the two are nearly mirror images in effect). Known warning: in Kuhlmann et al. 2016 the inhibitory PSP amplitude tracked propofol as expected but the inhibitory rate constant did not.

## 7.5 State estimation (LOCKED design; PLACEHOLDER values)

- **Filter:** unscented Kalman filter on the 19-D augmented state (Kuhlmann et al. 2016). The reference implementation is filterpy's standard scaled unscented transform. **This is a real numerical-stability risk, not a style choice.** Multiple sources report that the classical scaled unscented transform becomes unreliable as state dimension grows, with divergence and negative sigma-point weights reported once dimension exceeds roughly 20 (Chang et al.; the power-system UKF literature on dimensions of 20+). At 19 dimensions this design sits right at that edge.

- **Performance path (§16.5):** filterpy is pure NumPy with Python-level dispatch on every predict/update call, and a performance audit found this dispatch overhead, not FLOPs, dominates cost at this state size. The production implementation is therefore a **single Numba-compiled (`@njit`) function reproducing the identical filterpy algorithm** (same sigma-point weights, same Cholesky routine, same summation order, float64 throughout, fastmath=False). This is validated, not assumed: the Numba version's filtered and smoothed means and covariances are compared against filterpy's own output on synthetic data (rtol=1e-8, atol=1e-10, PLACEHOLDER) before it is used on real recordings, and every downstream statistic (gain estimates, selected equation, NRMSE) must be unchanged. The Numba port must replicate filterpy's exact square-root convention (SciPy's default upper-triangular Cholesky, rows of U as the sigma-point offsets; verify against the installed version's source), because a lower-triangular factor gives a different, equally valid sigma-point set and therefore slightly different output. A pure relative tolerance at 1e-12 over ~60,000 recursive steps could fail on harmless rounding differences between LAPACK builds, which is why the tolerance is looser and paired with the downstream check. If that check fails, fall back to plain filterpy and treat the speedup as unavailable rather than silently accepting a different filter.

- **Numerical-stability check, added to G0 (§9.3):** during the synthetic positive-control run, monitor the covariance matrix for negative eigenvalues and monitor for NaN/Inf propagation. If either appears, this is the dimensionality problem surfacing, and the response is to reduce state dimension using the preregistered order in §7.4, or to move to a **square-root UKF** formulation (numerically stabilizes the covariance update without changing the state count), not to rewrite the filter from scratch.

- **Delayed coupling (implementation, LOCKED):** the 10-ms delay is rounded to an integer number of model sub-steps (10 sub-steps at the 1024-Hz effective step ≈ 9.77 ms). The delayed input to node j is read from a **ring buffer of the sigmoid output S of node i's posterior-mean pyramidal potential (the filtered mean at every 4th sub-step, the predicted mean at the sub-steps in between)**, held outside the sigma points and treated as a known exogenous input at each step (its uncertainty is ignored). This keeps the state at 19-D instead of adding lagged states past the stability edge above. It is an approximation, and G0 tests it directly because the synthetic data are generated with the true delay at 2,048 Hz. In the smoother pass the buffer holds the forward-filter means, not smoothed values. After a (re)initialization the buffer is filled with the steady-state value of S.

- **Parameter handling (v0.6, LOCKED).** The slow parameters (p₁, p₂, the two E/I terms, g₁₂, g₂₁, m) are recording-level constants by model design. They are therefore estimated **once per recording**, by a continuous forward-filter + smoother pass over the recording's clean segments; the recording-level value is the mean of the smoothed parameter trajectory over the post-burn-in samples. The PySR training windows (§10.2) then **hold these values fixed** and re-initialize only the 12 neural states. Estimating the gains inside each 2-s window instead would leave them dominated by their prior (centred on 0), so the residual formed in §8.3 would contain a leftover copy of the base coupling's own shape, which PySR would find in every refit and C2 would count as a stable discovery.

- **Divergence (defined).** A segment is diverged if it contains a NaN/Inf, a covariance that is not positive definite after adding 1e-9·I, or any neural state beyond 10× its steady-state SD. A recording counts as diverged if more than 10% of its clean samples lie in diverged segments; otherwise the diverged segments are dropped and logged.

- **Smoother for training targets:** an unscented Rauch–Tung–Striebel smoother (as in Freestone et al. 2011) produces the state trajectories PySR is trained on, because it uses the whole segment and gives cleaner targets.

- **Forward filter for scoring:** held-out evaluation uses the forward filter only, so scores reflect causal prediction.

- **Test scoring re-runs the filter with each model variant** (§13): base coupling, and base coupling plus the frozen PySR residual.

- **Q and R are set by a preregistered rule, never hand-tuned on real data.** Q (process noise) and R (observation noise) are chosen on **training subjects only**, using the base-coupling model without any PySR term, by the value that makes the normalized innovation squared statistic closest to its expected value of 2, the observation dimension (innovation consistency). The Q/R family is fixed in §7.6, so the rule has exactly one free scale q. They are then frozen and used unchanged for PySR training and all test scoring. **Ordering (v0.5):** G0 runs first and never uses real-data Q/R. G0 tunes its own Q/R with the same rule on a separate **G0 tuning set of 20 extra synthetic series** (generated like the positive control at a mid coupling level, fresh seeds, never used for pass/fail), then freezes it for all G0 series. Real-data Q/R is tuned only **after G0 passes**, on the training subjects of each split (so it is **re-tuned per split seed**, never touching that seed's test subjects). Any change is a deviation.

## 7.6 Filter and simulation defaults (PLACEHOLDER)

Every value here is a PLACEHOLDER default chosen so that code can be written. The pilot may adjust them, G0 (§9) is the check that they work, and they live in config.yml with provenance placeholder.

| **Item** | **Default** | **Note** |
| --- | --- | --- |
| Observation noise R | R = σ_R²·I₂ with σ_R² = 25% of the rescaled observation variance (0.25·σ_ref²) | Fixed share, so NIS tunes one scale only |
| Process noise Q, neural states | Q = q·diag(steady-state variance of each state) per observation step; one scale q on an 8-value log grid from 1e-4 to 1e-1, chosen by the §7.5 NIS rule on 20 random training subjects per split seed | One free scale, one equation |
| Process noise Q, parameters | Slow random walks, variance per second = 1e-3 × prior variance | Keeps parameters slow |
| Initial neural states | Deterministic steady state at the prior parameter means; covariance = per-state variance from a 600-s noise-driven simulation at literature parameters; used for every (re)initialization |  |
| Parameter priors | p ~ N(220, 50²) s⁻¹ per node; log ρ ~ N(log(A₀/B₀), 0.2²) with ρ = A/B and the product A·B held at 3.25 × 22 (A = √(A·B·ρ), B = √(A·B/ρ)); g₁₂, g₂₁ ~ N(0, (0.1·C₂)²); m ~ N(0.2, 0.15²) truncated to [0, 0.5] | Defines the E/I reparameterization; gain prior scaled to the published medium coupling |
| Sigma points | α = 1, β = 2, κ = 0 (scaled unscented transform). At n = 19: centre weight 0, all other weights 1/38, no negative weights, spread √19 ≈ 4.4σ | α = 1e-3 would give a centre weight near −10⁶ and α = 0.5 gives −3. The wide spread is watched by the §9.3 stability check |
| Integrator | Heun (second order), 4 sub-steps per 256-Hz observation, process noise added once per observation step | As in Escuain-Poole et al. 2018; the pilot compares 1, 2, 4 sub-steps (§10.1) |
| M1 versus M2 filters | M1 removes g₁₂, g₂₁ from the state (17-D) and reuses M2's Q/R; M2 and M3 keep them (19-D) |  |
| Delay | d = 10 sub-steps (≈ 9.77 ms), fixed; buffer defined in §7.5 |  |
| Rescaling constants | μ_ref, σ_ref computed once (§5.1) and stored in config.yml | Not tuned to G0 or data |

# 8. Coupling Term and PySR Configuration

## 8.1 Refine, do not discover from scratch

**Base coupling (fixed form):** the sigmoid-transformed pyramidal output of node i, delayed by d, drives the excitatory input of node j with gain gᵢⱼ, in both directions. This is the standard form from Jansen and Rit (1995), David and Friston (2003) and DCM, and it is exactly the coupling written by Escuain-Poole et al. (2018, eq. 5): a delayed pyramidal sigmoid output, times a gain, added in the same bracket as the external input p. The gains are estimated by the UKF (§7.4).

**PySR target:** only a **residual correction** to that coupling, defined operationally in §8.3. Full open-form discovery of the coupling was considered and **rejected** because the identifiability literature does not support it.

## 8.2 PySR settings

| **Setting** | **Value** | **Status** |
| --- | --- | --- |
| Library version | Pin exact PySR version at setup | LOCKED |
| Binary operators | +, −, ×, / | PLACEHOLDER |
| Unary operators | tanh, exp (sigmoid-like shapes) | PLACEHOLDER |
| Candidate inputs | Normalized pyramidal potentials of the target node (at t) and the source node (at t−d), S of the source potential at t−d, and their pairwise products. Target and inputs are **z-scored with the fit fold's mean and SD** (constants stored with the frozen equation and applied unchanged to validation and test). | PLACEHOLDER |
| Parsimony penalty | Start near 0.01 (C. elegans value), recalibrate on G0 using fresh seeds | PLACEHOLDER |
| procs / populations | procs = worker count allowed by the core cap (8 logical CPUs). **populations left at PySR default** (§16.5.3), which for the pinned version already exceeds 2 × procs; verify at setup. | LOCKED |
| Timeout per fit | Primary fit 1,800 s; ensemble refits 600 s; synthetic fits 120 s | PLACEHOLDER |
| Seed / reproducibility | **Real fits run in parallel** (procs as above). Parallel PySR is not bit-reproducible even with a fixed seed (confirmed in a peer-reviewed PySR review), so the seed is recorded but "same equations found" is defined as the same frozen Pareto front and matching downstream statistics on synthetic data, not bit-identical output (§16.5.4). **Serial mode** (parallelism="serial", one thread) is used only for the pilot's small determinism checks, and those use a fixed niterations on a small dataset, not a timeout, since timeouts are themselves non-deterministic. | LOCKED |
| **turbo (§16.5)** | Try turbo=True for the ~20%+ speedup the maintainer reports (no published methodology behind the figure). Verify, with a fixed-iteration serial run on a small synthetic dataset, that the Pareto front is the same with and without it before using it on real data, since SIMD reordering can perturb losses. | PLACEHOLDER (verify in pilot) |
| Warm session (§16.5) | Reuse one long-lived Python/Julia process for every PySR call (primary fit, 25-refit ensemble, 5 split seeds) so JIT compilation is paid once, not per call. Structural change; cannot affect results. | LOCKED |
| **Equation selection from the Pareto front** | **Fit on 80% of training subjects, select, freeze; no refit on 100%.** The remaining 20% of training subjects form the validation fold. Pick the **simplest** Pareto equation whose validation loss is within 5% of the minimum validation loss, then freeze that equation. For G0, where each series is one recording and has no subjects, the validation fold is the **last 20% of the recording's 2-s windows in time order**. **A bare constant counts as "no residual term"** (§9.2, §20). Test subjects are never used. | LOCKED (rule); PLACEHOLDER (5%) |
| Primary fit | One fit on the 80% fit fold of training subjects (Pareto selection on the 20% validation fold, then freeze); the frozen equation is used for C1, C3, C4 | LOCKED |
| **Refit ensemble (for C2 only)** | 25 refits, each on a random half of the training subjects, 600 s each (~4.2 h). Each refit uses the same selection rule. This is the only place PySR is refit. | LOCKED |
| Refit inside the bootstrap | **Never.** Confidence intervals for C1 come from re-scoring the frozen equation over subject resamples. | LOCKED |
| Large inputs (subsampling) | Roughly 3.6M training rows exist (78 subjects × ~120 windows × 384 post-burn-in samples), doubled by pooling both nodes. Each fit sees a **fixed-size random subsample of 50,000 rows** (PLACEHOLDER), drawn **with equal rows per subject** from that fit's training subjects and a **fixed recorded seed**; the validation loss uses a separate fixed 50,000-row subsample of the validation subjects. PySR's own batching option is **not used** (§16.5.3): the subsample is applied before PySR sees the data. | LOCKED (scheme); PLACEHOLDER (50,000) |

## 8.3 The regression target, defined

None of the audits covers how to form a residual from filtered states (the coupling audit notes no study solves the partial-observation problem), so this is a design choice, validated by G0 rather than by literature.

**Derivative estimator: weak/integral formulation, primary; total-variation regularization, comparison baseline.** Point-wise derivative estimation is ill-posed under noise, and the error grows with signal frequency (directly relevant given the points-per-cycle limits in §10.1); weak-form and integral formulations avoid computing a point-wise derivative at all by smoothing through integration instead, and are reported to be more robust to noise than TV, splines or finite differences (Weak SINDy; WmSINDy literature). TV regularization is the older, still-standard baseline (it is what the original SINDy paper and Kato et al.'s own convention used), so it remains the comparison point, not the default. **Both are run on the pilot's synthetic set (§17), not on the full G0 series**, so the choice is never made on the series that certify the nulls and G0's cost is not doubled. The one with better NRMSE recovery at the real noise level and sampling rate is locked before the full run and used for real data. Changing it after the pilot is a recorded deviation.

**Definitions (v0.6).** *Primary, the test-function-kernel derivative:* dy₄/dt is estimated as ∫ φ′(τ−t)·y₄(τ) dτ, i.e. the smoothed y₄ trajectory convolved with the derivative of a compactly supported Gaussian-window test function φ (by integration by parts this equals −∫ φ(τ−t)·ẏ₄ dτ, so no point-wise difference is ever taken). Support 100 ms (PLACEHOLDER; the pilot compares 60, 100 and 160 ms). *Comparison:* TV-regularized differentiation with the regularization weight chosen on the pilot synthetic set from a fixed 5-value grid. In both cases PySR then fits ordinary mean squared error to the resulting target; **no custom weak-form PySR loss is used**.

- **Two passes on the training data (§7.5).** First a continuous forward-filter + smoother pass with the **base-coupling model** (fixed-form coupling, gains as slow states) gives each recording's parameter values. Then, with those parameters held fixed, the UKF + smoother is run per 2-s window (12 neural states re-initialized) to give the state trajectories.

- **Which state (notation: the standard six-state Jansen–Rit form of Grimbert & Faugeras, y₀…y₅, output y₁−y₂).** External input and the base coupling enter the equation for the velocity state **y₄ = ẏ₁** (the excitatory PSP onto the pyramidal population), i.e. ẏ₄ = A·a·[p + coupling + C₂·S(C₁·y₀)] − 2a·y₄ − a²·y₁. For each node j, compute dy₄/dt from the smoothed y₄ trajectory with the estimator selected above.

- Subtract the base model's predicted dy₄/dt, including the base coupling term at the recording-level estimated gain (§7.5). The difference is the **residual rⱼ(t)**, the only thing PySR fits.

- Rows come only from the 2-s training windows defined in §10.2, with the burn-in after each filter initialization discarded.

## 8.4 One shared equation, and what counts as a term

- **One shared residual equation for both nodes.** Rows from both directions (node 1 driving node 2, and node 2 driving node 1) are pooled and PySR fits a single equation in source-node and target-node input variables. This doubles the data, keeps a “term” well defined for C2, and avoids adding a second free equation to a weakly identifiable problem. It assumes the two directions share a functional form (the gains g₁₂ and g₂₁ still differ); asymmetric residual forms are not explored, and this is stated as a limitation.

- **Term signature (preregistered matching rule for C2).** Expand each selected equation with sympy into additive components. For each component, set every multiplicative numeric constant to 1 and every additive numeric constant inside a function argument to 0, then simplify; the result is that component's **signature** (so tanh(1.02·x₁ + 0.01) and tanh(x₁) share the signature tanh(x₁)). A signature “appears” in a refit if any component of that refit's selected equation has it. A bare constant has no signature, i.e. it is “no residual term”.

- **Edge case, accepted:** constants inside denominators are kept literally, so x/(x+0.3) and x/(x+0.5) get different signatures.

# 9. Synthetic-Validation Gate (G0)

This runs **before any real EEG is fit**. It caught a real problem last time (both FHN targets failed at NRMSE 0.72 and 1.55), and this time it has two harder null tests. The closest precedent predicts difficulty: with one channel per node, Escuain-Poole et al. (2018) found the UKF performed poorly except for an input-free column, because a single channel could not separate a node's own dynamics from its afferent coupling. This design is exactly one channel per node, so a G0 failure would be informative, not surprising.

## 9.1 Positive control

- **Generation rate:** simulate at **2,048 Hz** with a fine stochastic integrator (e.g., Euler–Maruyama or stochastic Heun at small step), then pass the result through the real pipeline, including anti-alias filtering and downsampling to 256 Hz. This avoids the “inverse crime” of generating and fitting with the same discretization.

- **Coupling-strength grid:** at least four gain levels, including two weak ones near the lower end of plausible values (PLACEHOLDER: 0.02, 0.05, 0.1, 0.25 × C₂ = 0.8·C = 108, i.e. about 2.2, 5.4, 10.8 and 27 s⁻¹; C₂ is the gain of the local excitatory feedback that enters the same equation as the coupling). This brackets the coupling strengths Escuain-Poole et al. (2018) describe as medium (k = 5–10 in that same slot); the top level is about 3–5× that. Each level carries a planted residual (next bullet).

- **Planted residual (defined):** r_res(t) = c·ũ_src(t−d)·ũ_tgt(t), added inside the same bracket as the base coupling, where ũ are the pyramidal potentials y₁−y₂ divided by their SD and c is set per level so that RMS(r_res) = 50% of the RMS of the base coupling term at that level (PLACEHOLDER). The product form is deliberately unlike the base coupling's sigmoid shape, so recovery cannot be a rescaled copy of the base term.

- **Operating regime matched to real data:** each series draws its operating point from a pre-computed grid of simulations over (p, input-noise SD, observation-noise level) (PLACEHOLDER: 12 × 8 × 5), choosing the grid point whose **scale-free** features (alpha peak frequency, relative alpha power = the alpha share of 1–45 Hz power, and aperiodic slope via specparam) match a feature vector drawn at random from the **training subjects only**. Absolute power is not matched, because it depends on the rescaling reference (§5.1). The NMM audit notes Jansen–Rit alpha can arise from noise-driven or limit-cycle regimes (Bastiaens et al. 2025; a non-peer-reviewed preprint places Hopf points near P = 89 and 320 Hz), so report which regime the matched simulations sit in.

- Realistic 1/f plus white noise, mixing m ~ U(0.1, 0.4) per series (all arms), two channels, delay d = 10 ms (≈ 2.6 samples at 256 Hz; resolution of the delay is itself checked).

- **Units:** before the preprocessing steps, each synthetic channel is scaled to the training subjects' median real bipolar SD in µV, so the amplitude-based steps (5–150 µV bounds, EMG screen) behave as on real data; it is then rescaled with the fixed reference like real data (§5.1).

- **Pass (per level):** over a level's 15 series, the **median** coupling-residual NRMSE ≤ 0.25 (PLACEHOLDER, carried over from the C. elegans preregistration) **and** the **median** absolute relative error of the estimated gains ≤ 15% (PLACEHOLDER; the same criterion as §15), at every coupling level from the second-weakest up; the weakest level is reported as the detection floor, not required to pass. **NRMSE definition:** the recovered residual function and the planted residual are both evaluated on the recording's true simulated input trajectories, and the RMS difference is divided by the SD of the planted residual. **Failure handling:** a positive-control failure does not stop the run; it sets a low_confidence flag (see the single gate rule in §18).

- **Parsimony recalibration and rows (defined):** the parsimony penalty is chosen from {0.003, 0.01, 0.03} as the value with the lowest median NRMSE on the **pilot's** positive series (ties: the larger penalty) and frozen before the full G0; it is never tuned on the full G0 series. A G0 series yields about 73,700 training rows (subsampled to 50,000 as in §8.2) and about 18,400 validation rows (all used).

## 9.2 Null controls (two arms, both mandatory)

- **Null A (mixing):** zero coupling (g₁₂ = g₂₁ = 0), nonzero mixing.

- **Null B (common input):** zero coupling, nonzero mixing, plus a **shared stochastic input** to both nodes with a random lag between 0 and 20 ms per series, carrying 30–50% (uniform per series, PLACEHOLDER) of each node's total input variance. wPLI and imaginary coherency cannot catch this confound because a lagged common drive is not zero-lag, and per-channel surrogates destroy coupling and common input alike, so it must be tested directly.

- **Pass for a null series:** PySR selects no residual term (under the §8.2 selection rule; a bare constant counts as none) **AND** both estimated gains are within **δ of zero**. **The estimator, defined once before any null result exists:** the mean of the forward-filtered gain trajectory over the post-burn-in recording; the same estimator is used for C3. **δ = half the weakest planted coupling level** (PLACEHOLDER: about 1.1 s⁻¹ for the 2.2 s⁻¹ level); the pilot may raise δ but never above the weakest planted level, and it is fixed before the full G0. **Why this replaced an interval test:** a series passes an “interval includes 0” test only 95% of the time even for a correct pipeline, so with two gains and 0-of-60 required, a correct pipeline would pass an arm only about 0.2% of the time (4.6% if the two gains moved together), and the 20-series artifact-only null only 13–36% of the time (computed in v0.6). The band rule asks what the gate exists to ask: does the pipeline invent coupling as large as the weakest coupling it is meant to detect. If the estimator's own noise is larger than δ, the gate fails legitimately and §20 applies. Both conditions are needed: without the second, the filter's gain could silently absorb the confound.

- **If either null arm fails, the pipeline is invalid and the run stops (hard stop).** It is fixed before real data is touched (§20).

## 9.3 Identifiability and numerical-stability check

On the positive-control data, compute posterior contraction (prior vs posterior width) for every estimated quantity in §7.4 and a profile of the coupling gain. Hartoyo et al. (2019) found about 5 identifiable parameter **combinations** from 22 Liley parameters, which is the realistic expectation here. **This is a gate, not just a record:** any quantity with contraction below 50% (PLACEHOLDER) triggers the reduction order in §7.4 before real data is fit. Contraction is aggregated by the **median across a level's series**, at every level except the weakest. The coupling gains and m are never dropped (§7.4), so if they fail to contract the outcome is not a further reduction: it is reported as “not identifiable at this level” and sets low_confidence.

**Numerical-stability check (§7.5).** The 19-D augmented state sits close to the dimension range where the classical scaled unscented transform is reported to become unreliable. During this same synthetic run, monitor the filter's covariance matrix for negative eigenvalues and monitor for NaN/Inf propagation. Either is treated as a gate failure with its own fix path (§20): first the §7.4 dimension-reduction order, then a square-root UKF formulation if reduction alone doesn't resolve it.

## 9.4 Size and statistics

Because the pass criterion is about a false-positive rate, the number of null series has to support it. With zero false positives in n series, the one-sided 95% upper bound on the rate is about 3/n, so reaching 5% needs about 60 null series per arm with zero hits.

| **Run** | **Positive series** | **Null A** | **Null B** | **Purpose** |
| --- | --- | --- | --- | --- |
| Pilot | 20 | 20 | 20 | Mechanics only; bound ≈ 14%, not a pass/fail on 5% |
| G0 tuning set (Q/R only) | 20 | — | — | Not scored; fixes G0's own Q/R (§7.5) |
| Full G0 | 60 (15 per coupling level) | 60 | 60 | Pass stated as the 95% upper bound ≤ 5%, i.e. 0 of 60 |
| Preprocessing gate (§5.2) | 20 | 20 (artifact-only) | — | Bias and artifact-coupling check |

Synthetic fits use a 120-s PySR timeout, since each synthetic series is one 4-minute recording. **After any fix to the pipeline, both nulls and the positive control are re-run on fresh seeds**, so the gate is not overfitted to one set of series, and so a penalty change that passes the null cannot silently break positive recovery.

# 10. Sampling, Windows and Epochs

## 10.1 Sampling

| **Decision** | **Value** | **Reason** | **Status** |
| --- | --- | --- | --- |
| Native rate | 1024 Hz | Confirmed from dataset paper | LOCKED |
| Observation rate | 256 Hz | ~4× fewer UKF updates; covers 1–45 Hz | LOCKED |
| Model integration inside the UKF | 4 sub-steps per observation (effective 1024 Hz) | Kuhlmann et al. resampled to 1000 Hz for UKF numerical accuracy (preprocessing audit); sub-stepping keeps that accuracy while observing at 256 Hz | PLACEHOLDER (pilot compares 1, 2 and 4 sub-steps) |
| Points per cycle | 256 Hz gives ~25.6 samples per 10 Hz alpha cycle, below the ~42 found necessary for SINDy on FitzHugh–Nagumo (Prokop & Gelens 2024) | Why smoothed derivatives are required (§8.3) | LOCKED (known risk) |

The pilot's synthetic gate also compares 256 Hz and 512 Hz observation rates. If recovery clearly degrades at 256 Hz, raising the rate is a recorded deviation, at a compute cost.

## 10.2 Training windows and scoring windows (defines C4)

- **Training windows (2 s).** For PySR training, the 12 neural states are (re)initialized at the start of each non-overlapping 2-s window (the slow parameters are held fixed at the recording-level values from §7.5), and the first 0.5 s after initialization is discarded as burn-in (PLACEHOLDER; the pilot checks convergence). No training target's neural-state trajectory therefore uses observations beyond a 2-s span; only the recording-level parameters carry longer-range information, which is by design because they are recording-level constants.

- **C1 scoring.** Held-out one-step prediction error from the **forward filter run continuously** over each clean segment of a test recording: the error between each observed sample and the filter's one-step-ahead prediction, in rescaled units, as mean squared error averaged over the two channels. **The burn-in after each filter (re)initialization is excluded, and the VAR baseline is scored on exactly the same samples**, so M1–M3 are not penalized against M0 for filter convergence. Per-subject values are then averaged over subjects (§15). This is independent of window length, which is why C1 has no epoch-length dimension.

- **C4 scoring (free-run, per 2-s segment).** From the filtered neural state at the start of each 2-s, 5-s and 10-s window (non-overlapping), the model is simulated forward **without observation updates and stochastically**: process noise stays on for the 12 neural states (a noise-free run would decay to a flat spectrum if the alpha is noise-driven, §9.1), parameters stay frozen at their window-start values, and observation noise R is added to the simulated output so the spectral floor matches. Use **20 independent realizations** per window. **Spectral estimator:** Welch with non-overlapping 2-s Hann segments on one common 0.5-Hz grid over 1–45 Hz. **Scoring:** every 2-s segment of a window is scored **separately**: the observed log₁₀ power spectrum of that segment is compared with the mean over the 20 realizations of the simulated log₁₀ spectrum at the same segment position; the segment error is the mean absolute difference over the grid, and a window's error is the mean over its segments. A window of length L therefore contributes L/2 comparable single-periodogram errors. **Why:** averaging spectra over a longer window makes the observed side less noisy, so the long-to-short error ratio came out about 0.38 even for a perfect model (about 0.77 with a 0.5 log₁₀ systematic error), which made C4 impossible to fail. Scoring per segment gives a ratio of about 1.00 for a model that does not drift, whatever its bias, and above 1 if the free-run degrades with time (checked by Monte Carlo in v0.6). C4 therefore tests drift over time, not constant bias.

- **The C4 condition:** C4 is scored only if G0 **fully passes (no `low_confidence` flag)** and at least 80% of test recordings yield a stable free-run (no divergence) at all three window lengths. Otherwise Figure 9 is the labelled “not attempted” placeholder.

# 11. Train/Test Split Design

## 11.1 Primary split

- **Subject-level**, never epoch-level.

- Primary analysis uses **t1 recordings only**, from all 111 subjects (minus exclusions): **78 train / 33 test** (70/30), split by subject ID with a fixed, recorded seed.

- **The split seed is fixed before the pilot: LOCKED at 42** (matching the C. elegans project's convention; the value itself is arbitrary). Pilot subjects are drawn only from the training partition (§17), so no pilot subject can enter the confirmatory test set. **The 4 extra split seeds used for C2(b) are constrained the same way: the 12 pilot subjects are forced onto the training side in every seed** (a small departure from a pure random split, disclosed). **Order (LOCKED):** the split is made once by main.py on all 111 subject IDs **before** any exclusion is applied and written to `outputs/split_<seed>.json`; exclusions (§12) then remove subjects from whichever side they fall on, and the split is never redrawn. This keeps the test set independent of the placeholder values (clean-data minimum, EMG multiple) that the pilot may still adjust.

- A subject's t2 session always goes to the same side as their t1.

## 11.2 Robustness across splits

- Repeat the full primary pipeline with **5 split seeds** (the primary seed plus 4). Each needs its own primary PySR fit (~30 min each). Report C1 per seed.

## 11.3 Session reliability sub-analysis (C3)

- **What is tested:** agreement between sessions of the per-session **coupling-gain estimate** (g₁₂ and g₂₁, §7.4), not of equation structure.

- **Estimator:** for each session, run the UKF with the frozen primary model (base coupling + frozen PySR residual); the session's estimate is the mean of the filtered gain over the recording's clean segments, after burn-in.

- **Statistic:** ICC(3,1), consistency form, over the ~42 subjects with both sessions, reported per direction (g₁₂, g₂₁).

- **In-sample vs out-of-sample:** roughly 70% of these subjects are training subjects whose t1 was in the PySR fit. Report ICC for all pairs (primary) and for test-partition pairs separately (secondary, small n).

- **Vigilance:** as a sensitivity analysis, regress every session's estimate on that session's mean alpha/theta ratio (all sessions pooled), take the residuals, and re-compute the ICC on them. **The unadjusted ICC decides C3 (§15)**; the adjusted one is reported alongside to help explain a failure. Session differences in drowsiness could otherwise masquerade as unreliability (Østergaard et al. 2024).

- Cognitive scores exist only at t1, so any score correlation uses t1 only.

## 11.4 Bootstrap rules

- Resample **subjects**, not epochs. Wherever t1 and t2 appear together, use a **cluster bootstrap** that keeps a subject's sessions together.

- Bootstraps re-score the frozen equation. They never refit PySR (§8.2).

## 11.5 Statistical power

- 33 test subjects: check what any “X% of subjects” criterion can distinguish from chance (binomial test) before fixing it.

- 42 session pairs: an observed ICC of 0.50 has a 95% CI of roughly 0.24–0.70, which is why C3's threshold is stated on the CI (§15).

- G0 nulls: see §9.4.

# 12. Exclusion Rules

All rules are applied uniformly, by code, before results are inspected. No subject is removed case by case after seeing their fit.

| **Rule** | **Action** | **Status** |
| --- | --- | --- |
| Units check fails (§4.2) | Exclude recording and log; halt the pipeline if > 5% of recordings fail | LOCKED |
| Any of the four bipolar-pair electrodes fails the gross bounds (§5.1 step 3) | **LOCKED: exclude, judged per recording.** Simplest and cleanest, with no flagged-covariate analysis to justify later. A failure at **t2 only** drops that t2 recording (subject stays in C1, drops out of C3). A failure at **t1** excludes the subject from C1 and from C3, and their t2 recording is then unused. | LOCKED |
| Fewer than a minimum amount of clean data after gross-artifact rejection | Exclude recording (minimum: PLACEHOLDER, e.g. 60 s) | PLACEHOLDER |
| sub-010 ses-t1 storage problem | LOCKED: see §4.2 (keep EEG, skip scans.tsv) | LOCKED |
| UKF divergence on a recording | **Report divergence counts per model variant** (M1, M2, M3; divergence is defined in §7.5). The primary comparison uses recordings where **all** of M0–M3 succeeded (matched sets). Sensitivity analysis: count any recording where M3 diverges but M2 does not as an **M3 loss** rather than dropping it, so M3's worst cases are not removed from the headline. If M3-only divergences exceed 5% of test recordings, state it in Limitations. | LOCKED |
| t2 recording excluded but t1 kept | Subject stays in C1; drops out of C3 (see the per-recording bad-electrode rule above) | LOCKED |
| Pilot subjects | Remain in the training partition only; never in the test set | LOCKED |

# 13. Baseline and Ablation (Non-Negotiable)

Two separate risks make baselines essential. Nozari et al. (2024) found linear models match or beat nonlinear ones for one-step prediction of resting fMRI/iEEG dynamics. And the closest method precedent itself lost to a linear model: in Kuhlmann et al. (2016), Jansen–Rit + UKF classified anaesthetic state with sensitivity 0.51 versus 0.58 for a linear ARMA model. That comparison is imperfect (propofol induction, not rest; the 0.51 came from a stricter rejection variant that included the normality test this project forbids), but it is a direct warning.

## 13.1 Nested comparison (all scored identically, same subjects, same metric)

| **Step** | **Model** | **What the step-to-step difference tests** |
| --- | --- | --- |
| M0 | Two-channel linear VAR: order 1–32 samples chosen by 5-fold subject-wise cross-validation on training subjects (one-step MSE, fitted within clean segments); coefficients estimated by least squares **pooled over training subjects and frozen** | — |
| M1 | Jansen–Rit, coupling gains fixed at 0 | M0 → M1: is the neural-mass skeleton worth anything? |
| M2 | Jansen–Rit, base coupling (gains estimated) | M1 → M2: does coupling help? |
| M3 | Jansen–Rit, base coupling + frozen PySR residual | **M2 → M3: does the refinement help? (the project's headline question)** |

- **M0b (secondary):** the same VAR order with coefficients **re-fitted on each test recording** itself. This is deliberately generous to the baseline, because the UKF re-estimates p, the E/I terms, the gains and m online on every recording, so a frozen pooled VAR alone would be handicapped. M3 is reported against M0 (primary, preregistered) and M0b (secondary); if M3 beats M0 but not M0b, the paper says so.

- Last project's AVA→AVB sanity check never passed and that surfaced late. These comparisons are built in from day one.

# 14. Robustness and Validation Tests

| **Test** | **What it answers** | **Source** | **Feeds** |
| --- | --- | --- | --- |
| Synthetic gate + two nulls | Does the pipeline recover real coupling and reject mixing and common input? | Coupling audit | G0 |
| Preprocessing gate | Do preprocessing choices bias estimates or create coupling? | Preprocessing audit | G0 |
| Nested ablation M0–M3 | Is the skeleton, the coupling, and the refinement each worth anything? | Nozari 2024; Kuhlmann 2016 | C1 |
| Refit ensemble (25 half-sample refits) | Which residual terms recur across independent refits? | Fasel et al. 2022 (E-SINDy-style stability selection) | C2 |
| Multi-seed splits (5) | Is C1 a lucky partition? | This document | C2 |
| Volume-conduction diagnostic | Is the coupling just shared zero-lag signal? (imaginary coherency, wPLI) | Nolte 2004; Vinck 2011 | C1 interpretation |
| Surrogate test on coupling gain | Does the estimated gain exceed what independent phase-randomized (AAFT) surrogates of each channel give? **10 test recordings** (fixed seed) × **50 surrogates**, each channel randomized separately (this destroys cross-channel coupling); statistic │mean filtered M2 gain│; per-recording p = share of surrogates at least as large; reported as the number of the 10 recordings with p < 0.05. It cannot separate coupling from lagged common input (only Null B does). | As in C. elegans project | C1 interpretation |
| Reliability (ICC) | Is the per-session coupling-gain estimate as reliable as features we already trust? | Politanskaia 2026 (0.51–0.88); Jafarian 2024 | C3 |
| Identifiability profile | Which estimated quantities are recoverable at all? | Hartoyo 2019 | G0, all |

Report within-node parameters and the coupling gain **separately** throughout. The research shows they are not equally recoverable. Note that neither wPLI nor per-channel surrogates can distinguish true coupling from lagged common input; only Null B in §9.2 addresses that.

**Defaults (PLACEHOLDER):** subject bootstrap B = 10,000 (percentile) from a pre-drawn index matrix; ICC confidence interval by the F-based method; wPLI and imaginary coherency in the alpha band (8–13 Hz) and over 1–45 Hz; the Holm family is the four §15.2 bullets; C4 is scored with M3.

# 15. Pass/Fail Criteria

PLACEHOLDER values are finalized after the pilot and **before** full-cohort results are inspected. Any later change is a recorded deviation.

## 15.1 Primary criteria

| **Claim** | **Pass criterion** | **Status** |
| --- | --- | --- |
| G0 positive | Per coupling level (second-weakest and up): **median** coupling-residual NRMSE ≤ 0.25 and **median** absolute relative gain error ≤ 15% over the level's 15 series (definitions in §9.1) | PLACEHOLDER |
| G0 nulls (A and B) | 0 false positives in 60 series per arm (95% upper bound ≤ 5%), where a false positive is a selected residual term (a bare constant does not count) OR either estimated gain with absolute value ≥ δ (§9.2) | LOCKED (structure); PLACEHOLDER (n) |
| Preprocessing gate | Bias ≤ ±15% for the coupling gains and E/I terms (p excluded, §5.1); 0 false positives in the artifact-only null (same rule as the §9.2 nulls) | PLACEHOLDER |
| C1 (primary) | **Primary metric:** subject-mean one-step held-out prediction error under continuous forward filtering (§10.2). **Pass:** M3 lower than M2 AND M3 lower than M0, each with the 95% subject-bootstrap CI of the paired difference excluding zero, on the primary split. | LOCKED |
| C2 | (a) At least one residual-term **signature** (§8.4) appears in ≥ 70% of the 25 ensemble refits, and that signature is in the primary equation; AND (b) C1 passes in ≥ 4 of 5 split seeds | LOCKED (structure); PLACEHOLDER (70%) |
| C3 | ICC(3,1) of per-session coupling-gain estimates, all 42 pairs: 95% CI lower bound ≥ 0.40 in at least one direction (≈ an observed ICC of 0.6 at n = 42; compute exactly before the pilot). **The unadjusted ICC decides;** the vigilance-adjusted ICC is a reported sensitivity analysis (§11.3). | PLACEHOLDER |
| C4 | For each of 5-s and 10-s windows, the **upper 95% subject-bootstrap CI bound of the ratio** (mean per-segment free-run spectral error at that window length) / (mean per-segment error at 2 s) is ≤ 1.20, on held-out subjects (error and grid defined in §10.2; condition in §10.2) | PLACEHOLDER |

## 15.2 Secondary analyses (Holm-corrected, never used to rescue a failed primary)

- C1 on the other steps of the ablation (M0→M1, M1→M2).

- Free-run spectral error per window length (2/5/10 s) as an alternative C1 metric.

- Test-partition-only C3 ICC.

- Split-seed-level C1 results.

# 16. Compute Plan

## 16.1 Hardware and the core cap

The Ryzen 5 5600 is a **6-core / 12-thread** chip. The project caps itself at **4 cores / 8 threads**, a deliberate choice that leaves headroom for the rest of the machine.

| **Two things to check before trusting the cap** **1. Which OS?** The C. elegans run hit a Windows-style EXCEPTION_ACCESS_VIOLATION, which suggests Windows. The audit's recipe (taskset, os.sched_getaffinity) is Linux only. The Windows equivalents are below. **2. “4 cores / 8 threads” is 8 logical CPUs, not 4.** taskset -c 0-3 would give 4 logical CPUs (about 2 physical cores with SMT). To get 4 full cores with both threads each, you need 8 logical CPUs that are SMT sibling pairs. Check the pairing with lscpu -e (Linux) or Task Manager / Coreinfo (Windows) before setting the mask. |
| --- |

## 16.2 Threading recipe

**Critical detail from the audit:** the usual BLAS variables do **not** cap PySR. PySR's Julia backend reads PYTHON_JULIACALL_THREADS, and only at startup, so it must be set before Python imports PySR.

**Linux:**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHON_JULIACALL_THREADS=8
taskset -c <8 logical CPUs = 4 cores + SMT siblings> python main.py
```

**Windows (PowerShell):**

```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
$env:PYTHON_JULIACALL_THREADS="8"
cmd /c start /affinity FF /wait python main.py   # FF = logical CPUs 0-7
```

- Set BLAS variables to 1 inside multiprocessing workers; use more only for single-process BLAS work.

- Size worker pools from the affinity mask: len(os.sched_getaffinity(0)) on Linux, len(psutil.Process().cpu_affinity()) on Windows. Never os.cpu_count().

- Do not rely on docker --cpus; OpenBLAS and libomp ignore cgroup quotas.

- Verify with python -m threadpoolctl -i numpy and by watching thread count while PySR runs.

- PySR still runs Julia under the hood via juliacall. Last time PySR itself ran fine; the crash came from diffeqpy callbacks in UDE escalation, which is now gone.

## 16.3 Time budget (full run)

**Unit correction from v0.1:** the UKF runs as **one coupled 19-D filter per recording**, not one filter per channel. The only published timing (Kuhlmann 2016) is for a 6-D + 3-parameter single-node filter on a 2012 laptop, and cost grows faster than linearly with state dimension, so the UKF row below is the least certain and **must be re-estimated from the pilot**.

| **Stage** | **Workload** | **Estimated wall time** |
| --- | --- | --- |
| Preprocessing | 153 recordings × ~1–2 min | ~1–2 h |
| UKF runs | Q/R tuning; training (2-s windows) on 78 subjects; scoring M1–M3 on 33 test subjects; per-session runs for C3; × 5 split seeds for training-side runs | ~6–20 h (highly uncertain) |
| G0 + preprocessing gate | ~220 scored synthetic series (180 full G0 + 40 preprocessing gate) × (two UKF passes + 120-s PySR), plus 20 tuning series (UKF only) | ~8–15 h |
| Q/R search | 8 grid values × 20 training subjects × 5 split seeds ≈ 800 UKF passes | ~2–7 h |
| AAFT surrogate test | 10 test recordings × 50 surrogates = 500 UKF runs | ~2–8 h |
| §5.2 sensitivity (real data) | 4 preprocessing variants × 20 training subjects = 80 UKF passes | ~0.5–2 h |
| PySR primary fits | 5 split seeds × 30 min | ~2.5 h |
| PySR refit ensemble (C2) | 25 × 600 s | ~4.2 h |
| VAR, bootstraps, ICC, diagnostics | Re-scoring only | < 1 h |
| **Total** |  | **~26–62 h: two to four overnight runs** |

This is larger than v0.1's 8–20 h because v0.1's design could not actually test C2, C3 or the nulls properly. (v0.5: the earlier ~380-series count double-counted; the full-run count is ~220, which lowers the G0 row and the total. v0.6: four workloads that had no line, the Q/R search, the second UKF pass per G0 series, the AAFT test and the §5.2 sensitivity analysis, are now budgeted; every UKF-side row is a guess until the pilot supplies per-run timing.) The G0 row is the easiest to shrink: it can run first, on its own, over one or two nights before any real data is touched.

## 16.4 The settings that would blow up the budget

- Running the UKF on more channels than the two bipolar pairs.

- Refitting PySR inside the bootstrap (≈ days).

- Observing at 1024 Hz instead of 256 Hz (≈ 4× per filter update).

- Growing the refit ensemble or split seeds beyond the numbers above without a recorded deviation.

## 16.5 Performance optimizations (a dedicated audit; the estimates in §16.3 do not assume any of these)

A separate evidence audit searched specifically for ways to reduce wall-clock time **without changing any statistical result**. Its central finding: the largest, safest gains come from **how the stages are structured**, not from swapping libraries. Every change below is sorted into two tiers, and the distinction is load-bearing, not a formality: **structural changes cannot alter results** (they only change how work is scheduled across cores/processes); **numerical changes can move floating-point bits or the search path**, and must be checked against the synthetic gate (G0, §9) before being trusted on real data, the same discipline already applied to every other pipeline choice in this document.

## 16.5.1 Structural changes (safe; adopt directly)

- **Parallelize preprocessing and UKF fitting across recordings**, not within one recording's fit. These 153 recordings are independent, so a 4-worker process pool (joblib, loky backend, n_jobs=4) run over them is expected to approach a 4× speedup on the preprocessing and UKF stages (rows 1–2 of §16.3) — this is an expectation from the audit, not yet a measured result, and should be benchmarked in the pilot at 4 vs 6 vs 8 workers, since SMT threads typically add little for floating-point-heavy code.

- **Pin each worker to a single BLAS thread** (OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1, set before NumPy is imported) and run MNE with n_jobs=1 inside each worker. Without this, total threads become n_jobs × `<LIB>_NUM_THREADS`, oversubscribing the 4-core cap (scikit-learn's own parallelism documentation states this explicitly). This is an extension of the existing §16.2 threading recipe, not a new rule.

- **Never overlap PySR's own stage with the cross-recording pool.** PySR's Julia thread count is fixed once at process start via PYTHON_JULIACALL_THREADS and cannot be changed after launch (confirmed directly by the PySR maintainer). **Never run PySR concurrently with the pool.** The order (v0.6, now including G0) is: preprocessing of all recordings; the pilot-only choices (derivative estimator, sub-steps, Numba validation, worker count); then **G0** (its own Q/R tuning, all its UKF passes in the pool, then all its PySR fits); then, only if G0 allows, the per-seed real-data loop: real Q/R tuning, training-side UKF runs across recordings, all PySR fits as their own stage inside one persistent Julia session (so JIT compilation is paid once), then the scoring-side UKF runs that need the frozen equation (M2/M3 scoring, C3 estimates, the AAFT test), which parallelize across recordings again.

- **Reuse one warm Julia session for every PySR call** (the primary fit, the 25-refit ensemble, all 5 split seeds), rather than starting a fresh process each time. An early PySR user report found a trivial fit took about 2 minutes dominated entirely by environment activation and compilation, not the search itself — this cost is avoidable and is pure overhead with no effect on the result.

- **Vectorize the bootstrap re-scoring and ICC computation as one pass over a pre-drawn resample index matrix**, rather than a Python loop per resample, provided the index matrix is generated by the same RNG calls in the same order. At 33–153 subjects this stage is milliseconds of arithmetic once it is out of a Python loop, so the entire gain is in removing loop overhead, not in any numerical change.

## 16.5.2 Numerical changes (verify against G0 before trusting on real data)

- **Rewrite the filterpy UKF/RTS loop as a single Numba-compiled (`@njit`) function**, keeping the identical algorithm: same sigma-point weights, same Cholesky routine, same summation order, float64 throughout (Numba silently keeps float32 precision in mixed float32/int32 operations where NumPy would promote to float64 — a documented pitfall to avoid explicitly), and fastmath=False (which licenses floating-point reassociation, i.e. a different answer, not just a faster one). This is expected to be the single largest per-fit speedup in the pipeline: at 19 dimensions, filterpy's cost is dominated by Python-level dispatch overhead and small-array allocation, not by FLOPs, and a same-algorithm Numba port removes that overhead directly. **Validation, not optional:** compare the Numba filter's means and covariances against filterpy's output on synthetic data (rtol=1e-8, atol=1e-10, PLACEHOLDER), replicating filterpy's square-root convention (§7.5), and confirm every downstream statistic matches, before using it on real recordings.

- **Try `turbo=True` in PySR** (SIMD loop vectorization; maintainer claims ~20%+, no published methodology behind the number) and confirm a serial-deterministic run returns the same Pareto front with and without it, since reordered floating-point sums can perturb losses and therefore the search path even when nothing else changes.

- **Do not change BLAS backend, FFT backend, or numeric precision casually.** No Zen-3-specific benchmark at this pipeline's actual matrix size (19×19-class operations) was found in the audit; historically MKL under-performed on AMD chips without a since-removed workaround (MKL_DEBUG_CPU_TYPE), and OpenBLAS ships a dedicated Zen kernel, but the honest state of evidence is that this needs to be measured on this machine, not assumed from a general MKL-vs-OpenBLAS reputation. At 19×19 scale, BLAS multi-threading is pure overhead regardless of backend, so the one confident recommendation is: whichever backend is used, pin it to 1 thread per worker process.

## 16.5.3 Explicitly not adopted (would change results, or no benefit at this scale)

- Swapping PySR for a different symbolic-regression library (e.g. Operon, which is reported 5–25× faster single-threaded but finds different equations and was benchmarked on different problems); PyTorch-CPU or JAX for small per-window matrix operations (a documented benchmark found NumPy roughly 15× faster than PyTorch on 3×3-scale operations, and QuantEcon's own guidance is that Numba, not JAX, is the right tool for inherently sequential loops on CPU); Polars in place of pandas (advantage only appears at data scales far larger than this project's); MNE's method="polyphase" resampling (a genuinely different algorithm, not just a faster implementation of the same one); float32 anywhere in the pipeline; and PySR settings that change the search itself (ncycles_per_iteration, populations, batching, fast_cycle) — the maintainer's own guidance is that these are tuned for large-cluster runs and should be left at default for a laptop-class 4–8-thread run.

## 16.5.4 A definitional note carried into every use of these optimizations

“Identical output” has to mean something specific before it can be checked. This document treats it as: the same equation structure and statistically equivalent parameter estimates on synthetic data at the §9 tolerances, not bit-for-bit identical floating-point output. A parallel PySR run is not bit-reproducible even with a fixed seed today (confirmed in a peer-reviewed review of the library: fixing the seed “has no effect when multi-processing is active”), so any claim of “the same equations found” after adopting these optimizations is necessarily a claim about the frozen Pareto front and downstream statistics, checked against the synthetic gate, not about bit-identical numbers. **Real fits therefore run in parallel**; serial mode is used only for the pilot's small fixed-iteration determinism checks (§8.2).

# 17. Pilot Run

The full pipeline runs end to end on a small slice before the full cohort. The pilot is also where the §16.5 performance changes are benchmarked and validated (Numba UKF vs filterpy at rtol=1e-8 / atol=1e-10; 4 vs 6 vs 8 parallel workers; turbo on/off Pareto-front comparison) before they are trusted on the full run.

| **Setting** | **Pilot value** |
| --- | --- |
| Subjects | 12, **all drawn from the training partition of the fixed primary split** (4 of them with t2 sessions); split 8 / 4 internally for mechanics |
| Windows | 2-s training windows; 2-s and 10-s scoring windows |
| Channels | The two bipolar pairs (4 electrodes) |
| Sampling | 256 Hz observation; 1 vs 2 vs 4 UKF sub-steps; one 512 Hz comparison in the synthetic gate |
| Synthetic gate | 20 positive, 20 Null A, 20 Null B, plus a small preprocessing gate |
| PySR | One primary fit at default populations, timeout 600 s; 3 ensemble refits to test the mechanism |
| Bootstrap | 100 subject resamples, re-scoring the frozen equation |
| Gate behaviour | Pilot mode: gates report but never hard-stop, since 20 series cannot support the 5% claim. The pilot also tunes real-data Q/R before the full G0 exists; this is allowed only in pilot mode and its results are discarded (§7.5's after-G0 ordering applies to the full run) |
| Performance benchmarks | Numba UKF vs filterpy (accuracy at the §7.5 tolerance + speed); derivative estimator comparison (weak-form vs TV) on the pilot synthetic set; 4/6/8-worker pool timing; turbo on/off Pareto-front check; all on this pilot's 12 subjects and synthetic series |

## 17.1 What the pilot can validate

- Units check and amplitude scaling; file I/O across sessions.

- Both null arms and the positive control run correctly (not the 5% threshold, which needs the full G0).

- UKF stability and burn-in length on 2-s windows; sub-step accuracy.

- Q/R tuning rule behaves sensibly.

- Thread capping and timeouts actually work.

- Split, ensemble and bootstrap code run without leakage.

- Per-stage timings, used to rescale the full-run budget (especially the UKF row).

## 17.2 What the pilot cannot validate

- The G0 false-positive threshold (bound ≈ 14% at 20 series).

- Cross-subject performance (4 internal test subjects).

- Stable bootstrap CIs, the 5-s window results for C4 (the pilot scores only 2-s and 10-s windows), C3 reliability (≈ 4 pairs).

**Pilot results are used only to fix placeholders and catch bugs.** They are not reported as findings, and because the pilot uses only training-partition subjects, the confirmatory test set is untouched.

# 18. Repository Structure

Minimal layout: one runnable main.py, a separate download.py, and seven pipeline scripts. Folders are created by the scripts, never committed. Each script does one job, saves its output, and never recomputes another script's work.

```text
eeg-jr-pysr-pipeline/
  main.py              # sets thread env vars, runs a phase (--phase 1-4), progress bars
  download.py          # fetches ds003775; creates data/ itself
  config.yml           # every parameter + provenance tag
  PREREGISTRATION.md   # this document, frozen
  DEVIATIONS.md        # dated log of any post-data changes
  README.md            # how to run it (§18.4) — not a copy of this document
  requirements.txt     # pinned package versions, incl. exact PySR version (§18.4)
  LICENSE
  .gitignore           # §18.3
  src/
    preprocess.py      # units check, filters, bipolar pairs, artifacts, 256 Hz, segments
    model.py           # two-node Jansen-Rit, observation/mixing, UKF + smoother, Q/R rule
    synthetic_gate.py  # G0 positive + Null A + Null B + preprocessing gate; writes gate flag
    regression.py      # residual targets, primary PySR fit, Pareto rule, refit ensemble
    baseline.py        # M0 VAR on identical splits
    robustness.py      # ablation scoring, bootstrap, ICC, wPLI, surrogates, scoring across split seeds (each seed's PySR fit is run by regression.py), C4 free-run, §5.2 sensitivity analysis, AAFT test
    figures.py         # figure set, placeholders for failed stages; reads outputs/ only, computes nothing
  (created at run time) data/, cache/, outputs/, results/, logs/
    results/summary.json   # all key numbers, gate status, both equation forms (sympy + latex)
    results/equations.tex  # frozen residual + full model, publication-ready LaTeX
    results/figures/       # 9 PNGs, 300 DPI, written by phase 4
```

## 18.1 Four-phase run

The pipeline runs as four phases, one python main.py --phase N call per phase, so a multi-night run is four separate, resumable invocations rather than one long process. Each phase checks that the previous phase's flag exists before it runs, so a crash in one phase cannot silently corrupt the next.

| **Phase** | **Runs** | **Reads** | **Writes** | **Flag written on success** |
| --- | --- | --- | --- | --- |
| 1 | download (if needed), preprocess, synthetic_gate | data/ | cache/, outputs/gate.json, config.yml (μ_ref, σ_ref) | outputs/phase1.done |
| 2 | model (real-data Q/R + training UKF), regression, baseline | cache/, outputs/gate.json | outputs/ (frozen equation, Q/R, VAR fit) | outputs/phase2.done |
| 3 | robustness (scoring UKF, bootstrap, ICC, wPLI, AAFT, §5.2 sensitivity) | outputs/ | results/ (C1–C4 scores, CIs, diagnostics, summary.json, equations.tex) | outputs/phase3.done |
| 4 | figures | results/, outputs/gate.json | results/figures/*.png | outputs/phase4.done |

- **Phase 1 is the hard gate.** If either null arm fails (§9.2, §5.2's artifact-only null), main.py --phase 1 exits without writing phase1.done, and phase 2 refuses to start. If only the positive control or the preprocessing-bias check fails (both nulls pass), phase 1 still writes phase1.done, but with low_confidence: true in gate.json, and every phase 3/4 result and figure carries that label. A numerical-stability failure (§9.3) is treated as a null failure (hard stop); a contraction failure sets low_confidence.

- **`--pilot` is a separate flag, combinable with any phase** (e.g. --phase 1 --pilot), running the reduced §17 settings. In pilot mode no gate hard-stops (§9.4: 20 series cannot support the 5% claim), and pilot subjects are forced onto the training side of the split (§11.1) so they can never enter phase 3's confirmatory scoring. Pilot output goes under results/pilot/, kept separate from the full run so the two are never confused.

- **`outputs/` vs `results/`:** outputs/ holds machine-readable artifacts that later phases read (the gate flags, the split file, frozen equations, Q/R values, config-stored constants). results/ holds what a person reads: the C1–C4 scores and CIs, the diagnostics (wPLI, AAFT, ICC), and results/figures/. Nothing in results/ is ever read back in by an earlier phase, so results can be deleted and phase 3–4 rerun without touching phases 1–2.

- **Unowned outputs, now owned:** main.py writes the split file `outputs/split_<seed>.json` before exclusions (§11.1); model.py computes μ_ref and σ_ref once and stores them in config.yml (§5.1), so preprocess.py never imports model.py at run time; synthetic_gate.py writes the §9.3 gain profile; robustness.py runs the §5.2 sensitivity analysis.

- regression.py is the **only** script that ever calls PySR on real data. robustness.py only loads frozen equations.

- preprocess.py and model.py cache per subject so a crash does not force a full rerun.

- figures.py (phase 4) reads only from results/ and outputs/gate.json; it computes nothing new, so it is the one phase that is always safe to rerun on its own.

## 18.2 Summary file and equation export

Two additional files, both written once by robustness.py at the end of phase 3, using the fitted equation object's own .sympy() and .latex() methods so the equation form is captured directly rather than hand-converted.

- **`results/summary.json`.** One machine-readable file holding everything a reader would otherwise have to dig through the pipeline to find: the gate status (§18.1), the frozen residual equation in **both** sympy and LaTeX string form plus its term signature (§8.4), the C1–C4 verdicts with their key statistics (p-values, ICC and CI, recurrence percentage), the estimated coupling gains, and a per-phase compute-time breakdown. Structure is illustrative, not a fixed schema: keys should follow the claim IDs (C1–C4) and gate names already used throughout this document, so the file reads as a direct mirror of §1.2 and §15 rather than an arbitrary dump.

- **`results/equations.tex`.** A small, standalone, human-readable LaTeX file: the frozen PySR residual on its own, and the full model equation (base coupling plus residual, §8.1–8.3) with the residual term underbraced, ready to \input{} directly into a paper draft rather than requiring hand-transcription from summary.json. Auto-generated, not hand-edited; regenerating it overwrites the file.

- If low_confidence is set (§18.1) or no residual term was selected (a bare constant, §8.2), both files still get written, with the residual field explicitly recorded as absent or low-confidence rather than omitted, so a reader of summary.json or equations.tex alone always sees the true status without having to cross-check gate.json.

## 18.3 .gitignore

A .gitignore at the repo root enforces the same rule already stated in §18: everything under “(created at run time)” in the tree above is generated, not committed. config.yml, PREREGISTRATION.md and DEVIATIONS.md **are** tracked, since they are the project's decision record, not generated output.

```gitignore
data/
cache/
outputs/
results/
logs/
__pycache__/
*.pyc
.venv/
venv/
.julia_env/
*.jl.cov
.ipynb_checkpoints/
.DS_Store
Thumbs.db
.vscode/
.idea/
```

- **Disclosed consequence:** ignoring data/ means the raw EEG never reaches GitHub; anyone cloning the repo runs download.py themselves rather than receiving the data with the clone. This is treated as the intended behavior (repo size, and ds003775's own terms), not an oversight.

## 18.4 README, dependencies, license

A short web search of current research-software practice (cookiecutter-data-science and similar templates; a 2025 survey of ICML repositories; RSE best-practice guidance) converges on a small set of files that essentially every reproducible research repository has and this one did not yet list. Two are added as real requirements; the third is optional.

- **`README.md` (new, required).** Distinct from PREREGISTRATION.md: the README is operational, not a design document. It should cover, in order: a one-paragraph summary of what the pipeline does; install steps; how to run each phase (python main.py --phase 1, etc., and the --pilot flag, §18.1); the approximate wall-clock cost of each phase (§16.3); and where to look for results (results/summary.json, results/equations.tex, results/figures/, §18.2). It does not restate the scientific reasoning behind any LOCKED decision; that stays in PREREGISTRATION.md, which the README links to.

- **`requirements.txt` (new, required).** Every package this pipeline depends on, pinned to an exact version, generated with pip freeze after the environment is finalized in the pilot — most importantly the **exact PySR version** (§8.2 already requires this to be pinned; this file is where that pin actually lives) and the filterpy version the Numba port is validated against (§7.5). An unpinned dependency file would make the pilot's Numba-vs-filterpy comparison and the G0 recovery numbers unreproducible even by the same person on a reinstalled machine.

- **`LICENSE` (optional).** Not required for the pipeline to run or the results to be valid, but conventional if the repository is ever made public; a permissive license (MIT or similar) is the common default for this kind of single-author research code, per the RSE guidance above.

- **What was already in line with current practice:** the code/results/generated-output separation, src/ for pipeline code, and treating figures.py as its own accountable phase rather than an afterthought are all explicitly recommended in the RSE literature above (analysis and plotting scripts "demand the same scrutiny" as the rest of the code), so §18.1's structure did not need to change for this.

# 19. Figure Plan

**Output format (LOCKED):** every figure (1–9, including any placeholder version) is written as **PNG at 300 DPI**. This is one shared setting in figures.py (e.g. plt.savefig(path, dpi=300, format="png") for matplotlib), not a per-figure choice, so the whole figure set stays visually consistent and print-quality.

Each figure maps to a claim. If a stage fails, its figure is still produced as a labelled placeholder rather than dropped.

| **Fig** | **Content** | **Claim** |
| --- | --- | --- |
| 1 | Data overview: raw vs units-corrected bipolar traces for one subject, units check, window scheme | Data |
| 2 | Synthetic gate: recovery by coupling level; Null A and Null B false-positive counts with upper bounds; thresholds marked | G0 |
| 3 | UKF hidden-state example beside the observed channel, with estimated gain over time | Method |
| 4 | Core result: held-out one-step error for M0–M3 (ablation ladder), 33 test subjects | C1 |
| 5 | Residual-term recurrence across the 25 ensemble refits | C2 |
| 6 | Volume-conduction diagnostic and AAFT surrogate for the estimated gain | C1 interpretation |
| 7 | C1 across 5 split seeds | C2 |
| 8 | t1 vs t2 coupling-gain estimates, ICC with CI, against the literature range; with and without vigilance adjustment | C3 |
| 9 | Free-run spectral error at 2/5/10 s, or labelled “not attempted” placeholder if the §10.2 condition fails | C4 |

If both nulls pass but the positive control misses a required coupling level, the run continues and Figure 4 is still produced with a low-confidence label, mirroring last project's pysr_no_escalation flag. (The weakest coupling level is exempt from the pass criterion, so missing only that level is not a failure.) If either null fails, the run has already stopped (§18).

# 20. Known Risks and Pre-Committed Responses

| **If this happens** | **Then** | **Why decided now** |
| --- | --- | --- |
| Either G0 null arm fails | Stop. Fix the pipeline (mixing term, library, penalty, reduction order in §7.4). Re-run **both nulls and the positive control on fresh seeds**. Do not fit real data until all pass. | Otherwise any real coupling result is uninterpretable, and fixing against the same series overfits the gate. |
| G0 positive fails | Apply the §7.4 reduction order, then smoothed-derivative alternatives, then 512 Hz observation. If still failing, report as a gate failure and proceed only with results flagged low confidence. **Expected possibility:** the closest precedent (Escuain-Poole 2018) found one channel per node could not separate a node's dynamics from its afferent coupling. | Mirrors the pysr_no_escalation flag, without the escalation that no longer exists. |
| M3 diverges on recordings where M2 does not | Report per-variant divergence counts; primary comparison uses matched sets; sensitivity analysis counts M3-only divergences as M3 losses (§12). State in Limitations if above 5% of test recordings. | Dropping only M3's worst cases would flatter the headline comparison. |
| Preprocessing gate fails | Try the next-gentler option (noise-adaptive R instead of hard rejection; lower high-pass). If artifacts create coupling in the artifact-only null, stop as for a G0 null failure. | No published protocol is validated for this task. |
| UKF shows negative covariance eigenvalues or NaN/Inf during G0 (§9.3) | Apply the §7.4 dimension-reduction order first; if that alone doesn't resolve it, move to a square-root UKF formulation. Do not fit real data until resolved. | The classical scaled unscented transform is reported to become unreliable near 20-D state; this design is 19-D. |
| Exclusions far above 1–2% (e.g. > 10%) | Inspect the failing rule's log before the full run; any rule change is a deviation. Units-check failures above 5% halt automatically. | An unexpectedly high rate usually means a rule, not the data, is wrong. |
| PySR returns bare constants on real data | The selected residual is empty; M3 = M2; C1 fails. Report as a genuine null. No UDE fallback exists; this was chosen in advance. | Prevents recasting a null as an infrastructure excuse. |
| M1 does not beat M0, or M3 does not beat M0 | Report plainly. Both Nozari et al. 2024 and Kuhlmann et al. 2016 (JR+UKF 0.51 vs ARMA 0.58) make this a real possibility. | Two independent warnings exist. |
| M3 does not beat M2 | C1 fails. The refinement is not shown to add value, whatever M2 does. | Headline claim. |
| No residual term recurs across the ensemble | C2 fails for the coupling residual; report within-node stability separately. Leading explanation per §1.4. | The audits predict partial identifiability. |
| Per-session coupling-gain ICC below threshold | C3 fails. Compare to Acharya et al. 2024 as a warning from a different setting (evoked iEEG, linear/black-box models), not as confirmation. | Only loosely related prior evidence exists. |
| Runtime far over budget in pilot | Reduce, in this order: the AAFT test (surrogates per recording), the Q/R search grid, then PySR timeouts, then G0 series count (keeping ≥ 60 per null arm); record the change. The UKF-side workloads are the largest and least certain, so they are the first levers. | Keeps the home-CPU plan feasible. |

# 21. Decisions

## 21.1 Resolved in v0.2 – v0.6

| **Decision** | **Resolution** | **Section** |
| --- | --- | --- |
| Estimated vs fixed quantities (gain, delay, mixing, observation gain) | Gains and m estimated; delay and observation gain fixed | §7.4 |
| Q/R, filter vs smoother, residual definition | Innovation-consistency rule on training subjects; smoother for targets, filter for scoring; residual defined | §7.5, §8.3 |
| Pareto selection rule | Simplest within 5% of best validation loss on a training-side fold | §8.2 |
| C1 primary metric | One-step error, continuous forward filter; M3 vs M2 and M3 vs M0 | §15 |
| C2 recurrence mechanism | 25 half-sample refits + 5 split seeds | §8.2, §15 |
| C3 estimation route | Per-session filtered gain, ICC(3,1), CI-based threshold | §11.3, §15 |
| C4 window handling and condition | 2-s training windows; free-run spectral scoring; 80% stability condition | §10.2 |
| Pilot subject allocation | Training partition only | §11.1, §17 |
| Synthetic generation rate, regime, null size | 2,048 Hz; coupling grid; PSD-matched; 60 per null arm | §9 |
| UKF implementation | filterpy's standard scaled unscented transform as reference, Numba rewrite as the validated performance path, plus a numerical-stability check inside G0 (negative covariance eigenvalues / NaN propagation), since the classical transform is reported to become unreliable near 20-D state and this design is 19-D. Fallback: dimension reduction, then square-root UKF. | §7.5, §9.3, §16.5 |
| Derivative estimator for the residual | Weak/integral formulation as primary (more robust to noise than point-wise methods at low points-per-cycle); total-variation regularization as the comparison baseline; both run on the pilot's synthetic set and the better performer at the real noise level is locked in before the full run. | §8.3 |
| Bipolar montage | **LOCKED: Option C, parieto-occipital (P3–PO3, P4–PO4).** Simplest, safest choice: strongest, cleanest eyes-closed alpha signal, easiest for the model to fit. No literature favors one montage over another for coupling, so this is a pragmatic signal-quality choice, stated as such. | §6 |
| sub-010 handling | **LOCKED: keep the EEG, skip only the scans.tsv for this subject.** | §4.2, §12 |
| Bad-electrode policy | **LOCKED: exclude the subject.** Simpler than tracking a flagged covariate. | §12 |
| Line-noise method | **LOCKED: standard notch filter.** No design reason favors the added complexity of ZapLine-style methods here. | §5.1 |
| Common-input term in the fitted model | **LOCKED: not included**, to avoid adding a parameter to an already 19-D state near its stability edge. The synthetic null still tests for this confound regardless. | §7.2 |
| Primary split seed | **LOCKED: 42**, matching the C. elegans project's convention. | §11.1 |
| PySR execution settings | Real fits parallel; serial only for pilot determinism checks; populations at default; fixed 50,000-row subsample per fit with equal rows per subject; no batching flag. | §8.2, §16.5 |
| Q/R versus G0 ordering | G0 tunes its own Q/R on a separate 20-series synthetic tuning set; real-data Q/R tuned only after G0 passes, per split seed. | §7.5, §9.4 |
| Null-gain estimator | Mean filtered gain per recording (same estimator as C3). The v0.5 block-interval pass rule was superseded in v0.6 by the δ-band rule (see Null-gate rule below). | §9.2, §11.3 |
| Gate failure rule | Null failure stops the run; positive-control failure continues flagged low confidence. | §9, §18, §19, §20 |
| Coupling delay implementation | Integer sub-step delay; ring buffer of filtered means outside the sigma points; state stays 19-D. | §7.5 |
| Residual definition and sharing | Residual on dy₄/dt (velocity of the excitatory PSP onto pyramidal cells); one shared equation for both directions; term signature defined for C2. | §8.3, §8.4 |
| Pareto fit-then-freeze; C1, C3, C4 details | Fit on 80%, select on 20%, freeze; C1 excludes burn-in with matched VAR samples; C3 decided on the unadjusted ICC; C4 uses a stochastic free-run on a common 0.5-Hz spectral grid, judged on the CI bound of the error ratio. | §8.2, §10.2, §15 |
| Per-recording exclusions and divergence | Bad-electrode exclusion judged per recording; divergence counted per variant with a sensitivity analysis. | §12 |
| Null-gate rule | │estimated gain│ < δ (half the weakest planted level) instead of an interval test; 0-of-60 structure kept. | §9.2, §15 |
| C4 scoring | Per 2-s segment, so the long/short ratio is about 1.00 for a non-drifting model; scored only after a full G0 pass. | §10.2, §15 |
| Parameter handling in training windows | Recording-level parameters from a continuous pass, held fixed in 2-s windows. | §7.5, §10.2 |
| Rescaling reference | Fixed constants from literature settings; p excluded from the bias check; scale-free G0 matching. | §5.1, §5.2, §9.1 |
| Filter and simulation defaults | Q/R family, priors, sigma points, integrator, initial states. | §7.6 |
| Planted residual, null strengths, G0 aggregation | Defined in §9.1–9.2; parsimony penalty recalibrated on pilot series. | §9 |
| Jansen–Rit constants and equations | Verified against independent sources; sigmoid formula and notation stated; coupling grid and gain prior re-centred on Escuain-Poole's medium coupling. | §7.1, §7.3, §9.1 |
| Order and ownership | Split before exclusions, written by main.py; reference constants in config.yml; gate.json carries §9.3 flags; pilot mode never hard-stops. | §11.1, §18 |

## 21.2 Still open (genuinely pilot-dependent; cannot be decided in advance)

| **#** | **Decision** | **Section** |
| --- | --- | --- |
| 1 | Hard segment rejection vs noise-adaptive UKF observation covariance — locked as hard rejection by default, compared against the alternative in the pilot | §5.2 |
| 2 | Operating system and exact CPU affinity mask (a 5-minute check, not a research question, but needs doing once on the actual machine) | §16 |
| 3 | Cross-recording worker count for the joblib pool: benchmark 4 vs 6 vs 8 in the pilot before locking | §16.5 |
| 4 | Whether the Numba UKF passes the §7.5 validation against filterpy (rtol=1e-8, atol=1e-10, plus unchanged downstream statistics); if not, fall back to filterpy for the full run | §7.5, §16.5 |

# 22. Key References

- Hatlestad-Hall C, Rygvold TW, Andersson S (2022). BIDS-structured resting-state EEG data extracted from an experimental paradigm. Data in Brief 45:108647.

- Jansen BH, Rit VG (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biological Cybernetics 73:357–366.

- David O, Friston KJ (2003). A neural mass model for MEG/EEG: coupling and neuronal dynamics. NeuroImage 20:1743–1755.

- Kuhlmann L, et al. (2016). Neural mass model-based tracking of anesthetic brain states. NeuroImage 133:438–456.

- Valdes PA, et al. (1999). Nonlinear EEG analysis based on a neural mass model. Biological Cybernetics 81:415–424.

- Freestone DR, et al. (2011). A data-driven framework for neural field modeling. NeuroImage 56(3):1043–1058.

- Omejc N, Roman S, Todorovski L, Džeroski S (2026). Neural population models for EEG: from canonical models to alternative model structures. PLOS Comput Biol 22(8):e1014222.

- Stankovski T, et al. (2017). Neural cross-frequency coupling functions. (Details and DOI: see coupling audit; verify before citing.)

- Delabays R, De Pasquale G, Dörfler F, Zhang Y (2025). Hypergraph reconstruction from dynamics. Nature Communications. DOI: 10.1038/s41467-025-57664-2.

- Hartoyo A, et al. (2019). Parameter estimation and identifiability in a neural population model for electro-cortical activity. PLOS Comput Biol 15(5):e1006694.

- Escuain-Poole L, Garcia-Ojalvo J, Pons AJ (2018). Extracranial estimation of neural mass model parameters using the Unscented Kalman Filter. Front Appl Math Stat 4:46. (Also used in v0.6 to verify the Jansen–Rit equations, constants and coupling form.)

- Grimbert F, Faugeras O (2006). Bifurcation analysis of Jansen's neural mass model. Neural Computation 18(12):3052–3068. (Constants and equations confirmed in v0.6 through secondary sources; the 1995 original was not read.)

- Bastiaens SP, Momi D, Griffiths JD (2025). A comprehensive investigation of intracortical and corticothalamic models of the alpha rhythm. PLOS Comput Biol 21(4):e1012926.

- Bastos AM, Schoffelen J-M (2016). A tutorial review of functional connectivity analysis methods and their interpretational pitfalls. Front Syst Neurosci 9:175.

- Nolte G, et al. (2004). Identifying true brain interaction from EEG data using the imaginary part of coherency. Clin Neurophysiol 115:2292–2307.

- Vinck M, et al. (2011). An improved index of phase-synchronization for electrophysiological data. NeuroImage 55:1548–1565.

- Mazzoni A, et al. (2015). Computing the local field potential (LFP) from integrate-and-fire network models. PLOS Comput Biol 11(12):e1004584.

- Teleńczuk B, et al. (2017). Local field potentials primarily reflect inhibitory neuron activity in human and monkey cortex. Sci Rep 7:40211.

- Nozari E, et al. (2024). Macroscopic resting-state brain dynamics are best described by linear models. Nat Biomed Eng 8:68–84.

- Acharya G, Davis KA, Nozari E (2024). Predictive modeling of evoked intracranial EEG response to medial temporal lobe stimulation. Commun Biol 7:1210.

- Prokop B, Gelens L (2024). From biological data to oscillator models using SINDy. iScience 27(4):109316.

- Fasel U, et al. (2022). Ensemble-SINDy: robust sparse model discovery in the low-data, high-noise limit. Proc R Soc A 478:20210904.

- de Vries S, Keemink SW, van Gerven MAJ (2026). Symbolic discovery of stochastic differential equations with genetic programming. arXiv 2603.09597 (preprint).

- Politanskaia P, et al. (2026). Long-term reliability and stability of parameterized resting state EEG. Cerebral Cortex 36(7):bhag113.

- Østergaard FG, et al. (2024). The aperiodic exponent of neural activity varies with vigilance state in mice and men. PLOS ONE 19(8):e0301406.

- Muthukumaraswamy SD (2013). High-frequency brain activity and muscle artifacts in MEG/EEG. Front Hum Neurosci 7:138.

- Yokoyama H, Kitajo K (2023). A data assimilation method to track excitation-inhibition balance change using scalp EEG. Communications Engineering 2:92. (Listed as unvetted in the E/I audit.)

- Widmann A, Schröger E, Maess B (2015). Digital filter design for electrophysiological data — a practical approach. J Neurosci Methods 250:34–46.

- Tanner D, Morgan-Short K, Luck SJ (2015). How inappropriate high-pass filters can produce artifactual effects in ERP studies. Psychophysiology 52(8):997–1009.

- Barnett L, Seth AK (2011). Behaviour of Granger causality under filtering. J Neurosci Methods 201(2):404–419.

- Castellanos NP, Makarov VA (2006). Recovering EEG brain signals: artifact suppression with wavelet enhanced ICA. J Neurosci Methods 158:300–312.

- Klug M, Gramann K (2021). Identifying key factors for improving ICA-based decomposition of EEG data. Eur J Neurosci 54(12):8406–8420.

- Cranmer M (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv 2305.01582.

- Chang L, et al. High-dimensional instability and negative sigma-point weights in scaled unscented transforms (general finding cited across multiple sources; verify specific reference before citing). See also arXiv 2009.13079 (Geometric UKF) and the multi-machine power-system UKF stability literature (arXiv:1509.07394) for dimension ≥ 20 divergence reports.

- Weak SINDy / WmSINDy literature on integral-formulation derivative estimation as more noise-robust than point-wise (TV, spline, finite-difference) methods for sparse dynamical-system identification (e.g. arXiv 2410.17838; arXiv 2504.01289). Verify exact primary citations before the paper draft.

- La Cava W, et al. Contemporary symbolic regression methods and their relative performance (SRBench). PMC11074949. — Operon on the accuracy-simplicity Pareto front; AI Feynman leads exact recovery on low-noise problems; parallel execution not benchmarked.

- eggp paper, arXiv 2501.17848 — single-threaded PySR ran 5-25x the runtime of Operon depending on dataset (used for §16.5's Operon comparison, not adopted).

- Cranmer M, GitHub Discussion #873 (PySR/SymbolicRegression.jl) — PYTHON_JULIACALL_THREADS is fixed at Julia interpreter startup and cannot be changed by later PySRRegressor configuration.

- PySR official tuning documentation (astroautomata.com/PySR/tuning) — turbo/bumper flags, precision, ncycles_per_iteration, populations guidance for cluster vs laptop-class runs.

- Numba documentation, "Floating-point pitfalls" — Numba may use a different algorithm than NumPy/Python; results are not guaranteed bit-for-bit compatible; float32/int32 promotion differs from NumPy.

- QuantEcon, "NumPy vs Numba vs JAX" lecture and style manual — for inherently sequential CPU work without autodiff, prefer Numba; JAX wins primarily for vectorized/batched (vmap) operations.

- scikit-learn parallelism documentation — total thread count equals n_jobs × `<LIB>_NUM_THREADS` unless inner BLAS/OpenMP threads are explicitly limited per worker.

**Verification before submission.** Full source lists with DOIs are in the seven evidence-audit PDFs (six on the core project design, one dedicated to performance optimization). Several DOIs there are marked unverified, and the coupling audit flags that the Hartoyo quotation and some Hashemi and Ahmadizadeh details came via a subagent. Check every one against the original before the paper is submitted.