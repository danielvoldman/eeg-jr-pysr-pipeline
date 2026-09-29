# DEVIATIONS.md

This is the dated record of every change to, or clarification of, the preregistered design in `PREREGISTRATION.md` (v0.6, frozen 2026-09-27).

**Rules**
- Add an entry before, or together with, the commit that makes the change.
- Never edit or delete an old entry. If a decision changes, add a new entry that supersedes it and names the old ID.
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
