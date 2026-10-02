# eeg-jr-pysr-pipeline

> **Status: this repository ends at a preregistered hard stop.** The synthetic-validation gate G0 failed at full size: 51 of 60 Null A and 55 of 60 Null B series were false positives on the gain half (criterion: 0 of 60). Under the preregistration (§9.2, §20) that stops the pipeline. No real-data model fit, PySR run or claim verdict (C1 to C4) exists, and none was attempted. Nothing was tuned or loosened. Read [docs/G0_HARD_STOP_REPORT.md](docs/G0_HARD_STOP_REPORT.md).

## What it does

A four-phase Python pipeline. It fits a two-node Jansen–Rit neural-mass model to two bipolar resting-state EEG channels (P3–PO3, P4–PO4; OpenNeuro ds003775) with an unscented Kalman filter, and uses PySR to refine only a residual on the inter-node coupling term. A synthetic gate (G0) must pass before any real-data fit. The scientific reasoning, every decision and every pass/fail rule are in [PREREGISTRATION.md](PREREGISTRATION.md) (v0.6, tag `prereg-v0.6`); this README does not restate them. Every change after that is in [DEVIATIONS.md](DEVIATIONS.md).

## Install

Windows, PowerShell, Python 3.12 (3.12.10 was used).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is a pinned `pip freeze` (PySR 2.6.0, filterpy 1.4.5, joblib 1.6.0, specparam 2.0.0rc7). Julia 1.11.9 and SymbolicRegression.jl 2.5.1 are installed by PySR on its first import (several minutes) under `.venv`. Always call `.venv\Scripts\python.exe` explicitly.

The data are not in the repository. `download.py` fetches ds003775 v1.2.1 into `data/` (`--verify` checks the stored SHA-256 manifest). **ds003775 has its own terms of use; read and follow them on OpenNeuro.** The code in this repository is MIT-licensed (see `LICENSE`); that licence does not cover the data.

## How to run

All commands from the repository root. `main.py` sets the thread variables before NumPy is imported. For a full run on Windows use the affinity launch of CLAUDE.md:

```powershell
cmd /c start /affinity FF /wait .venv\Scripts\python.exe main.py --phase 1
```

| Command | What it does |
|---|---|
| `main.py --make-split` | Writes the train/test split files (seed 42 plus four extra seeds) once, on all 111 subject IDs, before exclusions. Never redraw them. |
| `main.py --phase 1 [--pilot]` | Preprocessing (four §5.2 variants, exclusions, impulse responses), then the synthetic gate G0. Writes `outputs/gate.json` and, only if the gate allows, `outputs/phase1.done`. |
| `main.py --phase 2 [--pilot]` | Real-data Q/R report, primary PySR fit, VAR baseline. Refused unless phase 1 passed. |
| `main.py --phase 3 [--pilot]` | Scoring (C1 to C4), bootstrap, ICC, diagnostics, `summary.json`, `equations.tex`. |
| `main.py --phase 4 [--pilot]` | Figures. |
| `python -m src.synthetic_gate --g0-null-arms [--estimate-only]` | The formal full-size G0 null arms (UKF only). Writes `results/g0_null_arms_r<round>.json` once; never rerun on burnt seeds (`results/g0_seed_ledger.json`). |
| `python -m src.figures --figure 2` or `--all` | Figures from `outputs/gate.json` and the formal result file only (see below). |

`--pilot` runs the reduced §17 settings on the 12 pilot subjects (training side) and writes under `results/pilot/`; it never hard-stops and its output is mechanics only. `--force` recomputes a phase; on a full run that replaces never-redrawn outputs it needs a DEVIATIONS entry. `--dry-run` is for `--make-split` only (reports, writes nothing).

## What the hard stop means

- `outputs/gate.json` exists with `hard_stop: true` and `complete: false` (the positive control, contraction, stability and preprocessing-bias checks were not run on the formal arms).
- `outputs/phase1.done` does not exist, so `main.py --phase 2` (and 3, 4) refuse to start. This is deliberate and is not a bug to work around.
- The only figure with content is Fig 2 (the gate); the other eight are labelled placeholders that say why. `python -m src.figures --all` writes them to `results/figures/` (PNG, 300 DPI). `main.py --phase 4` stays refused.
- Going further would need a pipeline fix with its own deviation entry and a fresh-seed round 2 (§9.4, §20). The decision recorded in IMP-099 is to write the result up instead.

## Where results are

`results/` and `outputs/` are generated and not committed. The committed record of the result is `docs/G0_HARD_STOP_REPORT.md` (every number there carries its source file and SHA-256). When present: `outputs/gate.json`, `results/g0_null_arms_r1.json`, `results/g0_seed_ledger.json`, `results/figures/`, and mechanics-only pilot files under `results/pilot/`. The full-run files `results/summary.json` and `results/equations.tex` (§18.2) were not produced.

## Wall-clock cost

Measured, on the 4-core / 8-thread machine used (8 logical CPUs, 4 workers): the formal null-arm run (120 series, UKF only, preprocessing cached) took about 11 minutes: series generation 118 s, grid 19 s, evaluation 516 s (IMP-098). The full test suite takes about 14 to 17 minutes. The §16.3 estimates for the other stages (for example 6 to 20 h of UKF runs, 4.2 h for the PySR ensemble) were **not measured**: phases 2 to 4 never ran at full size.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -rs -q      # PySR tests are excluded by default
.venv\Scripts\python.exe -m pytest -m pysr     # real Julia fits, minutes
```

A few tests read `results/` or `outputs/` and skip with an explicit reason when those folders are absent.

## Layout

`main.py`, `download.py`, `config.yml` (every parameter with a provenance tag), `src/` (pipeline scripts, §18), `tests/`, `tools/` (read-only diagnostics, not part of the pipeline), `docs/`, and the decision record (`PREREGISTRATION.md`, `DEVIATIONS.md`, `PLAN.md`, `HANDOFF.md`). `data/`, `cache/`, `outputs/`, `results/`, `logs/` are created at run time and never committed.
