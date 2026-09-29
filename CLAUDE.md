# CLAUDE.md

Standing instructions for Claude Code in this repository. Read this whole file at the start of every session.

## The project

This is a minimal, four-phase Python pipeline. It fits a two-node Jansen–Rit neural-mass model to two bipolar resting-state EEG channels (OpenNeuro ds003775) with an unscented Kalman filter. PySR then refines only a residual on the inter-node coupling term. The pipeline tests whether that refinement adds anything (gate G0, claims C1–C4).

## Source of truth

- `PREREGISTRATION.md` (v0.6) is the complete specification. When a task names sections (for example "§5.1, §7.6"), read those sections in full before planning. Cite section numbers in plans, code comments and commit messages.
- Every decision in it carries a tag. **LOCKED** means settled. **PLACEHOLDER** means a default the pilot may calibrate. Items that are still open are listed in §21.2.
- If the code and the document disagree, the document wins.
- If the document is ambiguous or silent on something the code needs, **stop and ask**. Do not guess. If a default is unavoidable to keep going:
  - put it in `config.yml` tagged `placeholder`;
  - log it in `DEVIATIONS.md` under "Implementation clarifications";
  - tell me in your reply.
- `PLAN.md` holds the build order and the definition of done for every step. Work on one step at a time. When a step is finished, update its checkbox and add a note.

## Hard rules

1. **Never change a LOCKED value or procedure** without my explicit approval and a `DEVIATIONS.md` entry. If you think one is wrong, say so and stop.
2. **No magic numbers in `src/`.** Every parameter, threshold, seed and grid lives in `config.yml` with a provenance tag: `literature`, `locked`, `placeholder`, `computed`, `fitted` or `unset`. `unset` means the document gives no value: stop and ask, log the answer in `DEVIATIONS.md` §2, then change the tag to `placeholder`. Config entries use the format `{value, prov, ref}`; code reads only `value`.
3. **Every random operation uses an explicit seed** from `config.yml`, through `numpy.random.default_rng(seed)`. Never use the global `np.random` state.
4. **float64 everywhere.** Numba functions use `fastmath=False`. Cast explicitly to avoid float32/int32 promotion (§16.5.2).
5. **Script ownership follows §18 exactly.**
   - One job per script. Each script saves its output and never recomputes another script's work.
   - `regression.py` is the only script that calls PySR on real data.
   - `figures.py` computes nothing.
   - `preprocess.py` never imports `model.py`. The reverse direction is allowed.
6. **Data discipline.**
   - The split (seed 42) is made once by `main.py`, on all 111 subject IDs, **before** exclusions (§11.1).
   - Test subjects are never used to tune anything: not Q/R, not penalties, not placeholders.
   - Pilot subjects are always on the training side.
   - No real-data model fitting (Q/R, UKF training, PySR) happens until `outputs/gate.json` exists and permits it. `--pilot` mode is the only exception (§17).
7. **Closed questions. Do not reopen or suggest these:**
   - Wilson–Cowan.
   - Splitting one channel into E/I.
   - Open-form coupling discovery.
   - UDE escalation.
   - The §18 repo layout and the 4-phase design.
   - Everything in §16.5.3: Operon or other SR libraries, PyTorch/JAX, Polars, MNE `method="polyphase"`, float32, and changes to PySR's `ncycles_per_iteration`, `populations`, `batching` or `fast_cycle`.
8. **No faking.**
   - Never hard-code values to make a test pass.
   - Never present mock or synthetic output as a result.
   - Never silently skip or weaken a failing check.
   - Report failures plainly.

## Windows environment

- The setup is Windows, VS Code and PowerShell, with Python in `.venv`. Use `pathlib` for every path.
- `main.py` sets these environment variables **before** importing NumPy, MNE or PySR:
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHON_JULIACALL_THREADS=8`.
  PySR's Julia thread count is fixed at startup and cannot be changed later.
- Full runs are launched as:
  `cmd /c start /affinity FF /wait python main.py --phase N`
  FF means logical CPUs 0–7. The SMT pairing still has to be verified on this machine (§21.2 #2).
- Size worker pools from `len(psutil.Process().cpu_affinity())`, never from `os.cpu_count()`.
- Windows uses the `spawn` start method, which has three consequences:
  - Every script that uses joblib or multiprocessing needs an `if __name__ == "__main__":` guard.
  - Worker functions must be defined at module top level.
  - Each worker sets its BLAS threads to 1.
- joblib uses the `loky` backend with `n_jobs=4` by default, until the pilot benchmark decides otherwise (§21.2 #3).
- Never run PySR at the same time as the cross-recording worker pool (§16.5.1).
- The first PySR import installs Julia and can take several minutes. Never include it in a timing.
- Read EDF files with MNE using `units="uV"` (§4.2).

## How to work

1. **Plan before coding.** For each PLAN.md step, read the named sections, then give me:
   - (a) your understanding of the step in a few sentences;
   - (b) the files and functions you will create or change;
   - (c) every ambiguity or conflict you found;
   - (d) the tests you will write.

   Then wait for my approval.
2. **Verify with evidence.** Write tests in `tests/` with pytest. Run them and show me the command and its output. A step is done only when every "done when" item for it in PLAN.md is met, with evidence shown.
3. **Keep changes scoped** to the current step. If you notice a problem elsewhere, note it; don't fix it unasked.
4. **Commit** after each completed step, with a message naming the step and its sections, for example `A4: reference constants mu_ref/sigma_ref (§5.1, §7.6)`. Never commit `data/`, `cache/`, `outputs/`, `results/` or `logs/`.
5. **Log** to `logs/` with Python's `logging` module, one file per phase. Don't use bare `print` in `src/`.