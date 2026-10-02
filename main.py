"""Pipeline entry point: `python main.py --phase N [--pilot]` (§16.2, §18, §18.1).

The thread variables must be set before NumPy, MNE or PySR are imported, so
only the standard library, PyYAML and src.config are imported above the
bootstrap call. Julia's thread count is fixed at startup (§16.2).

Subject split (§11.1, §11.2, §17, IMP-008): `python main.py --make-split
[--dry-run] [--force]` writes outputs/split_<seed>.json for the five split
seeds, on all subject IDs and before any exclusion. The split is never redrawn.

Draw order, so that the split can be reproduced from this description alone.
All IDs are sorted (plain string order) before any draw; numpy
default_rng(seed) is the only source of randomness.
  1. Primary seed (split.primary_seed, 42): rng = default_rng(seed);
     perm = rng.permutation(n_subjects); the IDs at perm[:n_train] are train,
     the rest are test. Unstratified.
  2. Pilots: a fresh rng_p = default_rng(split.pilot_draw_seed). Sorted t2
     subjects of the primary training side: idx = rng_p.choice(n, 4,
     replace=False). Then, with the same generator, sorted non-t2 subjects of
     the primary training side: idx = rng_p.choice(n, 8, replace=False).
  3. Extra seeds (split.extra_seeds): pool = sorted IDs without the 12 pilots;
     rng = default_rng(seed); perm = rng.permutation(len(pool));
     pool[perm[:n_test]] are test; the other pool IDs plus the 12 pilots are
     train.
The pilots are therefore the same in every seed and always on the training side.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import yaml  # noqa: F401  (allowed before the env bootstrap)

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_PREREQ = 2

LOGGER_NAME = "pipeline"


def _bootstrap_env(config_path=None):
    """Set the thread variables from config.yml (§16.2). Overrides inherited values."""
    cfg = load_config(config_path or REPO_ROOT / "config.yml")
    for name, value in cfg["compute"]["env_vars_before_numpy_import"].items():
        os.environ[name] = str(value)


_bootstrap_env()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EEG Jansen-Rit PySR pipeline")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--phase", type=int, choices=[1, 2, 3, 4])
    what.add_argument("--make-split", action="store_true",
                      help="write outputs/split_<seed>.json for all split seeds (§11.1)")
    parser.add_argument("--pilot", action="store_true",
                        help="reduced §17 settings; output under results/pilot/")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --make-split: report, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="with --make-split: overwrite differing split files (must be logged in DEVIATIONS.md "
                             "section 1); with --phase: recompute every step instead of reusing its output (a full run "
                             "must log replaced never-redrawn outputs in DEVIATIONS.md)")
    args = parser.parse_args(argv)
    if not args.make_split and args.dry_run:
        parser.error("--dry-run needs --make-split")
    if args.make_split and args.pilot:
        parser.error("--pilot cannot be combined with --make-split")
    return args


# ---------------------------------------------------------------- subject split (§11.1, IMP-008)

class SplitError(RuntimeError):
    """The data tree, a split or a split file violates §11.1 / §17."""


def _subject_ids_sha256(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def discover_subjects(cfg, root):
    """Return (sorted subject IDs, set of IDs with a t2 EDF) read from data/ (§4.1).

    Fails loudly unless the tree holds exactly the configured numbers.
    """
    ds = cfg["dataset"]
    data = Path(root) / cfg["paths"]["data_dir"]
    subjects = {p.name for p in data.glob(ds["subject_dir_glob"]) if p.is_dir()}
    sessions = {}
    n_edf = 0
    for edf in data.glob(ds["eeg_glob"]):
        rel = edf.relative_to(data).parts
        sessions.setdefault(rel[0], set()).add(rel[1])
        n_edf += 1
    if len(subjects) != ds["n_subjects"]:
        raise SplitError(f"expected {ds['n_subjects']} subjects in {data}, found {len(subjects)}")
    if n_edf != ds["n_recordings"]:
        raise SplitError(f"expected {ds['n_recordings']} EDF files, found {n_edf}")
    no_t1 = sorted(s for s in subjects if ds["session_first"] not in sessions.get(s, ()))
    if no_t1:
        raise SplitError(f"subjects without a {ds['session_first']} EDF: {no_t1}")
    t2 = {s for s in subjects if ds["session_second"] in sessions.get(s, ())}
    if len(t2) != ds["n_subjects_with_t2"]:
        raise SplitError(f"expected {ds['n_subjects_with_t2']} subjects with "
                         f"{ds['session_second']}, found {len(t2)}")
    return sorted(subjects), t2


def read_manifest_summary(cfg, root):
    """The '# files=...' summary line of the download manifest, or None (IMP-007)."""
    path = Path(root) / cfg["paths"]["manifest_file"]
    if not path.is_file():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.startswith("# files=")]
    return lines[-1] if lines else None


def split_seeds(cfg):
    sp = cfg["split"]
    seeds = [sp["primary_seed"]] + list(sp["extra_seeds"])
    if len(set(seeds)) != len(seeds) or len(seeds) != sp["n_split_seeds_total"]:
        raise SplitError(f"split seeds must be {sp['n_split_seeds_total']} distinct values, "
                         f"got {seeds}")
    return seeds


def _numpy():
    """numpy, imported on first use so that `import main` never loads it (§16.2)."""
    import numpy
    return numpy


def _shuffled(ids, seed):
    perm = _numpy().random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in perm]


def make_split(subject_ids, t2_ids, cfg, seed, manifest_summary=None):
    """Build one split dict (draw order in the module docstring)."""
    sp, ds = cfg["split"], cfg["dataset"]
    ids = sorted(subject_ids)
    t2_ids = set(t2_ids)
    n_train, n_test = sp["n_train"], sp["n_test"]
    if n_train + n_test != len(ids) or round(sp["train_fraction"] * len(ids)) != n_train:
        raise SplitError(f"n_train {n_train} + n_test {n_test} must equal {len(ids)} subjects "
                         f"and match train_fraction {sp['train_fraction']}")
    if seed not in split_seeds(cfg):
        raise SplitError(f"seed {seed} is not one of the configured split seeds")
    order = _shuffled(ids, sp["primary_seed"])
    primary_train = sorted(order[:n_train])
    # pilots: own generator, primary training side only (§11.1, §17)
    np = _numpy()
    rng_p = np.random.default_rng(sp["pilot_draw_seed"])
    t2_train = [i for i in primary_train if i in t2_ids]
    other_train = [i for i in primary_train if i not in t2_ids]
    n_p, n_p2 = sp["n_pilot_subjects"], sp["n_pilot_subjects_with_t2"]
    if len(t2_train) < n_p2 or len(other_train) < n_p - n_p2:
        raise SplitError("primary training side has too few t2 / non-t2 subjects "
                         "for the pilot set")
    pilots = sorted(
        [t2_train[int(i)] for i in rng_p.choice(len(t2_train), n_p2, replace=False)]
        + [other_train[int(i)]
           for i in rng_p.choice(len(other_train), n_p - n_p2, replace=False)])
    if seed == sp["primary_seed"]:
        procedure = "primary_unstratified"
        train, test = primary_train, sorted(order[n_train:])
    else:
        procedure = "pilots_forced_to_train"
        pool = [i for i in ids if i not in set(pilots)]
        order = _shuffled(pool, seed)
        test = sorted(order[:n_test])
        train = sorted(order[n_test:] + pilots)
    return {
        "seed": seed,
        "procedure": procedure,
        "dataset_id": ds["openneuro_id"],
        "dataset_version": ds["version"],
        "subject_ids_sha256": _subject_ids_sha256(ids),
        "train": train,
        "test": test,
        "pilot": pilots,
        "subjects_with_t2": sorted(t2_ids),
        "manifest_summary": manifest_summary,
        "numpy_version": np.__version__,
    }


def validate_split(split, cfg):
    """Raise SplitError listing every §11.1 / §17 violation; return None if clean."""
    sp = cfg["split"]
    train, test, pilot = split["train"], split["test"], split["pilot"]
    t2 = set(split["subjects_with_t2"])
    bad = []
    for name, lst in (("train", train), ("test", test), ("pilot", pilot)):
        if len(set(lst)) != len(lst):
            bad.append(f"duplicate IDs in {name}")
    overlap = set(train) & set(test)
    if overlap:
        bad.append(f"overlap between train and test: {sorted(overlap)}")
    union = set(train) | set(test)
    if (_subject_ids_sha256(union) != split["subject_ids_sha256"]
            or len(union) != cfg["dataset"]["n_subjects"]):
        bad.append(f"union of train and test is not all {cfg['dataset']['n_subjects']} subjects")
    if len(train) != sp["n_train"]:
        bad.append(f"train size {len(train)} != {sp['n_train']}")
    if len(test) != sp["n_test"]:
        bad.append(f"test size {len(test)} != {sp['n_test']}")
    outside = sorted(set(pilot) - set(train))
    if outside:
        bad.append(f"pilot subjects not in train: {outside}")
    if len(pilot) != sp["n_pilot_subjects"]:
        bad.append(f"pilot count {len(pilot)} != {sp['n_pilot_subjects']}")
    if len(set(pilot) & t2) != sp["n_pilot_subjects_with_t2"]:
        bad.append(f"pilot t2 count {len(set(pilot) & t2)} != {sp['n_pilot_subjects_with_t2']}")
    wrong = sorted(s for s in t2 if (s in set(train)) + (s in set(test)) != 1)
    if wrong:
        bad.append(f"t2 subjects not on exactly one side: {wrong}")
    if bad:
        raise SplitError(f"invalid split (seed {split.get('seed')}): " + "; ".join(bad))


def _split_bytes(split):
    text = json.dumps(split, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def split_path(cfg, root, seed):
    return Path(root) / cfg["paths"]["split_file_pattern"].format(seed=seed)


def _without_numpy_version(d):
    return {k: v for k, v in d.items() if k != "numpy_version"}


def split_status(split, cfg, root):
    """'created', 'unchanged' or 'differs' versus the file on disk.

    numpy_version is not compared: if it is the only difference the file counts
    as unchanged and a warning is logged.
    """
    path = split_path(cfg, root, split["seed"])
    if not path.is_file():
        return "created"
    old = path.read_bytes()
    if old == _split_bytes(split):
        return "unchanged"
    try:
        old_split = json.loads(old.decode("utf-8"))
    except ValueError:
        return "differs"
    if (isinstance(old_split, dict) and _split_bytes(old_split) == old
            and _without_numpy_version(old_split) == _without_numpy_version(split)):
        logging.getLogger(LOGGER_NAME).warning(
            "split %s: only numpy_version differs (file %s, now %s); treated as unchanged",
            split["seed"], old_split.get("numpy_version"), split["numpy_version"])
        return "unchanged"
    return "differs"


def write_split(split, root, cfg, dry_run=False, force=False):
    """Write one split file. Returns 'created', 'unchanged', 'differs-refused' or 'overwritten'.

    A differing existing file is never replaced without force (§11.1: never redrawn).
    """
    validate_split(split, cfg)
    status = split_status(split, cfg, root)
    if status == "unchanged":
        return "unchanged"
    if status == "differs" and not force:
        return "differs-refused"
    if not dry_run:
        path = split_path(cfg, root, split["seed"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(_split_bytes(split))
        os.replace(tmp, path)
    return "created" if status == "created" else "overwritten"


def run_make_split(cfg, root, dry_run=False, force=False):
    """Build, validate and write all split files; none is written if any would be refused."""
    logger = logging.getLogger(LOGGER_NAME)
    ids, t2 = discover_subjects(cfg, root)
    summary = read_manifest_summary(cfg, root)
    splits = [make_split(ids, t2, cfg, s, summary) for s in split_seeds(cfg)]
    statuses = [write_split(sp, root, cfg, dry_run=True, force=force) for sp in splits]
    for sp, st in zip(splits, statuses):
        logger.info("seed %d [%s]: train %d, test %d, pilot %d (t2 %d), t2 on test side %d"
                    " -> %s%s", sp["seed"], sp["procedure"], len(sp["train"]), len(sp["test"]),
                    len(sp["pilot"]), len(set(sp["pilot"]) & t2), len(set(sp["test"]) & t2),
                    st, " (dry run)" if dry_run else "")
    logger.info("pilot subjects: %s", ", ".join(splits[0]["pilot"]))
    logger.info("numpy %s; manifest: %s", _numpy().__version__, summary)
    if "differs-refused" in statuses:
        logger.error("a split file differs from the regenerated split; nothing written. "
                     "The split is never redrawn (§11.1); --force overwrites and must be "
                     "logged in DEVIATIONS.md section 1")
        return False
    if force and "overwritten" in statuses:
        logger.warning("--force: split file(s) overwritten; log this in DEVIATIONS.md section 1")
    if not dry_run:
        for sp in splits:
            write_split(sp, root, cfg, force=force)
    return True


def load_split(seed, root=None, cfg=None):
    """Read and validate outputs/split_<seed>.json; returns the split dict.

    Code that must not see test IDs (tuning, fitting) calls training_subjects().
    """
    root = Path(root) if root is not None else REPO_ROOT
    cfg = cfg if cfg is not None else load_config(root / "config.yml")
    path = split_path(cfg, root, seed)
    if not path.is_file():
        raise SplitError(f"split file {path} not found; run main.py --make-split")
    split = json.loads(path.read_text(encoding="utf-8"))
    validate_split(split, cfg)
    return split


def training_subjects(seed, root=None, cfg=None):
    return list(load_split(seed, root, cfg)["train"])


def test_subjects(seed, root=None, cfg=None):
    """Confirmatory test IDs: phase 3 scoring only, never for tuning (§11.1)."""
    return list(load_split(seed, root, cfg)["test"])


test_subjects.__test__ = False  # not a pytest test when imported into a test module


def flag_path(cfg, root, phase, pilot):
    key = "pilot_phase_flag_pattern" if pilot else "phase_flag_pattern"  # IMP-002
    return Path(root) / cfg["paths"][key].format(n=phase)


def ensure_dirs(cfg, root, pilot):
    paths = cfg["paths"]
    names = ["cache_dir", "outputs_dir", "results_dir", "logs_dir"]
    if pilot:
        names.append("pilot_results_dir")
    for name in names:
        (Path(root) / paths[name]).mkdir(parents=True, exist_ok=True)


def setup_logging(cfg, root, phase, pilot):
    suffix = "_pilot" if pilot else ""
    return _setup_logger(Path(root) / cfg["paths"]["logs_dir"] / f"phase{phase}{suffix}.log")


def _setup_logger(log_file):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def check_prerequisite(cfg, root, phase, pilot):
    """Return None if phase may start, else the missing flag path (§18.1)."""
    if phase == 1:
        return None
    prev = flag_path(cfg, root, phase - 1, pilot)
    return None if prev.is_file() else prev


# ---------------------------------------------------------------- gate state and phase flags (H0, IMP-089)

def _sha256_file(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git_head(root):
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gate_state(cfg, root, pilot):
    """The gate file of the mode (outputs/gate.json, or results/pilot/gate_<g0.filter>.json for a pilot): whether it exists
    and its hard_stop and low_confidence flags (None when unknown)."""
    from src import passes, synthetic_gate
    path = synthetic_gate.gate_path(cfg, root, pilot, passes.resolve_filter(cfg))
    state = {"path": path, "exists": path.is_file(), "hard_stop": None, "low_confidence": None}
    if state["exists"]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        state["hard_stop"] = None if doc.get("hard_stop") is None else bool(doc["hard_stop"])
        state["low_confidence"] = None if doc.get("low_confidence") is None else bool(doc["low_confidence"])
    return state


def check_phase_gate(cfg, root, phase, pilot):
    """(refusal reason or None, low_confidence). Phases 2 to 4 of a full run need a gate file without hard_stop (§18.1: a
    hard stop writes no phase1.done, and the gate is read here as a second line); a pilot never stops. low_confidence is
    carried forward from the gate into every later flag."""
    if phase == 1:
        return None, None
    g = gate_state(cfg, root, pilot)
    if not pilot:
        if not g["exists"]:
            return f"gate file {g['path']} not found (CLAUDE.md rule 6)", None
        if g["hard_stop"] is not False:
            return f"gate file {g['path']} records a hard stop (or none was recorded)", g["low_confidence"]
    return None, g["low_confidence"]


def read_flag(path):
    """The phase flag as a dict: the JSON of this version, {"legacy": True} for an older timestamp-only flag, None if absent."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        return {"legacy": True, "text": text.strip()}
    return doc if isinstance(doc, dict) else {"legacy": True, "text": text.strip()}


def write_flag(cfg, root, phase, pilot, *, low_confidence, wall_seconds, steps):
    """outputs/phase<N>.done (pilot: results/pilot/phase<N>.done) as JSON: timestamp, commit, config hash, low_confidence,
    wall seconds and the per-step record (answer 10 of 2026-10-02)."""
    flag = flag_path(cfg, root, phase, pilot)
    flag.parent.mkdir(parents=True, exist_ok=True)
    doc = {"phase": int(phase), "pilot": bool(pilot), "mechanics_only": bool(pilot),
           "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_commit": _git_head(root),
           "config_sha256": _sha256_file(Path(root) / "config.yml"), "low_confidence": low_confidence,
           "wall_seconds": float(wall_seconds), "steps": steps}
    flag.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return flag


# ---------------------------------------------------------------- steps, resumability (H0, IMP-089)

class PhaseError(RuntimeError):
    """A phase step cannot continue; the phase fails and writes no flag."""


class StepRefused(PhaseError):
    """An existing output cannot be reused and would have to be replaced: needs --force."""


class NotBuilt(PhaseError):
    """The step belongs to a stage that has not been built yet (named in the message)."""


RUN_STATE = {"force": False, "steps": {}}


@dataclass
class Step:
    name: str
    outputs: list                  # the files whose existence marks the step as done
    run: object                    # run(force: bool) -> None
    always: bool = False           # a derived step (the summary): rerun on every call


def stale_reason(path, cfg, root):
    """(reason or None, drift list) for one existing output. JSON outputs with a provenance block must still match the
    split file they were made with, and the frozen equation (a file made without one, or with another, is stale once a
    different one exists). Config and code hash differences are drift: logged, never a refusal (a rerun is guarded by the
    drivers' write-once rule anyway)."""
    path = Path(path)
    if path.suffix != ".json":
        return None, []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return f"{path.name} is not valid JSON", []
    prov = doc.get("provenance") if isinstance(doc, dict) else None
    if not isinstance(prov, dict):
        return None, []
    seed, drift = doc.get("split_seed"), []
    if seed is not None and "split_file_sha256" in prov:
        if prov["split_file_sha256"] != _sha256_file(split_path(cfg, root, seed)):
            return f"{path.name} was made with a different split file (the split is never redrawn, §11.1)", []
    if seed is not None and "frozen_equation_sha256" in prov:
        from src import regression
        fp = regression.frozen_equation_path(cfg, root, seed, bool(doc.get("pilot")))
        if prov["frozen_equation_sha256"] != _sha256_file(fp):
            return f"{path.name} was made with another (or no) frozen equation than the one on disk", []
    if prov.get("config_yml_sha256") not in (None, _sha256_file(Path(root) / "config.yml")):
        drift.append("config.yml changed since the output was written")
    return None, drift


def run_steps(steps, cfg, root, pilot, force):
    """Run the steps in order: an existing, reusable output is skipped (resumability), a stale one refuses without --force,
    a missing one runs. The per-step record goes to RUN_STATE for the phase flag."""
    import time
    log = logging.getLogger(LOGGER_NAME)
    from src import preprocess
    with (nullcontext() if pilot else preprocess.all_subjects_permitted()):
        for st in steps:
            t0 = time.monotonic()
            existing = bool(st.outputs) and all(Path(p).is_file() for p in st.outputs)
            if existing and not force and not st.always:
                drift = []
                for p in st.outputs:
                    reason, d = stale_reason(p, cfg, root)
                    if reason:
                        raise StepRefused(f"step {st.name}: {reason}; re-run with --force "
                                          f"(a deviation entry for outputs that are never redrawn)")
                    drift += d
                for d in sorted(set(drift)):
                    log.warning("step %s: %s (the output is reused; --force recomputes it)", st.name, d)
                RUN_STATE["steps"][st.name] = {"status": "skipped", "seconds": 0.0, "drift": sorted(set(drift))}
                log.info("step %s: up to date, skipped", st.name)
                continue
            log.info("step %s: running%s", st.name, " (--force)" if force and existing else "")
            st.run(force)
            RUN_STATE["steps"][st.name] = {"status": "ran", "seconds": time.monotonic() - t0, "drift": []}


def _seeds(cfg, pilot):
    sp = cfg["split"]
    return [sp["primary_seed"]] if pilot else [sp["primary_seed"]] + list(sp["extra_seeds"])


def phase1_steps(cfg, root, pilot):
    from src import preprocess, synthetic_gate
    root = Path(root)
    ids = sorted(preprocess.load_pilot_ids(cfg, root)) if pilot else None
    pilot_ids = preprocess.load_pilot_ids(cfg, root) if pilot else frozenset()
    summary = read_manifest_summary(cfg, root)
    steps = []
    if not pilot:
        def download_step(force):
            import download
            download.run_download(cfg, root, logging.getLogger(LOGGER_NAME))
        steps.append(Step("download", [root / cfg["paths"]["manifest_file"]], download_step))
    variants = cfg["statistics"]["sensitivity"]["variants"]

    def variant_step(variant):
        path = preprocess.exclusions_path(cfg, root, variant["name"], pilot)

        def run(force):
            structure = preprocess.preprocess_variant(cfg, root, variant, ids, allow_all=not pilot, pilot_ids=pilot_ids)
            preprocess.write_exclusions(path, structure, variant=variant["name"],
                                        config_sha256=_sha256_file(root / "config.yml"), manifest_summary=summary)
            h = structure["units_check_halt"]
            logging.getLogger(LOGGER_NAME).info("preprocess %s: units check failed %d of %d recordings (%.3f)",
                                                variant["name"], h["n_failed"], h["n_total"], h["fraction"])
            if h["halt"]:
                raise PhaseError(f"variant {variant['name']}: units-check halt, {h['n_failed']} of {h['n_total']} recordings "
                                 f"failed (more than the §4.2 limit); {path.name} written, no phase flag")
        return Step(f"preprocess:{variant['name']}", [path], run)

    steps += [variant_step(v) for v in variants]
    ir = preprocess.impulse_response_path(cfg, root, pilot)
    steps.append(Step("impulse_response", [ir], lambda force: preprocess.save_impulse_responses(cfg, ir)))
    from src import passes
    if pilot:
        fname = passes.resolve_filter(cfg)
        gate = synthetic_gate.gate_path(cfg, root, True, fname)
        steps.append(Step("g0_pilot", [gate], lambda force: synthetic_gate.run_pilot(cfg, root, options=[fname])))
    else:
        def g0_full(force):
            raise NotBuilt("the full G0 driver (PLAN Stage K, phase 1) is not built; gate.json cannot be written")
        steps.append(Step("g0", [root / cfg["paths"]["gate_file"]], g0_full))
    return steps


def phase2_steps(cfg, root, pilot):
    from src import baseline, tuning
    root = Path(root)
    steps = []
    for s in _seeds(cfg, pilot):
        paths = list(baseline.output_paths(cfg, root, s, pilot))
        steps.append(Step(f"baseline:{s}", paths, lambda force, s=s: baseline.run_baseline(cfg, root, s, pilot=pilot, force=force)))
    s0 = cfg["split"]["primary_seed"]
    steps.append(Step(f"qr_report:{s0}", [tuning.qr_path(cfg, root, s0, pilot)],
                      lambda force: tuning.run_real(cfg, root, s0, pilot=pilot, force=force)))
    from src import regression

    def primary_fit(force):
        raise NotBuilt("the real-data primary PySR fit and the frozen-equation write (the phase 2 PySR driver) are not built")
    steps.append(Step("primary_fit", [regression.frozen_equation_path(cfg, root, s, pilot) for s in _seeds(cfg, pilot)],
                      primary_fit))
    return steps


def phase3_steps(cfg, root, pilot):
    from src import robustness as rb
    root = Path(root)
    seeds, s0 = _seeds(cfg, pilot), cfg["split"]["primary_seed"]
    conf = not pilot
    steps = [Step(f"c1:{s}", [rb.output_path(cfg, root, s, pilot)],
                  lambda force, s=s: rb.run_c1(cfg, root, s, pilot=pilot, confirmatory=conf, force=force)) for s in seeds]
    steps += [
        Step("c2", [rb._result_path(cfg, root, s0, pilot, "c2")], lambda force: rb.run_c2(cfg, root, pilot=pilot, confirmatory=conf, force=force)),
        Step("c3", [rb.c3_output_path(cfg, root, s0, pilot)], lambda force: rb.run_c3(cfg, root, s0, pilot=pilot, confirmatory=conf, force=force)),
        Step("c4", [rb._result_path(cfg, root, s0, pilot, "c4")], lambda force: rb.run_c4(cfg, root, s0, pilot=pilot, confirmatory=conf, force=force)),
        Step("diagnostics", [rb.diagnostics_output_path(cfg, root, s0, pilot)],
             lambda force: rb.run_diagnostics(cfg, root, s0, pilot=pilot, confirmatory=conf, force=force)),
        Step("sensitivity", [rb._result_path(cfg, root, s0, pilot, "sensitivity")],
             lambda force: rb.run_sensitivity(cfg, root, s0, pilot=pilot, confirmatory=conf, force=force))]

    def summary(force):
        secs = {k: v["seconds"] for k, v in RUN_STATE["steps"].items()}
        rb.run_summary(cfg, root, s0, pilot=pilot, timings={"phase3": float(sum(secs.values())), "phase3_steps": secs})
    steps.append(Step("summary", list(rb.summary_paths(cfg, root, pilot)), summary, always=True))
    return steps


def phase4_steps(cfg, root, pilot):
    def figures(force):
        raise NotBuilt("src/figures.py (PLAN H1) is not built")
    return [Step("figures", [Path(root) / cfg["paths"]["figures_dir"]], figures)]


PHASE_STEPS = {1: phase1_steps, 2: phase2_steps, 3: phase3_steps, 4: phase4_steps}


def _make_runner(phase):
    def runner(cfg, root, pilot):
        log = logging.getLogger(LOGGER_NAME)
        try:
            run_steps(PHASE_STEPS[phase](cfg, root, pilot), cfg, root, pilot, RUN_STATE["force"])
        except PhaseError as exc:
            log.error("phase %d: %s: %s", phase, type(exc).__name__, exc)
            return False
        except Exception as exc:                                                       # noqa: BLE001
            log.exception("phase %d: step failed: %s", phase, exc)
            return False
        return True
    return runner


# A runner returns True only on success; run_phase() writes the .done flag on that basis alone.
PHASE_RUNNERS = {n: _make_runner(n) for n in (1, 2, 3, 4)}


def run_phase(cfg, root, phase, pilot, force=False):
    import time
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("phase %d start (pilot=%s, force=%s)", phase, pilot, force)
    if force and not pilot:
        logger.warning("--force on a full run: any output replaced that is never redrawn (split, frozen equation, "
                       "confirmatory results) must be logged in DEVIATIONS.md")
    RUN_STATE.update(force=bool(force), steps={})
    t0 = time.monotonic()
    if PHASE_RUNNERS[phase](cfg, root, pilot) is not True:
        logger.error("phase %d did not succeed; no flag written", phase)
        return EXIT_FAILED
    low = None
    try:
        low = gate_state(cfg, root, pilot)["low_confidence"]
    except Exception:                                                                  # noqa: BLE001
        pass                                                                           # no gate file: unknown, recorded as null
    flag = write_flag(cfg, root, phase, pilot, low_confidence=low, wall_seconds=time.monotonic() - t0,
                      steps=dict(RUN_STATE["steps"]))
    logger.info("phase %d done; flag %s (low_confidence %s)", phase, flag, low)
    return EXIT_OK


def main(argv=None, root=None):
    args = parse_args(argv)
    root = Path(root) if root is not None else REPO_ROOT
    cfg = load_config(root / "config.yml")
    ensure_dirs(cfg, root, args.pilot)
    if args.make_split:
        _setup_logger(root / cfg["paths"]["logs_dir"] / "split.log")
        try:
            ok = run_make_split(cfg, root, dry_run=args.dry_run, force=args.force)
        except SplitError as exc:
            logging.getLogger(LOGGER_NAME).error("split failed: %s", exc)
            return EXIT_FAILED
        return EXIT_OK if ok else EXIT_FAILED
    logger = setup_logging(cfg, root, args.phase, args.pilot)
    missing = check_prerequisite(cfg, root, args.phase, args.pilot)
    if missing is not None:
        logger.error("phase %d refused: required flag %s not found",
                     args.phase, missing)
        return EXIT_PREREQ
    reason, _low = check_phase_gate(cfg, root, args.phase, args.pilot)
    if reason is not None:
        logger.error("phase %d refused: %s", args.phase, reason)
        return EXIT_PREREQ
    return run_phase(cfg, root, args.phase, args.pilot, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
