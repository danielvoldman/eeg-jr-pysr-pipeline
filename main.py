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
                        help="with --make-split: overwrite differing split files "
                             "(must be logged in DEVIATIONS.md section 1)")
    args = parser.parse_args(argv)
    if not args.make_split and (args.dry_run or args.force):
        parser.error("--dry-run and --force need --make-split")
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


def _make_stub(phase):
    def runner(cfg, root, pilot):
        logging.getLogger(LOGGER_NAME).error(
            "phase %d: not implemented", phase)
        return False
    return runner


# Real runners replace these stubs as their scripts are built. A runner returns
# True only on success; main() writes the .done flag on that basis alone.
PHASE_RUNNERS = {n: _make_stub(n) for n in (1, 2, 3, 4)}


def run_phase(cfg, root, phase, pilot):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("phase %d start (pilot=%s)", phase, pilot)
    if PHASE_RUNNERS[phase](cfg, root, pilot) is not True:
        logger.error("phase %d did not succeed; no flag written", phase)
        return EXIT_FAILED
    flag = flag_path(cfg, root, phase, pilot)
    flag.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    flag.write_text(f"{stamp}\n", encoding="utf-8")
    logger.info("phase %d done; flag %s", phase, flag)
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
    return run_phase(cfg, root, args.phase, args.pilot)


if __name__ == "__main__":
    sys.exit(main())
