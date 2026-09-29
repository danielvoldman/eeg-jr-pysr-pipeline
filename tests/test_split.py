"""B2: subject split (§11.1, §11.2, §17, IMP-008). Offline, on a fake data tree in tmp_path."""
import copy
import json
import logging

import numpy as np
import pytest

import main
from src.config import load_config

N, N_T2 = 111, 42
SEEDS = (42, 43, 44, 45, 46)


def build_tree(root, n=N, n_t2=N_T2, extra_files=0, tree_seed=0):
    """Fake ds003775 tree: n subjects, n_t2 of them with ses-t2, empty EDF files.

    Subjects and t2 are chosen and created in a shuffled order, so the file-system
    creation order differs between tree_seed values.
    """
    ids = [f"sub-{i:03d}" for i in range(1, n + 1)]
    t2 = {ids[i] for i in np.random.default_rng(0).choice(n, n_t2, replace=False)}
    rng = np.random.default_rng(1000 + tree_seed)
    for i in rng.permutation(n):
        sub = ids[i]
        for ses in ("ses-t1", "ses-t2") if sub in t2 else ("ses-t1",):
            d = root / "data" / sub / ses / "eeg"
            d.mkdir(parents=True)
            (d / f"{sub}_{ses}_task-rest_eeg.edf").write_bytes(b"")
    for k in range(extra_files):
        (root / "data" / ids[0] / "ses-t1" / "eeg" / f"extra{k}.edf").write_bytes(b"")
    return ids, t2


@pytest.fixture
def cfg(temp_root):
    return load_config(temp_root / "config.yml")


@pytest.fixture
def tree(temp_root):
    ids, t2 = build_tree(temp_root)
    return ids, t2


@pytest.fixture
def splits(tree, cfg):
    ids, t2 = tree
    return {s: main.make_split(ids, t2, cfg, s) for s in SEEDS}


# ---- sizes, structure --------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_sizes_disjoint_union(splits, tree, seed):
    ids, _ = tree
    sp = splits[seed]
    assert len(sp["train"]) == 78 and len(sp["test"]) == 33
    assert not set(sp["train"]) & set(sp["test"])
    assert sorted(sp["train"] + sp["test"]) == ids


def test_config_seeds_and_leaves(cfg):
    assert main.split_seeds(cfg) == list(SEEDS)
    assert cfg["split"]["pilot_draw_seed"] == 42


@pytest.mark.parametrize("seed", SEEDS)
def test_split_is_by_subject_and_t2_follows_t1(splits, tree, seed):
    _, t2 = tree
    sp = splits[seed]
    assert set(sp["subjects_with_t2"]) == t2
    # the file lists subject IDs only, so a t2 recording cannot sit on the other side
    assert all(s.startswith("sub-") and "ses" not in s for s in sp["train"] + sp["test"])
    for s in t2:
        assert (s in sp["train"]) != (s in sp["test"])


# ---- pilots ------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_pilots_count_t2_and_train_side(splits, tree, seed):
    _, t2 = tree
    sp = splits[seed]
    assert len(sp["pilot"]) == 12 and len(set(sp["pilot"])) == 12
    assert len(set(sp["pilot"]) & t2) == 4
    assert set(sp["pilot"]) <= set(sp["train"])


def test_pilots_identical_across_seeds_and_from_seed42_train(splits):
    assert len({tuple(splits[s]["pilot"]) for s in SEEDS}) == 1
    assert set(splits[42]["pilot"]) <= set(splits[42]["train"])


@pytest.mark.parametrize("seed", SEEDS[1:])
def test_extra_seed_test_sets_contain_no_pilot(splits, seed):
    assert not set(splits[seed]["test"]) & set(splits[42]["pilot"])


# ---- independent recomputation (amendment 2) ---------------------------------

def _expected(ids, t2, cfg):
    """Own few lines of numpy, not calling main.make_split or its helpers."""
    ids = sorted(ids)
    order = np.random.default_rng(42).permutation(111)
    train42 = sorted(ids[i] for i in order[:78])
    test42 = sorted(ids[i] for i in order[78:])
    rp = np.random.default_rng(cfg["split"]["pilot_draw_seed"])
    a = [s for s in train42 if s in t2]
    b = [s for s in train42 if s not in t2]
    pilots = sorted([a[i] for i in rp.choice(len(a), 4, replace=False)]
                    + [b[i] for i in rp.choice(len(b), 8, replace=False)])
    out = {42: (train42, test42)}
    for seed in (43, 44, 45, 46):
        pool = [s for s in ids if s not in pilots]
        o = np.random.default_rng(seed).permutation(99)
        test = sorted(pool[i] for i in o[:33])
        train = sorted([pool[i] for i in o[33:]] + pilots)
        out[seed] = (train, test)
    return out, pilots


def test_matches_independent_recomputation(splits, tree, cfg):
    ids, t2 = tree
    exp, pilots = _expected(ids, t2, cfg)
    for seed in SEEDS:
        assert splits[seed]["train"] == exp[seed][0], seed
        assert splits[seed]["test"] == exp[seed][1], seed
        assert splits[seed]["pilot"] == pilots


def test_pilot_draw_does_not_change_seed42_train_test(tree, cfg):
    ids, t2 = tree
    base = main.make_split(ids, t2, cfg, 42)
    other_cfg = copy.deepcopy(cfg)
    other_cfg["split"]["pilot_draw_seed"] = 7
    moved = main.make_split(ids, t2, other_cfg, 42)
    assert moved["pilot"] != base["pilot"]  # the pilot draw did change
    assert moved["train"] == base["train"] and moved["test"] == base["test"]
    exp, _ = _expected(ids, t2, cfg)
    assert (base["train"], base["test"]) == exp[42]


# ---- determinism -------------------------------------------------------------

def test_two_runs_byte_identical(tmp_path_factory, cfg):
    outs = []
    for k in range(2):
        root = tmp_path_factory.mktemp(f"run{k}")
        build_tree(root, tree_seed=0)
        assert main.run_make_split(cfg, root)
        outs.append({s: main.split_path(cfg, root, s).read_bytes() for s in SEEDS})
    assert outs[0] == outs[1]


def test_file_system_order_does_not_matter(tmp_path_factory, cfg):
    made = []
    for tree_seed in (0, 1):  # same subjects and t2 set, different creation order
        root = tmp_path_factory.mktemp(f"order{tree_seed}")
        build_tree(root, tree_seed=tree_seed)
        ids_found, t2_found = main.discover_subjects(cfg, root)
        made.append([main.make_split(ids_found, t2_found, cfg, s) for s in SEEDS])
    assert made[0] == made[1]


def test_changing_seed_changes_split(splits, tree, cfg):
    assert len({tuple(splits[s]["test"]) for s in SEEDS}) == 5
    ids, t2 = tree
    other = copy.deepcopy(cfg)
    other["split"]["primary_seed"] = 99
    other["split"]["extra_seeds"] = [43, 44, 45, 46]
    assert main.make_split(ids, t2, other, 99)["test"] != splits[42]["test"]


def test_file_layout(temp_root, cfg, tree):
    main.run_make_split(cfg, temp_root)
    raw = main.split_path(cfg, temp_root, 42).read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    d = json.loads(raw.decode("utf-8"))
    assert set(d) == {"seed", "procedure", "dataset_id", "dataset_version",
                      "subject_ids_sha256", "train", "test", "pilot",
                      "subjects_with_t2", "manifest_summary", "numpy_version"}
    assert d["dataset_id"] == "ds003775" and d["dataset_version"] == "1.2.1"
    assert d["numpy_version"] == np.__version__
    assert d["manifest_summary"] is None
    assert list(d) == sorted(d)  # sorted keys


def test_manifest_summary_line_recorded(temp_root, cfg, tree):
    line = "# files=632 manifest_sha256=abc dataset=ds003775 version=1.2.1 openneuro_py=x"
    (temp_root / "data" / "MANIFEST.sha256").write_text(f"h\t1\ta\n{line}\n", encoding="utf-8")
    ids, t2 = tree
    assert main.make_split(ids, t2, cfg, 42,
                           main.read_manifest_summary(cfg, temp_root))["manifest_summary"] == line


# ---- overwrite rule ----------------------------------------------------------

def test_identical_regeneration_is_noop(temp_root, cfg, splits):
    sp = splits[42]
    assert main.write_split(sp, temp_root, cfg) == "created"
    path = main.split_path(cfg, temp_root, 42)
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    assert main.write_split(sp, temp_root, cfg) == "unchanged"
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime


def test_differing_result_refuses_then_force_overwrites(temp_root, cfg, splits):
    sp = splits[42]
    main.write_split(sp, temp_root, cfg)
    path = main.split_path(cfg, temp_root, 42)
    before = path.read_bytes()
    changed = dict(sp, manifest_summary="# files=1 changed")
    assert main.write_split(changed, temp_root, cfg) == "differs-refused"
    assert path.read_bytes() == before
    assert main.write_split(changed, temp_root, cfg, dry_run=True, force=True) == "overwritten"
    assert path.read_bytes() == before  # dry run wrote nothing
    assert main.write_split(changed, temp_root, cfg, force=True) == "overwritten"
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_summary"] == "# files=1 changed"


def test_only_numpy_version_differs_is_unchanged_with_warning(temp_root, cfg, splits, caplog):
    sp = splits[42]
    main.write_split(dict(sp, numpy_version="0.0.0-old"), temp_root, cfg)
    path = main.split_path(cfg, temp_root, 42)
    before = path.read_bytes()
    with caplog.at_level(logging.WARNING, logger=main.LOGGER_NAME):
        assert main.write_split(sp, temp_root, cfg) == "unchanged"
    assert path.read_bytes() == before  # file untouched, old numpy_version kept
    assert any("numpy_version" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_numpy_version_and_other_field_differing_refuses(temp_root, cfg, splits):
    sp = splits[42]
    main.write_split(dict(sp, numpy_version="0.0.0-old"), temp_root, cfg)
    changed = dict(sp, manifest_summary="# files=1 changed")
    assert main.write_split(changed, temp_root, cfg) == "differs-refused"


def test_run_make_split_writes_none_if_any_refused(temp_root, cfg, tree, caplog):
    assert main.run_make_split(cfg, temp_root)
    main.split_path(cfg, temp_root, 44).write_text("{}", encoding="utf-8")  # corrupt one
    main.split_path(cfg, temp_root, 43).unlink()
    assert not main.run_make_split(cfg, temp_root)
    assert not main.split_path(cfg, temp_root, 43).exists()  # nothing was written
    assert main.run_make_split(cfg, temp_root, force=True)
    assert main.split_path(cfg, temp_root, 43).is_file()


# ---- validate_split: one test per rule ---------------------------------------

def _good(splits):
    return copy.deepcopy(splits[43])


def test_validate_accepts_good_splits(splits, cfg):
    for sp in splits.values():
        main.validate_split(sp, cfg)


def test_validate_catches_overlap(splits, cfg):
    sp = _good(splits)
    sp["test"][0] = sp["train"][0]
    with pytest.raises(main.SplitError, match="overlap"):
        main.validate_split(sp, cfg)


def test_validate_catches_missing_subject(splits, cfg):
    sp = _good(splits)
    sp["train"].pop()
    with pytest.raises(main.SplitError, match="union"):
        main.validate_split(sp, cfg)


def test_validate_catches_pilot_in_test(splits, cfg):
    sp = _good(splits)
    pilot = sp["pilot"][0]
    sp["train"].remove(pilot)
    sp["train"].append(next(s for s in sp["test"] if s not in sp["subjects_with_t2"]))
    sp["test"] = [pilot if s == sp["train"][-1] else s for s in sp["test"]]
    with pytest.raises(main.SplitError, match="pilot subjects not in train"):
        main.validate_split(sp, cfg)


def test_validate_catches_wrong_pilot_t2_count(splits, cfg):
    sp = _good(splits)
    t2 = set(sp["subjects_with_t2"])
    drop = next(s for s in sp["pilot"] if s in t2)
    add = next(s for s in sp["train"] if s not in t2 and s not in sp["pilot"])
    sp["pilot"] = sorted([s for s in sp["pilot"] if s != drop] + [add])
    with pytest.raises(main.SplitError, match="pilot t2 count 3"):
        main.validate_split(sp, cfg)


def test_validate_catches_t2_subject_split_across_sides(splits, cfg):
    sp = _good(splits)
    t2 = next(s for s in sp["test"] if s in sp["subjects_with_t2"])
    sp["train"].append(t2)  # now on both sides
    with pytest.raises(main.SplitError, match="t2 subjects not on exactly one side"):
        main.validate_split(sp, cfg)


def test_validate_catches_wrong_sizes_and_pilot_count(splits, cfg):
    sp = _good(splits)
    sp["pilot"] = sp["pilot"][:-1]
    with pytest.raises(main.SplitError, match="pilot count 11"):
        main.validate_split(sp, cfg)


# ---- count check -------------------------------------------------------------

def test_count_check_110_subjects(tmp_path, cfg):
    build_tree(tmp_path, n=110, n_t2=42, extra_files=1)  # 153 EDFs, so only the subject count is off
    with pytest.raises(main.SplitError, match="expected 111 subjects, found 110|found 110"):
        main.discover_subjects(cfg, tmp_path)


def test_count_check_41_t2_subjects(tmp_path, cfg):
    build_tree(tmp_path, n=111, n_t2=41, extra_files=1)  # 153 EDFs, so only the t2 count is off
    with pytest.raises(main.SplitError, match="found 41"):
        main.discover_subjects(cfg, tmp_path)


def test_count_check_edf_total(tmp_path, cfg):
    build_tree(tmp_path, extra_files=1)
    with pytest.raises(main.SplitError, match="EDF files"):
        main.discover_subjects(cfg, tmp_path)


def test_discover_on_good_tree(tmp_path, cfg):
    ids, t2 = build_tree(tmp_path)
    found_ids, found_t2 = main.discover_subjects(cfg, tmp_path)
    assert found_ids == ids and found_t2 == t2


# ---- accessors ---------------------------------------------------------------

def test_load_split_and_accessors(temp_root, cfg, tree):
    main.run_make_split(cfg, temp_root)
    sp = main.load_split(43, root=temp_root)
    assert main.training_subjects(43, root=temp_root) == sp["train"]
    assert main.test_subjects(43, root=temp_root) == sp["test"]
    with pytest.raises(main.SplitError, match="not found"):
        main.load_split(1, root=temp_root)


def test_load_split_rejects_tampered_file(temp_root, cfg, tree):
    main.run_make_split(cfg, temp_root)
    path = main.split_path(cfg, temp_root, 42)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["pilot"] = d["test"][:12]
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(main.SplitError):
        main.load_split(42, root=temp_root)


# ---- CLI ---------------------------------------------------------------------

def test_cli_dry_run_writes_nothing(temp_root, tree):
    assert main.main(["--make-split", "--dry-run"], root=temp_root) == main.EXIT_OK
    assert not list((temp_root / "outputs").glob("split_*"))


def test_cli_make_split_then_rerun_is_noop(temp_root, cfg, tree):
    assert main.main(["--make-split"], root=temp_root) == main.EXIT_OK
    files = {s: main.split_path(cfg, temp_root, s).read_bytes() for s in SEEDS}
    assert main.main(["--make-split"], root=temp_root) == main.EXIT_OK
    assert files == {s: main.split_path(cfg, temp_root, s).read_bytes() for s in SEEDS}
    assert (temp_root / "logs" / "split.log").is_file()


def test_cli_refuses_when_result_differs(temp_root, cfg, tree):
    main.main(["--make-split"], root=temp_root)
    path = main.split_path(cfg, temp_root, 42)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["manifest_summary"] = "# files=1"
    path.write_bytes(main._split_bytes(d))
    assert main.main(["--make-split"], root=temp_root) == main.EXIT_FAILED
    assert main.main(["--make-split", "--dry-run"], root=temp_root) == main.EXIT_FAILED
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_summary"] == "# files=1"
    assert main.main(["--make-split", "--force"], root=temp_root) == main.EXIT_OK
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_summary"] is None


def test_cli_bad_tree_fails_loudly(temp_root):
    build_tree(temp_root, n=110, n_t2=42, extra_files=1)
    assert main.main(["--make-split"], root=temp_root) == main.EXIT_FAILED
    assert not list((temp_root / "outputs").glob("split_*"))


@pytest.mark.parametrize("argv", [["--make-split", "--pilot"], ["--dry-run"], ["--force"],
                                  ["--phase", "1", "--dry-run"], ["--phase", "1", "--make-split"],
                                  []])
def test_cli_argument_errors(temp_root, argv):
    with pytest.raises(SystemExit) as e:
        main.main(argv, root=temp_root)
    assert e.value.code == 2


def test_cli_phase_arguments_unchanged():
    a = main.parse_args(["--phase", "3", "--pilot"])
    assert (a.phase, a.pilot, a.make_split) == (3, True, False)


# ---- statistical sanity (printed, not a pass rule) ---------------------------

def test_print_statistical_sanity(splits, capsys):
    counts = {}
    for s in SEEDS:
        for sub in splits[s]["test"]:
            counts[sub] = counts.get(sub, 0) + 1
    multi = sum(1 for c in counts.values() if c > 1)
    t2 = set(splits[42]["subjects_with_t2"])
    per_seed = {s: len(set(splits[s]["test"]) & t2) for s in SEEDS}
    with capsys.disabled():
        print(f"\n[split sanity] subjects on the test side in more than one seed: {multi} "
              f"of {len(counts)} ever tested")
        print(f"[split sanity] t2 subjects on the test side per seed: {per_seed}")
