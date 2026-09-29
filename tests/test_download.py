"""Offline tests for download.py (PLAN.md B1; IMP-007). Fake dataset trees in tmp_path only:
no network, no real data/, cache/, outputs/, results/ or logs/."""
import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import download
from src.config import load_config

N_SUBJECTS, N_T2 = 111, 42
SKIPPED = "sub-010/ses-t1/sub-010_ses-t1_scans.tsv"


def _sub(i):
    return f"sub-{i:03d}"


def make_remote():
    """rel path -> bytes for a fake snapshot: root files, code/, derivatives/, 111 subjects, 42 with t2."""
    remote = {
        "README": b"readme", "CHANGES": b"changes", "dataset_description.json": b"{}",
        "participants.tsv": b"p", "participants.json": b"{}", ".gitattributes": b"x",
        "code/s2_preprocess.m": b"% matlab", "derivatives/sub-001/ses-t1/eeg/d.set": b"D" * 50,
        "derivatives/participants.tsv": b"d",
    }
    for i in range(1, N_SUBJECTS + 1):
        for ses in ["ses-t1"] + (["ses-t2"] if i <= N_T2 else []):
            base = f"{_sub(i)}/{ses}"
            stem = f"{_sub(i)}_{ses}_task-rest"
            remote[f"{base}/{_sub(i)}_{ses}_scans.tsv"] = f"scans {i} {ses}".encode()
            remote[f"{base}/eeg/{stem}_channels.tsv"] = b"ch"
            remote[f"{base}/eeg/{stem}_eeg.json"] = b"{}"
            remote[f"{base}/eeg/{stem}_eeg.edf"] = f"edf {i} {ses}".encode() * 7
    return remote


REMOTE = make_remote()


def fetch_fake(cfg, remote=REMOTE):
    return [(name, len(data)) for name, data in remote.items()]


class FakeDownloader:
    """Mimics openneuro.download: honours include/exclude with openneuro's own glob rules."""

    def __init__(self, remote=REMOTE, truncate=None):
        self.remote, self.truncate, self.calls = remote, truncate, []

    def __call__(self, **kwargs):
        from openneuro._glob import glob_filter
        self.calls.append(kwargs)
        names = list(self.remote)
        inc = {n for v in glob_filter(names, kwargs["include"]).values() for n in v}
        exc = {n for v in glob_filter(names, kwargs["exclude"]).values() for n in v}
        keep = (inc - exc) | (download.OPENNEURO_ESSENTIAL_FILES & set(names))
        for name in keep:
            path = Path(kwargs["target_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self.remote[name]
            if name == self.truncate:
                data = data[:-3]
            path.write_bytes(data)


@pytest.fixture(autouse=True)
def _close_download_handlers():
    yield
    logger = logging.getLogger(download.LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def cfg(temp_root):
    return load_config(temp_root / "config.yml")


@pytest.fixture
def logger(cfg, temp_root):
    return download.setup_logging(cfg, temp_root)


@pytest.fixture
def tree(temp_root, cfg):
    """A complete downloaded tree (what a correct run leaves), written by hand."""
    data = temp_root / cfg["paths"]["data_dir"]
    for name in download.select_files(fetch_fake(cfg), cfg).files:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(REMOTE[name])
    return data


def edf_of(subject, ses="ses-t1"):
    return f"{subject}/{ses}/eeg/{subject}_{ses}_task-rest_eeg.edf"


# ---------------------------------------------------------------- counts
def test_counts_pass_on_full_tree(tree, cfg):
    counts = download.validate_tree(tree, cfg)
    assert counts["subjects"] == 111
    assert counts["EDF files"] == 153
    assert counts["subjects with a ses-t2 EDF"] == 42
    assert counts["scans.tsv files (recordings minus the skipped one)"] == 152


def test_counts_fail_missing_subject(tree, cfg):
    import shutil
    shutil.rmtree(tree / "sub-100")
    with pytest.raises(download.DownloadValidationError, match=r"subjects: found 110, expected 111"):
        download.validate_tree(tree, cfg)


def test_counts_fail_missing_edf(tree, cfg):
    (tree / edf_of("sub-100")).unlink()
    with pytest.raises(download.DownloadValidationError, match=r"EDF files: found 152, expected 153"):
        download.validate_tree(tree, cfg)


def test_counts_fail_extra_edf(tree, cfg):
    extra = tree / "sub-100/ses-t1/eeg/sub-100_ses-t1_task-other_eeg.edf"
    extra.write_bytes(b"x")
    with pytest.raises(download.DownloadValidationError, match=r"EDF files: found 154, expected 153"):
        download.validate_tree(tree, cfg)


def test_counts_fail_wrong_t2_count(tree, cfg):
    # a t2 EDF for a subject without one: t2 subjects become 43
    path = tree / edf_of("sub-100", "ses-t2")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    with pytest.raises(download.DownloadValidationError, match=r"ses-t2 EDF: found 43, expected 42"):
        download.validate_tree(tree, cfg)


def test_counts_report_every_failure_together(tree, cfg):
    (tree / edf_of("sub-100")).unlink()
    (tree / edf_of("sub-001", "ses-t2")).unlink()
    with pytest.raises(download.DownloadValidationError) as exc:
        download.validate_tree(tree, cfg)
    assert "EDF files" in str(exc.value) and "ses-t2 EDF" in str(exc.value)


def test_scans_count_is_152_not_153(tree, cfg):
    (tree / SKIPPED).parent.mkdir(parents=True, exist_ok=True)
    (tree / SKIPPED).write_bytes(b"x")  # the skipped file appears: 153
    with pytest.raises(download.DownloadValidationError, match=r"scans.tsv files.*found 153, expected 152"):
        download.validate_tree(tree, cfg)


def test_scans_missing_one_fails(tree, cfg):
    (tree / "sub-011/ses-t1/sub-011_ses-t1_scans.tsv").unlink()
    with pytest.raises(download.DownloadValidationError, match=r"scans.tsv files.*found 151, expected 152"):
        download.validate_tree(tree, cfg)


# ---------------------------------------------------------------- manifest
def test_manifest_lists_every_file_once_with_independent_hashes(tree, cfg, temp_root):
    files = download.list_local_files(tree)
    entries = download.build_entries(tree, files, cfg["download"]["hash_chunk_bytes"])
    assert [e[2] for e in entries] == sorted(files) and len({e[2] for e in entries}) == len(files)
    for sha, size, rel in entries:
        assert sha == hashlib.sha256(REMOTE[rel]).hexdigest()
        assert size == len(REMOTE[rel])
    path = temp_root / cfg["paths"]["manifest_file"]
    download.write_manifest(path, entries, cfg)
    lines = path.read_text(encoding="utf-8").splitlines()
    body = "".join(f"{hashlib.sha256(REMOTE[r]).hexdigest()}\t{len(REMOTE[r])}\t{r}\n" for r in sorted(files))
    assert lines[:-1] == body.splitlines()
    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert lines[-1] == (f"# files={len(files)} manifest_sha256={expected_hash} dataset=ds003775 "
                         f"version=1.2.1 openneuro_py={download.openneuro_py_version()}")
    assert path.name not in "".join(lines[:-1])  # the manifest does not list itself
    back, summary = download.read_manifest(path)
    assert sorted(back) == sorted(entries) and summary["dataset"] == "ds003775"


def test_summary_metadata_does_not_change_the_content_hash(tree, cfg):
    files = download.list_local_files(tree)
    entries = download.build_entries(tree, files, cfg["download"]["hash_chunk_bytes"])
    h1 = download.content_sha256(entries)
    cfg2 = load_config(Path(__file__).resolve().parent.parent / "config.yml")
    cfg2["dataset"]["version"] = "9.9.9"
    assert f"manifest_sha256={h1}" in download.manifest_text(entries, cfg2)
    assert "version=9.9.9" in download.manifest_text(entries, cfg2)


def test_edited_manifest_is_detected(tree, cfg, temp_root):
    entries = download.build_entries(tree, download.list_local_files(tree), cfg["download"]["hash_chunk_bytes"])
    path = temp_root / cfg["paths"]["manifest_file"]
    download.write_manifest(path, entries, cfg)
    path.write_text(path.read_text(encoding="utf-8").replace("\tREADME", "\tREADMX", 1), encoding="utf-8")
    with pytest.raises(download.IntegrityError):
        download.read_manifest(path)


def _manifest_of(tree, cfg):
    return download.build_entries(tree, download.list_local_files(tree), cfg["download"]["hash_chunk_bytes"])


def test_verify_ok_then_modified_truncated_missing_unexpected(tree, cfg):
    entries = _manifest_of(tree, cfg)
    assert download.verify_manifest(tree, entries, cfg).ok
    target = tree / edf_of("sub-005")
    original = target.read_bytes()

    target.write_bytes(bytes([original[0] ^ 1]) + original[1:])  # same size, one bit different
    rep = download.verify_manifest(tree, entries, cfg)
    assert [r for r, _ in rep.mismatched] == [edf_of("sub-005")] and "sha256" in rep.mismatched[0][1]

    target.write_bytes(original[:-1])  # truncated
    rep = download.verify_manifest(tree, entries, cfg)
    assert [r for r, _ in rep.mismatched] == [edf_of("sub-005")] and "size" in rep.mismatched[0][1]

    target.unlink()
    assert download.verify_manifest(tree, entries, cfg).missing == [edf_of("sub-005")]

    target.write_bytes(original)
    (tree / "stray.txt").write_bytes(b"?")
    assert download.verify_manifest(tree, entries, cfg).unexpected == ["stray.txt"]


# ---------------------------------------------------------------- selection
def test_skip_rule_logged_and_only_that_file(cfg, caplog):
    sel = download.select_files(fetch_fake(cfg), cfg)
    assert SKIPPED not in sel.files and sel.skipped == [SKIPPED]
    assert edf_of("sub-010") in sel.files                       # its EEG is kept
    assert "sub-010/ses-t2/sub-010_ses-t2_scans.tsv" in sel.files  # other session kept
    assert "sub-011/ses-t1/sub-011_ses-t1_scans.tsv" in sel.files  # other subject kept
    assert download.count_matching(sel.files, cfg["dataset"]["scans_glob"]) == 152


def test_skip_is_logged_in_list_output(cfg, temp_root, caplog):
    with caplog.at_level(logging.INFO, logger=download.LOGGER_NAME):
        assert download.main(["--list"], root=temp_root, fetch=fetch_fake) == 0
    assert any("Skipped by §4.2 rule" in r.message and SKIPPED in r.message for r in caplog.records)
    assert any("scans.tsv" in r.message and "152" in r.message for r in caplog.records)


def test_derivatives_excluded_code_and_root_files_included(cfg):
    sel = download.select_files(fetch_fake(cfg), cfg)
    assert not any(n.startswith("derivatives/") for n in sel.files)
    assert len(sel.excluded_by_config) == 2
    assert "code/s2_preprocess.m" in sel.files
    for root_file in ("README", "CHANGES", "dataset_description.json", "participants.tsv", "participants.json"):
        assert root_file in sel.files
    assert ".gitattributes" not in sel.files


# ---------------------------------------------------------------- run_download
def test_downloader_receives_include_exclude_and_pins(cfg, temp_root, logger):
    fake = FakeDownloader()
    mock = MagicMock(side_effect=fake)
    download.run_download(cfg, temp_root, logger, downloader=mock, fetch=fetch_fake)
    assert mock.call_count == 1
    kw = mock.call_args.kwargs
    assert kw["include"] == ["/sub-*", "/code"]
    assert kw["exclude"] == ["/derivatives", "sub-010/ses-t1/*_scans.tsv"]
    assert kw["dataset"] == "ds003775" and kw["tag"] == "1.2.1"
    assert kw["verify_hash"] is True and kw["verify_size"] is True
    assert not (temp_root / cfg["paths"]["data_dir"] / "derivatives").exists()
    assert (temp_root / cfg["paths"]["manifest_file"]).is_file()
    # the fake remote write really followed the arguments: no skipped file on disk
    assert not (temp_root / cfg["paths"]["data_dir"] / SKIPPED).exists()


def test_second_run_makes_zero_download_calls(cfg, temp_root, logger):
    download.run_download(cfg, temp_root, logger, downloader=FakeDownloader(), fetch=fetch_fake)
    manifest = (temp_root / cfg["paths"]["manifest_file"]).read_bytes()
    mock = MagicMock()
    fetch = MagicMock()
    download.run_download(cfg, temp_root, logger, downloader=mock, fetch=fetch)
    mock.assert_not_called()
    fetch.assert_not_called()  # fully offline when everything is present and verified
    assert (temp_root / cfg["paths"]["manifest_file"]).read_bytes() == manifest


def test_missing_file_is_refetched_and_must_match_manifest(cfg, temp_root, logger):
    download.run_download(cfg, temp_root, logger, downloader=FakeDownloader(), fetch=fetch_fake)
    (temp_root / "data" / edf_of("sub-050")).unlink()
    fake = FakeDownloader()
    download.run_download(cfg, temp_root, logger, downloader=fake, fetch=fetch_fake)
    assert len(fake.calls) == 1 and (temp_root / "data" / edf_of("sub-050")).is_file()


def test_modified_file_stops_run_and_is_never_overwritten(cfg, temp_root, logger):
    download.run_download(cfg, temp_root, logger, downloader=FakeDownloader(), fetch=fetch_fake)
    target = temp_root / "data" / edf_of("sub-050")
    target.write_bytes(b"tampered")
    mock = MagicMock()
    with pytest.raises(download.IntegrityError, match="differ from the manifest"):
        download.run_download(cfg, temp_root, logger, downloader=mock, fetch=fetch_fake)
    mock.assert_not_called()
    assert target.read_bytes() == b"tampered"


def test_truncated_download_fails_and_writes_no_manifest(cfg, temp_root, logger):
    fake = FakeDownloader(truncate=edf_of("sub-020"))
    with pytest.raises(download.IntegrityError, match="differ in size from the remote listing"):
        download.run_download(cfg, temp_root, logger, downloader=fake, fetch=fetch_fake)
    assert not (temp_root / cfg["paths"]["manifest_file"]).exists()


def test_missing_remote_file_after_download_fails(cfg, temp_root, logger):
    class Dropping(FakeDownloader):
        def __call__(self, **kw):
            super().__call__(**kw)
            (Path(kw["target_dir"]) / edf_of("sub-020")).unlink()
    with pytest.raises(download.IntegrityError):
        download.run_download(cfg, temp_root, logger, downloader=Dropping(), fetch=fetch_fake)
    assert not (temp_root / cfg["paths"]["manifest_file"]).exists()


def test_count_mismatch_in_remote_listing_fails_before_download(cfg, temp_root, logger):
    remote = {k: v for k, v in REMOTE.items() if k != edf_of("sub-020")}
    mock = MagicMock()
    with pytest.raises(download.DownloadValidationError, match="EDF files: found 152"):
        download.run_download(cfg, temp_root, logger, downloader=mock,
                              fetch=lambda c: fetch_fake(c, remote))
    mock.assert_not_called()


# ---------------------------------------------------------------- free space
def test_free_space_check():
    with pytest.raises(download.InsufficientSpaceError):
        download.check_free_space(100, 199, 2)
    download.check_free_space(100, 200, 2)


def test_low_free_space_stops_before_download(cfg, temp_root, logger, monkeypatch):
    monkeypatch.setattr(download, "_free_bytes", lambda p: 10)
    mock = MagicMock()
    with pytest.raises(download.InsufficientSpaceError):
        download.run_download(cfg, temp_root, logger, downloader=mock, fetch=fetch_fake)
    mock.assert_not_called()


# ---------------------------------------------------------------- --list failures, --verify, main
def test_list_fails_loudly_when_fetch_raises(temp_root, caplog):
    def boom(cfg):
        raise download.SnapshotUnavailable("private metadata function failed")
    with caplog.at_level(logging.ERROR, logger=download.LOGGER_NAME):
        assert download.main(["--list"], root=temp_root, fetch=boom) == 1
    assert any("SnapshotUnavailable" in r.message for r in caplog.records)


def test_fetch_snapshot_files_wraps_raise_and_missing_function(cfg, monkeypatch):
    import openneuro._download as od
    monkeypatch.setattr(od, "_get_download_metadata", MagicMock(side_effect=OSError("offline")))
    with pytest.raises(download.SnapshotUnavailable, match="private metadata function"):
        download.fetch_snapshot_files(cfg)
    monkeypatch.delattr(od, "_get_download_metadata")
    with pytest.raises(download.SnapshotUnavailable, match="missing or failed"):
        download.fetch_snapshot_files(cfg)


def test_list_via_main_exits_nonzero_when_metadata_function_missing(temp_root, monkeypatch):
    import openneuro._download as od
    monkeypatch.delattr(od, "_get_download_metadata")
    assert download.main(["--list"], root=temp_root) == 1


def test_list_fails_when_remote_has_153_scans(temp_root):
    remote = dict(REMOTE)
    remote["sub-010/ses-t1/sub-010_ses-t1_scans.tsv"] = b"x"  # still skipped by the rule -> 152
    assert download.main(["--list"], root=temp_root, fetch=lambda c: fetch_fake(c, remote)) == 0
    del remote["sub-011/ses-t1/sub-011_ses-t1_scans.tsv"]     # 151
    assert download.main(["--list"], root=temp_root, fetch=lambda c: fetch_fake(c, remote)) == 1


def test_list_downloads_nothing(temp_root, cfg):
    mock = MagicMock()
    assert download.main(["--list"], root=temp_root, downloader=mock, fetch=fetch_fake) == 0
    mock.assert_not_called()
    assert not (temp_root / cfg["paths"]["data_dir"]).exists()


def test_verify_mode(temp_root, cfg):
    assert download.main([], root=temp_root, downloader=FakeDownloader(), fetch=fetch_fake) == 0
    assert download.main(["--verify"], root=temp_root) == 0
    (temp_root / "data" / edf_of("sub-001")).write_bytes(b"bad")
    assert download.main(["--verify"], root=temp_root) == 1


def test_verify_without_manifest_fails(temp_root):
    assert download.main(["--verify"], root=temp_root) == 1


def test_log_file_is_under_the_temp_root(temp_root, cfg):
    download.main(["--list"], root=temp_root, fetch=fetch_fake)
    assert (temp_root / cfg["paths"]["download_log"]).is_file()
