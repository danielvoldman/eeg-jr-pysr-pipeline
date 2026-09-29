"""Fetch ds003775 into data/ and record a SHA-256 manifest (PLAN.md B1; §4.1, §4.2, §18; IMP-007).

Standalone script, not wired into main.py. It never opens an EDF.

    python download.py --list     what would be fetched, counts, total size; downloads nothing
    python download.py --verify   check data/ against the manifest only (no network)
    python download.py            download, check, write the manifest

Dataset facts (id, version, counts, globs, folders) all come from config.yml.
"""
import argparse
import fnmatch
import hashlib
import importlib.metadata
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.config import load_config

LOGGER_NAME = "download"
ROOT = Path(__file__).resolve().parent
MANIFEST_SEP = "\t"
SUMMARY_PREFIX = "# "
# openneuro-py always fetches these top-level files whatever `exclude` says
# (installed 2026.9.1, openneuro/_download.py); mirrored here so the selection agrees.
OPENNEURO_ESSENTIAL_FILES = frozenset(
    {"dataset_description.json", "participants.tsv", "participants.json",
     "README", "README.md", "CHANGES", ".bidsignore"}
)


class DownloadError(RuntimeError):
    """Base class: every failure that must stop the run with a non-zero exit code."""


class DownloadValidationError(DownloadError):
    """The file tree does not match the counts in config.yml."""


class SnapshotUnavailable(DownloadError):
    """The private openneuro metadata function is missing or failed."""


class InsufficientSpaceError(DownloadError):
    """Free disk space is below the required multiple of the download size."""


class IntegrityError(DownloadError):
    """A local file differs from the remote listing or from the manifest."""


# ---------------------------------------------------------------- paths and matching
def _match(path, pattern):
    """Glob match where `*` never crosses `/` and the depth must be equal."""
    p_parts = path.split("/")
    g_parts = pattern.split("/")
    return len(p_parts) == len(g_parts) and all(
        fnmatch.fnmatchcase(a, b) for a, b in zip(p_parts, g_parts)
    )


def count_matching(rel_paths, pattern):
    return sum(1 for p in rel_paths if _match(p, pattern))


def list_local_files(data_dir, manifest_name=None):
    """Sorted POSIX relative paths of every file under data_dir, minus the manifest itself."""
    data_dir = Path(data_dir)
    out = []
    for dirpath, _dirs, files in os.walk(data_dir):
        for name in files:
            rel = (Path(dirpath) / name).relative_to(data_dir).as_posix()
            if rel != manifest_name:
                out.append(rel)
    return sorted(out)


def manifest_rel_name(cfg):
    """Manifest path relative to data_dir (paths.manifest_file lives under paths.data_dir)."""
    return Path(cfg["paths"]["manifest_file"]).relative_to(cfg["paths"]["data_dir"]).as_posix()


# ---------------------------------------------------------------- counts (§4.1)
def skipped_scans_count(cfg):
    return 1  # exactly one file is skipped: skip_scans_tsv names one subject and one session


def validate_counts(rel_paths, cfg):
    """Check subjects, EDFs, t2 subjects and scans.tsv counts; raise with every failure listed."""
    ds = cfg["dataset"]
    rel_paths = list(rel_paths)
    subjects = {p.split("/")[0] for p in rel_paths if "/" in p
                and fnmatch.fnmatchcase(p.split("/")[0], ds["subject_dir_glob"])}
    edfs = [p for p in rel_paths if _match(p, ds["eeg_glob"])]
    t1 = {p.split("/")[0] for p in edfs if p.split("/")[1] == ds["session_first"]}
    t2 = {p.split("/")[0] for p in edfs if p.split("/")[1] == ds["session_second"]}
    n_scans = count_matching(rel_paths, ds["scans_glob"])
    expected_scans = ds["n_recordings"] - skipped_scans_count(cfg)
    checks = [
        ("subjects", len(subjects), ds["n_subjects"]),
        ("EDF files", len(edfs), ds["n_recordings"]),
        (f"subjects with a {ds['session_first']} EDF", len(t1), ds["n_subjects"]),
        (f"subjects with a {ds['session_second']} EDF", len(t2), ds["n_subjects_with_t2"]),
        ("scans.tsv files (recordings minus the skipped one)", n_scans, expected_scans),
    ]
    bad = [f"{name}: found {found}, expected {want}" for name, found, want in checks if found != want]
    if bad:
        raise DownloadValidationError("Count check failed: " + "; ".join(bad))
    return {name: found for name, found, _ in checks}


def validate_tree(data_dir, cfg):
    return validate_counts(list_local_files(data_dir, manifest_rel_name(cfg)), cfg)


# ---------------------------------------------------------------- remote listing and selection
def openneuro_py_version():
    return importlib.metadata.version("openneuro-py")


def fetch_snapshot_files(cfg):
    """[(filename, size)] of the pinned snapshot. The ONLY use of openneuro's private API (IMP-007)."""
    ds = cfg["dataset"]
    try:
        from openneuro._download import _get_download_metadata
        snapshot = _get_download_metadata(
            dataset_id=ds["openneuro_id"], tag=ds["version"],
            max_retries=cfg["download"]["metadata_max_retries"],
        )
        return [(f.filename, int(f.size or 0)) for f in snapshot.files]
    except Exception as exc:  # ImportError, AttributeError, network, schema: all fatal here
        raise SnapshotUnavailable(
            "Cannot list the remote dataset: openneuro-py's private metadata function "
            f"(openneuro._download._get_download_metadata) is missing or failed: {exc!r}"
        ) from exc


def _glob_filter(names, patterns):
    try:
        from openneuro._glob import glob_filter
    except Exception as exc:
        raise SnapshotUnavailable(
            f"openneuro-py's private glob_filter is unavailable: {exc!r}") from exc
    matches = glob_filter(names, patterns)
    return {n for found in matches.values() for n in found}


def exclude_patterns(cfg):
    skip = cfg["dataset"]["skip_scans_tsv"]
    return list(cfg["dataset"]["download_exclude"]) + [
        f"{skip['subject']}/{skip['session']}/{skip['pattern']}"
    ]


@dataclass
class Selection:
    files: dict  # filename -> size, what the downloader will be asked for
    skipped: list = field(default_factory=list)  # the sub-010 scans.tsv
    excluded_by_config: dict = field(default_factory=dict)  # filename -> size (derivatives/)

    @property
    def total_bytes(self):
        return sum(self.files.values())


def select_files(snapshot_files, cfg):
    """Apply the include/exclude rules exactly as openneuro.download does."""
    ds = cfg["dataset"]
    sizes = dict(snapshot_files)
    names = list(sizes)
    included = _glob_filter(names, ds["download_include"])
    config_excluded = _glob_filter(names, ds["download_exclude"])
    skip = ds["skip_scans_tsv"]
    skip_set = _glob_filter(names, [f"{skip['subject']}/{skip['session']}/{skip['pattern']}"])
    keep = (included - config_excluded - skip_set) | (OPENNEURO_ESSENTIAL_FILES & set(names))
    return Selection(
        files={n: sizes[n] for n in sorted(keep)},
        skipped=sorted(skip_set & included),
        excluded_by_config={n: sizes[n] for n in sorted(config_excluded)},
    )


def check_free_space(total_bytes, free_bytes, factor):
    if free_bytes < factor * total_bytes:
        raise InsufficientSpaceError(
            f"Free space {free_bytes} B is under {factor}x the download size "
            f"({total_bytes} B, needs {factor * total_bytes} B)")


def _free_bytes(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _gb(n):
    return f"{n / 1e9:.3f} GB"


# ---------------------------------------------------------------- manifest
def sha256_file(path, chunk_bytes):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def build_entries(data_dir, rel_paths, chunk_bytes):
    """[(sha256, size, rel_path)] sorted by path."""
    data_dir = Path(data_dir)
    return [(sha256_file(data_dir / rel, chunk_bytes), (data_dir / rel).stat().st_size, rel)
            for rel in sorted(rel_paths)]


def _file_lines(entries):
    return [MANIFEST_SEP.join((sha, str(size), rel)) + "\n" for sha, size, rel in sorted(entries, key=lambda e: e[2])]


def content_sha256(entries):
    return hashlib.sha256("".join(_file_lines(entries)).encode("utf-8")).hexdigest()


def manifest_text(entries, cfg):
    ds = cfg["dataset"]
    summary = (f"{SUMMARY_PREFIX}files={len(entries)} manifest_sha256={content_sha256(entries)} "
               f"dataset={ds['openneuro_id']} version={ds['version']} "
               f"openneuro_py={openneuro_py_version()}\n")
    return "".join(_file_lines(entries)) + summary


def write_manifest(path, entries, cfg):
    Path(path).write_text(manifest_text(entries, cfg), encoding="utf-8", newline="\n")


def read_manifest(path):
    """(entries, summary dict). Raises IntegrityError if the manifest is malformed or its own hash is wrong."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or not lines[-1].startswith(SUMMARY_PREFIX):
        raise IntegrityError(f"{path}: no summary line")
    summary = dict(kv.split("=", 1) for kv in lines[-1][len(SUMMARY_PREFIX):].split())
    entries = []
    for line in lines[:-1]:
        parts = line.split(MANIFEST_SEP)
        if len(parts) != 3 or not parts[1].isdigit():
            raise IntegrityError(f"{path}: malformed line {line!r}")
        entries.append((parts[0], int(parts[1]), parts[2]))
    if len({e[2] for e in entries}) != len(entries):
        raise IntegrityError(f"{path}: a path is listed more than once")
    if summary.get("files") != str(len(entries)) or summary.get("manifest_sha256") != content_sha256(entries):
        raise IntegrityError(f"{path}: summary line does not match the file lines (manifest edited or damaged)")
    return entries, summary


@dataclass
class VerifyReport:
    missing: list = field(default_factory=list)
    mismatched: list = field(default_factory=list)  # (rel, reason)
    unexpected: list = field(default_factory=list)

    @property
    def ok(self):
        return not (self.missing or self.mismatched or self.unexpected)


def verify_manifest(data_dir, entries, cfg):
    """Compare data_dir with the manifest. Size first; hash only when the size matches. Never modifies files."""
    data_dir = Path(data_dir)
    report = VerifyReport()
    chunk = cfg["download"]["hash_chunk_bytes"]
    listed = {rel for _, _, rel in entries}
    for sha, size, rel in entries:
        path = data_dir / rel
        if not path.is_file():
            report.missing.append(rel)
            continue
        actual = path.stat().st_size
        if actual != size:
            report.mismatched.append((rel, f"size {actual} != manifest {size}"))
        elif sha256_file(path, chunk) != sha:
            report.mismatched.append((rel, "sha256 differs from manifest"))
    report.unexpected = sorted(set(list_local_files(data_dir, manifest_rel_name(cfg))) - listed)
    return report


def _describe(report):
    parts = []
    if report.missing:
        parts.append(f"{len(report.missing)} missing (e.g. {report.missing[0]})")
    if report.mismatched:
        parts.append(f"{len(report.mismatched)} differ (e.g. {report.mismatched[0][0]}: {report.mismatched[0][1]})")
    if report.unexpected:
        parts.append(f"{len(report.unexpected)} not in manifest (e.g. {report.unexpected[0]})")
    return "; ".join(parts)


def log_longest_path(logger, data_dir):
    files = list_local_files(data_dir)
    if not files:
        return
    longest = max(files, key=lambda r: len((Path(data_dir).resolve() / r).as_posix()))
    absolute = str(Path(data_dir).resolve() / longest)
    logger.info("Longest path in data/: %d characters: %s", len(absolute), absolute)


# ---------------------------------------------------------------- run modes
def _log_selection(logger, sel, counts):
    skip_note = ", ".join(sel.skipped) or "none"
    logger.info("Selected %d files, %s", len(sel.files), _gb(sel.total_bytes))
    logger.info("Skipped by §4.2 rule (EEG still downloaded): %s", skip_note)
    logger.info("Excluded by config (derivatives): %d files, %s", len(sel.excluded_by_config),
                _gb(sum(sel.excluded_by_config.values())))
    for name, found in counts.items():
        logger.info("  %s: %d", name, found)


def run_list(cfg, root, fetch, logger):
    sel = select_files(fetch(cfg), cfg)
    counts = validate_counts(sel.files, cfg)
    _log_selection(logger, sel, counts)
    free = _free_bytes(Path(root) / cfg["paths"]["data_dir"])
    logger.info("Free space on the data drive: %s (required: %sx the total = %s)",
                _gb(free), cfg["download"]["min_free_space_factor"],
                _gb(cfg["download"]["min_free_space_factor"] * sel.total_bytes))
    check_free_space(sel.total_bytes, free, cfg["download"]["min_free_space_factor"])
    return sel


def run_verify(cfg, root, logger):
    data_dir = Path(root) / cfg["paths"]["data_dir"]
    manifest_path = Path(root) / cfg["paths"]["manifest_file"]
    if not manifest_path.is_file():
        raise IntegrityError(f"No manifest at {manifest_path}; run download.py first")
    entries, summary = read_manifest(manifest_path)
    report = verify_manifest(data_dir, entries, cfg)
    if not report.ok:
        raise IntegrityError("Manifest verification failed: " + _describe(report))
    counts = validate_tree(data_dir, cfg)
    logger.info("Manifest verified: %s files, %s", summary["files"], summary["manifest_sha256"])
    for name, found in counts.items():
        logger.info("  %s: %d", name, found)
    log_longest_path(logger, data_dir)


def run_download(cfg, root, logger, downloader=None, fetch=fetch_snapshot_files):
    ds, dl = cfg["dataset"], cfg["download"]
    data_dir = Path(root) / cfg["paths"]["data_dir"]
    manifest_path = Path(root) / cfg["paths"]["manifest_file"]
    data_dir.mkdir(parents=True, exist_ok=True)

    old_entries = None
    if manifest_path.is_file():
        old_entries, _ = read_manifest(manifest_path)
        report = verify_manifest(data_dir, old_entries, cfg)
        if report.mismatched or report.unexpected:
            raise IntegrityError("Existing files differ from the manifest; nothing was downloaded or "
                                 "changed: " + _describe(report))
        if report.ok:
            validate_tree(data_dir, cfg)
            logger.info("Everything present and matching the manifest; nothing to download.")
            log_longest_path(logger, data_dir)
            return
        logger.info("Manifest present, %d file(s) missing; fetching them.", len(report.missing))

    sel = select_files(fetch(cfg), cfg)
    _log_selection(logger, sel, validate_counts(sel.files, cfg))
    free = _free_bytes(data_dir)
    logger.info("Total to fetch %s; free space %s", _gb(sel.total_bytes), _gb(free))
    check_free_space(sel.total_bytes, free, dl["min_free_space_factor"])

    if downloader is None:
        import openneuro
        downloader = openneuro.download
    downloader(
        dataset=ds["openneuro_id"], tag=ds["version"], target_dir=data_dir,
        include=list(ds["download_include"]), exclude=exclude_patterns(cfg),
        verify_hash=True, verify_size=True,
        max_retries=dl["max_retries"], max_concurrent_downloads=dl["max_concurrent_downloads"],
    )

    local = list_local_files(data_dir, manifest_rel_name(cfg))
    extra, absent = sorted(set(local) - set(sel.files)), sorted(set(sel.files) - set(local))
    if extra or absent:
        raise IntegrityError(f"Local files differ from the remote selection: {len(absent)} absent "
                             f"(e.g. {absent[:1]}), {len(extra)} unexpected (e.g. {extra[:1]})")
    wrong = [(r, (data_dir / r).stat().st_size, sel.files[r]) for r in local
             if (data_dir / r).stat().st_size != sel.files[r]]
    if wrong:
        raise IntegrityError(f"{len(wrong)} file(s) differ in size from the remote listing, no manifest "
                             f"written (e.g. {wrong[0][0]}: local {wrong[0][1]} B, remote {wrong[0][2]} B)")
    validate_counts(local, cfg)
    logger.info("Counts and sizes match the remote listing.")

    if old_entries is not None:
        report = verify_manifest(data_dir, old_entries, cfg)
        if not report.ok:
            raise IntegrityError("Downloaded files do not match the existing manifest; it was not "
                                 "rewritten: " + _describe(report))
        logger.info("Existing manifest confirmed (%d files).", len(old_entries))
    else:
        entries = build_entries(data_dir, local, dl["hash_chunk_bytes"])
        write_manifest(manifest_path, entries, cfg)
        logger.info("Manifest written: %s (%d files)", manifest_path, len(entries))
    log_longest_path(logger, data_dir)


# ---------------------------------------------------------------- entry point
def setup_logging(cfg, root):
    log_file = Path(root) / cfg["paths"]["download_log"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download ds003775 and record a SHA-256 manifest.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="show what would be fetched; download nothing")
    group.add_argument("--verify", action="store_true", help="check data/ against the manifest only")
    return parser.parse_args(argv)


def main(argv=None, root=None, downloader=None, fetch=fetch_snapshot_files):
    args = parse_args(argv)
    root = Path(root) if root is not None else ROOT
    cfg = load_config(root / "config.yml")
    logger = setup_logging(cfg, root)
    try:
        if args.list:
            run_list(cfg, root, fetch, logger)
        elif args.verify:
            run_verify(cfg, root, logger)
        else:
            run_download(cfg, root, logger, downloader=downloader, fetch=fetch)
    except DownloadError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
