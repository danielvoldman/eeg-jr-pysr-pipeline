"""The preregistered Q/R tuning rule (PLAN.md C4; §7.5, §7.6, §9.4; IMP-037 to IMP-043).

R is fixed at 0.25 sigma_ref^2 I2 and Q = q * diag(steady-state variance) is a single scale q. q is chosen
on an 8-value log grid (1e-4 to 1e-1) as the value whose mean normalized innovation squared (NIS) is
closest to 2, the observation dimension, using the base-coupling 19-D M2 filter without any PySR term.

One core, two modes. tune_q(recordings, cfg) takes plain arrays (segments of shape (2, n), starts) and is
used both by the real-data driver run_real (20 random TRAINING subjects per split seed, ses-t1) and, later,
by Stage E for G0's own separate tuning set of 20 synthetic series (never used for pass/fail).

Definitions (IMP-037 to IMP-039): NIS_t = innovation' S^-1 innovation (S without Q, IMP-019). Samples: every
non-diverged clean segment, minus the first windows.training_burn_in_s of each segment and the first
passes.estimator_burn_in_s of cumulative processed samples (the same mask as the filtered-gain estimator).
Per recording the mean over its samples, then the unweighted mean over recordings. "Closest to 2" is the
absolute difference; an exact tie goes to the smaller q. A recording that diverged at ANY grid value (or has
no NIS samples) is left out at ALL grid values, so every q is averaged over the same recordings.

This module does not import preprocess.py at module level (the real-data loader does, lazily), never
touches test subjects, and never calls PySR.
"""
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import passes

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_CODE_FILES = ("model.py", "state_space.py", "ukf.py", "passes.py", "tuning.py", "ukf_ext.py")
_CONFIG_SECTIONS = ("rescaling", "jansen_rit", "simulator", "coupling", "observation", "state", "priors",
                    "ukf", "passes", "windows")


class TuningError(ValueError):
    """Raised for an unusable tuning request."""


class GateError(TuningError):
    """Real-data tuning was asked for without an outputs/gate.json that permits it (CLAUDE.md rule 6)."""


# ---- grid ---------------------------------------------------------------------------------------------

def q_grid(cfg):
    """The 8-value log grid of §7.6, ascending, read from config."""
    pn = cfg["ukf"]["process_noise"]
    return np.logspace(np.log10(pn["q_grid_low"]), np.log10(pn["q_grid_high"]), int(pn["q_grid_n"]))


# ---- one (recording, q) filter run -----------------------------------------------------------------------

def recording_nis(segments, starts, cfg, q, filter_name=None):
    """Forward-only pass-1 run of one recording at scale q (IMP-040) through the chosen filter (explicit, else
    g0.filter, F0). Returns a JSON-able dict with the per-recording mean NIS over the kept samples (None if there
    are none) and the divergence facts."""
    res = passes.run_pass1(segments, starts, cfg, q, forward_only=True, filter_name=filter_name)
    vals = [s.nis[s.nis_keep] for s in res.segments if not s.diverged and s.nis_keep is not None]
    kept = np.concatenate(vals) if vals else np.empty(0)
    return {"mean_nis": float(np.mean(kept)) if kept.size else None, "n_samples": int(kept.size),
            "recording_diverged": bool(res.recording_diverged), "n_clean": int(res.n_clean),
            "n_diverged_samples": int(res.n_diverged),
            "n_segments": len(res.segments), "n_segments_diverged": sum(s.diverged for s in res.segments)}


def _nis_worker(payload):
    """Top-level so that the Windows spawn start method can import it; one BLAS thread per worker."""
    from threadpoolctl import threadpool_limits
    segments, starts, cfg, q, filter_name = payload
    with threadpool_limits(limits=1):
        return recording_nis(segments, starts, cfg, q, filter_name)


# ---- cache -------------------------------------------------------------------------------------------------

def code_hash():
    """SHA-256 over the sources of every module the filter run depends on (IMP-041)."""
    h = hashlib.sha256()
    for name in _CODE_FILES:
        h.update((Path(__file__).resolve().parent / name).read_bytes())
    return h.hexdigest()


def filter_fingerprint(cfg):
    """The config leaves that affect a (recording, q) run: qr_rule and the rest of ukf, windows, passes,
    coupling, observation, state, priors, the simulator and Jansen-Rit constants, the rescaling constants
    and the observation rate (IMP-041)."""
    used = {k: cfg[k] for k in _CONFIG_SECTIONS}
    used["observation_fs_hz"] = cfg["preprocessing"]["observation_fs_hz"]
    return json.dumps(used, sort_keys=True, default=str)


def arrays_key(segments, starts):
    """Cache identity of a recording given only as arrays (synthetic series, G0 mode)."""
    h = hashlib.sha256()
    for seg, st in zip(segments, starts):
        a = np.ascontiguousarray(seg, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
        h.update(str(int(st)).encode())
    return "arrays:" + h.hexdigest()


def cache_key(rec_key, q, cfg, filter_name=None):
    payload = "\n".join(["qr", rec_key, repr(float(q)), str(filter_name), code_hash(), filter_fingerprint(cfg)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir, key):
    return Path(cache_dir) / f"{key}.json"


def _cache_read(cache_dir, key):
    if cache_dir is None:
        return None
    p = _cache_path(cache_dir, key)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _cache_write(cache_dir, key, entry):
    if cache_dir is None:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(p)


# ---- selection (pure) --------------------------------------------------------------------------------------

@dataclass
class QRResult:
    q: object                          # the selected q (None if the rule refused to select)
    q_index: object
    refused: bool
    refusal_reason: object
    grid: list
    target: float
    band: float
    table: list                        # per q: mean NIS over the matched recordings, divergence counts
    recordings: list                   # per recording: id, key, per-q means, matched flag, reason
    n_recordings: int
    n_matched: int
    in_band: bool = False
    at_grid_edge: bool = False
    nis_spread: object = None
    low_confidence_qr: bool = False
    tie_broken: bool = False
    q_flagged_divergence: list = field(default_factory=list)

    def to_dict(self):
        return dict(self.__dict__)


def select_q(entries, ids, keys, grid, cfg, min_recordings=None):
    """The rule, on precomputed entries[i][j] = recording_nis(...) dicts (recording i, grid value j)."""
    qr = cfg["ukf"]["qr_rule"]
    target, band = float(qr["nis_target"]), float(qr["nis_band"])
    min_rec = int(qr["min_recordings"] if min_recordings is None else min_recordings)
    flag_frac = float(qr["max_diverged_fraction_flag"])
    n_rec, n_q = len(entries), len(grid)

    def bad(e):
        return e["recording_diverged"] or e["mean_nis"] is None

    matched = [not any(bad(e) for e in row) for row in entries]
    n_matched = sum(matched)
    table = []
    for j in range(n_q):
        col = [entries[i][j] for i in range(n_rec)]
        n_div = sum(e["recording_diverged"] for e in col)
        ok = [e for e, m in zip(col, matched) if m]
        means = [e["mean_nis"] for e in ok]
        pooled_n = sum(e["n_samples"] for e in ok)
        pooled = (sum(e["mean_nis"] * e["n_samples"] for e in ok) / pooled_n) if pooled_n else None
        table.append({"q": float(grid[j]), "mean_nis": float(np.mean(means)) if means else None,
                      "pooled_mean_nis": pooled, "n_diverged": int(n_div),
                      "fraction_diverged": (n_div / n_rec) if n_rec else 0.0})
    flagged = [t["q"] for t in table if t["fraction_diverged"] > flag_frac]
    for qv in flagged:
        log.warning("q = %.3g: more than %.0f%% of the recordings diverged", qv, 100 * flag_frac)
    recs = []
    for i in range(n_rec):
        reason = None
        if not matched[i]:
            reason = "diverged_or_no_samples_at_some_q"
        recs.append({"id": ids[i], "key": keys[i], "matched": bool(matched[i]), "reason": reason,
                     "mean_nis": [e["mean_nis"] for e in entries[i]],
                     "n_samples": [e["n_samples"] for e in entries[i]],
                     "recording_diverged": [bool(e["recording_diverged"]) for e in entries[i]]})
    out = QRResult(q=None, q_index=None, refused=False, refusal_reason=None, grid=[float(v) for v in grid],
                   target=target, band=band, table=table, recordings=recs, n_recordings=n_rec,
                   n_matched=n_matched, q_flagged_divergence=flagged)
    if n_matched < min_rec:
        out.refused = True
        out.refusal_reason = f"only {n_matched} of {n_rec} recordings usable at every q; minimum {min_rec}"
        log.error("Q/R tuning refuses to select: %s", out.refusal_reason)
        return out
    dev = np.array([abs(t["mean_nis"] - target) for t in table])
    best = float(dev.min())
    j = int(np.argmin(dev))                 # first minimum: the smaller q on an exact tie (grid ascending)
    out.tie_broken = int(np.count_nonzero(dev == best)) > 1
    out.q, out.q_index = float(grid[j]), j
    out.in_band = best <= band
    out.at_grid_edge = j in (0, n_q - 1)
    means = [t["mean_nis"] for t in table]
    out.nis_spread = float(max(means) - min(means))
    out.low_confidence_qr = not out.in_band
    if out.low_confidence_qr:
        log.warning("no grid value has a mean NIS within %.2f of %.1f (closest: q = %.3g, NIS %.3f); "
                    "the closest is still selected, low_confidence_qr is set", band, target, out.q, means[j])
    if out.at_grid_edge:
        log.warning("the selected q = %.3g is at the edge of the grid; the optimum may lie outside it", out.q)
    return out


# ---- the core: run the grid and select -----------------------------------------------------------------------

def tune_q(recordings, cfg, cache_dir=None, n_jobs=1, min_recordings=None, filter_name=None):
    """Tune q on a list of recordings. Each is a dict {"id", "segments", "starts"[, "key"]} or a tuple
    (id, segments, starts). Used unchanged for the 20 real training recordings of a split seed and for the
    20-series synthetic G0 tuning set (Stage E). With a cache_dir, every (recording, q) result is stored and
    reused (the result does not depend on the split seed). n_jobs > 1 runs joblib/loky workers. filter_name:
    explicit, else g0.filter (F0); it is part of every cache key."""
    filter_name = passes.resolve_filter(cfg, filter_name)
    recs = []
    for r in recordings:
        d = dict(zip(("id", "segments", "starts"), r)) if isinstance(r, (tuple, list)) else dict(r)
        d.setdefault("key", arrays_key(d["segments"], d["starts"]))
        recs.append(d)
    if not recs:
        raise TuningError("no recordings to tune on")
    grid = q_grid(cfg)
    entries = [[None] * len(grid) for _ in recs]
    todo = []
    for i, r in enumerate(recs):
        for j, qv in enumerate(grid):
            ck = cache_key(r["key"], qv, cfg, filter_name)
            hit = _cache_read(cache_dir, ck)
            if hit is not None:
                entries[i][j] = hit
            else:
                todo.append((i, j, ck))
    log.info("Q/R tuning: %d recordings x %d q = %d runs, %d cached, %d to run (n_jobs=%d)",
             len(recs), len(grid), len(recs) * len(grid), len(recs) * len(grid) - len(todo), len(todo), n_jobs)
    payloads = [(recs[i]["segments"], recs[i]["starts"], cfg, float(grid[j]), filter_name) for i, j, _ in todo]
    if n_jobs > 1 and len(payloads) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend=cfg["compute"]["joblib_backend"])(
            delayed(_nis_worker)(p) for p in payloads)
    else:
        results = [_nis_worker(p) for p in payloads]
    for (i, j, ck), entry in zip(todo, results):
        entries[i][j] = entry
        _cache_write(cache_dir, ck, entry)
    return select_q(entries, [r["id"] for r in recs], [r["key"] for r in recs], grid, cfg, min_recordings)


# ---- real-data mode: draw, gate, output ---------------------------------------------------------------------------

def draw_for_split(split, seed, cfg):
    """The random order in which training subjects are considered for split seed `seed` (IMP-042). Reads
    ONLY split["train"]: the test list is never touched. The first n_tuning_subjects eligible subjects in
    this order are used. Draw seed = ukf.qr_rule.draw_seed_offset + split seed (default_rng)."""
    train = sorted(split["train"])
    rng = np.random.default_rng(int(cfg["ukf"]["qr_rule"]["draw_seed_offset"]) + int(seed))
    return [train[i] for i in rng.permutation(len(train))]


def collect_recordings(order, n, loader):
    """Walk `order` and keep the first n subjects whose ses-t1 recording the loader accepts. The loader
    returns a dict with "reason" None (usable) or a string, plus "segments", "starts" and "key"."""
    chosen, skipped = [], []
    for subject in order:
        if len(chosen) == n:
            break
        r = loader(subject)
        if r["reason"] is None:
            chosen.append({"id": subject, "segments": r["segments"], "starts": r["starts"], "key": r["key"]})
        else:
            skipped.append({"subject": subject, "reason": r["reason"]})
            log.warning("Q/R tuning: %s not used: %s", subject, r["reason"])
    return chosen, skipped


def check_gate(cfg, root):
    """Real-data fitting needs outputs/gate.json (CLAUDE.md rule 6). Its schema belongs to Stage E; until
    then: the file must exist, parse, and not carry hard_stop = true (IMP-043). Returns the parsed gate."""
    path = Path(root) / cfg["paths"]["gate_file"]
    if not path.is_file():
        raise GateError(f"{path} does not exist: real-data Q/R tuning is not permitted before G0 (rule 6)")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(gate, dict) and gate.get("hard_stop") is True:
        raise GateError(f"{path} records a hard stop: real-data Q/R tuning is not permitted")
    return gate


def make_real_loader(cfg, root, pilot_ids, allow_all):
    """Loader for collect_recordings: the ses-t1 recording of a subject through B4/B5 (cached) and the B6
    decision. Imports preprocess lazily."""
    from src import preprocess as pp
    root = Path(root)
    manifest = pp.load_manifest(root / cfg["paths"]["manifest_file"])
    cache_root = root / cfg["paths"]["cache_dir"]
    data_root = root / cfg["paths"]["data_dir"]
    session = cfg["dataset"]["session_first"]

    def load(subject):
        pattern = cfg["dataset"]["eeg_glob"].replace("sub-*", subject, 1)
        files = [f for f in sorted(data_root.glob(pattern)) if session in f.parts]
        if len(files) != 1:
            return {"reason": f"{len(files)} {session} EDF files found"}
        res = pp.segment_recording(cfg, files[0], data_root, manifest, pilot_ids, allow_all=allow_all,
                                   cache_root=cache_root)
        decision = pp.recording_decision(cfg, res.meta, res.meta)
        if decision["status"] != "kept":
            return {"reason": f"excluded: {decision['reason']}"}
        key = "b5:" + pp.cache_key_b5(cfg, res.meta["sha256"], False, False)
        return {"reason": None, "segments": res.segments, "starts": res.starts, "key": key}

    return load


def _git_state(root):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def qr_path(cfg, root, seed, pilot):
    base = cfg["paths"]["pilot_results_dir"] if pilot else cfg["paths"]["outputs_dir"]
    return Path(root) / base / cfg["ukf"]["qr_rule"]["output_name_pattern"].format(seed=seed)


def build_document(result, cfg, root, seed, order, skipped, pilot, gate, filter_name=None):
    head, dirty = _git_state(root)
    filter_name = passes.resolve_filter(cfg, filter_name)
    adopted = filter_name != "19D"           # DEV-005: q is declared, the section 7.5 rule is only reported
    cfg_file = Path(root) / "config.yml"
    body = {"schema": 1, "split_seed": int(seed), "pilot": bool(pilot), "result": result.to_dict(),
            "subjects": [r["id"] for r in result.recordings], "skipped_subjects": skipped,
            "draw": {"order_head": order[:len(result.recordings) + len(skipped)],
                     "seed": int(cfg["ukf"]["qr_rule"]["draw_seed_offset"]) + int(seed),
                     "session": cfg["dataset"]["session_first"], "train_only": True},
            "gate_low_confidence": bool(gate.get("low_confidence", False)) if isinstance(gate, dict) else False,
            "filter": filter_name,
            "rule_role": "reported_not_used" if adopted else "selects_q",
            "q_used": passes.resolve_q(cfg, None, filter_name) if adopted else None}
    prov = {"git_commit": head, "git_dirty": dirty, "code_sha256": code_hash(),
            "config_yml_sha256": hashlib.sha256(cfg_file.read_bytes()).hexdigest() if cfg_file.is_file() else None,
            "filter_fingerprint_sha256": hashlib.sha256(filter_fingerprint(cfg).encode()).hexdigest()}
    return dict(body, provenance=prov)


def write_qr(path, doc, force=False):
    """Write once. A rerun that would change the recorded result (everything but the provenance block) is
    refused unless force; an identical result is left untouched. Returns "written", "unchanged"."""
    path = Path(path)
    if path.is_file() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        strip = lambda d: {k: v for k, v in d.items() if k != "provenance"}    # noqa: E731
        if strip(old) == strip(json.loads(json.dumps(doc))):
            return "unchanged"
        raise TuningError(f"{path} exists with a different result; pass force to overwrite (a deviation)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return "written"


def run_real(cfg, root, seed, pilot=False, n_jobs=None, force=False, loader=None, pilot_ids=None, filter_name=None):
    """Real-data NIS rule for one split seed: gate check (skipped in pilot mode), the training-only draw,
    the tuning, and outputs/qr_<seed>.json (pilot: results/pilot/). Not wired into main.py.

    For the adopted filter (A, DEV-005, DEV-006) the rule's q is REPORTED ONLY: downstream runs use
    ukf.process_noise.q_fixed (passes.resolve_q), never this file; the document says so (rule_role, q_used)."""
    import main as main_mod
    root = Path(root)
    filter_name = passes.resolve_filter(cfg, filter_name)
    if filter_name != "19D":
        log.warning("Q/R tuning with filter %s: the NIS rule is run for the report only; q used downstream is "
                    "q_fixed = %s (DEV-005)", filter_name, passes.resolve_q(cfg, None, filter_name))
    gate = {} if pilot else check_gate(cfg, root)
    train = main_mod.training_subjects(seed, root, cfg)
    if pilot:
        if pilot_ids is None:
            from src import preprocess as pp
            pilot_ids = pp.load_pilot_ids(cfg, root)
        train = [s for s in train if s in set(pilot_ids)]
    order = draw_for_split({"train": train}, seed, cfg)
    n_want = int(cfg["ukf"]["qr_rule"]["n_tuning_subjects"])
    if loader is None:
        loader = make_real_loader(cfg, root, pilot_ids if pilot_ids is not None else [], allow_all=not pilot)
    chosen, skipped = collect_recordings(order, n_want, loader)
    min_rec = min(int(cfg["ukf"]["qr_rule"]["min_recordings"]), len(chosen)) if pilot else None
    jobs = int(cfg["compute"]["joblib_n_jobs"]) if n_jobs is None else int(n_jobs)
    cache_dir = root / cfg["paths"]["cache_dir"] / cfg["ukf"]["qr_rule"]["cache_subdir"]
    result = tune_q(chosen, cfg, cache_dir=cache_dir, n_jobs=jobs, min_recordings=min_rec, filter_name=filter_name)
    doc = build_document(result, cfg, root, seed, order, skipped, pilot, gate, filter_name=filter_name)
    status = write_qr(qr_path(cfg, root, seed, pilot), doc, force=force)
    log.info("Q/R tuning, split seed %s: q = %s (%s)", seed, result.q, status)
    return result
