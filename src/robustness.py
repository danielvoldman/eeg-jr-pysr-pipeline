"""C1 scoring: ablation M0 to M3, matched sets, subject bootstrap, Holm (PLAN G1; §10.2, §11.4, §12, §13.1, §15; IMP-078).

M1, M2 and M3 are scored from ONE continuous forward pass-1 filter per recording (filter A, q_fixed): the one-step squared
error, averaged over the two channels, on the samples of passes.scoring_mask (the same mask baseline.py used for M0 and
M0b), restricted to segments no variant diverged in. Per-subject score = mean over the subject's scored samples; headline =
mean over subjects. M3 is the frozen PySR residual added to dy4/dt (G0.5): the filter that carries it is the NumPy
extended filter of ukf_resid; M1 and M2 keep their Numba kernels. Nothing here calls PySR, refits anything, or reads a
recording that is not in the SubjectPlan.

Guard (rule 6, "pilot subjects only until the full run"), four layers:
  1. resolve_subjects is the only source of IDs. Pilot mode: the pilot subjects, checked to be on the training side, then
     only the internal-test side of the pilot's mechanics split; the test list is never loaded. Confirmatory mode needs
     confirmatory=True, a gate that permits it, a frozen-equation file and the baseline scores of exactly these subjects.
  2. The loader is built with allow_all = (mode is confirmatory), so preprocess.assert_dev_subject refuses any non-pilot
     recording in a development run, and guarded_loader refuses any ID outside the plan before the loader is called.
  3. The baseline scores (outputs of baseline.run_baseline) must cover exactly the planned subjects, and their stored mask
     must equal passes.scoring_mask of the recording: the "same samples" guarantee is checked, not assumed.
  4. Pilot output goes to results/pilot/ and is stamped mechanics_only.

Bootstrap (§11.4): subjects (or clusters of them) are resampled from a pre-drawn index matrix; the per-subject scores of
the already-run frozen model are re-averaged, so nothing is ever refitted or re-filtered. C1 uses ses-t1 only, so the
cluster argument defaults to one cluster per subject (C3 passes session clusters later).

G2 (IMP-080 to IMP-082) adds, in the same file (§18): the Holm family registry (11 members), the C3 ICC(3,1) per
direction with F-based and cluster-bootstrap CIs and the vigilance-adjusted sensitivity, the wPLI / imaginary coherency
diagnostic and the AAFT surrogate test. Same guard: pilot subjects only until the full run; the C3 plan needs both
sessions and (confirmatory) both partitions, so it has its own resolver.
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import baseline, passes
from src import state_space as ss

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("M1", "M2", "M3")


class RobustnessError(ValueError):
    """Raised for an unusable C1 scoring request."""


class GuardError(RobustnessError):
    """Raised when a request would read a subject or file the run is not allowed to use."""


# ---- guard ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectPlan:
    mode: str                  # "pilot" or "confirmatory"
    ids: tuple                 # the subjects scored, sorted
    pilot_ids: frozenset       # for the loader's development guard
    allow_all: bool
    partition: object = None   # {id: "train" | "test"} (C3 plans only)


def resolve_subjects(cfg, root, seed, *, pilot, confirmatory=False, pilot_ids=None):
    """The subjects of a C1 run. See the module docstring (guard layer 1)."""
    import main as main_mod
    from src import tuning
    root = Path(root)
    if pilot:
        if confirmatory:
            raise GuardError("confirmatory scoring is not a pilot run")
        if pilot_ids is None:
            from src import preprocess as pp
            pilot_ids = pp.load_pilot_ids(cfg, root)
        pilot_ids = frozenset(pilot_ids)
        train = set(main_mod.training_subjects(seed, root, cfg))
        if not pilot_ids <= train:
            raise GuardError("a pilot subject is not on the training side of the split")
        ids = tuple(baseline.pilot_internal_split(pilot_ids, cfg)[1])
        if not set(ids) <= train:
            raise GuardError("a planned subject is not on the training side")
        return SubjectPlan("pilot", ids, pilot_ids, allow_all=False)
    if confirmatory is not True:
        raise GuardError("a non-pilot run must pass confirmatory=True explicitly")
    tuning.check_gate(cfg, root)
    from src import regression
    fpath = regression.frozen_equation_path(cfg, root, seed, False)
    if not fpath.is_file():
        raise GuardError(f"{fpath} does not exist: confirmatory scoring needs the frozen equation")
    ids = tuple(sorted(main_mod.test_subjects(seed, root, cfg)))
    return SubjectPlan("confirmatory", ids, frozenset(), allow_all=True)


def guarded_loader(loader, plan):
    """Layer 2: refuse an ID outside the plan before the underlying loader is called."""
    allowed = set(plan.ids)

    def load(subject, *args, **kwargs):
        if subject not in allowed:
            raise GuardError(f"{subject} is not one of the {len(allowed)} planned subjects; refusing to read it")
        return loader(subject, *args, **kwargs)

    return load


def load_baseline(cfg, root, seed, plan):
    """Layer 3: the M0/M0b scores of exactly the planned subjects (no recording is read here)."""
    pilot = plan.mode == "pilot"
    jpath, npath = baseline.output_paths(cfg, root, seed, pilot)
    if not jpath.is_file() or not npath.is_file():
        raise GuardError(f"{jpath} / {npath} not found: run baseline.run_baseline first")
    doc = json.loads(jpath.read_text(encoding="utf-8"))
    if bool(doc.get("pilot")) != pilot or int(doc["split_seed"]) != int(seed):
        raise GuardError("the baseline file belongs to a different mode or split seed")
    scores = baseline.load_scores(npath)
    skipped = {s["subject"] for s in doc.get("skipped_subjects", [])}
    if set(scores) | skipped != set(plan.ids) or set(scores) & skipped:
        raise GuardError("the baseline scores do not cover exactly the planned subjects")
    return scores, doc, jpath, npath


# ---- forward-filter scoring -----------------------------------------------------------------------------

def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _code_sha():
    names = ("robustness.py", "passes.py", "ukf_resid.py", "ukf_ext.py", "ukf_numba.py", "ukf.py", "state_space.py",
             "model.py", "regression.py")
    return hashlib.sha256(b"".join((REPO_ROOT / "src" / n).read_bytes() for n in names)).hexdigest()


def _variant_result(p1):
    segs = p1.segments
    return {"sq_err": passes.filter_sq_errors(p1), "seg_diverged": [bool(s.diverged) for s in segs],
            "seg_reason": [s.reason for s in segs], "seg_n": [int(s.n) for s in segs],
            "recording_diverged": bool(p1.recording_diverged), "n_clean": int(p1.n_clean), "n_diverged": int(p1.n_diverged)}


def score_recording(rec, cfg, residual=None, filter_name=None, q=None, cache_dir=None, equation_sha=None):
    """{variant: result} for M1 and M2 (Numba filter A) and, with a residual, M3 (NumPy filter), each a continuous
    forward pass 1 over the clean segments. rec: dict(id, segments, starts, key?). A cache entry is used only when the
    recording has a key (the preprocessing cache key) and everything the result depends on is unchanged."""
    segs, starts = rec["segments"], rec["starts"]
    fname = passes.resolve_filter(cfg, filter_name)
    qv = passes.resolve_q(cfg, q, fname)
    spec = passes.make_spec(cfg, fname, segs)
    layouts = {"M1": ss.make_layout(cfg, include_gains=False), "M2": ss.make_layout(cfg)}
    out = {}
    wanted = ["M1", "M2"] + (["M3"] if residual is not None else [])
    base_key = None
    if cache_dir is not None and rec.get("key") is not None:
        base_key = {"rec": rec["key"], "filter": fname, "q": qv, "cfg": _digest(cfg), "code": _code_sha(),
                    "starts": [int(s) for s in starts]}
    for v in wanted:
        path = None
        if base_key is not None:
            key = _digest(dict(base_key, variant=v, eq=equation_sha if v == "M3" else None))
            path = Path(cache_dir) / f"{rec['id']}_{v}_{key}.npz"
            if path.is_file():
                out[v] = _cache_read(path)
                continue
        p1 = passes.run_pass1(segs, starts, cfg, q=qv, layout=layouts.get(v, layouts["M2"]), forward_only=True,
                              filter_name=fname, spec=spec, residual=residual if v == "M3" else None)
        out[v] = _variant_result(p1)
        if path is not None:
            _cache_write(path, out[v])
    return out


def _cache_write(path, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"err{k}": e for k, e in enumerate(res["sq_err"]) if e is not None}
    meta = {k: v for k, v in res.items() if k != "sq_err"}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)


def _cache_read(path):
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        meta["sq_err"] = [z[f"err{k}"] if f"err{k}" in z.files else None for k in range(len(meta["seg_diverged"]))]
    return meta


# ---- per-subject scores, matched sets, M3 loss ------------------------------------------------------------

def _split(arr, lengths):
    out, i = [], 0
    for n in lengths:
        out.append(np.asarray(arr[i:i + int(n)], dtype=np.float64))
        i += int(n)
    if i != len(arr):
        raise RobustnessError(f"{len(arr)} samples for segment lengths summing to {i}")
    return out


def _score(errs, keep):
    errs = [np.full(k.shape, np.nan) if e is None else e for e, k in zip(errs, keep)]
    return passes.subject_score(errs, keep)


def subject_scores(sid, cfg, base, runs, m3_mode):
    """One subject's row. base: baseline.load_scores entry; runs: score_recording output; m3_mode 'scored', 'no_term'
    (M3 is M2 exactly, no re-filtering) or 'absent' (no M3).

    status 'matched': no variant of M1..M3 diverged at recording level; the scored samples are scoring_mask restricted to
    segments no variant diverged in. status 'm3_loss': M3 diverged at recording level but M2 did not (§12); scored on the
    samples valid for M0, M1 (when it did not diverge) and M2, with M3's error := the worst of M0, M1, M2. Any other
    status is an exclusion with its reason; nothing is dropped silently."""
    lengths = [int(n) for n in base["lengths"]]
    masks = passes.scoring_mask(lengths, cfg)
    if not np.array_equal(np.concatenate(masks), base["mask"]):
        raise GuardError(f"{sid}: the baseline mask differs from passes.scoring_mask of the recording")
    r = dict(runs)
    if m3_mode == "no_term":
        r["M3"] = r["M2"]
    present = [v for v in VARIANTS if v in r]
    for v in present:
        if [int(n) for n in r[v]["seg_n"]] != lengths:
            raise GuardError(f"{sid}: {v} was run on segments of different lengths than the baseline")
    rec_div = {v: bool(r[v]["recording_diverged"]) for v in present}
    row = {"id": sid, "recording_diverged": rec_div, "status": None, "reason": None}
    m0, m0b = _split(base["M0"], lengths), _split(base["M0b"], lengths)

    def keep_for(variants):
        seg_div = np.zeros(len(lengths), dtype=bool)
        for v in variants:
            seg_div |= np.asarray(r[v]["seg_diverged"], dtype=bool)
        return [m & ~d for m, d in zip(masks, [np.full(n, bool(x)) for n, x in zip(lengths, seg_div)])], int(seg_div.sum())

    m3_only = "M3" in r and rec_div["M3"] and not rec_div["M2"] and m3_mode == "scored"
    if not any(rec_div.values()):
        keep, n_seg_div = keep_for(present)
        if not any(k.any() for k in keep):
            row.update(status="excluded", reason="no_scored_samples")
            return row
        row["scores"] = {"M0": _score(m0, keep), "M0b": _score(m0b, keep)}
        for v in present:
            row["scores"][v] = _score(r[v]["sq_err"], keep)
        row.update(status="matched", n_scored=int(sum(int(k.sum()) for k in keep)),
                   n_mask=int(sum(int(m.sum()) for m in masks)), n_segments_dropped=n_seg_div)
        return row
    if m3_only:
        avail = [v for v in ("M1", "M2") if not rec_div[v]]
        keep, n_seg_div = keep_for(avail)
        if not any(k.any() for k in keep):
            row.update(status="excluded", reason="m3_loss_no_scored_samples")
            return row
        sc = {"M0": _score(m0, keep), "M0b": _score(m0b, keep)}
        for v in avail:
            sc[v] = _score(r[v]["sq_err"], keep)
        sc["M3"] = max(sc["M0"], *(sc[v] for v in avail))
        row.update(status="m3_loss", scores=sc, n_scored=int(sum(int(k.sum()) for k in keep)),
                   n_mask=int(sum(int(m.sum()) for m in masks)), n_segments_dropped=n_seg_div)
        return row
    row.update(status="excluded", reason="recording_diverged_in_" + "_".join(v for v in present if rec_div[v]))
    return row


def divergence_report(rows, runs_by_subject, m3_mode, cfg):
    """Divergence counts per variant (§12) over the scored test recordings, and the removals of each matched-set rule."""
    present = [v for v in VARIANTS if any(v in r for r in runs_by_subject.values())]
    n = len(rows)
    per = {}
    for v in present + ["M0", "M0b"]:
        if v in ("M0", "M0b"):
            per[v] = {"recordings_diverged": 0, "segments_diverged": 0, "segments": None, "samples_diverged": 0,
                      "note": "a VAR cannot diverge"}
            continue
        runs = [r[v] if v in r else r["M2"] for r in runs_by_subject.values()]
        per[v] = {"recordings_diverged": int(sum(x["recording_diverged"] for x in runs)),
                  "segments_diverged": int(sum(sum(x["seg_diverged"]) for x in runs)),
                  "segments": int(sum(len(x["seg_diverged"]) for x in runs)),
                  "samples_diverged": int(sum(x["n_diverged"] for x in runs)),
                  "samples": int(sum(x["n_clean"] for x in runs))}
    if m3_mode == "no_term":
        per["M3"] = dict(per["M2"], note="identical to M2 (no residual term, no re-filtering)")
    m3_only = [r["id"] for r in rows if r["status"] == "m3_loss"]
    limit = float(cfg["statistics"]["c1"]["m3_only_divergence_limit"])
    return {"per_variant": per, "n_recordings": n, "m3_mode": m3_mode,
            "m3_only_divergence": {"n": len(m3_only), "subjects": m3_only, "fraction": (len(m3_only) / n) if n else None,
                                   "limit": limit, "exceeds_limit_state_in_limitations":
                                   bool(n and passes.exceeds_fraction(len(m3_only), n, limit))},
            "removed": {"recording_level_divergence": int(sum(r["reason"] is not None and r["reason"].startswith("recording_diverged")
                                                              for r in rows)),
                        "no_scored_samples": int(sum(r["reason"] in ("no_scored_samples", "m3_loss_no_scored_samples") for r in rows)),
                        "matched": int(sum(r["status"] == "matched" for r in rows)),
                        "m3_loss_kept_in_sensitivity": len(m3_only),
                        "segment_level": {"subjects_with_dropped_segments": int(sum(r.get("n_segments_dropped", 0) > 0
                                                                                    for r in rows if r["status"] == "matched")),
                                          "mask_samples_removed": int(sum(r["n_mask"] - r["n_scored"] for r in rows
                                                                          if r["status"] == "matched"))}}}


# ---- bootstrap, p-values, Holm ----------------------------------------------------------------------------

def draw_index_matrix(n_clusters, n_boot, seed, split_seed):
    """The pre-drawn (B, n_clusters) matrix of cluster indices, default_rng([seed, split_seed, n_clusters]); one matrix
    per (split seed, n), shared by every comparison of an analysis so the differences are paired."""
    if n_clusters < 1 or n_boot < 1:
        raise RobustnessError("the bootstrap needs at least one cluster and one resample")
    rng = np.random.default_rng([int(seed), int(split_seed), int(n_clusters)])
    return rng.integers(0, int(n_clusters), size=(int(n_boot), int(n_clusters)), dtype=np.int64)


def resample_means(d, idx, cluster=None):
    """Mean of the per-subject values d over each resample, vectorized. cluster: label per subject (default: each subject
    is its own cluster); idx indexes the clusters in sorted label order, a resample's mean is the mean over the member
    subjects of its drawn clusters."""
    d = np.asarray(d, dtype=np.float64)
    if cluster is None:
        if idx.shape[1] != d.size:
            raise RobustnessError("the index matrix has a different width than the number of subjects")
        return d[idx].mean(axis=1)
    labels, inv = np.unique(np.asarray(cluster), return_inverse=True)
    if idx.shape[1] != labels.size:
        raise RobustnessError("the index matrix has a different width than the number of clusters")
    csum = np.bincount(inv, weights=d, minlength=labels.size)
    cn = np.bincount(inv, minlength=labels.size).astype(np.float64)
    return csum[idx].sum(axis=1) / cn[idx].sum(axis=1)


def paired_bootstrap(d, idx, cfg, cluster=None):
    """Percentile CI and two-sided p of the mean of the paired per-subject differences d (§11.4, §15.1)."""
    level = float(cfg["statistics"]["ci_level"])
    if cfg["statistics"]["c1"]["p_value_rule"] != "two_sided_percentile":
        raise RobustnessError("only the two_sided_percentile p-value rule is implemented")
    means = resample_means(d, idx, cluster)
    a = (1.0 - level) / 2.0
    lo, hi = (float(v) for v in np.quantile(means, [a, 1.0 - a]))
    b = means.size
    p = min(1.0, 2.0 * min((1 + int(np.sum(means <= 0.0))) / (b + 1), (1 + int(np.sum(means >= 0.0))) / (b + 1)))
    return {"mean": float(np.mean(d)), "ci": [lo, hi], "ci_level": level, "p": float(p), "n": int(np.asarray(d).size),
            "B": int(b)}


def holm(p, m):
    """Holm step-down adjusted p-values for a family of EXACTLY m tests. An incomplete family raises (answer 6: no adjusted
    value is reported until the family is complete), so a partial list can never produce a number."""
    p = np.asarray(p, dtype=np.float64)
    if p.size != int(m):
        raise RobustnessError(f"the Holm family has {int(m)} tests but {p.size} p-values were given")
    order = np.argsort(p, kind="stable")
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---- analyses and the C1 verdict -----------------------------------------------------------------------------

COMPARISONS = (("M3_vs_M2", "M3", "M2", "primary"), ("M3_vs_M0", "M3", "M0", "primary"),
               ("M3_vs_M0b", "M3", "M0b", "secondary_descriptive"), ("M1_vs_M0", "M1", "M0", "secondary_holm_family"),
               ("M2_vs_M1", "M2", "M1", "secondary_holm_family"))


def analyse(rows, cfg, seed, n_boot, include=None, cluster=None):
    """Means and paired-difference bootstrap of the given rows (all with the same variants scored). d = later - earlier,
    so a negative difference favours the later model."""
    keys = sorted(set.intersection(*(set(r["scores"]) for r in rows))) if rows else []
    scores = {v: np.array([r["scores"][v] for r in rows]) for v in keys}
    n = len(rows)
    out = {"n_subjects": n, "means": {v: float(s.mean()) for v, s in scores.items()}, "comparisons": {}}
    if n == 0:
        return out
    n_clusters = n if cluster is None else len(set(cluster))
    idx = draw_index_matrix(n_clusters, n_boot, cfg["statistics"]["bootstrap_seed"], seed)
    out["bootstrap"] = {"B": int(n_boot), "n_clusters": int(n_clusters), "seed": [int(cfg["statistics"]["bootstrap_seed"]),
                                                                              int(seed), int(n_clusters)],
                        "index_matrix_sha256": hashlib.sha256(idx.tobytes()).hexdigest()}
    for name, later, earlier, role in COMPARISONS:
        if include is not None and name not in include:
            continue
        if later in scores and earlier in scores:
            out["comparisons"][name] = dict(paired_bootstrap(scores[later] - scores[earlier], idx, cfg, cluster), role=role,
                                            difference=f"{later} - {earlier}")
    return out


def c1_verdict(analysis, m3_mode):
    """C1 (§15.1): M3 lower than M2 AND lower than M0, each with the whole 95% CI of the paired difference on M3's side of
    zero (upper bound below zero). A CI that excludes zero on the wrong side is a fail. No M3: no verdict."""
    if m3_mode == "absent":
        return {"passed": None, "reason": "no frozen equation: M3 was not scored"}
    c = analysis["comparisons"]
    if "M3_vs_M2" not in c or "M3_vs_M0" not in c:
        return {"passed": None, "reason": "no matched subjects"}
    below = {k: bool(c[k]["ci"][1] < 0.0) for k in ("M3_vs_M2", "M3_vs_M0")}
    reason = "no residual term (a bare constant): M3 is M2 exactly, so the M3 - M2 difference is zero" \
        if m3_mode == "no_term" else ("both CIs lie below zero" if all(below.values()) else "a CI does not lie below zero")
    return {"passed": bool(all(below.values())), "reason": reason, "ci_upper_below_zero": below}


# ---- driver and writer ----------------------------------------------------------------------------------------

def output_path(cfg, root, seed, pilot):
    base = cfg["paths"]["pilot_results_dir"] if pilot else cfg["paths"]["results_dir"]
    return Path(root) / base / cfg["statistics"]["c1"]["output_pattern"].format(seed=seed)


def write_c1(path, doc, force=False):
    """Write once. An identical result (everything but the provenance block) is left alone; a different one is refused
    unless force (a deviation). Returns "written" or "unchanged"."""
    path = Path(path)
    new = json.loads(json.dumps(doc))
    if path.is_file() and not force:
        strip = lambda d: {k: v for k, v in d.items() if k != "provenance"}              # noqa: E731
        if strip(json.loads(path.read_text(encoding="utf-8"))) == strip(new):
            return "unchanged"
        raise RobustnessError(f"{path} exists with a different result; pass force to overwrite (a deviation)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    return "written"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path is not None and Path(path).is_file() else None


def run_c1(cfg, root, seed, *, pilot, confirmatory=False, loader=None, pilot_ids=None, frozen=None, use_cache=True,
           force=False, filter_name=None):
    """C1 for one split seed. Pilot mode: the internal-test pilot subjects only (mechanics), results/pilot/. The recordings
    are read only after the plan, the baseline files and the frozen equation have been checked."""
    from src import regression, tuning
    root = Path(root)
    plan = resolve_subjects(cfg, root, seed, pilot=pilot, confirmatory=confirmatory, pilot_ids=pilot_ids)
    scores, bdoc, bjson, bnpz = load_baseline(cfg, root, seed, plan)
    fpath = regression.frozen_equation_path(cfg, root, seed, pilot)
    if frozen is None and fpath.is_file():
        frozen = regression.load_frozen_equation(fpath)
    if frozen is None:
        if plan.mode == "confirmatory":
            raise GuardError("confirmatory scoring needs the frozen equation")
        m3_mode, residual = "absent", None
    elif frozen.no_term:
        m3_mode, residual = "no_term", None
    else:
        m3_mode, residual = "scored", frozen.residual()
    if loader is None:
        loader = tuning.make_real_loader(cfg, root, plan.pilot_ids, allow_all=plan.allow_all)
    loader = guarded_loader(loader, plan)
    cache_dir = (root / cfg["paths"]["cache_dir"] / cfg["statistics"]["c1"]["cache_subdir"]) if use_cache else None
    rows, runs, skipped = [], {}, []
    for sid in plan.ids:
        if sid not in scores:
            skipped.append({"subject": sid, "reason": "not scored by the baseline (see its skipped_subjects)"})
            continue
        res = loader(sid)
        if res["reason"] is not None:
            raise GuardError(f"{sid} was scored by the baseline but the loader now says: {res['reason']}")
        rec = {"id": sid, "segments": res["segments"], "starts": res["starts"], "key": res.get("key")}
        runs[sid] = score_recording(rec, cfg, residual=residual, filter_name=filter_name, cache_dir=cache_dir,
                                    equation_sha=None if frozen is None else frozen.sha256)
        rows.append(subject_scores(sid, cfg, scores[sid], runs[sid], m3_mode))
        log.info("C1 %s: %s %s", sid, rows[-1]["status"], rows[-1]["reason"] or "")
    n_boot = int(cfg["statistics"]["bootstrap_B_pilot" if pilot else "bootstrap_B"])
    matched = [r for r in rows if r["status"] == "matched"]
    sens = [r for r in rows if r["status"] in ("matched", "m3_loss")]
    primary = analyse(matched, cfg, seed, n_boot)
    sensitivity = analyse(sens, cfg, seed, n_boot, include=("M3_vs_M2", "M3_vs_M0", "M3_vs_M0b")) \
        if m3_mode == "scored" else None
    p_sec = {k: primary["comparisons"][k]["p"] for k in ("M1_vs_M0", "M2_vs_M1") if k in primary["comparisons"]}
    head, dirty = tuning._git_state(root)
    doc = {"schema": int(cfg["statistics"]["c1"]["schema_version"]), "split_seed": int(seed), "pilot": bool(pilot),
           "mechanics_only": bool(pilot), "mode": plan.mode, "filter": passes.resolve_filter(cfg, filter_name),
           "q": passes.resolve_q(cfg, None, passes.resolve_filter(cfg, filter_name)),
           "m3": {"mode": m3_mode, "frozen_equation_sha256": None if frozen is None else frozen.sha256,
                  "equation": None if frozen is None else frozen.equation},
           "subjects": list(plan.ids), "skipped_subjects": skipped + list(bdoc.get("skipped_subjects", [])),
           "per_subject": rows, "divergence": divergence_report(rows, runs, m3_mode, cfg),
           "primary": primary, "sensitivity_m3_loss": sensitivity,
           "m3_loss_rule": cfg["statistics"]["c1"]["m3_loss_rule"],
           "c1_verdict": c1_verdict(primary, m3_mode),
           "c1_verdict_sensitivity": c1_verdict(sensitivity, m3_mode) if sensitivity is not None else None,
           "holm": {"family_size_config": int(cfg["statistics"]["holm_family_size"]), "holm_status": "partial_family",
                    "raw_p": p_sec, "adjusted": None,
                    "note": "the family is the four §15.2 bullets; the other members come from G3 to G5; no adjusted value "
                            "is reported until the family is complete (IMP-078)"},
           "baseline": {"file": bjson.name, "order": bdoc["order"], "mechanics_only": bool(bdoc.get("mechanics_only"))}}
    doc["provenance"] = {"git_commit": head, "git_dirty": dirty, "code_sha256": _code_sha(),
                         "config_yml_sha256": _sha(root / "config.yml"), "baseline_json_sha256": _sha(bjson),
                         "baseline_scores_sha256": _sha(bnpz), "frozen_equation_sha256": None if frozen is None else frozen.sha256,
                         "split_file_sha256": _sha(root / cfg["paths"]["outputs_dir"] / f"split_{seed}.json")}
    path = output_path(cfg, root, seed, pilot)
    status = write_c1(path, doc, force=force)
    log.info("C1, split seed %s: %s (%s)", seed, status, path)
    return {"doc": doc, "path": path, "status": status}


# ==== G2: Holm family, C3 ICC, wPLI / imaginary coherency, AAFT (IMP-080 to IMP-082) ==============================

# ---- Holm family (§14, §15.2; IMP-080) ----------------------------------------------------------------------------

def holm_members(cfg):
    """The members of the Holm family, every test of the four §15.2 bullets (IMP-080): C1 M0->M1 and M1->M2 (G1), the
    free-run error per scoring window (G5), the test-partition ICC per direction (G4) and C1 per non-primary split
    seed (G3). The count must equal statistics.holm_family_size."""
    members = [{"name": f"c1_step:{name}", "bullet": "c1_other_steps", "stage": "G1"}
               for name, _later, _earlier, role in COMPARISONS if role == "secondary_holm_family"]
    members += [{"name": f"freerun:{w}s", "bullet": "freerun_error_per_window", "stage": "G5"}
                for w in cfg["windows"]["scoring_windows_s"]]
    members += [{"name": f"c3_test_icc:{d}", "bullet": "c3_test_partition_icc", "stage": "G4"}
                for d in cfg["statistics"]["c3"]["directions"]]
    members += [{"name": f"c1_seed:{s}", "bullet": "c1_split_seed_level", "stage": "G3"} for s in cfg["split"]["extra_seeds"]]
    size = int(cfg["statistics"]["holm_family_size"])
    if len(members) != size:
        raise RobustnessError(f"the Holm family has {len(members)} members but statistics.holm_family_size is {size}")
    return members


def holm_report(cfg, raw_p):
    """raw_p: {member name: p or None}. Adjusted values only for a COMPLETE family (every member present with a finite p
    in [0, 1]); otherwise status 'partial_family', the missing members, and no adjusted value (IMP-078, IMP-080)."""
    members = holm_members(cfg)
    names = [m["name"] for m in members]
    unknown = sorted(set(raw_p) - set(names))
    if unknown:
        raise RobustnessError(f"not members of the Holm family: {unknown}")
    missing = [n for n in names if raw_p.get(n) is None or not np.isfinite(raw_p[n])]
    rows = [dict(m, raw_p=raw_p.get(m["name"]), adjusted_p=None) for m in members]
    if missing:
        return {"family_size": len(members), "status": "partial_family", "missing": missing, "members": rows}
    p = np.array([raw_p[n] for n in names], dtype=np.float64)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise RobustnessError("a raw p-value lies outside [0, 1]")
    adj = holm(p, len(members))
    for r, a in zip(rows, adj):
        r["adjusted_p"] = float(a)
    return {"family_size": len(members), "status": "complete", "missing": [], "members": rows}


# ---- ICC(3,1) (§11.3, §14, §15.1; IMP-081) --------------------------------------------------------------------------

def icc31_values(Y):
    """ICC(3,1), consistency, of arrays (..., n subjects, k sessions), vectorised over the leading axes (two-way mixed
    model: (MSR - MSE) / (MSR + (k - 1) MSE)). Returns (icc, MSR, MSE)."""
    Y = np.asarray(Y, dtype=np.float64)
    n, k = Y.shape[-2], Y.shape[-1]
    if n < 2 or k < 2:
        raise RobustnessError("an ICC needs at least two subjects and two sessions")
    grand = Y.mean(axis=(-2, -1), keepdims=True)
    row = Y.mean(axis=-1, keepdims=True)
    col = Y.mean(axis=-2, keepdims=True)
    msr = k * np.sum((row - grand) ** 2, axis=(-2, -1)) / (n - 1)
    mse = np.sum((Y - row - col + grand) ** 2, axis=(-2, -1)) / ((n - 1) * (k - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (msr - mse) / (msr + (k - 1) * mse), msr, mse


def _f_ci(F, k, df1, df2, level):
    """F-based two-sided CI of ICC(3,1) (McGraw and Wong 1996 / Shrout and Fleiss 1979): FL = F / F(1-a/2; df1, df2),
    FU = F F(1-a/2; df2, df1), bound = (F* - 1) / (F* + k - 1)."""
    from scipy import stats
    if np.isnan(F):
        return [float("nan"), float("nan")]
    if np.isinf(F):
        return [1.0, 1.0]
    tail = 1.0 - (1.0 - level) / 2.0
    fl = F / stats.f.ppf(tail, df1, df2)
    fu = F * stats.f.ppf(tail, df2, df1)
    return [float((fl - 1.0) / (fl + k - 1.0)), float((fu - 1.0) / (fu + k - 1.0))]


def icc31(Y, level):
    """ICC(3,1) with its F-based CI and the upper-tail F-test of H0 ICC = 0 (the raw p of the Holm member). Reported as is
    with flags (never clipped): 'negative_icc', 'mse_zero' (F infinite, CI [1, 1]), 'undefined' (0/0)."""
    from scipy import stats
    Y = np.asarray(Y, dtype=np.float64)
    n, k = Y.shape
    icc, msr, mse = (float(v) for v in icc31_values(Y))
    df1, df2 = n - 1, (n - 1) * (k - 1)
    flags = []
    with np.errstate(divide="ignore", invalid="ignore"):
        F = float(msr / mse) if mse != 0.0 else (float("inf") if msr > 0.0 else float("nan"))
    if mse == 0.0:
        flags.append("mse_zero")
    if np.isnan(icc):
        flags.append("undefined")
    elif icc < 0.0:
        flags.append("negative_icc")
    p = float(stats.f.sf(F, df1, df2)) if not np.isnan(F) else float("nan")
    return {"icc": icc, "ci": _f_ci(F, k, df1, df2, level), "ci_level": float(level), "F": F, "df": [df1, df2], "MSR": msr,
            "MSE": mse, "p": p, "n": int(n), "k": int(k), "flags": flags}


def icc_for_ci_lower(lower, n, k, level):
    """The observed ICC(3,1) whose F-based CI lower bound equals `lower` at n subjects and k sessions (PLAN Stage I, O6):
    F_L = (1 + (k - 1) lower) / (1 - lower), F = F_L F(1-a/2; n-1, (n-1)(k-1)), ICC = (F - 1) / (F + k - 1)."""
    from scipy import stats
    tail = 1.0 - (1.0 - level) / 2.0
    fl = (1.0 + (k - 1) * lower) / (1.0 - lower)
    F = fl * stats.f.ppf(tail, n - 1, (n - 1) * (k - 1))
    return float((F - 1.0) / (F + k - 1.0))


def vigilance_adjust(Y, V):
    """Residuals of the pooled regression (all sessions of all subjects, one intercept and one slope) of the session
    estimates Y on the session covariate V (§11.3); same shape as Y."""
    Y, V = np.asarray(Y, dtype=np.float64), np.asarray(V, dtype=np.float64)
    if Y.shape != V.shape:
        raise RobustnessError("the covariate must have the shape of the estimates")
    x, y = V.ravel(), Y.ravel()
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return (y - design @ beta).reshape(Y.shape)


def icc_cluster_bootstrap(Y, idx, level):
    """Percentile CI of the ICC over resamples of whole subjects (a pair keeps its sessions together, §11.4), from the
    pre-drawn index matrix idx (B, n); vectorised. Undefined resamples (0/0) are excluded and counted."""
    Y = np.asarray(Y, dtype=np.float64)
    if idx.shape[1] != Y.shape[0]:
        raise RobustnessError("the index matrix has a different width than the number of pairs")
    vals = icc31_values(Y[idx])[0]
    ok = np.isfinite(vals)
    if not ok.any():
        return {"ci": [float("nan"), float("nan")], "B": int(idx.shape[0]), "n_undefined": int(idx.shape[0])}
    a = (1.0 - level) / 2.0
    lo, hi = (float(v) for v in np.quantile(vals[ok], [a, 1.0 - a]))
    return {"ci": [lo, hi], "B": int(idx.shape[0]), "n_undefined": int((~ok).sum())}


def c3_direction(Y, V, cfg, idx):
    """One direction: unadjusted ICC (decides C3), vigilance-adjusted ICC (sensitivity), cluster-bootstrap CI of each."""
    level = float(cfg["statistics"]["ci_level"])
    adj = vigilance_adjust(Y, V)
    out = {"n_pairs": int(Y.shape[0]), "unadjusted": icc31(Y, level), "vigilance_adjusted": icc31(adj, level)}
    out["unadjusted"]["cluster_bootstrap"] = icc_cluster_bootstrap(Y, idx, level)
    out["vigilance_adjusted"]["cluster_bootstrap"] = icc_cluster_bootstrap(adj, idx, level)
    return out


def c3_analysis(pairs, cfg, seed, n_boot):
    """pairs: [{"id", "g12": [t1, t2], "g21": [t1, t2], "vigilance": [t1, t2]}]. One index matrix per analysis (shared by
    both directions and by the adjusted ICCs), width = number of pairs."""
    n = len(pairs)
    if n < 2:
        return {"n_pairs": n, "status": "too_few_pairs"}
    idx = draw_index_matrix(n, n_boot, cfg["statistics"]["bootstrap_seed"], seed)
    V = np.array([p["vigilance"] for p in pairs], dtype=np.float64)
    out = {"n_pairs": n, "status": "ok", "B": int(n_boot),
           "index_matrix_sha256": hashlib.sha256(idx.tobytes()).hexdigest(), "directions": {}}
    for d in cfg["statistics"]["c3"]["directions"]:
        out["directions"][d] = c3_direction(np.array([p[d] for p in pairs], dtype=np.float64), V, cfg, idx)
    return out


def c3_verdict(analysis, cfg, estimator_mode):
    """C3 (§15.1): the F-based CI lower bound of the UNADJUSTED all-pairs ICC is at least the minimum in at least one
    direction. A stand-in estimator (no frozen equation) makes no verdict."""
    if estimator_mode == "M2_stand_in":
        return {"passed": None, "reason": "no frozen equation: the gain estimate is the M2 stand-in, so this is not C3"}
    if analysis.get("status") != "ok":
        return {"passed": None, "reason": "fewer than two usable pairs"}
    lo = float(cfg["statistics"]["criteria"]["c3_icc_ci_lower_min"])
    ok = {d: bool(v["unadjusted"]["ci"][0] >= lo) for d, v in analysis["directions"].items()}
    return {"passed": bool(any(ok.values())), "ci_lower_min": lo, "direction_meets_minimum": ok,
            "reason": "at least one direction has an F-based CI lower bound at the minimum" if any(ok.values())
            else "no direction has an F-based CI lower bound at the minimum"}


# ---- per-session filtered-gain estimate ----------------------------------------------------------------------------

def _pool_size(cfg):
    import psutil
    return max(1, min(int(cfg["compute"]["joblib_n_jobs"]), len(psutil.Process().cpu_affinity())))


def _map(fn, payloads, cfg, n_jobs):
    """Run fn over payloads, inline for one worker, else on the loky pool (spawn-safe: fn is a module-level function)."""
    n_jobs = _pool_size(cfg) if n_jobs is None else int(n_jobs)
    if n_jobs <= 1 or len(payloads) <= 1:
        return [fn(p) for p in payloads]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend=cfg["compute"]["joblib_backend"])(delayed(fn)(p) for p in payloads)


def session_gain(segments, starts, cfg, residual=None, filter_name=None):
    """The §11.3 session estimate: mean FILTERED gain over the clean segments after both burn-ins (passes.Pass1Result
    .gain_estimate), M2 layout, filter A, q_fixed, forward-only pass 1. With a residual this is the M3 estimate (the NumPy
    filter of G0.5), without one the base model (M2) estimate."""
    fname = passes.resolve_filter(cfg, filter_name)
    qv = passes.resolve_q(cfg, None, fname)
    p1 = passes.run_pass1(segments, starts, cfg, q=qv, layout=ss.make_layout(cfg), forward_only=True, filter_name=fname,
                          spec=passes.make_spec(cfg, fname, segments), residual=residual)
    ge = p1.gain_estimate
    return {"g12": None if ge is None else float(ge["g12"]), "g21": None if ge is None else float(ge["g21"]),
            "n": 0 if ge is None else int(ge["n"]), "recording_diverged": bool(p1.recording_diverged),
            "n_clean": int(p1.n_clean), "n_diverged": int(p1.n_diverged), "n_segments": len(p1.segments),
            "n_segments_diverged": int(sum(bool(s.diverged) for s in p1.segments))}


def _gain_worker(payload):
    """Top-level (Windows spawn): one recording's session gain; the frozen equation travels as its document."""
    from threadpoolctl import threadpool_limits
    segments, starts, cfg, frozen_doc, frozen_sha, filter_name = payload
    residual = None
    if frozen_doc is not None:
        from src import regression
        residual = regression.FrozenEquation(doc=frozen_doc, sha256=frozen_sha).residual()
    with threadpool_limits(limits=1):
        return session_gain(segments, starts, cfg, residual=residual, filter_name=filter_name)


def _cache_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else None


def _cache_json_write(path, obj):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8", newline="\n")
    tmp.replace(path)


def vigilance_value(vig):
    """The session's vigilance covariate (O2): mean alpha/theta over every 2-s epoch of every clean segment and both
    channels, from the stored B5 vigilance (pre-rescaling; the epochs span the whole clean data, including the estimator
    burn-in). Returns (value, number of epochs per channel)."""
    ratios = [np.asarray(r, dtype=np.float64) for r in vig["ratios"]]
    n_epochs = int(ratios[0].size)
    if n_epochs == 0:
        raise RobustnessError("the recording has no vigilance epochs")
    return float(np.mean(np.concatenate(ratios))), n_epochs


def build_pairs(rows, cfg):
    """rows: [{"id", "partition", "sessions": {session: {"reason", "gain", "vigilance"}}}]. A pair needs both sessions kept,
    neither diverged at recording level and a gain estimate (O1); every other subject is listed with its reason."""
    s1, s2 = cfg["dataset"]["session_first"], cfg["dataset"]["session_second"]
    pairs, excluded = [], []
    for r in rows:
        why = None
        for s in (s1, s2):
            ses = r["sessions"].get(s)
            if ses is None or ses["reason"] is not None:
                why = f"{s}: {'not loaded' if ses is None else ses['reason']}"
            elif ses["gain"]["recording_diverged"]:
                why = f"{s}: diverged at recording level"
            elif ses["gain"]["g12"] is None or ses["gain"]["g21"] is None:
                why = f"{s}: no gain estimate (no sample after both burn-ins)"
            if why:
                break
        if why:
            excluded.append({"id": r["id"], "partition": r["partition"], "reason": why})
            continue
        pair = {"id": r["id"], "partition": r["partition"], "vigilance": [r["sessions"][s]["vigilance"][0] for s in (s1, s2)]}
        for d in cfg["statistics"]["c3"]["directions"]:
            pair[d] = [r["sessions"][s]["gain"][d] for s in (s1, s2)]
        pairs.append(pair)
    return pairs, excluded


# ---- plans (guard) ---------------------------------------------------------------------------------------------------

def _subjects_with_both_sessions(cfg, root, ids):
    """Of ids, those with an EEG file in both sessions (file names only; no recording is read)."""
    data_root = Path(root) / cfg["paths"]["data_dir"]
    sessions = (cfg["dataset"]["session_first"], cfg["dataset"]["session_second"])
    out = []
    for sid in sorted(ids):
        files = list(data_root.glob(cfg["dataset"]["eeg_glob"].replace("sub-*", sid, 1)))
        if all(any(s in f.parts for f in files) for s in sessions):
            out.append(sid)
    return out


def resolve_c3_subjects(cfg, root, seed, *, pilot, confirmatory=False, pilot_ids=None):
    """The subjects of a C3 run (guard layer 1). Pilot: the pilot subjects that have both sessions (file names only),
    checked to be on the training side; the test list is never loaded. Confirmatory (needs confirmatory=True, a gate that
    permits it and the frozen-equation file): every subject with a second session, both partitions, labelled, because
    C3's primary ICC is over all pairs and the test-partition ICC is secondary (§11.3)."""
    import main as main_mod
    from src import tuning
    root = Path(root)
    if pilot:
        if confirmatory:
            raise GuardError("confirmatory scoring is not a pilot run")
        if pilot_ids is None:
            from src import preprocess as pp
            pilot_ids = pp.load_pilot_ids(cfg, root)
        pilot_ids = frozenset(pilot_ids)
        train = set(main_mod.training_subjects(seed, root, cfg))
        if not pilot_ids <= train:
            raise GuardError("a pilot subject is not on the training side of the split")
        ids = tuple(_subjects_with_both_sessions(cfg, root, pilot_ids))
        return SubjectPlan("pilot", ids, pilot_ids, allow_all=False, partition={s: "train" for s in ids})
    if confirmatory is not True:
        raise GuardError("a non-pilot run must pass confirmatory=True explicitly")
    tuning.check_gate(cfg, root)
    from src import regression
    fpath = regression.frozen_equation_path(cfg, root, seed, False)
    if not fpath.is_file():
        raise GuardError(f"{fpath} does not exist: confirmatory C3 needs the frozen equation")
    split = main_mod.load_split(seed, root, cfg)
    ids = tuple(sorted(split["subjects_with_t2"]))
    test = set(split["test"])
    part = {s: ("test" if s in test else "train") for s in ids}
    return SubjectPlan("confirmatory", ids, frozenset(), allow_all=True, partition=part)


def resolve_diag_subjects(cfg, root, seed, *, pilot, confirmatory=False, pilot_ids=None):
    """The subjects of the wPLI and AAFT diagnostics (ses-t1 recordings). Pilot: all pilot subjects (training side only).
    Confirmatory: the test partition, behind the gate (the diagnostics use the base model M2, so no frozen equation)."""
    import main as main_mod
    from src import tuning
    root = Path(root)
    if pilot:
        if confirmatory:
            raise GuardError("confirmatory scoring is not a pilot run")
        if pilot_ids is None:
            from src import preprocess as pp
            pilot_ids = pp.load_pilot_ids(cfg, root)
        pilot_ids = frozenset(pilot_ids)
        if not pilot_ids <= set(main_mod.training_subjects(seed, root, cfg)):
            raise GuardError("a pilot subject is not on the training side of the split")
        return SubjectPlan("pilot", tuple(sorted(pilot_ids)), pilot_ids, allow_all=False)
    if confirmatory is not True:
        raise GuardError("a non-pilot run must pass confirmatory=True explicitly")
    tuning.check_gate(cfg, root)
    return SubjectPlan("confirmatory", tuple(sorted(main_mod.test_subjects(seed, root, cfg))), frozenset(), allow_all=True)


# ---- JSON hygiene and provenance --------------------------------------------------------------------------------------

def _clean(obj):
    """Python scalars only; NaN and infinities become None (a NaN would also break the write-once comparison)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _provenance(cfg, root, seed, extra):
    from src import tuning
    head, dirty = tuning._git_state(root)
    prov = {"git_commit": head, "git_dirty": dirty, "code_sha256": _code_sha(),
            "config_yml_sha256": _sha(Path(root) / "config.yml"),
            "split_file_sha256": _sha(Path(root) / cfg["paths"]["outputs_dir"] / f"split_{seed}.json")}
    prov.update(extra)
    return prov


def _result_path(cfg, root, seed, pilot, leaf):
    base = cfg["paths"]["pilot_results_dir"] if pilot else cfg["paths"]["results_dir"]
    return Path(root) / base / cfg["statistics"][leaf]["output_pattern"].format(seed=seed)


def c3_output_path(cfg, root, seed, pilot):
    return _result_path(cfg, root, seed, pilot, "c3")


def diagnostics_output_path(cfg, root, seed, pilot):
    return _result_path(cfg, root, seed, pilot, "diagnostics")


# ---- C3 driver ------------------------------------------------------------------------------------------------------

def run_c3(cfg, root, seed, *, pilot, confirmatory=False, loader=None, pilot_ids=None, frozen=None, use_cache=True,
           force=False, filter_name=None, n_jobs=None):
    """C3 for the primary split seed: per-session filtered-gain estimates, pairs, ICC(3,1) per direction. Pilot: the pilot
    subjects with both sessions, results/pilot/, mechanics only; with no frozen equation the estimate is the M2 stand-in
    and no verdict is made."""
    from src import regression, tuning
    root = Path(root)
    plan = resolve_c3_subjects(cfg, root, seed, pilot=pilot, confirmatory=confirmatory, pilot_ids=pilot_ids)
    fpath = regression.frozen_equation_path(cfg, root, seed, pilot)
    if frozen is None and fpath.is_file():
        frozen = regression.load_frozen_equation(fpath)
    if frozen is None:
        if plan.mode == "confirmatory":
            raise GuardError("confirmatory C3 needs the frozen equation")
        mode, doc_frozen = "M2_stand_in", None
    elif frozen.no_term:
        mode, doc_frozen = "M3_no_term_equals_M2", None
    else:
        mode, doc_frozen = "M3", frozen.doc
    if loader is None:
        loader = tuning.make_real_loader(cfg, root, plan.pilot_ids, allow_all=plan.allow_all)
    loader = guarded_loader(loader, plan)
    sessions = (cfg["dataset"]["session_first"], cfg["dataset"]["session_second"])
    cache_dir = (root / cfg["paths"]["cache_dir"] / cfg["statistics"]["c3"]["cache_subdir"]) if use_cache else None
    fname = passes.resolve_filter(cfg, filter_name)
    sha = None if frozen is None else frozen.sha256
    loaded, todo = {}, []
    for sid in plan.ids:
        for ses in sessions:
            res = loader(sid, ses)
            loaded[sid, ses] = res
            if res["reason"] is not None:
                continue
            path = None
            if cache_dir is not None and res.get("key") is not None:
                key = _digest({"rec": res["key"], "filter": fname, "q": passes.resolve_q(cfg, None, fname), "cfg": _digest(cfg),
                               "code": _code_sha(), "starts": [int(x) for x in res["starts"]], "mode": mode, "eq": sha})
                path = cache_dir / f"{sid}_{ses}_{key}.json"
            hit = _cache_json(path)
            if hit is not None:
                res["gain"] = hit
            else:
                todo.append((sid, ses, path))
    payloads = [(loaded[s, ses]["segments"], loaded[s, ses]["starts"], cfg, doc_frozen, sha, fname) for s, ses, _ in todo]
    for (sid, ses, path), gain in zip(todo, _map(_gain_worker, payloads, cfg, n_jobs)):
        loaded[sid, ses]["gain"] = gain
        _cache_json_write(path, gain)
    rows, epochs = [], {}
    for sid in plan.ids:
        sess = {}
        epochs[sid] = {}
        for ses in sessions:
            res = loaded[sid, ses]
            if res["reason"] is not None:
                sess[ses] = {"reason": res["reason"]}
                epochs[sid][ses] = None
                continue
            vig = vigilance_value(res["vigilance"])
            sess[ses] = {"reason": None, "gain": res["gain"], "vigilance": vig}
            epochs[sid][ses] = vig[1]
            log.info("C3 %s %s: g12 %s g21 %s", sid, ses, res["gain"]["g12"], res["gain"]["g21"])
        rows.append({"id": sid, "partition": plan.partition[sid], "sessions": sess})
    pairs, excluded = build_pairs(rows, cfg)
    n_boot = int(cfg["statistics"]["bootstrap_B_pilot" if pilot else "bootstrap_B"])
    level = float(cfg["statistics"]["ci_level"])
    k = len(sessions)
    test_pairs = [p for p in pairs if p["partition"] == "test"]
    all_an = c3_analysis(pairs, cfg, seed, n_boot)
    test_an = c3_analysis(test_pairs, cfg, seed, n_boot) if test_pairs else \
        {"n_pairs": 0, "status": "no_test_partition_pairs" + (" (a pilot run never reads test subjects)" if pilot else "")}
    raw_p = {f"c3_test_icc:{d}": (test_an["directions"][d]["unadjusted"]["p"] if test_an.get("status") == "ok" else None)
             for d in cfg["statistics"]["c3"]["directions"]}
    lower = float(cfg["statistics"]["criteria"]["c3_icc_ci_lower_min"])
    n_t2 = int(cfg["dataset"]["n_subjects_with_t2"])
    doc = {"schema": int(cfg["statistics"]["c3"]["schema_version"]), "split_seed": int(seed), "pilot": bool(pilot),
           "mechanics_only": bool(pilot), "mode": plan.mode, "filter": fname, "q": passes.resolve_q(cfg, None, fname),
           "estimator": {"mode": mode, "frozen_equation_sha256": sha,
                         "equation": None if frozen is None else frozen.equation,
                         "definition": "mean filtered gain over the clean segments after both burn-ins (§11.3)"},
           "subjects": list(plan.ids), "n_subjects_planned": len(plan.ids),
           "pairs": pairs, "excluded_pairs": excluded,
           "gains_per_session": {r["id"]: {s: v.get("gain") for s, v in r["sessions"].items()} for r in rows},
           "vigilance_coverage": {"covariate": cfg["statistics"]["c3"]["vigilance_covariate"], "epochs_per_channel": epochs,
                                  "note": "all 2-s epochs of all clean segments of the session, pre-rescaling, including the "
                                          "estimator burn-in; the gain estimate excludes the burn-ins"},
           "all_pairs": all_an, "test_partition_pairs": test_an, "verdict": c3_verdict(all_an, cfg, mode),
           "criterion": {"ci_lower_min": lower, "direction_rule": "at least one direction, unadjusted all-pairs ICC decides",
                         "observed_icc_equivalent_to_the_minimum": {
                             "at_n_pairs": icc_for_ci_lower(lower, len(pairs), k, level) if len(pairs) >= k else None,
                             "n_pairs": len(pairs), "at_n_subjects_with_t2": icc_for_ci_lower(lower, n_t2, k, level),
                             "n_subjects_with_t2": n_t2}},
           "holm_raw_p": raw_p}
    doc["provenance"] = _provenance(cfg, root, seed, {"frozen_equation_sha256": sha})
    path = c3_output_path(cfg, root, seed, pilot)
    status = write_c1(path, _clean(doc), force=force)
    log.info("C3, split seed %s: %s (%s)", seed, status, path)
    return {"doc": doc, "path": path, "status": status}


# ---- wPLI and imaginary coherency (§14; IMP-082) ------------------------------------------------------------------

def connectivity(segments, cfg):
    """wPLI (Vinck et al. 2011, plain) and imaginary coherency (Nolte et al. 2004) of the two channels over the
    non-overlapping Hann epochs of the clean segments (mean removed, partial epoch dropped). S = X1 conj(X2). Per bin:
    wPLI = |mean Im S| / mean |Im S|, icoh = Im(mean S) / sqrt(mean |X1|^2 mean |X2|^2). Band value = mean over the bins
    of the band (both edges included); the imaginary coherency is signed, and also given as the mean of the bin-wise
    absolute values (the signed band mean can cancel)."""
    fs = cfg["preprocessing"]["observation_fs_hz"]
    ep = int(round(float(cfg["statistics"]["wpli"]["epoch_s"]) * fs))
    win = np.hanning(ep)
    blocks = []
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        if seg.shape[0] != 2:
            raise RobustnessError("connectivity needs exactly two channels")
        n = seg.shape[1] // ep
        if n:
            b = seg[:, :n * ep].reshape(2, n, ep)
            blocks.append(np.fft.rfft((b - b.mean(axis=-1, keepdims=True)) * win, axis=-1))
    if not blocks:
        raise RobustnessError("no complete epoch in the clean segments")
    X = np.concatenate(blocks, axis=1)
    S = X[0] * np.conj(X[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        wpli = np.abs(S.imag.mean(axis=0)) / np.abs(S.imag).mean(axis=0)
        icoh = S.mean(axis=0).imag / np.sqrt((np.abs(X[0]) ** 2).mean(axis=0) * (np.abs(X[1]) ** 2).mean(axis=0))
    freqs = np.fft.rfftfreq(ep, 1.0 / fs)
    bands = {}
    for lo, hi in cfg["statistics"]["wpli_bands_hz"]:
        sel = (freqs >= lo) & (freqs <= hi)
        bands[f"{lo}-{hi}"] = {"wpli": float(np.mean(wpli[sel])), "icoh": float(np.mean(icoh[sel])),
                               "abs_icoh": float(abs(np.mean(icoh[sel]))),
                               "abs_icoh_binwise": float(np.mean(np.abs(icoh[sel]))), "n_bins": int(sel.sum())}
    return {"n_epochs": int(X.shape[1]), "bands": bands}


def connectivity_summary(per_recording, percentiles):
    """Per band: median and IQR (25th and 75th percentiles) over recordings of wPLI and |Im coherency|. No test."""
    out = {}
    if not per_recording:
        return out
    for band in next(iter(per_recording.values()))["bands"]:
        out[band] = {}
        for key in ("wpli", "abs_icoh", "abs_icoh_binwise"):
            v = np.array([r["bands"][band][key] for r in per_recording.values()], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                q1, med, q3 = np.percentile(v, percentiles)
                out[band][key] = {"median": float(med), "iqr": [float(q1), float(q3)], "n": int(v.size)}
            else:
                out[band][key] = {"median": None, "iqr": None, "n": 0}
    return out


# ---- AAFT surrogate test (§14; IMP-082) ---------------------------------------------------------------------------

def aaft_surrogate(x, rng):
    """Amplitude-adjusted Fourier transform surrogate of one 1-D series (Theiler et al. 1992): Gaussianise by rank,
    randomise the phases (DC and Nyquist kept real), rank-remap onto the sorted original values. The surrogate has exactly
    the original values (so the amplitude distribution) and approximately its power spectrum."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="stable")
    gauss = np.empty(n)
    gauss[order] = np.sort(rng.standard_normal(n))
    f = np.fft.rfft(gauss)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=f.size)
    phase[0] = 0.0
    if n % 2 == 0:
        phase[-1] = 0.0
    shuffled = np.fft.irfft(np.abs(f) * np.exp(1j * phase), n=n)
    ranks = np.argsort(np.argsort(shuffled, kind="stable"), kind="stable")
    return x[order][ranks]


def aaft_segments(segments, rng):
    """Every clean segment, every channel, its own AAFT surrogate (O16); starts and gaps are the caller's, unchanged."""
    return [np.stack([aaft_surrogate(ch, rng) for ch in np.asarray(seg, dtype=np.float64)]) for seg in segments]


def aaft_statistic(gain):
    """max(|g12|, |g21|) of the session gain estimate; None when there is no estimate or the recording diverged."""
    if gain["recording_diverged"] or gain["g12"] is None or gain["g21"] is None:
        return None
    return float(max(abs(gain["g12"]), abs(gain["g21"])))


def aaft_p(real, surrogate_stats, n_surrogates):
    """p = (1 + k) / (N + 1), k = surrogates whose statistic is at least the real one; a diverged surrogate (None) counts as
    at least as large (IMP-082). Returns (p, k, number of diverged surrogates)."""
    n_div = int(sum(s is None for s in surrogate_stats))
    k = n_div + int(sum(s is not None and s >= real for s in surrogate_stats))
    return float((1 + k) / (int(n_surrogates) + 1)), k, n_div


def _aaft_worker(payload):
    """Top-level (Windows spawn). kind 'real': the real recording's statistic; kind 'surrogates': the statistics of the
    surrogates `indices`, each from default_rng([seed, recording_index, surrogate_index])."""
    from threadpoolctl import threadpool_limits
    kind, rec_idx, indices, segments, starts, cfg, seed, filter_name = payload
    out = []
    with threadpool_limits(limits=1):
        if kind == "real":
            return [{"index": None, "stat": aaft_statistic(session_gain(segments, starts, cfg, filter_name=filter_name))}]
        for k in indices:
            rng = np.random.default_rng([int(seed), int(rec_idx), int(k)])
            surr = aaft_segments(segments, rng)
            out.append({"index": int(k), "stat": aaft_statistic(session_gain(surr, starts, cfg, filter_name=filter_name))})
    return out


def run_aaft(cfg, plan, loader, *, cache_dir=None, filter_name=None, n_jobs=None):
    """The AAFT test over a plan's ses-t1 recordings: the recordings are drawn by seed (default_rng([aaft.seed, number of
    planned subjects]) permutation of the sorted IDs); a recording that is excluded or whose real M2 run diverges is
    replaced by the next draw (and counted); each accepted recording gets n_surrogates surrogates."""
    ac = cfg["statistics"]["aaft"]
    n_rec, n_sur = int(ac["n_recordings"]), int(ac["n_surrogates"])
    per_task = int(ac["surrogates_per_task"])
    fname = passes.resolve_filter(cfg, filter_name)
    aseed = int(ac["seed"])
    ids = list(plan.ids)
    order = np.random.default_rng([aseed, len(ids)]).permutation(len(ids))
    accepted, replaced, real_stat, recs = [], [], {}, {}
    pos = 0
    while len(accepted) < n_rec and pos < len(order):
        batch = [int(i) for i in order[pos:pos + (n_rec - len(accepted))]]
        pos += len(batch)
        ready = []
        for i in batch:
            res = loader(ids[i])
            if res["reason"] is not None:
                replaced.append({"id": ids[i], "reason": res["reason"]})
                continue
            recs[i] = res
            ready.append(i)
        stats = _map(_aaft_worker, [("real", i, None, recs[i]["segments"], recs[i]["starts"], cfg, aseed, fname) for i in ready],
                     cfg, n_jobs)
        for i, st in zip(ready, stats):
            if st[0]["stat"] is None:
                replaced.append({"id": ids[i], "reason": "the real M2 run diverged (or has no gain estimate)"})
            else:
                real_stat[i] = st[0]["stat"]
                accepted.append(i)
    surr, todo = {}, []
    for i in accepted:
        path = None
        if cache_dir is not None and recs[i].get("key") is not None:
            key = _digest({"rec": recs[i]["key"], "filter": fname, "q": passes.resolve_q(cfg, None, fname), "cfg": _digest(cfg),
                           "code": _code_sha(), "starts": [int(x) for x in recs[i]["starts"]], "seed": aseed, "idx": i,
                           "n": n_sur})
            path = Path(cache_dir) / f"{ids[i]}_{key}.json"
        hit = _cache_json(path)
        if hit is not None:
            surr[i] = hit
            continue
        surr[i] = [None] * n_sur
        todo.append((i, path))
    tasks = []
    for i, _ in todo:
        for a in range(0, n_sur, per_task):
            tasks.append(("surrogates", i, list(range(a, min(a + per_task, n_sur))), recs[i]["segments"], recs[i]["starts"], cfg,
                          aseed, fname))
    for task, res in zip(tasks, _map(_aaft_worker, tasks, cfg, n_jobs)):
        for r in res:
            surr[task[1]][r["index"]] = {"stat": r["stat"]}
    for i, path in todo:
        _cache_json_write(path, surr[i])
    thr = float(ac["p_threshold"])
    rows = []
    for i in sorted(accepted):
        stats = [None if s["stat"] is None else float(s["stat"]) for s in surr[i]]
        p, k, n_div = aaft_p(real_stat[i], stats, n_sur)
        kept = [s for s in stats if s is not None]
        rows.append({"id": ids[i], "recording_index": i, "real_statistic": real_stat[i], "n_surrogates": n_sur,
                     "k_at_least_real": k, "n_surrogates_diverged": n_div, "p": p, "significant": bool(p < thr),
                     "surrogate_median": float(np.median(kept)) if kept else None})
    return {"statistic": ac["statistic"], "p_rule": ac["p_rule"], "p_threshold": thr, "n_recordings_requested": n_rec,
            "n_recordings": len(rows), "n_surrogates": n_sur, "seed": aseed, "draw_order": [ids[int(i)] for i in order],
            "replaced": replaced, "recordings": rows, "n_significant": int(sum(r["significant"] for r in rows)),
            "limitation": "cannot separate coupling from lagged common input (only Null B does, §14)"}


def run_diagnostics(cfg, root, seed, *, pilot, confirmatory=False, loader=None, pilot_ids=None, use_cache=True, force=False,
                    filter_name=None, n_jobs=None):
    """wPLI, imaginary coherency and the AAFT test over the ses-t1 recordings of the plan (pilot: the pilot subjects;
    confirmatory: the test partition). Pilot output: results/pilot/, mechanics only."""
    from src import tuning
    root = Path(root)
    plan = resolve_diag_subjects(cfg, root, seed, pilot=pilot, confirmatory=confirmatory, pilot_ids=pilot_ids)
    if loader is None:
        loader = tuning.make_real_loader(cfg, root, plan.pilot_ids, allow_all=plan.allow_all)
    loader = guarded_loader(loader, plan)
    per, skipped = {}, []
    for sid in plan.ids:
        res = loader(sid)
        if res["reason"] is not None:
            skipped.append({"subject": sid, "reason": res["reason"]})
            continue
        per[sid] = connectivity(res["segments"], cfg)
    cache_dir = (root / cfg["paths"]["cache_dir"] / cfg["statistics"]["diagnostics"]["cache_subdir"]) if use_cache else None
    aaft = run_aaft(cfg, plan, loader, cache_dir=cache_dir, filter_name=filter_name, n_jobs=n_jobs)
    doc = {"schema": int(cfg["statistics"]["diagnostics"]["schema_version"]), "split_seed": int(seed), "pilot": bool(pilot),
           "mechanics_only": bool(pilot), "mode": plan.mode, "session": cfg["dataset"]["session_first"],
           "subjects": list(plan.ids), "skipped_subjects": skipped,
           "connectivity": {"estimator": cfg["statistics"]["wpli"]["estimator"], "epoch_s": cfg["statistics"]["wpli"]["epoch_s"],
                            "bands_hz": cfg["statistics"]["wpli_bands_hz"], "per_recording": per,
                            "summary": connectivity_summary(per, cfg["statistics"]["diagnostics"]["summary_percentiles"]),
                            "note": "a volume-conduction diagnostic; no test, and it cannot separate coupling from lagged "
                                    "common input (§14)"},
           "aaft": aaft}
    doc["provenance"] = _provenance(cfg, root, seed, {})
    path = diagnostics_output_path(cfg, root, seed, pilot)
    status = write_c1(path, _clean(doc), force=force)
    log.info("diagnostics, split seed %s: %s (%s)", seed, status, path)
    return {"doc": doc, "path": path, "status": status}
