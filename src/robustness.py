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

    def load(subject):
        if subject not in allowed:
            raise GuardError(f"{subject} is not one of the {len(allowed)} planned subjects; refusing to read it")
        return loader(subject)

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
