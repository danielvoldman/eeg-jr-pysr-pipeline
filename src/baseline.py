"""M0 and M0b: the two-channel linear VAR baselines (PLAN F1; §13.1, §10.2; IMP-075, IMP-076).

M0: VAR(p), p in 1..32 samples, order chosen by 5-fold SUBJECT-WISE cross-validation on training subjects (one-step MSE),
coefficients (and one intercept per channel) from least squares POOLED over the training subjects, then frozen.
M0b: the same order, coefficients re-fitted on each test recording itself (in-sample, deliberately generous).

Rules (all in IMP-075 / IMP-076; values in config baseline.* and the burn-in leaves read by passes.scoring_mask):
- A design row never reaches across a gap: lags come from the same clean segment only.
- Every order is fitted on the same rows (sample index >= the largest order inside its segment), so the CV curve compares
  orders on identical data.
- Least squares by per-subject QR with stacked R factors (the lags of a 256-Hz signal are strongly collinear, so normal
  equations would square the condition number). The columns are [intercept, lag 1 (ch 1, ch 2), lag 2, ...], so the
  fit of order p is the leading block of the R factor of order 32.
- Scoring (§10.2): one-step squared error in rescaled units averaged over the two channels, on the samples of
  passes.scoring_mask (the same function the filter side uses): the first 0.5 s of every segment and the first 6 s of the
  recording are out. The mask never depends on a divergence. The burn-in must be at least the largest order, so VAR can
  predict every scored sample; otherwise this module raises (it never silently drops samples).
- Per-subject score = mean over that subject's scored samples; the headline is the mean over subjects.

Nothing here touches preprocess.py at module level, imports model.py, or reads the test side before a frozen fit exists.
Stage G reads outputs/baseline_scores_<seed>.npz (format in `write_outputs`).
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from src import passes

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


class BaselineError(ValueError):
    """Raised for an unusable baseline request."""


# ---- config ---------------------------------------------------------------------------------------------

def _bcfg(cfg):
    b = cfg["baseline"]
    lo, hi = (int(v) for v in b["var_order_range"])
    if not 1 <= lo <= hi:
        raise BaselineError(f"baseline.var_order_range {b['var_order_range']} is not a valid range")
    if b["cv_aggregation"] != "per_subject_mean":
        raise BaselineError(f"baseline.cv_aggregation {b['cv_aggregation']!r} is not implemented")
    for key in ("cv_seed", "pilot_internal_seed"):
        if b[key] is None:
            raise BaselineError(f"baseline.{key} is unset")
    return lo, hi, int(b["var_cv_folds"]), bool(b["intercept"])


def _check_burn_in(masks, hi):
    """Every scored sample must have `hi` earlier samples in its own segment."""
    for m in masks:
        idx = np.flatnonzero(m)
        if idx.size and idx[0] < hi:
            raise BaselineError(f"a scored sample has only {int(idx[0])} earlier samples in its segment, fewer than the "
                                f"largest VAR order {hi}: the burn-in is shorter than the order range")


# ---- design ---------------------------------------------------------------------------------------------

def lag_design(segment, order, first, intercept=True):
    """(X, Y, rows): for one (2, n) segment the rows t = first .. n-1 (first >= order). X columns:
    [1,] z(t-1) (2 values), z(t-2), ..., z(t-order); Y = z(t); rows = t. Lags never leave the segment."""
    z = np.asarray(segment, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != 2:
        raise BaselineError(f"a segment must have shape (2, n), got {z.shape}")
    if first < order:
        raise BaselineError("the first row must have `order` earlier samples")
    n = z.shape[1]
    m = n - first
    if m <= 0:
        return np.empty((0, int(intercept) + 2 * order)), np.empty((0, 2)), np.empty(0, dtype=np.int64)
    cols = np.empty((m, int(intercept) + 2 * order))
    if intercept:
        cols[:, 0] = 1.0
    for lag in range(1, order + 1):
        c = int(intercept) + 2 * (lag - 1)
        cols[:, c:c + 2] = z[:, first - lag:n - lag].T
    return cols, np.ascontiguousarray(z[:, first:].T), np.arange(first, n)


def _r_of(blocks):
    """R factor of the stacked [X | Y] blocks (a list of (X, Y)); exact least squares information, no normal equations."""
    r = None
    for x, y in blocks:
        if x.shape[0] == 0:
            continue
        a = np.hstack([x, y])
        rr = np.linalg.qr(a, mode="r") if r is None else np.linalg.qr(np.vstack([r, a]), mode="r")
        r = rr
    if r is None:
        raise BaselineError("no usable rows (every segment shorter than the largest order)")
    return r


def recording_r(rec, order_max, intercept):
    """R factor of one recording's rows (t >= order_max inside every segment)."""
    return _r_of([lag_design(s, order_max, order_max, intercept)[:2] for s in rec["segments"]])


def combine_r(rs):
    return np.linalg.qr(np.vstack(rs), mode="r")


def coef_from_r(r, order, order_max, intercept):
    """Coefficients (k, 2) of the order-`order` fit from the R factor of order `order_max`."""
    n_ic = int(intercept)
    c_full = n_ic + 2 * order_max
    k = n_ic + 2 * order
    rxx = r[:k, :k]
    d = np.abs(np.diag(rxx))
    if rxx.shape[0] < k or not np.all(np.isfinite(rxx)) or d.min() <= 1e-12 * d.max():
        raise BaselineError(f"the design of order {order} is rank deficient")
    return solve_triangular(rxx, r[:k, c_full:c_full + 2], lower=False)


@dataclass
class VarFit:
    order: int
    coef: np.ndarray            # (n_ic + 2 order, 2): column i predicts channel i
    intercept: bool

    @property
    def A(self):
        """order x 2 x 2: A[l-1][i, j] multiplies z_j(t - l) in channel i's prediction."""
        n_ic = int(self.intercept)
        return np.stack([self.coef[n_ic + 2 * l:n_ic + 2 * l + 2, :].T for l in range(self.order)])

    @property
    def c(self):
        return self.coef[0].copy() if self.intercept else np.zeros(2)


def predict_segment(fit, segment):
    """(2, n) one-step predictions of one segment; NaN for the first `order` samples (no full history)."""
    x, _, rows = lag_design(segment, fit.order, fit.order, fit.intercept)
    out = np.full((2, np.asarray(segment).shape[1]), np.nan)
    if rows.size:
        out[:, rows] = (x @ fit.coef).T
    return out


# ---- scoring (§10.2) ------------------------------------------------------------------------------------

def score_recording(fit, rec, cfg):
    """Per-sample channel-mean squared one-step error over the whole recording (concatenated segments), NaN where VAR has no
    full history, with the shared scoring mask. Returns dict(err, mask, lengths, starts)."""
    lengths = [int(np.asarray(s).shape[1]) for s in rec["segments"]]
    masks = passes.scoring_mask(lengths, cfg)
    _check_burn_in(masks, _bcfg(cfg)[1])
    errs = []
    for seg, _ in zip(rec["segments"], masks):
        seg = np.asarray(seg, dtype=np.float64)
        errs.append(np.mean((seg - predict_segment(fit, seg)) ** 2, axis=0))
    return {"err": np.concatenate(errs), "mask": np.concatenate(masks), "lengths": np.array(lengths, dtype=np.int64),
            "starts": np.array([int(s) for s in rec["starts"]], dtype=np.int64)}


def subject_mse(scored):
    """Mean squared error of one scored recording over its scored samples; raises if a scored sample has no prediction."""
    return passes.subject_score([scored["err"]], [scored["mask"]])


def fit_recording(rec, order, cfg):
    """M0b: the order-`order` fit on this recording alone (rows t >= order inside each segment)."""
    _, hi, _, ic = _bcfg(cfg)
    r = _r_of([lag_design(s, order, order, ic)[:2] for s in rec["segments"]])
    return VarFit(order, coef_from_r(r, order, order, ic), ic)


# ---- folds and CV (§13.1) -------------------------------------------------------------------------------

def make_folds(ids, n_folds, seed):
    """Subject-wise folds: a seeded permutation of the sorted IDs split into n_folds nearly equal groups."""
    ids = sorted(ids)
    if len(set(ids)) != len(ids):
        raise BaselineError("duplicate subject IDs")
    if n_folds > len(ids):
        raise BaselineError(f"{n_folds} folds need at least {n_folds} subjects, got {len(ids)}")
    perm = np.random.default_rng(int(seed)).permutation(len(ids))
    return [sorted(ids[i] for i in part) for part in np.array_split(perm, n_folds)]


@dataclass
class CVResult:
    orders: list
    mse: list                   # mean over subjects of the held-out per-subject MSE, per order
    per_fold: list              # [fold][order] mean over that fold's subjects
    per_subject: dict           # id -> [per order]
    folds: list
    order: int                  # chosen: smallest MSE, ties to the smaller order
    seed: int
    fold_coefs: list = None     # [fold] {order: coefficients} fitted WITHOUT that fold's subjects (audit)


def choose_order(mse, orders):
    """The order with the smallest CV error; an exact tie goes to the smaller order (orders ascending)."""
    return orders[int(np.argmin(mse))]


def cv_select_order(recs, cfg):
    """5-fold subject-wise CV of the order (M0). Each recording is a dict(id, segments, starts)."""
    lo, hi, n_folds, ic = _bcfg(cfg)
    by_id = {r["id"]: r for r in recs}
    seed = int(cfg["baseline"]["cv_seed"])
    folds = make_folds(list(by_id), n_folds, seed)
    for r in recs:
        _check_burn_in(passes.scoring_mask([np.asarray(s).shape[1] for s in r["segments"]], cfg), hi)
    r_sub = {rid: recording_r(rec, hi, ic) for rid, rec in by_id.items()}
    orders = list(range(lo, hi + 1))
    per_subject, fold_coefs = {}, []
    for held in folds:
        train = combine_r([r_sub[i] for i in by_id if i not in set(held)])
        coefs = {p: coef_from_r(train, p, hi, ic) for p in orders}
        fold_coefs.append(coefs)
        for sid in held:
            rec = by_id[sid]
            lengths = [np.asarray(s).shape[1] for s in rec["segments"]]
            masks = passes.scoring_mask(lengths, cfg)
            xs, ys, keeps = [], [], []
            for seg, m in zip(rec["segments"], masks):
                x, y, rows = lag_design(seg, hi, hi, ic)
                xs.append(x), ys.append(y), keeps.append(m[rows])
            x, y, keep = np.vstack(xs), np.vstack(ys), np.concatenate(keeps)
            if not keep.any():
                raise BaselineError(f"subject {sid} has no scored samples")
            x, y = x[keep], y[keep]
            per_subject[sid] = [float(np.mean((y - x[:, :int(ic) + 2 * p] @ coefs[p]) ** 2)) for p in orders]
    mse = [float(np.mean([per_subject[i][j] for i in per_subject])) for j in range(len(orders))]
    per_fold = [[float(np.mean([per_subject[i][j] for i in held])) for j in range(len(orders))] for held in folds]
    return CVResult(orders=orders, mse=mse, per_fold=per_fold, per_subject=per_subject, folds=folds,
                    order=choose_order(mse, orders), seed=seed, fold_coefs=fold_coefs)


def fit_pooled(recs, order, cfg):
    """M0 coefficients: least squares pooled over the given training recordings (rows t >= the largest order)."""
    _, hi, _, ic = _bcfg(cfg)
    r = combine_r([recording_r(rec, hi, ic) for rec in recs])
    return VarFit(order, coef_from_r(r, order, hi, ic), ic)


def fit_m0(train_recs, cfg):
    """Order by CV, then the frozen pooled fit on all the training recordings. Returns (VarFit, CVResult)."""
    cv = cv_select_order(train_recs, cfg)
    return fit_pooled(train_recs, cv.order, cfg), cv


# ---- driver: real recordings (pilot mode or the confirmatory split) ----------------------------------------

def pilot_internal_split(pilot_ids, cfg):
    """The 8/4 mechanics split of the pilot subjects (split.pilot_internal_split): a seeded permutation of the sorted IDs,
    the first n_train are the 'training' side. Mechanics only; both sides are training-side subjects of the real split."""
    n_train, n_test = (int(v) for v in cfg["split"]["pilot_internal_split"])
    ids = sorted(pilot_ids)
    if len(ids) != n_train + n_test:
        raise BaselineError(f"{len(ids)} pilot IDs for a {n_train}/{n_test} internal split")
    perm = np.random.default_rng(int(cfg["baseline"]["pilot_internal_seed"])).permutation(len(ids))
    return sorted(ids[i] for i in perm[:n_train]), sorted(ids[i] for i in perm[n_train:])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def output_paths(cfg, root, seed, pilot):
    base = Path(root) / (cfg["paths"]["pilot_results_dir"] if pilot else cfg["paths"]["outputs_dir"])
    b = cfg["baseline"]
    return base / b["output_var_pattern"].format(seed=seed), base / b["output_scores_pattern"].format(seed=seed)


def build_document(cfg, root, seed, pilot, fit, cv, train_ids, test_ids, skipped, summary, scores_name):
    from src import tuning
    head, dirty = tuning._git_state(root)
    n_ic = int(fit.intercept)
    return {
        "schema": 1, "split_seed": int(seed), "pilot": bool(pilot),
        "mechanics_only": bool(pilot),
        "order": fit.order, "order_range": list(cfg["baseline"]["var_order_range"]),
        "A": fit.A.tolist(), "intercept": fit.c.tolist(), "intercept_fitted": bool(fit.intercept),
        "cv": {"seed": cv.seed, "n_folds": len(cv.folds), "folds": cv.folds, "orders": cv.orders, "mse_by_order": cv.mse,
               "mse_by_fold_and_order": cv.per_fold, "per_subject_mse": cv.per_subject,
               "criterion": cfg["baseline"]["cv_aggregation"], "tie_rule": "smaller order",
               "scored_on": "scoring_mask samples of the held-out subjects"},
        "fit": {"subjects": sorted(train_ids), "n_subjects": len(train_ids), "rows": "t >= largest order inside each segment",
                "method": "per-subject QR, stacked R factors (exact least squares)"},
        "mask_rule": {"burn_in_segment_s": cfg["windows"]["training_burn_in_s"], "burn_in_recording_s": cfg["passes"]["estimator_burn_in_s"],
                      "divergence_independent": True, "ref": "IMP-075"},
        "scored_subjects": sorted(test_ids), "skipped_subjects": skipped, "summary": summary,
        "scores_file": scores_name,
        "scores_format": "npz keys '<id>|starts', '<id>|lengths', '<id>|mask', '<id>|M0', '<id>|M0b'; M0 and M0b are per-sample "
                         "channel-mean squared one-step errors over the concatenated clean segments (NaN where VAR has no full "
                         "history); mask = passes.scoring_mask; Stage G intersects it with the matched filter segments and takes "
                         "passes.subject_score",
        "n_coefficients_check": n_ic + 2 * fit.order,
        "provenance": {"git_commit": head, "git_dirty": dirty,
                       "code_sha256": hashlib.sha256(b"".join((REPO_ROOT / "src" / n).read_bytes()
                                                              for n in ("baseline.py", "passes.py"))).hexdigest(),
                       "config_yml_sha256": _sha(Path(root) / "config.yml"),
                       "split_file_sha256": _sha(Path(root) / cfg["paths"]["outputs_dir"] / f"split_{seed}.json")}}


def write_outputs(paths, doc, arrays, force=False):
    """Write once. An identical result (everything but the provenance block) is left alone; a different one is refused
    unless force (a deviation). Returns "written" or "unchanged"."""
    jpath, npath = Path(paths[0]), Path(paths[1])
    strip = lambda d: {k: v for k, v in d.items() if k != "provenance"}              # noqa: E731
    new_doc = json.loads(json.dumps(doc))
    if jpath.is_file() and not force:
        same_json = strip(json.loads(jpath.read_text(encoding="utf-8"))) == strip(new_doc)
        same_npz = False
        if npath.is_file():
            with np.load(npath) as old:
                same_npz = set(old.files) == set(arrays) and all(np.array_equal(old[k], arrays[k], equal_nan=old[k].dtype.kind == "f")
                                                                  for k in arrays)
        if same_json and same_npz:
            return "unchanged"
        raise BaselineError(f"{jpath} exists with a different result; pass force to overwrite (a deviation)")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = npath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(npath)
    jpath.write_text(json.dumps(new_doc, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return "written"


def load_scores(npz_path):
    """The format Stage G reads: {id: {starts, lengths, mask, M0, M0b}}."""
    out = {}
    with np.load(npz_path) as z:
        for key in z.files:
            rid, name = key.rsplit("|", 1)
            out.setdefault(rid, {})[name] = z[key]
    return out


def _load(ids, loader):
    recs, skipped = [], []
    for sid in ids:
        r = loader(sid)
        if r["reason"] is None:
            recs.append({"id": sid, "segments": r["segments"], "starts": r["starts"]})
        else:
            skipped.append({"subject": sid, "reason": r["reason"]})
            log.warning("baseline: %s not used: %s", sid, r["reason"])
    return recs, skipped


def run_baseline(cfg, root, seed, pilot=False, loader=None, pilot_ids=None, force=False):
    """Fit M0 on the training side and score M0 and M0b on the held-out side. Pilot mode: the 12 pilot subjects split 8/4
    (mechanics only, results/pilot/); otherwise the split_<seed>.json 78/33 and a gate.json that permits it (rule 6).
    The held-out recordings are read only after the frozen fit exists. Not wired into main.py."""
    import main as main_mod
    from src import tuning
    root = Path(root)
    if not pilot:
        tuning.check_gate(cfg, root)
    train_all = list(main_mod.training_subjects(seed, root, cfg))
    if pilot:
        if pilot_ids is None:
            from src import preprocess as pp
            pilot_ids = pp.load_pilot_ids(cfg, root)
        if not set(pilot_ids) <= set(train_all):
            raise BaselineError("a pilot subject is not on the training side of the split")
        train_ids, test_ids = pilot_internal_split(pilot_ids, cfg)
    else:
        train_ids, test_ids = sorted(train_all), sorted(main_mod.test_subjects(seed, root, cfg))
    if loader is None:
        loader = tuning.make_real_loader(cfg, root, pilot_ids if pilot_ids is not None else [], allow_all=not pilot)
    train_recs, skipped = _load(train_ids, loader)
    fit, cv = fit_m0(train_recs, cfg)
    log.info("baseline: order %d chosen on %d training subjects (%s)", fit.order, len(train_recs),
             "pilot, mechanics only" if pilot else "confirmatory split")
    test_recs, skipped_test = _load(test_ids, loader)
    arrays, summary = {}, {"M0": {"per_subject": {}}, "M0b": {"per_subject": {}}}
    for rec in test_recs:
        a, b = score_recording(fit, rec, cfg), score_recording(fit_recording(rec, fit.order, cfg), rec, cfg)
        for key, val in (("starts", a["starts"]), ("lengths", a["lengths"]), ("mask", a["mask"]), ("M0", a["err"]), ("M0b", b["err"])):
            arrays[f"{rec['id']}|{key}"] = val
        summary["M0"]["per_subject"][rec["id"]] = subject_mse(a)
        summary["M0b"]["per_subject"][rec["id"]] = subject_mse(b)
    for v in summary.values():
        v["mean"] = float(np.mean(list(v["per_subject"].values()))) if v["per_subject"] else None
    paths = output_paths(cfg, root, seed, pilot)
    doc = build_document(cfg, root, seed, pilot, fit, cv, [r["id"] for r in train_recs], [r["id"] for r in test_recs],
                         skipped + skipped_test, summary, paths[1].name)
    status = write_outputs(paths, doc, arrays, force=force)
    log.info("baseline, split seed %s: %s", seed, status)
    return {"fit": fit, "cv": cv, "summary": summary, "paths": paths, "status": status, "doc": doc}
