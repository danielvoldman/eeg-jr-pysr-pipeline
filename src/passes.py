"""Two-pass parameter handling (PLAN.md C3; §5.1, §7.5, §8.3, §10.2, §11.3; IMP-030 to IMP-036).

Pass 1 (recording level): a continuous forward filter plus unscented RTS smoother over a recording's
clean segments with the full augmented layout. The slow parameters are recording-level constants, so
their posterior mean and covariance are carried from one segment to the next while the 12 neural states
are re-initialized at every segment start (§5.1), with the parameter covariance inflated by the
random-walk variance over the gap (IMP-030). The recording-level value of each parameter is the mean
of its SMOOTHED trajectory over the post-burn-in samples (§7.5).

Pass 2 (windows): non-overlapping 2-s windows cut inside each clean segment, a 12-D neural-only filter
and smoother per window with the recording-level parameters HELD FIXED (they are not in the state, so
they cannot be re-estimated inside a window: the shrinkage problem of §7.5), 12 neural states
re-initialized per window, the first 0.5 s of each window discarded from the returned trajectories
(§10.2).

Nothing here tunes Q or R (q is an argument), derives a residual, estimates a derivative or runs PySR
(Stage D). Inputs are plain arrays: segments of shape (2, n), float64, rescaled mV at the observation
rate, with their start indices; this module never imports preprocess.py.

Filter and q (F0, DEV-005, IMP-074): every entry point resolves the filter from an explicit `filter_name` or else from
config g0.filter (A: 21-D pass 1, 14-D windows) and q from an explicit `q` or else from ukf.process_noise.q_fixed.
The dropped 19-D filter is reachable only as filter_name="19D", and then q must be given explicitly.
"""
import logging
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from src import model
from src import state_space as ss
from src import ukf

log = logging.getLogger(__name__)


class PassError(ValueError):
    """Raised for an inconsistent pass request."""


# option name -> ukf_ext kind ("N" = no extra state = the 19-D filter, which runs through ukf.run_filter itself)
FILTER_KINDS = {"19D": "N", "A": "A", "B": "B"}


def resolve_filter(cfg, filter_name=None):
    """The filter of a run: the explicit `filter_name`, else config g0.filter (the one place that leaf is read for runs;
    the full G0 gate reads it through here too). Raises if unset or not one of g0.filter_options."""
    name = cfg["g0"]["filter"] if filter_name is None else filter_name
    if name is None:
        raise PassError("g0.filter is unset (the DEV-005 decision): no filter can be chosen")
    options = list(cfg["g0"]["filter_options"])
    if name not in options or name not in FILTER_KINDS:
        raise PassError(f"filter {name!r} is not one of {options}")
    return name


def resolve_q(cfg, q=None, filter_name=None):
    """The process-noise scale of a run: the explicit `q`, else ukf.process_noise.q_fixed (DEV-005). The dropped 19-D
    filter never gets q_fixed implicitly. qr_<seed>.json is never read."""
    if q is not None:
        return float(q)
    if filter_name == "19D":
        raise PassError("the 19D filter needs an explicit q (q_fixed is declared for the adopted filter only)")
    value = cfg["ukf"]["process_noise"]["q_fixed"]
    if value is None:
        raise PassError("ukf.process_noise.q_fixed is unset")
    return float(value)


def make_spec(cfg, filter_name, segments):
    """The recording's own (s2, tau) spec for A and B (estimated from its clean segments); None for 19D."""
    if filter_name == "19D":
        return None
    from src import ukf_ext
    return ukf_ext.spec_for(FILTER_KINDS[filter_name], segments, cfg)


def _runners(filter_name, spec):
    """(run_filter, run_smoother) for the filter: ukf's own (looked up at call time) for 19D, the extended ones of
    ukf_ext.make_runners (looked up now) for A and B. No global patching."""
    if filter_name == "19D":
        return (lambda *a, **k: ukf.run_filter(*a, **k)), (lambda *a, **k: ukf.run_smoother(*a, **k))
    from src import ukf_ext
    return ukf_ext.make_runners(spec)


# ---- helpers ------------------------------------------------------------------------------------

def _observation_fs(cfg):
    return cfg["preprocessing"]["observation_fs_hz"]


def _samples(seconds, cfg):
    return int(round(seconds * _observation_fs(cfg)))


def _as_z(segment):
    a = np.asarray(segment, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != ss.N_NODES:
        raise PassError(f"a segment must have shape ({ss.N_NODES}, n), got {a.shape}")
    return np.ascontiguousarray(a.T)


def exceeds_fraction(n_diverged, n_total, fraction):
    """True if n_diverged / n_total is strictly more than `fraction`, in exact arithmetic (§7.5)."""
    return Fraction(int(n_diverged)) > Fraction(str(fraction)) * int(n_total)


def _check_inputs(segments, starts):
    if len(segments) != len(starts):
        raise PassError("segments and starts must have the same length")
    ends = [int(s) + np.asarray(seg).shape[1] for s, seg in zip(starts, segments)]
    for k in range(1, len(starts)):
        if int(starts[k]) < ends[k - 1]:
            raise PassError("segments must be ordered and must not overlap")


# ---- result objects -------------------------------------------------------------------------------

@dataclass
class RecordingParams:
    """Recording-level parameters: means of the smoothed trajectories over post-burn-in samples."""
    p1: float
    p2: float
    log_rho1: float
    log_rho2: float
    rho1: float
    rho2: float
    A1: float
    B1: float
    A2: float
    B2: float
    g12: float
    g21: float
    m: float                    # mean of the per-sample CLIPPED smoothed m (IMP-016, IMP-031)
    m_raw: float                # mean of the unclipped smoothed m
    m_clipped_fraction: float   # fraction of post-burn-in samples with m outside the clip range
    posterior_sd: dict          # per state quantity: mean smoothed posterior SD over the same samples
    n_samples: int
    layout_names: tuple

    def fixed(self):
        """The seven values a fixed-parameter layout needs."""
        return {k: getattr(self, k) for k in ss.PARAM_NAMES_FULL}


@dataclass
class SegmentPass1:
    index: int
    start: int
    n: int
    diverged: bool
    reason: object
    step: object
    monitor: dict
    gap_s: object                    # seconds since the last used segment (None for the first)
    carry_in_mean: object            # parameter mean at the segment start (None: the prior)
    carry_in_var: object             # its diagonal variance, after the gap inflation
    carry_out_mean: object = None    # filtered parameter mean at the segment end (None if diverged)
    carry_out_var: object = None     # its diagonal variance
    x_filt_params: object = None     # (n, n_params) filtered parameter trajectory
    x_smooth_params: object = None   # (n, n_params) smoothed parameter trajectory
    sd_smooth_params: object = None  # (n, n_params) smoothed posterior SD
    n_used: int = 0                  # post-burn-in samples entering the recording-level means
    nis: object = None               # (n,) NIS per step (C4, IMP-040); None if the segment diverged
    nis_keep: object = None          # (n,) bool: samples after both burn-ins (the C4 tuning samples)
    z_pred: object = None            # (n, 2) one-step-ahead predicted observation, rescaled units (F0); None if diverged
    sq_err: object = None            # (n,) one-step squared error averaged over the two channels (F0); None if diverged


@dataclass
class Pass1Result:
    params: object                   # RecordingParams or None
    segments: list
    layout: object
    q: float
    n_clean: int
    n_diverged: int
    diverged_fraction: float
    recording_diverged: bool
    gain_estimate: object            # {"g12", "g21", "n"}: mean FILTERED gain, the §9.2 null-gate estimator
    burn_in_samples: int
    estimator_burn_in_samples: int
    filter_name: str = None
    state_dim: int = None            # pass-1 state width: base layout + the noise states of A / B (21 for A)
    spec: object = None              # the recording's (s2, tau) spec (None for 19D)


@dataclass
class WindowResult:
    segment: int
    window: int
    start: int                       # index of the window's first sample, same axis as `starts`
    diverged: bool
    reason: object
    step: object
    monitor: dict
    x_smooth: object = None          # (n_window - burn, 12) smoothed neural states
    s_delayed: object = None         # (n_window - burn, 2) delayed source S from the forward buffer


@dataclass
class Pass2Result:
    windows: list
    fixed_layout: object
    q: float
    n_attempted: int                 # samples in attempted windows
    n_diverged: int
    diverged_fraction: float
    recording_diverged: bool
    burn_in_samples: int
    window_samples: int
    filter_name: str = None
    state_dim: int = None            # window state width (14 for A)

    @property
    def kept(self):
        return [w for w in self.windows if not w.diverged]


@dataclass
class RecordingResult:
    pass1: Pass1Result
    pass2: object
    recording_diverged: bool


# ---- pass 1 -----------------------------------------------------------------------------------------

def run_pass1(segments, starts, cfg, q=None, layout=None, forward_only=False, filter_name=None, spec=None):
    """Recording-level pass over the clean segments (§7.5). See the module docstring and IMP-030/031.

    filter_name / q: explicit, else config (g0.filter, ukf.process_noise.q_fixed); see resolve_filter, resolve_q. `spec`
    (A and B) is the recording's (s2, tau); it is estimated from `segments` when not given.

    forward_only=True (C4, IMP-040) skips the smoother and the stored covariances: params is None, the
    parameter carry and the divergence rule are unchanged, and each segment carries its NIS and the mask
    of samples after both burn-ins. The default is unchanged."""
    _check_inputs(segments, starts)
    filter_name = resolve_filter(cfg, filter_name)
    q = resolve_q(cfg, q, filter_name)
    spec = make_spec(cfg, filter_name, segments) if spec is None and filter_name != "19D" else spec
    run_filter, run_smoother = _runners(filter_name, spec)
    layout = ss.make_layout(cfg) if layout is None else layout
    if layout.fixed_params is not None:
        raise PassError("pass 1 needs a layout with the parameters in the state")
    n_neural = ss.N_NEURAL
    burn = _samples(cfg["windows"]["training_burn_in_s"], cfg)
    est_burn = _samples(cfg["passes"]["estimator_burn_in_s"], cfg)
    fs = _observation_fs(cfg)
    walk = cfg["ukf"]["process_noise"]["parameter_random_walk_factor"]
    lo, hi = cfg["priors"]["m_truncate"]
    prior_x, prior_P = ss.prior_mean(layout, cfg), ss.prior_cov(layout, cfg)
    prior_var_par = np.diag(prior_P)[n_neural:]
    par_names = layout.names[n_neural:]

    carry_x = carry_P = None
    last_end = None
    processed = 0
    sums = {k: 0.0 for k in ss.PARAM_NAMES_FULL}
    m_raw_sum = m_clip_count = 0.0
    sd_sums = {k: 0.0 for k in par_names}
    n_used = 0
    gain_sums = {"g12": 0.0, "g21": 0.0}
    n_gain = 0
    seg_results = []
    n_clean = n_div = 0

    for k, (seg, start) in enumerate(zip(segments, starts)):
        z = _as_z(seg)
        T = z.shape[0]
        n_clean += T
        gap_s = carry_in_mean = carry_in_var = None
        x0, P0 = prior_x.copy(), prior_P.copy()
        if carry_x is not None:
            gap_s = max(0, int(start) - last_end) / fs
            x0[n_neural:] = carry_x
            P0[n_neural:, n_neural:] = carry_P + np.diag(walk * prior_var_par * gap_s)
            carry_in_mean, carry_in_var = carry_x.copy(), np.diag(P0)[n_neural:].copy()
        res = run_filter(z, cfg, layout, q, x0=x0, P0=P0, keep_cov=not forward_only)
        seg_res = SegmentPass1(index=k, start=int(start), n=T, diverged=res.diverged,
                               reason=res.divergence_reason, step=res.divergence_step,
                               monitor=res.monitor, gap_s=gap_s, carry_in_mean=carry_in_mean,
                               carry_in_var=carry_in_var)
        seg_results.append(seg_res)
        if res.diverged:
            n_div += T
            log.warning("pass 1: segment %d (start %d, %d samples) diverged at step %s (%s); dropped",
                        k, start, T, res.divergence_step, res.divergence_reason)
            continue
        if not forward_only:
            xs, Ps = run_smoother(res, cfg)
            seg_res.x_filt_params = res.x[:, n_neural:].copy()
            seg_res.x_smooth_params = xs[:, n_neural:].copy()
            seg_res.sd_smooth_params = np.sqrt(np.einsum("tii->ti", Ps[:, n_neural:, n_neural:]))
        if getattr(res, "z_pred", None) is not None:        # stub filters of some tests carry no prediction
            seg_res.z_pred = np.array(res.z_pred[:T], dtype=np.float64)
            seg_res.sq_err = np.mean((z - seg_res.z_pred) ** 2, axis=1)
        carry_x = res.x[-1, n_neural:].copy()
        carry_P = (res.P_last if forward_only else res.P[-1])[n_neural:, n_neural:].copy()
        seg_res.carry_out_mean, seg_res.carry_out_var = carry_x.copy(), np.diag(carry_P).copy()
        last_end = int(start) + T

        if not forward_only and T > burn:                     # smoothed means over post-burn-in samples
            q_s = layout.params(xs[burn:])
            m_col = q_s["m"]
            for key in ss.PARAM_NAMES_FULL:
                sums[key] += float(np.sum(np.clip(m_col, lo, hi) if key == "m" else q_s[key]))
            m_raw_sum += float(np.sum(m_col))
            m_clip_count += float(np.count_nonzero((m_col < lo) | (m_col > hi)))
            for i, name in enumerate(par_names):
                sd_sums[name] += float(np.sum(seg_res.sd_smooth_params[burn:, i]))
            seg_res.n_used = T - burn
            n_used += T - burn
        cum = processed + np.arange(T)
        keep = (np.arange(T) >= burn) & (cum >= est_burn)     # samples after both burn-ins
        if forward_only:                                      # only the tuning path reads them (C4)
            seg_res.nis, seg_res.nis_keep = res.nis.copy(), keep
        if layout.include_gains:                              # filtered gains after both burn-ins
            if keep.any():
                q_f = layout.params(res.x[keep])
                gain_sums["g12"] += float(np.sum(q_f["g12"]))
                gain_sums["g21"] += float(np.sum(q_f["g21"]))
                n_gain += int(keep.sum())
        processed += T

    params = None
    if n_used:
        mean = {k: sums[k] / n_used for k in sums}
        rho1, rho2 = np.exp(mean["log_rho1"]), np.exp(mean["log_rho2"])
        A1, B1 = (float(v) for v in ss.rho_to_AB(rho1, cfg))
        A2, B2 = (float(v) for v in ss.rho_to_AB(rho2, cfg))
        params = RecordingParams(
            p1=mean["p1"], p2=mean["p2"], log_rho1=mean["log_rho1"], log_rho2=mean["log_rho2"],
            rho1=float(rho1), rho2=float(rho2), A1=A1, B1=B1, A2=A2, B2=B2,
            g12=mean["g12"], g21=mean["g21"], m=mean["m"], m_raw=m_raw_sum / n_used,
            m_clipped_fraction=m_clip_count / n_used,
            posterior_sd={k: v / n_used for k, v in sd_sums.items()}, n_samples=n_used,
            layout_names=tuple(layout.names))
    gain_estimate = None
    if layout.include_gains and n_gain:
        gain_estimate = {"g12": gain_sums["g12"] / n_gain, "g21": gain_sums["g21"] / n_gain, "n": n_gain}
    frac = cfg["ukf"]["divergence"]["recording_fraction"]
    return Pass1Result(params=params, segments=seg_results, layout=layout, q=q, n_clean=n_clean,
                       n_diverged=n_div, diverged_fraction=(n_div / n_clean) if n_clean else 0.0,
                       recording_diverged=exceeds_fraction(n_div, n_clean, frac),
                       gain_estimate=gain_estimate, burn_in_samples=burn, estimator_burn_in_samples=est_burn,
                       filter_name=filter_name, state_dim=layout.n + (spec.nx if spec is not None else 0), spec=spec)


# ---- pass 2 -----------------------------------------------------------------------------------------

def cut_windows(n_samples, window_samples):
    """Start offsets of the whole non-overlapping windows inside one segment; the tail is dropped."""
    return [w * window_samples for w in range(int(n_samples) // int(window_samples))]


def run_pass2(segments, starts, params, cfg, q=None, skip_segments=(), filter_name=None, spec=None):
    """2-s windows with the recording-level parameters held fixed (§7.5, §10.2). See IMP-032. filter_name, q and spec as
    in run_pass1 (pass the pass-1 spec so both passes use the recording's one (s2, tau)); A windows are 14-D."""
    _check_inputs(segments, starts)
    if params is None:
        raise PassError("pass 2 needs the recording-level parameters of pass 1")
    filter_name = resolve_filter(cfg, filter_name)
    q = resolve_q(cfg, q, filter_name)
    spec = make_spec(cfg, filter_name, segments) if spec is None and filter_name != "19D" else spec
    run_filter, run_smoother = _runners(filter_name, spec)
    win = _samples(cfg["windows"]["training_window_s"], cfg)
    burn = _samples(cfg["windows"]["training_burn_in_s"], cfg)
    delay = cfg["coupling"]["delay_substeps"]
    flayout = ss.make_fixed_layout(cfg, params.fixed())
    windows = []
    n_attempted = n_div = 0
    for k, (seg, start) in enumerate(zip(segments, starts)):
        if k in skip_segments:
            continue
        z = _as_z(seg)
        for w, off in enumerate(cut_windows(z.shape[0], win)):
            zw = np.ascontiguousarray(z[off:off + win])          # exactly the window's own samples
            buf = ss.make_buffer(cfg)                            # fresh buffer, steady-state fill
            res = run_filter(zw, cfg, flayout, q, buffer=buf, keep_cov=True)
            n_attempted += win
            rec = WindowResult(segment=k, window=w, start=int(start) + off, diverged=res.diverged,
                               reason=res.divergence_reason, step=res.divergence_step,
                               monitor=res.monitor)
            if res.diverged:
                n_div += win
                log.warning("pass 2: window %d of segment %d diverged at step %s (%s); dropped",
                            w, k, res.divergence_step, res.divergence_reason)
            else:
                xs, _ = run_smoother(res, cfg)
                # the delayed source S seen by sample t is lag `delay` of the buffer as it stands after
                # step t, i.e. at the start of step t + 1 (forward-filter means, never smoothed, §7.5)
                snaps = np.concatenate([res.snapshots[1:], ukf.buffer_snapshot(buf)[None]])
                rec.x_smooth = xs[burn:].copy()
                rec.s_delayed = snaps[burn:, delay, :].copy()
            windows.append(rec)
    frac = cfg["ukf"]["divergence"]["recording_fraction"]
    return Pass2Result(windows=windows, fixed_layout=flayout, q=q, n_attempted=n_attempted,
                       n_diverged=n_div, diverged_fraction=(n_div / n_attempted) if n_attempted else 0.0,
                       recording_diverged=exceeds_fraction(n_div, n_attempted, frac),
                       burn_in_samples=burn, window_samples=win, filter_name=filter_name,
                       state_dim=flayout.n + (spec.nx if spec is not None else 0))


def run_recording(segments, starts, cfg, q=None, layout=None, filter_name=None):
    """Both passes. recording_diverged is the OR of the two passes, each on its own clean-sample
    denominator (IMP-035). Pass 2 is skipped for a recording that already diverged in pass 1, and
    skips the segments that diverged in pass 1. The recording's (s2, tau) spec is estimated once and shared by both
    passes."""
    filter_name = resolve_filter(cfg, filter_name)
    q = resolve_q(cfg, q, filter_name)
    p1 = run_pass1(segments, starts, cfg, q, layout=layout, filter_name=filter_name,
                   spec=make_spec(cfg, filter_name, segments))
    p2 = None
    if p1.params is not None and not p1.recording_diverged:
        skip = {s.index for s in p1.segments if s.diverged}
        p2 = run_pass2(segments, starts, p1.params, cfg, q, skip_segments=skip, filter_name=filter_name, spec=p1.spec)
    flag = p1.recording_diverged or (p2 is not None and p2.recording_diverged)
    return RecordingResult(pass1=p1, pass2=p2, recording_diverged=flag)


# ---- base model prediction for §8.3 (no residual, no derivative estimator here) -------------------------

def base_dy4dt(x_smooth, s_delayed, params, cfg):
    """Base-coupling model's dy4/dt per node, shape (n, 2), from smoothed neural states (n, 12), the
    delayed source S (n, 2) and the recording-level parameters (§7.1, §8.1, §8.3):
    A a [p + C2 S(C1 y0) + g S_source(t - d)] - 2 a y4 - a^2 y1. The residual of §8.3 is formed in Stage D."""
    k = model.constants(cfg)
    x = np.asarray(x_smooth, dtype=np.float64)
    s = np.asarray(s_delayed, dtype=np.float64)
    out = np.empty((x.shape[0], ss.N_NODES))
    for j, (p, A, g, src) in enumerate(((params.p1, params.A1, params.g21, 1),
                                        (params.p2, params.A2, params.g12, 0))):
        y0, y1, y4 = (x[:, j * model.N_STATES + i] for i in (model.Y0, model.Y1, model.Y4))
        s0 = model.sigmoid(k["C1"] * y0, k["e0"], k["v0"], k["r"])
        out[:, j] = A * k["a"] * (p + k["C2"] * s0 + g * s[:, src]) - 2.0 * k["a"] * y4 - k["a"] ** 2 * y1
    return out
