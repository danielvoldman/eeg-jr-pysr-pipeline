"""DEV-007 tests: the dwell requirement of the state-SD divergence clause and the parameter-excursion clause (§7.5).

Expected values are independent of the code under test: the filter is run once with both clauses OFF (state multiple inf,
parameter multiple null), and the step at which each rule must fire is computed here from that trajectory by a plain
replay (ukf.DivergenceReference for the centre, the stored SDs, prior means and SDs from the config). A filter run with a
rule on follows the flag-off trajectory exactly until it stops, so the replayed step is the expected divergence step.
"""
import copy

import numpy as np
import pytest

from src import freerun as fr
from src import state_space as ss
from src import ukf, ukf_ext
from src.config import load_config
from tests import sim_data
from tests.legacy_rule import legacy

CFG = load_config()
Q = 1.0e-2
FS = CFG["preprocessing"]["observation_fs_hz"]
N_EXEMPT = int(round(CFG["ukf"]["divergence"]["startup_exempt_s"] * FS))
N_DWELL = 128
SD12 = np.sqrt(np.tile(ss.neural_variance(CFG), ss.N_NODES))
PARAMS = ("p1", "p2", "log_rho1", "log_rho2", "g12", "g21", "m")


@pytest.fixture(scope="module")
def layout():
    return ss.make_layout(CFG)


def flag_off(cfg):
    c = copy.deepcopy(cfg)
    c["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
    c["ukf"]["divergence"]["parameter_sd_multiple"] = None
    return c


def data(seconds=6, seed=31):
    return sim_data.simulate_stream(CFG, seconds, seed)[1]


def reference_run(z, layout):
    return ukf.run_filter(z, flag_off(CFG), layout, Q, backend="numba")


def state_beyond(res, layout):
    ref = ukf.DivergenceReference(layout, CFG)
    X = res.x[:res.n_done]
    dev = np.array([np.abs(X[t, :ss.N_NEURAL] - np.asarray(ref.center(X[t]))) / SD12 for t in range(len(X))])
    beyond = np.any(dev > CFG["ukf"]["divergence"]["state_sd_multiple"], axis=1)
    beyond[:N_EXEMPT] = False
    return beyond


def param_beyond(res, layout, mult=8.0):
    X = res.x[:res.n_done]
    sd = dict(ss.parameter_prior_sd(CFG))
    sd["m"] = CFG["priors"]["m_sd"]
    prior = ss.prior_mean(layout, CFG)
    out = np.zeros(len(X), dtype=bool)
    for name in PARAMS:
        if name in layout.idx:
            c = layout.idx[name]
            out |= np.abs(X[:, c] - prior[c]) > mult * sd[name]
    return out


def first_run_end(beyond, n):
    """Index of the sample that completes the first run of n consecutive True values, or None."""
    run = 0
    for t, b in enumerate(beyond):
        run = run + 1 if b else 0
        if run >= n:
            return t
    return None


# ---- configuration ----------------------------------------------------------------------------------------

def test_config_values_and_settings(layout):
    dv = CFG["ukf"]["divergence"]
    assert (dv["state_sd_multiple"], dv["state_dwell_s"], dv["parameter_sd_multiple"], dv["parameter_dwell_s"]) == (10, 0.5, 8, 0.5)
    n_dwell, n_pdwell, pcols, pmean, plim = ukf.dwell_settings(layout, CFG)
    assert (n_dwell, n_pdwell) == (N_DWELL, N_DWELL)
    assert [int(c) for c in pcols] == [layout.idx[k] for k in PARAMS]
    sd = ss.parameter_prior_sd(CFG)
    assert plim[0] == pytest.approx(8 * sd["p1"]) and plim[4] == pytest.approx(8 * sd["g12"]) and plim[6] == pytest.approx(8 * CFG["priors"]["m_sd"])
    assert pmean[0] == pytest.approx(CFG["priors"]["p_mean"])
    off = legacy(CFG)
    n1, _, _, _, plim_off = ukf.dwell_settings(layout, off)
    assert n1 == 1 and np.all(np.isinf(plim_off))


def test_m1_and_fixed_layouts_have_no_absent_parameters():
    m1 = ss.make_layout(CFG, include_gains=False)
    _, _, pcols, _, plim = ukf.dwell_settings(m1, CFG)
    assert pcols[4] == -1 and pcols[5] == -1 and pcols[0] >= 0
    fixed = ss.make_fixed_layout(CFG, {k: 1.0 for k in PARAMS})
    _, _, pcols_f, _, _ = ukf.dwell_settings(fixed, CFG)
    assert np.all(pcols_f == -1)


# ---- the monitor (NumPy oracle) -----------------------------------------------------------------------------

def monitor_inputs(layout):
    x = ss.prior_mean(layout, CFG).copy()
    return x, np.eye(layout.n), ss.initial_neural_state(CFG)


def test_state_counter_needs_128_consecutive_samples_and_resets(layout):
    mon = ukf.DwellMonitor(layout, CFG)
    x, P, centre = monitor_inputs(layout)
    far = x.copy()
    far[1] = centre[1] + 11.0 * SD12[1]
    for _ in range(N_DWELL - 1):
        assert mon.check(far, P, CFG, centre, SD12, 1.0, True) is None
    assert mon.check(x, P, CFG, centre, SD12, 1.0, True) is None                   # one sample back inside: reset
    for _ in range(N_DWELL - 1):
        assert mon.check(far, P, CFG, centre, SD12, 1.0, True) is None
    assert mon.check(far, P, CFG, centre, SD12, 1.0, True) == "state_beyond_sd_multiple"   # the 128th


def test_samples_inside_the_startup_exemption_do_not_count(layout):
    mon = ukf.DwellMonitor(layout, CFG)
    x, P, centre = monitor_inputs(layout)
    far = x.copy()
    far[1] = centre[1] + 11.0 * SD12[1]
    for _ in range(300):
        assert mon.check(far, P, CFG, centre, SD12, 1.0, False) is None
    assert mon.cnt_s == 0
    for _ in range(N_DWELL - 1):
        assert mon.check(far, P, CFG, centre, SD12, 1.0, True) is None


def test_parameter_counter_boundary_reset_and_columns(layout):
    mon = ukf.DwellMonitor(layout, CFG)
    x, P, centre = monitor_inputs(layout)
    sd = ss.parameter_prior_sd(CFG)
    inside, outside = x.copy(), x.copy()
    inside[layout.idx["p1"]] += 7.9 * sd["p1"]
    outside[layout.idx["p1"]] += 8.1 * sd["p1"]
    for _ in range(3 * N_DWELL):
        assert mon.check(inside, P, CFG, centre, SD12, 1.0, True) is None
    for _ in range(N_DWELL - 1):
        assert mon.check(outside, P, CFG, centre, SD12, 1.0, False) is None          # no exemption for this clause
    assert mon.check(inside, P, CFG, centre, SD12, 1.0, False) is None               # reset
    for _ in range(N_DWELL - 1):
        assert mon.check(outside, P, CFG, centre, SD12, 1.0, False) is None
    assert mon.check(outside, P, CFG, centre, SD12, 1.0, False) == "parameter_beyond_prior_sd"
    for name in PARAMS:                                                              # every listed parameter is watched, below too
        mon2 = ukf.DwellMonitor(layout, CFG)
        y = x.copy()
        s = dict(sd)
        s["m"] = CFG["priors"]["m_sd"]
        y[layout.idx[name]] -= 8.5 * s[name]
        got = [mon2.check(y, P, CFG, centre, SD12, 1.0, False) for _ in range(N_DWELL)]
        assert got[-1] == "parameter_beyond_prior_sd" and all(g is None for g in got[:-1]), name


def test_state_clause_wins_a_tie_and_nan_and_pd_stay_immediate(layout):
    mon = ukf.DwellMonitor(layout, CFG)
    x, P, centre = monitor_inputs(layout)
    both = x.copy()
    both[1] = centre[1] + 11.0 * SD12[1]
    both[layout.idx["p1"]] += 9.0 * ss.parameter_prior_sd(CFG)["p1"]
    got = [mon.check(both, P, CFG, centre, SD12, 1.0, True) for _ in range(N_DWELL)]
    assert got[-1] == "state_beyond_sd_multiple"
    Pn = P.copy()
    Pn[0, 0] = np.nan
    assert ukf.DwellMonitor(layout, CFG).check(x, Pn, CFG, centre, SD12, 1.0, True) == "nan_inf"
    assert ukf.DwellMonitor(layout, CFG).check(x, P, CFG, centre, SD12, -1e-3, True) == "covariance_not_pd"


def test_legacy_settings_fire_on_the_first_crossing_and_never_on_parameters(layout):
    off = legacy(CFG)
    mon = ukf.DwellMonitor(layout, off)
    x, P, centre = monitor_inputs(layout)
    far = x.copy()
    far[1] = centre[1] + 11.0 * SD12[1]
    assert mon.check(far, P, off, centre, SD12, 1.0, True) == "state_beyond_sd_multiple"
    huge = x.copy()
    huge[layout.idx["p1"]] += 1e6
    assert ukf.DwellMonitor(layout, off).check(huge, P, off, centre, SD12, 1.0, True) is None


# ---- filter runs, both backends, against the replayed expectation ------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_brief_spike_is_survived_by_the_new_rule_and_stops_the_legacy_one(layout, backend):
    z = data()
    z[300:310] += 20.0
    ref = reference_run(z, layout)
    b = state_beyond(ref, layout)
    assert 0 < b.sum() and first_run_end(b, N_DWELL) is None                          # a real crossing, shorter than the dwell
    assert first_run_end(param_beyond(ref, layout), N_DWELL) is None                  # and no parameter run either
    first = int(np.argmax(b))
    leg = ukf.run_filter(z, legacy(CFG), layout, Q, backend=backend)
    assert leg.diverged and leg.divergence_reason == "state_beyond_sd_multiple" and leg.divergence_step == first
    new = ukf.run_filter(z, CFG, layout, Q, backend=backend)
    assert not new.diverged and new.n_done == z.shape[0]
    np.testing.assert_allclose(new.x, ref.x, rtol=1e-8, atol=1e-10)                   # the trajectory is the flag-off one


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_sustained_offset_is_stopped_at_the_replayed_step(layout, backend):
    z = data() + 100.0
    ref = reference_run(z, layout)
    exp_state = first_run_end(state_beyond(ref, layout), N_DWELL)
    exp_par = first_run_end(param_beyond(ref, layout), N_DWELL)
    assert exp_par is not None                                                         # the parameters run away here
    expected = min(t for t in (exp_state, exp_par) if t is not None)
    reason = "state_beyond_sd_multiple" if exp_state is not None and exp_state <= exp_par else "parameter_beyond_prior_sd"
    res = ukf.run_filter(z, CFG, layout, Q, backend=backend)
    assert res.diverged and res.divergence_step == expected and res.divergence_reason == reason
    assert res.n_done == expected


def test_the_two_backends_agree_on_the_dwell_rule(layout):
    for z in (data() + 100.0, np.where(np.arange(1536)[:, None] == 400, data() + 80.0, data())):
        a = ukf.run_filter(z, CFG, layout, Q, backend="numpy")
        b = ukf.run_filter(z, CFG, layout, Q, backend="numba")
        assert (a.diverged, a.divergence_step, a.divergence_reason, a.n_done) == (b.diverged, b.divergence_step, b.divergence_reason, b.n_done)


def test_legacy_settings_reproduce_the_flag_off_trajectory_bit_for_bit_when_nothing_fires(layout):
    z = data()
    ref = reference_run(z, layout)
    for backend in ("numpy", "numba"):
        leg = ukf.run_filter(z, legacy(CFG), layout, Q, backend=backend)
        new = ukf.run_filter(z, CFG, layout, Q, backend=backend)
        assert not leg.diverged and not new.diverged
        assert np.array_equal(leg.x, new.x) and np.array_equal(leg.nis, new.nis)
    leg_np = ukf.run_filter(z, legacy(CFG), layout, Q, backend="numpy")
    leg_nb = ukf.run_filter(z, legacy(CFG), layout, Q, backend="numba")
    np.testing.assert_allclose(leg_nb.x, leg_np.x, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ref.x, leg_nb.x, rtol=1e-8, atol=1e-10)


def test_the_extended_kernel_follows_the_same_rule(layout):
    z2 = data() + 100.0
    spec2 = ukf_ext.spec_for("A", [z2.T], CFG)
    run2, _ = ukf_ext.make_runners(spec2)
    off = run2(z2, flag_off(CFG), layout, Q)                    # the extended filter's own flag-off trajectory
    assert not off.diverged
    ref = type("R", (), {"x": off.x[:, :layout.n], "n_done": off.n_done})
    exp_state = first_run_end(state_beyond(ref, layout), N_DWELL)
    exp_par = first_run_end(param_beyond(ref, layout), N_DWELL)
    cand = [t for t in (exp_state, exp_par) if t is not None]
    assert cand
    bad = run2(z2, CFG, layout, Q)
    assert bad.diverged and bad.divergence_step == min(cand)
    assert bad.divergence_reason == ("state_beyond_sd_multiple" if exp_state is not None and exp_state <= (exp_par or 10**9) else "parameter_beyond_prior_sd")
    leg = run2(z2, legacy(CFG), layout, Q)
    first = int(np.argmax(state_beyond(ref, layout)))
    assert leg.diverged and leg.divergence_reason == "state_beyond_sd_multiple" and leg.divergence_step == first
    z = data()
    ok = ukf_ext.make_runners(ukf_ext.spec_for("A", [z.T], CFG))[0](z, CFG, layout, Q)
    assert not ok.diverged


def test_a_fixed_parameter_window_has_only_the_state_clause(layout):
    params = {k: 1.0 for k in PARAMS}
    params.update(p1=220.0, p2=220.0, log_rho1=float(np.log(3.25 / 22)), log_rho2=float(np.log(3.25 / 22)), g12=0.0, g21=0.0, m=0.2)
    fixed = ss.make_fixed_layout(CFG, params)
    z = data()[:512]
    res = ukf.run_filter(z + 100.0, CFG, fixed, Q, backend="numba")
    assert res.diverged and res.divergence_reason != "parameter_beyond_prior_sd"


# ---- free-run (IMP-086 amended) ------------------------------------------------------------------------------

def test_free_run_needs_the_dwell_before_a_realization_is_unstable(monkeypatch):
    from test_freerun import LAYOUT, start_state, Q as QF
    spec0 = ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0))
    sd0 = np.sqrt(CFG["ukf"]["initial_state"]["neural_variance"][0])
    st = start_state(spec0, shift=30.0 * sd0)
    monkeypatch.setattr(fr, "propagate", lambda X, ring, pots, layout, cfg, residual: X)     # hold the state far away
    out = fr.simulate(st, N_DWELL + 20, np.random.default_rng(1), CFG, LAYOUT, spec0, QF, n_real=3)
    assert out["unstable"].all()
    assert np.isfinite(out["y"][:, :N_DWELL]).all() and np.isnan(out["y"][:, N_DWELL:]).all()   # stops after the 128th step
    near = fr.simulate(start_state(spec0, shift=9.0 * sd0), N_DWELL + 20, np.random.default_rng(1), CFG, LAYOUT, spec0, QF, n_real=3)
    assert not near["unstable"].any()                                                              # 9 SD held for ever is inside the bound
    short = fr.simulate(st, N_DWELL - 1, np.random.default_rng(1), CFG, LAYOUT, spec0, QF, n_real=3)
    assert not short["unstable"].any()


def test_the_m3_numpy_filter_follows_the_same_rule(layout):
    from src import ukf_resid
    z = data()
    z[300:310] += 20.0                                                   # the brief crossing of the spike test (survived)
    spec = ukf_ext.Spec("N")
    ok = ukf_resid.run_filter_numpy(z, CFG, layout, Q, spec)
    assert not ok.diverged and ok.n_done == z.shape[0]
    leg = ukf_resid.run_filter_numpy(z, legacy(CFG), layout, Q, spec)
    assert leg.diverged and leg.divergence_reason == "state_beyond_sd_multiple"
    z2 = data() + 100.0
    ref = ukf_resid.run_filter_numpy(z2, flag_off(CFG), layout, Q, spec)
    exp = min(t for t in (first_run_end(state_beyond(ref, layout), N_DWELL), first_run_end(param_beyond(ref, layout), N_DWELL)) if t is not None)
    bad = ukf_resid.run_filter_numpy(z2, CFG, layout, Q, spec)
    assert bad.diverged and bad.divergence_step == exp


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_parameter_far_BELOW_its_prior_is_flagged_at_the_replayed_step(layout, backend):
    z = data()
    x0, P0, _ = ukf.initial_state(layout, CFG)
    x0 = np.array(x0)
    sd = ss.parameter_prior_sd(CFG)
    x0[layout.idx["g12"]] -= 9.0 * sd["g12"]
    P0 = np.array(P0)
    P0[layout.idx["g12"], layout.idx["g12"]] = 1e-4                                 # the filter trusts the displaced start
    ref = ukf.run_filter(z, flag_off(CFG), layout, Q, x0=x0, P0=P0, backend="numba")
    exp = first_run_end(param_beyond(ref, layout), N_DWELL)
    assert exp is not None and exp == N_DWELL - 1                                  # beyond from the first sample, below the prior
    res = ukf.run_filter(z, CFG, layout, Q, x0=x0, P0=P0, backend=backend)
    assert res.diverged and res.divergence_reason == "parameter_beyond_prior_sd" and res.divergence_step == exp


def test_the_extended_kernel_flags_a_parameter_below_its_prior(layout):
    z = data()
    x0, P0, _ = ukf.initial_state(layout, CFG)
    x0 = np.array(x0)
    x0[layout.idx["g12"]] -= 9.0 * ss.parameter_prior_sd(CFG)["g12"]
    P0 = np.array(P0)
    P0[layout.idx["g12"], layout.idx["g12"]] = 1e-4
    run, _ = ukf_ext.make_runners(ukf_ext.spec_for("A", [z.T], CFG))
    res = run(z, CFG, layout, Q, x0=x0, P0=P0)
    assert res.diverged and res.divergence_reason == "parameter_beyond_prior_sd" and res.divergence_step == N_DWELL - 1


def test_free_run_counter_resets_on_a_sample_inside_the_bound(monkeypatch):
    from test_freerun import LAYOUT, start_state, Q as QF
    spec0 = ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0))
    sd0 = np.sqrt(CFG["ukf"]["initial_state"]["neural_variance"][0])
    far = np.tile(start_state(spec0, shift=30.0 * sd0)["x"][:LAYOUT.n], (3, 1))
    near = np.tile(start_state(spec0)["x"][:LAYOUT.n], (3, 1))
    calls = [0]

    def prop(X, ring, pots, layout, cfg, residual):
        calls[0] += 1
        return (near if calls[0] % (N_DWELL - 10) == 0 else far).copy()          # one inside sample every 118 steps

    monkeypatch.setattr(fr, "propagate", prop)
    out = fr.simulate(start_state(spec0), 4 * N_DWELL, np.random.default_rng(1), CFG, LAYOUT, spec0, QF, n_real=3)
    assert not out["unstable"].any() and np.isfinite(out["y"]).all()
