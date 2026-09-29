"""C1 tests: augmented state-space model, priors, delay buffer (§7.1-§7.6, §8.1; IMP-014 to IMP-018).

Expected values come from plain numpy/scipy written here, never from the code under test
(the right-hand side and the sigmoid are re-implemented locally). Nothing touches the real
config.yml except reading it.
"""
import ast
import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from src import model, state_space as ss
from src.config import DEFAULT_CONFIG_PATH, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG0 = load_config()
JR = CFG0["jansen_rit"]
A0, B0, a, b = JR["A"], JR["B"], JR["a"], JR["b"]
C = JR["C"]
C1, C2, C3, C4 = C * JR["C1_multiplier"], C * JR["C2_multiplier"], C * JR["C3_multiplier"], C * JR["C4_multiplier"]
E0, V0, R_ = JR["e0"], JR["v0"], JR["r"]
P0 = CFG0["priors"]["p_mean"]
DUMMY_VAR = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def S(v):
    """Independent sigmoid (§7.3, no offset)."""
    return 2.0 * E0 / (1.0 + np.exp(R_ * (V0 - np.asarray(v, dtype=float))))


def rhs_node(y, p, drive, A, B):
    """Independent §7.1 right-hand side of one node: derivatives of (y0, y1, y2, y3, y4, y5)."""
    y0, y1, y2, y3, y4, y5 = y
    return [y3, y4, y5,
            A * a * S(y1 - y2) - 2 * a * y3 - a * a * y0,
            A * a * (p + C2 * S(C1 * y0) + drive) - 2 * a * y4 - a * a * y1,
            B * b * C4 * S(C3 * y0) - 2 * b * y5 - b * b * y2]


def rhs_full(t, y, p, g12, g21, s_delayed, A=A0, B=B0):
    """d/dt of the 12 neural states, node-major, with a constant delayed S."""
    return np.array(rhs_node(y[:6], p, g21 * s_delayed[1], A, B)
                    + rhs_node(y[6:], p, g12 * s_delayed[0], A, B))


@pytest.fixture
def cfg():
    c = copy.deepcopy(CFG0)
    c["ukf"]["initial_state"]["neural_variance"] = list(DUMMY_VAR)
    return c


def with_switches(cfg, **kw):
    c = copy.deepcopy(cfg)
    c["state"]["reduction_switches"].update(kw)
    return c


# ---- layout ---------------------------------------------------------------------------

def test_layout_dimensions_and_indices(cfg):
    full = ss.make_layout(cfg)
    m1 = ss.make_layout(cfg, include_gains=False)
    assert full.n == 19 == cfg["state"]["n_augmented_states"]
    assert m1.n == 17 == cfg["state"]["n_augmented_states_M1"]
    assert full.idx["p1"] == 12 == ss.P1 and full.idx["m"] == 18 == ss.M
    assert (ss.P2, ss.LOG_RHO1, ss.LOG_RHO2, ss.G12, ss.G21) == (13, 14, 15, 16, 17)
    assert "g12" not in m1.idx and "g21" not in m1.idx and m1.idx["m"] == 16
    assert full.names[:12] == m1.names[:12]


@pytest.mark.parametrize("kw,include_gains,n", [
    ({"fix_EI_terms": True}, True, 17),
    ({"tie_p1_p2": True}, True, 18),
    ({"tie_g12_g21": True}, True, 18),
    ({"fix_EI_terms": True, "tie_p1_p2": True, "tie_g12_g21": True}, True, 15),
    ({"fix_EI_terms": True}, False, 15),
    ({"tie_g12_g21": True}, False, 17),       # M1 has no gains to tie
    ({"fix_EI_terms": True, "tie_p1_p2": True}, False, 14),
])
def test_layout_reduction_switches(cfg, kw, include_gains, n):
    layout = ss.make_layout(with_switches(cfg, **kw), include_gains=include_gains)
    assert layout.n == n
    assert prior_shapes_ok(with_switches(cfg, **kw), layout)


def prior_shapes_ok(cfg, layout):
    return ss.prior_mean(layout, cfg).shape == (layout.n,) and ss.prior_cov(layout, cfg).shape == (layout.n, layout.n)


def test_fixed_EI_pins_log_rho_at_prior_mean_and_ties_hold(cfg):
    c = with_switches(cfg, fix_EI_terms=True, tie_p1_p2=True, tie_g12_g21=True)
    layout = ss.make_layout(c)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, layout.n))
    q = layout.params(X)
    expected = math.log(3.25 / 22.0)
    assert np.all(q["log_rho1"] == expected) and np.all(q["log_rho2"] == expected)
    assert np.array_equal(q["p2"], q["p1"]) and np.array_equal(q["g21"], q["g12"])
    # M1: gains are zero and m is still there
    m1 = ss.make_layout(cfg, include_gains=False)
    q1 = m1.params(rng.normal(size=(4, m1.n)))
    assert np.all(q1["g12"] == 0) and np.all(q1["g21"] == 0)


def test_prior_values_match_section_7_6(cfg):
    layout = ss.make_layout(cfg)
    mean = ss.prior_mean(layout, cfg)
    cov = ss.prior_cov(layout, cfg)
    assert np.array_equal(cov, np.diag(np.diag(cov)))
    # section 7.6: p ~ N(220, 50^2); log rho ~ N(log(3.25/22), 0.2^2); g ~ N(0, (0.1*C2)^2), C2 = 108;
    # m ~ N(0.2, 0.15^2) (raw, truncation only inside observe)
    np.testing.assert_allclose(mean[12:], [220, 220, math.log(3.25 / 22), math.log(3.25 / 22), 0, 0, 0.2], rtol=1e-14)
    np.testing.assert_allclose(np.diag(cov)[12:], [2500, 2500, 0.04, 0.04, 10.8 ** 2, 10.8 ** 2, 0.15 ** 2], rtol=1e-12)
    # neural: same variance vector on both nodes, means = the steady state of each node
    np.testing.assert_allclose(np.diag(cov)[:12], DUMMY_VAR * 2)
    np.testing.assert_allclose(mean[:6], mean[6:12])
    assert mean[3] == mean[4] == mean[5] == 0.0


def test_prior_cov_needs_stored_variance():
    c = copy.deepcopy(CFG0)
    c["ukf"]["initial_state"]["neural_variance"] = None
    with pytest.raises(ss.StateSpaceError):
        ss.prior_cov(ss.make_layout(c), c)


# ---- reparameterization ----------------------------------------------------------------

def test_reparameterization(cfg):
    rho = np.exp(np.random.default_rng(2).normal(math.log(3.25 / 22), 0.2, size=50))
    A, B = ss.rho_to_AB(rho, cfg)
    np.testing.assert_allclose(A * B, 3.25 * 22, rtol=1e-13)
    np.testing.assert_allclose(ss.AB_to_rho(A, B), rho, rtol=1e-13)
    A0_, B0_ = ss.rho_to_AB(3.25 / 22, cfg)
    assert A0_ == pytest.approx(3.25, rel=1e-13) and B0_ == pytest.approx(22.0, rel=1e-13)
    # the product is the config's and equals 3.25 * 22
    assert cfg["priors"]["AB_product"] == 3.25 * 22


# ---- fixed points -------------------------------------------------------------------------

def uncoupled_equilibrium(p, A, B):
    def eqs(u):
        y0, y1, y2 = u
        return [y0 - (A / a) * S(y1 - y2),
                y1 - (A / a) * (p + C2 * S(C1 * y0)),
                y2 - (B / b) * C4 * S(C3 * y0)]
    sol, info, ier, msg = fsolve(eqs, [0.1, 20.0, 15.0], xtol=1e-14, full_output=True)
    assert ier == 1, msg
    return sol


def test_fixed_point_drift_zero_and_helper_agrees(cfg):
    y0, y1, y2 = uncoupled_equilibrium(P0, A0, B0)
    node = [y0, y1, y2, 0.0, 0.0, 0.0]
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    np.testing.assert_allclose(x[:6], node, rtol=1e-9, atol=1e-10)     # helper agrees
    np.testing.assert_allclose(model.steady_state(P0, A0, B0, cfg), node, rtol=1e-9, atol=1e-10)
    s = S(y1 - y2)
    f = ss.drift(x, [s, s], layout, cfg)
    assert np.abs(f).max() < 1e-7        # scale of the terms is ~1e3 (A a S ~ 100); 1e-7 is ~1e-10 relative
    assert np.all(f[12:] == 0.0)
    # buffer fill is the same S
    np.testing.assert_allclose(ss.initial_buffer_fill(cfg), [s, s], rtol=1e-12)


def test_coupled_fixed_point_drift_zero(cfg):
    g12, g21 = 8.0, -5.0

    def eqs(u):
        n1, n2 = u[:3], u[3:]
        s1, s2 = S(n1[1] - n1[2]), S(n2[1] - n2[2])
        out = []
        for (y0, y1, y2), g_in, s_src in ((n1, g21, s2), (n2, g12, s1)):
            out += [y0 - (A0 / a) * S(y1 - y2),
                    y1 - (A0 / a) * (P0 + C2 * S(C1 * y0) + g_in * s_src),
                    y2 - (B0 / b) * C4 * S(C3 * y0)]
        return out
    u0 = uncoupled_equilibrium(P0, A0, B0)
    sol, _, ier, msg = fsolve(eqs, np.concatenate([u0, u0]), xtol=1e-14, full_output=True)
    assert ier == 1, msg
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    x[[0, 1, 2]], x[[6, 7, 8]] = sol[:3], sol[3:]
    x[3:6] = 0.0
    x[9:12] = 0.0
    x[ss.G12], x[ss.G21] = g12, g21
    s = [S(sol[1] - sol[2]), S(sol[4] - sol[5])]
    assert np.abs(ss.drift(x, s, layout, cfg)).max() < 1e-7
    assert abs(sol[3] - u0[0]) > 1e-4      # the coupling really moved the equilibrium


# ---- drift against solve_ivp ------------------------------------------------------------

def perturbed_start(cfg):
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    x[:12] += np.array([0.03, 1.5, -1.0, 4.0, 60.0, -150.0, -0.02, -1.0, 1.2, -3.0, -40.0, 90.0])
    return layout, x


def test_predict_matches_solve_ivp_with_heun_order(cfg):
    layout, x = perturbed_start(cfg)
    T = 1.0 / cfg["preprocessing"]["observation_fs_hz"]
    s = ss.initial_buffer_fill(cfg)                     # g = 0: the delayed input has no effect
    ref = solve_ivp(lambda t, y: rhs_full(t, y, P0, 0.0, 0.0, s), (0, T), x[:12],
                    method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    out = {}
    for n in (4, 8, 16):
        buf = ss.make_buffer(cfg)
        out[n] = ss.predict(x[None], [1.0], buf, layout, cfg, n_substeps=n)[0, :12]
    err = {n: np.abs(out[n] - ref).max() for n in out}
    order_1 = math.log2(err[4] / err[8])
    order_2 = math.log2(err[8] / err[16])

    assert 1.8 <= order_1 <= 2.2 and 1.8 <= order_2 <= 2.2, (err, order_1, order_2)
    # Richardson estimate of the 4-sub-step error for a second-order method: (4/3) |x4 - x8|
    richardson = 4.0 / 3.0 * np.abs(out[4] - out[8]).max()
    assert err[4] <= 1.5 * richardson, (err, richardson)
    assert err[4] < 1e-2 * np.abs(x[:12]).max()          # and small in absolute terms
    # parameters do not move in predict
    buf = ss.make_buffer(cfg)
    assert np.array_equal(ss.predict(x[None], [1.0], buf, layout, cfg)[0, 12:], x[12:])


# ---- delay, direction, gain sign ----------------------------------------------------------

def test_drift_coupling_direction_and_gain_sign(cfg):
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    s = np.array([2.0, 3.0])
    for g12, g21 in ((7.0, -3.0), (-7.0, 3.0), (7.0, 0.0), (0.0, -3.0)):
        x[ss.G12], x[ss.G21] = g12, g21
        d = ss.drift(x, s, layout, cfg)
        d0 = ss.drift(x, [0.0, 0.0], layout, cfg)
        delta = d - d0
        expect = np.zeros(19)
        expect[6 + 4] = A0 * a * g12 * s[0]          # node 1 -> node 2, into node 2's y4 (index 10)
        expect[4] = A0 * a * g21 * s[1]              # node 2 -> node 1, into node 1's y4
        np.testing.assert_allclose(delta, expect, rtol=1e-12, atol=1e-9)


def test_delay_reads_exactly_ten_substeps(cfg, monkeypatch):
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    x[ss.G12], x[ss.G21] = 5.0, -4.0
    delay = cfg["coupling"]["delay_substeps"]
    assert delay == 10
    buf = ss.make_buffer(cfg)
    impulse = 1000.0
    buf.replace_latest([impulse, impulse + 1.0])       # entry of sub-step 0: node 1 = 1000, node 2 = 1001
    calls = []
    real = ss.drift

    def spy(X, s_delayed, layout_, cfg_):
        calls.append(np.array(s_delayed, dtype=float))
        return real(X, s_delayed, layout_, cfg_)
    monkeypatch.setattr(ss, "drift", spy)
    X = x[None]
    for _ in range(3):                                  # 12 sub-steps
        X = ss.predict(X, [1.0], buf, layout, cfg)
    assert len(calls) == 24
    # sub-step j: stage 1 is call 2j (entry j - 10), stage 2 is call 2j + 1 (entry j - 9)
    hit = [k for k, s in enumerate(calls) if s[0] == impulse]
    assert hit == [2 * 9 + 1, 2 * 10]      # stage 2 of sub-step 9, stage 1 of sub-step 10, and no others
    hit2 = [k for k, s in enumerate(calls) if s[1] == impulse + 1.0]
    assert hit2 == hit
    # the impulse was in node 1's slot only: node 1's slot carries 1000, node 2's carries 1001
    assert all(s[0] != impulse + 1.0 for s in calls)


def test_buffer_read_write_reset_replace():
    buf = ss.DelayBuffer(3, 2)
    buf.reset([1.0, 2.0])
    assert all(np.array_equal(buf.read(l), [1.0, 2.0]) for l in range(4))
    for n in range(1, 6):
        buf.write([n, -n])
    assert [buf.read(l)[0] for l in range(4)] == [5, 4, 3, 2]
    buf.replace_latest([9.0, 9.0])
    assert buf.read(0)[0] == 9.0 and buf.read(1)[0] == 4
    with pytest.raises(ss.StateSpaceError):
        buf.read(4)
    with pytest.raises(ss.StateSpaceError):
        ss.DelayBuffer(0)
    buf.reset([0.5, 0.5])
    assert all(np.array_equal(buf.read(l), [0.5, 0.5]) for l in range(4))


# ---- S of the mean, not the mean of S --------------------------------------------------------

def test_buffer_holds_S_of_mean_not_mean_of_S(cfg):
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    pts = np.tile(x, (2, 1))
    pts[0, 1] += 6.0
    pts[1, 1] -= 6.0                              # node 1 potentials y1 - y2 spread by +-6 mV
    pts[0, 7] += 3.0
    pts[1, 7] -= 3.0
    wm = np.array([0.5, 0.5])
    v = np.stack([pts[:, 1] - pts[:, 2], pts[:, 7] - pts[:, 8]], axis=1)
    s_of_mean = S(wm @ v)
    mean_of_s = wm @ S(v)
    assert np.abs(s_of_mean - mean_of_s).min() > 1e-3        # the two really differ
    np.testing.assert_allclose(ss.sigmoid_of_mean_potential(pts, wm, cfg), s_of_mean, rtol=1e-13)
    buf = ss.make_buffer(cfg)
    out = ss.predict(pts, wm, buf, ss.make_layout(cfg), cfg)
    v_out = ss.potentials(out)
    np.testing.assert_allclose(buf.read(0), S(wm @ v_out), rtol=1e-12)
    assert np.abs(buf.read(0) - wm @ S(v_out)).min() > 1e-6


# ---- predict against the A2 simulator -----------------------------------------------------------

def coupled_setup(cfg):
    layout, x = perturbed_start(cfg)
    x[ss.G12], x[ss.G21] = 6.0, -4.0
    return layout, x


def run_predict(cfg, layout, x, delay, n_sub, n_calls, wm=(1.0,)):
    buf = ss.DelayBuffer(delay)
    buf.reset(ss.initial_buffer_fill(cfg))
    buf.replace_latest(ss.sigmoid_of_mean_potential(x[None], wm, cfg))      # S of the initial state
    X, traj = x[None], []
    for _ in range(n_calls):
        X = ss.predict(X, list(wm), buf, layout, cfg, n_substeps=n_sub)
        traj.append(X[0, :12].copy())
    return np.array(traj)


def a2_states(cfg, x, n_steps):
    res = model.simulate(cfg, n_steps, input=np.full((n_steps, 2), P0), n_nodes=2, p=P0,
                         g12=x[ss.G12], g21=x[ss.G21], y_init=x[:12].reshape(2, 6))
    return res.states.reshape(n_steps + 1, 12)


def test_predict_at_simulator_rate_equals_A2_exactly(cfg):
    """8 sub-steps at 256 Hz = 2048 Hz, delay 20 = the simulator's: identical scheme, identical numbers."""
    layout, x = coupled_setup(cfg)
    n_calls = 6                                       # 48 steps, past the 20-step delay
    traj = run_predict(cfg, layout, x, cfg["coupling"]["sim_delay_steps"], 8, n_calls)
    states = a2_states(cfg, x, 8 * n_calls)
    np.testing.assert_allclose(traj, states[8::8], rtol=1e-9, atol=1e-9)


def test_predict_at_ukf_rate_close_to_A2_within_heun_difference(cfg):
    """4 sub-steps, delay 10 (1024 Hz) against the 2048 Hz simulator with delay 20: same physical
    delay, so the difference is the Heun step-size difference. Bound: 3x the same difference
    without coupling (which contains the pure step-size effect); a one-step delay error at
    these gains would add a term of the size of the coupling drive change, well above it."""
    layout, x = coupled_setup(cfg)
    n_calls = 8
    coupled = run_predict(cfg, layout, x, cfg["coupling"]["delay_substeps"], 4, n_calls)
    ref = a2_states(cfg, x, 8 * n_calls)[8::8]
    diff = np.abs(coupled - ref).max()
    x0 = x.copy()
    x0[ss.G12] = x0[ss.G21] = 0.0
    unc = np.abs(run_predict(cfg, layout, x0, cfg["coupling"]["delay_substeps"], 4, n_calls)
                 - a2_states(cfg, x0, 8 * n_calls)[8::8]).max()

    assert diff <= 3.0 * unc, (diff, unc)


# ---- observation ------------------------------------------------------------------------------

def x_with_potentials(cfg, v1, v2, m):
    layout = ss.make_layout(cfg)
    x = ss.prior_mean(layout, cfg)
    x[:12] = 0.0
    x[1], x[7] = v1, v2                    # y1 - y2 = v (y2 = 0)
    x[ss.M] = m
    return layout, x


def test_observation_symmetric_unit_diagonal_and_dc(cfg):
    mu = cfg["rescaling"]["mu_ref"]
    for m in (0.0, 0.1, 0.2, 0.5):
        layout, x = x_with_potentials(cfg, mu + 1.0, mu + 0.0, m)
        y = ss.observe(x, layout, cfg)
        # M = [[1, m], [m, 1]] on the deviations (1, 0): column 0 of M
        np.testing.assert_allclose(y - mu, [1.0, m], atol=1e-12)
        layout, x = x_with_potentials(cfg, mu + 0.0, mu + 1.0, m)
        np.testing.assert_allclose(ss.observe(x, layout, cfg) - mu, [m, 1.0], atol=1e-12)   # symmetric
    for m in (-0.3, 0.0, 0.2, 0.5, 0.9, 5.0):
        layout, x = x_with_potentials(cfg, mu, mu, m)
        np.testing.assert_allclose(ss.observe(x, layout, cfg), [mu, mu], atol=1e-12)        # DC kept


def test_observation_clips_m_inside_observe_only(cfg):
    mu = cfg["rescaling"]["mu_ref"]
    for m_raw, m_used in ((-0.3, 0.0), (0.9, 0.5), (0.3, 0.3)):
        layout, x = x_with_potentials(cfg, mu + 2.0, mu - 1.0, m_raw)
        expect = mu + np.array([2.0 + m_used * -1.0, m_used * 2.0 - 1.0])
        np.testing.assert_allclose(ss.observe(x, layout, cfg), expect, atol=1e-12)
        assert x[ss.M] == m_raw                       # the state itself is not clipped
    assert cfg["priors"]["m_truncate"] == [0.0, 0.5]


def test_observation_batched_and_m1(cfg):
    m1 = ss.make_layout(cfg, include_gains=False)
    X = np.tile(ss.prior_mean(m1, cfg), (4, 1))
    assert ss.observe(X, m1, cfg).shape == (4, 2)


# ---- sigma-point arithmetic (no filterpy, no sigma-point code in src) ----------------------------

def test_sigma_point_arithmetic_n19(cfg):
    sp = cfg["ukf"]["sigma_points"]
    n = ss.make_layout(cfg).n
    assert n == 19
    lam = sp["alpha"] ** 2 * (n + sp["kappa"]) - n
    wm0 = lam / (n + lam)
    wmi = 1.0 / (2.0 * (n + lam))
    wc0 = wm0 + (1.0 - sp["alpha"] ** 2 + sp["beta"])
    assert lam == 0.0 and wm0 == 0.0
    assert wmi == pytest.approx(1.0 / 38.0, rel=1e-15)
    assert wc0 == 2.0
    assert math.sqrt(n + lam) == pytest.approx(math.sqrt(19.0))
    assert wm0 + 2 * n * wmi == pytest.approx(1.0, rel=1e-14)
    assert wmi > 0                                     # no negative weights


# ---- config hygiene ----------------------------------------------------------------------------

def test_new_config_leaves_are_tagged():
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["ukf"]["initial_state"]
    assert raw["covariance_seed"]["prov"] == "placeholder" and raw["covariance_seed"]["value"] == 42
    assert raw["neural_variance"]["prov"] == "computed"
    assert raw["neural_variance_provenance"]["prov"] == "computed"
    for leaf in ("covariance_seed", "neural_variance", "neural_variance_provenance"):
        assert raw[leaf]["ref"]


def test_no_numeric_parameter_literals_in_module():
    tree = ast.parse((REPO_ROOT / "src" / "state_space.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2, 0.5, 0.0, 1.0}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool) and node.value not in allowed:
            bad.append((node.lineno, node.value))
    assert not bad, bad


def test_module_imports_no_filter_or_preprocess():
    path = REPO_ROOT / "src" / "state_space.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {n.name.split(".")[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
            imported |= {n.name for n in node.names if node.module == "src"}
    assert not imported & {"filterpy", "preprocess", "scipy"}, imported
    assert not re.search(r"print\(", path.read_text(encoding="utf-8"))


# ---- writer and guard ----------------------------------------------------------------------------

def test_writer_changes_only_intended_lines(tmp_path):
    src = DEFAULT_CONFIG_PATH.read_bytes()
    path = tmp_path / "config.yml"
    path.write_bytes(src)
    before = path.read_bytes().decode().splitlines(keepends=True)
    if load_config(path)["ukf"]["initial_state"]["neural_variance"] is not None:
        pytest.skip("neural_variance already written; the writer refuses without force")
    var = [1e-4, 0.5, 1.5, 2.0, 400.0, 6500.0]
    ss.write_initial_covariance(path, var, {"seed": 1})
    after = path.read_bytes().decode().splitlines(keepends=True)
    changed = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
    assert len(before) == len(after) and len(changed) == 2
    c = load_config(path)["ukf"]["initial_state"]
    assert c["neural_variance"] == var and all(type(v) is float for v in c["neural_variance"])
    assert c["neural_variance_provenance"] == {"seed": 1}
    with pytest.raises(ss.StateSpaceError):
        ss.write_initial_covariance(path, var, {"seed": 2})


def test_recompute_deterministic_and_short_run_positive(cfg):
    c = copy.deepcopy(cfg)
    c["ukf"]["initial_state"]["covariance_sim_duration_s"] = 5
    r1 = ss.neural_variance_from_simulation(c)
    r2 = ss.neural_variance_from_simulation(c)
    assert np.array_equal(r1["variance"], r2["variance"])
    assert np.all(r1["variance"] > 0) and np.all(np.isfinite(r1["variance"]))
    assert r1["variance"].shape == (6,) and r1["per_node"].shape == (2, 6)


def test_stored_neural_variance_matches_code():
    """Guard: the stored variances are what the code produces (skips while null)."""
    cfg = load_config()
    stored = cfg["ukf"]["initial_state"]["neural_variance"]
    if stored is None:
        pytest.skip("neural_variance is still null (written in the second C1 commit)")
    assert len(stored) == 6 and all(type(v) is float and v > 0 and math.isfinite(v) for v in stored)
    result = ss.neural_variance_from_simulation(cfg)
    assert np.array_equal(result["variance"], np.asarray(stored))
    prov = cfg["ukf"]["initial_state"]["neural_variance_provenance"]
    assert prov["seed"] == 42 and prov["kept_duration_s"] == 600 and prov["burn_in_s"] == 10
    assert prov["ddof"] == 1 and prov["sim_fs_hz"] == 2048
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"])


# ---- C2b: guard, per-point parameters, buffer after replace_latest ----------------------------------------

def test_delay_guard_needs_delay_at_least_substeps(cfg):
    n_sub = cfg["ukf"]["substeps_per_observation"]
    ok = copy.deepcopy(cfg)
    ok["coupling"]["delay_substeps"] = n_sub                       # equal is allowed
    ss.make_layout(ok)
    ss.make_buffer(ok)
    bad = copy.deepcopy(cfg)
    bad["coupling"]["delay_substeps"] = n_sub - 1
    with pytest.raises(ss.StateSpaceError, match="delay_substeps"):
        ss.make_layout(bad)
    with pytest.raises(ss.StateSpaceError, match="delay_substeps"):
        ss.make_buffer(bad)
    bad2 = copy.deepcopy(cfg)
    bad2["ukf"]["substeps_per_observation"] = cfg["coupling"]["delay_substeps"] + 1
    with pytest.raises(ss.StateSpaceError):
        ss.make_layout(bad2)


def test_drift_uses_each_sigma_points_own_parameters(cfg):
    layout = ss.make_layout(cfg)
    rng = np.random.default_rng(8)
    n_pts = 7
    X = np.tile(ss.prior_mean(layout, cfg), (n_pts, 1))
    X[:, :12] += rng.normal(size=(n_pts, 12)) * np.array([0.01, 0.2, 1.0, 1.0, 20.0, 60.0] * 2)
    X[:, ss.P1] = rng.uniform(150, 300, n_pts)
    X[:, ss.P2] = rng.uniform(150, 300, n_pts)
    X[:, ss.LOG_RHO1] = rng.normal(math.log(3.25 / 22), 0.2, n_pts)
    X[:, ss.LOG_RHO2] = rng.normal(math.log(3.25 / 22), 0.2, n_pts)
    X[:, ss.G12] = rng.normal(0, 10, n_pts)
    X[:, ss.G21] = rng.normal(0, 10, n_pts)
    s = rng.uniform(0.5, 4.0, size=(n_pts, 2))                     # a different delayed S per point too
    batched = ss.drift(X, s, layout, cfg)
    ab = cfg["priors"]["AB_product"]
    for i in range(n_pts):
        np.testing.assert_allclose(batched[i], ss.drift(X[i], s[i], layout, cfg), rtol=1e-13, atol=1e-13)
        expected = np.zeros(19)
        for j, (p, lr, g_in, s_src) in enumerate(((X[i, ss.P1], X[i, ss.LOG_RHO1], X[i, ss.G21], s[i, 1]),
                                                  (X[i, ss.P2], X[i, ss.LOG_RHO2], X[i, ss.G12], s[i, 0]))):
            rho = math.exp(lr)
            expected[6 * j:6 * j + 6] = rhs_node(X[i, 6 * j:6 * j + 6], p, g_in * s_src,
                                                 math.sqrt(ab * rho), math.sqrt(ab / rho))
        np.testing.assert_allclose(batched[i], expected, rtol=1e-10, atol=1e-9)
    # the rows really differ from each other (a "row 0 for all points" bug would be visible)
    assert np.abs(batched[1:, :12] - batched[0, :12]).max() > 1.0


def test_replace_latest_touches_only_the_newest_entry():
    buf = ss.DelayBuffer(10, 2)
    buf.reset([1.0, 1.0])
    for n in range(1, 5):                                          # four sub-steps of one observation step
        buf.write([n, -n])
    before = [buf.read(lag).copy() for lag in range(11)]
    buf.replace_latest([99.0, -99.0])
    for lag in range(1, 11):
        assert np.array_equal(buf.read(lag), before[lag]), lag
    assert np.array_equal(buf.read(0), [99.0, -99.0]) and not np.array_equal(before[0], buf.read(0))
    assert [buf.read(l)[0] for l in (1, 2, 3, 4)] == [3, 2, 1, 1.0]   # predicted entries 3, 2, 1, then the fill
