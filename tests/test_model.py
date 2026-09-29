import ast
import copy
import math

import numpy as np
import pytest

from conftest import REPO_ROOT
from src import model
from src.config import ConfigError, load_config

CFG = load_config(REPO_ROOT / "config.yml")
K = model.constants(CFG)
FS = CFG["rescaling"]["reference_simulation"]["sim_fs_hz"]
DELAY = CFG["coupling"]["sim_delay_steps"]


def _steady_s(p=220.0):
    s = model.steady_state(p, CFG["jansen_rit"]["A"], CFG["jansen_rit"]["B"], CFG)
    return float(model.sigmoid(s[model.Y1] - s[model.Y2], K["e0"], K["v0"], K["r"]))


# ---- sigmoid (§7.3) ----------------------------------------------------------

def test_sigmoid_midpoint_limits_and_no_offset():
    e0, v0, r = K["e0"], K["v0"], K["r"]
    assert model.sigmoid(v0, e0, v0, r) == 2.5           # 5 / (1 + e^0) = 2.5
    assert model.sigmoid(1.0e3, e0, v0, r) == pytest.approx(5.0, rel=1e-12)
    assert model.sigmoid(-1.0e3, e0, v0, r) == pytest.approx(0.0, abs=1e-12)
    # S(0) = 5 / (1 + exp(0.56 * 6)) = 5 / (1 + e^3.36), e^3.36 = 28.789190879242675
    hand = 5.0 / (1.0 + 28.789190879242675)
    assert hand == pytest.approx(0.1678461164, rel=1e-9)
    assert model.sigmoid(0.0, e0, v0, r) == pytest.approx(hand, rel=1e-12)
    assert model.sigmoid(0.0, e0, v0, r) != 0.0


# ---- right-hand side, hand-computed (§7.1) -----------------------------------

# State (y0..y5) = (0.1, 2.0, 1.0, 5.0, -3.0, 4.0), p = 220, coupling drive = 7,
# literature A = 3.25, B = 22, a = 100, b = 50, C1 = 135, C2 = 108, C3 = C4 = 33.75.
#   S(y1 - y2 = 1.0)     = 5 / (1 + exp(0.56 * (6 - 1.0)))     = 5 / (1 + e^2.8)
#   S(C1 y0 = 13.5)      = 5 / (1 + exp(0.56 * (6 - 13.5)))    = 5 / (1 + e^-4.2)
#   S(C3 y0 = 3.375)     = 5 / (1 + exp(0.56 * (6 - 3.375)))   = 5 / (1 + e^1.47)
Y_HAND = [0.1, 2.0, 1.0, 5.0, -3.0, 4.0]
S_A = 5.0 / (1.0 + 16.444646771097048)      # e^2.8
S_B = 5.0 / (1.0 + 0.014995576820477703)    # e^-4.2
S_C = 5.0 / (1.0 + 4.349235141062741)       # e^1.47


def _expected_hand(drive):
    return [
        5.0,                                                          # y0' = y3
        -3.0,                                                         # y1' = y4
        4.0,                                                          # y2' = y5
        # y3' = A a S(y1-y2) - 2a y3 - a^2 y0 = 325 S_A - 200*5 - 10000*0.1
        325.0 * S_A - 1000.0 - 1000.0,
        # y4' = A a [p + C2 S(C1 y0) + drive] - 2a y4 - a^2 y1
        #     = 325 (220 + 108 S_B + drive) + 200*3 - 10000*2
        325.0 * (220.0 + 108.0 * S_B + drive) + 600.0 - 20000.0,
        # y5' = B b C4 S(C3 y0) - 2b y5 - b^2 y2 = 22*50*33.75 S_C - 100*4 - 2500*1
        37125.0 * S_C - 400.0 - 2500.0,
    ]


def test_rhs_matches_hand_computation_term_by_term():
    got = model.jr_rhs(Y_HAND, 220.0, 7.0, 3.25, 22.0, CFG)
    want = _expected_hand(7.0)
    # cross-checked once with plain math: -1906.848214..., 227282.157430..., 31801.222718...
    assert want[3] == pytest.approx(-1906.8482141643, rel=1e-9)
    assert want[4] == pytest.approx(227282.1574378, rel=1e-9)
    assert want[5] == pytest.approx(31801.22271782, rel=1e-9)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_coupling_enters_y4_bracket_alongside_p():
    with_drive = model.jr_rhs(Y_HAND, 220.0, 7.0, 3.25, 22.0, CFG)
    without = model.jr_rhs(Y_HAND, 220.0, 0.0, 3.25, 22.0, CFG)
    diff = with_drive - without
    assert diff[4] == pytest.approx(325.0 * 7.0, rel=1e-12)   # A a * drive, same slot as p
    assert np.all(diff[[0, 1, 2, 3, 5]] == 0.0)
    more_p = model.jr_rhs(Y_HAND, 227.0, 0.0, 3.25, 22.0, CFG)   # p and drive are interchangeable
    assert more_p[4] == pytest.approx(with_drive[4], rel=1e-12)


# ---- steady state ------------------------------------------------------------

def test_steady_state_single_root_is_a_fixed_point():
    s = model.steady_state(220.0, 3.25, 22.0, CFG)
    d = model.jr_rhs(s, 220.0, 0.0, 3.25, 22.0, CFG)
    assert np.max(np.abs(d)) < 1e-7
    assert np.all(s[3:] == 0.0)


def test_steady_state_raises_when_several_roots():
    # Three fixed points exist at p = 0 with the literature A and B (found by a
    # scratch scan; also at p = 50). One root at p = 220.
    with pytest.raises(model.ModelError, match="3 roots"):
        model.steady_state(0.0, 3.25, 22.0, CFG)


def test_grid_root_count_does_not_double_count_an_exact_zero():
    count = model._count_grid_roots
    assert count(np.array([1.0, 0.0, -1.0])) == 1         # zero between a + and a - side
    assert count(np.array([-2.0, 0.0, 3.0, 4.0])) == 1
    assert count(np.array([1.0, -1.0, -2.0])) == 1        # plain crossing, no zero
    assert count(np.array([1.0, -1.0, 0.0, -1.0])) == 2   # crossing plus a separate zero
    assert count(np.array([1.0, 1.0, 1.0])) == 0


# ---- integrator: independence, delay, determinism ----------------------------

def test_uncoupled_two_node_equals_two_single_nodes():
    u = model.draw_input(np.random.default_rng(7), 4096, 2, 220.0, CFG)
    two = model.simulate(CFG, 4096, input=u, n_nodes=2, g12=0.0, g21=0.0)
    for k in range(2):
        one = model.simulate(CFG, 4096, input=np.ascontiguousarray(u[:, [k]]), n_nodes=1)
        assert np.array_equal(two.states[:, k, :], one.states[:, 0, :])


def test_prefilled_buffer_gives_steady_drive_before_d():
    g12, g21 = 10.0, -6.0
    r = model.simulate(CFG, 4 * DELAY, seed=3, g12=g12, g21=g21)
    s = _steady_s()
    np.testing.assert_allclose(r.drive[:DELAY, 1], g12 * s, rtol=1e-14, atol=0)
    np.testing.assert_allclose(r.drive[:DELAY, 0], g21 * s, rtol=1e-14, atol=0)


@pytest.mark.parametrize("delay", [DELAY, 5, 1])
def test_delayed_coupling_reads_exactly_d_steps_back(delay):
    cfg = copy.deepcopy(CFG)
    cfg["coupling"]["sim_delay_steps"] = delay
    n, k0 = 200, 30
    base = np.full((n, 2), 220.0)
    pulse = base.copy()
    pulse[k0, 0] += 1000.0
    ref = model.simulate(cfg, n, input=base, g12=10.0, g21=0.0)
    hit = model.simulate(cfg, n, input=pulse, g12=10.0, g21=0.0)
    # Node 1's state first differs at step k0 + 1, so its S does too; node 2's stage-1
    # drive at step k reads S of step k - d, so it first differs at k0 + 1 + d.
    first = int(np.flatnonzero(hit.drive[:, 1] != ref.drive[:, 1])[0])
    assert first == k0 + 1 + delay
    assert np.array_equal(hit.drive[:first, 1], ref.drive[:first, 1])


# ---- independent two-node Heun reference (plain math, no numpy in the stepping) -------
#
# Derivation of the stage-2 delayed read. Let S_k = S(y1 - y2) of the state at step k, and
# d the delay in steps; S_k for k < 0 is the steady-state fill. Heun advances y_k -> y_{k+1}:
#   stage 1 evaluates the RHS at t_k,      with delayed drive g * S(t_k - d dt)     = g * S_{k-d}
#   stage 2 evaluates the RHS at t_{k+1},  with delayed drive g * S(t_{k+1} - d dt) = g * S_{k+1-d}
# so the stage-2 index is the stage-1 index plus one. Since d >= 1, k + 1 - d <= k and S_{k+1-d}
# is already known when step k is taken: the scheme stays explicit. Reading S_{k-d} in stage 2
# as well would make the second stage a stale copy of the first.

_RJ = CFG["jansen_rit"]
_C = _RJ["C"]
_a, _b = float(_RJ["a"]), float(_RJ["b"])
_C1, _C2 = float(_C * _RJ["C1_multiplier"]), float(_C * _RJ["C2_multiplier"])
_C3, _C4 = float(_C * _RJ["C3_multiplier"]), float(_C * _RJ["C4_multiplier"])
_e0, _v0, _r = float(_RJ["e0"]), float(_RJ["v0"]), float(_RJ["r"])


def _ref_sig(v):
    return 2.0 * _e0 / (1.0 + math.exp(_r * (_v0 - v)))


def _ref_rhs(y, p_in, drive, A, B):
    return [
        y[3],
        y[4],
        y[5],
        A * _a * _ref_sig(y[1] - y[2]) - 2.0 * _a * y[3] - _a * _a * y[0],
        A * _a * (p_in + _C2 * _ref_sig(_C1 * y[0]) + drive) - 2.0 * _a * y[4] - _a * _a * y[1],
        B * _b * _C4 * _ref_sig(_C3 * y[0]) - 2.0 * _b * y[5] - _b * _b * y[2],
    ]


def _ref_steady(p, A, B):
    """Root of y0 = (A/a) S(y1 - y2) by plain bisection (single root at p = 220)."""
    def res(y0):
        y1 = (A / _a) * (p + _C2 * _ref_sig(_C1 * y0))
        y2 = (B / _b) * _C4 * _ref_sig(_C3 * y0)
        return y0 - (A / _a) * _ref_sig(y1 - y2), y1, y2
    lo, hi = 0.0, 2.0 * _e0 * A / _a
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if res(mid)[0] * res(lo)[0] < 0.0:
            hi = mid
        else:
            lo = mid
    y0 = 0.5 * (lo + hi)
    _, y1, y2 = res(y0)
    return [y0, y1, y2, 0.0, 0.0, 0.0]


def _ref_simulate(u, g12, g21, delay, p, A, B):
    """u: list of [u_node1, u_node2] per step. Returns states[k][node] = six floats."""
    dt = 1.0 / FS
    y = [_ref_steady(p, A, B), _ref_steady(p, A, B)]
    s_fill = _ref_sig(y[0][1] - y[0][2])
    states = [[list(y[0]), list(y[1])]]
    s_hist = []
    gain = [[0.0, g21], [g12, 0.0]]   # gain[j][i]: from node i to node j (g12: 1 -> 2)

    def s_at(step):
        return s_fill if step < 0 else s_hist[step]

    for k, u_k in enumerate(u):
        s_hist.append([_ref_sig(n[1] - n[2]) for n in states[k]])
        s_hist[k] = tuple(s_hist[k])
        new = []
        for j in range(2):
            d1 = sum(gain[j][i] * (s_fill if k - delay < 0 else s_hist[k - delay][i]) for i in range(2))
            d2 = sum(gain[j][i] * (s_fill if k + 1 - delay < 0 else s_hist[k + 1 - delay][i])
                     for i in range(2))
            y_k = states[k][j]
            k1 = _ref_rhs(y_k, u_k[j], d1, A, B)
            y_pred = [y_k[m] + dt * k1[m] for m in range(6)]
            k2 = _ref_rhs(y_pred, u_k[j], d2, A, B)
            new.append([y_k[m] + 0.5 * dt * (k1[m] + k2[m]) for m in range(6)])
        states.append(new)
    return states


def _ref_inputs(n_steps, seed=21):
    rng = np.random.default_rng(seed)
    return rng.uniform(120.0, 320.0, size=(n_steps, 2))


def test_two_node_heun_matches_independent_reference_including_stage2_delayed_read():
    n_steps, delay, g12, g21 = 50, 3, 10.0, -6.0
    cfg = copy.deepcopy(CFG)
    cfg["coupling"]["sim_delay_steps"] = delay
    u = _ref_inputs(n_steps)
    real = model.simulate(cfg, n_steps, input=u, g12=g12, g21=g21)
    ref = _ref_simulate(u.tolist(), g12, g21, delay, _RJ["p_mean"], _RJ["A"], _RJ["B"])
    want = np.array(ref, dtype=np.float64)
    assert want.shape == real.states.shape
    np.testing.assert_allclose(real.states, want, rtol=1e-12, atol=0.0)


def test_coupling_reaches_node2_y4_at_the_step_the_reference_predicts():
    # The pulse is in node 1's input at step k0. Node 1's S first differs at step k0 + 1.
    # Node 2's stage-2 read at step k is S_{k+1-d}, so it first differs at k = k0 + d and
    # states[k0 + d + 1] is the first row to deviate; only y4 is fed directly (stage 1 is
    # still unperturbed there, and y1' = y4 of the predictor).
    n_steps, delay, k0, g12 = 40, 3, 10, 10.0
    cfg = copy.deepcopy(CFG)
    cfg["coupling"]["sim_delay_steps"] = delay
    base = _ref_inputs(n_steps)
    pulse = base.copy()
    pulse[k0, 0] += 1000.0

    def first_dev_ref(a, b):
        for k in range(len(a)):
            if a[k][1][4] != b[k][1][4]:
                return k
        return None

    args = (delay, _RJ["p_mean"], _RJ["A"], _RJ["B"])
    r0 = _ref_simulate(base.tolist(), g12, 0.0, *args)
    r1 = _ref_simulate(pulse.tolist(), g12, 0.0, *args)
    predicted = first_dev_ref(r0, r1)
    assert predicted == k0 + delay + 1
    # The baseline is the coupled run without the pulse: with g12 != 0 the pre-filled
    # history already drives node 2 (g12 * S_steady), so it differs from a g = 0 run from
    # step 1 on. Node 1 is the same in both, so the pulse response is what separates them.
    quiet = model.simulate(cfg, n_steps, input=base, g12=g12, g21=0.0)
    hit = model.simulate(cfg, n_steps, input=pulse, g12=g12, g21=0.0)
    dev = np.flatnonzero(hit.states[:, 1, model.Y4] != quiet.states[:, 1, model.Y4])
    assert int(dev[0]) == predicted
    # and with g12 = 0 the pulse never reaches node 2
    off_q = model.simulate(cfg, n_steps, input=base, g12=0.0, g21=0.0)
    off_h = model.simulate(cfg, n_steps, input=pulse, g12=0.0, g21=0.0)
    assert np.array_equal(off_h.states[:, 1], off_q.states[:, 1])


# ---- config guards and consistency ---------------------------------------------

@pytest.mark.parametrize("key, bad", [("integrator", "rk4"), ("noise_redraw_interval", "every 10 steps")])
def test_simulate_rejects_unimplemented_integrator_or_redraw_interval(key, bad):
    cfg = copy.deepcopy(CFG)
    cfg["rescaling"]["reference_simulation"][key] = bad
    with pytest.raises(ConfigError, match=key):
        model.simulate(cfg, 4, seed=1)


def test_reference_sim_rate_equals_generation_rate():
    assert (CFG["rescaling"]["reference_simulation"]["sim_fs_hz"]
            == CFG["g0"]["generation_fs_hz"])


def test_same_seed_bit_identical_different_seed_differs():
    a = model.simulate(CFG, 2048, seed=11, g12=5.0, g21=5.0)
    b = model.simulate(CFG, 2048, seed=11, g12=5.0, g21=5.0)
    c = model.simulate(CFG, 2048, seed=12, g12=5.0, g21=5.0)
    assert np.array_equal(a.states, b.states)
    assert not np.array_equal(a.states, c.states)


def test_input_is_uniform_p_centred_with_optional_half_width():
    rng = np.random.default_rng(0)
    u = model.draw_input(rng, 20000, 2, 220.0, CFG)
    assert u.dtype == np.float64 and u.min() >= 120.0 and u.max() <= 320.0
    assert u.mean() == pytest.approx(220.0, abs=2.0)
    w = model.draw_input(np.random.default_rng(0), 20000, 2, 150.0, CFG, half_width=10.0)
    assert w.min() >= 140.0 and w.max() <= 160.0


def test_simulate_requires_exactly_one_of_seed_or_input():
    with pytest.raises(model.ModelError):
        model.simulate(CFG, 10)
    with pytest.raises(model.ModelError):
        model.simulate(CFG, 10, seed=1, input=np.zeros((10, 2)))


def test_numba_kernel_agrees_with_pure_python():
    n = 2 * FS  # 2 s
    fast = model.simulate(CFG, n, seed=5, g12=8.0, g21=-4.0)
    slow = model.simulate(CFG, n, seed=5, g12=8.0, g21=-4.0,
                          kernel=model._simulate_kernel.py_func)
    np.testing.assert_allclose(fast.states, slow.states, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fast.drive, slow.drive, rtol=1e-9, atol=1e-12)


def test_600s_run_is_finite():
    n = CFG["rescaling"]["reference_simulation"]["duration_s"] * FS
    r = model.simulate(CFG, n, seed=CFG["rescaling"]["reference_simulation"]["seed"],
                       g12=5.0, g21=5.0)
    assert r.states.dtype == np.float64
    assert np.all(np.isfinite(r.states)) and np.all(np.isfinite(r.drive))


# ---- hygiene -----------------------------------------------------------------

def test_no_numeric_literals_beyond_allowed_in_model():
    tree = ast.parse((REPO_ROOT / "src" / "model.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2, 0.5}
    bad = [(n.value, n.lineno) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, f"numeric literals in src/model.py: {bad}"


def test_model_imports_no_scipy_and_preprocess_never_imports_model():
    def imported(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                names |= {n.module or ""} | {f"{n.module}.{a.name}" for a in n.names}
        return names

    assert not any(m.split(".")[0] == "scipy"
                   for m in imported(REPO_ROOT / "src" / "model.py"))
    pre = REPO_ROOT / "src" / "preprocess.py"
    if pre.exists():
        assert not any(m.split(".")[-1] == "model" for m in imported(pre))


# ---- sanity report (not a pass rule beyond the 4-16 Hz stop condition) --------

def test_sanity_report_alpha_peak_at_p220(capsys):
    from scipy.signal import welch
    ref = CFG["rescaling"]["reference_simulation"]
    burn = ref["burn_in_s"] * FS
    n = burn + 120 * FS
    r = model.simulate(CFG, n, seed=ref["seed"], n_nodes=1, p=220.0)
    y = r.output[burn:, 0]
    seg = int(CFG["simulator"]["sanity_welch_segment_s"] * FS)
    f, pxx = welch(y - y.mean(), fs=FS, nperseg=seg)
    band = (f >= 0.5) & (f <= 45.0)
    peak = float(f[band][np.argmax(pxx[band])])
    order = np.argsort(pxx[band])[::-1][:5]
    with capsys.disabled():
        print(f"\n[sanity] single node, p=220, seed {ref['seed']}, 120 s after {ref['burn_in_s']} s "
              f"burn-in, Welch {CFG['simulator']['sanity_welch_segment_s']} s segments")
        print(f"[sanity] output mean {y.mean():.4f} mV, SD {y.std():.4f} mV")
        print(f"[sanity] spectral peak {peak:.3f} Hz; top bins (Hz): "
              f"{[round(float(x), 3) for x in f[band][order]]}")
    assert 4.0 <= peak <= 16.0, f"peak {peak} Hz outside 4-16 Hz: STOP and report the spectrum"
