import ast
import copy
import math

import numpy as np
import pytest

from conftest import REPO_ROOT
from src import model
from src.config import load_config

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
