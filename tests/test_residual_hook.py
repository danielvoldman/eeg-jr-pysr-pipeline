"""G0.5 tests: the M3 residual hook (state_space.drift / predict, ukf_resid) and the frozen equation on disk
(regression.FrozenResidual, write/load_frozen_equation); IMP-077. Expected values are computed by hand or by independent
code, never by the function under test."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "c7"))

import sim_data  # noqa: E402
from src import passes, regression, ukf, ukf_ext, ukf_resid  # noqa: E402
from src import state_space as ss  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
Q = 1.0e-2
LAYOUT = ss.make_layout(CFG)
SPEC_A = ukf_ext.Spec("A", s2=(6.0, 9.0), tau=(0.1, 0.2))
Y4 = ss.Y4


class ZeroHook(ss.ResidualHook):
    def value(self, u_tgt, u_src, S_src):
        return np.zeros(len(u_tgt))


class ConstHook(ss.ResidualHook):
    def __init__(self, c):
        self.c = c

    def value(self, u_tgt, u_src, S_src):
        return np.full(len(u_tgt), self.c)


class LinearHook(ss.ResidualHook):
    def value(self, u_tgt, u_src, S_src):
        return np.asarray(u_tgt) + 10.0 * np.asarray(u_src) + 100.0 * np.asarray(S_src)


def series(seconds, seed):
    r = sim_data.make_recording(CFG, [seconds], seed, g12=8.0, g21=2.0, p=(220.0, 250.0), burn_seconds=2.0)
    return np.ascontiguousarray(r["segments"][0].T)


def points(n=5, seed=3):
    rng = np.random.default_rng(seed)
    x0 = ss.prior_mean(LAYOUT, CFG)
    sd = np.sqrt(np.diag(ss.prior_cov(LAYOUT, CFG)))
    return x0 + 0.2 * sd * rng.standard_normal((n, LAYOUT.n))


# ---- drift: where the residual enters ---------------------------------------------------------------

def test_drift_adds_the_residual_to_y4_of_the_target_node_with_the_other_nodes_inputs():
    X = points()
    s = np.array([[0.4, 0.9]] * X.shape[0]) + np.arange(X.shape[0])[:, None] * 0.01
    v = np.array([[3.0, -2.0]] * X.shape[0]) + np.arange(X.shape[0])[:, None] * 0.1
    base = ss.drift(X, s, LAYOUT, CFG)
    got = ss.drift(X, s, LAYOUT, CFG, LinearHook(), v)
    pot = np.stack([X[:, 1] - X[:, 2], X[:, 7] - X[:, 8]], axis=1)       # y1 - y2 of node 1, node 2 (hand-indexed)
    exp0 = pot[:, 0] + 10.0 * v[:, 1] + 100.0 * s[:, 1]                    # node 1 is driven by node 2
    exp1 = pot[:, 1] + 10.0 * v[:, 0] + 100.0 * s[:, 0]
    diff = got - base
    np.testing.assert_allclose(diff[:, 4], exp0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(diff[:, 10], exp1, rtol=1e-12, atol=1e-12)
    other = np.delete(diff, [4, 10], axis=1)
    assert np.all(other == 0.0)


def test_drift_without_residual_is_untouched_and_a_residual_needs_v_delayed():
    X = points()
    s = np.array([0.4, 0.9])
    assert np.array_equal(ss.drift(X, s, LAYOUT, CFG), ss.drift(X, s, LAYOUT, CFG, None, None))
    with pytest.raises(ss.StateSpaceError):
        ss.drift(X, s, LAYOUT, CFG, ZeroHook())


# ---- predict ------------------------------------------------------------------------------------------

def _wm(n):
    return ukf.weights_for(n, CFG)[1]


def test_zero_residual_predict_is_bit_identical_to_no_residual():
    X = points(2 * LAYOUT.n + 1)
    wm = _wm(LAYOUT.n)
    b1, b2 = ss.make_buffer(CFG), ss.make_buffer(CFG)
    hook = ZeroHook().bind(ss.make_potential_buffer(CFG))
    a = ss.predict(X, wm, b1, LAYOUT, CFG)
    b = ss.predict(X, wm, b2, LAYOUT, CFG, residual=hook)
    assert np.array_equal(a, b) and np.array_equal(b1._buf, b2._buf) and b1._head == b2._head


def test_constant_residual_matches_a_hand_composed_heun_step():
    c = 7.5
    X = points(2 * LAYOUT.n + 1)
    wm = _wm(LAYOUT.n)
    buf = ss.make_buffer(CFG)
    hook = ConstHook(c).bind(ss.make_potential_buffer(CFG))
    got = ss.predict(X, wm, buf, LAYOUT, CFG, residual=hook)
    # independent composition: the plain drift plus c on both y4 columns, Heun by hand on a second buffer
    dt, n_sub = ss.substep_dt(CFG)
    ref_buf = ss.make_buffer(CFG)
    add = np.zeros(LAYOUT.n)
    add[[4, 10]] = c
    Y = X.copy()
    for _ in range(n_sub):
        k1 = ss.drift(Y, ref_buf.read(ref_buf.delay), LAYOUT, CFG) + add
        k2 = ss.drift(Y + dt * k1, ref_buf.read(ref_buf.delay - 1), LAYOUT, CFG) + add
        Y = Y + 0.5 * dt * (k1 + k2)
        ref_buf.write(ss.sigmoid_of_mean_potential(Y, wm, CFG))
    np.testing.assert_allclose(got, Y, rtol=1e-12, atol=1e-12)
    assert np.array_equal(buf._buf, ref_buf._buf)


def test_potential_ring_holds_the_weighted_mean_potential_of_each_substep():
    X = points(2 * LAYOUT.n + 1)
    wm = _wm(LAYOUT.n)
    hook = ZeroHook().bind(ss.make_potential_buffer(CFG))
    fill = hook.pots.read(0).copy()
    out = ss.predict(X, wm, ss.make_buffer(CFG), LAYOUT, CFG, n_substeps=1, residual=hook)
    pot = np.stack([out[:, 1] - out[:, 2], out[:, 7] - out[:, 8]], axis=1)
    np.testing.assert_allclose(hook.pots.read(0), wm @ pot, rtol=1e-12, atol=1e-12)
    assert np.array_equal(hook.pots.read(1), fill)


def test_potential_buffer_fill_is_the_steady_state_potential():
    s = ss.initial_node_state(CFG)
    buf = ss.make_potential_buffer(CFG)
    assert buf.delay == CFG["coupling"]["delay_substeps"]
    np.testing.assert_allclose(buf.read(0), [s[ss.Y1] - s[ss.Y2]] * 2, rtol=0, atol=0)


def test_residual_without_potential_ring_is_refused():
    with pytest.raises(ss.StateSpaceError):
        ss.predict(points(2 * LAYOUT.n + 1), _wm(LAYOUT.n), ss.make_buffer(CFG), LAYOUT, CFG, residual=ZeroHook())


# ---- the NumPy extended filter ----------------------------------------------------------------------

def test_numpy_filter_without_residual_matches_the_numba_extended_kernel():
    z = series(4.0, 12)
    a = ukf_ext.run_filter_ext(z, CFG, LAYOUT, Q, SPEC_A)
    b = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, SPEC_A)
    assert not a.diverged and not b.diverged and a.n_done == b.n_done
    for name in ("x", "z_pred", "nis", "S"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-8, atol=1e-10, err_msg=name)


def test_numpy_filter_without_extra_state_matches_the_reference_filter():
    z = series(3.0, 15)
    a = ukf.run_filter(z, CFG, LAYOUT, Q, backend="numpy")
    b = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, ukf_ext.Spec("N"))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.z_pred, b.z_pred)


def test_zero_residual_filter_is_bit_identical_and_a_real_residual_changes_the_prediction():
    z = series(3.0, 13)
    base = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, SPEC_A)
    zero = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, SPEC_A, residual=ZeroHook())
    assert np.array_equal(base.x, zero.x) and np.array_equal(base.z_pred, zero.z_pred) and np.array_equal(base.nis, zero.nis)
    big = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, SPEC_A, residual=ConstHook(500.0))
    both = np.isfinite(base.z_pred) & np.isfinite(big.z_pred)
    assert both.any() and not np.allclose(base.z_pred[both], big.z_pred[both], rtol=1e-6, atol=1e-6)


def test_filtered_mean_replaces_the_newest_potential_ring_entry():
    z = series(1.0, 14)
    res = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, Q, SPEC_A, residual=ZeroHook())
    last = res.x[res.n_done - 1, :LAYOUT.n]
    np.testing.assert_allclose(res.pots.read(0), ss.potentials(last[None, :])[0], rtol=1e-12, atol=1e-12)


def test_a_residual_run_refuses_a_given_buffer():
    with pytest.raises(ukf.UKFError):
        ukf_resid.run_filter_numpy(series(1.0, 14), CFG, LAYOUT, Q, SPEC_A, residual=ZeroHook(), buffer=ss.make_buffer(CFG))


def test_residual_runners_have_no_smoother():
    rf, rs = ukf_resid.make_runners(SPEC_A, ZeroHook())
    res = rf(series(1.0, 14), CFG, LAYOUT, Q)
    with pytest.raises(ukf.UKFError):
        rs(res, CFG)


def test_run_pass1_with_a_zero_residual_equals_the_numpy_filter_and_needs_forward_only():
    z = series(4.0, 16)
    seg, starts = [z.T.copy()], [0]
    spec = passes.make_spec(CFG, "A", seg)
    p0 = passes.run_pass1(seg, starts, CFG, forward_only=True, spec=spec, residual=ZeroHook())
    ref = ukf_resid.run_filter_numpy(z, CFG, LAYOUT, 1.0e-2, spec)
    assert not p0.segments[0].diverged
    np.testing.assert_allclose(p0.segments[0].z_pred, ref.z_pred[:z.shape[0]], rtol=1e-12, atol=1e-12)
    expected = np.mean((z - ref.z_pred) ** 2, axis=1)
    np.testing.assert_allclose(p0.segments[0].sq_err, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(passes.PassError):
        passes.run_pass1(seg, starts, CFG, forward_only=False, spec=spec, residual=ZeroHook())


# ---- FrozenResidual: z-scoring and products ---------------------------------------------------------

def _z():
    return regression.ZScore(mean_X=np.array([1.0, 2.0, 3.0]), sd_X=np.array([2.0, 4.0, 5.0]), mean_y=7.0, sd_y=3.0, n=100)


def test_frozen_residual_applies_zscore_products_and_the_inverse_target_scaling_by_hand():
    # inputs (5, 10, 8): z-scored ((5-1)/2, (10-2)/4, (8-3)/5) = (2, 2, 1) -> hand arithmetic: S_src = 1,
    # u_tgt_u_src = 2 * 2 = 4, f = 1 + 4 = 5, value = mean_y + sd_y * f = 7 + 3 * 5 = 22
    r = regression.FrozenResidual("S_src + u_tgt_u_src", _z())
    np.testing.assert_allclose(r.value([5.0], [10.0], [8.0]), [22.0], rtol=0, atol=1e-12)


def test_frozen_residual_equals_design_columns_on_random_rows():
    z = _z()
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3)) * [2, 4, 5] + [1, 2, 3]
    text = "tanh(u_tgt) + 0.3 * u_src_S_src - S_src / (u_src * u_src + 1.5)"
    rows = regression.Rows(X_raw=X, y=np.zeros(40), node=np.zeros(40, int), window=np.zeros(40, np.int64),
                           t=np.zeros(40, int), subject=np.full(40, "s"))
    Xd, _ = regression.design(rows, z)
    expected = z.mean_y + z.sd_y * regression.evaluate_equation(text, Xd)
    got = regression.FrozenResidual(text, z).value(X[:, 0], X[:, 1], X[:, 2])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_frozen_residual_constant_equation_broadcasts():
    r = regression.FrozenResidual("2.0", _z())
    np.testing.assert_allclose(r.value([0, 1, 2], [0, 1, 2], [0, 1, 2]), [7.0 + 3.0 * 2.0] * 3)


# ---- frozen equation on disk -----------------------------------------------------------------------

def _record(**kw):
    rec = {"role": "primary", "k": 0, "seeds": {"fold": 1}, "versions": {"pysr": "2.6.0"}, "no_term": False,
           "reason": "selected", "equation": "S_src + u_tgt_u_src", "complexity": 5, "val_loss": 0.5,
           "min_val_loss": 0.49, "zscore": {**_z().to_dict()}, "signatures": ["S_src"], "fit_subjects": ["sub-001"],
           "val_subjects": ["sub-002"], "front": [{"complexity": 5, "equation": "S_src", "loss": 0.6, "val_loss": 0.5}]}
    rec.update(kw)
    return rec


def test_frozen_equation_roundtrip_and_residual(tmp_path):
    doc = regression.build_frozen_document(_record(), CFG, REPO, 42, True)
    path = tmp_path / "frozen_equation_42.json"
    assert regression.write_frozen_equation(path, doc) == "written"
    fe = regression.load_frozen_equation(path)
    assert fe.equation == "S_src + u_tgt_u_src" and not fe.no_term and fe.split_seed == 42
    assert len(fe.sha256) == 64 and fe.doc["pilot"] is True and fe.doc["fit_subjects"] == ["sub-001"]
    np.testing.assert_allclose(fe.residual().value([5.0], [10.0], [8.0]), [22.0], atol=1e-12)
    assert np.array_equal(fe.zscore.sd_X, _z().sd_X)


def test_write_once_refuses_a_different_equation_and_leaves_an_identical_one(tmp_path):
    path = tmp_path / "f.json"
    d1 = regression.build_frozen_document(_record(), CFG, REPO, 42, True)
    regression.write_frozen_equation(path, d1)
    before = path.read_bytes()
    assert regression.write_frozen_equation(path, d1) == "unchanged"
    d2 = regression.build_frozen_document(_record(equation="u_tgt"), CFG, REPO, 42, True)
    with pytest.raises(regression.FrozenEquationError):
        regression.write_frozen_equation(path, d2)
    assert path.read_bytes() == before
    assert regression.write_frozen_equation(path, d2, force=True) == "written"


def test_no_term_loads_and_gives_no_residual(tmp_path):
    path = tmp_path / "f.json"
    rec = _record(no_term=True, equation="0.37", reason="selected equation is a bare constant", signatures=[])
    regression.write_frozen_equation(path, regression.build_frozen_document(rec, CFG, REPO, 42, True))
    fe = regression.load_frozen_equation(path)
    assert fe.no_term and fe.residual() is None


def test_only_the_primary_fit_is_frozen_and_bad_documents_are_refused(tmp_path):
    with pytest.raises(regression.FrozenEquationError):
        regression.build_frozen_document(_record(role="refit"), CFG, REPO, 42, True)
    with pytest.raises(regression.FrozenEquationError):
        regression.load_frozen_equation(tmp_path / "missing.json")
    good = regression.build_frozen_document(_record(), CFG, REPO, 42, True)
    for mutate in (lambda d: d.pop("zscore"), lambda d: d.update(variable_names=["a"]),
                   lambda d: d["zscore"].update(sd_y=0.0), lambda d: d.update(equation="u_tgt +")):
        d = copy.deepcopy(good)
        mutate(d)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises((regression.FrozenEquationError, Exception)):
            regression.load_frozen_equation(p)


def test_path_pattern_pilot_and_full():
    assert regression.frozen_equation_path(CFG, "/r", 42, True).as_posix().endswith("results/pilot/frozen_equation_42.json")
    assert regression.frozen_equation_path(CFG, "/r", 42, False).as_posix().endswith("outputs/frozen_equation_42.json")


def test_importing_and_loading_never_imports_pysr(tmp_path):
    path = tmp_path / "f.json"
    regression.write_frozen_equation(path, regression.build_frozen_document(_record(), CFG, REPO, 42, True))
    code = ("import sys; from src import regression; fe = regression.load_frozen_equation(r'%s'); fe.residual(); "
            "bad = [m for m in ('pysr', 'juliacall', 'julia') if m in sys.modules]; print('BAD' if bad else 'OK', bad)" % path)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith("OK"), out.stdout


def test_state_space_has_no_regression_or_sympy_import():
    text = (REPO / "src" / "state_space.py").read_text(encoding="utf-8")
    assert "regression" not in text.replace("regression.py", "") or "import regression" not in text
    assert "import sympy" not in text
