"""C8: the gain-prior widening used by tools/c8_gain_prior_diagnostic.py (PLAN.md C8; §7.3, §7.6). Nothing here touches src/.

Variant K (config copy) must move all four places the prior SD enters; variant P (wrapped prior_cov) must move only P0, the
carry reset and the gap inflation, leaving the gain random-walk Q and the reference-refresh threshold at the reference values.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from src import state_space as ss
from src import ukf
from src.config import load_config
from tests import sim_data

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "c7"))
import c6_gain_recovery_diagnostic as C6  # noqa: E402
import c8_gain_prior_diagnostic as C8  # noqa: E402

CFG = load_config()
LAYOUT = ss.make_layout(CFG)
GAIN = [i for i, n in enumerate(LAYOUT.names) if n in ("g12", "g21")]
OTHER = [i for i in range(LAYOUT.n) if i not in GAIN]
Q = 1.0e-3


def test_k_scales_all_four_places_and_nothing_else():
    cfg3 = C8.scaled_cfg(CFG, "K", 3.0)
    P1, P3 = np.diag(ss.prior_cov(LAYOUT, CFG)), np.diag(ss.prior_cov(LAYOUT, cfg3))
    np.testing.assert_allclose(P3[GAIN], 9.0 * P1[GAIN], rtol=1e-14)
    np.testing.assert_array_equal(P3[OTHER], P1[OTHER])
    Q1, Q3 = np.diag(ukf.process_noise(LAYOUT, CFG, Q)), np.diag(ukf.process_noise(LAYOUT, cfg3, Q))
    np.testing.assert_allclose(Q3[GAIN], 9.0 * Q1[GAIN], rtol=1e-14)
    np.testing.assert_array_equal(Q3[OTHER], Q1[OTHER])
    s1, s3 = ss.parameter_prior_sd(CFG), ss.parameter_prior_sd(cfg3)
    assert s3["g12"] == pytest.approx(3.0 * s1["g12"]) and s3["g21"] == pytest.approx(3.0 * s1["g21"])
    for k in ("p1", "p2", "log_rho1", "log_rho2"):
        assert s3[k] == s1[k]
    np.testing.assert_array_equal(ss.prior_mean(LAYOUT, cfg3), ss.prior_mean(LAYOUT, CFG))   # prior mean stays 0


def test_scaled_cfg_does_not_touch_the_original_and_x1_is_identical():
    before = copy.deepcopy(CFG)
    C8.scaled_cfg(CFG, "K", 10.0)
    assert CFG == before
    assert C8.scaled_cfg(CFG, "K", 1.0) == CFG
    assert C8.scaled_cfg(CFG, "P", 10.0) == CFG        # P leaves the config alone; the widening is the wrapper


def test_p_widens_p0_and_carry_but_not_q_or_refresh_threshold_and_restores():
    orig_cov, orig_pn = ss.prior_cov, ukf.process_noise
    Q_ref = ukf.process_noise(LAYOUT, CFG, Q)
    sd_ref = ss.parameter_prior_sd(CFG)
    P_ref = np.diag(ss.prior_cov(LAYOUT, CFG))
    with C8.widen_prior_only(3.0):
        P = np.diag(ss.prior_cov(LAYOUT, CFG))
        np.testing.assert_allclose(P[GAIN], 9.0 * P_ref[GAIN], rtol=1e-14)
        np.testing.assert_array_equal(P[OTHER], P_ref[OTHER])
        np.testing.assert_allclose(np.diag(ukf.initial_state(LAYOUT, CFG)[1]), P, rtol=0, atol=0)   # P0 is the widened one
        np.testing.assert_array_equal(ukf.process_noise(LAYOUT, CFG, Q), Q_ref)                      # Q unchanged
        assert ss.parameter_prior_sd(CFG) == sd_ref                                                   # refresh threshold unchanged
        assert ss.prior_cov is not orig_cov                                                           # prior_cov is the wrapper again after pn
    assert ss.prior_cov is orig_cov and ukf.process_noise is orig_pn


def test_p_restores_on_exception():
    orig_cov, orig_pn = ss.prior_cov, ukf.process_noise
    with pytest.raises(RuntimeError):
        with C8.widen_prior_only(3.0):
            raise RuntimeError("boom")
    assert ss.prior_cov is orig_cov and ukf.process_noise is orig_pn


@pytest.fixture(scope="module")
def short_series():
    r = sim_data.make_recording(CFG, [12.0, 10.0], 7, g12=8.0, g21=8.0, p=(220.0, 250.0), burn_seconds=2.0)
    return r["segments"], r["starts"]


def _pass1(segs, starts, cfg):
    return C6._pass1_numba(segs, starts, cfg, Q, False)


def test_x1_equals_reference_and_widening_changes_the_estimate(short_series):
    segs, starts = short_series
    ref = _pass1(segs, starts, CFG)
    x1 = _pass1(segs, starts, C8.scaled_cfg(CFG, "K", 1.0))
    with C8.widen_prior_only(1.0):
        p1 = _pass1(segs, starts, CFG)
    for other in (x1, p1):
        assert other.params.g12 == ref.params.g12 and other.params.posterior_sd["g12"] == ref.params.posterior_sd["g12"]
    k3 = _pass1(segs, starts, C8.scaled_cfg(CFG, "K", 3.0))
    with C8.widen_prior_only(3.0):
        p3 = _pass1(segs, starts, CFG)
    assert k3.params.g12 != ref.params.g12 and p3.params.g12 != ref.params.g12
    assert k3.params.g12 != p3.params.g12                 # K and P are different runs (Q differs)
    assert p3.params.posterior_sd["g12"] > 0.0 and k3.params.posterior_sd["g12"] > 0.0


def _row(arm, v, s, c, si, rel, null_val=0.3, **kw):
    g = 0.0 if c == "null" else 10.0
    r = {"arm": arm, "variant": v, "setting": s, "cell": c, "series": si, "g_true": g, "off_failures": [], "nan_inf": False,
         "on_recording_diverged": False, "contraction": {"g12": 0.9, "g21": 0.9, "m": 0.6}}
    val = null_val if c == "null" else g * (1.0 + rel)
    r.update({"g12": val, "g21": val, "g12_filt": val, "g21_filt": val})
    r.update(kw)
    return r


def _stub(rel, null_val):
    C6.DELTA = 1.08
    return [_row("A", "S1", "x1", c, si, rel, null_val) for c in C8.CELLS for si in range(6)]


def test_criteria_logic_trade_off_case():
    fl, first = C8.criteria(_stub(0.05, 2.0), "A", "S1", "x1")      # accurate gains, nulls beyond delta
    assert fl[0] and not fl[1] and first == "c2"
    fl, first = C8.criteria(_stub(0.40, 0.3), "A", "S1", "x1")      # attenuated gains, nulls fine
    assert not fl[0] and fl[1] and first == "c1"
    fl, first = C8.criteria(_stub(0.05, 0.3), "A", "S1", "x1")
    assert all(fl) and first is None
