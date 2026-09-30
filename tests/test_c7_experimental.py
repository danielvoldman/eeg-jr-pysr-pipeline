"""C7: the experimental extended UKF (tools/c7) and the aperiodic estimator (PLAN.md C7; §7.5, §9.3).

Nothing here touches src/. The extended kernels are edited copies of src/ukf_numba; with no extra state and no
adaptation they must reproduce it, and with extra states they must reproduce an independent NumPy implementation
built from the reference filter pieces (ukf.UnscentedFilter, ss.predict, ss.observe, ukf.rts_smooth).
"""
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from src import passes, ukf, ukf_numba
from src import state_space as ss
from src.config import load_config
from tests import sim_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "c7"))
import aperiodic_prior as ap  # noqa: E402
import ext_ukf  # noqa: E402

CFG = load_config()
Q = 1.0e-2
RTOL, ATOL = 1e-8, 1e-10
RTOL_SC, ATOL_SC = CFG["ukf"]["numba"]["rtol_smoothed_cov"], CFG["ukf"]["numba"]["atol_smoothed_cov"]
LAYOUT = ss.make_layout(CFG)
SPEC_A = ext_ukf.Spec("A", s2=(6.0, 9.0), tau=(0.1, 0.2))
SPEC_B = ext_ukf.Spec("B", s2=(6.0, 9.0), tau=(0.1, 0.2))
SPEC_N = ext_ukf.Spec("N")


def series(seconds, seed, g12=8.0, g21=2.0, p=(220.0, 250.0)):
    r = sim_data.make_recording(CFG, [seconds], seed, g12=g12, g21=g21, p=p, burn_seconds=2.0)
    return np.ascontiguousarray(r["segments"][0].T)


def numpy_oracle(z, spec, keep_smoother=True):
    """Independent NumPy filter (and smoother) for the extended state, from the reference pieces."""
    nb = LAYOUT.n
    nx = spec.nx
    n = nb + nx
    phi, q_ex, p0_ex = ext_ukf.extra_terms(spec, CFG)
    x0, P0, buf = ukf.initial_state(LAYOUT, CFG)
    Qm = np.zeros((n, n))
    Qm[:nb, :nb] = ukf.process_noise(LAYOUT, CFG, Q)
    Qm[nb:, nb:] = np.diag(q_ex)
    P = np.zeros((n, n))
    P[:nb, :nb] = P0
    P[nb:, nb:] = np.diag(p0_ex)

    def propagate(s, wm):
        return np.hstack([ss.predict(s[:, :nb], wm, buf, LAYOUT, CFG), s[:, nb:] * phi])

    def observe(s):
        return ss.observe(s[:, :nb], LAYOUT, CFG) + s[:, nb:]

    filt = ukf.UnscentedFilter(n, ukf.weights_for(n, CFG), Qm, ukf.obs_noise(CFG), propagate, observe,
                               jitter=CFG["ukf"]["divergence"]["covariance_jitter"])
    filt.x, filt.P = np.concatenate([x0, np.zeros(nx)]), P
    xs, Ps, snaps, nis = [], [], [], []
    for t in range(z.shape[0]):
        snaps.append(ukf.buffer_snapshot(buf))
        filt.predict()
        filt.update(z[t])
        buf.replace_latest(ss.sigmoid_of_mean_potential(filt.x[None, :nb], [1.0], CFG))
        xs.append(filt.x.copy())
        Ps.append(filt.P.copy())
        nis.append(float(filt.y @ filt.SI @ filt.y))
    xs, Ps, snaps = np.array(xs), np.array(Ps), np.array(snaps)
    out = {"x": xs, "P": Ps, "nis": np.array(nis), "snaps": snaps}
    if keep_smoother:
        def propagate_k(k, sigmas, wm):
            b = ukf.buffer_from_snapshot(snaps[k + 1])
            return np.hstack([ss.predict(sigmas[:, :nb], wm, b, LAYOUT, CFG), sigmas[:, nb:] * phi])
        out["xs"], out["Ps"] = ukf.rts_smooth(xs, Ps, ukf.weights_for(n, CFG), Qm, propagate_k,
                                              CFG["ukf"]["divergence"]["covariance_jitter"], [])
    return out


# ---- the copy is faithful ---------------------------------------------------------------------------------

def test_no_extra_state_no_adaptation_reproduces_src_numba():
    z = series(6.0, 11)
    a = ukf_numba.run_filter(z, CFG, LAYOUT, Q, keep_cov=True)
    b = ext_ukf.run_filter_ext(z, CFG, LAYOUT, Q, SPEC_N, keep_cov=True)
    assert (a.n_done, a.diverged) == (b.n_done, b.diverged)
    for name in ("x", "P", "z_pred", "S", "innovation", "nis", "snapshots", "min_eig"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-12, atol=1e-12, err_msg=name)
    xa, Pa = ukf_numba.run_smoother(a, CFG)
    xb, Pb = ext_ukf.run_smoother_ext(b, CFG, SPEC_N, LAYOUT)
    np.testing.assert_allclose(xb, xa, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Pb, Pa, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B], ids=["A_ou", "B_random_walk"])
def test_extended_filter_and_smoother_match_numpy_oracle(spec):
    z = series(4.0, 12)
    ora = numpy_oracle(z, spec)
    res = ext_ukf.run_filter_ext(z, CFG, LAYOUT, Q, spec, keep_cov=True)
    assert res.x.shape[1] == LAYOUT.n + 2 and not res.diverged
    np.testing.assert_allclose(res.x, ora["x"], rtol=RTOL, atol=ATOL, err_msg="x")
    np.testing.assert_allclose(res.P, ora["P"], rtol=RTOL, atol=ATOL, err_msg="P")
    np.testing.assert_allclose(res.nis, ora["nis"], rtol=RTOL, atol=ATOL, err_msg="nis")
    np.testing.assert_allclose(res.snapshots, ora["snaps"], rtol=RTOL, atol=ATOL, err_msg="buffer")
    xs, Ps = ext_ukf.run_smoother_ext(res, CFG, spec, LAYOUT)
    np.testing.assert_allclose(xs, ora["xs"], rtol=RTOL, atol=ATOL, err_msg="smoothed x")
    np.testing.assert_allclose(Ps, ora["Ps"], rtol=RTOL_SC, atol=ATOL_SC, err_msg="smoothed P")


def test_noise_state_is_seen_in_the_observation():
    """A constant +40 mV offset on one channel is absorbed by the noise state of that channel, not by the neural states."""
    z = series(4.0, 13, g12=0.0, g21=0.0)
    off = z.copy()
    off[:, 0] += 3.0
    spec = ext_ukf.Spec("B", s2=(25.0, 25.0), tau=(0.1, 0.1))
    res = ext_ukf.run_filter_ext(off, CFG, LAYOUT, Q, spec)
    assert not res.diverged
    assert res.x[-256:, LAYOUT.n].mean() > 1.5           # channel-1 noise state carries the offset
    assert abs(res.x[-256:, LAYOUT.n + 1].mean()) < 1.0


# ---- terms, weights, stability bookkeeping ------------------------------------------------------------------

def test_extra_terms_keep_the_ou_variance_stationary_and_give_the_walk_the_same_drive():
    dt = 1.0 / CFG["preprocessing"]["observation_fs_hz"]
    phi, qx, p0 = ext_ukf.extra_terms(SPEC_A, CFG)
    np.testing.assert_allclose(phi ** 2 * p0 + qx, p0, rtol=1e-13)
    np.testing.assert_allclose(phi, np.exp(-dt / np.array(SPEC_A.tau)), rtol=1e-14)
    phi_b, qb, p0b = ext_ukf.extra_terms(SPEC_B, CFG)
    np.testing.assert_array_equal(phi_b, np.ones(2))
    np.testing.assert_allclose(qb, 2.0 * np.array(SPEC_B.s2) / np.array(SPEC_B.tau) * dt, rtol=1e-14)
    np.testing.assert_allclose(qb, qx, rtol=0.05)         # same drive as the OU up to O(dt / tau) (4% at tau = 0.1 s)
    assert ext_ukf.extra_terms(ext_ukf.Spec("C"), CFG)[0].size == 0


def test_sigma_weights_at_21_dimensions():
    lam, Wm, Wc = ukf_numba.weights_for(21, CFG)
    assert lam == 0.0 and Wm.size == 43
    assert Wm[0] == 0.0 and Wc[0] == 2.0
    np.testing.assert_allclose(Wm[1:], 1.0 / 42.0, rtol=1e-14)
    assert abs(Wm.sum() - 1.0) < 1e-14
    assert not (Wm < 0).any() and not (Wc < 0).any()     # no negative weights at 21-D with the locked alpha = 1


def test_monitor_reports_the_stability_quantities():
    res = ext_ukf.run_filter_ext(series(3.0, 14), CFG, LAYOUT, Q, SPEC_A)
    for key in ("min_eig_overall", "n_negative_eig_steps", "n_jitter_fallbacks", "nan_inf_seen", "diverged"):
        assert key in res.monitor


# ---- candidate C ----------------------------------------------------------------------------------------

def test_adaptive_r_stays_between_floor_and_cap_and_rises_on_noisy_data():
    spec = ext_ukf.Spec("C")
    z = series(8.0, 15)
    quiet = ext_ukf.run_filter_ext(z, CFG, LAYOUT, Q, spec)
    R0 = ukf.obs_noise(CFG)[0, 0]
    cap = CFG["rescaling"]["sigma_ref"] ** 2
    assert np.all(quiet.R_used >= R0 - 1e-12) and np.all(quiet.R_used <= cap + 1e-12)
    rng = np.random.default_rng(31)
    noisy = z + rng.normal(size=z.shape) * 1.0 * CFG["rescaling"]["sigma_ref"]
    cfg_off = copy.deepcopy(CFG)                          # state-SD flag off: this test is about R, not the rule
    cfg_off["ukf"]["divergence"]["state_sd_multiple"] = 1e30
    loud = ext_ukf.run_filter_ext(noisy, cfg_off, LAYOUT, Q, spec)
    assert np.median(loud.R_used[-1024:, :]) > 2.0 * np.median(quiet.R_used[-1024:, :])
    fixed = ext_ukf.run_filter_ext(noisy, cfg_off, LAYOUT, Q, SPEC_N)
    assert not loud.diverged and not fixed.diverged
    assert np.nanmean(loud.nis[512:]) < np.nanmean(fixed.nis[512:])      # adaptation pulls the NIS down


# ---- pass 1 through the unchanged src/passes.py ----------------------------------------------------------------

def test_pass1_with_patched_filters_trims_carries_only_parameters_and_restores():
    r = sim_data.make_recording(CFG, [6.0, 6.0], 16, g12=6.0, g21=6.0, burn_seconds=2.0)
    orig = (ukf.run_filter, ukf.run_smoother)
    with ext_ukf.patched_filters(SPEC_A) as pf:
        res = passes.run_pass1(r["segments"], r["starts"], CFG, Q)
    assert (ukf.run_filter, ukf.run_smoother) == orig
    assert res.params is not None and len(pf.monitors) == 2
    seg1 = res.segments[1]
    assert seg1.carry_in_mean.shape == (LAYOUT.n - ss.N_NEURAL,)         # the 7 parameters, no noise state
    assert res.segments[0].x_smooth_params.shape[1] == LAYOUT.n - ss.N_NEURAL
    assert np.isfinite(res.params.g12) and np.isfinite(res.gain_estimate["g12"])


# ---- the estimator ----------------------------------------------------------------------------------------

def _ou(n, s2, tau, fs, seed):
    rng = np.random.default_rng(seed)
    phi = np.exp(-1.0 / (fs * tau))
    e = rng.normal(size=n) * np.sqrt(s2 * (1.0 - phi ** 2))
    x = np.empty(n)
    x[0] = rng.normal() * np.sqrt(s2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_estimator_recovers_a_known_lorentzian_and_returns_little_on_white_noise():
    fs = 256.0
    s2_true, tau_true = 4.0, 0.08
    segs = [np.vstack([_ou(int(90 * fs), s2_true, tau_true, fs, 100 + k * 2 + c) for c in range(2)]) for k in range(3)]
    est = ap.estimate(segs, fs, 0.0)
    np.testing.assert_allclose(est["s2"], s2_true, rtol=0.25)
    np.testing.assert_allclose(est["tau"], tau_true, rtol=0.35)
    rng = np.random.default_rng(5)
    white = [rng.normal(size=(2, int(90 * fs))) * 2.0 for _ in range(3)]
    est_w = ap.estimate(white, fs, 4.0)                   # R equals the white variance: nothing is left to fit
    assert np.all(est_w["s2"] < 0.10 * est_w["var"])


def test_estimator_is_deterministic_and_uses_the_documented_bins():
    fs = 256.0
    segs = [np.vstack([_ou(int(30 * fs), 3.0, 0.1, fs, 7), _ou(int(30 * fs), 3.0, 0.1, fs, 8)])]
    a, b = ap.estimate(segs, fs, 0.0), ap.estimate(segs, fs, 0.0)
    np.testing.assert_array_equal(a["s2"], b["s2"])
    assert ap.FIT_HZ == (1.0, 40.0) and ap.EXCLUDE_HZ == (7.0, 14.0)


# ---- hygiene ------------------------------------------------------------------------------------------------

def test_new_kernels_have_no_fastmath_and_sources_have_no_global_random_or_print():
    for name in ("_extra_decay", "_extra_observe", "_filter_kernel_ext", "_smooth_kernel_ext"):
        assert getattr(ext_ukf, name).targetoptions.get("fastmath") is False, name
    for f in ("ext_ukf.py", "aperiodic_prior.py"):
        text = (Path(__file__).resolve().parent.parent / "tools" / "c7" / f).read_text(encoding="utf-8")
        assert not re.search(r"np\.random\.(?!default_rng)", text)
        assert not re.search(r"^\s*print\(", text, re.M)
        assert "fastmath=True" not in text
