"""E4 tests: src/ukf_ext.py, the promoted copy of the C7 extended filter (filter options A and B of G0, IMP-068).
Oracles: src.ukf_numba (no extra state) and an independent NumPy filter built from the reference pieces
(ukf.UnscentedFilter, ss.predict, ss.observe, ukf.rts_smooth), for the 19-D layout AND the 12-D fixed-parameter window
layout of pass 2 (which makes the A / B windows 14-D)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "c7"))

import sim_data  # noqa: E402
from src import ukf, ukf_ext, ukf_numba  # noqa: E402
from src import state_space as ss  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
Q = 1.0e-2
RTOL, ATOL = 1e-8, 1e-10
RTOL_SC, ATOL_SC = CFG["ukf"]["numba"]["rtol_smoothed_cov"], CFG["ukf"]["numba"]["atol_smoothed_cov"]
LAYOUT = ss.make_layout(CFG)
SPEC_A = ukf_ext.Spec("A", s2=(6.0, 9.0), tau=(0.1, 0.2))
SPEC_B = ukf_ext.Spec("B", s2=(6.0, 9.0), tau=(0.1, 0.2))
SPEC_N = ukf_ext.Spec("N")


def series(seconds, seed, g12=8.0, g21=2.0, p=(220.0, 250.0)):
    r = sim_data.make_recording(CFG, [seconds], seed, g12=g12, g21=g21, p=p, burn_seconds=2.0)
    return np.ascontiguousarray(r["segments"][0].T)


def numpy_oracle(z, spec, layout, buf=None):
    """Independent NumPy filter and smoother for the extended state on any base layout."""
    nb, nx = layout.n, spec.nx
    n = nb + nx
    phi, q_ex, p0_ex = ukf_ext.extra_terms(spec, CFG)
    x0, P0, d_buf = ukf.initial_state(layout, CFG)
    buf = d_buf if buf is None else buf
    Qm = np.zeros((n, n))
    Qm[:nb, :nb] = ukf.process_noise(layout, CFG, Q)
    Qm[nb:, nb:] = np.diag(q_ex)
    P = np.zeros((n, n))
    P[:nb, :nb] = P0
    P[nb:, nb:] = np.diag(p0_ex)

    def propagate(s, wm):
        return np.hstack([ss.predict(s[:, :nb], wm, buf, layout, CFG), s[:, nb:] * phi])

    def observe(s):
        return ss.observe(s[:, :nb], layout, CFG) + s[:, nb:]

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

    def propagate_k(k, sigmas, wm):
        b = ukf.buffer_from_snapshot(snaps[k + 1])
        return np.hstack([ss.predict(sigmas[:, :nb], wm, b, layout, CFG), sigmas[:, nb:] * phi])

    sm_x, sm_P = ukf.rts_smooth(xs, Ps, ukf.weights_for(n, CFG), Qm, propagate_k,
                                CFG["ukf"]["divergence"]["covariance_jitter"], [])
    return {"x": xs, "P": Ps, "nis": np.array(nis), "snaps": snaps, "xs": sm_x, "Ps": sm_P}


def test_no_extra_state_reproduces_src_numba():
    z = series(6.0, 11)
    a = ukf_numba.run_filter(z, CFG, LAYOUT, Q, keep_cov=True)
    b = ukf_ext.run_filter_ext(z, CFG, LAYOUT, Q, SPEC_N, keep_cov=True)
    assert (a.n_done, a.diverged) == (b.n_done, b.diverged)
    for name in ("x", "P", "z_pred", "S", "innovation", "nis", "snapshots", "min_eig"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-12, atol=1e-12, err_msg=name)
    xa, Pa = ukf_numba.run_smoother(a, CFG)
    xb, Pb = ukf_ext.run_smoother_ext(b, CFG, SPEC_N, LAYOUT)
    np.testing.assert_allclose(xb, xa, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Pb, Pa, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B], ids=["A_ou", "B_random_walk"])
def test_19d_base_plus_two_noise_states_matches_numpy_oracle(spec):
    z = series(4.0, 12)
    ora = numpy_oracle(z, spec, LAYOUT)
    res = ukf_ext.run_filter_ext(z, CFG, LAYOUT, Q, spec, keep_cov=True)
    assert res.x.shape[1] == 21 and not res.diverged
    np.testing.assert_allclose(res.x, ora["x"], rtol=RTOL, atol=ATOL, err_msg="x")
    np.testing.assert_allclose(res.P, ora["P"], rtol=RTOL, atol=ATOL, err_msg="P")
    np.testing.assert_allclose(res.nis, ora["nis"], rtol=RTOL, atol=ATOL, err_msg="nis")
    np.testing.assert_allclose(res.snapshots, ora["snaps"], rtol=RTOL, atol=ATOL, err_msg="buffer")
    xs, Ps = ukf_ext.run_smoother_ext(res, CFG, spec, LAYOUT)
    np.testing.assert_allclose(xs, ora["xs"], rtol=RTOL, atol=ATOL, err_msg="smoothed x")
    np.testing.assert_allclose(Ps, ora["Ps"], rtol=RTOL_SC, atol=ATOL_SC, err_msg="smoothed P")


@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B], ids=["A_ou", "B_random_walk"])
def test_fixed_parameter_window_layout_gives_14d_windows_matching_the_oracle(spec):
    """Pass 2 holds the recording-level parameters fixed (12 neural states): with the two noise states the window filter
    is 14-D and equals the independent NumPy filter on the fixed layout, with a fresh steady-state buffer."""
    params = {"p1": 225.0, "p2": 245.0, "log_rho1": float(np.log(3.25 / 22.0)), "log_rho2": float(np.log(3.25 / 22.0)),
              "g12": 8.0, "g21": 2.0, "m": 0.3}
    flayout = ss.make_fixed_layout(CFG, params)
    assert flayout.n == 12
    z = series(2.0, 14)
    ora = numpy_oracle(z, spec, flayout, buf=ss.make_buffer(CFG))
    res = ukf_ext.run_filter_ext(z, CFG, flayout, Q, spec, buffer=ss.make_buffer(CFG), keep_cov=True)
    assert res.x.shape[1] == 14 and not res.diverged
    np.testing.assert_allclose(res.x, ora["x"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res.nis, ora["nis"], rtol=RTOL, atol=ATOL)
    xs, Ps = ukf_ext.run_smoother_ext(res, CFG, spec, flayout)
    np.testing.assert_allclose(xs, ora["xs"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Ps, ora["Ps"], rtol=RTOL_SC, atol=ATOL_SC)


def test_noise_state_absorbs_an_offset_not_the_neural_states():
    z = series(4.0, 13, g12=0.0, g21=0.0)
    off = z.copy()
    off[:, 0] += 3.0
    res = ukf_ext.run_filter_ext(off, CFG, LAYOUT, Q, ukf_ext.Spec("B", s2=(25.0, 25.0), tau=(0.1, 0.1)))
    assert not res.diverged
    assert res.x[-256:, LAYOUT.n].mean() > 1.5 and abs(res.x[-256:, LAYOUT.n + 1].mean()) < 1.0


def test_spec_kinds_and_extra_terms():
    with pytest.raises(ukf.UKFError):
        ukf_ext.Spec("C")
    dt = 1.0 / CFG["preprocessing"]["observation_fs_hz"]
    phi, qx, p0 = ukf_ext.extra_terms(SPEC_A, CFG)
    np.testing.assert_allclose(phi ** 2 * p0 + qx, p0, rtol=1e-13)                  # stationary OU variance
    np.testing.assert_allclose(phi, np.exp(-dt / np.array(SPEC_A.tau)), rtol=1e-14)
    phi_b, qb, _ = ukf_ext.extra_terms(SPEC_B, CFG)
    np.testing.assert_array_equal(phi_b, np.ones(2))
    np.testing.assert_allclose(qb, 2.0 * np.array(SPEC_B.s2) / np.array(SPEC_B.tau) * dt, rtol=1e-14)
    assert (SPEC_N.nx, SPEC_A.nx, SPEC_B.nx) == (0, 2, 2)


# ---- the aperiodic estimator --------------------------------------------------------------------------

def test_fit_channel_recovers_a_lorentzian_with_a_white_floor_and_ignores_alpha():
    f = np.arange(0.0, 128.5, 0.5)
    s2, tau, floor = 5.0, 0.08, 0.02
    psd = 4.0 * s2 * tau / (1.0 + (2 * np.pi * f * tau) ** 2) + floor
    psd = psd + 3.0 * np.exp(-((f - 10.0) ** 2) / 2.0)                              # an alpha peak inside the excluded band
    got_s2, got_tau = ukf_ext.fit_channel(f, psd, floor, CFG)
    assert got_tau == pytest.approx(tau, rel=0.06) and got_s2 == pytest.approx(s2, rel=0.12)


def test_fit_channel_without_positive_bins_returns_zero_and_the_upper_bound():
    f = np.arange(0.0, 64.5, 0.5)
    assert ukf_ext.fit_channel(f, np.full_like(f, 0.01), 0.5, CFG) == (0.0, CFG["ukf"]["aperiodic"]["tau_bounds_s"][1])


def test_spec_for_floors_s2_and_defaults():
    rng = np.random.default_rng(3)
    segs = [rng.standard_normal((2, 256 * 30)) * 1.0]
    spec = ukf_ext.spec_for("A", segs, CFG)
    var = np.concatenate(segs, axis=1).var(axis=1, ddof=1)
    assert all(s >= 1e-6 * v * 0.999 for s, v in zip(spec.s2, var))
    assert ukf_ext.spec_for("N", segs, CFG) == SPEC_N and spec.kind == "A" and ukf_ext.spec_for("B", segs, CFG).kind == "B"
    with pytest.raises(ukf.UKFError):
        ukf_ext.pooled_psd([np.zeros((2, 10))], 256, CFG)


def test_config_constants_equal_the_c7_tool_constants():
    import aperiodic_prior as ap
    ac = CFG["ukf"]["aperiodic"]
    assert tuple(ac["fit_hz"]) == ap.FIT_HZ and tuple(ac["exclude_hz"]) == ap.EXCLUDE_HZ
    assert tuple(ac["tau_bounds_s"]) == ap.TAU_BOUNDS_S and ac["n_tau"] == ap.N_TAU
    assert ac["welch_segment_s"] == 2 and ac["welch_overlap_fraction"] == 0.5 and ac["s2_floor_fraction"] == 1e-6
    rng = np.random.default_rng(4)
    segs = [rng.standard_normal((2, 256 * 40)) * 1.5]
    mine = ukf_ext.estimate(segs, CFG)
    theirs = ap.estimate(segs, 256, float(ukf.obs_noise(CFG)[0, 0]))
    np.testing.assert_allclose(mine["s2"], theirs["s2"], rtol=1e-12)
    np.testing.assert_allclose(mine["tau"], theirs["tau"], rtol=1e-12)


def test_source_hygiene():
    text = (Path(ukf_ext.__file__)).read_text(encoding="utf-8")
    assert not re.search(r"(^|\s)print\(", text)
    assert not re.search(r"^\s*(import|from)\s+(filterpy|tools|ext_ukf|aperiodic_prior)", text, re.M)
    assert "fastmath=False" in text and "fastmath=True" not in text
