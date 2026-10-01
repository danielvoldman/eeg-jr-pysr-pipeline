"""Numbers pinned from HEAD c25886b (before the F0 wiring change) for the 19-D and A filters (PLAN F0).

The literals below were printed by running the pre-change code (19D: plain passes.run_pass1/run_pass2; A: the
same inside synthetic_gate.filter_context) on the fixed recording of tests/golden_head.py at q = 1e-2. They are
NOT produced by the code under test at the time the test runs; they must never be regenerated from a later commit.
"""
import numpy as np
import pytest

from src import passes, synthetic_gate as sg
from src.config import load_config
import golden_head as gh

CFG = load_config()

GOLDEN = {
    "19D": {"g12": 7.839621514199388, "g21": 4.629830715861255, "m": 0.26282895671650364, "p1": 215.9644319632628,
            "p2": 246.20379920300977, "log_rho1": -1.888511013240705, "gf12": 7.609324631813059,
            "gf21": 4.868187873337444, "n_div": 0, "win_x_sum": 307479.441645082, "n_windows": 9},
    "A": {"g12": 7.9190489308343, "g21": 4.6227412897398095, "m": 0.2736560271214204, "p1": 215.4725190125892,
          "p2": 245.82310981048272, "log_rho1": -1.8874357903316914, "gf12": 7.68119081595137,
          "gf21": 4.822533183304866, "n_div": 0, "win_x_sum": 307505.25519860454, "n_windows": 9},
}


def _run(name):
    r = gh.recording(CFG)
    with sg.filter_context(CFG, name, r["segments"]):
        p1 = passes.run_pass1(r["segments"], r["starts"], CFG, gh.Q)
        p2 = passes.run_pass2(r["segments"], r["starts"], p1.params, CFG, gh.Q)
    return gh.summary(p1, p2)


@pytest.mark.parametrize("name", ["19D", "A"])
def test_head_numbers_are_reproduced(name):
    got = _run(name)
    assert set(got) == set(GOLDEN[name])
    for k, v in GOLDEN[name].items():
        np.testing.assert_allclose(got[k], v, rtol=1e-10, atol=0, err_msg=f"{name}.{k}")
