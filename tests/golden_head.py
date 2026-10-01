"""Helper for tests/test_filter_golden.py: the fixed recording and the summary numbers (no src logic here)."""
import numpy as np

from sim_data import make_recording

Q = 1.0e-2
SEG_SECONDS = (10, 8)


def recording(cfg):
    return make_recording(cfg, SEG_SECONDS, seed=77, g12=8.0, g21=4.0, m=0.3, p=np.array([220.0, 250.0]))


def summary(p1, p2):
    """Small scalar fingerprint of a pass-1 (+ pass-2) result."""
    prm = p1.params
    nis = np.concatenate([s.nis[s.nis_keep] for s in p1.segments]) if p1.segments[0].nis is not None else None
    out = {"g12": prm.g12, "g21": prm.g21, "m": prm.m, "p1": prm.p1, "p2": prm.p2, "log_rho1": prm.log_rho1,
           "gf12": p1.gain_estimate["g12"], "gf21": p1.gain_estimate["g21"], "n_div": p1.n_diverged}
    if p2 is not None:
        out["win_x_sum"] = float(sum(np.sum(w.x_smooth) for w in p2.kept))
        out["n_windows"] = len(p2.windows)
    return out
