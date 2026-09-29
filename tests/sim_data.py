"""Simulated two-node recordings with known truth for the C3 tests (§7.5, §9.1 style).

One long A2 run at 2048 Hz, sampled every 8th step (256 Hz), mixed on the deviations from mu_ref
(IMP-015) with observation noise at the filter's own R, then cut into clean segments separated by
gaps. The truth states are returned alongside, on the same sample axis.
"""
import math

import numpy as np

from src import model


def simulate_stream(cfg, seconds, seed, g12=0.0, g21=0.0, m=0.2, p=None):
    fs_sim, fs_obs = cfg["rescaling"]["reference_simulation"]["sim_fs_hz"], cfg["preprocessing"]["observation_fs_hz"]
    step = int(round(fs_sim / fs_obs))
    res = model.simulate(cfg, int(seconds * fs_sim), seed=seed, n_nodes=2, g12=g12, g21=g21, p=p)
    states = res.states[step::step]
    y = states[:, :, 1] - states[:, :, 2]
    mu, sigma = cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    M = np.array([[1.0, m], [m, 1.0]])
    rng = np.random.default_rng(seed + 1000)
    z = mu + (y - mu) @ M.T + rng.normal(size=y.shape) * math.sqrt(cfg["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"]) * sigma
    return states.reshape(len(states), 12), z


def make_recording(cfg, seg_seconds, seed, g12=0.0, g21=0.0, m=0.2, p=None, gap_seconds=3.0, burn_seconds=4.0):
    """Clean segments cut from one continuous stream. Returns dict(segments (2, n) arrays, starts, truths
    (n, 12) arrays, fs, z_all, truth_all). `starts` are sample indices on the stream's axis."""
    fs = cfg["preprocessing"]["observation_fs_hz"]
    total = burn_seconds + sum(seg_seconds) + gap_seconds * (len(seg_seconds) - 1) + 1.0
    truth, z = simulate_stream(cfg, total, seed, g12, g21, m, p)
    starts, segments, truths = [], [], []
    pos = int(round(burn_seconds * fs))
    for sec in seg_seconds:
        n = int(round(sec * fs))
        starts.append(pos)
        segments.append(np.ascontiguousarray(z[pos:pos + n].T))
        truths.append(truth[pos:pos + n].copy())
        pos += n + int(round(gap_seconds * fs))
    return {"segments": segments, "starts": starts, "truths": truths, "fs": fs, "z_all": z, "truth_all": truth}
