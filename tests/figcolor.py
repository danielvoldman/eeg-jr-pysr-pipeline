"""Colour checks for the figure palette (IMP-099): CIE76 dE under CVD simulation and grayscale dL*.

CVD simulation: Machado, Oliveira and Fernandes (2009), severity 1.0, applied to linear sRGB. Used by tests only.
"""
import numpy as np

CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = _M @ np.ones(3)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_lab(lin):
    xyz = (_M @ lin) / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_of(hex_color, cvd=None):
    lin = to_linear(hex_to_rgb(hex_color))
    if cvd is not None:
        lin = np.clip(CVD[cvd] @ lin, 0.0, 1.0)
    return to_lab(lin)


def delta_e(a, b, cvd=None):
    return float(np.linalg.norm(lab_of(a, cvd) - lab_of(b, cvd)))


def delta_l(a, b):
    """Grayscale lightness difference (L* of the luminance, which is what a grayscale print keeps)."""
    return abs(float(lab_of(a)[0] - lab_of(b)[0]))


def min_pair_metrics(colors):
    """(min dE over normal and the three CVD simulations, min dL*) over all pairs of the given hex colours."""
    de, dl = [], []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            de.append(min(delta_e(colors[i], colors[j], c) for c in (None, "protan", "deutan", "tritan")))
            dl.append(delta_l(colors[i], colors[j]))
    return min(de), min(dl)
