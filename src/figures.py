"""Figure set (§19, §18; PLAN H1, reduced scope of IMP-099).

Computes nothing: every drawn number is read from outputs/gate.json and the formal result file the gate names
(results/g0_null_arms_r<round>.json). Fig 2 is the only figure with content at the G0 hard stop; every other figure is a
labelled placeholder with the reason. PNG at 300 DPI only (§19, LOCKED). The style (fonts, sizes, colours) lives here
and its values in config.yml `figures` (IMP-099). `main.py --phase 4` stays refused by the hard stop (§18.1); the entry
points are `python -m src.figures --figure N` and `--all`.
"""
import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import REPO_ROOT, load_config  # noqa: E402

log = logging.getLogger("figures")

# §19 titles. Claim labels follow the table in §19.
FIGURES = {
    1: ("data_overview", "Data overview", "Data"),
    2: ("synthetic_gate", "Synthetic gate (G0)", "G0"),
    3: ("ukf_states", "UKF hidden-state example", "Method"),
    4: ("c1_ablation", "Held-out one-step error, M0 to M3", "C1"),
    5: ("c2_recurrence", "Residual-term recurrence", "C2"),
    6: ("c1_interpretation", "Volume conduction and AAFT", "C1 interpretation"),
    7: ("c1_split_seeds", "C1 across split seeds", "C2"),
    8: ("c3_icc", "Session reliability (ICC)", "C3"),
    9: ("c4_freerun", "Free-run spectral error, 2/5/10 s", "C4"),
}
DOUBLE_WIDTH = {2}
# Why a placeholder exists, by figure, when the gate hard-stopped (outputs/gate.json hard_stop true).
PLACEHOLDER_REASON = {
    1: "Not built for this outcome: the only real data used are pilot recordings (mechanics only).",
    3: "Not built for this outcome: no frozen model exists, and pilot fits are mechanics only.",
    4: "C1 not attempted: G0 hard stop, phase 2 refused.",
    5: "C2 not attempted: G0 hard stop, no primary fit and no ensemble.",
    6: "Not attempted: G0 hard stop, no estimated gain to diagnose.",
    7: "C2 not attempted: G0 hard stop, no split-seed fits.",
    8: "C3 not attempted: G0 hard stop (the pilot has 2 usable pairs, mechanics only).",
    9: "C4 not attempted: G0 hard stop.",
}


class FigureError(RuntimeError):
    """Raised when an input of a figure is missing or inconsistent. Never silently skipped."""


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FigureError(f"required input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(cfg=None, root=None):
    """Read outputs/gate.json and the formal result file it names (§19: reads only results/ and the gate)."""
    cfg = cfg or load_config()
    root = Path(root or REPO_ROOT)
    gate = read_json(root / cfg["paths"]["outputs_dir"] / "gate.json")
    formal = None
    if gate.get("formal_results_file"):
        formal = read_json(root / gate["formal_results_file"])
    return {"gate": gate, "formal": formal}


def rc_params(cfg):
    f = cfg["figures"]
    return {
        "font.family": f["font_family"],
        "font.size": f["font_tick_pt"],
        "axes.labelsize": f["font_label_pt"],
        "axes.titlesize": f["font_panel_pt"],
        "axes.titleweight": "bold",
        "axes.linewidth": f["line_axes_pt"],
        "xtick.labelsize": f["font_tick_pt"],
        "ytick.labelsize": f["font_tick_pt"],
        "xtick.major.width": f["line_axes_pt"],
        "ytick.major.width": f["line_axes_pt"],
        "legend.fontsize": f["font_tick_pt"],
        "lines.linewidth": f["line_data_pt"],
        "lines.markersize": f["marker_pt"],
        "figure.dpi": f["dpi"],
        "savefig.dpi": f["dpi"],
        "text.color": f["colors"]["text"],
        "axes.edgecolor": f["colors"]["text"],
        "axes.labelcolor": f["colors"]["text"],
        "xtick.color": f["colors"]["text"],
        "ytick.color": f["colors"]["text"],
        "figure.constrained_layout.use": True,
    }


@contextmanager
def figstyle(cfg):
    """The one shared style (IMP-099)."""
    with matplotlib.rc_context(rc_params(cfg)):
        yield


def footer_for(gate, cfg, pilot=False):
    """Footer text, only for pilot or low-confidence figures (IMP-099); None otherwise."""
    f = cfg["figures"]
    if pilot:
        return f["footer_pilot"]
    if gate.get("low_confidence"):
        return f["footer_low_confidence"]
    return None


def size_for(number, cfg):
    f = cfg["figures"]
    if number in DOUBLE_WIDTH:
        return f["width_double_in"], f["height_row_double_in"]
    return f["width_single_in"], f["height_single_in"]


def new_figure(number, cfg):
    return plt.figure(figsize=size_for(number, cfg))


def add_footer(fig, gate, cfg, pilot=False):
    text = footer_for(gate, cfg, pilot)
    if text:
        fig.text(0.5, 0.0, text, ha="center", va="bottom", fontsize=cfg["figures"]["font_min_pt"])


def fig_title(number):
    _, title, claim = FIGURES[number]
    return f"Fig {number}  {title} [{claim}]"


# ------------------------------------------------------------------ placeholders
def build_placeholder(number, inputs, cfg, pilot=False):
    """Labelled placeholder for a stage that failed or was not attempted (§19)."""
    gate = inputs["gate"]
    if number not in PLACEHOLDER_REASON:
        raise FigureError(f"no placeholder text for figure {number}")
    if gate.get("hard_stop") is not True:
        raise FigureError("placeholder reasons are written for the G0 hard stop; outputs/gate.json does not say hard_stop true")
    with figstyle(cfg):
        fig = new_figure(number, cfg)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        fig.suptitle(fig_title(number), fontsize=cfg["figures"]["font_label_pt"], fontweight="bold")
        ax.text(0.5, 0.62, "PLACEHOLDER", ha="center", va="center", fontsize=cfg["figures"]["font_panel_pt"], fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.40, PLACEHOLDER_REASON[number], ha="center", va="center", wrap=True,
                fontsize=cfg["figures"]["font_label_pt"], transform=ax.transAxes)
        add_footer(fig, gate, cfg, pilot)
    return fig


# ------------------------------------------------------------------ Fig 2
def fig2_data(inputs):
    """Everything Fig 2 draws, read from the files (nothing computed)."""
    gate, formal = inputs["gate"], inputs["formal"]
    if formal is None:
        raise FigureError("outputs/gate.json names no formal result file")
    arms = {}
    for key in ("null_A", "null_B"):
        a = formal["arms"][key]
        arms[key] = {
            "n": a["n"],
            "dropped": a["rule_on"]["dropped"],
            "outside": a["rule_on"]["outside_delta"],
            "pending": a["rule_on"]["inside_delta"],
            "false_positives": a["false_positives"],
            "bound": a["upper_bound_95"],
            "flag_off_outside": a["flag_off_diagnostic"]["counts"]["outside_delta"],
        }
    cited = formal["cited_artifact_only_null"]
    return {
        "arms": arms,
        "cited": {"n": cited["n"], "n_fail": cited["n_fail"]},
        "delta": formal["delta"],
        "hard_stop": gate["hard_stop"],
        "positive": gate["verdicts"]["positive"],
        "round": formal["round"],
    }


def build_fig2(inputs, cfg, pilot=False):
    d = fig2_data(inputs)
    f = cfg["figures"]
    c = f["colors"]
    arms = d["arms"]
    names = {"null_A": "Null A", "null_B": "Null B"}
    with figstyle(cfg):
        fig = new_figure(2, cfg)
        gs = fig.add_gridspec(1, 3, width_ratios=f["panel_width_ratios"])
        ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
        banner = "G0 HARD STOP" if d["hard_stop"] else "G0"
        fig.suptitle(f"Fig 2  {FIGURES[2][1]}: {banner}, both null arms failed", fontsize=f["font_label_pt"], fontweight="bold")

        # A: recovery by coupling level (positive control not run)
        ax_a.set_title("A", loc="left", fontsize=f["font_panel_pt"], fontweight="bold")
        ax_a.set_axis_off()
        ax_a.text(0.5, 0.5, "Recovery by coupling level\n\nPositive control:\nnot run (formal G0)", ha="center", va="center",
                  fontsize=f["font_label_pt"], transform=ax_a.transAxes)

        # B: null arms, rule on (verdict)
        ax_b.set_title("B", loc="left", fontsize=f["font_panel_pt"], fontweight="bold")
        n = max(a["n"] for a in arms.values())
        parts = (("dropped", "dropped by divergence rule", c["bar_dropped"], None),
                 ("outside", f"outside δ = {d['delta']}", c["bar_outside_delta"], None),
                 ("pending", "inside δ (residual half not run)", c["bar_pending"], f["hatch_pending"]))
        for i, key in enumerate(("null_A", "null_B")):
            bottom = 0
            for field, label, color, hatch in parts:
                ax_b.bar(i, arms[key][field], f["bar_width"], bottom=bottom, color=color, hatch=hatch,
                         edgecolor=c["text"], linewidth=f["line_axes_pt"], label=label if i == 0 else None)
                bottom += arms[key][field]
            ax_b.annotate(f"{arms[key]['false_positives']}/{arms[key]['n']}\nfalse pos.", (i, bottom), xytext=(0, f["label_pad_pt"]),
                          textcoords="offset points", ha="center", va="bottom", fontsize=f["font_tick_pt"])
        ax_b.set_xticks(range(2), [f"{names[k]}\nbound {arms[k]['bound']:.4f}" for k in ("null_A", "null_B")])
        ax_b.set_xlabel("bound: one-sided 95% upper\nbound on the false-positive rate")
        ax_b.set_xlim(-f["bar_xlim_pad"], 1 + f["bar_xlim_pad"])
        ax_b.set_ylim(0, n * (1 + f["headroom_fraction"]))
        ax_b.set_ylabel(f"series (n = {n} per arm)")
        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=f["font_tick_pt"])

        # C: diagnostic and cited rows (not verdicts), criterion at 0
        ax_c.set_title("C", loc="left", fontsize=f["font_panel_pt"], fontweight="bold")
        rows = [("Null A, rule off\n(diagnostic)", arms["null_A"]["flag_off_outside"], arms["null_A"]["n"]),
                ("Null B, rule off\n(diagnostic)", arms["null_B"]["flag_off_outside"], arms["null_B"]["n"]),
                ("Artifact-only null,\npilot, cited", d["cited"]["n_fail"], d["cited"]["n"])]
        for y, (label, k, nn) in enumerate(rows):
            share = f["percent_scale"] * k / nn
            ax_c.plot([share], [y], marker="o", markerfacecolor="none", markeredgecolor=c["text"], linestyle="none",
                      markersize=f["marker_pt"])
            ax_c.annotate(f"{k} of {nn}", (share, y), xytext=(0, f["label_pad_pt"] + f["marker_pt"]), textcoords="offset points",
                          ha="center", va="bottom", fontsize=f["font_tick_pt"])
        ax_c.axvline(0, color=c["bar_outside_delta"], linestyle="--", linewidth=f["line_data_pt"])
        ax_c.annotate(f"required: 0 of {arms['null_A']['n']}", (0, f["panel_c_required_y"]), xytext=(f["label_pad_pt"], 0),
                      textcoords="offset points", ha="left", va="center", fontsize=f["font_tick_pt"], color=c["bar_outside_delta"])
        ax_c.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax_c.set_ylim(*f["panel_c_ylim"])
        ax_c.set_xlim(*f["percent_xlim"])
        ax_c.set_xticks(f["percent_ticks"])
        ax_c.set_xlabel("series outside δ or dropped (%)\nrule-off, pilot rows: not verdicts")
        add_footer(fig, inputs["gate"], cfg, pilot)
    return fig


BUILDERS = {2: build_fig2}


def build_figure(number, inputs, cfg, pilot=False):
    if number not in FIGURES:
        raise FigureError(f"unknown figure {number}")
    if number in BUILDERS:
        return BUILDERS[number](inputs, cfg, pilot)
    return build_placeholder(number, inputs, cfg, pilot)


def figure_path(number, cfg, root=None):
    slug = FIGURES[number][0]
    return Path(root or REPO_ROOT) / cfg["paths"]["figures_dir"] / f"fig{number}_{slug}.{cfg['figures']['format']}"


def save_figure(fig, path, cfg):
    """PNG at the configured DPI, nothing else (§19, LOCKED)."""
    path = Path(path)
    if path.suffix.lower() != f".{cfg['figures']['format']}":
        raise FigureError(f"figures are PNG only (§19): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["figures"]["dpi"], format=cfg["figures"]["format"])
    plt.close(fig)
    return path


def make(numbers, cfg=None, root=None, pilot=False):
    cfg = cfg or load_config()
    inputs = load_inputs(cfg, root)
    written = []
    for number in numbers:
        fig = build_figure(number, inputs, cfg, pilot)
        written.append(save_figure(fig, figure_path(number, cfg, root), cfg))
        log.info("wrote %s", written[-1])
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.figures", description=__doc__.split("\n")[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--figure", type=int, choices=sorted(FIGURES))
    what.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config()
    logs = REPO_ROOT / cfg["paths"]["logs_dir"]
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(logs / "phase4_figures.log"), logging.StreamHandler(sys.stderr)])
    numbers = sorted(FIGURES) if args.all else [args.figure]
    try:
        written = make(numbers, cfg)
    except FigureError as exc:
        log.error("%s", exc)
        return 2
    sys.stdout.write("\n".join(str(p) for p in written) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
