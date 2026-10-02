"""Figure checks (IMP-099, §19): style, quality, placeholders, Fig 2 data, source scope.

The rendering tests use a fixture shaped like outputs/gate.json and the formal result file (values chosen
independently of the real files), so they run without results/. The real-file Fig 2 check skips with an explicit reason.
"""
import copy
import itertools
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest
from PIL import Image

matplotlib.use("Agg")
from figcolor import delta_e, delta_l, min_pair_metrics  # noqa: E402

from src import figures as F  # noqa: E402
from src.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
S = CFG["figures"]


def make_inputs(hard_stop=True, low_confidence=False):
    arm = lambda fp, d, o, p, ub, off: {  # noqa: E731
        "n": 20, "false_positives": fp, "upper_bound_95": ub,
        "rule_on": {"dropped": d, "outside_delta": o, "inside_delta": p},
        "flag_off_diagnostic": {"counts": {"outside_delta": off, "inside_delta": 20 - off, "dropped": 0}}}
    return {
        "gate": {"hard_stop": hard_stop, "low_confidence": low_confidence, "verdicts": {"positive": "not_run"},
                 "formal_results_file": "results/x.json"},
        "formal": {"delta": 0.5, "round": 3,
                   "arms": {"null_A": arm(14, 7, 7, 6, 0.81, 9), "null_B": arm(17, 8, 9, 3, 0.93, 11)},
                   "cited_artifact_only_null": {"n": 8, "n_fail": 5}},
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


@pytest.fixture(scope="module")
def inputs():
    return make_inputs()


def all_figs(inputs):
    return {n: F.build_figure(n, inputs, CFG) for n in F.FIGURES}


# ------------------------------------------------------------------ palette
def palette_colors():
    c = S["colors"]
    return [c[k] for k in ("bar_dropped", "bar_outside_delta", "bar_pending")], [c[k] for k in ("m0", "m1", "m2", "m3")]


def test_bar_palette_cvd_and_grayscale():
    bars, _ = palette_colors()
    de, dl = min_pair_metrics(bars)
    assert de >= S["palette_min_de_cvd"]
    assert dl >= S["palette_min_dl_gray"]


def test_model_palette_cvd_and_marker_linestyle_alternative():
    _, models = palette_colors()
    de, _ = min_pair_metrics(models)
    assert de >= S["palette_min_de_cvd"]
    names = ("m0", "m1", "m2", "m3")
    styles = {n: (S["series_style"][n]["marker"], S["series_style"][n]["linestyle"]) for n in names}
    colors = dict(zip(names, models))
    for a, b in itertools.combinations(names, 2):
        assert delta_l(colors[a], colors[b]) >= S["palette_min_dl_gray"] or styles[a] != styles[b]
    assert len(set(styles.values())) == 4


def test_m0b_is_hollow_dotted_like_m0():
    m0, m0b = S["series_style"]["m0"], S["series_style"]["m0b"]
    assert m0b["hollow"] is True and m0["hollow"] is False
    assert m0b["linestyle"] == ":" and m0["linestyle"] != ":"
    assert m0b["marker"] == m0["marker"]


def test_palette_checker_rejects_a_bad_palette():
    assert min_pair_metrics(["#D55E00", "#D95F02"])[0] < 10
    assert delta_e("#000000", "#FFFFFF") > 90


# ------------------------------------------------------------------ style and quality
def undrawn_tick_labels(fig):
    """Tick labels whose tick lies outside the view limits: matplotlib keeps them but never draws them."""
    out = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    out.update((tick.label1, tick.label2))
    return out


def visible_texts(fig):
    skip = undrawn_tick_labels(fig)
    return [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and t not in skip]


def test_config_values_follow_the_owner_settings():
    assert (S["width_single_in"], S["width_double_in"], S["height_single_in"], S["height_row_double_in"]) == (3.5, 7.0, 2.6, 3.0)
    assert (S["font_label_pt"], S["font_tick_pt"], S["font_min_pt"], S["font_panel_pt"]) == (8, 7, 7, 9)
    assert (S["line_data_pt"], S["line_axes_pt"], S["marker_min_pt"], S["dpi"], S["format"]) == (1.0, 0.5, 3, 300, "png")
    assert S["font_family"] == "DejaVu Sans" and S["marker_pt"] >= S["marker_min_pt"]


def test_figure_sizes(inputs):
    for n, fig in all_figs(inputs).items():
        w, h = fig.get_size_inches()
        assert (w, h) == ((S["width_double_in"], S["height_row_double_in"]) if n in F.DOUBLE_WIDTH
                          else (S["width_single_in"], S["height_single_in"]))


def test_every_text_is_dejavu_and_at_least_min_size(inputs):
    for n, fig in all_figs(inputs).items():
        for t in visible_texts(fig):
            assert t.get_fontsize() >= S["font_min_pt"], (n, t.get_text())
            assert t.get_fontname() == S["font_family"], (n, t.get_text(), t.get_fontname())


def test_panel_letters_are_nine_point_bold(inputs):
    fig = F.build_figure(2, inputs, CFG)
    letters = [t for ax in fig.axes for t in (ax.title, ax._left_title) if t.get_text() in {"A", "B", "C"}]
    assert {t.get_text() for t in letters} == {"A", "B", "C"}
    for t in letters:
        assert t.get_fontsize() == S["font_panel_pt"] and t.get_fontweight() == "bold"


def test_line_widths_markers_and_spines(inputs):
    fig = F.build_figure(2, inputs, CFG)
    for ax in fig.axes:
        for sp in ax.spines.values():
            assert sp.get_linewidth() == S["line_axes_pt"]
        for ln in ax.get_lines():
            if ln.get_linestyle() not in ("None", "none", ""):
                assert ln.get_linewidth() == S["line_data_pt"]
            if ln.get_marker() not in ("None", None, "", " "):
                assert ln.get_markersize() >= S["marker_min_pt"]


def test_no_text_overlap_and_inside_canvas(inputs):
    tol = S["overlap_tol_px"]
    for n, fig in all_figs(inputs).items():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = [(t, t.get_window_extent(r)) for t in visible_texts(fig)]
        W, H = fig.canvas.get_width_height()
        for t, b in boxes:
            assert b.x0 >= -tol and b.y0 >= -tol and b.x1 <= W + tol and b.y1 <= H + tol, (n, t.get_text())
        for (ta, a), (tb, b) in itertools.combinations(boxes, 2):
            ov_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            ov_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert not (ov_x > tol and ov_y > tol), (n, ta.get_text(), tb.get_text())


def test_saved_png_is_300_dpi_and_the_right_size(tmp_path, inputs):
    for n in (2, 4):
        fig = F.build_figure(n, inputs, CFG)
        w, h = fig.get_size_inches()
        path = F.save_figure(fig, tmp_path / f"f{n}.png", CFG)
        im = Image.open(path)
        assert im.format == "PNG"
        assert abs(im.info["dpi"][0] - S["dpi"]) <= S["dpi_tolerance"]
        assert abs(im.size[0] - round(w * S["dpi"])) <= 1 and abs(im.size[1] - round(h * S["dpi"])) <= 1
        assert im.convert("L").getextrema()[0] < 255  # not blank


def test_only_png_is_written(tmp_path, inputs):
    fig = F.build_figure(4, inputs, CFG)
    with pytest.raises(F.FigureError):
        F.save_figure(fig, tmp_path / "x.jpg", CFG)
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------------ content
def test_placeholders_carry_their_reason(inputs):
    for n in F.FIGURES:
        if n == 2:
            continue
        texts = [t.get_text() for t in visible_texts(F.build_figure(n, inputs, CFG))]
        assert "PLACEHOLDER" in texts
        assert F.PLACEHOLDER_REASON[n] in texts
    assert "not attempted" in F.PLACEHOLDER_REASON[9]  # §19 wording for Fig 9


def test_placeholder_refused_without_a_hard_stop():
    with pytest.raises(F.FigureError):
        F.build_figure(4, make_inputs(hard_stop=False), CFG)


def test_fig2_draws_exactly_the_file_values(inputs):
    d = F.fig2_data(inputs)
    assert d["arms"]["null_A"] == {"n": 20, "dropped": 7, "outside": 7, "pending": 6, "false_positives": 14, "bound": 0.81,
                                   "flag_off_outside": 9}
    assert d["arms"]["null_B"]["false_positives"] == 17 and d["cited"] == {"n": 8, "n_fail": 5}
    fig = F.build_figure(2, inputs, CFG)
    texts = [t.get_text() for t in visible_texts(fig)]
    assert "14/20 false pos." in texts and "17/20 false pos." in texts
    assert any("0.8100" in t for t in texts) and any("0.9300" in t for t in texts)
    assert "5 of 8" in texts and "9 of 20" in texts and "11 of 20" in texts
    assert any(t.startswith("required: 0 of 20") for t in texts)
    assert any("not run" in t for t in texts)
    bars = [p.get_height() for ax in fig.axes for p in ax.patches if abs(p.get_width() - S["bar_width"]) < 1e-9]
    assert sorted(bars) == sorted([7, 7, 6, 8, 9, 3])


def test_fig2_needs_the_formal_file():
    bad = make_inputs()
    bad["formal"] = None
    with pytest.raises(F.FigureError):
        F.fig2_data(bad)


def test_missing_gate_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(F.FigureError):
        F.load_inputs(CFG, tmp_path)


def test_footer_only_on_pilot_or_low_confidence(inputs):
    g = inputs["gate"]
    assert F.footer_for(g, CFG) is None
    assert F.footer_for(g, CFG, pilot=True) == S["footer_pilot"]
    assert F.footer_for({**g, "low_confidence": True}, CFG) == S["footer_low_confidence"]
    plain = [t.get_text() for t in visible_texts(F.build_figure(2, inputs, CFG))]
    assert S["footer_pilot"] not in plain and S["footer_low_confidence"] not in plain
    low = [t.get_text() for t in visible_texts(F.build_figure(2, make_inputs(low_confidence=True), CFG))]
    assert S["footer_low_confidence"] in low
    pil = [t.get_text() for t in visible_texts(F.build_figure(4, inputs, CFG, pilot=True))]
    assert S["footer_pilot"] in pil


def test_real_gate_files_draw_fig2(tmp_path):
    gate = ROOT / "outputs" / "gate.json"
    if not gate.is_file():
        pytest.skip("outputs/gate.json absent (not committed)")
    inp = F.load_inputs(CFG, ROOT)
    if inp["formal"] is None:
        pytest.skip("gate.json names no formal result file")
    d = F.fig2_data(inp)
    assert d["hard_stop"] is True
    assert d["arms"]["null_A"]["false_positives"] == inp["formal"]["arms"]["null_A"]["false_positives"]
    F.save_figure(F.build_figure(2, inp, CFG), tmp_path / "f.png", CFG)


# ------------------------------------------------------------------ scope
def test_figures_py_reads_only_results_and_gate_and_imports_no_pipeline_code():
    text = (ROOT / "src" / "figures.py").read_text(encoding="utf-8")
    imports = set(re.findall(r"^import\s+([\w.]+)", text, re.M))
    for mod, names in re.findall(r"^from\s+([\w.]+)\s+import\s+([^#\n]+)", text, re.M):
        imports.add(mod)
        imports.update(f"{mod}.{n.strip()}" for n in names.split(","))
    forbidden = {"src.model", "src.ukf", "src.ukf_numba", "src.ukf_ext", "src.ukf_resid", "src.passes", "src.regression",
                 "src.robustness", "src.synthetic_gate", "src.preprocess", "src.baseline", "src.freerun", "src.tuning",
                 "src.state_space", "pysr", "numba", "mne", "scipy"}
    assert not forbidden & set(imports)
    for key in ("data_dir", "cache_dir", "pilot_results_dir"):
        assert key not in text
    assert "np.random" not in text and "default_rng" not in text


def test_phase4_stays_refused_at_the_hard_stop():
    r = subprocess.run([sys.executable, "main.py", "--phase", "4"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "refused" in (r.stdout + r.stderr)
    assert not (ROOT / "outputs" / "phase4.done").exists()


def test_cli_rejects_unknown_figure():
    r = subprocess.run([sys.executable, "-m", "src.figures", "--figure", "10"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
