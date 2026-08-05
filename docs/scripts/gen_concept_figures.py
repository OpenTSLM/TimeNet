# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=2.1"]
# ///
"""Generate the TimeNet concept figures: schematic diagrams of the data model.

Run once; the SVGs are committed under ``docs/assets/figures/`` and referenced from the docs. This is
deliberately not wired into ``make docs`` -- regenerate by hand when the concepts change:

    uv run docs/scripts/gen_concept_figures.py

The figures are deliberately schematic: a plain signal plus the minimum marks that explain one concept
(a span, a point, an input-to-output arrow). They carry no domain detail; that lives in the page HTML
around the image. Set ``FIG_PNG_DIR=/some/dir`` to also drop PNG copies there for eyeballing; the
committed artifacts are SVG only. The structural diagrams (pipeline, sample, annotation/task) are
mermaid, embedded inline in the pages; their sources are under ``docs/scripts/diagrams/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent.parent / "assets" / "figures"
PNG_DIR = os.environ.get("FIG_PNG_DIR")

BLUE = "#4c6ef5"
GREY = "#868e96"
FORECAST = "#f76707"
POINT = "#e8590c"
SPAN_FACE = "#ffe3bf"
SPAN_EDGE = "#f08c00"
GREEN = "#0ca678"
GREEN_FC = "#eefbf5"
STEP_FC = "#f1f3f5"
AXIS = "#adb5bd"
INK = "#212529"
MUTED = "#868e96"

plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# --- signal + primitives ---------------------------------------------------------------------------


def wave(t, seed=0):
    """Build a simple, smooth schematic curve over time array ``t``.

    Each seed picks its own periods, amplitudes, and phases (deterministically), so two curves in one
    figure look independent rather than phase-shifted copies of each other. No noise: it stays a clean
    sum of two sines.

    Returns:
        The curve sampled at ``t``.
    """
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(2.6, 4.2)
    p2 = rng.uniform(1.3, 2.2)
    a2 = rng.uniform(0.25, 0.45)
    ph1 = rng.uniform(0.0, 2 * np.pi)
    ph2 = rng.uniform(0.0, 2 * np.pi)
    return np.sin(2 * np.pi * t / p1 + ph1) + a2 * np.sin(2 * np.pi * t / p2 + ph2)


def bare(ax):
    """Strip ticks and spines so an axis carries only the drawn marks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def plot_series(ax, t, y, color=BLUE, lw=1.7):
    """Plot one signal on ``ax`` in the shared minimal style."""
    ax.plot(t, y, color=color, lw=lw)
    bare(ax)
    ax.margins(y=0.3)


def unit_axis(ax):
    """Turn ``ax`` into a blank 0..1 canvas for drawing schematic output shapes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    bare(ax)


def box(ax, text, *, cx=0.5, cy=0.5, w=0.8, h=0.42, ec=GREEN, fc=GREEN_FC, fs=10):  # noqa: PLR0913
    """Draw a centered rounded box with a label on a unit axis."""
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=1.3, joinstyle="round", capstyle="round"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK)


def caption(fig, text, y=0.9):
    """Write one short grey caption across the top of a figure."""
    fig.text(0.5, y, text, ha="center", va="center", fontsize=9, color=MUTED)


def block_label(fig, ax, text):
    """Write a short grey label centered above ``ax`` in figure coords, so stacked axes align."""
    pos = ax.get_position()
    fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.03, text, ha="center", va="bottom", fontsize=9, color=MUTED)


def draw_axes(ax, *, x_right=1.0, x_axis=True, y_axis=True, time_label=True):
    """Draw schematic x and y axes as light arrows, with ``Time`` at the right end of the x axis."""
    props = {"arrowstyle": "->", "color": AXIS, "lw": 1.0}
    if y_axis:
        ax.annotate("", xy=(0, 1.0), xytext=(0, 0), xycoords="axes fraction", arrowprops=props, annotation_clip=False)
    if x_axis:
        ax.annotate(
            "", xy=(x_right, 0), xytext=(0, 0), xycoords="axes fraction", arrowprops=props, annotation_clip=False
        )
        if time_label:
            ax.text(x_right, -0.12, "Time", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)


def arrow(fig, x=0.635, y=0.46):
    """Draw the input-to-output arrow glyph between two panels."""
    fig.text(x, y, "→", ha="center", va="center", fontsize=22, color=GREY)


def save(fig, name):
    """Write ``fig`` as an SVG under ``OUT`` (and a PNG under ``FIG_PNG_DIR`` when set)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    if PNG_DIR:
        fig.savefig(Path(PNG_DIR) / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {name}.svg")


# --- annotations: one shape on one signal ----------------------------------------------------------


def _annotation_fig(name, cap, draw, seed):
    """Build an annotation exemplar: a plain signal plus one drawn shape and a short caption."""
    t = np.linspace(0, 8, 500)
    fig, ax = plt.subplots(figsize=(6.8, 1.7))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.18)
    plot_series(ax, t, wave(t, seed))
    draw(ax)
    draw_axes(ax)
    caption(fig, cap)
    save(fig, name)


def fig_annotation_static():
    """Annotation: one fact about the whole recording."""
    _annotation_fig(
        "annotation-static",
        "one fact about the whole recording",
        lambda ax: ax.axvspan(0, 8, color=SPAN_FACE, alpha=0.5, zorder=0),
        seed=10,
    )


def fig_annotation_point():
    """An annotation with a PointSpan: one time offset in time."""

    def draw(ax):
        ax.axvline(3.4, color=POINT, lw=1.6, zorder=3)
        ax.plot(3.4, ax.get_ylim()[1] * 0.92, marker="v", color=POINT, ms=7, zorder=4)

    _annotation_fig("annotation-point", "one time offset in time", draw, seed=11)


def fig_annotation_interval():
    """An annotation with an IntervalSpan: a region in time."""

    def draw(ax):
        ax.axvspan(3.0, 5.2, color=SPAN_FACE, alpha=0.85, zorder=0)
        for x in (3.0, 5.2):
            ax.axvline(x, color=SPAN_EDGE, lw=1.1, ls="--", zorder=1)

    _annotation_fig("annotation-interval", "a span in time", draw, seed=12)


def fig_cross_sensor():
    """One span shared across several channels."""
    t = np.linspace(0, 8, 500)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 3.0), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.13, hspace=0.4)
    for i, ax in enumerate(axes):
        plot_series(ax, t, wave(t, seed=20 + i))
        ax.axvspan(3.0, 5.2, color=SPAN_FACE, alpha=0.85, zorder=0)
        for x in (3.0, 5.2):
            ax.axvline(x, color=SPAN_EDGE, lw=1.0, ls="--", zorder=1)
        ax.text(-0.03, 0.5, f"channel {i + 1}", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=INK)
        bottom = i == len(axes) - 1
        draw_axes(ax, x_axis=bottom, time_label=bottom)
    caption(fig, "one span, several channels", y=0.94)
    save(fig, "cross-sensor")


# --- tasks: an input series, an arrow, an output shape ---------------------------------------------


def _io_fig(name, seeds, render_right, *, right_title, question=False, window=None, height=2.3):  # noqa: PLR0913
    """Build a task exemplar: one axis per input channel, an arrow, and the output shape on the right.

    ``seeds`` gives one channel per entry (each its own stacked axis). ``question`` adds a ``+ ?`` panel
    after the channels; ``window`` shades a span across every channel (for a scoped task).
    """
    t = np.linspace(0, 8, 500)
    fig = plt.figure(figsize=(6.8, height))
    outer = fig.add_gridspec(1, 2, width_ratios=[2, 1.2], left=0.03, right=0.97, top=0.80, bottom=0.16, wspace=0.5)
    if question:
        left = outer[0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
        chan_host = left[0]
    else:
        chan_host = outer[0]
    chan_gs = chan_host.subgridspec(len(seeds), 1, hspace=0.4)
    channels = []
    for r, s in enumerate(seeds):
        ax = fig.add_subplot(chan_gs[r], sharex=channels[0] if channels else None)
        ax.plot(t, wave(t, s), color=BLUE, lw=1.6)
        bare(ax)
        ax.margins(y=0.28)
        ax.set_xlim(0, 8)
        if window is not None:
            ax.axvspan(*window, color=SPAN_FACE, alpha=0.85, zorder=0)
        bottom = r == len(seeds) - 1
        draw_axes(ax, x_axis=bottom, time_label=bottom)
        channels.append(ax)
    block_label(fig, channels[0], "time series")
    if question:
        q_ax = fig.add_subplot(left[1])
        unit_axis(q_ax)
        q_ax.text(0.32, 0.5, "+", ha="center", va="center", fontsize=16, color=MUTED)
        q_ax.text(0.72, 0.5, "?", ha="center", va="center", fontsize=18, color=MUTED, fontweight="bold")
        block_label(fig, q_ax, "question")
    r_ax = fig.add_subplot(outer[1])
    unit_axis(r_ax)
    render_right(r_ax)
    r_ax.set_title(right_title, fontsize=9, color=MUTED)
    arrow(fig)
    save(fig, name)


def _text_block(ax):
    """Draw the schematic 'free text' block used as a task output."""
    ax.add_patch(Rectangle((0.08, 0.26), 0.84, 0.48, fc="#f8f9fa", ec=GREY, lw=1.2))
    for yy, ww in [(0.60, 0.70), (0.48, 0.74), (0.36, 0.5)]:
        ax.plot([0.16, 0.16 + ww], [yy, yy], color=GREY, lw=2.4, solid_capstyle="round")


def _spans_axis(ax, spans, points=()):
    """Draw predicted spans and points on a schematic 0..8 timeline axis (a localization output)."""
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    bare(ax)
    for start, end in spans:
        ax.add_patch(Rectangle((start, 0.34), end - start, 0.32, fc=SPAN_FACE, ec=SPAN_EDGE, lw=1.1))
    for x in points:
        ax.axvline(x, color=POINT, lw=1.6, ymin=0.28, ymax=0.72)
        ax.plot(x, 0.78, marker="v", color=POINT, ms=6)


def fig_task_classification():
    """ClassificationTask: a series in, one label out."""
    _io_fig("task-classification", (30, 71), lambda ax: box(ax, "class A", w=0.6), right_title="one label")


def fig_task_classification_scoped():
    """ClassificationTask with a scope: the same label, over a supplied window."""
    _io_fig(
        "task-classification-scoped",
        (31, 72),
        lambda ax: box(ax, "a label\nfor the window", h=0.5),
        right_title="a label for the scope",
        window=(3.0, 5.2),
    )


def fig_task_answer_caption():
    """AnswerTask with no prompt: a series in, free text out."""
    _io_fig("task-answer-caption", (32, 73), _text_block, right_title="free text")


def fig_task_answer():
    """AnswerTask: a question about a series, one answer out."""
    _io_fig("task-answer", (33, 74), lambda ax: box(ax, "one answer"), right_title="an answer", question=True)


def fig_task_answer_rationale():
    """Any task may carry a rationale: a question, a chain of steps, then the answer."""

    def render(ax):
        for yy in (0.82, 0.58):
            box(ax, "step", cx=0.5, cy=yy, w=0.72, h=0.16, ec=GREY, fc=STEP_FC, fs=8)
        box(ax, "answer", cx=0.5, cy=0.24, w=0.72, h=0.2, ec=GREEN, fc=GREEN_FC, fs=9)
        for y0, y1 in [(0.73, 0.67), (0.49, 0.35)]:
            ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.1})

    _io_fig(
        "task-answer-rationale", (30, 75), render, right_title="reasoning, then an answer", question=True, height=2.8
    )


def fig_task_scalar_prediction():
    """ScalarPredictionTask: a series over a window, one typed number out."""
    _io_fig(
        "task-scalar-prediction",
        (36, 76),
        lambda ax: box(ax, "62 bpm", w=0.66),
        right_title="one number, with a unit",
        window=(1.4, 5.0),
    )


def _localization_fig(name, spans, points, *, right_title, caption_text):
    """Build a localization exemplar: a query plus the series in, a set of regions out."""
    t = np.linspace(0, 8, 500)
    fig = plt.figure(figsize=(6.8, 2.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[2, 1.2], left=0.03, right=0.97, top=0.72, bottom=0.18, wspace=0.5)
    left = outer[0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
    ax = fig.add_subplot(left[0])
    ax.plot(t, wave(t, 37), color=BLUE, lw=1.6)
    bare(ax)
    ax.margins(y=0.28)
    ax.set_xlim(0, 8)
    draw_axes(ax)
    block_label(fig, ax, "time series")
    q_ax = fig.add_subplot(left[1])
    unit_axis(q_ax)
    q_ax.text(0.32, 0.5, "+", ha="center", va="center", fontsize=16, color=MUTED)
    q_ax.text(0.72, 0.5, "?", ha="center", va="center", fontsize=18, color=MUTED, fontweight="bold")
    block_label(fig, q_ax, "query")
    r_ax = fig.add_subplot(outer[1])
    _spans_axis(r_ax, spans, points)
    r_ax.set_title(right_title, fontsize=9, color=MUTED)
    arrow(fig, y=0.42)
    caption(fig, caption_text, y=0.95)
    save(fig, name)


def fig_task_localization_sparse():
    """TemporalLocalizationTask (sparse): a few regions out; unmarked time is unlabeled."""
    _localization_fig(
        "task-localization-sparse",
        spans=[(1.0, 2.2)],
        points=(4.1, 6.4),
        right_title="points and spans",
        caption_text="sparse: unmarked time is simply unlabeled",
    )


def fig_task_localization_exhaustive():
    """TemporalLocalizationTask (exhaustive): contiguous segments tile the recording."""
    _localization_fig(
        "task-localization-exhaustive",
        spans=[(0.0, 2.4), (2.4, 5.1), (5.1, 8.0)],
        points=(),
        right_title="segments that tile the span",
        caption_text="exhaustive: a gap is an error",
    )


def fig_task_forecasting():
    """ForecastingTask: observed past, predicted future, on one timeline."""
    t = np.linspace(0, 12, 700)
    y = wave(t, seed=34)
    cut = 430
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.18)
    ax.plot(t[:cut], y[:cut], color=GREY, lw=1.7)
    ax.plot(t[cut - 1 :], y[cut - 1 :], color=FORECAST, lw=1.9)
    ax.axvline(t[cut], color=AXIS, ls="--", lw=1.0)
    bare(ax)
    ax.margins(y=0.3)
    ax.set_xlim(t[0], t[-1])
    draw_axes(ax)
    caption(fig, "observed past → predicted future")
    save(fig, "task-forecasting")


def _series_out_fig(name, *, seed_in, seed_out, caption_text, prompt_only=False):
    """Build a series-out exemplar: an input panel (or a text spec), an arrow, and the produced series."""
    t = np.linspace(0, 8, 500)
    fig = plt.figure(figsize=(6.8, 2.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.03, right=0.97, top=0.70, bottom=0.2, wspace=0.42)
    left_ax = fig.add_subplot(outer[0])
    if prompt_only:
        unit_axis(left_ax)
        _text_block(left_ax)
        block_label(fig, left_ax, "a text spec")
    else:
        left_ax.plot(t, wave(t, seed_in), color=GREY, lw=1.6)
        bare(left_ax)
        left_ax.margins(y=0.3)
        left_ax.set_xlim(0, 8)
        draw_axes(left_ax)
        block_label(fig, left_ax, "the source series")
    right_ax = fig.add_subplot(outer[1])
    right_ax.plot(t, wave(t, seed_out), color=FORECAST, lw=1.7)
    bare(right_ax)
    right_ax.margins(y=0.3)
    right_ax.set_xlim(0, 8)
    draw_axes(right_ax)
    block_label(fig, right_ax, "the produced series")
    arrow(fig, x=0.5, y=0.45)
    caption(fig, caption_text, y=0.95)
    save(fig, name)


def fig_task_editing():
    """TSEditingTask: a series in, a transformed series out."""
    _series_out_fig(
        "task-editing", seed_in=38, seed_out=39, caption_text="an instruction transforms one series into another"
    )


def fig_task_generation():
    """TSGenerationTask: a text spec in, a new series out."""
    _series_out_fig(
        "task-generation",
        seed_in=0,
        seed_out=41,
        caption_text="a specification alone produces a series",
        prompt_only=True,
    )


def fig_task_correspondence():
    """TSCorrespondenceTask: a query series plus candidates in, the matching candidate out."""
    t = np.linspace(0, 8, 500)
    fig = plt.figure(figsize=(6.8, 2.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.03, right=0.93, top=0.70, bottom=0.16, wspace=0.42)
    query_ax = fig.add_subplot(outer[0])
    query_ax.plot(t, wave(t, 42), color=BLUE, lw=1.7)
    bare(query_ax)
    query_ax.margins(y=0.3)
    query_ax.set_xlim(0, 8)
    draw_axes(query_ax)
    block_label(fig, query_ax, "the query series")
    pool = [(43, False), (44, True), (45, False)]
    candidates = outer[1].subgridspec(len(pool), 1, hspace=0.5)
    for row, (seed, matched) in enumerate(pool):
        ax = fig.add_subplot(candidates[row])
        ax.plot(t, wave(t, seed), color=GREEN if matched else GREY, lw=1.5 if matched else 1.1)
        bare(ax)
        ax.margins(y=0.3)
        ax.set_xlim(0, 8)
        if matched:
            ax.text(8.2, 0.0, "✓", ha="left", va="center", fontsize=12, color=GREEN)
        last = row == len(pool) - 1
        draw_axes(ax, x_axis=last, time_label=last)
        if row == 0:
            block_label(fig, ax, "the candidates")
    arrow(fig, x=0.48, y=0.45)
    caption(fig, "which candidate corresponds to the query", y=0.95)
    save(fig, "task-correspondence")


# --- dataset + time series -------------------------------------------------------------------------


def fig_dataset():
    """A dataset: a versioned, immutable set of samples."""
    fig = plt.figure(figsize=(6.8, 2.7))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.12, left=0.03, right=0.97, top=0.66, bottom=0.05)
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        t = np.linspace(0, 6, 400)
        plot_series(ax, t, wave(t, seed=40 + i), lw=0.9)
        draw_axes(ax, time_label=False)
        ax.set_title(f"sample {i + 1}", fontsize=7.5, color=MUTED, pad=1)
    fig.text(0.5, 0.94, "org/name@1.0.0", ha="center", va="top", fontsize=12, color=INK, fontweight="bold")
    caption(fig, "a versioned, immutable set of samples", y=0.80)
    save(fig, "dataset-example")


def fig_time_series():
    """A time series: one channel of float32 values over time."""
    t = np.linspace(0, 8, 500)
    fig, ax = plt.subplots(figsize=(6.8, 1.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.2)
    plot_series(ax, t, wave(t, seed=50))
    ax.set_xlim(0, 8)
    ax.text(-0.03, 0.5, "channel_1", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=INK)
    draw_axes(ax)
    caption(fig, "one channel, float32 over time")
    save(fig, "time-series-example")


def main():
    """Generate every concept figure into ``docs/assets/figures``."""
    fig_dataset()
    fig_time_series()
    fig_annotation_static()
    fig_annotation_point()
    fig_annotation_interval()
    fig_cross_sensor()
    fig_task_classification()
    fig_task_classification_scoped()
    fig_task_answer_caption()
    fig_task_answer()
    fig_task_answer_rationale()
    fig_task_scalar_prediction()
    fig_task_localization_sparse()
    fig_task_localization_exhaustive()
    fig_task_forecasting()
    fig_task_editing()
    fig_task_generation()
    fig_task_correspondence()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
