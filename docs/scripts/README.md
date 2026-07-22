# Docs figures

Sources for the concept figures in the docs. Two kinds, handled differently.

## Signal plots (matplotlib -> SVG)

`gen_concept_figures.py` draws the schematic concept figures (one plain signal plus the minimum marks
that explain a span, a point, or an input-to-output arrow) and writes them to `../assets/figures/*.svg`.
They carry no domain detail; that lives in the page HTML around each image. Those SVGs are committed and
referenced from the doc pages. This is **not** wired into `make docs`; regenerate by hand when the
concepts change:

```bash
uv run docs/scripts/gen_concept_figures.py
```

The script is a self-contained `uv` script (matplotlib + numpy declared inline). All randomness is
seeded, so re-running produces byte-stable output. Set `FIG_PNG_DIR=/some/dir` to also drop PNG copies
there for eyeballing; the committed artifacts are SVG only.

Figures produced: `dataset-example`, `time-series-example`, `annotation-static`, `annotation-point`,
`annotation-interval`, `cross-sensor`, and `task-classification`, `task-labeling`, `task-captioning`,
`task-qa`, `task-reasoning`, `task-forecasting`.

## Structural diagrams (mermaid, inline)

The pipeline, sample-composition, and annotation/task diagrams are **mermaid**, embedded directly in
the doc pages as ` ```mermaid ` code blocks. Zensical renders them client-side (the `superfences`
mermaid fence is already enabled in `zensical.toml`), so they pick up the site fonts and adapt to the
light and dark colour schemes automatically. There is nothing to pre-render.

The diagram definitions live under `diagrams/` (`pipeline.mmd`, `sample.mmd`, `annotation-task.mmd`) as
the source of truth; paste their contents into a ` ```mermaid ` block on the relevant page. They are
kept plain (no hard-coded theme or colours) so Zensical's theming applies.
