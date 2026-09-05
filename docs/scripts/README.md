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
`annotation-interval`, `cross-sensor`, and `task-classification`, `task-classification-scoped`,
`task-answer`, `task-answer-caption`, `task-answer-rationale`, `task-scalar-prediction`,
`task-localization-sparse`, `task-localization-exhaustive`, `task-forecasting`, `task-editing`,
`task-generation`, `task-correspondence`.

## Structural diagrams (mermaid, inline)

The pipeline, record-composition, and annotation/task diagrams are **mermaid**, written directly in the
doc pages as ` ```mermaid ` code blocks (the `superfences` mermaid fence is enabled in `zensical.toml`).
The block on the page is the source; there is nothing to pre-render and no separate `.mmd` files to keep
in sync. Zensical renders them client-side, so they pick up the site fonts and adapt to the light and
dark colour schemes automatically. Keep the blocks plain (no hard-coded theme or colours).
