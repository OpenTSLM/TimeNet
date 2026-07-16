# Curation CLI

`timenet-curate` is the producer command-line interface, run from a `timenet-connectors` checkout. It
drives a [connector](connectors.md) through the [engine](engine.md) and produces a dataset-layout
directory (a `manifest.json` plus its parquet) ready for a [registry](registry.md). It ships with
`timenet-connectors`, distinct from the consumer [`timenet`](client.md#cli) CLI. Built with
[Typer](https://typer.tiangolo.com).

## `timenet-curate build`

```bash
timenet-curate build <dataset_id> --out <dir>
```

Runs the connector for `<dataset_id>` through the engine (`download -> convert -> derive_schema ->
store`) and writes the dataset into `<dir>`. The output directory is itself a valid local registry, so
you can verify the result immediately:

```bash
timenet-curate build timenet/hello-world --out ./local_registry
python -c "from timenet.client import TimeNet; print(TimeNet('./local_registry').list())"
```

The curator loop: add a connector at `datasets/<org>/<name>/` (its `__init__.py` a
[`BaseConnector`](connectors.md) exposing `CONNECTOR`) with its [dataset card](manifest.md),
`dataset.yaml`, beside it, `build` locally, verify with the SDK, then publish. The card is validated
against the packaged [`dataset-card.schema.json`](https://ai-x-labs.github.io/TimeNet/schemas/dataset-card-v1.schema.json)
when the connector loads it (`BaseConnector` reads `dataset.yaml` by convention; set `CARD` to override
the path). This needs the `timenet[curation]` extra, pulled in by `timenet-connectors`. Additional verbs
(`validate`, `inspect`, `publish`) and a remote registry backend are planned.
