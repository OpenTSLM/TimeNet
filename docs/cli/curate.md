---
icon: lucide/factory
description: "The timenet-curate producer CLI: run a connector through the pipeline into a registry."
tags:
  - cli
---

# `timenet-curate`

The producer command-line tool, run from a `timenet-connectors` checkout. It drives a
[connector](../connectors.md) through the [curation pipeline](../curation.md) and writes a
dataset-layout directory (a `manifest.json` plus its parquet) ready for a [registry](../registry.md).
It ships with `timenet-connectors`, separate from the consumer [`timenet`](timenet.md) tool, and needs
the `timenet[curation]` extra that `timenet-connectors` already pulls in.

## `timenet-curate build`

```bash
timenet-curate build <dataset_id> [--out <dir>] [--force] [--keep-cache]
```

Runs the connector for `<dataset_id>` through the pipeline (download, convert, derive_schema, store) and
writes the dataset into `--out <dir>`, defaulting to the local registry at `<home>/registry`. An
already-curated version is reused unless you pass `--force`/`-f`. The raw download cache is removed after
a successful build unless you pass `--keep-cache`. Add `--quiet`/`-q` to suppress progress output.

The output directory is itself a valid local registry, so you can verify a build straight away:

```bash
timenet-curate build timenet/hello-world --out ./local_registry
python -c "from timenet.client import TimeNet; print(TimeNet('./local_registry').list())"
```

## Planned commands

Only `build` exists today. `validate`, `inspect`, and `publish` (to an S3 or hosted registry) are
planned and land with the remote registry backends.

For the authoring loop behind these commands, see [Curate & publish](../curation.md) and
[Connectors](../connectors.md).
