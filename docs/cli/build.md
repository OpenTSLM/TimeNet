---
icon: lucide/factory
description: "The timenet-build producer CLI: run a connector through the pipeline into a registry."
tags:
  - cli
---

# `timenet-build`

The producer command-line tool. It drives a [connector](../connectors.md) through the
[build pipeline](../build.md). It writes a TimeF version directory that a
[registry](../registry.md) can use. The directory has a `manifest.json`, Parquet control tables, and
a Parquet or Zarr values plane. The tool ships with `timenet-connectors`, separate from the consumer
[`timenet`](timenet.md) tool. It needs the `timenet[build]` extra that `timenet-connectors`
already installs.

```bash
uv tool install timenet-connectors
```

The built-in connectors ship inside the package. The tool resolves them by dataset id, so that
install is enough to build them. Clone the repository only to *author* a connector.

## `timenet-build build`

```bash
timenet-build [--quiet] build <dataset_id> \
    [--out <dir>] [--force] [--keep-cache] [--values-backend <parquet|zarr>] [--no-isolation]
```

This command runs the connector for `<dataset_id>` through the pipeline: download, convert,
derive_schema, store. Then it writes the dataset into the output registry.

| Option | Default | What it does |
| --- | --- | --- |
| `<dataset_id>` | required | The `org/name` id to build. An unknown id lists the ones that exist. |
| `--out <dir>` | `$TIMENET_REGISTRY`, else `<home>/registry` | Where to write. If the directory is absent, the tool creates it. |
| `--force`, `-f` | off | Rebuild a version that is already built instead of reusing it. |
| `--keep-cache` | off | Keep the raw download cache. The tool removes it after a successful build. |
| `--values-backend <parquet\|zarr>` | the connector's declared backend | Override the values-plane storage backend. |
| `--isolation` / `--no-isolation` | on | Build in an environment made from the connector's `requirements.txt`. `--no-isolation` (or `TIMENET_ISOLATION=off`) builds in the current interpreter. |
| `--quiet`, `-q` | off | Suppress status output. Belongs to `timenet-build`, not to `build`. |

!!! warning "`--quiet` goes before the subcommand"
    `timenet-build --quiet build <id>` works. `timenet-build build <id> --quiet` exits `2` with
    `No such option '--quiet'`.

If `$TIMENET_REGISTRY` (or `--out`) names a remote registry (`timenet://`, `s3://`, `http(s)://`),
`build` publishes straight to it instead of writing a local directory, and prints the published version
to stdout instead of a path.

## Output streams

Status lines go to stderr. The committed version directory goes to stdout on its own. A script can
capture the path without parsing anything:

```bash
DIR=$(timenet-build --quiet build timenet/hello-world)
ls "$DIR"/manifest.json
```

`--quiet` silences the status lines but never warnings, errors, or that stdout path.

## Exit codes

| Code | When |
| --- | --- |
| `0` | The tool built the dataset, or reused an already-built version. |
| `1` | An expected failure (a `TimeNetError`): a one-line `Error: ...` on stderr, no traceback. |
| `2` | A usage error: an unknown dataset id, a bad flag. |

Anything else is a bug and surfaces its traceback.

## Verifying a build

The output directory is a valid local registry. You can point the SDK straight at it:

```bash
timenet-build build timenet/hello-world --out ./local_registry
python -c "from timenet.client import TimeNet; \
    print(TimeNet('./local_registry').list())"
```

## Planned commands

Only `build` exists today. Publishing to an S3 or hosted registry already works, through `--out` (see
above). `validate` and `inspect` are planned as separate subcommands.

For the authoring loop behind these commands, see [Build & publish](../build.md) and
[Connectors](../connectors.md).
