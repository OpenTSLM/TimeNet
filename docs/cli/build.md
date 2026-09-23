---
icon: lucide/factory
description: "Use timenet-build to convert a source and publish one TimeF version."
tags:
  - cli
---

# `timenet-build`

`timenet-build` is the producer CLI. It resolves an installed connector, runs the build pipeline,
and writes or publishes one TimeF version. It ships with `timenet-connectors`.

```bash
uv tool install timenet-connectors
```

The package includes the built-in connectors. Clone the source repository only when you want to
author or change a connector.

## Build command

```bash
timenet-build [--quiet] build <dataset_id> \
    [--out <registry>] \
    [--values-backend parquet|zarr] \
    [--force] [--keep-cache] [--no-isolation]
```

| Option | Default | Purpose |
| --- | --- | --- |
| `<dataset_id>` | required | Select the connector by its `org/name` ID. |
| `--out` | `$TIMENET_REGISTRY`, then the local registry | Select a local or remote output registry. |
| `--values-backend` | connector setting | Override the Parquet or Zarr values backend. |
| `--force`, `-f` | off | Rebuild or republish an existing version. |
| `--keep-cache` | off | Keep downloaded source files after a successful build. |
| `--no-isolation` | isolation on | Use the current environment instead of a connector-specific one. |
| `--quiet`, `-q` | off | Hide status output. Place this option before `build`. |

The output selector accepts:

- a local path or `file://` URI;
- `s3://bucket/prefix`;
- `http://` or `https://` for a registry API;
- `timenet://` for the hosted TimeNet registry.

Remote publishing needs the credentials required by that registry. For example, the hosted API can
use `TIMENET_TOKEN`.

## Output

Status goes to stderr. A local build prints the committed version directory to stdout. A remote
build prints the published version string.

```bash
VERSION_DIR=$(
    timenet-build --quiet build timenet/hello-world \
        --out .timenet-registry
)
test -f "$VERSION_DIR/manifest.json"
```

This separation lets scripts capture the result without parsing progress messages.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The version was built, published, or reused. |
| `1` | An expected TimeNet operation failed. |
| `2` | The command or one of its options is invalid. |

Unexpected programming errors keep their traceback.

## Inspect a local build

Use the same explicit registry for the build and the consumer:

```bash
timenet-build build timenet/hello-world --out .timenet-registry
timenet list --registry .timenet-registry
```

Or load it in Python:

```python
from timenet.client import TimeNet

dataset = TimeNet(".timenet-registry").load("timenet/hello-world")
```

See [Build and publish](../build.md) for the pipeline and [Connectors](../connectors.md) for the
authoring contract.
