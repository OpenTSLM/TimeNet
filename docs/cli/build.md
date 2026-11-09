---
icon: lucide/factory
description: "Use timenet-build to convert a source and publish one TimeF version."
tags:
  - cli
---

# `timenet-build`

`timenet-build` is the producer CLI. It resolves an installed connector, runs the build pipeline,
writes one TimeF version into the local registry, and can publish it. It ships with
`timenet-connectors`.

```bash
uv tool install timenet-connectors
```

The package includes the built-in connectors. Clone the source repository only when you want to
author or change a connector.

## Build command

```bash
timenet-build [--quiet] build <dataset_id>[@<version>] \
    [--force] [--keep-cache] [--isolation | --no-isolation] \
    [--publish-to <url|path>] [--dry-run] \
    [--local-registry <path>] [--remote-registry <url|path>] \
    [--auto-build | --no-auto-build]
```

| Option | Default | Purpose |
| --- | --- | --- |
| `<dataset_id>[@<version>]` | required | Select the connector by its `org/name` ID. A version must be the one that the card of the connector declares. |
| `--force`, `-f` | off | Build the release again in the local registry, even if the local registry holds it. |
| `--keep-cache` | off | Keep downloaded source files after a successful build. |
| `--isolation`, `--no-isolation` | `$TIMENET_ISOLATION`, else on | Run this build in an environment built from the requirements of the connector, or in the current one. |
| `--publish-to` | off | After the build, publish the release and its payload sources to the registry at this URL or path. |
| `--dry-run` | off | Print the plan to stdout and stop. It copies, builds, and publishes nothing. |
| `--local-registry` | `$TIMENET_LOCAL_REGISTRY`, else `~/.timenet/registry` | A local registry: the first source that a lookup checks, and the local registry where the build lands. |
| `--remote-registry` | `$TIMENET_REMOTE_REGISTRY`, else `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the local registry does not have a parent. |
| `--auto-build`, `--no-auto-build` | `$TIMENET_AUTO_BUILD`, else on | Whether installed connectors build a parent that neither the local registry nor the remote registry has. |
| `--quiet`, `-q` | off | Hide status output. Place this option before `build`. |

The build lands in the local registry, where `TimeNet()` reads.

`--local-registry` takes a local path or a `file://` URI. `--publish-to` and `--remote-registry`
accept:

- a local path or `file://` URI;
- `s3://bucket/prefix`;
- `http://` or `https://` for a registry API;
- `timenet://` for the hosted TimeNet registry.

Remote publishing needs the credentials required by that registry. For example, the hosted API can
use `TIMENET_TOKEN`.

## Publish after the build

`--publish-to` gets the release to another registry in two steps:

1. The release builds into the local registry, as without `--publish-to`. Its parents land there too.
2. `timenet.publish()` copies the release, and each payload source that the destination lacks,
   from the local registry to the destination.

If the destination holds the version and `--force` is not set, the command builds and copies
nothing. While `--force` builds the release again in the local registry. It cannot overwrite a published version. If the new build wrote the same files, the publish copies nothing. If it wrote other files, the publish
raises `TimeNetVersionConflictError`, and the change needs a new version.

## Output

Status goes to stderr. Before the first step, the command shows its plan as a tree: the requested
release at the top, and below each release the releases that it needs. Each line names a release,
what the build does with it, and the source:

```text
📋 Plan for 'physionet/ecg-qa-cot' 1.2.0:
   physionet/ecg-qa-cot@1.2.0: build with installed connectors
   └── physionet/ptb-xl@1.0.0: copy from registry https://registry.timenet.ai
```

Pass `--quiet` to hide the plan and the status lines.

```bash
VERSION_DIR=$(timenet-build --quiet build timenet/hello-world)
test -f "$VERSION_DIR/manifest.json"
```

This separation lets scripts capture the result without parsing progress messages.

`--dry-run` prints only the tree to stdout, and stops. It reads cards and manifests, but it copies,
builds, and publishes nothing.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The version was built, published, or reused. |
| `1` | An expected TimeNet operation failed. |
| `2` | The command or one of its options is invalid, for example an unknown dataset ID, or a registry flag that names no registry. |

If an isolated build cannot make its environment, for example because `uv` is missing, the command
raises `TimeNetIsolationError`.

Unexpected programming errors keep their traceback.

## Inspect a local build

Build into a local registry of your own, and pass the same registry to the consumer:

```bash
timenet-build build timenet/hello-world --local-registry .timenet-registry
timenet info timenet/hello-world --local-registry .timenet-registry
```

Or load it in Python:

```python
import timenet

client = timenet.TimeNet(sources=[".timenet-registry"])
dataset = client.load("timenet/hello-world")
```

See [Build and publish](../build.md) for the pipeline and [Connectors](../connectors.md) for the
authoring contract.
