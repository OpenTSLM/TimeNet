---
icon: lucide/factory
description: "Use timenet-build to convert a source and publish one TimeF version."
tags:
  - cli
---

# `timenet-build`

`timenet-build` is the producer CLI. It resolves an installed connector, runs the build pipeline,
writes one TimeF version into the store, and can publish it. It ships with `timenet-connectors`.

```bash
uv tool install timenet-connectors
```

The package includes the built-in connectors. Clone the source repository only when you want to
author or change a connector.

## Build command

```bash
timenet-build [--quiet] build <dataset_id>[@<version>] \
    [--force] [--keep-cache] [--isolation | --no-isolation] \
    [--publish-to <url>] [--dry-run] \
    [--local-registry <path>] [--remote-registry <url>] \
    [--auto-build | --no-auto-build]
```

| Option | Default | Purpose |
| --- | --- | --- |
| `<dataset_id>[@<version>]` | required | Select the connector by its `org/name` ID. A version must be the one that the card of the connector declares. |
| `--force`, `-f` | off | Build the release again in the store, even if the store holds it. |
| `--keep-cache` | off | Keep downloaded source files after a successful build. |
| `--isolation`, `--no-isolation` | `$TIMENET_ISOLATION`, else on | Run this build in an environment built from the requirements of the connector, or in the current one. |
| `--publish-to` | off | After the build, publish the release and its payload sources to the registry at this URL or path. |
| `--dry-run` | off | Print the plan to stdout and stop. It copies, builds, and publishes nothing. |
| `--local-registry` | `$TIMENET_LOCAL_REGISTRY`, else `~/.timenet/registry` | A local registry: the first source that a lookup checks, and the store where the build lands. |
| `--remote-registry` | `$TIMENET_REMOTE_REGISTRY`, else `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the store does not have a parent. |
| `--auto-build`, `--no-auto-build` | `$TIMENET_AUTO_BUILD`, else on | Whether installed connectors build a parent that neither the store nor the remote registry has. |
| `--quiet`, `-q` | off | Hide status output. Place this option before `build`. |

The build lands in the store, where `TimeNet()` reads. Each parent and payload source that the
store lacks comes from the default list of sources: first the remote registry, then the installed
connectors. `--local-registry`, `--remote-registry`, and `--auto-build`/`--no-auto-build` set that
list for one command, as they do for [`timenet`](timenet.md#selecting-the-sources). The connector
of the requested dataset always builds it, also with `--no-auto-build`. See
[Prepare parent releases](../build.md#prepare-parent-releases) for the steps.

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

1. The release builds into the store, as without `--publish-to`. Its parents land there too.
2. `timenet.publish()` copies the release, and each payload source that the destination lacks,
   from the store to the destination.

The destination is not a source of parents. To copy a parent from that registry, name it with
`--remote-registry` too.

If the destination holds the version and `--force` is not set, the command builds and copies
nothing. If the store holds the release too, the two must hold the same release: the command
compares both manifests without the build environment. For the same release, it says "Nothing to
do". For another release, it raises `TimeNetVersionConflictError` before any build. If the store
lacks the release, the version in the destination is enough, and the command says "Nothing to do".

`--force` builds the release again in the store. It cannot overwrite a published version. If the
new build wrote the same files, the publish copies nothing. If it wrote other files, the publish
raises `TimeNetVersionConflictError`, and the change needs a new version.

A parent that the build copied from `--remote-registry` can differ from the copy of that parent in
the destination. If the release reads files of that parent, the publish raises
`TimeNetVersionConflictError` for it, and copies nothing.

A registry that is not local holds only releases with access declared `open`, so for such a
destination the command checks the access of each release before any download.

## Output

Status goes to stderr. Before the first step, the command shows its plan as a tree: the requested
release at the top, and below each release the releases that it needs. Each line names a release,
what the build does with it, and the source: `build with`, `copy from`, or `read from`. A release
that the tree already shows reads `see above` the second time.

```text
📋 Plan for 'physionet/ecg-qa-cot' 1.2.0:
   physionet/ecg-qa-cot@1.2.0: build with installed connectors
   └── physionet/ptb-xl@1.0.0: copy from registry https://registry.timenet.ai
```

Then the command shows one line when each copy or build of a release starts and one when it ends.
If nothing runs, because the store holds the release or the destination the version, one line says
so instead of the plan. `--quiet` hides the plan and the status lines.

A build prints the release directory in the store to stdout, and with `--publish-to` the version.

```bash
VERSION_DIR=$(timenet-build --quiet build timenet/hello-world)
test -f "$VERSION_DIR/manifest.json"
```

This separation lets scripts capture the result without parsing progress messages.

`--dry-run` prints only the tree to stdout, and stops. It reads cards and manifests, but it copies,
builds, and publishes nothing. With `--publish-to`, a last line names the publish:

```bash
timenet-build build physionet/ecg-qa-cot --dry-run \
    --publish-to s3://bucket/registry
```

```text
physionet/ecg-qa-cot@1.2.0: build with installed connectors
└── physionet/ptb-xl@1.0.0: copy from registry https://registry.timenet.ai
then publish physionet/ecg-qa-cot@1.2.0 to S3 registry s3://bucket/registry
```

If nothing would run, the "Nothing to do" line goes to stderr, and stdout stays empty.
`timenet.plan_build()` returns the same plan in Python (see
[Programmatic builds](../build.md#programmatic-builds)).

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The version was built, published, or reused. |
| `1` | An expected TimeNet operation failed. |
| `2` | The command or one of its options is invalid, for example an unknown dataset ID. |

A registry flag that names no registry is a usage error, so the command exits with code 2 and
names the flag. This applies to `--local-registry`, `--remote-registry` and `--publish-to`. A blank
value, an unknown scheme, and a URL for `--local-registry` name no registry. A bad `TIMENET_*`
variable exits with code 1, and the error names the variable.

If an isolated build cannot make its environment, for example because `uv` is missing, the command
raises `TimeNetIsolationError`. Its hint names `--no-isolation`.

Unexpected programming errors keep their traceback.

## Inspect a local build

Build into a store of your own, and pass the same store to the consumer:

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
