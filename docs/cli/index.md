---
icon: lucide/terminal
description: "TimeNet's two command-line tools: timenet for consumers and timenet-build for producers."
tags:
  - cli
---

# CLI

TimeNet ships two console scripts, one for each side of the workflow:

| Command | Ships with | Role | Mirrors |
| --- | --- | --- | --- |
| [`timenet`](timenet.md) | `timenet` | Consumer: browse a registry and download datasets. | The [client SDK](../client.md). |
| [`timenet-build`](build.md) | `timenet-connectors` | Producer: run a connector through the pipeline into a registry. | [Build & publish](../build.md). |

Both tools use [Typer](https://typer.tiangolo.com). The `timenet` command needs the `cli` extra:

```bash
uv add 'timenet[cli]'
```

If you run it without the extra, it prints a one-line install hint instead of a traceback.

Both tools split their output the same way: status lines go to stderr, the machine-readable result (a
path) goes to stdout. `--quiet`/`-q` silences the status. The flag belongs to the tool, not the
subcommand, so it comes first: `timenet --quiet list`, `timenet-build --quiet build <id>`.

Both tools honor `$TIMENET_REGISTRY`, but their final defaults differ. The consumer uses the hosted
registry when no registry is set. The build tool writes to the local registry. Pass `--registry` or
`--out` when a workflow must use a specific location.

If you want to use datasets, start with [`timenet`](timenet.md). If you build a dataset, start with
[`timenet-build`](build.md).
