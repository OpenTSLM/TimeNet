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

Both tools use [Typer](https://typer.tiangolo.com). `timenet` keeps its heavier dependencies behind an
extra. This choice keeps the core package small. `timenet` needs the `cli` extra (`pip install
'timenet[cli]'`) to use these dependencies. Without the extra, `timenet` prints a one-line install
hint instead of a traceback. `timenet-connectors` ships Typer and Rich without an extra. It has no
use case outside the CLI, so it does not need to stay small.

Both tools split their output the same way. Status lines go to stderr. The machine-readable result
(a path or version identifier) goes to stdout. `--quiet`/`-q` silences the status. The flag belongs
to the tool, not the subcommand, so it comes first: `timenet --quiet list`,
`timenet-build --quiet build <id>`.

Every command reads the same configuration as the SDK. Consumer commands
(`timenet list`/`search`/`info`/`download`) resolve the registry in this order: the `--registry`
flag, then `$TIMENET_REGISTRY`, then the hosted registry (`timenet://`). `timenet-build build`'s
`--out` resolves the same first two steps. Then it falls back to the local default
(`<home>/registry`) instead, because a build needs a place to write. See
[Configuration](../client.md#configuration) for the full set of `TIMENET_*` variables.

If you want to use datasets, start with [`timenet`](timenet.md). If you build a dataset, start with
[`timenet-build`](build.md).
