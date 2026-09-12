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
extra, so the core package stays small: it needs the `cli` extra (`pip install 'timenet[cli]'`), and
without it prints a one-line install hint instead of a traceback. `timenet-connectors` ships Typer and
Rich unconditionally, since it has no non-CLI use case to keep slim.

Both tools split their output the same way: status lines go to stderr, the machine-readable result (a
path or version identifier) goes to stdout. `--quiet`/`-q` silences the status. The flag belongs to the
tool, not the subcommand, so it comes first: `timenet --quiet list`, `timenet-build --quiet build <id>`.

Every command reads the same configuration as the SDK. Consumer commands (`timenet list`/`search`/
`info`/`download`) resolve the registry in this order: the `--registry` flag, then `$TIMENET_REGISTRY`,
then the hosted registry (`timenet://`). `timenet-build build`'s `--out` resolves the same first two
steps, but falls back to the local default at `<home>/registry` instead, since a build needs a place to
write. See [Configuration](../client.md#configuration) for the full set of `TIMENET_*` variables.

If you want to use datasets, start with [`timenet`](timenet.md). If you build a dataset, start with
[`timenet-build`](build.md).
