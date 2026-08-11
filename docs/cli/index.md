---
icon: lucide/terminal
description: "TimeNet's two command-line tools: timenet for consumers and timenet-curate for producers."
tags:
  - cli
---

# CLI

TimeNet ships two console scripts, one for each side of the workflow:

| Command | Ships with | Role | Mirrors |
| --- | --- | --- | --- |
| [`timenet`](timenet.md) | `timenet` | Consumer: browse a registry and download datasets. | The [client SDK](../client.md). |
| [`timenet-curate`](curate.md) | `timenet-connectors` | Producer: run a connector through the pipeline into a registry. | [Curate & publish](../curation.md). |

Both tools use [Typer](https://typer.tiangolo.com). They keep their heavier dependencies behind an
extra, so the core package stays small. `timenet` needs the `cli` extra (`pip install 'timenet[cli]'`).
If you run it without the extra, it prints a one-line install hint instead of a traceback.

Both tools split their output the same way: status lines go to stderr, the machine-readable result (a
path) goes to stdout. `--quiet`/`-q` silences the status. The flag belongs to the tool, not the
subcommand, so it comes first: `timenet --quiet list`, `timenet-curate --quiet build <id>`.

Every command reads the same configuration as the SDK. TimeNet resolves the registry to use in this
order: the `--registry` flag (or `--out` for `timenet-curate build`), then `$TIMENET_REGISTRY`, then the
local default at `<home>/registry`. See [Configuration](../client.md#configuration) for the full set of
`TIMENET_*` variables.

If you want to use datasets, start with [`timenet`](timenet.md). If you build a dataset, start with
[`timenet-curate`](curate.md).
