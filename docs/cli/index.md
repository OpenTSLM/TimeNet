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

Both are built with [Typer](https://typer.tiangolo.com) and gate their heavier dependencies behind an
extra, so the core package stays small. `timenet` needs the `cli` extra (`pip install 'timenet[cli]'`);
run it without the extra and it prints a one-line install hint instead of a traceback.

Every command reads the same configuration the SDK does. The registry to talk to is resolved as
`--registry` flag, then `$TIMENET_REGISTRY`, then the local default at `<home>/registry`. See
[Configuration](../client.md#configuration) for the full set of `TIMENET_*` variables.

Start with [`timenet`](timenet.md) if you want to use datasets, or [`timenet-curate`](curate.md) if you
are building one.
