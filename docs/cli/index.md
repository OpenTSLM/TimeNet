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
| [`timenet`](timenet.md) | `timenet` | Consumer: find, read, download and publish TimeNet datasets. | The [client SDK](../client.md). |
| [`timenet-build`](build.md) | `timenet-connectors` | Producer: run a connector through the pipeline into the local registry, and publish the result. | [Build & publish](../build.md). |

Both tools use [Typer](https://typer.tiangolo.com). The `timenet` command needs the `cli` extra:

```bash
uv add 'timenet[cli]'
```

If you run it without the extra, it prints a one-line install hint instead of a traceback.

Both tools split their output the same way: status lines go to stderr, the machine-readable result (a
path or a version) goes to stdout. `--quiet`/`-q` silences the status. The flag belongs to the tool,
not the subcommand, so it comes first: `timenet --quiet list`, `timenet-build --quiet build <id>`.

Both tools read the same [settings](../client.md#settings). The consumer copies into the local
registry, `$TIMENET_LOCAL_REGISTRY`, and the build tool writes to it. The commands that look up
datasets take `--local-registry`, `--remote-registry`, and `--auto-build`/`--no-auto-build`, which
override these settings for one command (see
[Selecting the sources](timenet.md#selecting-the-sources)).

If you want to use datasets, start with [`timenet`](timenet.md). If you build a dataset, start with
[`timenet-build`](build.md).
