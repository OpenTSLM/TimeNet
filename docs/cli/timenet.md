---
icon: lucide/terminal
description: "The timenet consumer CLI: browse a registry and download datasets from the shell."
tags:
  - cli
---

# `timenet`

The consumer command-line tool. It mirrors the [client SDK](../client.md). You can do the same tasks
from the shell that you do in Python. Install it with the `cli` extra:

```bash
pip install 'timenet[cli]'
```

You must build a registry first (see [Get started](../get-started.md)). Then you can browse and
download datasets:

```bash
timenet list
timenet search --query ecg --domain cardiology --limit 10
timenet info chengsenwang/tsqa@1.0.0   # pin a version; omit @ for the latest
timenet download chengsenwang/tsqa
```

## Commands

| Command | What it does |
| --- | --- |
| `timenet list` | Print every dataset in the registry with its latest version. |
| `timenet search [flags]` | Filter datasets. The flags map one-to-one to [`registry.search`](../registry.md#search). Repeat a flag for list values (`--spec` for `time_series_spec`, `--id` for `dataset_id`). |
| `timenet info <id>[@version]` | Show a dataset's [manifest](../manifest.md): metadata, schema, and counts. |
| `timenet download <id>[@version]` | Copy a version's files into local storage and print the directory. `--storage <dir>` picks the target (else `$TIMENET_STORAGE`, then `<home>/storage`). If a local version already exists, it skips the copy. |
| `timenet cache info` | List downloaded datasets on disk (location, id, version, size) and the total. |
| `timenet cache clear` | Remove downloads and the raw cache. It prompts first. `-y` skips the prompt. `--all` also clears built data. |

Every command writes its status to stderr. It writes its machine-readable result (a path) to stdout.
Therefore, you can capture `timenet download <id>` in a script safely.

`--quiet`/`-q` removes that status output. It belongs to `timenet` itself. Therefore, it goes before
the subcommand: `timenet --quiet list`, not `timenet list --quiet`. Be careful with the collision:
after `search`, `-q` is the short form of `--query`.

## Selecting a registry

The tool resolves the registry in this order: `--registry`, then `$TIMENET_REGISTRY`, then the local
default (`<home>/registry`). [`timenet-build build`](build.md) resolves the same way. Therefore, the
tool that writes a dataset and the tool that reads it always agree. Today, only local registries serve
data. The `s3://` and hosted backends are [deferred](../registry.md). See
[Configuration](../client.md#configuration) for the storage and cache paths that the commands read and
write.

## Pinning versions

Add the suffix `@<version>` to an id to pin it. You can also use `@latest` (the default when you omit
the suffix):

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # latest
```

If you pin a version that is not committed, the tool exits with a `TimeNetDatasetNotFoundError`.
