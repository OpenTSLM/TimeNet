---
icon: lucide/terminal
description: "The timenet consumer CLI: browse a registry and download datasets from the shell."
tags:
  - cli
---

# `timenet`

The consumer command-line tool. It mirrors the [client SDK](../client.md), so anything you can do in
Python you can do from the shell. Install it with the `cli` extra:

```bash
pip install 'timenet[cli]'
```

With a registry in place (build one first, see [Get started](../get-started.md)), browse and pull:

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
| `timenet search [flags]` | Filter datasets. Flags map one-to-one to [`registry.search`](../registry.md#search) and repeat for list values (`--spec` for `time_series_spec`, `--id` for `dataset_id`). |
| `timenet info <id>[@version]` | Show a dataset's [manifest](../manifest.md): schema, counts, and files. |
| `timenet download <id>[@version]` | Copy a version's files into local storage and print the directory. `--storage <dir>` picks the target (else `$TIMENET_STORAGE`, then `<home>/storage`). Idempotent unless `--force`. |
| `timenet cache info` | List downloaded datasets on disk (location, id, version, size) and the total. |
| `timenet cache clear` | Remove downloads and the raw cache. Prompts first; `-y` skips the prompt, `--all` also clears curated data. |

Every command writes its status to stderr and its machine-readable result (a path) to stdout, so
`timenet download <id>` is safe to capture in a script.

`--quiet`/`-q` suppresses that status output. It belongs to `timenet` itself, so it goes before the
subcommand: `timenet --quiet list`, not `timenet list --quiet`. Watch the collision: after `search`,
`-q` is the short form of `--query`.

## Selecting a registry

The registry is resolved as `--registry`, then `$TIMENET_REGISTRY`, then the local default
(`<home>/registry`). [`timenet-curate build`](curate.md) resolves the same way, so the tool that
writes a dataset and the tool that reads it always agree. Only local registries serve data today; the
`s3://` and hosted backends are [deferred](../registry.md). See
[Configuration](../client.md#configuration) for the storage and cache paths the commands read and
write.

## Pinning versions

Suffix an id with `@<version>` to pin it, or use `@latest` (the default when you omit the suffix):

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # latest
```

Pinning a version that is not committed exits with a `DatasetNotFoundError`.
