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

Point it at a registry, then browse and pull:

```bash
export TIMENET_REGISTRY=~/.timenet/local   # a local registry (build one with timenet-curate)
timenet list
timenet search --query ecg --domain cardiology --limit 10
timenet info chengsenwang/tsqa@1.0.0       # pin a version; omit @ for the latest
timenet download chengsenwang/tsqa
```

## Commands

| Command | What it does |
| --- | --- |
| `timenet list` | Print every dataset in the registry with its latest version. |
| `timenet search [flags]` | Filter datasets. Flags map one-to-one to [`registry.search`](../registry.md#search) and repeat for list values (`--spec` for `time_series_spec`, `--id` for `dataset_id`). |
| `timenet info <id>[@version]` | Show a dataset's [manifest](../manifest.md): schema, counts, and files. |
| `timenet download <id>[@version]` | Copy a version's files into local storage and print the directory. Idempotent unless `--force`. |
| `timenet cache info` | List downloaded datasets on disk (location, id, version, size) and the total. |
| `timenet cache clear` | Remove downloads and the raw cache. Prompts first; `-y` skips the prompt, `--all` also clears curated data. |

## Selecting a registry

The registry is resolved as `--registry`, then `$TIMENET_REGISTRY`, then the local default
(`<home>/registry`). Only local registries serve data today; the `s3://` and hosted backends are
[deferred](../registry.md). See [Configuration](../client.md#configuration) for the storage and cache
paths the commands read and write.

## Pinning versions

Suffix an id with `@<version>` to pin it, or use `@latest` (the default when you omit the suffix):

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # latest
```

Pinning a version that is not committed exits with a `DatasetNotFoundError`.
