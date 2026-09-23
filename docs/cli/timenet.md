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
uv add 'timenet[cli]'
```

By default, the command uses the hosted TimeNet registry. You can also select a local or S3
registry:

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

The tool resolves the registry in this order: `--registry`, then `$TIMENET_REGISTRY`, then
`timenet://`. The last value selects the hosted registry. Local paths, `file://`, `s3://`,
`http(s)://`, and `timenet://` are valid selectors.

The build tool has a different final default: it writes to the local registry. Pass the same explicit
path to both tools when you build and inspect a dataset locally.

## Pinning versions

Add the suffix `@<version>` to an id to pin it. You can also use `@latest` (the default when you omit
the suffix):

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # latest
```

If you pin a version that is not committed, the tool exits with a `TimeNetDatasetNotFoundError`.
