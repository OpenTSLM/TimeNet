---
icon: lucide/terminal
description: "The timenet consumer CLI: find, read, download and publish TimeNet datasets from the shell."
tags:
  - cli
---

# `timenet`

The consumer command-line tool. It mirrors the [client SDK](../client.md). You can do the same tasks
from the shell that you do in Python. Install it with the `cli` extra:

```bash
uv add 'timenet[cli]'
```

By default, the command uses the hosted TimeNet registry:

```bash
timenet list
timenet search --query ecg --domain cardiology --limit 10
timenet info chengsenwang/tsqa@1.0.0   # pin a version
timenet download chengsenwang/tsqa
```

To use a local or S3 registry instead, see [Selecting the sources](#selecting-the-sources).

## Commands

| Command | What it does |
| --- | --- |
| `timenet list` | Print every dataset of the sources (id, name, license, domains), from the metadata of the highest version that a source lists. The cards of the installed connectors count too. See [`list()`](../client.md#browse-datasets). |
| `timenet search [flags]` | Filter the datasets of the sources, the cards of the installed connectors included. The flags map one-to-one to [`registry.search`](../registry.md#search). Repeat a flag for list values (`--spec` for `time_series_spec`, `--id` for `dataset_id`). `--limit` applies after the merge of the sources, and the search checks only the highest version of each dataset. |
| `timenet info <id>[@version]` | Show the [manifest](../manifest.md) of the release that `open()` reads: metadata, schema, and counts. For a release that only an installed connector provides, show its card, and the `timenet-build` command that adds the schema and the counts. See [`get_manifest()`](../client.md#inspect-a-release). |
| `timenet download <id>[@version] [--force]` | Copy a release and its data files into the local registry, and print the release directory. The release and each payload source that the local registry lacks come from the first registry that provides them. With `--force`, the release is copied again, also when the local registry holds a complete copy. It never builds. See [`download()`](../client.md#open-load-or-download). |
| `timenet publish <id>[@version] --dest` `URL` or `PATH` | Copy a complete release and its payload sources to the registry `--dest`, and print the release. The release comes from the local registry, which `--local-registry` selects, or from the registry `--source` `URL` or `PATH`, and without a version it is the latest version there. A published version cannot change, so another release under the version in `--dest` is an error. See [`publish()`](../build.md#programmatic-builds). |
| `timenet local info` | Print the path and size of the cache. Then print one line for the downloads and one line for the builds in the local registry, such as `Downloads in <path>: 2 releases, 1.4 GB`. |
| `timenet local clear [--yes]` | Delete the cache and every release in the local registry. The prompt counts the builds. |
| `timenet local cache info` | Print the path and size of the cache, `$TIMENET_CACHE`. |
| `timenet local cache clear [--yes]` | Delete the raw source files (`downloads/`) and the control databases (`control/`) of the cache that carry TimeNet's `CACHEDIR.TAG` file. |
| `timenet local downloads info` | List the downloads in the local registry: dataset, version, source, size, and the total. |
| `timenet local downloads clear [--yes]` | Delete the downloads in the local registry, except the ones that a release that stays needs. |
| `timenet local builds info` | List the builds in the local registry: dataset, version, source, size, and the total. |
| `timenet local builds clear [--yes]` | Delete the builds in the local registry, except the ones that a release that stays needs. |

Every command writes its status to stderr and its machine-readable result (a path or a release) to
stdout, so you can capture `timenet download <id>` in a script safely.

Most errors end with a hint that names a command or a flag that helps.

`list` and `search` leave out a source without a catalog and a connector card that cannot be read.
For each, they print a warning line to stderr, without a Python file and line number.

`--quiet`/`-q` removes the status output. It belongs to `timenet` itself. Therefore, it goes before
the subcommand: `timenet --quiet list`, not `timenet list --quiet`. Be careful with the collision:
after `search`, `-q` is the short form of `--query`.

## Downloads and builds in the local registry

The local registry holds two kinds of release, downloads and local builds. Releases can depend on
each other (parent-child relationships). In that case `timenet local downloads clear` and
`timenet local builds clear` might keep releases that are depended on, and print
`Kept <release>: <release> needs it.` for each. Because `local clear` deletes everything, no
dependency relationships can persist, and everything is removed. A symlinked dataset or release
folder in the local registry stops every clear, `local clear` included, before it deletes anything.
A provenance record that is not valid also stops them. Its hint names
`timenet download <id>@<version> --force` and `timenet-build build <id>@<version> --force`, which
replace the release and its record. Do not run a clear while a build or a download runs.

## Selecting the sources

`list`, `search`, `info`, and `download` create a client with the default list of sources,
`[local registry, remote registry, installed connectors]` (see
[Choose the sources](../client.md#choose-the-sources)). Three flags set this list, and each flag
overrides one [variable](../client.md#settings):

| Flag | Variable | Default | What it sets |
| --- | --- | --- | --- |
| `--local-registry PATH` | `TIMENET_LOCAL_REGISTRY` | `~/.timenet/registry` | The local registry, where downloads and builds land and which a lookup checks first. |
| `--remote-registry` `URL` or `PATH` | `TIMENET_REMOTE_REGISTRY` | `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the local registry does not have the dataset. |
| `--auto-build` / `--no-auto-build` | `TIMENET_AUTO_BUILD` | `true` | Whether installed connectors build a dataset that neither the local nor the remote registry has. |

A flag wins over its variable, and the variable wins over its default. Flags go after the
subcommand:

```bash
timenet list --remote-registry https://registry.example.com
timenet download chengsenwang/tsqa --local-registry ./data --no-auto-build
```

`publish` copies out of the local registry by default. To mirror a hosted release in a registry of
your own, name the hosted registry as the source:

```bash
timenet publish org/name --source timenet:// --dest s3://my-bucket/timenet
```

## Pinning versions

Add the suffix `@<version>` to an id to pin it. Without a suffix, the command reads the newest
version of the first source that serves the dataset, so a copy in the local registry reads offline.
`@latest` reads the highest version of all sources:

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # newest of the first source
timenet info chengsenwang/tsqa@latest    # highest of all sources
```

If you pin a version that is not committed, the tool exits with a `TimeNetDatasetNotFoundError`.
