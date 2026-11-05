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
timenet info chengsenwang/tsqa@1.0.0   # pin a version; omit @ for the latest
timenet download chengsenwang/tsqa
```

To use a local or S3 registry instead, see [Selecting the sources](#selecting-the-sources).

## Commands

| Command | What it does |
| --- | --- |
| `timenet list` | Print every dataset of the sources (id, name, license, domains), from the metadata of the highest version that a source lists. The cards of the installed connectors count too. See [`list()`](../client.md#browse-datasets). |
| `timenet search [flags]` | Filter the datasets of the sources, the cards of the installed connectors included. The flags map one-to-one to [`registry.search`](../registry.md#search). Repeat a flag for list values (`--spec` for `time_series_spec`, `--id` for `dataset_id`). `--limit` applies after the merge of the sources. |
| `timenet info <id>[@version]` | Show the [manifest](../manifest.md) of the release that `open()` reads: metadata, schema, and counts. For a release that only an installed connector provides, show its card, and the `timenet-build` command that adds the schema and the counts. See [`get_manifest()`](../client.md#inspect-a-release). |
| `timenet download <id>[@version]` | Copy a release and its data files into the store, and print the release directory. The release and each payload source that the store lacks come from the first registry that provides them. It never builds. See [`download()`](../client.md#open-load-or-download). |
| `timenet publish <id>[@version] --dest URL` | Copy a complete release and its payload sources to the registry `--dest`, and print the release. The release comes from the store, which `--local-registry` selects, or from the registry `--source URL`, and without a version it is the latest version there. A remote `--source` reads the complete copies of the same releases in the store. A published version cannot change, so another release under the version in `--dest` is an error. See [`publish()`](../build.md#programmatic-builds). |
| `timenet local info` | Print the path and size of the cache. Then print one line for the downloads and one line for the builds in the store, such as `Downloads in <path>: 2 releases, 1.4 GB`. |
| `timenet local clear [--yes]` | Delete the cache and every release in the store. The prompt counts the builds. |
| `timenet local cache info` | Print the path and size of the cache, `$TIMENET_CACHE`. |
| `timenet local cache clear [--yes]` | Delete the raw source files (`downloads/`) and the control databases (`control/`) of the cache that carry TimeNet's `CACHEDIR.TAG` file. |
| `timenet local downloads info` | List the downloads in the store: dataset, version, source, size, and the total. |
| `timenet local downloads clear [--yes]` | Delete the downloads in the store, except the ones that a release that stays needs. |
| `timenet local builds info` | List the builds in the store: dataset, version, source, size, and the total. |
| `timenet local builds clear [--yes]` | Delete the builds in the store, except the ones that a release that stays needs. |

Every command writes its status to stderr and its machine-readable result (a path or a release) to
stdout, so you can capture `timenet download <id>` in a script safely. `timenet download` and
`timenet publish` show the same progress as `timenet-build`: a status line when the copy of each
release starts and when it ends. For a copy into a local registry, a terminal also shows a progress
bar for each release, and other output gets a line at about every 20% of the copy. If
`timenet download` copies nothing, it prints
`Nothing to do: '<dataset_id>' <version> is already in local registry <store>`.

An error ends with a hint that names a command or a flag that helps. A command in a hint repeats
the registry flags of the command that failed. For example:

- If no source provides the release, the hint names `--remote-registry <url>`. For `info`, it also
  names `--auto-build` if an installed connector builds the release.
- If only an installed connector provides the release, `timenet download` names
  `timenet-build build <id>`. If a registry holds another version, it also names
  `timenet download <id>@<version>`.
- For a damaged copy in the store, the hint names `timenet download` or
  `timenet-build build ... --force`.
- If a registry cannot be reached, the hint names `--remote-registry <url>` for a registry that you
  can reach.

If the store lacks the release that `timenet publish` copies, the hint of the error names the
command that puts it there. This is `timenet download` if the remote registry holds the release, or
`timenet-build build` if an installed connector builds it. A `--dest` that is not a local registry
holds only open releases. For a release without open access, the command raises
`TimeNetAccessError`, and the hint names a registry of your own, such as a local directory.

`--quiet`/`-q` removes the status output. It belongs to `timenet` itself. Therefore, it goes before
the subcommand: `timenet --quiet list`, not `timenet list --quiet`. Be careful with the collision:
after `search`, `-q` is the short form of `--query`.

## Downloads and builds in the store

The store holds two kinds of release. A download is a release whose
[provenance record](../client.md#the-store) says `copy`. A build is a release whose record says
`build`, or a release without a record. `timenet local downloads clear` and
`timenet local builds clear` keep each release that a release that stays needs: its payload
sources, and the parents of a build. This holds in both directions, so `builds clear` keeps a build
that a download reads files of. For each release that a command keeps, it prints
`Kept <release>: <release> needs it.` `timenet local clear` deletes everything without this check.

`downloads clear` and `builds clear` stop before they delete anything if a dataset or release
folder in the store is a symlink, or if the manifest of a release that stays is not valid. For a
manifest that is not valid, the hint says
`Delete <dir> and download or build <release> again, then clear again.` Do not run a clear while a
build or a download runs.

## Selecting the sources

`list`, `search`, `info`, and `download` create a client with the default list of sources,
`[store, remote registry, installed connectors]` (see
[Choose the sources](../client.md#choose-the-sources)). Three flags set this list, and each flag
overrides one [variable](../client.md#settings):

| Flag | Variable | Default | What it sets |
| --- | --- | --- | --- |
| `--local-registry PATH` | `TIMENET_LOCAL_REGISTRY` | `~/.timenet/registry` | A local registry: the first source that a lookup checks, and the place where downloads and builds land. This is the store. |
| `--remote-registry URL` | `TIMENET_REMOTE_REGISTRY` | `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the local registry does not have the dataset. |
| `--auto-build` / `--no-auto-build` | `TIMENET_AUTO_BUILD` | `true` | Whether installed connectors build a dataset that neither the local nor the remote registry has. |

A flag wins over its variable, and the variable wins over its default. A registry flag that names
no registry is a usage error. The command exits with code 2 and names the flag. This applies to
`--local-registry`, `--remote-registry`, `--dest` and `--source`. A blank value, an unknown scheme,
and a URL for `--local-registry` name no registry. A bad `TIMENET_*` variable stops the command
with exit code 1, and the error names the variable. `--remote-registry` takes a local path or a
`file://`, `s3://`, `http(s)://` or `timenet://` URL, and the flags go after the subcommand:

```bash
timenet list --remote-registry https://registry.example.com
timenet download chengsenwang/tsqa --local-registry ./data --no-auto-build
```

`list` and `search` merge every source of the list. `info` resolves the release through the whole
list, as `open()` does. `download` resolves the release through the whole list and copies it
into the store. `publish` and the `timenet local` commands do not use the list of sources. Of the
three flags, `publish`, `local info`, `local clear`, and the `local downloads` and `local builds`
commands take only `--local-registry`. The `local cache` commands take none of them.

`publish` copies out of the store by default. To mirror a hosted release in a registry of your own,
name the hosted registry as the source:

```bash
timenet publish org/name --source timenet:// --dest s3://my-bucket/timenet
```

The build tool writes to the store by default, and `list`, `search` and `info` read the store too
(see [Inspect a local build](build.md#inspect-a-local-build)).

## Pinning versions

Add the suffix `@<version>` to an id to pin it. You can also use `@latest` (the default when you omit
the suffix):

```bash
timenet info chengsenwang/tsqa@1.0.0     # pinned
timenet download chengsenwang/tsqa       # latest
```

If you pin a version that is not committed, the tool exits with a `TimeNetDatasetNotFoundError`.
