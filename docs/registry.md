---
icon: lucide/database
description: "Registry backends that serve compiled TimeF versions to the SDK."
tags:
  - guide
  - registry
---

# Registry

A registry serves compiled TimeF versions to the SDK. A compiled TimeF version has manifests,
Parquet control tables, and a values plane in Parquet or Zarr format. A registry never runs
connector code. The registry code lives in `timenet.registry`.

You can have several registries: a public registry, private internal registries, or a local
directory. The output of [curation](curation.md) is itself a valid local registry.

## Choosing a registry

```python
from timenet.registry import open_registry

registry = open_registry("./local_registry")
```

`open_registry(uri)` dispatches by scheme:

| Scheme | Backend |
| --- | --- |
| `file://`, plain path | `LocalRegistry` |
| `s3://` | `S3Registry` (deferred) |
| `http(s)://` | `RemoteRegistry` (deferred) |
| `timenet://` | `RemoteRegistry`, an alias for the hosted `https://registry.timenet.ai` |

An unrecognized scheme, for example `gs://` or `az://`, raises `ValueError` instead of becoming a
local path. `open_registry` expands `~` in a local path or a `file://` URI.

!!! note "What works today"
    `LocalRegistry` and the `http(s)://` / `timenet://` `RemoteRegistry` serve reads, and both publish
    with `store` (remote publishing needs a writer token). The `s3://` backend still raises
    `NotImplementedError` until it lands.

### Remote registries

`http(s)://` and `timenet://` both select `RemoteRegistry`. `timenet://` is a shorthand for the hosted
`https://registry.timenet.ai` service root and takes no path: dataset ids are passed to the client
methods, not folded into the registry URI. An `http(s)://` URI is used verbatim as the service root,
which is how you point at a private or a dev deployment.

The HTTP transport authenticates with a bearer token read from `$TIMENET_TOKEN`. When it is unset the
SDK talks to the registry anonymously (no `Authorization` header), which is enough for public reads.
Set it to reach private datasets or to write:

```bash
export TIMENET_TOKEN=your-token
```

## `BaseRegistry`

Every backend implements this contract: four data-access methods and one shared `search` method.

| Method | Description |
| --- | --- |
| `list_datasets()` | Latest-version `DatasetMetadata` for every dataset, sorted by id. |
| `get_manifest(dataset_id, version=None)` | A dataset's [manifest](manifest.md) (latest if `version` is `None`). Raises `DatasetNotFoundError` for an unknown id/version, or `TimeFFormatError` if the stored manifest's own id disagrees with the directory it was loaded from. |
| `open_file(dataset_id, version, relpath)` | A file of a dataset version, opened for seekable binary reading (an object store must return a range-capable handle, not a forward-only stream). |
| `open_version(dataset_id, version=None)` | A `DatasetVersion`: the parsed manifest plus a filesystem-rooted, picklable handle to the version's files. This is the storage seam the [reader](timef-reader.md) reads through, so a read never re-opens the registry nor re-parses `manifest.json`. |
| `search(...)` | Filter datasets (shared implementation). |

`LocalRegistry` serves a `<root>/<dataset_id>/<version>/` tree. `RemoteRegistry` serves the same layout
over the versioned REST contract (`GET /api/v1/datasets`, `/api/v1/datasets/{id}/{version}/manifest`,
`.../download/{relpath}`); `S3Registry` targets the same layout under an S3 prefix and still raises
`NotImplementedError`.

A dataset id is an `org/name` pair, for example `chengsenwang/tsqa`. The id nests one level deep on
disk, at `<root>/chengsenwang/tsqa/<version>/`. `list_datasets` finds these ids at any depth. Use
lowercase ids to avoid casing clashes on case-insensitive filesystems.

## Writing to a registry

A `WritableRegistry` adds one write method to the read contract. As a result,
[curation](curation.md) can publish into any backend, not only a local directory:

| Method | Description |
| --- | --- |
| `store(dataset, *, force=False, progress_cb=None)` | Compile a dataset and publish it; returns the stored version. Derives the schema first if absent, and skips an already-committed version unless `force`. |
| `exists(dataset_id, version)` | Whether a committed version already exists (shared implementation). |

`LocalRegistry` implements `store` by streaming the dataset through a [`TimeFWriter`](timef-writer.md),
which stages under `<version>.tmp-*` and publishes with a single atomic rename. `RemoteRegistry` runs
the [publish flow](#remote-publishing) below. `S3Registry` is still a write stub.
`open_writable_registry(uri)` resolves a URI like `open_registry` but returns a `WritableRegistry`.

```python
from timenet.registry import open_writable_registry

registry = open_writable_registry("./local_registry")
version = registry.store(dataset)   # schema derived if needed, atomic commit
```

Every backend is a `WritableRegistry`. As a result, this type alone does not show if a backend is a
directory the engine can write to or a remote stub. `local_registry_path(uri)` gives this
information. It returns the directory that a `file://` URI or a plain path names. It raises
`RegistryError` for a remote scheme. `default_registry_path()` uses `local_registry_path` to find
the default local registry. It uses `$TIMENET_REGISTRY` when its value is a local path. Otherwise,
it uses `<home>/registry`. This is how [`timenet-curate build`](cli/curate.md) resolves its output
when `--out` is absent. The `timenet_connectors.build` and `load` helpers use it too.

### Remote publishing

`RemoteRegistry.store` compiles the dataset locally, then hands the artifacts to the service over the
REST contract. Dataset bytes never pass through the API: the service issues a presigned PUT per file and
the SDK uploads straight to the object store. The token must be a writer token (`$TIMENET_TOKEN`); an
anonymous or read-only caller is rejected by the service.

1. Compile into a temporary staging directory with a [`TimeFWriter`](timef-writer.md), deriving the
   schema first if the dataset has none. An already-committed version returns early unless `force`.
2. `POST /datasets/{id}/{version}/publish` with the compiled `manifest.json` as the body. The service
   registers the pending version and returns the list of files it expects.
3. For each file, `POST /datasets/{id}/{version}/publish/upload-url` with `{"path": relpath}` to get a
   presigned grant (`{"url", "headers"}`), then `PUT` the file's bytes to that URL with the returned
   headers.
4. `POST /datasets/{id}/{version}/finalize` to commit the version. The staging directory is removed
   whether or not the publish succeeds.

```python
from timenet.registry import open_writable_registry

registry = open_writable_registry("timenet://")  # needs $TIMENET_TOKEN
version = registry.store(dataset)                 # compile, upload, finalize
```

## `search`

```python
registry.search(
    query=None, domain=None, task=None, license=None,
    time_series_spec=None, dataset_id=None, tag=None, limit=100,
)
```

Every filter takes a scalar value or a list. The registry ignores `None` filters and combines all
non-`None` filters with AND logic. The consumer CLI, `timenet search`, mirrors this behavior
exactly. `limit` caps the number of results, with a default of 100. `limit=0` returns no results.
A negative `limit` raises `ValueError`.

| Filter | Matches |
| --- | --- |
| `query` | any term is a case-insensitive substring of name/description/tags |
| `domain` | dataset shares any of these domains |
| `task` | dataset's schema includes any of these task classes (reads the manifest) |
| `license` | dataset has any of these licenses |
| `time_series_spec` | dataset declares all of these `spec_type` values (reads the manifest) |
| `dataset_id` | dataset id is any of these |
| `tag` | dataset declares all of these tags |

The type-filters, `task` and `time_series_spec`, resolve each dataset's schema from its committed
manifest. The manifest always carries the derived schema, so the registry does not need a
`precomputed_schema`.

---

See the [API reference for `timenet.registry`](api/registry.md) for the full symbol listing.
