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
directory. The output of [build](build.md) is itself a valid local registry.

## Choosing a registry

```python
from timenet.registry import open_registry

registry = open_registry("./local_registry")
```

`open_registry(uri)` dispatches by scheme:

| Scheme | Backend |
| --- | --- |
| `file://`, plain path | `LocalRegistry` |
| `s3://` | `S3Registry` |
| `http(s)://` | `RemoteRegistry` |
| `timenet://` | `RemoteRegistry`, an alias for the hosted `https://registry.timenet.ai` |

An unrecognized scheme, for example `gs://` or `az://`, raises `ValueError` instead of becoming a
local path. `open_registry` expands `~` in a local path or a `file://` URI.

!!! note "Catalog support"
    Every backend serves reads and publishes with `store` (remote publishing needs a writer token; the
    S3 backend reads AWS credentials from the environment). The S3 backend has no catalog, so `list` and
    `search` are unsupported there.

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

### S3 registries

An `s3://bucket/prefix` URI opens an `S3Registry`. boto3 reads credentials, region, and an optional
endpoint override from the environment (`AWS_*` vars, `AWS_PROFILE`, `AWS_ENDPOINT_URL`), so nothing is
hardcoded. Install the extra:

```bash
pip install 'timenet[s3]'
```

The layout under the prefix matches the hosted registry's bucket convention,
`datasets/org/name/version/…` with `manifest.json` beside the data files. `store` uploads to a
temporary `version.tmp-<uuid>` prefix and moves each object into place
(`manifest.json` last), so a failed publish never leaves a partial version. `load` reads lazily through
a pyarrow `S3FileSystem` (range reads, no whole-version download), or from the local cache when a prior
`download` populated it. The backend has no catalog, so `list` and `search` are unsupported: publish and
fetch by an explicit `org/name@version`.

## `BaseRegistry`

Every backend implements this contract: four data-access methods and one shared `search` method.

| Method | Description |
| --- | --- |
| `list_datasets()` | Latest-version `DatasetMetadata` for every dataset, sorted by id. |
| `get_manifest(dataset_id, version=None)` | A dataset's [manifest](manifest.md) (latest if `version` is `None`). Raises `TimeNetDatasetNotFoundError` for an unknown id/version, or `TimeFFormatError` if the stored manifest's own id disagrees with the directory it was loaded from. |
| `open_file(dataset_id, version, relpath)` | A file of a dataset version, opened for binary reading. |
| `open_version(dataset_id, version=None)` | A `DatasetVersion`: the parsed manifest plus a filesystem-rooted, picklable handle to the version's files. This is the storage seam the [reader](timef-reader.md) reads through, so a read never re-opens the registry nor re-parses `manifest.json`. The filesystem of the handle must serve range reads, for out-of-order Parquet footer and value-slice reads. |
| `search(...)` | Filter datasets (shared implementation). |

`LocalRegistry` serves a `<root>/<dataset_id>/<version>/` tree. `RemoteRegistry` serves the same layout
over the versioned REST contract (`GET /api/v1/datasets`, `/api/v1/datasets/{id}/{version}/manifest`,
`.../download/{relpath}`); `S3Registry` serves the same layout under an `s3://bucket/prefix` root. The
S3 backend has no catalog, so `list_datasets` and `search` raise `NotImplementedError`.

A dataset id is an `org/name` pair, for example `chengsenwang/tsqa`. The id nests one level deep on
disk, at `<root>/chengsenwang/tsqa/<version>/`. `list_datasets` finds these ids at any depth. Use
lowercase ids to avoid casing clashes on case-insensitive filesystems.

## Writing to a registry

A `WritableRegistry` adds one write method to the read contract. As a result,
[build](build.md) can publish into any backend, not only a local directory:

| Method | Description |
| --- | --- |
| `store(dataset, *, force=False, values_backend="parquet", progress_cb=None)` | Compile a dataset and publish it; returns the stored version. Derives the schema first if absent, skips an already-committed version unless `force`, and `values_backend` (`"parquet"` or `"zarr"`) selects the values-plane backend. |
| `exists(dataset_id, version)` | Whether a committed version already exists (shared implementation). |

`LocalRegistry` implements `store` by streaming the dataset through a [`TimeFWriter`](timef-writer.md),
which stages under `<version>.tmp-*` and publishes with a single atomic rename. `RemoteRegistry` runs
the [publish flow](#remote-publishing) below. `S3Registry` uploads to a temporary prefix and moves each
object into place (`manifest.json` last).
`open_writable_registry(uri)` resolves a URI like `open_registry` but returns a `WritableRegistry`.

```python
from timenet.registry import open_writable_registry

registry = open_writable_registry("./local_registry")
version = registry.store(dataset)   # schema derived if needed, atomic commit
```

Every backend is a `WritableRegistry`. As a result, this type alone does not show if a backend is a
directory the engine can write to or a remote stub. `local_registry_path(uri)` gives this
information. It returns the directory that a `file://` URI or a plain path names. It raises
`TimeNetRegistryError` for a remote scheme. `default_registry_path()` uses `local_registry_path` to find
the default local registry. It uses `$TIMENET_REGISTRY` when its value is a local path. Otherwise,
it uses `<home>/registry`. This is how [`timenet-build build`](cli/build.md) resolves its output
when `--out` is absent. The `timenet_connectors.builder` and `load` helpers use it too.

### Remote publishing

`RemoteRegistry.store` compiles the dataset locally, then hands the artifacts to the service over the
REST contract. Dataset bytes never pass through the API: the service issues a presigned PUT per file and
the SDK uploads straight to the object store. The token must be a writer token (`$TIMENET_TOKEN`); an
anonymous or read-only caller is rejected by the service.

1. Compile into a temporary staging directory with a [`TimeFWriter`](timef-writer.md), deriving the
   schema first if the dataset has none. An already-committed version returns early unless `force`.
2. `POST /api/v1/datasets/{id}/{version}/publish` with the compiled `manifest.json` as the body. The
   service registers the pending version and returns the list of files it expects.
3. For each file, `POST /api/v1/datasets/{id}/{version}/publish/upload-url` with `{"path": relpath}` to
   get a presigned grant (`{"url", "headers"}`), then `PUT` the file's bytes to that URL with the
   returned headers.
4. `POST /api/v1/datasets/{id}/{version}/finalize` to commit the version. The staging directory is
   removed whether or not the publish succeeds.

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
manifest. The manifest always carries the derived schema, so search reads the schema straight from
the manifest instead of caching it separately.

---

See the [API reference for `timenet.registry`](api/registry.md) for the full symbol listing.
