---
icon: lucide/database
description: "Registry backends that serve compiled TimeF versions to the SDK."
tags:
  - guide
  - registry
---

# Registry

A registry serves compiled TimeF versions to the SDK. A compiled TimeF version has a manifest,
an immutable DuckDB control database, and a values plane in Parquet or Zarr format. A registry never
runs connector code. The registry code lives in `timenet.registry`.

You can have several registries: a public registry, private internal registries, or a local
directory. The output of [build](build.md) is itself a valid local registry.

## Choosing a registry

To read a dataset, use [`TimeNet.open()`](client.md#open-load-or-download) with its ID, or
[`TimeFReader.open()`](timef-reader.md) with a version directory.

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

`open_registry()` returns a `WritableRegistry`, so you can also publish to the registry that it
opens (see [Writing to a registry](#writing-to-a-registry)). A remote registry reads only its own
files. To read a local copy first, put its `LocalRegistry` before the remote registry in the
sources.

!!! note "Catalog support"
    Every backend serves reads and takes new releases. Remote publishing needs a writer token, and
    the S3 backend reads AWS credentials from the environment. The S3 backend has no catalog, so
    `list` and `search` do not work there.

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
uv add 'timenet[s3]'
```

The layout under the prefix is `datasets/org/name/version/…`, with `manifest.json` beside the data
files. `put_dataset` uploads to a temporary `version.tmp-<uuid>` prefix and moves each object into
place (`manifest.json` last), so that failed publishing does not leak partial versions.
`TimeNet.open()` reads lazily and `load()` reads every value. Both read remote payload ranges
through a pyarrow `S3FileSystem` (see [Remote reads](#remote-reads)). The backend has no catalog, so
`list` and `search` are unsupported: publish and fetch by an explicit `org/name@version`.

## `BaseRegistry`

Every backend implements this contract.

| Method | Description |
| --- | --- |
| `list_datasets()` | Latest-version `DatasetMetadata` for every dataset, sorted by id. |
| `get_manifest(dataset_id, version=None)` | A dataset's [manifest](manifest.md) (latest if `version` is `None`). Raises `TimeNetDatasetNotFoundError` for an unknown id/version. Raises `TimeNetManifestError` if the stored manifest is not valid, `TimeFFormatVersionError` if it has a newer TimeF format, or `TimeFFormatError` if its own id disagrees with the directory it was loaded from. |
| `open_file(dataset_id, version, relpath)` | A file of a dataset version, opened for seekable binary reading (an object store must return a range-capable handle, not a forward-only stream). |
| `location` | Where the registry is: the base URL of a remote registry, the `s3://bucket/prefix` root of an S3 registry, or the resolved root directory of a local registry. Each backend must define it. |
| `requires_open_access` | Whether the registry holds and serves only releases with open access. `False` only for `LocalRegistry`. |
| `find_manifest(release)` | The manifest of an exact release, or `None` if the registry does not hold it. Raises `TimeFFormatError` if the stored manifest names another release. |
| `download_release(reference, directory, *, force=False, manifest=None, progress_cb=None)` | Download the files of one release, without its payload sources, into a local directory. A failed download leaves no partial copy, and it keeps the copy that was there before. |
| `open_release(reference, *, manifest=None)` | A `DatasetVersion` of one exact release without its payload sources: the parsed manifest and a handle to the files. Pass `manifest` if you already read it from this registry. A local registry raises `TimeNetFileError` for a missing or resized file. |
| `resolve(dataset_id, version)` | Whether the registry serves a release, as a [source](#sources) answer: `Found`, `NotFound` or `NotServed`. |
| `search(filter)` | Find the datasets that match a `DatasetFilter`. |

To copy a release with its payload sources, use
[`TimeNet.download()`](client.md#open-load-or-download) for the local registry where downloads and
builds land, or [`copy_release()`](#copying-a-release) for another registry. To read a release
with its payload sources, use [`TimeNet.open()`](client.md#open-load-or-download).

`LocalRegistry` serves a `<root>/<dataset_id>/<version>/` tree. `RemoteRegistry` serves the same layout
over the versioned REST contract (`GET /api/v1/datasets`, `/api/v1/datasets/{id}/{version}/manifest`,
`.../download/{relpath}`); `S3Registry` serves the same layout under an `s3://bucket/prefix` root. The
S3 backend has no catalog: its `has_catalog` is `False`, and `list_datasets` and `search` raise
`NotImplementedError`. `TimeNet.list()` and `TimeNet.search()` leave a registry without a catalog
out, with a `CatalogWarning` that names it. If no source of the list has a catalog, they raise
`TimeNetNoCatalogError`.

A dataset id is an `org/name` pair, for example `chengsenwang/tsqa`. The id nests one level deep on
disk, at `<root>/chengsenwang/tsqa/<version>/`. `list_datasets` finds these ids at any depth. Use
lowercase ids to avoid casing clashes on case-insensitive filesystems.

### Payload sources

A version can reuse payload files that other versions store: its payload sources. Its manifest
names each one and pins the files it reads from it (see
[Shared payloads](timef-format.md#shared-payloads)). The version needs its payload sources:

- A read through `TimeNet.open()` resolves each payload source through the list of sources, and
  checks it against the pins. It reads only manifests and, for local files, file sizes. See
  [Opening a version](timef-reader.md#opening-a-version) for the checks and their errors.
- `TimeNet.download()` and `copy_release()` copy complete copies of the version and of each payload
  source, and check each source against the pins. If a downloaded file has another size or checksum
  than its manifest records, the copy raises `TimeNetFileError` and keeps no partial copy.
- `put_dataset` refuses a version if the registry does not hold one of its payload sources.

### Remote reads

HTTP reads use the range reader and the read-ahead cache of fsspec. The object server must support
byte ranges. The client requests a new signed URL after one expires. The registry token goes only
with API requests.

A read of byte ranges cannot make sure that a whole file matches its checksum. `TimeNet.download()`
copies complete files, and makes sure that each file has its recorded size and checksum.
`TimeFReader.verify()` reads and hashes every file that a release reads.

## Sources

A source is something that can provide a release. `timenet.sources` defines the type
`ReleaseSource`. Each source has the method `resolve(dataset_id, version)`. `version` is a `Version`
object for an exact version, or `None` for the newest version of that source. Two kinds of source
exist:

- A registry. Every `BaseRegistry` is a source.
- `InstalledConnectors()`: every installed package that registers a builder backend under the
  `timenet.builders` entry point group.

A source gives one of these answers:

| Answer | Meaning |
| --- | --- |
| `FoundInRegistry(version, registry)` | The registry serves the release. |
| `FoundInConnector(builder)` | An installed connector builds the release. `version` is the version that its card declares. |
| `NotFound()` | The source does not know the release. |
| `NotServed(version, access, requirements)` | The source lists the release, but does not serve its files. `access` is the effective access of the release, and `requirements` names each card that is not open, with its access page. `access_error(dataset_id)` returns the `TimeNetAccessError` that names each page. |

The type `Found` covers `FoundInRegistry` and `FoundInConnector`. `FoundInRegistry` and
`NotServed` also carry the `manifest` that the source read, or `None`, so that you do not need to
fetch it again.

A `LocalRegistry` serves every release that it holds, so it never answers `NotServed`. A registry
that is not local serves the files of a release only if its effective access is `open`, and
answers `NotServed` otherwise.

`InstalledConnectors()` answers `Found` only for the version that the card of a connector declares,
because a connector builds only that version. For any other request, it answers `NotFound`.

`resolve_release(sources, dataset_id, version)` asks an ordered list of sources and follows four
rules:

1. Without a version, the first source that serves the dataset gives its newest version. A source
   that lists the dataset without serving it falls through to the next source.
2. An exact version comes from the first source that provides it.
3. `"latest"` is the highest version across all sources, the connectors included. The order then
   decides only which source provides that version.
4. Only `NotFound` and `NotServed` fall through to the next source. Any other error of a source
   stops the resolution.

So the order of the list is the read order: a release that two sources hold comes from the first
one.

```python
from timenet.registry import LocalRegistry, open_registry
from timenet.sources import InstalledConnectors, resolve_release

sources = [
    LocalRegistry("~/.timenet/registry"),
    open_registry("timenet://"),
    InstalledConnectors(),
]
answer = resolve_release(sources, "timenet/hello-world", None)
```

A build uses such a list to find the parent releases that its local registry lacks. Each missing
parent comes from the first source that provides its exact version.

`TimeNet` takes such a list as `sources`. The first `LocalRegistry` of the list that is not
`read_only=True` gets the downloads and builds, and a list with `InstalledConnectors()` needs one.
See [Choose the sources](client.md#choose-the-sources).

## Writing to a registry

A `WritableRegistry` adds two write methods to the read contract. Every backend implements them,
so you can publish to a local directory, a remote registry, or an S3 bucket:

| Method | Description |
| --- | --- |
| `put_dataset(dataset, *, force=False, values_backend="parquet", progress_cb=None)` | Compile a dataset and publish it. Returns the stored release as a `DatasetRef`. Derives the schema first if absent, skips an already-committed version unless `force`, and checks every payload source in this registry. `values_backend` picks the values plane format: `"parquet"` or `"zarr"`. On a remote or S3 registry, `force=True` raises `TimeFValidationError` and uploads nothing, because a published version never changes. |
| `put_release(from_directory)` | Publish a complete local release directory as it is; returns the published release as a `DatasetRef`. Only reads `from_directory` and never changes it. Makes sure that the directory holds a readable `manifest.json` and every file that the manifest declares with its recorded size, and raises `TimeFFormatError` or `TimeNetFileError` otherwise. Publishes only those files and `manifest.json`. |
| `exists(dataset_id, version)` | Whether a committed version already exists (shared implementation). |
| `requires_clean_code_provenance` | Whether the registry refuses releases built from locally modified TimeNet code. `False` only for `LocalRegistry`. Otherwise `put_dataset` and `put_release` raise `TimeNetBuildError` for a release whose manifest records local changes. |

`LocalRegistry`, `RemoteRegistry` and `S3Registry` take `read_only=True`. A read-only registry
refuses each write with `TimeNetRegistryError`.

`put_dataset` writes the dataset with a [`TimeFWriter`](timef-writer.md) and checks each payload
source of the new release in the registry that receives it. In a local registry, `force=True`
replaces a committed version only once the new one is complete, so a failed write keeps the old
one. A replace can break readers that already opened the old version, and versions that reuse its
payloads then fail to open.
`RemoteRegistry` publishes through the [publish flow](#remote-publishing) below.

`put_release` publishes a release directory that already exists, without a new compile. The
directory must hold `manifest.json` and every file that it lists, with the recorded size. Only those
files reach the registry.

`open_registry(uri)` returns a `WritableRegistry`, so you can publish to every registry that it
opens:

```python
from timenet.registry import open_registry

registry = open_registry("./local_registry")
# schema derived if needed, atomic commit
release = registry.put_dataset(dataset)
```

Every backend is a `WritableRegistry`. As a result, this type alone does not show if a backend is a
directory the engine can write to or a remote stub. `local_registry_path(uri)` gives this
information. It returns the directory that a `file://` URI or a plain path names. It raises
`TimeNetRegistryError` for a remote scheme or an empty string.

### Remote publishing

`RemoteRegistry.put_dataset` compiles the dataset locally and then sends the files to the service.
The data does not pass through the API. The service gives a presigned URL for each file, and the SDK
uploads the file to the object store directly. You need a writer token in `$TIMENET_TOKEN`. The
service refuses an anonymous or read-only caller. `RemoteRegistry.put_release` runs steps 2 to 4
for a release directory that already exists.

1. Compile into a temporary staging directory with a [`TimeFWriter`](timef-writer.md), deriving the
   schema first if the dataset has none. An already-committed version returns early. The writer
   checks every payload source against the service, so a version whose sources the service does not
   hold is refused before anything is published.
2. `POST /datasets/{id}/{version}/publish` with the compiled `manifest.json` as the body. The service
   registers the pending version and returns the list of files it expects.
3. For each file, `POST /datasets/{id}/{version}/publish/upload-url` with `{"path": relpath}` to get a
   presigned grant (`{"url", "headers"}`), then `PUT` the file's bytes to that URL with the returned
   headers.
4. `POST /datasets/{id}/{version}/finalize` to commit the version. The staging directory is removed
   whether or not the publish succeeds.

```python
from timenet.registry import open_registry

registry = open_registry("timenet://")   # needs $TIMENET_TOKEN
release = registry.put_dataset(dataset)  # compile, upload, finalize
```

### Copying a release

`timenet.registry.copy_release(reference, source, dest, *, with_payload_sources=True, force=False,
progress=None)` copies one release from `source` to the writable registry `dest`.

A dataset can depend on payload files stored as part of another dataset (parent/child relationship).
`copy_release` copies each payload source that the manifest of the release lists too, so that
`dest` can open the copied release. With `with_payload_sources=False`, it copies the release alone.
It copies only the releases that `dest` does not hold yet.

`copy_release` checks every release before it copies anything, so a refused copy leaves `dest`
unchanged. A `dest` with `requires_clean_code_provenance` refuses a release whose manifest records TimeNet
code with local changes, as
[Clean code provenance for remote registries](build.md#clean-code-provenance-for-remote-registries) describes.

## `search`

```python
from timenet.types import DatasetFilter, Domain

registry.search(DatasetFilter.of(domain=Domain.CARDIOLOGY, tag="ecg", limit=10))
```

A `DatasetFilter` from `timenet.types` holds the filters and the limit. Each filter is a tuple, and
an empty tuple sets no restriction. `DatasetFilter.of()` takes one value, a list, or `None` for
each filter. The registry combines the filters with AND logic. `limit` caps the number of
results, with a default of 100. `limit=0` returns no results, and a negative limit raises
`TimeFValidationError`.

`DatasetFilter.select()` does the filtering. It takes `ListedRelease` rows, each a card and the
registry that lists it, and keeps their order. It checks the card first. A `task` or
`time_series_spec` filter then reads the manifest of a matching card from its registry. A row
without a registry never matches these two filters. `select()` stops at `limit` matches, so it
reads no more manifests than it needs.

A registry sorts its catalog by dataset ID and gives it to `select()`, so `search()` returns the
first matches by dataset ID. `TimeNet.search()` and the consumer CLI, `timenet search`, take the
same filters as keywords. They give `select()` the merged rows of all sources (see
[Browse datasets](client.md#browse-datasets)). So they check only the highest version of each
dataset, and they do not find an older version that matches.

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

See the [API reference for `timenet.registry`](api/timenet.registry.md) for the full symbol listing.
