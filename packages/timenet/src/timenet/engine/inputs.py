"""Resolve exact stored inputs before importing or running a missing connector."""

from contextlib import ExitStack
from pathlib import Path

from timenet.builders import find_builder
from timenet.config import settings
from timenet.connectors.context import BuildContext, active_build_path
from timenet.errors import TimeNetBuildError, TimeNetDatasetNotFoundError
from timenet.reader import TimeFReader
from timenet.registry import open_registry
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetMetadata, DatasetRef


def _check_identity(version: DatasetVersion, reference: DatasetRef) -> None:
    actual = DatasetRef(dataset_id=version.manifest.dataset_id, version=version.manifest.metadata.dataset_version)
    if actual != reference:
        raise TimeNetBuildError(f"requested input {reference}, but the stored artifact identifies {actual}")


def _input(reference: DatasetRef, root: Path) -> DatasetVersion:
    if reference in active_build_path():
        path = " -> ".join(str(item) for item in (*active_build_path(), reference))
        raise TimeNetBuildError(f"connector inputs form a cycle: {path}")
    target = root / reference.dataset_id / str(reference.version)
    if (target / "manifest.json").exists():
        version = DatasetVersion.open_local(target)
        _check_identity(version, reference)
        return version
    cfg = settings()
    for location in (cfg.storage_dir, cfg.registry or cfg.registry_path):
        registry = open_registry(location)
        try:
            manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
        except TimeNetDatasetNotFoundError:
            continue
        actual = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        if actual != reference:
            raise TimeNetBuildError(f"requested input {reference}, but the registry returned {actual}")
        registry.download_version(reference.dataset_id, str(reference.version), target, manifest=manifest)
        # Physical sources are flat: provenance inputs do not cause downloads.
        for source in manifest.payload_sources:
            owner = source.dataset
            registry.download_version(
                owner.dataset_id, str(owner.version), root / owner.dataset_id / str(owner.version)
            )
        return DatasetVersion.open_local(target)
    builder = find_builder(reference.dataset_id)
    if builder is None:
        raise TimeNetBuildError(f"no stored input or installed connector can provide {reference}")
    declared = builder.declared_version(reference.dataset_id)
    if declared != str(reference.version):
        raise TimeNetBuildError(f"input {reference} is unavailable; its connector declares release {declared!r}")
    built = builder.build(reference.dataset_id, root, force=False)
    version = DatasetVersion.open_local(built)
    _check_identity(version, reference)
    if built.resolve() != target.resolve():
        raise TimeNetBuildError(f"input builder wrote {reference} outside its requested directory {target}")
    return version


def open_inputs(metadata: DatasetMetadata, root: Path, stack: ExitStack) -> BuildContext:
    """Open stored inputs and retain their readers until the output is committed.

    Returns:
        The aliased exact input readers.
    """
    return BuildContext(
        {
            alias: stack.enter_context(TimeFReader(_input(reference, root)))
            for alias, reference in metadata.inputs.items()
        }
    )
