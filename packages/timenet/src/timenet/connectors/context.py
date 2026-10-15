"""Resolve exact build inputs before connector conversion starts."""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import os
from typing import Any

from timenet.builders import find_builder
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.reader import DatasetHandle, TimeFReader
from timenet.registry.base import BaseRegistry
from timenet.registry.local import LocalRegistry
from timenet.types import Access, DatasetMetadata
from timenet.types.references import DatasetRef, InputLock, manifest_digest


_CHAIN_ENV = "TIMENET_BUILD_INPUT_CHAIN"


@dataclass
class BuildContext:
    """Opened parent handles and immutable provenance for one connector build."""

    inputs: dict[str, DatasetHandle] = field(default_factory=dict)
    locks: dict[str, InputLock] = field(default_factory=dict)
    lineage: dict[str, InputLock] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        metadata: DatasetMetadata,
        registry: BaseRegistry,
        *,
        build_parents: bool = False,
        parent_builder: Callable[[DatasetRef], None] | None = None,
    ) -> "BuildContext":
        """Resolve inputs, optionally building missing exact releases.

        Returns:
            Opened handles and their pins.

        Raises:
            TimeFValidationError: If dependencies cycle, pins conflict, or a builder cannot satisfy a pin.
            TimeNetDatasetNotFoundError: If a parent is absent and parent building is disabled.
        """
        current = f"{metadata.dataset_id}@{metadata.dataset_version}"
        chain = json.loads(os.environ.get(_CHAIN_ENV, "[]"))
        if current in chain:
            raise TimeFValidationError(f"dataset input cycle: {' -> '.join([*chain, current])}")
        context = cls()
        try:
            for alias, ref in metadata.inputs.items():
                if str(ref) in {*chain, current}:
                    raise TimeFValidationError(f"dataset input cycle through {ref}")
                try:
                    version = registry.open_version(ref.dataset_id, str(ref.version))
                except TimeNetDatasetNotFoundError as exc:
                    if not build_parents:
                        raise TimeNetDatasetNotFoundError(
                            f"missing input {alias!r}: {ref}; build it first or pass --build-parents"
                        ) from exc
                    _build_input(ref, registry, parent_builder, [*chain, current])
                    version = registry.open_version(ref.dataset_id, str(ref.version))
                context.inputs[alias] = TimeFReader(version).get_handle()
                lock = InputLock(
                    source=ref,
                    manifest_checksum=manifest_digest(version.manifest.to_dict()),
                    metadata=version.manifest.metadata.model_dump(mode="json"),
                )
                context.locks[alias] = lock
                for ancestor in (*version.manifest.lineage, lock):
                    existing = context.lineage.get(str(ancestor.source))
                    if existing is not None and existing.manifest_checksum != ancestor.manifest_checksum:
                        raise TimeFValidationError(f"conflicting input snapshots for {ancestor.source}")
                    context.lineage[str(ancestor.source)] = ancestor
        except BaseException:
            context.close()
            raise
        return context

    def apply(self, dataset: Any) -> None:
        """Attach provenance and carry forward applicable access restrictions."""
        dataset.input_locks.update(self.locks)
        dataset.lineage.update(self.lineage)
        ranks = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}
        strictest = dataset.metadata
        for parent in self.inputs.values():
            if ranks[parent.metadata.access] > ranks[strictest.access]:
                strictest = parent.metadata
        if strictest.access != dataset.metadata.access:
            dataset._metadata = dataset.metadata.model_copy(
                update={"access": strictest.access, "access_url": strictest.access_url}
            )

    def close(self) -> None:
        """Release connections after the writer finishes consuming shared handles."""
        for parent in self.inputs.values():
            parent.close()

    def __enter__(self) -> "BuildContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _build_input(
    ref: DatasetRef, registry: BaseRegistry, parent_builder: Callable[[DatasetRef], None] | None, chain: list[str]
) -> None:
    """Build a missing parent while carrying the cycle guard across isolated processes.

    Raises:
        TimeFValidationError: If no matching builder or publishing callback is available.
    """
    previous = os.environ.get(_CHAIN_ENV)
    os.environ[_CHAIN_ENV] = json.dumps(chain)
    try:
        if parent_builder is not None:
            parent_builder(ref)
            return
        builder = find_builder(ref.dataset_id)
        if builder is None or builder.declared_version(ref.dataset_id) != str(ref.version):
            raise TimeFValidationError(f"no installed connector builds exact input {ref}")
        if not isinstance(registry, LocalRegistry):
            raise TimeFValidationError("remote parent builds require a publishing builder")
        builder.build(ref.dataset_id, registry.root, build_parents=True)
    finally:
        if previous is None:
            os.environ.pop(_CHAIN_ENV, None)
        else:
            os.environ[_CHAIN_ENV] = previous
