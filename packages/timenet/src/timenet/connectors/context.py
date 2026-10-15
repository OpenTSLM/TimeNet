"""Resolve exact build inputs before connector conversion starts."""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import os
from typing import Any

from timenet.builders import build_exact
from timenet.errors import TimeFValidationError
from timenet.reader import DatasetHandle, TimeFReader
from timenet.registry.base import BaseRegistry
from timenet.registry.factory import open_registry
from timenet.registry.local import LocalRegistry
from timenet.registry.version import prepare_dependency_graph
from timenet.types import Access, DatasetMetadata
from timenet.types.references import DatasetRef, InputLock


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
        fetch_parents: str | None = None,
        parent_builder: Callable[[DatasetRef], None] | None = None,
    ) -> "BuildContext":
        """Prepare complete parent releases before connector conversion.

        Args:
            metadata: The child dataset card with exact input references.
            registry: The output registry and local cache for prepared parents.
            build_parents: Build a missing exact release when fetching cannot find it.
            fetch_parents: Registry URI to fetch missing releases from.
            parent_builder: Optional callback that builds and publishes an exact parent.

        Returns:
            Opened handles and their pins.

        Raises:
            TimeFValidationError: If build inputs cycle or a builder cannot satisfy an exact version.
        """
        current = f"{metadata.dataset_id}@{metadata.dataset_version}"
        chain = json.loads(os.environ.get(_CHAIN_ENV, "[]"))
        if current in chain:
            raise TimeFValidationError(f"dataset input cycle: {' -> '.join([*chain, current])}")
        context = cls()
        parent_registry = open_registry(fetch_parents) if fetch_parents is not None else None

        def build_parent(ref: DatasetRef) -> None:
            _build_input(ref, registry, parent_builder, [*chain, current], fetch_parents)

        try:
            for alias, ref in metadata.inputs.items():
                if str(ref) in {*chain, current}:
                    raise TimeFValidationError(f"dataset input cycle through {ref}")
                fetch_from = parent_registry
                if fetch_from is None and not isinstance(registry, LocalRegistry):
                    fetch_from = registry
                version = prepare_dependency_graph(
                    registry.local_root,
                    ref,
                    fetch_from=fetch_from,
                    build=build_parent if build_parents else None,
                )
                context.inputs[alias] = TimeFReader(version).get_handle()
                lock = InputLock(
                    source=ref,
                    metadata=version.manifest.metadata.model_dump(mode="json"),
                )
                context.locks[alias] = lock
                for ancestor in (*version.manifest.lineage, lock):
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
    ref: DatasetRef,
    registry: BaseRegistry,
    parent_builder: Callable[[DatasetRef], None] | None,
    chain: list[str],
    fetch_parents: str | None,
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
            if not isinstance(registry, LocalRegistry):
                registry.download_version(
                    ref.dataset_id,
                    str(ref.version),
                    registry.local_root / ref.dataset_id / str(ref.version),
                )
            return
        if not isinstance(registry, LocalRegistry):
            raise TimeFValidationError("remote parent builds require a publishing builder")
        build_exact(ref, registry.root, fetch_parents=fetch_parents)
    finally:
        if previous is None:
            os.environ.pop(_CHAIN_ENV, None)
        else:
            os.environ[_CHAIN_ENV] = previous
