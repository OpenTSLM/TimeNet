"""Dependency declarations and resolved locks for composed datasets."""

from dataclasses import dataclass

from timenet.errors import TimeNetInvalidManifestError
from timenet.manifest.counts import ManifestCounts
from timenet.types import Access, DatasetRef, License, ParentDataset


@dataclass(frozen=True)
class DirectDependency:
    """One card-declared parent with its local alias."""

    alias: str
    dataset: DatasetRef

    def __post_init__(self) -> None:
        """Validate the alias with the same rules as a dataset card."""
        ParentDataset(alias=self.alias, dataset=self.dataset)


@dataclass(frozen=True)
class LockedDependency:
    """One version in the flattened, content-verified dependency closure."""

    dataset: DatasetRef
    manifest_checksum: str
    size: int
    counts: ManifestCounts
    license: License
    access: Access

    def __post_init__(self) -> None:
        """Validate the checksum and byte count.

        Raises:
            TimeNetInvalidManifestError: If the checksum or size is invalid.
        """
        if isinstance(self.license, str):
            object.__setattr__(self, "license", License(self.license))
        if isinstance(self.access, str):
            object.__setattr__(self, "access", Access(self.access))
        checksum_length = len("sha256:") + 64
        if not self.manifest_checksum.startswith("sha256:") or len(self.manifest_checksum) != checksum_length:
            raise TimeNetInvalidManifestError("dependency manifest_checksum must be a canonical sha256 checksum")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise TimeNetInvalidManifestError("dependency size must be a non-negative integer")


@dataclass(frozen=True)
class ManifestDependencies:
    """Direct parent edges and the flattened dependency lock."""

    direct: tuple[DirectDependency, ...] = ()
    lock: tuple[LockedDependency, ...] = ()

    def __post_init__(self) -> None:
        """Require unambiguous aliases and one lock row per version.

        Raises:
            TimeNetInvalidManifestError: If an alias or locked version is duplicated, or a direct
                dependency is missing from the lock.
        """
        aliases = [dependency.alias for dependency in self.direct]
        datasets = [dependency.dataset for dependency in self.lock]
        if len(aliases) != len(set(aliases)):
            raise TimeNetInvalidManifestError("direct dependency aliases must be unique")
        if len(datasets) != len(set(datasets)):
            raise TimeNetInvalidManifestError("dependency lock contains a duplicate dataset version")
        locked = set(datasets)
        for dependency in self.direct:
            if dependency.dataset not in locked:
                raise TimeNetInvalidManifestError(
                    f"direct dependency {dependency.dataset} is absent from the dependency lock"
                )
