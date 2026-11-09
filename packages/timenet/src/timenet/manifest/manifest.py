"""The :class:`Manifest`: the compiled ``manifest.json`` file and its JSON codec."""

from importlib.metadata import distributions
from pathlib import Path
import platform
from typing import Annotated, Any, Self

from pydantic import Field, StrictBool, StrictStr, StringConstraints, ValidationError, model_validator

from timenet.errors import ReleaseDifference, TimeFFormatError, TimeNetManifestError
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.manifest.payload_source import PayloadSource
from timenet.types import DatasetMetadata, DatasetSchema
from timenet.types._model import TimeFModel, _first_problem
from timenet.types.metadata_fields import DatasetId
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKeyField
from timenet.types.references import DatasetRef
from timenet.types.wire import ValueEncoding
from timenet.values_backends import ValuesBackend


GitSHA = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
"""A full SHA-1 git commit id: 40 lowercase hex digits."""


class GitSource(TimeFModel):
    """The git commit that one TimeNet distribution was built from."""

    sha: GitSHA
    """The commit of the checkout, or of the checkout that built the wheel."""
    contains_local_changes: StrictBool
    """Whether the package directory had uncommitted changes, untracked files included."""

    @classmethod
    def load_from_build_info(cls, path: Path) -> Self:
        """Read the record that ``scripts/build_dists.py`` wrote into a built package."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


DistributionName = StrictStr
"""The name of an installed distribution, for example ``timenet-connectors``."""

GitSourceMap = dict[DistributionName, GitSource]
"""The commit of each installed TimeNet distribution, keyed by distribution name."""


class BuildEnvironment(TimeFModel):
    """The Python version, the installed packages, and the TimeNet commits that produced a version."""

    python: StrictStr
    """The interpreter version, for example ``3.13.1``."""
    packages: dict[DistributionName, StrictStr]
    """Each installed distribution and its version, sorted by name."""
    git: GitSourceMap
    """The commit of each installed TimeNet distribution. A distribution that is not installed is absent."""

    @classmethod
    def capture(cls, git: GitSourceMap) -> Self:
        """Describe this interpreter, with ``git`` from ``TimeNetSnapshot.to_source_map()``."""
        # Under an isolated build the packages are exactly the base, the connector's requirements,
        # and their transitives. Under --no-isolation they are the whole ambient environment, which
        # is noisier but still the truth about what produced the dataset.
        packages = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
        return cls(python=platform.python_version(), packages=dict(sorted(packages.items())), git=git)


class Manifest(TimeFModel):
    """The single source of truth a consumer reads to interpret a dataset version."""

    dataset_id: DatasetId
    """A denormalized copy of ``metadata.dataset_id``."""
    metadata: DatasetMetadata
    """Descriptive identity of the dataset."""
    files: ManifestFiles
    """Descriptor for every data artifact, grouped by kind."""
    dataset_schema: Annotated[DatasetSchema, Field(alias="schema")] = Field(default_factory=DatasetSchema)
    """Structural schema: time-series specs, annotations, and tasks."""
    counts: ManifestCounts = Field(default_factory=ManifestCounts)
    """Row and entity counts recorded for quick inspection."""
    values_backend: ValuesBackend = ValuesBackend.PARQUET
    """Storage backend for the time-series values plane."""
    value_encoding: dict[StrictStr, ValueEncoding] = Field(default_factory=dict)
    """``spec_type`` to the values-column encoding used by its shards."""
    parent_cards: dict[ParentAlias, ParentDatasetCard] = Field(default_factory=dict)
    """Copies of the cards of the declared parents, indexed by parent alias."""
    lineage: tuple[ParentDatasetCard, ...] = ()
    """Flattened provenance snapshots; never traversed to read data."""
    payload_sources: dict[SourceKeyField, PayloadSource] = Field(default_factory=dict)
    """Every release that stores a payload file this release reads, by source key, pinned to those
    files. The writer flattens them, so a source's own sources are not listed unless this release
    reads their files too."""
    build_env: BuildEnvironment
    """The Python version, the package set, and the TimeNet commits that produced this version."""
    timef_format_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    """The TimeF manifest format version."""

    @model_validator(mode="after")
    def _matching_dataset_id(self) -> Self:
        if self.dataset_id != self.metadata.dataset_id:
            raise ValueError(
                f"manifest dataset_id {self.dataset_id!r} does not match "
                f"metadata.dataset_id {self.metadata.dataset_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _source_keys_name_their_release(self) -> Self:
        for key, source in self.payload_sources.items():
            if key != str(source.release):
                raise ValueError(f"payload source key {key!r} does not name its release {source.release}")
        return self

    @model_validator(mode="after")
    def _one_parent_card_per_release(self) -> Self:
        aliases: dict[str, list[str]] = {}
        for alias, card in self.parent_cards.items():
            aliases.setdefault(str(card.source), []).append(alias)
        for release, names in aliases.items():
            if len(names) > 1:
                raise ValueError(
                    f"parent_cards hold the parent {release} under the parent aliases {sorted(names)}; "
                    "a manifest holds each parent release under one alias only"
                )
        return self

    def required_releases(self, *, with_parents: bool) -> list[DatasetRef]:
        """List the releases that this release needs: its payload sources, and its parents if asked.

        Args:
            with_parents: Also list the declared parents.

        Returns:
            The releases of the payload sources, then the parents.
        """
        releases = [source.release for source in self.payload_sources.values()]
        if with_parents:
            releases.extend(self.metadata.parents.values())
        return releases

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a validated JSON-compatible dict with all keys present.

        Returns:
            The canonical dict form.
        """
        # Frozen models still contain mutable dictionaries; validate before publishing them.
        validated = type(self).model_validate(self)
        return validated.model_dump(mode="json", by_alias=True)

    def to_json(self) -> str:
        """Serialize the manifest to a pretty JSON string.

        Returns:
            The JSON text.
        """
        # Apply the same validation as to_dict() to catch changes in nested dictionaries.
        validated = type(self).model_validate(self)
        return validated.model_dump_json(indent=2, by_alias=True)

    def difference_from(self, other: "Manifest") -> ReleaseDifference | None:
        """Compare this manifest with another one of the same release, ignoring their build environments.

        Args:
            other: The other manifest.

        Returns:
            ``FILES`` if the file records differ, else ``CARD`` if the rest differs, or ``None`` for the same
            release.
        """
        if _file_records(self) != _file_records(other):
            return ReleaseDifference.FILES
        if self.model_copy(update={"build_env": {}}) != other.model_copy(update={"build_env": {}}):
            return ReleaseDifference.CARD
        return None


def _file_records(manifest: Manifest) -> frozenset[tuple[str, int, str]]:
    """Return the path, size, and checksum of each file of a manifest, in no order."""
    return frozenset((part.path, part.size, part.checksum) for part in manifest.files.all_files())


def parse_manifest(text: str | bytes, where: str, *, path: Path | None = None) -> Manifest:
    """Parse the JSON text of a stored manifest.

    Args:
        text: The JSON text.
        where: The source location to include in errors.
        path: The file of the manifest, or ``None`` for a manifest that a registry served.

    Returns:
        The manifest.

    Raises:
        TimeNetManifestError: If the text is not a valid manifest.
    """
    try:
        return Manifest.model_validate_json(text)
    except ValidationError as error:
        raise TimeNetManifestError(where, _first_problem(error), path) from error


def read_manifest(path: Path) -> Manifest:
    """Read a ``manifest.json`` file.

    Args:
        path: The file.

    Returns:
        The manifest.

    Raises:
        TimeFFormatError: If the file cannot be read.
        TimeNetManifestError: If the file is not a valid manifest.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by parse_manifest
    try:
        text = path.read_bytes()
    except OSError as error:
        raise TimeFFormatError(f"cannot read the manifest {path}: {error.strerror or error}") from error
    return parse_manifest(text, str(path), path=path)
