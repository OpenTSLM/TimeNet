"""A Record's metadata as a small shared core plus typed extensions, one per dataset or source.

This is a prototype of the shared metadata proposed on TimeNet #433, kept in the connectors package until
the core design is agreed. Nothing here is specific to one dataset, so it can move to ``timenet.types``
unchanged.

A Record's metadata is a ``RecordMetadata``: the fields every dataset shares, then any number of
extensions, each a ``MetadataExtension``. Each extension is stored under its own ``NAME``, so no extension can
overwrite a core field or another extension, and ``from_json`` restores each one as its own type::

    {"subject_id": "USW00093242", "trace": {"coordinates": ..., "sampling": ..., "windows": [...]}}

Metadata is provenance and never model input.
"""

from collections.abc import Iterable
from typing import Annotated, Any, ClassVar, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping


class MetadataExtension(BaseModel):
    """A dataset's or source's typed addition to ``RecordMetadata``, stored under its ``NAME``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    NAME: ClassVar[str]
    """The key the extension is stored under. It must differ from every core field and other extension."""


_Extension = TypeVar("_Extension", bound=MetadataExtension)


class RecordMetadata(BaseModel):
    """The metadata every Record shares, and the typed extensions its datasets add."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="Record metadata")

    subject_id: Annotated[StrictStr, StringConstraints(pattern=r"\S")] | None = None
    """What was measured: a person, a device, or a site such as a weather station."""
    extensions: tuple[MetadataExtension, ...] = Field(default=(), exclude=True)
    """The extensions, in the order they were added. Each is serialized under its ``NAME``."""

    def extend(self, extension: MetadataExtension) -> Self:
        """Return this metadata with one more extension.

        Returns:
            New metadata; this one is unchanged.

        Raises:
            TimeFValidationError: If the extension's name is a core field's, or another extension's.
        """
        if extension.NAME in _CORE_FIELDS or extension.NAME == "extensions":
            raise TimeFValidationError(f"extension name {extension.NAME!r} is a core metadata field")
        if any(existing.NAME == extension.NAME for existing in self.extensions):
            raise TimeFValidationError(f"metadata already has a {extension.NAME!r} extension")
        return self.model_copy(update={"extensions": (*self.extensions, extension)})

    def extension(self, kind: type[_Extension]) -> _Extension:
        """Return the extension of one type.

        Returns:
            The extension.

        Raises:
            TimeFValidationError: If the metadata has no extension of that type.
        """
        for extension in self.extensions:
            if isinstance(extension, kind):
                return extension
        raise TimeFValidationError(f"metadata has no {kind.NAME!r} extension")

    def to_json(self) -> JsonMapping:
        """Return the metadata as TimeNet stores it: the core fields, then each extension under its name.

        Fields that are ``None`` are left out, and enums are written as their values.

        Returns:
            The JSON object.
        """
        stored: dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        for extension in self.extensions:
            stored[extension.NAME] = extension.model_dump(mode="json", exclude_none=True)
        return stored

    @classmethod
    def from_json(cls, stored: JsonMapping, kinds: Iterable[type[MetadataExtension]]) -> Self:
        """Read stored metadata back, restoring each extension as its own type.

        Args:
            stored: The JSON object ``to_json`` wrote.
            kinds: The extension types the metadata may hold.

        Returns:
            The metadata, with its extensions in the order they were stored.

        Raises:
            TimeFValidationError: If a stored key is neither a core field nor one of ``kinds``.
        """
        by_name = {kind.NAME: kind for kind in kinds}
        core = {key: value for key, value in stored.items() if key in _CORE_FIELDS}
        unknown = sorted(key for key in stored if key not in _CORE_FIELDS and key not in by_name)
        if unknown:
            raise TimeFValidationError(
                f"metadata has keys that are neither core fields nor known extensions: {unknown}"
            )
        extensions = tuple(by_name[key].model_validate(value) for key, value in stored.items() if key in by_name)
        metadata = cls.model_validate(core)
        for extension in extensions:
            metadata = metadata.extend(extension)
        return metadata


# The fields stored at the top level; the extensions are stored under their names instead.
_CORE_FIELDS = frozenset(name for name in RecordMetadata.model_fields if name != "extensions")
