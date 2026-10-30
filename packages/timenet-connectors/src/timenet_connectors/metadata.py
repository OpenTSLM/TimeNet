"""A Record's metadata as a small shared core plus typed extensions, one per dataset or source.

This is a prototype of the shared metadata proposed on TimeNet #433, kept in the connectors package until
the core design is agreed. Nothing here is specific to one dataset, so it can move to ``timenet.types``
unchanged.

A Record's metadata is a ``RecordMetadata``: the fields every dataset shares, then any number of
extensions, each a ``MetadataExtension``. Each extension is stored under its own ``NAME``, so no extension can
overwrite a field or another extension, and ``from_json`` restores each one as its own type. An extension can
hold extensions of its own, stored the same way one level down::

    {"subject_id": "USW00093242", "trace": {"coordinates": ..., "sampling": ..., "windows": [...]}}

An extension that may change shape over time declares a ``version`` field as a ``Literal``, so a stored
version it does not read is rejected rather than misread.

Design choices:

- The stored JSON is plain keys and values, with no type tags or wrappers, so a reader without these classes,
  such as a consumer who has not installed the connector, can still read it.
- Values keep their types: each extension is a pydantic model with typed fields, not a bag of strings.
- Metadata is frozen, and ``extend`` returns a new copy, so metadata shared between objects never changes.
- ``from_json`` takes the extension types explicitly rather than looking them up in a global registry, so a
  name only has to be unique among the types passed, and reading does not depend on which modules are imported.

Metadata is provenance and never model input.
"""

from collections.abc import Iterable, Iterator
from typing import Annotated, Any, ClassVar, Self, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, StrictStr, StringConstraints

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping


_Extension = TypeVar("_Extension", bound="MetadataExtension")


class Metadata(BaseModel):
    """Typed fields, then extensions, each stored under its ``NAME``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    _extensions: dict[str, "MetadataExtension"] = PrivateAttr(default_factory=dict)
    """The extensions by name, in the order they were added. A private attribute rather than a field, so its
    own name never collides with an extension's."""

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:  # noqa: PLW3201 - pydantic's repr hook
        # pydantic leaves private attributes out of repr; show each extension under its name, as in the JSON.
        yield from super().__repr_args__()
        yield from self._extensions.items()

    def extend(self, *extensions: "MetadataExtension") -> Self:
        """Return this metadata with more extensions.

        Returns:
            New metadata; this one is unchanged.

        Raises:
            TimeFValidationError: If an extension's name is one of this metadata's fields, or another extension's.
        """
        extended = dict(self._extensions)
        for extension in extensions:
            if extension.NAME in type(self).model_fields:
                raise TimeFValidationError(f"extension name {extension.NAME!r} is a field of {type(self).__name__}")
            if extension.NAME in extended:
                raise TimeFValidationError(f"metadata already has a {extension.NAME!r} extension")
            extended[extension.NAME] = extension
        copy = self.model_copy()
        copy._extensions = extended
        return copy

    def extension(self, kind: type[_Extension]) -> _Extension:
        """Return the extension of one type.

        Returns:
            The extension.

        Raises:
            TimeFValidationError: If the metadata has no extension of that type.
        """
        extension = self._extensions.get(kind.NAME)
        if not isinstance(extension, kind):
            raise TimeFValidationError(f"metadata has no {kind.NAME!r} extension")
        return extension

    def to_json(self) -> JsonMapping:
        """Return the metadata as TimeNet stores it: the fields, then each extension under its name.

        Fields that are ``None`` are left out, and enums are written as their values.

        Returns:
            The JSON object.
        """
        stored: dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        for name, extension in self._extensions.items():
            stored[name] = extension.to_json()
        return stored

    @classmethod
    def from_json(cls, stored: JsonMapping, kinds: Iterable[type["MetadataExtension"]]) -> Self:
        """Read stored metadata back, restoring each extension, at any depth, as its own type.

        Args:
            stored: The JSON object ``to_json`` wrote.
            kinds: The extension types the metadata may hold, at any depth.

        Returns:
            The metadata, with its extensions in the order they were stored.

        Raises:
            TimeFValidationError: If two of ``kinds`` share a name, a stored key is neither a field nor one of
                ``kinds``, or an extension is not stored as a JSON object.
        """
        kinds = tuple(kinds)
        by_name = {kind.NAME: kind for kind in kinds}
        if len(by_name) != len(kinds):
            raise TimeFValidationError("two extension types share a name")
        fields = {key: value for key, value in stored.items() if key in cls.model_fields}
        unknown = sorted(key for key in stored if key not in fields and key not in by_name)
        if unknown:
            raise TimeFValidationError(f"metadata has keys that are neither fields nor known extensions: {unknown}")
        extensions: list[MetadataExtension] = []
        for key, value in stored.items():
            if key in fields:
                continue
            if not isinstance(value, dict):
                raise TimeFValidationError(f"metadata extension {key!r} is not a JSON object")
            extensions.append(by_name[key].from_json(value, kinds))
        return cls.model_validate(fields).extend(*extensions)


class MetadataExtension(Metadata):
    """A dataset's or source's typed addition to ``RecordMetadata``, stored under its ``NAME``."""

    NAME: ClassVar[str]
    """The key the extension is stored under. It must differ from its holder's fields and other extensions."""


class RecordMetadata(Metadata):
    """The metadata every Record shares, and the typed extensions its datasets add."""

    model_config = ConfigDict(title="Record metadata")

    subject_id: Annotated[StrictStr, StringConstraints(pattern=r"\S")] | None = None
    """What was measured: a person, a device, or a site such as a weather station."""
