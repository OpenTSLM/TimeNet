"""Copy physical payload artifacts without decoding or changing their codecs."""

from hashlib import sha256
from pathlib import Path
from typing import Any

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import CHECKSUM_PREFIX
from timenet.manifest import FilePart
from timenet.types.payload_references import PayloadReference
from timenet.types.payloads import Payload


class PayloadCopier:
    """Own one copy of each required artifact in a standalone output."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.parts: list[FilePart] = []
        self._copied: set[str] = set()

    def reference(self, payload: Payload[Any]) -> PayloadReference:
        """Copy the referenced artifacts and return a local locator.

        Returns:
            The same codec and coordinates under the new owner's directory.

        Raises:
            TimeFValidationError: If the handle has no pinned storage binding.
            TimeFFormatError: If an artifact differs from its owner manifest.
        """
        reference = payload.reference
        if reference is None or reference.source is None or payload._storage is None:
            raise TimeFValidationError("copying a stored payload requires a pinned storage binding")
        version = payload._storage.resolve(reference)
        old_prefix = f"{reference.prefix}/" if reference.prefix else ""
        prefix = "payloads/owners/" + reference.source.manifest_checksum.removeprefix(CHECKSUM_PREFIX)
        needed = {old_prefix + chunk.path for chunk in reference.chunks}
        if reference.codec == "zarr":
            values = tuple(needed)
            for path in values:
                relative = path.removeprefix(old_prefix).split("/")
                needed.add(old_prefix + "/".join([relative[0], "_validity", *relative[1:]]))
        for part in version.manifest.files.all_files():
            required = (
                part.path in needed
                if reference.codec != "zarr"
                else any(
                    part.path.startswith(path + "/")
                    or (part.path.endswith("/zarr.json") and path.startswith(part.path.removesuffix("zarr.json")))
                    for path in needed
                )
            )
            if not required:
                continue
            target_path = f"{prefix}/{part.path}"
            if target_path in self._copied:
                continue
            target = self.root / target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = sha256(), 0
            try:
                with (
                    version.filesystem.open_input_file(version.path(part.path.removeprefix(old_prefix))) as source,
                    target.open("wb") as sink,
                ):
                    while block := source.read(2**20):
                        sink.write(block)
                        digest.update(block)
                        size += len(block)
            except OSError as exc:
                raise TimeFFormatError(f"cannot copy payload artifact {part.path!r}: {exc}") from exc
            if size != part.size or CHECKSUM_PREFIX + digest.hexdigest() != part.checksum:
                raise TimeFFormatError(f"copied payload artifact {part.path!r} differs from its owner manifest")
            self.parts.append(part.model_copy(update={"path": target_path}))
            self._copied.add(target_path)
        return reference.model_copy(
            update={"source": None, "prefix": prefix + (f"/{reference.prefix}" if reference.prefix else "")}
        )
