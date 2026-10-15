"""Write-time collection of direct physical owners for shared payloads."""

import hashlib
from typing import TYPE_CHECKING

from timenet.errors import TimeFValidationError
from timenet.manifest.sources import PayloadSource
from timenet.types.references import DatasetRef, manifest_digest


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


class StorageSources:
    """Collect only the physical files referenced by a new catalog."""

    def __init__(self) -> None:
        self.sources: dict[str, PayloadSource] = {}

    def add(self, version: "DatasetVersion", paths: set[str]) -> str:
        """Pin one physical owner and merge the requested artifact descriptors.

        Returns:
            Its compact owner key in the child manifest.

        Raises:
            TimeFValidationError: If a requested payload is absent from the owner manifest.
        """
        manifest = version.manifest
        ref = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        key = hashlib.sha256(str(ref).encode()).hexdigest()[:24]
        files = {
            part.path: part
            for part in manifest.files.all_files()
            if any(part.path == path or part.path.startswith(path + "/") for path in paths)
        }
        for path in paths:
            if not any(name == path or name.startswith(path + "/") for name in files):
                raise TimeFValidationError(f"payload {path!r} is absent from {ref}")
        digest = manifest_digest(manifest.to_dict())
        previous = self.sources.get(key)
        if previous is not None:
            if previous.source != ref or previous.manifest_checksum != digest:
                raise TimeFValidationError(f"conflicting manifests for physical owner {ref}")
            files.update((part.path, part) for part in previous.files)
        self.sources[key] = PayloadSource(
            source=ref,
            manifest_checksum=digest,
            values_backend=manifest.values_backend,
            files=tuple(files[name] for name in sorted(files)),
        )
        return key
