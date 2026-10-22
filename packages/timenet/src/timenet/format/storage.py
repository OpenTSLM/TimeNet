"""Write-time collection of direct physical owners for shared payloads."""

from typing import TYPE_CHECKING

from timenet.errors import TimeFValidationError
from timenet.types.reference_keys import OwnerKey
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


class StorageSources:
    """Collect the exact releases that store a new catalog's shared payloads."""

    def __init__(self, target: DatasetRef | None = None) -> None:
        self.target = target
        self.sources: dict[OwnerKey, DatasetRef] = {}

    def add(self, version: "DatasetVersion", paths: set[str]) -> OwnerKey:
        """Check the requested paths and record their physical dataset release.

        Returns:
            Its owner key in the child manifest: the exact release as ``org/name@version``.

        Raises:
            TimeFValidationError: If a requested payload is absent from the owner manifest.
        """
        manifest = version.manifest
        ref = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        if ref == self.target:
            raise TimeFValidationError(
                f"shared payload owner {ref} is also the output release; use a different dataset ID or version"
            )
        # The exact release itself names the owner, so catalog rows stay readable without the manifest.
        key = OwnerKey(str(ref))
        files = {
            part.path: part
            for part in manifest.files.all_files()
            if any(part.path == path or part.path.startswith(path + "/") for path in paths)
        }
        for path in paths:
            if not any(name == path or name.startswith(path + "/") for name in files):
                raise TimeFValidationError(f"payload {path!r} is absent from {ref}")
        self.sources[key] = ref
        return key
