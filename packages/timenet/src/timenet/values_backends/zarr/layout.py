"""Paths of the arrays in a release's Zarr values store, without importing Zarr."""

VALIDITY_GROUP = "_validity"
"""Group that marks present timesteps in nullable series. It matches the writer's layout."""


def validity_path(values_rel_path: str) -> str:
    """Return the path of the validity array for a values array.

    The path preserves groups from the values locator. A legacy irregular path under
    ``_irregular/`` therefore remains under that group below ``_validity/``. Current layouts put
    every specification's values at the store root.

    Args:
        values_rel_path: The values array's path relative to the version directory.

    Returns:
        The parallel validity array's path.
    """
    parts = values_rel_path.split("/")
    return "/".join([parts[0], VALIDITY_GROUP, *parts[1:]])
