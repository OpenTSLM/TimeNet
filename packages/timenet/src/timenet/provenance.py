"""Describe the environment a build ran in, for the manifest's provenance block."""

from importlib.metadata import distributions
import platform
from typing import Any


def build_env() -> dict[str, Any]:
    """Describe this process's Python version and installed packages.

    Under an isolated build the recorded set is exactly the base, the connector's requirements, and
    their transitives. Under ``--no-isolation`` it is the whole ambient environment, which is noisier
    but still the truth about what produced the dataset.

    Returns:
        A dict with ``python`` (the interpreter version) and ``packages`` (name -> version, sorted).
    """
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version
    return {"python": platform.python_version(), "packages": dict(sorted(packages.items()))}
