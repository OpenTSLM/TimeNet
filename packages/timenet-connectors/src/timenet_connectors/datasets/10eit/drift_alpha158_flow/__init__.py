"""Build the research-source DRIFT stock-flow replay."""

from importlib import import_module


# The organization ID begins with a digit, so static import resolution cannot represent its path.
CONNECTOR = import_module(".connector", __name__).CONNECTOR  # ruff: ignore[non-empty-init-module]
