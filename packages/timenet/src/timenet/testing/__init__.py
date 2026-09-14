"""Deterministic builders shared across TimeNet tests.

This subpackage ships as real code under ``src`` (not test-directory helpers), so tests in both
workspace packages can import them under pytest's ``importlib`` mode.
"""

from timenet.testing.builders import make_dataset, make_metadata, make_record


__all__ = ["make_dataset", "make_metadata", "make_record"]
