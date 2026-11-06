"""Deterministic builders and assertions shared across TimeNet tests.

This subpackage ships as real code under ``src`` (not test-directory helpers), so tests in both
workspace packages can import them under pytest's ``importlib`` mode.
"""

from timenet.testing.builders import (
    CountingLoader,
    assert_datasets_equal,
    make_dataset,
    sine_loader,
)
from timenet.testing.round_trip import assert_round_trips


__all__ = [
    "CountingLoader",
    "assert_datasets_equal",
    "assert_round_trips",
    "make_dataset",
    "sine_loader",
]
