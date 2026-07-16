"""Deterministic builders and assertions shared across TimeNet tests.

Shipped as a real ``src`` subpackage (not test-directory helpers) so tests in both workspace packages can
import them under pytest's ``importlib`` mode.
"""

from timenet.testing.builders import (
    CountingLoader,
    assert_datasets_equal,
    make_dataset,
    sine_loader,
)


__all__ = ["CountingLoader", "assert_datasets_equal", "make_dataset", "sine_loader"]
