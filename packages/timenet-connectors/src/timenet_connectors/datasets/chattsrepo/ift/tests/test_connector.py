"""Tests for the ChatTS instruction-following connector."""

from timenet_connectors.datasets.chattsrepo.chatts_core.tests.helpers import assert_configuration_conversion
from timenet_connectors.datasets.chattsrepo.ift import ChatTSIftConnector


def test_convert_preserves_ift_configuration():
    """Convert fixture rows with instruction-following provenance."""
    assert_configuration_conversion(ChatTSIftConnector(), "ift")
