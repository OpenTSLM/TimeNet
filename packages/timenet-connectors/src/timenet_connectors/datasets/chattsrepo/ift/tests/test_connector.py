"""Tests for the ChatTS instruction-following connector."""

from timenet_connectors.datasets.chattsrepo.ift import ChatTSIftConnector
from timenet_connectors.datasets.chattsrepo.tests.helpers import assert_configuration_conversion


def test_convert_preserves_ift_configuration(tmp_path):
    """Convert fixture rows with instruction-following provenance."""
    assert_configuration_conversion(ChatTSIftConnector(), "ift", tmp_path)
