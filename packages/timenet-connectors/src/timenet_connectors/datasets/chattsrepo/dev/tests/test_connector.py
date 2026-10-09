"""Tests for the ChatTS development connector."""

from timenet_connectors.datasets.chattsrepo.dev import ChatTSDevConnector
from timenet_connectors.datasets.chattsrepo.tests.helpers import assert_configuration_conversion


def test_convert_preserves_development_configuration():
    """Convert fixture rows with development-set provenance."""
    assert_configuration_conversion(ChatTSDevConnector(), "dev")
