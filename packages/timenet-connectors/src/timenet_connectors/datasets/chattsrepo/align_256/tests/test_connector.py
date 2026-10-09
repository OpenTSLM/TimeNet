"""Tests for the ChatTS fixed-length alignment connector."""

from timenet_connectors.datasets.chattsrepo.align_256 import ChatTSAlign256Connector
from timenet_connectors.datasets.chattsrepo.tests.helpers import assert_configuration_conversion


def test_convert_preserves_fixed_length_configuration():
    """Convert fixture rows with fixed-length alignment provenance."""
    assert_configuration_conversion(ChatTSAlign256Connector(), "align_256")
