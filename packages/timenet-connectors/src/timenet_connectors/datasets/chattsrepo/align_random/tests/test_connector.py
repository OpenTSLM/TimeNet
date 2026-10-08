"""Tests for the ChatTS variable-length alignment connector."""

from timenet_connectors.datasets.chattsrepo.align_random import ChatTSAlignRandomConnector
from timenet_connectors.datasets.chattsrepo.chatts_core.tests.helpers import assert_configuration_conversion


def test_convert_preserves_variable_length_configuration():
    """Convert fixture rows with variable-length alignment provenance."""
    assert_configuration_conversion(ChatTSAlignRandomConnector(), "align_random")
