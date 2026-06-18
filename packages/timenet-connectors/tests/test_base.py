from abc import ABC

import pytest

from timenet_connectors import BaseConnector


def test_base_connector_is_abstract():
    assert issubclass(BaseConnector, ABC)
    assert BaseConnector.__abstractmethods__ == frozenset({"download", "convert"})


def test_base_connector_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseConnector()
