from pydantic import ValidationError
import pytest

from timenet_connectors.sources.readers import DataRow


class Row(DataRow):
    name: str
    count: int


def test_data_row_rejects_unexpected_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Row.model_validate({"name": "sensor", "count": 1, "extra": "value"})
