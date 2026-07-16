import jsonschema
import yaml

from timenet.schemas import DATASET_CARD_SCHEMA
from timenet_connectors.discovery import available, resolve


def test_every_connector_ships_a_valid_dataset_card():
    ids = available()
    assert ids, "discovery found no connectors"
    for dataset_id in ids:
        connector = resolve(dataset_id)()
        card_path = connector._card_path()
        assert card_path.name == "dataset.yaml", f"{dataset_id} card is not dataset.yaml"
        data = yaml.safe_load(card_path.read_text())
        jsonschema.validate(data, DATASET_CARD_SCHEMA)
        assert data["dataset_id"] == dataset_id
