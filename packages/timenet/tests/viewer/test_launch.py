"""Viewer selection through the CLI and local dataset paths."""

import pytest
from typer.testing import CliRunner

from timenet.cli.app import app
from timenet.viewer.launch import open_reader


@pytest.mark.parametrize("selection", ["registry", "path"])
def test_local_selection_opens_the_stored_dataset(stored, selection):
    directory, dataset_id = stored
    args = (
        (dataset_id, None, str(directory), None)
        if selection == "registry"
        else (None, None, None, directory / dataset_id / "1.0.0")
    )
    with open_reader(*args) as reader:
        assert reader.record_ids() == ("record-0", "record-1", "record-2")


def test_cli_explains_how_to_select_a_registry_path():
    result = CliRunner().invoke(app, ["view", "registry/chengsenwang/tsqa", "--no-browser"])
    assert result.exit_code == 2
    assert "org/name" in result.output
    assert "--registry" in result.output
    assert "--path" in result.output
