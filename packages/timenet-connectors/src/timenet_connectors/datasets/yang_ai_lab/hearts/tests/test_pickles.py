import os
import pickle
import re

import numpy as np
import pandas as pd
import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import (
    _ALLOWED_GLOBALS,
    dig,
    load_payload,
    require_pandas,
)


# The payloads here are hand-written. Their subject ids are not in the release's format, so no
# reader can mistake one for a released file.


@pytest.fixture(autouse=True)
def _empty_cache():
    load_payload.cache_clear()
    yield
    load_payload.cache_clear()


def _write(path, payload) -> str:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return str(path)


def test_require_pandas_gives_the_module_the_released_pickles_need():
    assert require_pandas() is pd


def test_a_payload_round_trips_through_the_restricted_unpickler(tmp_path):
    frame = pd.DataFrame({"Time (min)": [0.0, 1.0], "CGM (mg/dL)": [88.0, 91.5]})
    location = _write(tmp_path / "0.pkl", {"subject_id": "not-a-real-subject", "cgm_df": frame, "GT": 1.0})
    payload = load_payload(location)
    assert sorted(payload) == ["GT", "cgm_df", "subject_id"]
    assert np.array_equal(payload["cgm_df"]["CGM (mg/dL)"].to_numpy(), np.array([88.0, 91.5]))


def test_one_read_serves_every_loader_of_one_file(tmp_path):
    location = _write(tmp_path / "0.pkl", {"waveform": np.zeros(4, dtype=np.float32), "GT": 0})
    assert load_payload(location) is load_payload(location)


def test_a_file_holding_something_other_than_a_dict_fails_by_name(tmp_path):
    location = _write(tmp_path / "0.pkl", [1, 2, 3])
    with pytest.raises(TimeFFormatError, match="holds a list, not a dict"):
        load_payload(location)


class _RunsACommand:
    def __reduce__(self):
        return (os.system, ("true",))


def test_a_pickle_reaching_outside_the_allowlist_is_refused(tmp_path):
    # Rebuilding this payload would run a shell command. The restricted unpickler must refuse it
    # by name instead, before anything is imported.
    location = _write(tmp_path / "0.pkl", {"waveform": np.zeros(4, dtype=np.float32), "GT": _RunsACommand()})
    with pytest.raises(TimeFFormatError) as refusal:
        load_payload(location)
    # The allowlist is a record and not a proof, so the message names the file, the pair, that the
    # list is only a record of the pin, and where to add the pair.
    message = str(refusal.value)
    assert location in message
    assert "posix.system" in message or "nt.system" in message
    assert "a record of what the pinned revision asks for" in message
    assert "_ALLOWED_GLOBALS" in message


def test_the_allowlist_names_a_module_and_an_attribute_for_every_entry():
    assert _ALLOWED_GLOBALS
    assert all(isinstance(module, str) and isinstance(name, str) for module, name in _ALLOWED_GLOBALS)
    assert ("numpy", "ndarray") in _ALLOWED_GLOBALS
    assert ("os", "system") not in _ALLOWED_GLOBALS


def test_dig_follows_a_key_path():
    payload = {"segment_dfs": {"A": {"rsp": [1.0, 2.0]}}}
    assert dig(payload, ("segment_dfs", "A", "rsp")) == [1.0, 2.0]
    assert dig(payload, ()) is payload


def test_dig_names_the_missing_key_and_the_path_it_was_following():
    payload = {"segment_dfs": {"A": {}}}
    with pytest.raises(TimeFFormatError, match=re.escape("segment_dfs.A.rsp: 'rsp' is missing")):
        dig(payload, ("segment_dfs", "A", "rsp"))
