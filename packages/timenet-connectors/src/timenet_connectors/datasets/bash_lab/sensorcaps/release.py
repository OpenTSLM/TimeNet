"""The pinned SensorCaps release on the Hugging Face Hub: its repository, commit, and files."""

from collections.abc import Mapping


HF_REPO = "BASH-Lab/SensorCaps"
SOURCE_REVISION = "4242e07feaacb0ee3aff86182a62914fa7abacef"
# Pinned inventory of the original captioning files.
SOURCE_ROWS: Mapping[str, int] = {
    "hhar_v2": 9166,
    "motion_v2": 4534,
    "pamap2_v2": 9672,
    "shoaib_v2": 10500,
    "uci_v2": 2088,
}


def source_file(partition: str) -> str:
    """Return a partition's repository-relative JSONL file.

    Returns:
        The file's path in the pinned repository.
    """
    return f"data/{partition}-00000-of-00001.jsonl"
