from _origin_record import origin_file
import pytest

from timenet import _hints as hints_module
from timenet._hints import CliWording, Facts, RequestedRelease, SdkWording, hint
from timenet.errors import (
    FormatVersionSubject,
    TimeFFormatError,
    TimeFFormatVersionError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetInvalidCardError,
    TimeNetNoCatalogError,
    TimeNetOriginError,
    TimeNetRegistryError,
)
from timenet.registry import LocalRegistry
from timenet.testing import install_backends, make_dataset


def test_a_failed_hint_lookup_gives_no_hint_and_a_bug_in_it_surfaces(monkeypatch):
    error, facts = (
        TimeNetDatasetNotFoundError("x/y is in none of the sources."),
        Facts([], None, wanted=RequestedRelease("x/y")),
    )

    lookups = [
        TimeNetRegistryError("the registry is offline"),
        TimeFFormatError("the manifest is not valid"),
        TimeNetInvalidCardError("the card cannot be read"),
    ]
    for failure in lookups:
        monkeypatch.setattr(hints_module, "_remedy", lambda error, facts, wording, failure=failure: _raise(failure))
        assert hint(error, facts, SdkWording()) is None

    # Any other error of the lookup is a bug, also a TimeNet error.
    for bug in [AttributeError("a typo in the lookup"), TimeFValidationError("a manifest at an odd depth")]:
        monkeypatch.setattr(hints_module, "_remedy", lambda error, facts, wording, bug=bug: _raise(bug))
        with pytest.raises(type(bug), match=str(bug)):
            hint(error, facts, SdkWording())


def _raise(error):
    raise error


def test_a_subclass_of_an_error_without_its_own_remedy_gets_the_remedy_of_its_base_class(tmp_path):
    class _SubclassError(TimeNetDatasetNotFoundError):
        pass

    store = LocalRegistry(tmp_path / "store")
    facts = Facts([store], store, wanted=RequestedRelease("x/y"))

    assert hint(_SubclassError("x/y is in none of the sources."), facts, SdkWording()) == (
        "Add a registry that holds it to the sources, or install a connector package that builds it, such as "
        "timenet-connectors."
    )


def test_a_damaged_provenance_record_names_the_forced_download_and_build_of_the_sdk(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    release = store.put_dataset(make_dataset())
    origin_file(store, release).write_text("{")
    with pytest.raises(TimeNetOriginError) as caught:
        store.read_origin(release)

    wording = hint(caught.value, Facts([store], store), SdkWording())

    assert wording == (
        f"Get the release again with download('{release}', force=True) if it came from a registry, or with "
        f"timenet.build('{release}', local_registry={str(store.root)!r}, force=True) if it was built. Both replace "
        "the release and its record."
    )


def test_sources_without_a_catalog_name_the_read_of_an_exact_release_in_the_cli(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    wording = CliWording(remote_registry="s3://bucket/data")

    assert hint(TimeNetNoCatalogError("none of the sources has a catalog"), Facts([store], store), wording) == (
        "Read an exact release with `timenet info <id>@<version> --remote-registry s3://bucket/data`."
    )


class _Backend:
    """A builder backend that holds the card of the demo dataset, at version 1.0.0."""

    def dataset_ids(self):
        return [make_dataset().metadata.dataset_id]

    def card(self, dataset_id):
        return make_dataset().metadata

    def build(self, release, store, *, force, pinned_by, options):
        raise AssertionError(f"a hint never builds {release}")


def test_a_version_in_another_spelling_names_the_build_of_its_connector(tmp_path, monkeypatch):
    install_backends(monkeypatch, _Backend())
    store = LocalRegistry(tmp_path / "store")
    # 01.0.0 is the version 1.0.0 of the card, written in another way.
    facts = Facts([store], store, wanted=RequestedRelease("timenet/hello-world", "01.0.0"))
    error = TimeNetDatasetNotFoundError("timenet/hello-world@01.0.0 is in none of the sources.")
    assert hint(error, facts, SdkWording()) == (
        "Add a registry that holds it to the sources, or add InstalledConnectors() to build it."
    )


def test_a_format_version_error_names_an_upgrade_or_a_new_download_by_direction(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    path = tmp_path / "store/org/a/1.0.0/manifest.json"
    facts = Facts([store], store)

    def error(found, path=path):
        return TimeFFormatVersionError(FormatVersionSubject.TIMEF_FORMAT, str(path), found, 3, 3, path)

    # Deleting and downloading the release again gives the same manifest, so only an upgrade helps.
    assert hint(error(4), facts, SdkWording()) == "Upgrade the timenet package to read it."
    # An upgrade cannot read an older format, but a new download or build of the release can.
    assert hint(error(2), facts, SdkWording()) == (f"Delete {path.parent} and download or build org/a@1.0.0 again.")
    # A registry served the manifest, so no local delete replaces it.
    assert hint(error(2, path=None), facts, SdkWording()) is None
