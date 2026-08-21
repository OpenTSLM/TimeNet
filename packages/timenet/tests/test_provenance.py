import sys

from timenet.provenance import build_env


def test_build_env_records_the_running_python():
    assert build_env()["python"] == ".".join(str(part) for part in sys.version_info[:3])


def test_build_env_records_installed_packages():
    packages = build_env()["packages"]
    assert packages["timenet"]
    assert packages == dict(sorted(packages.items()))
