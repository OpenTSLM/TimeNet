from pydantic import ValidationError
import pytest

from timenet.config import settings


def test_token_defaults_to_none(monkeypatch):
    monkeypatch.delenv("TIMENET_TOKEN", raising=False)
    assert settings().token is None


def test_token_read_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_TOKEN", "tok_abc")
    assert settings().token == "tok_abc"


def test_download_mode_defaults_to_on_demand(monkeypatch):
    monkeypatch.delenv("TIMENET_DOWNLOAD_MODE", raising=False)
    assert settings().download_mode == "on_demand"


def test_download_mode_read_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_DOWNLOAD_MODE", "full")
    assert settings().download_mode == "full"


def test_download_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TIMENET_DOWNLOAD_MODE", "bogus")
    with pytest.raises(ValidationError):
        settings()
