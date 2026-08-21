from timenet.config import settings


def test_token_defaults_to_none(monkeypatch):
    monkeypatch.delenv("TIMENET_TOKEN", raising=False)
    assert settings().token is None


def test_token_read_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_TOKEN", "tok_abc")
    assert settings().token == "tok_abc"
