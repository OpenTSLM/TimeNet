from timenet.config import settings


def test_token_read_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_TOKEN", "tok_abc")
    assert settings().token == "tok_abc"
