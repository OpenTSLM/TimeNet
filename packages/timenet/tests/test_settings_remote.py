from timenet.config import settings


def test_token_defaults_to_none(monkeypatch, tmp_path):
    # A .env in the working directory supplies a token of its own, and pydantic-settings reads it
    # when the environment variable is absent. Run from an empty directory so the assertion tests
    # the default rather than whatever the checkout happens to sit next to.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TIMENET_TOKEN", raising=False)
    assert settings().token is None


def test_token_read_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_TOKEN", "tok_abc")
    assert settings().token == "tok_abc"
