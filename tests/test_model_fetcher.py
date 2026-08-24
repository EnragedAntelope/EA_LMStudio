"""Tests for model_fetcher: identifier validation, exclusion and discovery."""
import model_fetcher
from model_fetcher import (
    CUSTOM_MODEL_OPTION,
    _is_excluded,
    fetch_models_from_server,
    get_default_model_choice,
    validate_model_identifier,
)


# --- validate_model_identifier ------------------------------------------

def test_valid_org_slash_name():
    ok, err = validate_model_identifier(
        "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    )
    assert ok is True
    assert err is None


def test_valid_quant_suffix():
    ok, err = validate_model_identifier("qwen2.5-7b@q4_k_m")
    assert ok is True
    assert err is None


def test_empty_is_invalid():
    ok, err = validate_model_identifier("")
    assert ok is False
    assert err


def test_whitespace_only_is_invalid():
    ok, err = validate_model_identifier("   ")
    assert ok is False


def test_path_traversal_rejected():
    ok, err = validate_model_identifier("../../etc/passwd")
    assert ok is False
    assert ".." in err


def test_overlong_rejected():
    ok, err = validate_model_identifier("a" * 257)
    assert ok is False
    assert "length" in err.lower()


def test_invalid_characters_rejected():
    ok, err = validate_model_identifier("bad name with spaces")
    assert ok is False


# --- _is_excluded --------------------------------------------------------

def test_excluded_substring_match():
    assert _is_excluded("text-embedding-ada-002", ["embedding"]) is True


def test_excluded_is_case_insensitive_both_sides():
    assert _is_excluded("MyEmbeddingModel", ["embedding"]) is True
    assert _is_excluded("qwen3-coder-30b", ["Qwen3-Coder"]) is True


def test_not_excluded():
    assert _is_excluded("llama-3-8b", ["embedding"]) is False


def test_empty_patterns_excludes_nothing():
    assert _is_excluded("anything", []) is False


# --- fetch_models_from_server -------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_server(monkeypatch, ids, capture=None):
    payload = {"data": [{"id": model_id} for model_id in ids]}

    def fake_get(*args, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(payload)

    # The fetcher reuses one requests.Session; patch its get, not the module.
    monkeypatch.setattr(model_fetcher._session, "get", fake_get)


def test_fetch_sorts_and_excludes(monkeypatch):
    _fake_server(monkeypatch, ["zeta-7b", "text-embedding-small", "Alpha-3b"])
    models, error, rejected = fetch_models_from_server("http://x", 1.0, ["embedding"])
    assert error is None
    assert models == ["Alpha-3b", "zeta-7b"]
    assert rejected == []


def test_unsafe_identifier_is_reported_not_just_dropped(monkeypatch):
    """LM Studio really does serve ids like "model@?".

    Those fail validation, and before v2.0.0 they vanished from the dropdown
    with no explanation anywhere.
    """
    _fake_server(monkeypatch, ["good-model", "gemma4-31b-balanced-mtp@?"])
    models, error, rejected = fetch_models_from_server("http://x", 1.0, [])
    assert error is None
    assert models == ["good-model"]
    assert rejected == ["gemma4-31b-balanced-mtp@?"]


def test_missing_data_field_is_an_error(monkeypatch):
    monkeypatch.setattr(
        model_fetcher._session,
        "get",
        lambda *a, **kw: _FakeResponse({"oops": []}),
    )
    models, error, rejected = fetch_models_from_server("http://x", 1.0, [])
    assert models == []
    assert rejected == []
    assert "data" in error


# --- get_default_model_choice -------------------------------------------

def test_default_is_a_real_model_when_available(monkeypatch):
    """A fresh node must land on a usable model, not the Custom sentinel."""
    monkeypatch.setattr(model_fetcher, "_cached_models", ["alpha-3b", "zeta-7b"])
    assert get_default_model_choice() == "alpha-3b"


def test_default_falls_back_to_custom_when_discovery_failed(monkeypatch):
    monkeypatch.setattr(model_fetcher, "_cached_models", [])
    assert get_default_model_choice() == CUSTOM_MODEL_OPTION


# --- auth headers / origin guard -----------------------------------------

def test_auth_headers_empty_without_token():
    assert model_fetcher.auth_headers(None) == {}
    assert model_fetcher.auth_headers("") == {}
    assert model_fetcher.auth_headers("   ") == {}


def test_auth_headers_bearer_with_token():
    assert model_fetcher.auth_headers("sekrit") == {"Authorization": "Bearer sekrit"}


def test_fetch_passes_headers_to_the_session(monkeypatch):
    seen = {}
    _fake_server(monkeypatch, ["alpha-3b"], capture=seen)
    fetch_models_from_server(
        "http://x", 1.0, [], headers={"Authorization": "Bearer t"}
    )
    assert seen.get("headers") == {"Authorization": "Bearer t"}


def test_origin_guard_allows_missing_origin():
    """curl and server-to-server callers send no Origin header."""
    assert model_fetcher.origin_matches_host(None, "127.0.0.1:8188") is True


def test_origin_guard_rejects_foreign_origin():
    assert (
        model_fetcher.origin_matches_host(
            "http://evil.example", "127.0.0.1:8188"
        )
        is False
    )


def test_origin_guard_accepts_matching_origin():
    assert (
        model_fetcher.origin_matches_host(
            "http://127.0.0.1:8188", "127.0.0.1:8188"
        )
        is True
    )


def test_origin_guard_rejects_when_host_missing():
    assert model_fetcher.origin_matches_host("http://evil.example", None) is False
