"""Tests for lms_config.config_manager: config derivation from dicts.

All getters accept a pre-loaded config dict, so these tests exercise the
merge/derivation logic without touching the filesystem.
"""
from lms_config.config_manager import ConfigManager


def _cm():
    return ConfigManager()


# --- get_excluded_patterns ----------------------------------------------

def test_excluded_defaults_when_absent():
    assert _cm().get_excluded_patterns({}) == ["embedding"]


def test_excluded_merges_user_patterns():
    result = _cm().get_excluded_patterns(
        {"excluded_model_patterns": ["custom", "coder"]}
    )
    assert result == ["embedding", "custom", "coder"]


def test_excluded_always_keeps_embedding():
    # "embedding" cannot be removed even if the user omits it.
    result = _cm().get_excluded_patterns({"excluded_model_patterns": ["coder"]})
    assert "embedding" in result


def test_excluded_dedupes():
    result = _cm().get_excluded_patterns(
        {"excluded_model_patterns": ["embedding", "embedding"]}
    )
    assert result == ["embedding"]


def test_excluded_bad_type_falls_back_to_defaults():
    result = _cm().get_excluded_patterns({"excluded_model_patterns": "not-a-list"})
    assert result == ["embedding"]


# --- get_server_url / get_timeout ---------------------------------------

def test_server_url_from_config():
    assert _cm().get_server_url({"server_host": "h", "server_port": 42}) == "http://h:42"


def test_server_url_defaults():
    assert _cm().get_server_url({}) == "http://127.0.0.1:1234"


def test_timeout_from_config_is_float():
    val = _cm().get_timeout({"timeout_seconds": 10})
    assert val == 10.0
    assert isinstance(val, float)


def test_timeout_default():
    assert _cm().get_timeout({}) == 5.0


# --- junk-value tolerance (a non-numeric timeout used to crash the whole
# node pack at import, because the startup fetch coerces it eagerly) ---

def test_timeout_junk_falls_back_to_default():
    assert _cm().get_timeout({"timeout_seconds": "abc"}) == 5.0
    assert _cm().get_timeout({"timeout_seconds": None}) == 5.0


def test_port_junk_or_range_falls_back_to_1234():
    cm = _cm()
    assert cm.get_server_url({"server_host": "h", "server_port": "nope"}) == "http://h:1234"
    assert cm.get_server_address({"server_host": "h", "server_port": 99999}) == "h:1234"
    assert cm.get_server_address({"server_host": "h", "server_port": 0}) == "h:1234"


def test_host_with_pasted_scheme_is_stripped():
    """Users routinely paste "http://192.168.1.50" into server_host."""
    cfg = {"server_host": "http://192.168.1.50/", "server_port": 1234}
    assert _cm().get_server_url(cfg) == "http://192.168.1.50:1234"
    assert _cm().get_server_address(cfg) == "192.168.1.50:1234"


def test_get_config_returns_isolated_lists():
    """A shallow .copy() aliased excluded_model_patterns across callers."""
    cm = _cm()
    first = cm.get_config()
    first["excluded_model_patterns"].append("mutated")
    second = cm.get_config()
    assert "mutated" not in second["excluded_model_patterns"]


# --- api_token (config file first, then LM_API_TOKEN env) ---

def test_api_token_from_config(monkeypatch):
    monkeypatch.delenv("LM_API_TOKEN", raising=False)
    assert _cm().get_api_token({"api_token": "tok"}) == "tok"


def test_api_token_env_fallback(monkeypatch):
    monkeypatch.setenv("LM_API_TOKEN", "envtok")
    assert _cm().get_api_token({}) == "envtok"


def test_api_token_config_wins_over_env(monkeypatch):
    monkeypatch.setenv("LM_API_TOKEN", "envtok")
    assert _cm().get_api_token({"api_token": "filetok"}) == "filetok"


def test_api_token_absent_everywhere(monkeypatch):
    monkeypatch.delenv("LM_API_TOKEN", raising=False)
    assert _cm().get_api_token({}) == ""


def test_default_config_template_includes_empty_token():
    """The shipped template documents the key without suggesting a widget."""
    from lms_config.config_manager import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.get("api_token") == ""
