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
