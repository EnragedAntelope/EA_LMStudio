"""Tests for model_fetcher: identifier validation and exclusion matching."""
from model_fetcher import validate_model_identifier, _is_excluded


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
