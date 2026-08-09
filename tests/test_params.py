"""Tests for lms_params: widget strings -> LM Studio prediction config values."""
import pytest

from lms_params import (
    CONTEXT_OVERFLOW_OPTIONS,
    CONTEXT_OVERFLOW_POLICIES,
    OUTPUT_FORMAT_OPTIONS,
    build_structured_setting,
    missing_config_keys,
    parse_json_schema,
    parse_stop_strings,
    strip_json_code_fence,
    unescape_stop_string,
)


# --- parse_stop_strings ---------------------------------------------------

def test_empty_gives_no_stop_strings():
    assert parse_stop_strings("") == []
    assert parse_stop_strings("\n\n  \n") == []


def test_one_per_line():
    assert parse_stop_strings("END\nUser:") == ["END", "User:"]


def test_blank_lines_dropped():
    """A trailing newline must not become an empty stop string.

    An empty stop string matches immediately, so it would truncate every
    response to nothing.
    """
    assert parse_stop_strings("END\n\n") == ["END"]
    assert "" not in parse_stop_strings("A\n\n\nB\n")


def test_crlf_handled():
    assert parse_stop_strings("A\r\nB") == ["A", "B"]


def test_significant_whitespace_preserved():
    assert parse_stop_strings(" ###") == [" ###"]
    assert parse_stop_strings("END ") == ["END "]


def test_escape_sequences_expanded():
    assert parse_stop_strings("\\nUser:") == ["\nUser:"]
    assert parse_stop_strings("a\\tb") == ["a\tb"]


def test_literal_backslash_not_eaten_by_escape():
    # "\\n" (an escaped backslash followed by n) must stay a backslash + n,
    # not become a newline.
    assert unescape_stop_string("\\\\n") == "\\n"


# --- parse_json_schema ----------------------------------------------------

def test_empty_schema_is_an_error():
    schema, error = parse_json_schema("   ")
    assert schema is None
    assert "empty" in error.lower()


def test_invalid_json_reports_error():
    schema, error = parse_json_schema("{not json}")
    assert schema is None
    assert "not valid json" in error.lower()


def test_non_object_schema_rejected():
    schema, error = parse_json_schema('["a", "b"]')
    assert schema is None
    assert "object" in error.lower()


def test_valid_schema_parsed():
    schema, error = parse_json_schema('{"type": "object", "properties": {}}')
    assert error is None
    assert schema["type"] == "object"


# --- build_structured_setting --------------------------------------------

def test_text_sends_nothing():
    setting, error = build_structured_setting("Text", "")
    assert setting is None
    assert error is None


def test_free_form_json():
    setting, error = build_structured_setting("JSON (no schema)", "")
    assert error is None
    assert setting == {"type": "json"}


def test_schema_mode_includes_schema():
    setting, error = build_structured_setting(
        "JSON (schema below)", '{"type": "object"}'
    )
    assert error is None
    assert setting["type"] == "json"
    assert setting["jsonSchema"] == {"type": "object"}


def test_schema_mode_without_schema_errors():
    setting, error = build_structured_setting("JSON (schema below)", "")
    assert setting is None
    assert error


def test_free_form_json_ignores_schema_box():
    """Leftover text in json_schema must not break the 'any shape' mode."""
    setting, error = build_structured_setting("JSON (no schema)", "{not json}")
    assert error is None
    assert setting == {"type": "json"}


# --- strip_json_code_fence ------------------------------------------------

def test_fenced_json_is_unwrapped():
    """Verified live: 'JSON (no schema)' really does come back fenced."""
    text, stripped = strip_json_code_fence('```json\n{"sky": "blue"}\n```')
    assert stripped is True
    assert text == '{"sky": "blue"}'


def test_bare_fence_without_language_tag():
    text, stripped = strip_json_code_fence('```\n{"a": 1}\n```')
    assert stripped is True
    assert text == '{"a": 1}'


def test_unterminated_fence_still_unwrapped_if_valid():
    # Truncated responses often lose the closing fence.
    text, stripped = strip_json_code_fence('```json\n{"a": 1}')
    assert stripped is True
    assert text == '{"a": 1}'


def test_plain_json_left_alone():
    text, stripped = strip_json_code_fence('{"a": 1}')
    assert stripped is False
    assert text == '{"a": 1}'


def test_prose_never_becomes_structured():
    """Unwrapping must never dress up a non-JSON answer as JSON."""
    original = "```json\nI could not answer that.\n```"
    text, stripped = strip_json_code_fence(original)
    assert stripped is False
    assert text == original


def test_fenced_non_json_code_left_alone():
    original = "```python\nprint(1)\n```"
    text, stripped = strip_json_code_fence(original)
    assert stripped is False
    assert text == original


def test_empty_input_is_safe():
    assert strip_json_code_fence("") == ("", False)


# --- missing_config_keys --------------------------------------------------

def test_nothing_missing():
    assert missing_config_keys({"temperature": 0.7}, {"temperature": 0.7}) == []


def test_dropped_key_detected():
    """The check that would have caught presencePenalty/enableThinking.

    The lmstudio SDK discards config keys it does not know instead of raising,
    so the only evidence is the config the server echoes back.
    """
    requested = {"temperature": 0.7, "presencePenalty": 0.5}
    applied = {"temperature": 0.7, "cpuThreads": 8}
    assert missing_config_keys(requested, applied) == ["presencePenalty"]


def test_extra_applied_keys_are_not_reported():
    # LM Studio echoes back defaults we never asked for; those are not a problem.
    assert missing_config_keys({"temperature": 1.0}, {"temperature": 1.0, "rawTools": {}}) == []


# --- option tables --------------------------------------------------------

@pytest.mark.parametrize("label", CONTEXT_OVERFLOW_OPTIONS)
def test_every_context_option_maps_to_a_policy(label):
    """A widget label with no mapping would silently fall back to the default."""
    assert label in CONTEXT_OVERFLOW_POLICIES


def test_context_policies_are_valid_sdk_values():
    # LlmPredictionConfigDict["contextOverflowPolicy"] literal values.
    assert set(CONTEXT_OVERFLOW_POLICIES.values()) <= {
        "stopAtLimit",
        "truncateMiddle",
        "rollingWindow",
    }


def test_default_context_option_preserves_v1_behaviour():
    assert CONTEXT_OVERFLOW_POLICIES[CONTEXT_OVERFLOW_OPTIONS[0]] == "truncateMiddle"


def test_default_output_format_is_text():
    assert OUTPUT_FORMAT_OPTIONS[0] == "Text"


@pytest.mark.parametrize("label", OUTPUT_FORMAT_OPTIONS)
def test_every_output_format_is_handled(label):
    setting, error = build_structured_setting(label, '{"type": "object"}')
    # Either it produces a setting or it is the plain-text mode; never an error
    # for a well-formed schema.
    assert error is None
    assert setting is None or setting["type"] == "json"
