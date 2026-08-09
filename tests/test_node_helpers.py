"""Tests for the node's pure helpers.

Importing ``LMStudio`` pulls in the ``lmstudio``/``comfy`` stubs registered by
conftest and performs the startup model fetch, which fails harmlessly when no LM
Studio server is listening. The helpers exercised here never touch either.
"""
import importlib

# Synthetic package registered by conftest.py (pytest's importlib mode means
# conftest itself is not importable by name, so the name is repeated here).
NODE_PACKAGE = "ea_lmstudio_under_test"

_node = importlib.import_module(f"{NODE_PACKAGE}.LMStudio")
EALMStudio = _node.EALMStudio
_summarize_config = _node._summarize_config


class _Stats:
    """Stand-in for LlmPredictionStats.

    Every field except ``stop_reason`` is Optional on the real type, which is
    exactly what these tests pin down.
    """

    def __init__(self, **fields):
        self.stop_reason = "eosFound"
        self.tokens_per_second = None
        self.prompt_tokens_count = None
        self.predicted_tokens_count = None
        self.total_tokens_count = None
        self.time_to_first_token_sec = None
        self.num_gpu_layers = None
        self.total_draft_tokens_count = None
        self.accepted_draft_tokens_count = None
        self.rejected_draft_tokens_count = None
        self.used_draft_model_key = None
        for key, value in fields.items():
            setattr(self, key, value)


# --- _stats_lines ---------------------------------------------------------

def test_stats_survive_a_backend_that_reports_nothing():
    """The regression that discarded successful generations.

    tokens_per_second is Optional; formatting None with ":.2f" raised TypeError,
    which the outer handler reported as "Generation failed" *after* the model had
    already produced the text.
    """
    lines = EALMStudio._stats_lines(_Stats(), 1.5)
    assert any("Stop reason" in line for line in lines)
    assert any("Total time" in line for line in lines)
    assert not any("Tokens per second" in line for line in lines)


def test_stats_reported_when_present():
    lines = EALMStudio._stats_lines(
        _Stats(
            tokens_per_second=42.123,
            prompt_tokens_count=30,
            predicted_tokens_count=7,
            total_tokens_count=37,
            time_to_first_token_sec=0.1122,
        ),
        2.0,
    )
    joined = "\n".join(lines)
    assert "Tokens per second: 42.12" in joined
    assert "Input tokens: 30" in joined
    assert "Output tokens: 7" in joined
    assert "Total tokens: 37" in joined
    assert "Time to first token: 0.112s" in joined


def test_zero_output_tokens_still_reported():
    """0 is a meaningful count and must not be swallowed as falsy."""
    lines = EALMStudio._stats_lines(_Stats(predicted_tokens_count=0), 1.0)
    assert any("Output tokens: 0" in line for line in lines)


def test_speculative_decoding_acceptance_reported():
    lines = EALMStudio._stats_lines(
        _Stats(
            total_draft_tokens_count=6,
            accepted_draft_tokens_count=5,
            rejected_draft_tokens_count=1,
            used_draft_model_key="tiny-draft",
        ),
        1.0,
    )
    joined = "\n".join(lines)
    assert "tiny-draft" in joined
    assert "5/6 draft tokens accepted (83%)" in joined
    assert "HINT" not in joined  # good acceptance rate, no nagging


def test_poor_draft_acceptance_gets_a_hint():
    lines = EALMStudio._stats_lines(
        _Stats(
            total_draft_tokens_count=100,
            accepted_draft_tokens_count=5,
            rejected_draft_tokens_count=95,
        ),
        1.0,
    )
    assert any("HINT" in line for line in lines)


# --- _split_reasoning -----------------------------------------------------

def test_native_reasoning_wins_over_regex():
    """LM Studio's own parser is authoritative when it fired."""
    log = []
    answer, reasoning = EALMStudio._split_reasoning(
        "The answer is 42.",
        "let me work this out",
        "The answer is 42.",
        "Auto-detect (recommended)",
        "<think>",
        "</think>",
        log,
    )
    assert answer == "The answer is 42."
    assert reasoning == "let me work this out"
    assert any("LM Studio's own parser" in line for line in log)


def test_native_reasoning_falls_back_when_no_plain_fragments():
    answer, reasoning = EALMStudio._split_reasoning(
        "only content here", "thinking", "", "Auto-detect (recommended)",
        "<think>", "</think>", [],
    )
    assert answer == "only content here"
    assert reasoning == "thinking"


def test_regex_used_when_lmstudio_did_not_tag():
    answer, reasoning = EALMStudio._split_reasoning(
        "<think>hmm</think>Final answer.", "", "", "Auto-detect (recommended)",
        "<think>", "</think>", [],
    )
    assert answer == "Final answer."
    assert reasoning == "hmm"


def test_disabled_mode_passes_text_through():
    answer, reasoning = EALMStudio._split_reasoning(
        "<think>hmm</think>Final answer.", "", "", "Disabled",
        "<think>", "</think>", [],
    )
    assert answer == "<think>hmm</think>Final answer."
    assert reasoning == ""


def test_custom_tags_mode():
    answer, reasoning = EALMStudio._split_reasoning(
        "[R]secret[/R]Answer.", "", "", "Custom tags", "[R]", "[/R]", [],
    )
    assert answer == "Answer."
    assert reasoning == "secret"


def test_whitespace_only_native_reasoning_does_not_hijack():
    """A stream of blank reasoning fragments must not suppress the regex path."""
    answer, reasoning = EALMStudio._split_reasoning(
        "<think>hmm</think>Final answer.", "   \n ", "", "Auto-detect (recommended)",
        "<think>", "</think>", [],
    )
    assert answer == "Final answer."
    assert reasoning == "hmm"


# --- ui payload -----------------------------------------------------------

def test_output_carries_both_ui_and_result():
    """OUTPUT_NODE is only worth having because of the ui half."""
    out = EALMStudio._output("hello", "thinking", ["[INFO] a", "[INFO] b"])
    assert out["result"] == ("hello", "thinking", "[INFO] a\n[INFO] b")
    assert out["ui"]["text"] == ["hello"]
    assert out["ui"]["reasoning"] == ["thinking"]


# --- _summarize_config ----------------------------------------------------

def test_config_summary_truncates_long_values():
    summary = _summarize_config({"structured": {"jsonSchema": {"x": "y" * 500}}})
    assert "truncated" in summary
    assert len(summary) < 300


def test_config_summary_lists_every_key():
    summary = _summarize_config({"temperature": 0.7, "maxTokens": 10})
    assert "temperature=0.7" in summary
    assert "maxTokens=10" in summary
