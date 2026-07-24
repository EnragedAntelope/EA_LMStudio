"""Tests for lms_reasoning: reasoning/thinking extraction."""
from lms_reasoning import (
    extract_reasoning_auto,
    extract_reasoning_custom,
    extract_reasoning_gpt_oss,
    looks_like_leaked_thinking,
)


# --- Standard tag-based extraction --------------------------------------

def test_think_tag_basic():
    resp, reasoning, pattern = extract_reasoning_auto("<think>pondering</think>answer")
    assert resp == "answer"
    assert reasoning == "pondering"
    assert pattern == "<think>"


def test_multiple_think_blocks_joined():
    resp, reasoning, pattern = extract_reasoning_auto("<think>a</think>X<think>b</think>Y")
    assert resp == "XY"
    assert reasoning == "a\n---\nb"
    assert pattern == "<think>"


def test_thinking_variant():
    resp, reasoning, pattern = extract_reasoning_auto("<thinking>t</thinking>ans")
    assert (resp, reasoning, pattern) == ("ans", "t", "<thinking>")


def test_reasoning_variant():
    resp, reasoning, pattern = extract_reasoning_auto("<reasoning>r</reasoning>done")
    assert (resp, reasoning, pattern) == ("done", "r", "<reasoning>")


def test_reason_variant():
    resp, reasoning, pattern = extract_reasoning_auto("<reason>r</reason>done")
    assert (resp, reasoning, pattern) == ("done", "r", "<reason>")


def test_no_reasoning_passthrough():
    resp, reasoning, pattern = extract_reasoning_auto("just a plain answer")
    assert resp == "just a plain answer"
    assert reasoning == ""
    assert pattern is None


# --- Missing-open-tag fallbacks -----------------------------------------

def test_missing_open_tag_with_answer():
    resp, reasoning, pattern = extract_reasoning_auto("leaked reasoning</think>the answer")
    assert resp == "the answer"
    assert reasoning == "leaked reasoning"
    assert pattern == "</think> (missing open tag)"


def test_missing_open_tag_truncated_response():
    resp, reasoning, pattern = extract_reasoning_auto("all thinking no answer</think>")
    assert resp == ""
    assert reasoning == "all thinking no answer"
    assert pattern == "</think> (missing open tag, response truncated)"


# --- GPT-OSS harmony format ---------------------------------------------

def test_harmony_full():
    text = (
        "<|channel|>analysis<|message|>thinking<|end|>"
        "<|channel|>final<|message|>answer<|return|>"
    )
    resp, reasoning, pattern = extract_reasoning_auto(text)
    assert resp == "answer"
    assert reasoning == "thinking"
    assert pattern == "<|channel|> (GPT-OSS harmony)"


def test_harmony_analysis_only():
    resp, reasoning, pattern = extract_reasoning_gpt_oss(
        "<|channel|>analysis<|message|>thinking only"
    )
    assert reasoning == "thinking only"
    assert pattern == "<|channel|> (GPT-OSS harmony)"


def test_harmony_stray_terminator():
    resp, reasoning, pattern = extract_reasoning_gpt_oss("some reasoning<|end|>the answer")
    assert resp == "the answer"
    assert reasoning == "some reasoning"
    assert pattern == "<|end|> (stray terminator)"


def test_harmony_stripped_special_tokens():
    resp, reasoning, pattern = extract_reasoning_gpt_oss(
        "analysisUser wants a cat.assistantfinalHere is a cat."
    )
    assert resp == "Here is a cat."
    assert reasoning == "User wants a cat."
    assert pattern == "analysis...assistantfinal (stripped special tokens)"


def test_harmony_returns_none_without_markers():
    assert extract_reasoning_gpt_oss("plain text, no harmony") is None


# --- Custom tag extraction ----------------------------------------------

def test_custom_tags_basic():
    resp, reasoning = extract_reasoning_custom("<x>r</x>ans", "<x>", "</x>")
    assert resp == "ans"
    assert reasoning == "r"


def test_custom_tags_no_open_tag():
    resp, reasoning = extract_reasoning_custom("plain", "<x>", "</x>")
    assert resp == "plain"
    assert reasoning == ""


def test_custom_tags_no_close_tag_takes_rest():
    resp, reasoning = extract_reasoning_custom("<x>rest of text", "<x>", "</x>")
    assert resp == ""
    assert reasoning == "rest of text"


# --- Leaked-thinking heuristic ------------------------------------------

def test_leaked_thinking_detected_at_head():
    assert looks_like_leaked_thinking("Thinking process: let's work this out")


def test_leaked_thinking_not_in_normal_answer():
    assert not looks_like_leaked_thinking("The answer is 42.")


def test_leaked_thinking_only_checks_head():
    # Marker appears well past the first 200 chars -> not flagged.
    text = ("x" * 300) + " let me think"
    assert not looks_like_leaked_thinking(text)


def test_leaked_thinking_empty():
    assert not looks_like_leaked_thinking("")
