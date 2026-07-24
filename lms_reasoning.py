"""
Reasoning / thinking extraction for EA_LMStudio.

Pure text-processing helpers with no heavy dependencies (only ``re`` and
``typing``). Kept in a standalone module — separate from ``LMStudio.py``, which
imports the lmstudio SDK and ComfyUI and performs a network fetch at import time
— so this logic stays importable and unit-testable on its own.
"""
import re
from typing import Optional, Tuple

# Common reasoning tag patterns used by different models
# Order matters - most common first for efficiency
# Precompiled once at import; these run on every generation.
COMMON_REASONING_PATTERNS = [
    # DeepSeek R1, Qwen3, QwQ, GLM-4/Z1 - most common
    (re.compile(r"<think>(.*?)</think>", re.DOTALL), "<think>", "</think>"),
    # Alternative spelling
    (re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL), "<thinking>", "</thinking>"),
    # Some models use this
    (re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL), "<reasoning>", "</reasoning>"),
    # Occasionally seen
    (re.compile(r"<reason>(.*?)</reason>", re.DOTALL), "<reason>", "</reason>"),
]

# GPT-OSS models use OpenAI's "harmony" channel format:
#   <|start|>assistant<|channel|>analysis<|message|>...<|end|>
#   <|start|>assistant<|channel|>final<|message|>...<|return|>
# LM Studio usually parses this itself, but partial leaks are common: a final
# block with its markers still attached, an analysis block missing its <|end|>
# terminator, a trailing <|return|>, or (when special tokens get stripped
# during detokenization) a bare "analysis...assistantfinal..." string.
#
# Analysis/commentary segments end at <|end|> or at the start of the next
# message/channel header, or at end of text if truncated. The commentary
# channel carries preambles/tool chatter, so it is routed to reasoning too.
GPT_OSS_ANALYSIS_RE = re.compile(
    r"<\|channel\|>(?:analysis|commentary)<\|message\|>(.*?)"
    r"(?=<\|end\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)
GPT_OSS_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)\Z", re.DOTALL)
# Sweep for any harmony markers left over after extraction. <|...|> tokens
# never appear in legitimate prose, so removing them is safe.
HARMONY_MARKER_RE = re.compile(
    r"<\|start\|>(?:assistant|user|system|tool)?"
    r"|<\|channel\|>(?:analysis|commentary|final)?"
    r"|<\|message\|>|<\|end\|>|<\|return\|>|<\|constrain\|>"
)
# Detokenized-without-special-tokens variant: "analysisUser wants...assistantfinalAnswer".
# "assistantfinal" is the discriminator; it never occurs in normal prose.
GPT_OSS_STRIPPED_RE = re.compile(r"\s*analysis(.*?)assistantfinal(.*)\Z", re.DOTALL)

# Quick membership test for whether harmony markers are present at all
_HARMONY_TOKENS = ("<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>")

# Plain-text headings that some models (often community finetunes/merges) emit when
# they "think" without wrapping it in tags. We only use these to DETECT likely leaked
# thinking for a better troubleshooting hint -- we never strip them from the response.
LEAKED_THINKING_MARKERS = (
    "thinking process",
    "thought process",
    "reasoning process",
    "let me think",
    "let's think",
    "step 1:",
    "step 1.",
)


def looks_like_leaked_thinking(text: str) -> bool:
    """Heuristically detect tagless reasoning that leaked into the response.

    Checks only the start of the response (first ~200 chars) so that a marker
    appearing mid-answer in a legitimate reply doesn't trigger a false positive.
    Detection only -- callers must not strip anything based on this.
    """
    if not text:
        return False
    head = text[:200].lower()
    return any(marker in head for marker in LEAKED_THINKING_MARKERS)


def extract_reasoning_gpt_oss(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract reasoning from GPT-OSS "harmony" channel output, including
    partial leaks where only some markers survived LM Studio's own parsing.

    Args:
        text: Full response text

    Returns:
        Tuple of (response, reasoning, detected_pattern), or None if the
        text contains no harmony markers at all.
    """
    if any(token in text for token in _HARMONY_TOKENS):
        reasoning_parts = [m.group(1).strip() for m in GPT_OSS_ANALYSIS_RE.finditer(text)]
        reasoning_parts = [p for p in reasoning_parts if p]
        final_match = GPT_OSS_FINAL_RE.search(text)

        if final_match:
            response = final_match.group(1)
        else:
            response = GPT_OSS_ANALYSIS_RE.sub("", text)

        # Only stray terminators present (e.g. a bare <|end|> with no
        # channel headers): treat text before the terminator as leaked
        # reasoning, mirroring the missing-open-tag fallback below.
        if not reasoning_parts and not final_match and "<|channel|>" not in text:
            before, sep, after = text.partition("<|end|>")
            if sep and before.strip() and after.strip():
                return (
                    HARMONY_MARKER_RE.sub("", after).strip(),
                    before.strip(),
                    "<|end|> (stray terminator)",
                )

        # Sweep any remaining markers (<|return|>, <|start|>assistant, ...)
        response = HARMONY_MARKER_RE.sub("", response).strip()
        return response, "\n---\n".join(reasoning_parts), "<|channel|> (GPT-OSS harmony)"

    # Detokenized variant with special tokens stripped:
    # "analysisUser wants a cat pic.assistantfinalHere is a cat."
    stripped_match = GPT_OSS_STRIPPED_RE.match(text)
    if stripped_match:
        return (
            stripped_match.group(2).strip(),
            stripped_match.group(1).strip(),
            "analysis...assistantfinal (stripped special tokens)",
        )

    return None


def extract_reasoning_auto(text: str) -> Tuple[str, str, Optional[str]]:
    """
    Auto-detect and extract reasoning using common patterns.

    Args:
        text: Full response text

    Returns:
        Tuple of (response_without_reasoning, reasoning_content, detected_pattern)
        detected_pattern is None if no pattern matched
    """
    # Check for GPT-OSS harmony/channel-based format first
    gpt_oss_result = extract_reasoning_gpt_oss(text)
    if gpt_oss_result is not None:
        return gpt_oss_result

    # Check standard tag-based patterns
    for pattern, open_tag, close_tag in COMMON_REASONING_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            reasoning_parts = [m.group(1) for m in matches]
            # Remove all matched reasoning blocks from text
            clean_text = pattern.sub("", text)
            return clean_text.strip(), "\n---\n".join(reasoning_parts).strip(), open_tag

    # Fallback: Check for closing tag without opening tag (model bug/edge case)
    # Some models forget the opening <think> but include closing </think>
    for _, open_tag, close_tag in COMMON_REASONING_PATTERNS:
        if close_tag in text and open_tag not in text:
            # Split on closing tag - everything before is reasoning
            parts = text.split(close_tag, 1)
            if len(parts) == 2:
                reasoning = parts[0].strip()
                response = parts[1].strip()
                if reasoning and response:
                    return response, reasoning, f"{close_tag} (missing open tag)"
                if reasoning and not response:
                    # Model spent its whole budget thinking and never
                    # produced an answer (usually maxTokens truncation).
                    # Surface it as reasoning rather than passing the raw
                    # tagged text through as the response.
                    return "", reasoning, f"{close_tag} (missing open tag, response truncated)"

    # No pattern matched
    return text, "", None


def extract_reasoning_custom(text: str, open_tag: str, close_tag: str) -> Tuple[str, str]:
    """
    Extract reasoning using custom tags.

    Args:
        text: Full response text
        open_tag: Opening tag
        close_tag: Closing tag

    Returns:
        Tuple of (response_without_reasoning, reasoning_content)
    """
    if not open_tag or open_tag not in text:
        return text, ""

    reasoning_parts = []
    response_text = text

    # Extract all reasoning blocks
    while open_tag in response_text:
        start_idx = response_text.find(open_tag)
        end_idx = response_text.find(close_tag, start_idx + len(open_tag))

        if end_idx == -1:
            # No closing tag - take rest as reasoning
            reasoning_parts.append(response_text[start_idx + len(open_tag):])
            response_text = response_text[:start_idx]
            break

        # Extract reasoning content
        reasoning_content = response_text[start_idx + len(open_tag):end_idx]
        reasoning_parts.append(reasoning_content)

        # Remove from response
        response_text = response_text[:start_idx] + response_text[end_idx + len(close_tag):]

    return response_text.strip(), "\n---\n".join(reasoning_parts).strip()
