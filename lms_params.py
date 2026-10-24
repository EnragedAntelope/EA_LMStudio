"""
Prediction-parameter helpers for EA_LMStudio.

Pure functions that turn widget strings into the shapes the LM Studio SDK
expects. Kept out of ``LMStudio.py`` (which imports the lmstudio SDK and ComfyUI
and performs a network fetch at import time) so they stay unit-testable on their
own.

Everything here is validated against ``LlmPredictionConfigDict`` from the
``lmstudio`` Python SDK. That type is the authoritative list of accepted keys —
and the SDK *silently discards* keys it does not recognise rather than raising,
so a typo or a wished-for parameter becomes a no-op that looks like it worked.
``LMStudio.py`` guards against that by diffing the requested config against the
config the server echoes back on every run.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

# Widget label -> LM Studio contextOverflowPolicy value.
CONTEXT_OVERFLOW_OPTIONS = [
    "Truncate middle",
    "Rolling window",
    "Stop at limit (error)",
]

CONTEXT_OVERFLOW_POLICIES = {
    "Truncate middle": "truncateMiddle",
    "Rolling window": "rollingWindow",
    "Stop at limit (error)": "stopAtLimit",
}

# Widget label -> structured output mode.
#
# "JSON (no schema)" maps to LM Studio's ``{"type": "json"}``, which *asks* for
# JSON but does not constrain decoding: models frequently answer with a
# ```json fenced block. Only the schema form actually constrains the sampler,
# which is why the labels say so rather than promising valid JSON either way.
OUTPUT_FORMAT_OPTIONS = [
    "Text",
    "JSON (no schema)",
    "JSON (schema below)",
]

# stop_reason the server reports when generation hit maxTokens. Referenced by
# name in the structured-output hint; must match the server's spelling exactly.
MAX_TOKENS_STOP_REASON = "maxPredictedTokensReached"

# Escape sequences honoured inside a stop string, so a user can stop on a
# newline (which they cannot type into a single line of the widget).
_STOP_STRING_ESCAPES = (
    ("\\\\", "\x00"),  # protect literal backslashes first
    ("\\n", "\n"),
    ("\\r", "\r"),
    ("\\t", "\t"),
)


def unescape_stop_string(raw: str) -> str:
    """Expand ``\\n``/``\\r``/``\\t``/``\\\\`` in a single stop string."""
    out = raw
    for token, replacement in _STOP_STRING_ESCAPES:
        out = out.replace(token, replacement)
    return out.replace("\x00", "\\")


def parse_stop_strings(raw: str) -> List[str]:
    """Parse the ``stop_strings`` widget into a list for the SDK.

    One stop string per line. Blank lines are ignored so a trailing newline in
    the textarea does not become an empty stop string (which would stop the
    prediction immediately). Leading/trailing spaces are significant and kept —
    stopping on ``"\\nUser:"`` or ``" ###"`` is a real use case — so only the
    line terminator is stripped.
    """
    if not raw:
        return []
    stops = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        stops.append(unescape_stop_string(line))
    return stops


def parse_json_schema(raw: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse the ``json_schema`` widget.

    Returns:
        ``(schema, error)`` — exactly one is non-None. A schema must be a JSON
        object; LM Studio rejects bare arrays/scalars, so we catch that here
        where we can explain it, rather than letting the server fail mid-run.
    """
    text = (raw or "").strip()
    if not text:
        return None, "Output format is 'JSON (schema below)' but json_schema is empty"

    try:
        schema = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json_schema is not valid JSON: {e}"

    if not isinstance(schema, dict):
        return None, (
            f"json_schema must be a JSON object (got {type(schema).__name__}). "
            'Example: {"type": "object", "properties": {"caption": {"type": "string"}}}'
        )

    return schema, None


def build_structured_setting(
    output_format: str, json_schema_text: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build the ``structured`` prediction-config value for the chosen format.

    Returns ``(setting, error)``. ``(None, None)`` means plain text — send
    nothing, which is what a default workflow does.
    """
    if output_format == "JSON (no schema)":
        return {"type": "json"}, None

    if output_format == "JSON (schema below)":
        schema, error = parse_json_schema(json_schema_text)
        if error:
            return None, error
        return {"type": "json", "jsonSchema": schema}, None

    return None, None


def strip_json_code_fence(text: str) -> Tuple[str, bool]:
    """Unwrap a ```json ... ``` fence around an otherwise-valid JSON document.

    Returns ``(text, stripped)``. Only used when the caller asked for JSON
    output and the raw response did not parse: models routinely answer a JSON
    request with a fenced markdown block, which breaks every downstream parser.
    The unwrapped text is returned only if it actually parses as JSON, so this
    can never turn a prose answer into something that looks structured.
    """
    candidate = (text or "").strip()
    if not candidate.startswith("```"):
        return text, False

    # Drop the opening fence line (```json / ```JSON / bare ```) and the closer.
    newline = candidate.find("\n")
    if newline == -1:
        return text, False
    inner = candidate[newline + 1:]
    if inner.rstrip().endswith("```"):
        inner = inner.rstrip()[: -3]

    inner = inner.strip()
    try:
        json.loads(inner)
    except (json.JSONDecodeError, ValueError):
        return text, False

    return inner, True


def _order_like_schema(value: Any, schema: Any) -> Any:
    if isinstance(value, dict) and isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        props = schema["properties"]
        ordered = {k: _order_like_schema(value[k], props[k]) for k in props if k in value}
        ordered.update((k, v) for k, v in value.items() if k not in ordered)
        return ordered
    if isinstance(value, list) and isinstance(schema, dict) and isinstance(schema.get("items"), dict):
        return [_order_like_schema(v, schema["items"]) for v in value]
    return value


def reorder_json_to_schema(text: str, schema: Any) -> Tuple[str, bool]:
    """Put a JSON response's object keys back into the schema's property order.

    The lmstudio SDK serialises requests with msgspec ``order="deterministic"``,
    which sorts every dict key - the schema's ``properties`` included - so the
    model is constrained to emit keys alphabetically. Prompt-as-JSON consumers
    (FIBO) depend on the order. Returns ``(text, changed)``; text that does not
    parse, or is already in order, comes back untouched.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return text, False
    ordered = _order_like_schema(parsed, schema)
    # dict == ignores order, so compare the serialised forms.
    if json.dumps(ordered) == json.dumps(parsed):
        return text, False
    # Keep the model's layout: multi-line in, 2-space indented out; one line in, one line out.
    return json.dumps(ordered, indent=2 if "\n" in text.strip() else None, ensure_ascii=False), True


def missing_config_keys(
    requested: Dict[str, Any], applied: Dict[str, Any]
) -> List[str]:
    """Keys we asked for that LM Studio did not echo back as applied.

    The SDK drops unrecognised prediction-config keys without complaining, so a
    parameter can silently do nothing. Comparing what we sent against the
    ``prediction_config`` the server returns turns that class of bug into a
    visible warning instead of a mystery.
    """
    return [key for key in requested if key not in applied]
