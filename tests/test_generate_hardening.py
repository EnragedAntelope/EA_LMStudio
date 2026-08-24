"""Tests for the generate()-path hardening added in 2.1.0.

Covers the extracted post-processing step (``_finalize_response``), the
partial-output-preserving stream wrapper (``_stream``), and the error-hint
mapping - the pieces an earlier audit flagged as untested substring
heuristics with the power to discard successful generations.
"""
import importlib
from types import SimpleNamespace

import pytest

NODE_PACKAGE = "ea_lmstudio_under_test"

_node = importlib.import_module(f"{NODE_PACKAGE}.LMStudio")
EALMStudio = _node.EALMStudio


class _Stats:
    stop_reason = "eosFound"


def _response(content="the answer", structured=None):
    return SimpleNamespace(
        content=content,
        stats=_Stats(),
        structured=structured,
        prediction_config=None,
    )


# --- _error_hints ----------------------------------------------------------

@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("connection refused", "server enabled"),
        ("context length exceeded", "context length"),
        ("invalid schema in json_schema", "JSON Schema"),
        ("model not found", "matches LM Studio"),
        ("image rejected by the VLM backend", "single image"),
    ],
)
def test_error_hints_match_known_failures(fragment, expected):
    hints = EALMStudio._error_hints(fragment)
    assert any(expected.lower() in h.lower() for h in hints)


def test_timeout_gets_its_own_hint():
    """The SDK's sync timeout surfaces as LMStudioTimeoutError ('timed out')."""
    hints = EALMStudio._error_hints("prediction timed out after 60s")
    assert any("no data" in h for h in hints)


def test_unknown_error_gets_no_hint_rather_than_a_wrong_one():
    assert EALMStudio._error_hints("quantization exploded") == []


# --- _finalize_response ----------------------------------------------------

def _run_finalize(response, response_text="raw text", **overrides):
    kwargs = dict(
        gen_config={"temperature": 0.7},
        elapsed=1.0,
        structured=None,
        native_reasoning="",
        plain_content="",
        reasoning_mode="Disabled",
        custom_open_tag="<think>",
        custom_close_tag="</think>",
        troubleshooting_lines=[],
    )
    kwargs.update(overrides)
    return EALMStudio._finalize_response(response, response_text, **kwargs)


def test_finalize_returns_the_response_text_on_the_happy_path():
    final, reasoning = _run_finalize(_response("hello"), "hello")
    assert (final, reasoning) == ("hello", "")


def test_finalize_splits_tags_when_auto_mode_is_on():
    final, reasoning = _run_finalize(
        _response("<think>hmm</think>answer"),
        "<think>hmm</think>answer",
        reasoning_mode="Auto-detect (recommended)",
    )
    assert reasoning == "hmm"
    assert final == "answer"


def test_finalize_degrades_to_raw_text_when_post_processing_raises():
    """A crash AFTER generation must not discard what the model produced."""

    class _RaisingResponse:
        content = "precious text"
        stats = _Stats()
        prediction_config = None
        structured = property(lambda self: (_ for _ in ()).throw(RuntimeError("garbage")))

    lines = []
    final, reasoning = _run_finalize(
        _RaisingResponse(),
        response_text="precious text",
        structured={"type": "json"},
        troubleshooting_lines=lines,
    )

    assert final == "precious text"
    assert reasoning == ""
    assert any("post-processing failed" in line for line in lines)


# --- _stream partial-output preservation -----------------------------------

class _Fragment:
    def __init__(self, content):
        self.content = content
        self.reasoning_type = "none"
        self.tokens_count = 1


class _DyingStream:
    """Yields two fragments then dies, as a stalled socket/timeout would."""

    def __init__(self):
        self.cancel_calls = 0

    def __iter__(self):
        yield _Fragment("par")
        yield _Fragment("tial")
        raise RuntimeError("websocket closed")

    def cancel(self):
        self.cancel_calls += 1

    def result(self):  # pragma: no cover - must never be reached on error
        raise AssertionError("result() must not be called after a stream error")


def test_stream_preserves_partial_output_when_it_dies_midflight(monkeypatch):
    monkeypatch.setattr(
        _node.model_management, "processing_interrupted", lambda: False, raising=False
    )
    monkeypatch.setattr(_node, "ProgressBar", None, raising=False)

    model = SimpleNamespace(respond_stream=lambda chat, config=None: _DyingStream())
    result, native, plain, interrupted, error = EALMStudio()._stream(model, None, {}, 16)

    assert result is None
    assert plain == "partial"
    assert native == ""
    assert interrupted is False
    assert error is not None and "RuntimeError" in error


def test_stream_still_reports_clean_finishes(monkeypatch):
    monkeypatch.setattr(
        _node.model_management, "processing_interrupted", lambda: False, raising=False
    )
    monkeypatch.setattr(_node, "ProgressBar", None, raising=False)

    class _Stream:
        def __iter__(self):
            yield _Fragment("done")

        def cancel(self):  # pragma: no cover
            pass

        def result(self):
            return _response("done")

    model = SimpleNamespace(respond_stream=lambda chat, config=None: _Stream())
    result, _, plain, interrupted, error = EALMStudio()._stream(model, None, {}, 16)

    assert error is None
    assert interrupted is False
    assert plain == "done"
    assert result.content == "done"


# --- _prepare_images -------------------------------------------------------
#
# The image-upload path had no test at all, which is how a deleted
# ``from tempfile import NamedTemporaryFile`` shipped on this branch: every
# vision run raised NameError while 151 tests stayed green. These pin the
# whole round trip - temp file written, closed, uploaded, then removed.

class _RecordingFiles:
    def __init__(self):
        self.uploaded = []

    def prepare_image(self, path):
        with open(path, "rb") as fh:
            assert fh.read(2) == b"\xff\xd8", "uploaded file is not JPEG"
        self.uploaded.append(path)
        return f"handle:{len(self.uploaded)}"


class _FilesClient:
    def __init__(self):
        self.files = _RecordingFiles()


def _pil_image(size=(8, 8)):
    from PIL import Image

    return Image.new("RGB", size, "red")


def test_prepare_images_uploads_each_image_and_cleans_up():
    import os

    client = _FilesClient()
    handles = EALMStudio._prepare_images(client, [_pil_image(), _pil_image((4, 4))])

    assert handles == ["handle:1", "handle:2"]
    for path in client.files.uploaded:
        assert not os.path.exists(path), "temp image was left on disk"


def test_prepare_images_removes_the_temp_file_even_when_upload_fails():
    import os

    seen = []

    class _FailingFiles:
        def prepare_image(self, path):
            seen.append(path)
            raise RuntimeError("upload rejected")

    class _FailingClient:
        files = _FailingFiles()

    with pytest.raises(RuntimeError):
        EALMStudio._prepare_images(_FailingClient(), [_pil_image()])

    assert seen and not os.path.exists(seen[0])


class _StreamWithDeadResult:
    """Drains cleanly, then .result() raises - a socket dropping at the finish."""

    def __iter__(self):
        yield _Fragment("almost ")
        yield _Fragment("everything")

    def cancel(self):  # pragma: no cover - not interrupted here
        pass

    def result(self):
        raise RuntimeError("connection reset")


def test_stream_preserves_output_when_result_fails_after_a_clean_drain(monkeypatch):
    """Guarding only the fragment loop still lost a completed generation."""
    monkeypatch.setattr(
        _node.model_management, "processing_interrupted", lambda: False, raising=False
    )
    monkeypatch.setattr(_node, "ProgressBar", None, raising=False)

    model = SimpleNamespace(respond_stream=lambda chat, config=None: _StreamWithDeadResult())
    result, native, plain, interrupted, error = EALMStudio()._stream(model, None, {}, 16)

    assert result is None
    assert plain == "almost everything"
    assert interrupted is False
    assert error is not None and "RuntimeError" in error


# --- stream-error reporting honesty ---------------------------------------
#
# Measured live against LM Studio: asking a text-only model for a vision
# response fails mid-stream with "No engine protocol runtime is registered".
# The node reported "Returning the partial text received before the failure"
# while returning nothing at all, and offered no hint about the real cause.

def test_error_hints_fire_for_a_vision_capability_failure():
    hints = EALMStudio._error_hints(
        "lmstudioservererror: chat response error: no engine protocol runtime "
        "is registered for 'abc'.",
        has_images=True,
    )
    assert any("vision (VLM) model" in h for h in hints)


def test_engine_runtime_failure_without_images_points_at_the_runtime():
    hints = EALMStudio._error_hints(
        "no engine protocol runtime is registered for 'abc'.", has_images=False
    )
    assert any("runtime/format" in h for h in hints)
    assert not any("VLM" in h for h in hints)


def test_an_unrecognised_failure_on_an_image_run_names_the_likely_cause():
    assert EALMStudio._error_hints("quantization exploded", has_images=True)
    assert EALMStudio._error_hints("quantization exploded", has_images=False) == []
