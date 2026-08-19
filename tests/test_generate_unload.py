"""Tests for how ``generate()`` unloads the model.

These drive the real ``generate()`` with a fake ``lmstudio`` module patched onto
the node, which is the only way to pin down *when* the unload happens - the
point of the change is that it survives a failed generation.
"""
import importlib

import pytest

NODE_PACKAGE = "ea_lmstudio_under_test"

_node = importlib.import_module(f"{NODE_PACKAGE}.LMStudio")
EALMStudio = _node.EALMStudio
CUSTOM_MODEL_OPTION = _node.CUSTOM_MODEL_OPTION


class _Stats:
    stop_reason = "eosFound"
    tokens_per_second = None
    prompt_tokens_count = None
    predicted_tokens_count = None
    total_tokens_count = None
    time_to_first_token_sec = None
    num_gpu_layers = None
    total_draft_tokens_count = None
    accepted_draft_tokens_count = None
    rejected_draft_tokens_count = None
    used_draft_model_key = None


class _Fragment:
    def __init__(self, content, reasoning_type="none"):
        self.content = content
        self.reasoning_type = reasoning_type
        self.tokens_count = 1


class _Result:
    def __init__(self, content):
        self.content = content
        self.stats = _Stats()
        self.structured = None
        self.prediction_config = None


class _Stream:
    def __init__(self, content):
        self._content = content

    def __iter__(self):
        yield _Fragment(self._content)

    def result(self):
        return _Result(self._content)

    def cancel(self):  # pragma: no cover - never interrupted in these tests
        pass


class _ModelHandle:
    """A handle whose two names differ, as they do for `lms load --identifier`."""

    def __init__(self, identifier, model_key, fail=None):
        self.identifier = identifier
        self.model_key = model_key
        self._fail = fail
        self.direct_unload_calls = 0

    def respond_stream(self, chat, config=None):
        if self._fail is not None:
            raise self._fail
        return _Stream("hello")

    def unload(self):
        self.direct_unload_calls += 1


class _LLMNamespace:
    def __init__(self, handles):
        # Keyed by every name each handle answers to, mirroring LM Studio.
        self._by_name = {}
        for handle in handles:
            self._by_name[handle.identifier] = handle
            self._by_name[handle.model_key] = handle
        self.loaded = list(handles)

    def model(self, identifier):
        return self._by_name[identifier]

    def list_loaded(self):
        return list(self.loaded)


class _Client:
    def __init__(self, llm):
        self.llm = llm

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Chat:
    def __init__(self, system_message=None):
        self.system_message = system_message

    def add_user_message(self, prompt, images=None):
        pass


@pytest.fixture
def lm_studio(monkeypatch):
    """Patch the node's ``lms`` module and hand back the loaded-model namespace."""

    # ComfyUI's real model_management provides these; conftest's stub is bare.
    monkeypatch.setattr(
        _node.model_management, "processing_interrupted", lambda: False, raising=False
    )
    monkeypatch.setattr(
        _node.model_management,
        "throw_exception_if_processing_interrupted",
        lambda: None,
        raising=False,
    )

    def _install(handles):
        namespace = _LLMNamespace(handles)

        class _FakeLMS:
            Chat = _Chat

            @staticmethod
            def Client(server_address):
                return _Client(namespace)

        monkeypatch.setattr(_node, "lms", _FakeLMS)
        # A real unload makes LM Studio stop listing the model; the fake has to
        # do the same or the confirmation poll would never be satisfied.
        for handle in handles:
            original = handle.unload

            def _unload(h=handle, orig=original):
                orig()
                namespace.loaded = [x for x in namespace.loaded if x is not h]

            handle.unload = _unload
        return namespace

    return _install


def _run(**overrides):
    kwargs = dict(
        system_message="sys",
        prompt="hi",
        model_selection=CUSTOM_MODEL_OPTION,
        custom_model_name="prompt-gen",
        max_tokens=16,
        temperature=0.7,
        seed=0,
    )
    kwargs.update(overrides)
    return EALMStudio().generate(**kwargs)


def _troubleshooting(result):
    return result["result"][2]


def test_a_failed_generation_still_frees_the_vram(lm_studio):
    """The regression: an exception skipped the unload entirely, stranding VRAM
    on exactly the runs where the next node is about to ask for it."""
    handle = _ModelHandle("prompt-gen", "pub/repo", fail=RuntimeError("context overflow"))
    namespace = lm_studio([handle])

    result = _run()

    assert "[ERROR] Generation failed" in _troubleshooting(result)
    assert handle.direct_unload_calls == 1
    assert namespace.loaded == []


def test_the_draft_model_is_unloaded_alongside_the_main_model(lm_studio):
    """Speculative decoding loads a second set of weights that also holds VRAM."""
    main = _ModelHandle("prompt-gen", "pub/repo")
    draft = _ModelHandle("draft-0.5b", "pub/draft")
    namespace = lm_studio([main, draft])

    _run(draft_model_selection=CUSTOM_MODEL_OPTION, custom_draft_model="draft-0.5b")

    assert namespace.loaded == []


def test_a_model_served_under_an_identifier_is_matched_by_its_model_key(lm_studio):
    """The node's widget may hold either name; both must reach the same handle."""
    handle = _ModelHandle("prompt-gen", "pub/repo")
    namespace = lm_studio([handle])

    _run(custom_model_name="pub/repo")

    assert namespace.loaded == []


def test_unloading_an_identifier_warns_that_it_will_no_longer_resolve(lm_studio):
    """Once unloaded, a serving identifier is not JIT-loadable - the next run
    would fail with a bare 'model not found' unless the user is told why."""
    lm_studio([_ModelHandle("prompt-gen", "pub/repo")])

    lines = _troubleshooting(_run())

    assert "was a serving identifier, not a model key" in lines
    assert "pub/repo" in lines


def test_keeping_the_model_warm_leaves_it_loaded(lm_studio):
    handle = _ModelHandle("prompt-gen", "pub/repo")
    namespace = lm_studio([handle])

    _run(unload_llm=False)

    assert handle.direct_unload_calls == 0
    assert namespace.loaded == [handle]
