"""Tests for confirmed VRAM unloading (``lms_unload``).

The module is deliberately free of the ``lmstudio`` SDK: every function takes a
duck-typed client/handle, so these tests drive it with plain fakes rather than a
running LM Studio.
"""
import pytest

from lms_unload import list_loaded_llms, model_ids, unload_llm_instances


class _Handle:
    """Stand-in for a loaded-model handle returned by ``client.llm.list_loaded``."""

    def __init__(self, identifier=None, model_key=None, path=None, info=None, fail=False):
        if identifier is not None:
            self.identifier = identifier
        if model_key is not None:
            self.model_key = model_key
        if path is not None:
            self.path = path
        if info is not None:
            self.info = info
        self._fail = fail
        self.unload_calls = 0

    def unload(self):
        self.unload_calls += 1
        if self._fail:
            raise RuntimeError("boom")


class _Info:
    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


class _LLM:
    """``client.llm`` namespace whose ``list_loaded`` walks a script of states."""

    def __init__(self, states, attr="list_loaded"):
        self._states = list(states)
        self.model_calls = []
        setattr(self, attr, self._list_loaded)

    def _list_loaded(self):
        # Hold the final state once the script runs out, so a polling loop that
        # keeps asking sees a stable answer.
        return self._states.pop(0) if len(self._states) > 1 else self._states[0]

    def model(self, identifier):  # pragma: no cover - must never be reached
        self.model_calls.append(identifier)
        raise AssertionError("unload must not JIT-load a model")


class _Client:
    def __init__(self, llm):
        self.llm = llm


# --- model_ids ------------------------------------------------------------

def test_model_ids_reads_both_names_off_the_handle():
    assert model_ids(_Handle(identifier="prompt-gen", model_key="pub/repo")) == (
        "prompt-gen",
        "pub/repo",
    )


def test_model_ids_falls_back_to_path_then_to_info():
    assert model_ids(_Handle(path="pub/repo"))[1] == "pub/repo"
    handle = _Handle(info=_Info(identifier="from-info", model_key="pub/repo"))
    assert model_ids(handle) == ("from-info", "pub/repo")


def test_model_ids_ignores_non_string_attributes():
    """SDK builds differ; a stray object here must not become a match target."""
    assert model_ids(_Handle(identifier=object(), model_key=object())) == (None, None)


# --- list_loaded_llms -----------------------------------------------------

def test_list_loaded_llms_accepts_the_camelcase_spelling():
    handle = _Handle(identifier="a")
    assert list_loaded_llms(_Client(_LLM([[handle]], attr="listLoaded"))) == [handle]


def test_list_loaded_llms_reports_an_sdk_without_either_spelling():
    class _Bare:
        pass

    with pytest.raises(AttributeError):
        list_loaded_llms(_Client(_Bare()))


# --- unload_llm_instances -------------------------------------------------

def test_nothing_is_requested_when_every_target_is_empty():
    llm = _LLM([[]])
    lines = []
    unload_llm_instances(_Client(llm), [None, "", None], lines)
    assert lines == []


def test_a_loaded_instance_is_unloaded_without_being_reloaded_first():
    """The bug: ``client.llm.model(id)`` JIT-loads a model just to unload it."""
    handle = _Handle(identifier="prompt-gen", model_key="pub/repo")
    llm = _LLM([[handle], []])
    lines = []

    unload_llm_instances(_Client(llm), ["prompt-gen"], lines, poll_interval=0)

    assert handle.unload_calls == 1
    assert llm.model_calls == []
    assert any("Unload requested: prompt-gen" in line for line in lines)


def test_a_target_naming_the_model_key_matches_the_loaded_instance():
    handle = _Handle(identifier="prompt-gen", model_key="pub/repo")
    llm = _LLM([[handle], []])

    unload_llm_instances(_Client(llm), ["pub/repo"], [], poll_interval=0)

    assert handle.unload_calls == 1


def test_an_unmatched_target_says_so_instead_of_claiming_an_unload():
    handle = _Handle(identifier="something-else")
    lines = []

    unload_llm_instances(_Client(_LLM([[handle]])), ["pub/repo"], lines, poll_interval=0)

    assert handle.unload_calls == 0
    assert any("Nothing to unload" in line for line in lines)


def test_success_is_only_claimed_once_lm_studio_stops_reporting_the_model():
    """VRAM is freed asynchronously, so the first poll can still see the model."""
    handle = _Handle(identifier="prompt-gen")
    llm = _LLM([[handle], [handle], []])
    lines = []

    unload_llm_instances(_Client(llm), ["prompt-gen"], lines, poll_interval=0)

    assert any("confirmed all requested models unloaded" in line for line in lines)


def test_a_model_that_never_goes_away_warns_instead_of_hanging():
    handle = _Handle(identifier="prompt-gen")
    lines = []

    unload_llm_instances(
        _Client(_LLM([[handle]])), ["prompt-gen"], lines, timeout=0, poll_interval=0
    )

    assert any("Still loaded after" in line for line in lines)
    assert not any("confirmed" in line for line in lines)


def test_one_handle_refusing_to_unload_does_not_strand_the_others():
    bad = _Handle(identifier="bad", fail=True)
    good = _Handle(identifier="good")
    lines = []

    unload_llm_instances(
        _Client(_LLM([[bad, good]])), ["bad", "good"], lines, timeout=0, poll_interval=0
    )

    assert good.unload_calls == 1
    assert any("Failed to unload bad: boom" in line for line in lines)


def test_an_sdk_that_cannot_enumerate_warns_rather_than_raising():
    class _Broken:
        def list_loaded(self):
            raise RuntimeError("socket closed")

    lines = []
    unload_llm_instances(_Client(_Broken()), ["pub/repo"], lines, poll_interval=0)

    assert any("Could not enumerate loaded models: socket closed" in line for line in lines)
