"""Unloading LM Studio models and confirming the VRAM is actually free.

Kept free of the ``lmstudio`` SDK (every entry point takes a duck-typed client
or handle) so it stays unit-testable, in line with the rest of the non-node
modules.

Two things make this more involved than calling ``handle.unload()``:

* A loaded model answers to two different names. The *instance identifier* is
  what it was served with (``lms load --identifier prompt-gen``); the *model
  key* is the repo path (``publisher/repo``). Which one a node's ``model``
  widget holds depends on how the user loaded it, and the attribute layout
  differs across SDK versions, so both are read defensively.
* LM Studio frees VRAM asynchronously. Returning as soon as the unload is
  *requested* means the next ComfyUI node sizes its allocations against memory
  that has not been released yet - which is the whole reason the node offers to
  unload in the first place.
"""
import time
from typing import List, Optional, Tuple

__all__ = ["model_ids", "list_loaded_llms", "unload_llm_instances"]


def model_ids(handle) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(instance identifier, model key)`` for a loaded handle."""
    ident = getattr(handle, "identifier", None)
    key = getattr(handle, "model_key", None) or getattr(handle, "path", None)
    info = getattr(handle, "info", None)
    if info is not None:
        ident = ident or getattr(info, "identifier", None)
        key = key or getattr(info, "model_key", None) or getattr(info, "path", None)
    return (
        ident if isinstance(ident, str) else None,
        key if isinstance(key, str) else None,
    )


def list_loaded_llms(client) -> List:
    """Return the currently loaded LLM handles, tolerating SDK naming drift."""
    for attr in ("list_loaded", "listLoaded"):
        lister = getattr(client.llm, attr, None)
        if callable(lister):
            return list(lister())
    raise AttributeError("LM Studio SDK exposes no list_loaded() on client.llm")


def unload_llm_instances(
    client,
    wanted: List[Optional[str]],
    lines: List[str],
    timeout: float = 20.0,
    poll_interval: float = 0.25,
) -> None:
    """Unload the named models and wait until LM Studio reports them gone.

    Matching is done against *already loaded* instances rather than by calling
    ``client.llm.model(id)``, which would JIT-load a model just to unload it.

    Diagnostics are appended to ``lines``; nothing here raises, because a failed
    unload must not turn an otherwise successful generation into a node error.
    """
    targets = {name for name in wanted if name}
    if not targets:
        return

    def _still_loaded():
        found = []
        for handle in list_loaded_llms(client):
            ident, key = model_ids(handle)
            if (ident and ident in targets) or (key and key in targets):
                found.append((handle, ident or key))
        return found

    try:
        matched = _still_loaded()
    except Exception as e:
        lines.append(f"[WARNING] Could not enumerate loaded models: {e}")
        return

    if not matched:
        lines.append(
            f"[INFO] Nothing to unload; no loaded instance matched {sorted(targets)}"
        )
        return

    for handle, label in matched:
        try:
            handle.unload()
            lines.append(f"[INFO] Unload requested: {label}")
        except Exception as e:
            lines.append(f"[WARNING] Failed to unload {label}: {e}")

    deadline = time.time() + timeout
    while True:
        try:
            remaining = [label for _, label in _still_loaded()]
        except Exception as e:
            lines.append(f"[WARNING] Could not verify unload: {e}")
            return
        if not remaining:
            lines.append("[INFO] LM Studio confirmed all requested models unloaded")
            return
        if time.time() >= deadline:
            lines.append(
                f"[WARNING] Still loaded after {timeout:.0f}s: {remaining} - "
                "VRAM may not be free yet for the next node"
            )
            return
        time.sleep(poll_interval)
