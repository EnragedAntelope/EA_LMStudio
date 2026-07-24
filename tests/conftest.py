"""Pytest configuration for EA_LMStudio.

The tested modules (``lms_reasoning``, ``lms_image``, ``model_fetcher``,
``lms_config``) live at the repository root and are intentionally free of heavy
dependencies. Tests import them directly.

The repo root also contains the ComfyUI node package's ``__init__.py``, so pytest
treats the root as a package and imports it during collection — which pulls in
the ``lmstudio`` SDK and ``comfy`` (neither installed in a plain test/CI env).
To keep the suite dependency-light we register minimal stand-ins for those
modules *only when they are not actually importable*, so a real install is never
shadowed.
"""
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _REPO_ROOT)


def _ensure_stub(name: str) -> None:
    """Register an empty module under ``name`` if it can't be imported."""
    try:
        __import__(name)
        return
    except Exception:
        pass
    parts = name.split(".")
    for i in range(len(parts)):
        mod_name = ".".join(parts[: i + 1])
        if mod_name not in sys.modules:
            module = types.ModuleType(mod_name)
            sys.modules[mod_name] = module
            if i > 0:
                setattr(sys.modules[".".join(parts[:i])], parts[i], module)


# These are only referenced inside function bodies at runtime, never at import
# time, so bare module objects are enough to let the package import succeed.
_ensure_stub("lmstudio")
_ensure_stub("comfy.model_management")
