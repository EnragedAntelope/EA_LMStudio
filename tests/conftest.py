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


# ``LMStudio.py`` is a package module (``from .lms_config import ...``), so it
# cannot be imported as a top-level module the way the dependency-light modules
# are. Register the repo root as a synthetic package instead of relying on the
# checkout directory being named "EA_LMStudio" - a clone into any other folder
# name would otherwise break the suite. ``__init__.py`` is deliberately not
# executed: it registers ComfyUI server routes we have no use for here.
NODE_PACKAGE = "ea_lmstudio_under_test"

if NODE_PACKAGE not in sys.modules:
    _package = types.ModuleType(NODE_PACKAGE)
    _package.__path__ = [_REPO_ROOT]
    sys.modules[NODE_PACKAGE] = _package
