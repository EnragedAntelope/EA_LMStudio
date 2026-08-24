"""Guard the JS/Python constant duplication flagged as a sync hazard.

``CUSTOM_MODEL_OPTION`` exists verbatim in both ``model_fetcher.py`` (source of
truth) and ``web/ea_lmstudio.js``. Nothing else links them, so this test fails
loudly if they drift - the failure mode (dropdown value not recognised by the
backend) would otherwise only surface in a user's browser.
"""
import re
from pathlib import Path

import model_fetcher

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_js_custom_model_option_matches_python():
    js = (REPO_ROOT / "web" / "ea_lmstudio.js").read_text(encoding="utf-8")
    match = re.search(r'const CUSTOM_MODEL_OPTION = "([^"]+)"', js)
    assert match, "CUSTOM_MODEL_OPTION literal not found in ea_lmstudio.js"
    assert match.group(1) == model_fetcher.CUSTOM_MODEL_OPTION


def test_js_refresh_route_path_matches_backend():
    """The frontend POSTs a route registered in __init__.py; keep them locked."""
    js = (REPO_ROOT / "web" / "ea_lmstudio.js").read_text(encoding="utf-8")
    init_py = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    for route in re.findall(r'api\.fetchApi\("(/ea_lmstudio/[^"]+)"', js):
        assert f'"{route}"' in init_py, f"route {route} used by JS but not registered"
