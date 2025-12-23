"""
Model fetcher for EA_LMStudio.
Queries LM Studio server for available models via /v1/models endpoint.
"""
import re
import requests
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("EA_LMStudio")

# Module-level cache for models
_cached_models: List[str] = []
_last_fetch_error: Optional[str] = None
_last_fetch_success: bool = False

# Constants
CUSTOM_MODEL_OPTION = "-- Custom (enter below) --"

# Patterns to exclude from model list (embedding models, etc.)
EXCLUDED_MODEL_PATTERNS = ("embedding",)


def validate_model_identifier(model_id: str) -> Tuple[bool, str]:
    """
    Validate model identifier for safety.

    Args:
        model_id: The model identifier string to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not model_id or not model_id.strip():
        return False, "Model identifier cannot be empty"

    model_id = model_id.strip()

    # Check for path traversal attempts
    if ".." in model_id:
        return False, "Model identifier contains invalid path sequence"

    # Check reasonable length
    if len(model_id) > 256:
        return False, "Model identifier exceeds maximum length (256 characters)"

    # Allow alphanumeric, hyphens, underscores, dots, colons, at signs, forward slashes
    # These are common in model names like "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    # or "qwen2.5-7b@q4_k_m"
    if not re.match(r'^[\w\-.:@/]+$', model_id):
        return False, "Model identifier contains disallowed characters"

    return True, ""


def fetch_models_from_server(server_url: str, timeout: float = 5.0) -> Tuple[List[str], Optional[str]]:
    """
    Fetch available models from LM Studio server.

    Args:
        server_url: Base URL of LM Studio server (e.g., http://127.0.0.1:1234)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (model_list, error_message)
        - model_list: List of model IDs (excludes embedding models), empty list on failure
        - error_message: None on success, descriptive error on failure
    """
    models: List[str] = []
    error: Optional[str] = None

    endpoint = f"{server_url.rstrip('/')}/v1/models"

    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        if "data" not in data:
            error = "Unexpected response format from LM Studio (missing 'data' field)"
            logger.warning(f"EA_LMStudio: {error}")
            return models, error

        for model in data["data"]:
            model_id = model.get("id", "")

            if not model_id:
                continue

            # Exclude embedding models by checking id pattern
            model_id_lower = model_id.lower()
            is_excluded = any(pattern in model_id_lower for pattern in EXCLUDED_MODEL_PATTERNS)

            if not is_excluded:
                # Validate the model ID before adding
                is_valid, _ = validate_model_identifier(model_id)
                if is_valid:
                    models.append(model_id)

        # Sort alphabetically for easier navigation
        models.sort(key=str.lower)

        logger.info(f"EA_LMStudio: Fetched {len(models)} models from {server_url}")

    except requests.exceptions.ConnectionError:
        error = f"Cannot connect to LM Studio at {server_url}. Ensure LM Studio is running with server enabled."
        logger.warning(f"EA_LMStudio: {error}")
    except requests.exceptions.Timeout:
        error = f"Connection to LM Studio timed out ({timeout}s). Server may be busy or unreachable."
        logger.warning(f"EA_LMStudio: {error}")
    except requests.exceptions.HTTPError as e:
        error = f"LM Studio returned HTTP error: {e.response.status_code}"
        logger.warning(f"EA_LMStudio: {error}")
    except requests.exceptions.JSONDecodeError:
        error = "Invalid JSON response from LM Studio"
        logger.warning(f"EA_LMStudio: {error}")
    except Exception as e:
        error = f"Unexpected error fetching models: {type(e).__name__}: {str(e)}"
        logger.error(f"EA_LMStudio: {error}")

    return models, error


def get_model_choices() -> List[str]:
    """
    Get model choices for dropdown widget.

    Returns:
        List with Custom option first, followed by cached models.
    """
    global _cached_models

    choices = [CUSTOM_MODEL_OPTION]

    if _cached_models:
        choices.extend(_cached_models)

    return choices


def refresh_model_cache(server_url: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Refresh the cached model list from server.

    Args:
        server_url: Base URL of LM Studio server
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    global _cached_models, _last_fetch_error, _last_fetch_success

    models, error = fetch_models_from_server(server_url, timeout)

    if error:
        _last_fetch_error = error
        _last_fetch_success = False
        return False, error

    _cached_models = models
    _last_fetch_error = None
    _last_fetch_success = True

    if models:
        return True, f"Successfully loaded {len(models)} models from LM Studio"
    else:
        return True, "Connected to LM Studio but no models found (embedding models are excluded)"


def initialize_model_cache(server_url: str, timeout: float = 5.0) -> None:
    """
    Initialize model cache at startup. Silent failure - just logs warning.

    Args:
        server_url: Base URL of LM Studio server
        timeout: Request timeout in seconds
    """
    success, message = refresh_model_cache(server_url, timeout)
    if not success:
        logger.warning(f"EA_LMStudio startup: {message}")
        logger.warning("EA_LMStudio: Models will need to be entered manually or refreshed later")


def get_last_fetch_error() -> Optional[str]:
    """Get the last error that occurred during model fetching, or None if last fetch succeeded."""
    return _last_fetch_error


def get_last_fetch_success() -> bool:
    """Return True if the last fetch attempt was successful."""
    return _last_fetch_success


def get_cached_model_count() -> int:
    """Get the number of currently cached models."""
    return len(_cached_models)
