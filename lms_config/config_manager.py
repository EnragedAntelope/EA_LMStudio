"""
Configuration manager for EA_LMStudio.
Handles server settings with gitignore-protected user config.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional


logger = logging.getLogger("EA_LMStudio")

DEFAULT_CONFIG: Dict[str, Any] = {
    "server_host": "127.0.0.1",
    "server_port": 1234,
    "timeout_seconds": 5,
    "excluded_model_patterns": ["embedding"],
    # Never surfaced as a node widget: workflow JSON is plaintext and gets
    # shared, so a secret there would leak with every posted .json file.
    "api_token": "",
}


class ConfigManager:
    """Manages EA_LMStudio configuration with user override support."""

    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.default_config_path = self.config_dir / "default_config.json"
        self.user_config_path = self.config_dir / "user_config.json"

    def get_config(self) -> Dict[str, Any]:
        """
        Load configuration with user overrides applied to defaults.

        Returns:
            Dict containing merged configuration values.
        """
        # Copy lists as well as the dict itself: a shallow .copy() aliases
        # excluded_model_patterns across callers, so one mutation would leak
        # into every later get_config() result.
        config = {
            key: list(value) if isinstance(value, list) else value
            for key, value in DEFAULT_CONFIG.items()
        }

        # Load user overrides if they exist
        if self.user_config_path.exists():
            try:
                with open(self.user_config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Only update with valid keys (ignore unknown keys and comments)
                    for key in DEFAULT_CONFIG.keys():
                        if key in user_config:
                            config[key] = user_config[key]
                logger.debug(f"Loaded user config from {self.user_config_path}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in user_config.json: {e}. Using defaults.")
            except IOError as e:
                logger.warning(f"Could not read user_config.json: {e}. Using defaults.")

        return config

    @staticmethod
    def _sanitize_host(host: Any) -> str:
        """Accept pasted URLs and bare hosts alike; return a bare hostname."""
        text = str(host or "").strip()
        if "://" in text:
            # Users routinely paste "http://192.168.1.50" into server_host;
            # stripping the scheme beats a confusing connection failure.
            text = text.split("://", 1)[1]
        return text.rstrip("/")

    @staticmethod
    def _coerce_port(port: Any) -> int:
        """Coerce the configured port, falling back to 1234 with a warning."""
        try:
            value = int(port)
        except (TypeError, ValueError):
            logger.warning(
                f"server_port in user_config.json is not a number ({port!r}). Using 1234."
            )
            return 1234
        if not 1 <= value <= 65535:
            logger.warning(f"server_port {value} is out of range. Using 1234.")
            return 1234
        return value

    def get_server_url(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the full server base URL for API calls.

        Args:
            config: Optional pre-loaded config dict (from get_config()) to
                avoid re-reading the config file. Loaded fresh if omitted.

        Returns:
            URL string like "http://127.0.0.1:1234"
        """
        if config is None:
            config = self.get_config()
        host = self._sanitize_host(config.get("server_host", "127.0.0.1")) or "127.0.0.1"
        port = self._coerce_port(config.get("server_port", 1234))
        return f"http://{host}:{port}"

    def get_server_address(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Host:port for the lmstudio SDK client - the single source of truth.

        Args:
            config: Optional pre-loaded config dict to avoid a re-read.
        """
        if config is None:
            config = self.get_config()
        host = self._sanitize_host(config.get("server_host", "127.0.0.1")) or "127.0.0.1"
        port = self._coerce_port(config.get("server_port", 1234))
        return f"{host}:{port}"

    def get_api_token(self, config: Optional[Dict[str, Any]] = None) -> str:
        """API token for LM Studio servers that require authentication.

        Sources, in order: ``api_token`` in user_config.json, then the
        ``LM_API_TOKEN`` environment variable (the SDK's own convention).
        There is deliberately no node widget for this: workflows are shared
        as plaintext JSON, so the token only ever lives in this gitignored
        file or the environment.
        """
        if config is None:
            config = self.get_config()
        token = str(config.get("api_token") or "").strip()
        if token:
            return token
        return os.environ.get("LM_API_TOKEN", "").strip()

    def get_timeout(self, config: Optional[Dict[str, Any]] = None) -> float:
        """Get configured timeout in seconds, tolerating junk values.

        A non-numeric timeout_seconds used to raise ValueError at module
        import (the startup fetch reads it), failing the entire node pack
        load with a traceback. Warn and fall back instead.

        Args:
            config: Optional pre-loaded config dict to avoid a re-read.
        """
        if config is None:
            config = self.get_config()
        raw = config.get("timeout_seconds", 5)
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"timeout_seconds in user_config.json is not a number ({raw!r}). Using default 5."
            )
            return 5.0

    def get_excluded_patterns(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get model exclusion patterns from config.

        Args:
            config: Optional pre-loaded config dict to avoid a re-read.

        Returns a merged list: default patterns + user-specified additions.
        The "embedding" pattern is always included and cannot be removed.
        """
        if config is None:
            config = self.get_config()
        val = config.get("excluded_model_patterns")

        # Start with defaults
        result: List[str] = list(DEFAULT_CONFIG["excluded_model_patterns"])

        # Merge user patterns (type guard + dedup)
        if isinstance(val, list):
            for p in val:
                if isinstance(p, str) and p not in result:
                    result.append(p)
        elif val is not None:
            logger.warning(
                "excluded_model_patterns in user_config.json is not a list. "
                "Using defaults + any valid entries."
            )

        return result

    def create_user_config_template(self) -> None:
        """
        Create user config template file if it doesn't exist.
        Called at module initialization.
        """
        if not self.user_config_path.exists():
            template = {
                "_comment": "EA_LMStudio user configuration. This file is gitignored and survives updates.",
                "_instructions": "Modify values below to override defaults. Delete this file to reset.",
                "server_host": "127.0.0.1",
                "server_port": 1234,
                "timeout_seconds": 5,
                "excluded_model_patterns": ["embedding"],
                "api_token": "",
            }
            try:
                with open(self.user_config_path, 'w', encoding='utf-8') as f:
                    json.dump(template, f, indent=2)
                logger.info(f"Created user config template at {self.user_config_path}")
            except IOError as e:
                logger.warning(f"Could not create user_config.json template: {e}")

    def ensure_default_config_exists(self) -> None:
        """Create default config file if missing (for reference)."""
        if not self.default_config_path.exists():
            try:
                reference = {
                    "_comment": "Default configuration reference. Do not edit. Create user_config.json to override.",
                    **DEFAULT_CONFIG
                }
                with open(self.default_config_path, 'w', encoding='utf-8') as f:
                    json.dump(reference, f, indent=2)
            except IOError:
                pass  # Non-critical
