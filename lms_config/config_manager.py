"""
Configuration manager for YANC_LMStudio.
Handles server settings with gitignore-protected user config.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("YANC_LMStudio")

DEFAULT_CONFIG: Dict[str, Any] = {
    "server_host": "127.0.0.1",
    "server_port": 1234,
    "timeout_seconds": 5
}


class ConfigManager:
    """Manages YANC_LMStudio configuration with user override support."""

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
        config = DEFAULT_CONFIG.copy()

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

    def get_server_url(self) -> str:
        """
        Get the full server base URL for API calls.

        Returns:
            URL string like "http://127.0.0.1:1234"
        """
        config = self.get_config()
        host = config.get("server_host", "127.0.0.1")
        port = config.get("server_port", 1234)
        return f"http://{host}:{port}"

    def get_timeout(self) -> float:
        """Get configured timeout in seconds."""
        config = self.get_config()
        return float(config.get("timeout_seconds", 5))

    def create_user_config_template(self) -> None:
        """
        Create user config template file if it doesn't exist.
        Called at module initialization.
        """
        if not self.user_config_path.exists():
            template = {
                "_comment": "YANC_LMStudio user configuration. This file is gitignored and survives updates.",
                "_instructions": "Modify values below to override defaults. Delete this file to reset.",
                "server_host": "127.0.0.1",
                "server_port": 1234,
                "timeout_seconds": 5
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
