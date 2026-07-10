"""
EA_LMStudio - LM Studio integration for ComfyUI
Provides automatic model discovery and text/vision generation.
Many helpful optimizations and features!
"""
import logging

from .LMStudio import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serve web extension for frontend model refresh
WEB_DIRECTORY = "./web"

logger = logging.getLogger("EA_LMStudio")


def _register_routes() -> None:
    """Register the model-refresh API route with ComfyUI's PromptServer.

    Kept in a guarded function so the node pack still loads in contexts where
    the ComfyUI server isn't available (registry scanners, headless imports).
    """
    from aiohttp import web
    from server import PromptServer

    from .model_fetcher import refresh_model_cache, get_model_choices, CUSTOM_MODEL_OPTION
    from .lms_config.config_manager import ConfigManager

    @PromptServer.instance.routes.post("/ea_lmstudio/refresh_models")
    async def _refresh_models_route(request):
        """API endpoint to refresh model list from LM Studio server."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        server_url = config_manager.get_server_url(config)
        timeout = config_manager.get_timeout(config)
        excluded_patterns = config_manager.get_excluded_patterns(config)

        success, message = refresh_model_cache(server_url, timeout, excluded_patterns=excluded_patterns)
        choices = get_model_choices()
        models = [m for m in choices if m != CUSTOM_MODEL_OPTION]

        return web.json_response({
            "success": success,
            "message": message,
            "models": models,
        })


try:
    _register_routes()
except Exception as e:
    logger.warning(
        f"EA_LMStudio: could not register web routes ({type(e).__name__}: {e}). "
        "The node still works; the refresh_models toggle will refresh on the next queued run instead of instantly."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
