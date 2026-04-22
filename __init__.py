"""
EA_LMStudio - LM Studio integration for ComfyUI
Provides automatic model discovery and text/vision generation.
Many helpful optimizations and features!
"""
from .LMStudio import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serve web extension for frontend model refresh
WEB_DIRECTORY = "./web"

# Register server API route for model refresh
from aiohttp import web
from server import PromptServer

from .model_fetcher import refresh_model_cache, get_model_choices, CUSTOM_MODEL_OPTION
from .lms_config.config_manager import ConfigManager

@PromptServer.instance.routes.post("/ea_lmstudio/refresh_models")
async def _refresh_models_route(request):
    """API endpoint to refresh model list from LM Studio server."""
    config_manager = ConfigManager()
    server_url = config_manager.get_server_url()
    timeout = config_manager.get_timeout()
    excluded_patterns = config_manager.get_excluded_patterns()

    success, message = refresh_model_cache(server_url, timeout, excluded_patterns=excluded_patterns)
    choices = get_model_choices()
    models = [m for m in choices if m != CUSTOM_MODEL_OPTION]

    return web.json_response({
        "success": success,
        "message": message,
        "models": models,
    })

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
