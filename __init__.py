"""
EA_LMStudio - LM Studio integration for ComfyUI
Provides automatic model discovery and text/vision generation.
Many helpful optimizations and features!
"""
import asyncio
import logging
from typing import Optional

from .LMStudio import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serve web extension for frontend model refresh
WEB_DIRECTORY = "./web"

logger = logging.getLogger("EA_LMStudio")


def _register_routes() -> None:
    """Register the model-refresh API routes with ComfyUI's PromptServer.

    Kept in a guarded function so the node pack still loads in contexts where
    the ComfyUI server isn't available (registry scanners, headless imports).
    """
    from aiohttp import web
    from server import PromptServer

    from .model_fetcher import (
        CUSTOM_MODEL_OPTION,
        auth_headers,
        get_cached_model_count,
        get_last_fetch_error,
        get_last_fetch_success,
        get_model_choices,
        origin_matches_host,
        refresh_model_cache,
    )
    from .lms_config.config_manager import ConfigManager

    def _reject_cross_origin(request) -> Optional[web.Response]:
        if not origin_matches_host(request.headers.get("Origin"), request.headers.get("Host")):
            return web.json_response(
                {"success": False, "message": "Cross-origin request rejected"}, status=403
            )
        return None

    @PromptServer.instance.routes.post("/ea_lmstudio/refresh_models")
    async def _refresh_models_route(request):
        """API endpoint to refresh model list from LM Studio server."""
        reject = _reject_cross_origin(request)
        if reject is not None:
            return reject

        config_manager = ConfigManager()
        config = config_manager.get_config()
        server_url = config_manager.get_server_url(config)
        timeout = config_manager.get_timeout(config)
        excluded_patterns = config_manager.get_excluded_patterns(config)

        # requests.get blocks; running it inline here would stall ComfyUI's
        # whole event loop (API + websockets) for up to the full timeout.
        success, message = await asyncio.to_thread(
            refresh_model_cache,
            server_url,
            timeout,
            excluded_patterns=excluded_patterns,
            headers=auth_headers(config_manager.get_api_token(config)),
        )
        choices = get_model_choices()
        models = [m for m in choices if m != CUSTOM_MODEL_OPTION]

        return web.json_response({
            "success": success,
            "message": message,
            "models": models,
        })

    @PromptServer.instance.routes.get("/ea_lmstudio/models")
    async def _models_status_route(request):
        """Report the current cached model list and the last fetch outcome.

        Lets the frontend tell a user whose LM Studio was unreachable at
        startup why the dropdown is empty, instead of leaving them with a
        silent "-- Custom --".
        """
        models = [m for m in get_model_choices() if m != CUSTOM_MODEL_OPTION]
        return web.json_response({
            "success": get_last_fetch_success(),
            "error": get_last_fetch_error(),
            "model_count": get_cached_model_count(),
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
