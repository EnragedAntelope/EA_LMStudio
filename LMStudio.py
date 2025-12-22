"""
YANC LM Studio Node - ComfyUI integration for LM Studio
Provides text generation using local LLM/VLM models via LM Studio server.
"""
import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io
import base64

# LM Studio SDK
import lmstudio as lms

# ComfyUI imports
import comfy.model_management as model_management

# Local imports
from .lms_config.config_manager import ConfigManager
from .model_fetcher import (
    get_model_choices,
    refresh_model_cache,
    initialize_model_cache,
    validate_model_identifier,
    get_last_fetch_error,
    get_last_fetch_success,
    get_cached_model_count,
    CUSTOM_MODEL_OPTION,
)

# Setup logging
logger = logging.getLogger("YANC_LMStudio")

# Initialize configuration and model cache at module load
_config_manager = ConfigManager()
_config_manager.create_user_config_template()
_config_manager.ensure_default_config_exists()

# Attempt to fetch models at startup
_startup_server_url = _config_manager.get_server_url()
_startup_timeout = _config_manager.get_timeout()
initialize_model_cache(_startup_server_url, _startup_timeout)


class YANCLMStudio:
    """
    LM Studio integration node for ComfyUI.
    Queries local LM Studio server for text generation with LLM/VLM models.
    """

    CATEGORY = "YANC/LMStudio"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "troubleshooting")
    OUTPUT_NODE = True
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = get_model_choices()
        default_model = model_choices[0] if model_choices else CUSTOM_MODEL_OPTION

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The input prompt to send to the LLM. This is your main request or question."
                }),
                "model_selection": (model_choices, {
                    "default": default_model,
                    "tooltip": "Select a model from LM Studio. Models are fetched at ComfyUI startup. Select 'Custom' to manually enter a model identifier."
                }),
                "custom_model_name": ("STRING", {
                    "default": "",
                    "tooltip": "Manual model identifier. Only used when 'Custom' is selected above. Find identifiers in LM Studio's model list or server logs."
                }),
                "system_message": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "System prompt that defines the LLM's role and behavior. Sets the context for all responses."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower (0.1-0.3) = focused/deterministic. Higher (0.7-1.0) = creative/varied. Default: 0.7"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 131072,
                    "step": 64,
                    "tooltip": "Maximum tokens in the response. Longer responses need more tokens. 1 token ~ 4 characters."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible outputs. Use same seed + settings for identical results. 0 = random."
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision-enabled models (VLMs like LLaVA, Qwen-VL). Ignored by text-only models."
                }),
                "draft_model_selection": (model_choices, {
                    "default": default_model,
                    "tooltip": "Optional draft model for speculative decoding. Select 'Custom' and leave custom field empty to disable."
                }),
                "custom_draft_model": ("STRING", {
                    "default": "",
                    "tooltip": "Manual draft model identifier. Only used when draft 'Custom' is selected. Leave empty to disable speculative decoding."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold. Lower values (0.1-0.9) = more focused. 1.0 = disabled. Use with temperature. Default: 1.0"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalizes repeated tokens. Higher values (1.1-1.3) reduce repetition. 1.0 = disabled. Default: 1.0"
                }),
                "reasoning_tag": ("STRING", {
                    "default": "<think>",
                    "tooltip": "Opening tag to identify reasoning sections (e.g., '<think>' for DeepSeek R1). Reasoning is extracted to separate output."
                }),
                "unload_llm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the LLM from LM Studio after generation. Frees VRAM but adds reload time on next use."
                }),
                "unload_comfy_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload ComfyUI models (SD, VAE, etc.) before LLM inference. Frees VRAM for larger LLMs."
                }),
                "refresh_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-fetch model list from LM Studio server. Enable this, queue once, then disable. Updates the dropdown for next use."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> str:
        """Force re-execution when refresh_models is True."""
        if kwargs.get("refresh_models", False):
            return str(float("nan"))  # Always different
        return ""

    def _resolve_model_identifier(
        self,
        selection: str,
        custom_name: str,
        field_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve model selection to an identifier.

        Returns:
            Tuple of (model_identifier, error_message)
            model_identifier is None if error, error_message is None if success
        """
        if selection == CUSTOM_MODEL_OPTION:
            if not custom_name or not custom_name.strip():
                return None, None  # No custom model specified (valid for optional draft model)

            model_id = custom_name.strip()
        else:
            model_id = selection

        # Validate
        is_valid, error = validate_model_identifier(model_id)
        if not is_valid:
            return None, f"Invalid {field_name}: {error}"

        return model_id, None

    def _convert_image_to_base64(self, image_tensor) -> Optional[str]:
        """Convert ComfyUI image tensor to base64 string for vision models."""
        try:
            # ComfyUI images are [B, H, W, C] float tensors in 0-1 range
            if image_tensor is None:
                return None

            # Take first image if batch
            if len(image_tensor.shape) == 4:
                img_array = image_tensor[0].cpu().numpy()
            else:
                img_array = image_tensor.cpu().numpy()

            # Convert to uint8
            img_array = (img_array * 255).astype(np.uint8)

            # Create PIL Image
            pil_image = Image.fromarray(img_array)

            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return base64_string

        except Exception as e:
            logger.error(f"Failed to convert image: {e}")
            return None

    def _extract_reasoning(self, text: str, opening_tag: str) -> Tuple[str, str]:
        """
        Extract reasoning content from response.

        Args:
            text: Full response text
            opening_tag: Opening tag like "<think>"

        Returns:
            Tuple of (response_without_reasoning, reasoning_content)
        """
        if not opening_tag or opening_tag not in text:
            return text, ""

        # Derive closing tag
        if opening_tag.startswith("<") and opening_tag.endswith(">"):
            tag_name = opening_tag[1:-1]
            closing_tag = f"</{tag_name}>"
        else:
            # Non-XML style tag - just use same tag as closer
            closing_tag = opening_tag

        reasoning_parts = []
        response_text = text

        # Extract all reasoning blocks
        while opening_tag in response_text:
            start_idx = response_text.find(opening_tag)
            end_idx = response_text.find(closing_tag, start_idx)

            if end_idx == -1:
                # No closing tag - take rest as reasoning
                reasoning_parts.append(response_text[start_idx + len(opening_tag):])
                response_text = response_text[:start_idx]
                break

            # Extract reasoning content
            reasoning_content = response_text[start_idx + len(opening_tag):end_idx]
            reasoning_parts.append(reasoning_content)

            # Remove from response
            response_text = response_text[:start_idx] + response_text[end_idx + len(closing_tag):]

        return response_text.strip(), "\n---\n".join(reasoning_parts).strip()

    def generate(
        self,
        prompt: str,
        model_selection: str,
        custom_model_name: str,
        system_message: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        image=None,
        draft_model_selection: str = CUSTOM_MODEL_OPTION,
        custom_draft_model: str = "",
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        reasoning_tag: str = "<think>",
        unload_llm: bool = False,
        unload_comfy_models: bool = False,
        refresh_models: bool = False
    ) -> Tuple[str, str, str]:
        """
        Generate text using LM Studio.

        Returns:
            Tuple of (response_text, reasoning_text, troubleshooting_info)
        """
        troubleshooting_lines = []

        # Get current config
        config = _config_manager.get_config()
        server_url = _config_manager.get_server_url()
        timeout = _config_manager.get_timeout()

        troubleshooting_lines.append(f"[INFO] Server: {server_url}")
        troubleshooting_lines.append(f"[INFO] Cached models: {get_cached_model_count()}")

        # Handle model refresh request
        if refresh_models:
            success, message = refresh_model_cache(server_url, timeout)
            if success:
                troubleshooting_lines.append(f"[INFO] Model refresh: {message}")
                troubleshooting_lines.append("[INFO] Dropdown will update on next node load")
            else:
                troubleshooting_lines.append(f"[WARNING] Model refresh failed: {message}")

        # Check for startup fetch errors
        last_error = get_last_fetch_error()
        if last_error and not get_last_fetch_success():
            troubleshooting_lines.append(f"[WARNING] Startup model fetch: {last_error}")

        # Resolve main model
        model_identifier, error = self._resolve_model_identifier(
            model_selection, custom_model_name, "model"
        )
        if error:
            troubleshooting_lines.append(f"[ERROR] {error}")
            return "", "", "\n".join(troubleshooting_lines)

        if not model_identifier:
            error_msg = "No model selected. Choose a model from dropdown or enter a custom model name."
            troubleshooting_lines.append(f"[ERROR] {error_msg}")
            return "", "", "\n".join(troubleshooting_lines)

        troubleshooting_lines.append(f"[INFO] Model: {model_identifier}")

        # Resolve draft model (optional)
        draft_model, error = self._resolve_model_identifier(
            draft_model_selection, custom_draft_model, "draft model"
        )
        if error:
            troubleshooting_lines.append(f"[WARNING] Draft model error: {error}")
            draft_model = None
        elif draft_model:
            troubleshooting_lines.append(f"[INFO] Draft model: {draft_model}")

        # Unload ComfyUI models if requested
        if unload_comfy_models:
            troubleshooting_lines.append("[INFO] Unloading ComfyUI models...")
            model_management.unload_all_models()
            model_management.soft_empty_cache()

        # Prepare image for vision models
        image_base64 = None
        if image is not None:
            image_base64 = self._convert_image_to_base64(image)
            if image_base64:
                troubleshooting_lines.append("[INFO] Image prepared for vision model")
            else:
                troubleshooting_lines.append("[WARNING] Failed to process image input")

        # Build inference request
        try:
            troubleshooting_lines.append("[INFO] Connecting to LM Studio...")

            # Parse host and port from config
            host = config.get("server_host", "127.0.0.1")
            port = config.get("server_port", 1234)

            # Create LM Studio client
            with lms.Client(host=host, port=port) as client:
                # Load or get model
                model = client.llm.model(model_identifier)
                troubleshooting_lines.append(f"[INFO] Model loaded: {model_identifier}")

                # Build chat
                chat = lms.Chat(system_message)

                # Add user message (with optional image)
                if image_base64:
                    # For vision models, include image
                    chat.add_user_message([
                        {"type": "image", "image": image_base64},
                        {"type": "text", "text": prompt}
                    ])
                else:
                    chat.add_user_message(prompt)

                # Build generation config
                gen_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }

                # Add optional parameters if not at default
                if top_p < 1.0:
                    gen_config["topP"] = top_p
                if repeat_penalty != 1.0:
                    gen_config["repeatPenalty"] = repeat_penalty
                if seed > 0:
                    gen_config["seed"] = seed
                if draft_model:
                    gen_config["draftModel"] = draft_model

                troubleshooting_lines.append(f"[INFO] Generating (temp={temperature}, max_tokens={max_tokens})...")

                # Generate response
                response = model.respond(chat, config=gen_config)
                response_text = str(response)

                troubleshooting_lines.append("[INFO] Generation complete")

                # Extract reasoning if tag specified
                final_response, reasoning = self._extract_reasoning(response_text, reasoning_tag)
                if reasoning:
                    troubleshooting_lines.append(f"[INFO] Extracted reasoning ({len(reasoning)} chars)")

                # Unload LLM if requested
                if unload_llm:
                    try:
                        model.unload()
                        troubleshooting_lines.append("[INFO] LLM unloaded from LM Studio")
                    except Exception as e:
                        troubleshooting_lines.append(f"[WARNING] Failed to unload LLM: {e}")

                return final_response, reasoning, "\n".join(troubleshooting_lines)

        except Exception as e:
            error_msg = f"Generation failed: {type(e).__name__}: {e}"
            troubleshooting_lines.append(f"[ERROR] {error_msg}")

            # Provide hints based on error type
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                troubleshooting_lines.append("[HINT] Ensure LM Studio is running with server enabled")
            elif "not found" in error_str or "model" in error_str:
                troubleshooting_lines.append("[HINT] Check model identifier matches LM Studio exactly")

            logger.exception("YANC_LMStudio generation error")
            return "", "", "\n".join(troubleshooting_lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "YANC_LMStudio": YANCLMStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YANC_LMStudio": "YANC LM Studio"
}
