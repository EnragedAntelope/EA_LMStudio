"""
YANC LM Studio Node - ComfyUI integration for LM Studio
Provides text generation using local LLM/VLM models via LM Studio server.
"""
import logging
import re
from typing import Optional, Tuple, List
from tempfile import NamedTemporaryFile
import numpy as np
from PIL import Image

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

# Image resize options
IMAGE_RESIZE_OPTIONS = [
    "No Resize",
    "Low (512px)",
    "Medium (768px)",
    "High (1024px)",
    "Ultra (1536px)",
]

# Mapping of resize option to max dimension
RESIZE_DIMENSIONS = {
    "No Resize": None,
    "Low (512px)": 512,
    "Medium (768px)": 768,
    "High (1024px)": 1024,
    "Ultra (1536px)": 1536,
}

# Reasoning extraction modes
REASONING_MODE_OPTIONS = [
    "Auto-detect (recommended)",
    "Disabled",
    "Custom tags",
]

# Common reasoning tag patterns used by different models
# Order matters - most common first for efficiency
COMMON_REASONING_PATTERNS = [
    # DeepSeek R1, Qwen3, QwQ, GLM-4/Z1 - most common
    (r"<think>(.*?)</think>", "<think>", "</think>"),
    # Alternative spelling
    (r"<thinking>(.*?)</thinking>", "<thinking>", "</thinking>"),
    # Some models use this
    (r"<reasoning>(.*?)</reasoning>", "<reasoning>", "</reasoning>"),
    # Occasionally seen
    (r"<reason>(.*?)</reason>", "<reason>", "</reason>"),
]

# GPT-OSS uses a channel-based format: analysis channel for reasoning, final channel for response
GPT_OSS_ANALYSIS_PATTERN = r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
GPT_OSS_FINAL_PATTERN = r"<\|channel\|>final<\|message\|>(.*?)$"


class YANCLMStudio:
    """
    LM Studio integration node for ComfyUI.
    Queries local LM Studio server for text generation with LLM/VLM models.

    Note: model.respond() automatically applies the model's chat template.
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
                # --- Prompts first ---
                "system_message": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "System prompt that defines the LLM's role and behavior. Sets the context for all responses."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The user prompt to send to the LLM. This is your main request or question."
                }),
                # --- Model selection ---
                "model_selection": (model_choices, {
                    "default": default_model,
                    "tooltip": "Select a model from LM Studio. Models are fetched at ComfyUI startup. Select 'Custom' to manually enter a model identifier."
                }),
                "custom_model_name": ("STRING", {
                    "default": "",
                    "tooltip": "Manual model identifier. Only used when 'Custom' is selected above. Find identifiers in LM Studio's model list."
                }),
                # --- Generation parameters ---
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 131072,
                    "step": 1,
                    "tooltip": "Maximum OUTPUT tokens for the response. This limits reply length, not input. The model's context window (input+output) is set in LM Studio when loading. Context must exceed max_tokens for full output."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower (0.1-0.3) = focused/deterministic. Higher (0.7-1.0) = creative/varied."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for ComfyUI workflow reproducibility. Note: LM Studio SDK does not support inference-time seeding."
                }),
            },
            "optional": {
                # --- Image inputs (for VLMs) ---
                "image_resize": (IMAGE_RESIZE_OPTIONS, {
                    "default": "Medium (768px)",
                    "tooltip": "Resize images before processing. Smaller = faster inference. 'No Resize' keeps original size. Only applies when images are connected."
                }),
                "image1": ("IMAGE", {
                    "tooltip": "First image input for vision models (VLMs). Leave unconnected for text-only inference."
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Second image input for multi-image VLMs. Not all VLMs support multiple images."
                }),
                "image3": ("IMAGE", {
                    "tooltip": "Third image input for multi-image VLMs. Not all VLMs support multiple images."
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Fourth image input for multi-image VLMs. Not all VLMs support multiple images."
                }),
                # --- Advanced model options ---
                "draft_model_selection": (model_choices, {
                    "default": default_model,
                    "tooltip": "Optional draft model for speculative decoding (faster inference). Select 'Custom' and leave empty to disable."
                }),
                "custom_draft_model": ("STRING", {
                    "default": "",
                    "tooltip": "Manual draft model identifier. Only used when draft 'Custom' is selected. Leave empty to disable."
                }),
                # --- Sampling parameters ---
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling: only consider tokens with cumulative probability >= top_p. Lower = more focused. 1.0 = disabled."
                }),
                "top_k": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Top-K sampling: only consider the K most likely tokens. Lower = more focused. 0 = disabled. Recommended: 20-40 for thinking models."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalizes repeated tokens. Higher values (1.1-1.3) reduce repetition. 1.0 = disabled."
                }),
                # --- Reasoning extraction ---
                "reasoning_mode": (REASONING_MODE_OPTIONS, {
                    "default": "Auto-detect (recommended)",
                    "tooltip": "How to extract reasoning/thinking from model output. Auto-detect works with DeepSeek, Qwen, QwQ, GLM, GPT-OSS and similar models."
                }),
                "custom_open_tag": ("STRING", {
                    "default": "<think>",
                    "tooltip": "Custom opening tag for reasoning extraction. Only used when reasoning_mode is 'Custom tags'."
                }),
                "custom_close_tag": ("STRING", {
                    "default": "</think>",
                    "tooltip": "Custom closing tag for reasoning extraction. Only used when reasoning_mode is 'Custom tags'."
                }),
                # --- Management ---
                "unload_llm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the LLM from LM Studio after generation. Recommended to free VRAM for image generation."
                }),
                "unload_comfy_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload ComfyUI models (SD, VAE, etc.) before LLM inference. Frees VRAM for larger LLMs."
                }),
                "refresh_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-fetch model list from LM Studio server. Enable, queue once, then disable. Updates dropdown on next node load."
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

    def _resize_image(self, pil_image: Image.Image, max_dimension: Optional[int]) -> Image.Image:
        """
        Resize image to fit within max_dimension while preserving aspect ratio.

        Args:
            pil_image: PIL Image to resize
            max_dimension: Maximum size for longest edge, or None to skip resize

        Returns:
            Resized PIL Image (or original if no resize needed)
        """
        if max_dimension is None:
            return pil_image

        width, height = pil_image.size
        max_current = max(width, height)

        # Only resize if image is larger than target
        if max_current <= max_dimension:
            return pil_image

        # Calculate new dimensions preserving aspect ratio
        scale = max_dimension / max_current
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Use LANCZOS for high-quality downscaling
        return pil_image.resize((new_width, new_height), Image.LANCZOS)

    def _convert_image_to_pil(self, image_tensor, resize_option: str = "No Resize") -> Optional[Image.Image]:
        """
        Convert ComfyUI image tensor to PIL Image, optionally resizing.

        Args:
            image_tensor: ComfyUI image tensor
            resize_option: Resize option from IMAGE_RESIZE_OPTIONS

        Returns:
            PIL Image or None if conversion fails
        """
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

            # Apply resize if specified
            max_dim = RESIZE_DIMENSIONS.get(resize_option)
            if max_dim is not None:
                pil_image = self._resize_image(pil_image, max_dim)

            return pil_image

        except Exception as e:
            logger.error(f"Failed to convert image: {e}")
            return None

    def _extract_reasoning_auto(self, text: str) -> Tuple[str, str, Optional[str]]:
        """
        Auto-detect and extract reasoning using common patterns.

        Args:
            text: Full response text

        Returns:
            Tuple of (response_without_reasoning, reasoning_content, detected_pattern)
            detected_pattern is None if no pattern matched
        """
        # Check for GPT-OSS channel-based format first
        # Format: <|channel|>analysis<|message|>...<|end|>...<|channel|>final<|message|>...
        analysis_match = re.search(GPT_OSS_ANALYSIS_PATTERN, text, re.DOTALL)
        if analysis_match:
            reasoning = analysis_match.group(1).strip()
            # Try to extract the final response
            final_match = re.search(GPT_OSS_FINAL_PATTERN, text, re.DOTALL)
            if final_match:
                response = final_match.group(1).strip()
            else:
                # Fallback: remove analysis section and any remaining markers
                response = re.sub(GPT_OSS_ANALYSIS_PATTERN, "", text, flags=re.DOTALL)
                # Clean up any remaining channel markers
                response = re.sub(r"<\|start\|>assistant", "", response)
                response = re.sub(r"<\|channel\|>final<\|message\|>", "", response)
                response = re.sub(r"<\|end\|>", "", response)
                response = response.strip()
            return response, reasoning, "<|channel|>analysis"

        # Check standard tag-based patterns
        for pattern, open_tag, close_tag in COMMON_REASONING_PATTERNS:
            # Use DOTALL to match across newlines
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                reasoning_parts = [m.group(1) for m in matches]
                # Remove all matched reasoning blocks from text
                clean_text = re.sub(pattern, "", text, flags=re.DOTALL)
                return clean_text.strip(), "\n---\n".join(reasoning_parts).strip(), open_tag

        # No pattern matched
        return text, "", None

    def _extract_reasoning_custom(self, text: str, open_tag: str, close_tag: str) -> Tuple[str, str]:
        """
        Extract reasoning using custom tags.

        Args:
            text: Full response text
            open_tag: Opening tag
            close_tag: Closing tag

        Returns:
            Tuple of (response_without_reasoning, reasoning_content)
        """
        if not open_tag or open_tag not in text:
            return text, ""

        reasoning_parts = []
        response_text = text

        # Extract all reasoning blocks
        while open_tag in response_text:
            start_idx = response_text.find(open_tag)
            end_idx = response_text.find(close_tag, start_idx + len(open_tag))

            if end_idx == -1:
                # No closing tag - take rest as reasoning
                reasoning_parts.append(response_text[start_idx + len(open_tag):])
                response_text = response_text[:start_idx]
                break

            # Extract reasoning content
            reasoning_content = response_text[start_idx + len(open_tag):end_idx]
            reasoning_parts.append(reasoning_content)

            # Remove from response
            response_text = response_text[:start_idx] + response_text[end_idx + len(close_tag):]

        return response_text.strip(), "\n---\n".join(reasoning_parts).strip()

    def generate(
        self,
        system_message: str,
        prompt: str,
        model_selection: str,
        custom_model_name: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        image_resize: str = "Medium (768px)",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        draft_model_selection: str = CUSTOM_MODEL_OPTION,
        custom_draft_model: str = "",
        top_p: float = 1.0,
        top_k: int = 0,
        repeat_penalty: float = 1.0,
        reasoning_mode: str = "Auto-detect (recommended)",
        custom_open_tag: str = "<think>",
        custom_close_tag: str = "</think>",
        unload_llm: bool = True,
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

        # Collect and process images
        image_inputs = [image1, image2, image3, image4]
        pil_images: List[Image.Image] = []

        for idx, img_tensor in enumerate(image_inputs, start=1):
            if img_tensor is not None:
                pil_img = self._convert_image_to_pil(img_tensor, image_resize)
                if pil_img:
                    pil_images.append(pil_img)
                    if image_resize != "No Resize":
                        troubleshooting_lines.append(f"[INFO] Image {idx}: {pil_img.size[0]}x{pil_img.size[1]} (resized)")
                    else:
                        troubleshooting_lines.append(f"[INFO] Image {idx}: {pil_img.size[0]}x{pil_img.size[1]}")
                else:
                    troubleshooting_lines.append(f"[WARNING] Failed to process image {idx}")

        if pil_images:
            troubleshooting_lines.append(f"[INFO] Total images for VLM: {len(pil_images)}")

        # Build inference request
        try:
            troubleshooting_lines.append("[INFO] Connecting to LM Studio...")

            # Parse host and port from config
            host = config.get("server_host", "127.0.0.1")
            port = config.get("server_port", 1234)
            server_address = f"{host}:{port}"

            # Create LM Studio client
            with lms.Client(server_address) as client:
                # Load or get model
                model = client.llm.model(model_identifier)
                troubleshooting_lines.append(f"[INFO] Model loaded: {model_identifier}")

                # Build chat
                chat = lms.Chat(system_message)

                # Add user message (with optional images)
                if pil_images:
                    # Prepare all images for the SDK
                    image_handles = []
                    for pil_img in pil_images:
                        with NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                            pil_img.save(temp, format="JPEG", quality=95)
                            temp.flush()
                            image_handle = client.files.prepare_image(temp.name)
                            image_handles.append(image_handle)

                    chat.add_user_message(prompt, images=image_handles)
                else:
                    chat.add_user_message(prompt)

                # Build generation config
                gen_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    # Handle context overflow by truncating middle of conversation
                    "contextOverflowPolicy": "truncateMiddle",
                }

                # Add optional parameters if not at default
                # Note: Parameter names per LM Studio SDK docs
                if top_p < 1.0:
                    gen_config["topPSampling"] = top_p
                if top_k > 0:
                    gen_config["topKSampling"] = top_k
                if repeat_penalty != 1.0:
                    gen_config["repeatPenalty"] = repeat_penalty
                # Note: seed is not a valid inference-time parameter in LM Studio SDK
                if draft_model:
                    gen_config["draftModel"] = draft_model

                troubleshooting_lines.append(f"[INFO] Config: maxTokens={max_tokens}, temp={temperature}")
                if top_k > 0:
                    troubleshooting_lines.append(f"[INFO] Sampling: top_k={top_k}, top_p={top_p}")
                troubleshooting_lines.append("[INFO] Generating...")

                # Generate response
                response = model.respond(chat, config=gen_config)
                response_text = str(response)

                troubleshooting_lines.append("[INFO] Generation complete")
                troubleshooting_lines.append(f"[INFO] Raw response length: {len(response_text)} chars")

                # Extract reasoning based on mode
                final_response = response_text
                reasoning = ""

                if reasoning_mode == "Auto-detect (recommended)":
                    final_response, reasoning, detected_pattern = self._extract_reasoning_auto(response_text)
                    if detected_pattern:
                        troubleshooting_lines.append(f"[INFO] Auto-detected reasoning format: {detected_pattern}")
                    elif response_text != final_response:
                        troubleshooting_lines.append("[INFO] Reasoning extracted")
                elif reasoning_mode == "Custom tags":
                    final_response, reasoning = self._extract_reasoning_custom(
                        response_text, custom_open_tag, custom_close_tag
                    )
                # else: "Disabled" - no extraction

                if reasoning:
                    troubleshooting_lines.append(f"[INFO] Extracted reasoning: {len(reasoning)} chars")
                    troubleshooting_lines.append(f"[INFO] Clean response: {len(final_response)} chars")

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
            elif "context" in error_str or "length" in error_str or "2048" in error_str:
                troubleshooting_lines.append("[HINT] Context length exceeded. In LM Studio, increase the model's context length setting")
                troubleshooting_lines.append("[HINT] Note: maxTokens limits OUTPUT tokens; contextLength limits TOTAL tokens (input + output)")
            elif "not found" in error_str or "model" in error_str:
                troubleshooting_lines.append("[HINT] Check model identifier matches LM Studio exactly")
            elif "image" in error_str or "vision" in error_str or "multi" in error_str:
                troubleshooting_lines.append("[HINT] This model may not support images or multiple image inputs. Try with a single image or text-only.")

            logger.exception("YANC_LMStudio generation error")
            return "", "", "\n".join(troubleshooting_lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "YANC_LMStudio": YANCLMStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YANC_LMStudio": "YANC LM Studio"
}
