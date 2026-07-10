"""
EA LM Studio Node - ComfyUI integration for LM Studio
Provides text generation using local LLM/VLM models via LM Studio server.
"""
import logging
import re
from typing import Optional, Tuple, List
import os
import time
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
logger = logging.getLogger("EA_LMStudio")


# --- Dependency log noise suppression ---
# The lmstudio SDK and its HTTP/websocket transport emit a lot of INFO-level
# chatter on every request (websocket lifecycle events, "HTTP Request: GET ..."
# lines, etc.). Because ComfyUI configures the root logger at INFO, all of this
# propagates to the console. We quiet it down to WARNING without touching
# ComfyUI's own logging or other custom nodes.
#
# Two sources need handling:
#   1. Named third-party loggers (httpx, httpcore, websockets, urllib3) -> set level.
#   2. The lmstudio SDK, which creates loggers via new_logger(type(self).__name__),
#      i.e. bare class-name loggers (e.g. "SyncRemoteCall") that are direct children
#      of root. Their names are unpredictable, so we instead attach a filter to the
#      root handlers that drops their sub-WARNING records by source file path.
_NOISY_NAMED_LOGGERS = ("httpx", "httpcore", "websockets", "websocket", "urllib3")


class _SuppressDependencyInfo(logging.Filter):
    """Drop sub-WARNING log records originating from the lmstudio SDK package."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True  # always let warnings/errors through
        path = (record.pathname or "").replace(os.sep, "/").replace("\\", "/").lower()
        # Drop INFO/DEBUG records emitted from inside the lmstudio package.
        return "/lmstudio/" not in path


def _quiet_dependency_logs() -> None:
    """Suppress noisy INFO logs from the lmstudio SDK and its HTTP/ws deps.

    Idempotent: safe to call more than once (the filter is added at most once
    per root handler).
    """
    for name in _NOISY_NAMED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, _SuppressDependencyInfo) for f in handler.filters):
            handler.addFilter(_SuppressDependencyInfo())


# ComfyUI configures root logging before loading custom nodes, so its handlers
# already exist by the time this module is imported.
_quiet_dependency_logs()


# Initialize configuration and model cache at module load
_config_manager = ConfigManager()
_config_manager.create_user_config_template()
_config_manager.ensure_default_config_exists()

# Attempt to fetch models at startup (single config read)
_startup_config = _config_manager.get_config()
initialize_model_cache(
    _config_manager.get_server_url(_startup_config),
    _config_manager.get_timeout(_startup_config),
    excluded_patterns=_config_manager.get_excluded_patterns(_startup_config),
)

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

# enable_thinking toggle options. "Model default" leaves the model's own
# behavior untouched (nothing is sent); the other two force thinking on/off
# for hybrid reasoning models (e.g. Qwen3) via the SDK's enableThinking flag.
ENABLE_THINKING_OPTIONS = [
    "Model default",
    "Enabled",
    "Disabled",
]

# Common reasoning tag patterns used by different models
# Order matters - most common first for efficiency
# Precompiled once at import; these run on every generation.
COMMON_REASONING_PATTERNS = [
    # DeepSeek R1, Qwen3, QwQ, GLM-4/Z1 - most common
    (re.compile(r"<think>(.*?)</think>", re.DOTALL), "<think>", "</think>"),
    # Alternative spelling
    (re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL), "<thinking>", "</thinking>"),
    # Some models use this
    (re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL), "<reasoning>", "</reasoning>"),
    # Occasionally seen
    (re.compile(r"<reason>(.*?)</reason>", re.DOTALL), "<reason>", "</reason>"),
]

# GPT-OSS models use OpenAI's "harmony" channel format:
#   <|start|>assistant<|channel|>analysis<|message|>...<|end|>
#   <|start|>assistant<|channel|>final<|message|>...<|return|>
# LM Studio usually parses this itself, but partial leaks are common: a final
# block with its markers still attached, an analysis block missing its <|end|>
# terminator, a trailing <|return|>, or (when special tokens get stripped
# during detokenization) a bare "analysis...assistantfinal..." string.
#
# Analysis/commentary segments end at <|end|> or at the start of the next
# message/channel header, or at end of text if truncated. The commentary
# channel carries preambles/tool chatter, so it is routed to reasoning too.
GPT_OSS_ANALYSIS_RE = re.compile(
    r"<\|channel\|>(?:analysis|commentary)<\|message\|>(.*?)"
    r"(?=<\|end\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)
GPT_OSS_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)\Z", re.DOTALL)
# Sweep for any harmony markers left over after extraction. <|...|> tokens
# never appear in legitimate prose, so removing them is safe.
HARMONY_MARKER_RE = re.compile(
    r"<\|start\|>(?:assistant|user|system|tool)?"
    r"|<\|channel\|>(?:analysis|commentary|final)?"
    r"|<\|message\|>|<\|end\|>|<\|return\|>|<\|constrain\|>"
)
# Detokenized-without-special-tokens variant: "analysisUser wants...assistantfinalAnswer".
# "assistantfinal" is the discriminator; it never occurs in normal prose.
GPT_OSS_STRIPPED_RE = re.compile(r"\s*analysis(.*?)assistantfinal(.*)\Z", re.DOTALL)

# Quick membership test for whether harmony markers are present at all
_HARMONY_TOKENS = ("<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>")

# Plain-text headings that some models (often community finetunes/merges) emit when
# they "think" without wrapping it in tags. We only use these to DETECT likely leaked
# thinking for a better troubleshooting hint -- we never strip them from the response.
LEAKED_THINKING_MARKERS = (
    "thinking process",
    "thought process",
    "reasoning process",
    "let me think",
    "let's think",
    "step 1:",
    "step 1.",
)


def _looks_like_leaked_thinking(text: str) -> bool:
    """Heuristically detect tagless reasoning that leaked into the response.

    Checks only the start of the response (first ~200 chars) so that a marker
    appearing mid-answer in a legitimate reply doesn't trigger a false positive.
    Detection only -- callers must not strip anything based on this.
    """
    if not text:
        return False
    head = text[:200].lower()
    return any(marker in head for marker in LEAKED_THINKING_MARKERS)


class EALMStudio:
    """
    LM Studio integration node for ComfyUI.
    Queries local LM Studio server for text generation with LLM/VLM models.

    Note: model.respond() automatically applies the model's chat template.
    """

    CATEGORY = "EA/LMStudio"
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
                    "tooltip": "Maximum OUTPUT tokens for the response (default 1024). Limits reply length, not input. Raise for longer replies; lower to cap length/speed up. The model's context window (input+output) is set in LM Studio when loading and must exceed max_tokens for full output."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness (default 0.7). Lower (0.1-0.3) = more focused/deterministic; higher (0.7-1.2) = more creative/varied. 0.0 = greedy/most deterministic."
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
                    "tooltip": "Nucleus sampling: only consider tokens within cumulative probability top_p (default 1.0 = disabled). Lowering (e.g. 0.9-0.95) = more focused/coherent; raising toward 1.0 = more diverse."
                }),
                "top_k": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Top-K sampling: only consider the K most likely tokens (default 0 = disabled). Lowering (e.g. 20-40) = more focused; raising = more diverse. Recommended 20-40 for thinking models."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalizes tokens that already appeared, scaled by how often (default 1.0 = disabled). Raising (1.1-1.3) reduces repetition/loops; too high can hurt coherence. Below 1.0 encourages repetition."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-P sampling: drop tokens below this fraction of the top token's probability (default 0.0 = disabled). Raising (e.g. 0.05-0.1) = more focused/coherent; lowering toward 0 = more diverse. A modern alternative to top_p."
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Flat penalty on any token already used, encouraging new topics (default 0.0 = disabled). Raising (e.g. 0.3-0.8) reduces repetition / broadens topics; negative values encourage reuse. Distinct from repeat_penalty. Note: LM Studio has no frequency_penalty."
                }),
                "enable_thinking": (ENABLE_THINKING_OPTIONS, {
                    "default": "Model default",
                    "tooltip": "Force thinking/reasoning on hybrid models like Qwen3 without the '/think' prompt hack (default 'Model default' = leave the model's own behavior untouched). 'Enabled' turns thinking on; 'Disabled' turns it off. Pairs with reasoning_mode. Ignored by models/backends that don't support it."
                }),
                # --- Reasoning extraction ---
                "reasoning_mode": (REASONING_MODE_OPTIONS, {
                    "default": "Auto-detect (recommended)",
                    "tooltip": "How to extract reasoning/thinking from model output. Auto-detect works with DeepSeek, Qwen, QwQ, GLM, GPT-OSS and similar models. Note: Models don't always produce thinking output for simple queries. For Qwen3, add '/think' to your prompt to force thinking mode."
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
                    "tooltip": "Toggle ON to re-fetch the model list from LM Studio and update the dropdowns instantly. Automatically toggles back off."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when refresh_models is True.

        Returns float("nan") (not its string form): NaN != NaN, so ComfyUI
        sees a changed value every run. As a string, "nan" == "nan" and the
        cached result would be reused.
        """
        if kwargs.get("refresh_models", False):
            return float("nan")  # Always different
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

            # Convert to uint8. Clip first: ComfyUI tensors can slightly
            # exceed [0, 1] (VAE decode etc.), and out-of-range values would
            # otherwise wrap around during the uint8 cast (1.02 -> 4).
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

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

    def _extract_reasoning_gpt_oss(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Extract reasoning from GPT-OSS "harmony" channel output, including
        partial leaks where only some markers survived LM Studio's own parsing.

        Args:
            text: Full response text

        Returns:
            Tuple of (response, reasoning, detected_pattern), or None if the
            text contains no harmony markers at all.
        """
        if any(token in text for token in _HARMONY_TOKENS):
            reasoning_parts = [m.group(1).strip() for m in GPT_OSS_ANALYSIS_RE.finditer(text)]
            reasoning_parts = [p for p in reasoning_parts if p]
            final_match = GPT_OSS_FINAL_RE.search(text)

            if final_match:
                response = final_match.group(1)
            else:
                response = GPT_OSS_ANALYSIS_RE.sub("", text)

            # Only stray terminators present (e.g. a bare <|end|> with no
            # channel headers): treat text before the terminator as leaked
            # reasoning, mirroring the missing-open-tag fallback below.
            if not reasoning_parts and not final_match and "<|channel|>" not in text:
                before, sep, after = text.partition("<|end|>")
                if sep and before.strip() and after.strip():
                    return (
                        HARMONY_MARKER_RE.sub("", after).strip(),
                        before.strip(),
                        "<|end|> (stray terminator)",
                    )

            # Sweep any remaining markers (<|return|>, <|start|>assistant, ...)
            response = HARMONY_MARKER_RE.sub("", response).strip()
            return response, "\n---\n".join(reasoning_parts), "<|channel|> (GPT-OSS harmony)"

        # Detokenized variant with special tokens stripped:
        # "analysisUser wants a cat pic.assistantfinalHere is a cat."
        stripped_match = GPT_OSS_STRIPPED_RE.match(text)
        if stripped_match:
            return (
                stripped_match.group(2).strip(),
                stripped_match.group(1).strip(),
                "analysis...assistantfinal (stripped special tokens)",
            )

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
        # Check for GPT-OSS harmony/channel-based format first
        gpt_oss_result = self._extract_reasoning_gpt_oss(text)
        if gpt_oss_result is not None:
            return gpt_oss_result

        # Check standard tag-based patterns
        for pattern, open_tag, close_tag in COMMON_REASONING_PATTERNS:
            matches = list(pattern.finditer(text))
            if matches:
                reasoning_parts = [m.group(1) for m in matches]
                # Remove all matched reasoning blocks from text
                clean_text = pattern.sub("", text)
                return clean_text.strip(), "\n---\n".join(reasoning_parts).strip(), open_tag

        # Fallback: Check for closing tag without opening tag (model bug/edge case)
        # Some models forget the opening <think> but include closing </think>
        for _, open_tag, close_tag in COMMON_REASONING_PATTERNS:
            if close_tag in text and open_tag not in text:
                # Split on closing tag - everything before is reasoning
                parts = text.split(close_tag, 1)
                if len(parts) == 2:
                    reasoning = parts[0].strip()
                    response = parts[1].strip()
                    if reasoning and response:
                        return response, reasoning, f"{close_tag} (missing open tag)"
                    if reasoning and not response:
                        # Model spent its whole budget thinking and never
                        # produced an answer (usually maxTokens truncation).
                        # Surface it as reasoning rather than passing the raw
                        # tagged text through as the response.
                        return "", reasoning, f"{close_tag} (missing open tag, response truncated)"

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
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        enable_thinking: str = "Model default",
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

        # Get current config (read the file once and derive everything from it)
        config = _config_manager.get_config()
        server_url = _config_manager.get_server_url(config)
        timeout = _config_manager.get_timeout(config)
        excluded_patterns = _config_manager.get_excluded_patterns(config)

        troubleshooting_lines.append(f"[INFO] Server: {server_url}")
        troubleshooting_lines.append(f"[INFO] Cached models: {get_cached_model_count()}")

        # Handle model refresh request
        if refresh_models:
            success, message = refresh_model_cache(server_url, timeout, excluded_patterns=excluded_patterns)
            if success:
                troubleshooting_lines.append(f"[INFO] Model refresh: {message}")
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
                if len(img_tensor.shape) == 4 and img_tensor.shape[0] > 1:
                    troubleshooting_lines.append(
                        f"[WARNING] Image {idx}: batch of {img_tensor.shape[0]} received; only the first image is used"
                    )
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
                    temp_paths = []
                    try:
                        for pil_img in pil_images:
                            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                                pil_img.save(temp, format="JPEG", quality=95)
                                temp.flush()
                                temp_paths.append(temp.name)
                                image_handle = client.files.prepare_image(temp.name)
                                image_handles.append(image_handle)
                    finally:
                        for path in temp_paths:
                            try:
                                os.unlink(path)
                            except OSError:
                                pass

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

                # Add optional parameters only when set away from their disabled
                # default, so default workflows send nothing extra (and stay safe
                # on older LM Studio backends, which silently ignore params they
                # don't support rather than erroring).
                # Key names validated against LM Studio 0.4.17 / LLMPredictionConfigInput.
                if top_p < 1.0:
                    gen_config["topPSampling"] = top_p
                if top_k > 0:
                    gen_config["topKSampling"] = top_k
                if repeat_penalty != 1.0:
                    gen_config["repeatPenalty"] = repeat_penalty
                if min_p > 0.0:
                    gen_config["minPSampling"] = min_p
                if presence_penalty != 0.0:
                    gen_config["presencePenalty"] = presence_penalty
                if enable_thinking != "Model default":
                    gen_config["enableThinking"] = (enable_thinking == "Enabled")
                # Note: seed is not a valid inference-time parameter in LM Studio SDK
                if draft_model:
                    gen_config["draftModel"] = draft_model

                # Show every parameter actually being sent, so the user can confirm
                # exactly what is applied. Params left at their disabled default are
                # omitted above and therefore won't appear here.
                config_summary = ", ".join(f"{k}={v}" for k, v in gen_config.items())
                troubleshooting_lines.append(f"[INFO] Config: {config_summary}")
                troubleshooting_lines.append("[INFO] Generating...")

                start_time = time.time()
                # Generate response
                response = model.respond(chat, config=gen_config)
                response_text = str(response)

                troubleshooting_lines.append("[INFO] Generation complete")
                troubleshooting_lines.append(f"[INFO] Raw response length: {len(response_text)} chars")

                # Extract inference statistics
                tokens_per_sec = getattr(response.stats, 'tokens_per_second', 0.0)
                input_tokens = getattr(response.stats, 'prompt_tokens_count', 0)
                output_tokens = getattr(response.stats, 'predicted_tokens_count', 0)
                time_to_first_token = getattr(response.stats, 'time_to_first_token_sec', None)
                stop_reason = getattr(response.stats, 'stop_reason', 'unknown')
                elapsed = time.time() - start_time

                troubleshooting_lines.append(f"[INFO] Tokens per second: {tokens_per_sec:.2f}")
                troubleshooting_lines.append(f"[INFO] Input tokens: {input_tokens}")
                troubleshooting_lines.append(f"[INFO] Output tokens: {output_tokens}")
                if time_to_first_token is not None:
                    troubleshooting_lines.append(f"[INFO] Time to first token: {time_to_first_token:.3f}s")
                troubleshooting_lines.append(f"[INFO] Stop reason: {stop_reason}")
                troubleshooting_lines.append(f"[INFO] Total time: {elapsed:.2f}s")

                # Extract reasoning based on mode
                final_response = response_text
                reasoning = ""

                if reasoning_mode == "Auto-detect (recommended)":
                    final_response, reasoning, detected_pattern = self._extract_reasoning_auto(response_text)
                    if detected_pattern:
                        troubleshooting_lines.append(f"[INFO] Auto-detected reasoning format: {detected_pattern}")
                    elif _looks_like_leaked_thinking(response_text):
                        # The model thought, but in a tagless plain-text format that
                        # neither LM Studio's parser nor our tag-based extractor caught,
                        # so the reasoning leaked into the response output.
                        troubleshooting_lines.append("[WARNING] Output looks like tagless thinking that leaked into the response (no <think>-style tags found)")
                        if enable_thinking == "Disabled":
                            troubleshooting_lines.append("[HINT] This model kept thinking despite enable_thinking=Disabled - its chat template likely ignores the enableThinking flag (common for community merges/finetunes)")
                        troubleshooting_lines.append("[HINT] To fix in LM Studio: set this model's Reasoning Parsing delimiters, or edit its Jinja template to hard-disable thinking ({%- set enable_thinking = false %})")
                        troubleshooting_lines.append("[HINT] Or, if the model uses a consistent marker, switch reasoning_mode to 'Custom tags' and set the open/close tags")
                    else:
                        troubleshooting_lines.append("[INFO] No reasoning tags detected (model may not have used thinking for this query)")
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

            logger.exception("EA_LMStudio generation error")
            return "", "", "\n".join(troubleshooting_lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "EA_LMStudio": EALMStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EA_LMStudio": "EA LM Studio"
}
