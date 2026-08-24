"""
EA LM Studio Node - ComfyUI integration for LM Studio
Provides text generation using local LLM/VLM models via LM Studio server.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import inspect
import os
import threading
import time
from PIL import Image

# LM Studio SDK
import lmstudio as lms

# ComfyUI imports
import comfy.model_management as model_management

try:  # Optional: only used to drive the queue progress bar.
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover - ComfyUI always provides this at runtime
    ProgressBar = None

# Local imports
from .lms_config.config_manager import ConfigManager
from .model_fetcher import (
    auth_headers,
    get_model_choices,
    get_default_model_choice,
    refresh_model_cache,
    initialize_model_cache,
    validate_model_identifier,
    get_last_fetch_error,
    get_last_fetch_success,
    get_cached_model_count,
    get_last_rejected_models,
    CUSTOM_MODEL_OPTION,
)
from .lms_reasoning import (
    extract_reasoning_auto,
    extract_reasoning_custom,
    looks_like_leaked_thinking,
)
from .lms_params import (
    CONTEXT_OVERFLOW_OPTIONS,
    CONTEXT_OVERFLOW_POLICIES,
    MAX_TOKENS_STOP_REASON,
    OUTPUT_FORMAT_OPTIONS,
    build_structured_setting,
    missing_config_keys,
    parse_stop_strings,
    strip_json_code_fence,
)
from .lms_image import convert_image_to_pil
from .lms_unload import model_ids, unload_llm_instances

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

# Fetch models at startup in a daemon thread.
#
# The fetch is a blocking HTTP call; done inline at import it taxed every
# ComfyUI start by up to ~3s connect cap + read timeout whenever LM Studio
# was unreachable-but-not-refusing. A daemon thread keeps startup instant;
# failure is recorded in model_fetcher and surfaced through the
# /ea_lmstudio/models status route and per-run troubleshooting output.
_startup_config = _config_manager.get_config()
threading.Thread(
    target=initialize_model_cache,
    args=(
        _config_manager.get_server_url(_startup_config),
        _config_manager.get_timeout(_startup_config),
    ),
    kwargs={
        "excluded_patterns": _config_manager.get_excluded_patterns(_startup_config),
        "headers": auth_headers(_config_manager.get_api_token(_startup_config)),
    },
    name="EA_LMStudio-startup-model-fetch",
    daemon=True,
).start()

# ComfyUI's cancel button sets an interrupt flag. We poll it while streaming so a
# runaway generation can actually be stopped, then re-raise so the queue reports
# a cancellation rather than a node error. The class is looked up defensively so
# the module still imports under a stubbed ``comfy`` (tests, registry scanners).
_INTERRUPT_EXCEPTION = getattr(model_management, "InterruptProcessingException", None)
_INTERRUPT_EXCEPTIONS: Tuple[type, ...] = (
    (_INTERRUPT_EXCEPTION,) if isinstance(_INTERRUPT_EXCEPTION, type) else ()
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

# Reasoning/thinking extraction (regexes + helpers) lives in lms_reasoning.py
# so the pure text-processing logic stays importable and unit-testable without
# pulling in the lmstudio SDK, ComfyUI, or the startup network fetch.

# ``draftModel`` is excluded from the applied-config check: LM Studio only echoes
# speculative-decoding settings back when the draft model was actually accepted,
# and we have no way to distinguish "not echoed" from "not applied" here. A false
# "this did nothing" warning would be worse than no warning.
_UNVERIFIABLE_CONFIG_KEYS = frozenset({"draftModel"})

# Keep long values (stop strings, JSON schemas) from flooding the config summary.
_CONFIG_SUMMARY_VALUE_LIMIT = 120


def _summarize_config(gen_config: Dict[str, Any]) -> str:
    """Render the generation config as one readable line."""
    parts = []
    for key, value in gen_config.items():
        text = repr(value) if isinstance(value, (str, list, dict)) else str(value)
        if len(text) > _CONFIG_SUMMARY_VALUE_LIMIT:
            text = text[:_CONFIG_SUMMARY_VALUE_LIMIT] + "...(truncated)"
        parts.append(f"{key}={text}")
    return ", ".join(parts)


# Warn once, not once per run, when a token is set but the SDK lacks support.
_api_token_unsupported_warned = False


def _create_client(server_address: str, api_token: str):
    """Build an lmstudio SDK client, attaching the API token when supported.

    Token authentication needs an SDK new enough to accept an ``api_token"
    argument on Client (1.6.0b1 onwards). Older SDKs raise TypeError just by
    receiving it, so support is detected from the signature and the missing
    capability warns once instead of failing every run.
    """
    global _api_token_unsupported_warned
    if not api_token:
        return lms.Client(server_address)
    try:
        supports_token = "api_token" in inspect.signature(lms.Client).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic Client stand-ins
        supports_token = False
    if supports_token:
        return lms.Client(server_address, api_token=api_token)
    if not _api_token_unsupported_warned:
        _api_token_unsupported_warned = True
        logger.warning(
            "EA_LMStudio: api_token is configured but the installed lmstudio SDK "
            "cannot send it (needs lmstudio>=1.6). Model discovery still sends "
            "the token over REST; upgrade the package for authenticated generation."
        )
    return lms.Client(server_address)

class EALMStudio:
    """
    LM Studio integration node for ComfyUI.
    Queries local LM Studio server for text generation with LLM/VLM models.

    Note: model.respond() automatically applies the model's chat template.
    """

    CATEGORY = "EA/LMStudio"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "troubleshooting")
    # OUTPUT_NODE lets the node run as a graph terminal (so it can be queued
    # without wiring its outputs anywhere) AND carries the ``ui`` payload that
    # renders the response inside the node. Both halves matter: without the
    # payload the flag would only cost a wasted inference on disconnected nodes.
    OUTPUT_NODE = True
    FUNCTION = "generate"
    DESCRIPTION = (
        "Generate text with a local LM Studio model. Supports vision models, "
        "reasoning extraction, structured JSON output and VRAM management."
    )

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = get_model_choices()
        # Default to a real model when discovery worked, so a freshly dropped
        # node is runnable instead of landing on the "Custom" sentinel with an
        # empty identifier. The draft dropdown keeps the sentinel: speculative
        # decoding must stay opt-in.
        default_model = get_default_model_choice()

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
                    "tooltip": "Re-roll control only. LM Studio has no inference-time seed, so this does NOT make output reproducible - changing it simply tells ComfyUI the node is dirty so it generates again instead of reusing the cached response. Set control_after_generate to 'randomize' for a fresh answer every queue, or 'fixed' to keep the cached one."
                }),
            },
            # Widget order below is the on-node layout. Grouped most-used first:
            # sampling -> output shaping -> reasoning -> vision -> speculative
            # decoding -> management. Each group's dependent fields follow the
            # control that switches them on, and every dependent field's tooltip
            # names that control, so the layout reads top-to-bottom.
            "optional": {
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
                    "tooltip": "Penalizes tokens that already appeared, scaled by how often (default 1.0 = disabled). Raising (1.1-1.3) reduces repetition/loops; too high can hurt coherence. Below 1.0 encourages repetition. LM Studio has no presence or frequency penalty - this and min_p are the repetition controls it offers."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-P sampling: drop tokens below this fraction of the top token's probability (default 0.0 = disabled). Raising (e.g. 0.05-0.1) = more focused/coherent; lowering toward 0 = more diverse. A modern alternative to top_p."
                }),
                # --- Output shaping ---
                "stop_strings": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Stop generation when any of these strings appears. One per line; blank lines ignored. Leading/trailing spaces are kept, and \\n \\r \\t \\\\ are expanded - so a line of '\\nUser:' stops at a newline followed by 'User:'. Empty = no stop strings. Useful to stop a chatty model running on past the answer."
                }),
                "context_overflow": (CONTEXT_OVERFLOW_OPTIONS, {
                    "default": "Truncate middle",
                    "tooltip": "What LM Studio does when prompt + response exceed the model's context window. 'Truncate middle' (default) silently drops the middle of the conversation. 'Rolling window' drops from the start. 'Stop at limit (error)' fails loudly instead - pick it if a silently shortened prompt would be worse than no answer."
                }),
                "output_format": (OUTPUT_FORMAT_OPTIONS, {
                    "default": "Text",
                    "tooltip": "'Text' = normal prose. 'JSON (schema below)' constrains decoding to the schema in json_schema and is the reliable choice when a downstream node must parse the response. 'JSON (no schema)' only asks for JSON - it does NOT constrain decoding, and many models answer with a ```json fenced block (which is unwrapped automatically when the contents are valid JSON). Structured output and thinking models mix poorly."
                }),
                "json_schema": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "JSON Schema object, used only when output_format is 'JSON (schema below)'. Example: {\"type\": \"object\", \"properties\": {\"caption\": {\"type\": \"string\"}}, \"required\": [\"caption\"]}"
                }),
                # --- Reasoning extraction ---
                "reasoning_mode": (REASONING_MODE_OPTIONS, {
                    "default": "Auto-detect (recommended)",
                    "tooltip": "How to split thinking from the final answer. When LM Studio's own Reasoning Parsing is configured for the model, its tagging is used directly and this setting is not needed. Otherwise Auto-detect handles DeepSeek, Qwen, QwQ, GLM, GPT-OSS and similar tag formats. Models don't always think for simple queries."
                }),
                "custom_open_tag": ("STRING", {
                    "default": "<think>",
                    "tooltip": "Custom opening tag for reasoning extraction. Only used when reasoning_mode is 'Custom tags'."
                }),
                "custom_close_tag": ("STRING", {
                    "default": "</think>",
                    "tooltip": "Custom closing tag for reasoning extraction. Only used when reasoning_mode is 'Custom tags'."
                }),
                # --- Vision (only relevant when an image input is connected) ---
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
                # --- Speculative decoding ---
                "draft_model_selection": (model_choices, {
                    "default": CUSTOM_MODEL_OPTION,
                    "tooltip": "Optional draft model for speculative decoding (faster inference). Must share a tokenizer with the main model. Leave on 'Custom' with an empty box to disable. Acceptance stats are reported in troubleshooting."
                }),
                "custom_draft_model": ("STRING", {
                    "default": "",
                    "tooltip": "Manual draft model identifier. Only used when draft 'Custom' is selected. Leave empty to disable."
                }),
                # --- Management ---
                "unload_llm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the LLM from LM Studio after generation. Recommended to free VRAM for image generation. Turn off to keep the model warm across runs (this also unloads a model you loaded by hand in LM Studio)."
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

        The browser extension resets the toggle right after fetching, so UI
        users never queue with this True; the guarantee exists for headless,
        API-driven workflows that POST refresh_models=true - this is what
        makes such a refresh-bearing run actually execute.
        """
        if kwargs.get("refresh_models", False):
            return float("nan")  # Always different
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output(
        response: str, reasoning: str, troubleshooting_lines: List[str]
    ) -> Dict[str, Any]:
        """Build the node return value.

        Because this is an OUTPUT_NODE, the ``ui`` half is what makes the flag
        worth having: the response is rendered inside the node itself, so the
        common "just enhance my prompt" workflow needs no extra preview node.
        """
        troubleshooting = "\n".join(troubleshooting_lines)
        return {
            "ui": {
                "text": [response],
                "reasoning": [reasoning],
                "troubleshooting": [troubleshooting],
            },
            "result": (response, reasoning, troubleshooting),
        }

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

    @staticmethod
    def _prepare_images(client, pil_images: List[Image.Image]) -> List[Any]:
        """Upload PIL images to LM Studio and return its file handles.

        Each image is written to a temp file, closed, uploaded, then deleted.
        The file is closed before uploading because on Windows a second reader
        of a still-open handle is not guaranteed, and deletion of an open file
        fails outright.
        """
        handles = []
        for pil_img in pil_images:
            temp_path = None
            try:
                with NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                    temp_path = temp.name
                    pil_img.save(temp, format="JPEG", quality=95)
                handles.append(client.files.prepare_image(temp_path))
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError as e:
                        # Best effort only; Windows can hold the file briefly.
                        logger.debug(f"Could not remove temp image {temp_path}: {e}")
        return handles

    @staticmethod
    def _stats_lines(stats, elapsed: float) -> List[str]:
        """Format inference statistics, tolerating fields the backend omits.

        Every count on LlmPredictionStats except stop_reason is Optional, so a
        bare f-string format spec (e.g. ``:.2f`` on None) raises TypeError. That
        used to surface as "Generation failed" *after* a successful generation,
        discarding the text the model had already produced.
        """
        lines = []

        def value(name, default=None):
            return getattr(stats, name, default)

        tokens_per_sec = value("tokens_per_second")
        if tokens_per_sec is not None:
            lines.append(f"[INFO] Tokens per second: {tokens_per_sec:.2f}")

        for label, attr in (
            ("Input tokens", "prompt_tokens_count"),
            ("Output tokens", "predicted_tokens_count"),
            ("Total tokens", "total_tokens_count"),
        ):
            count = value(attr)
            if count is not None:
                lines.append(f"[INFO] {label}: {count}")

        ttft = value("time_to_first_token_sec")
        if ttft is not None:
            lines.append(f"[INFO] Time to first token: {ttft:.3f}s")

        gpu_layers = value("num_gpu_layers")
        if gpu_layers is not None:
            lines.append(f"[INFO] GPU layers: {gpu_layers:g}")

        # Speculative decoding: without acceptance numbers there is no way to
        # tell whether a draft model is helping or just burning time.
        drafted = value("total_draft_tokens_count")
        if drafted:
            accepted = value("accepted_draft_tokens_count") or 0
            rejected = value("rejected_draft_tokens_count") or 0
            rate = accepted / drafted * 100.0
            draft_key = value("used_draft_model_key")
            if draft_key:
                lines.append(f"[INFO] Draft model used: {draft_key}")
            lines.append(
                f"[INFO] Speculative decoding: {accepted}/{drafted} draft tokens accepted "
                f"({rate:.0f}%), {rejected} rejected"
            )
            if rate < 30.0:
                lines.append(
                    "[HINT] Low draft acceptance - this draft model may be slowing "
                    "generation down. Try a smaller/closer-matched draft model or disable it."
                )

        lines.append(f"[INFO] Stop reason: {value('stop_reason', 'unknown')}")
        lines.append(f"[INFO] Total time: {elapsed:.2f}s")
        return lines

    @staticmethod
    def _error_hints(error_str: str) -> List[str]:
        """Actionable hints keyed off the error text (pure, unit-tested)."""
        hints: List[str] = []
        if "timeout" in error_str or "timed out" in error_str:
            hints.append(
                "[HINT] LM Studio sent no data within the sync timeout (~60s idle). "
                "The model may still be loading or the server busy - retry, or check LM Studio's logs."
            )
        if "connection" in error_str or "refused" in error_str:
            hints.append("[HINT] Ensure LM Studio is running with server enabled")
        elif "context" in error_str or "length" in error_str or "2048" in error_str:
            hints.append("[HINT] Context length exceeded. In LM Studio, increase the model's context length setting")
            hints.append("[HINT] Note: maxTokens limits OUTPUT tokens; contextLength limits TOTAL tokens (input + output)")
        elif "schema" in error_str or "json" in error_str:
            hints.append("[HINT] Check json_schema is a valid JSON Schema object, or set output_format back to 'Text'")
        elif "not found" in error_str or "model" in error_str:
            hints.append("[HINT] Check model identifier matches LM Studio exactly")
        elif "image" in error_str or "vision" in error_str or "multi" in error_str:
            hints.append("[HINT] This model may not support images or multiple image inputs. Try with a single image or text-only.")
        return hints

    @staticmethod
    def _finalize_response(
        response,
        response_text: str,
        gen_config: Dict[str, Any],
        elapsed: float,
        structured: Optional[Dict[str, Any]],
        native_reasoning: str,
        plain_content: str,
        reasoning_mode: str,
        custom_open_tag: str,
        custom_close_tag: str,
        troubleshooting_lines: List[str],
    ) -> Tuple[str, str]:
        """Post-generation processing: verify config, log stats, split reasoning.

        Every step here runs AFTER the text is safely in hand, and any
        exception degrades to returning the raw model output instead of
        discarding it (a bug class that has bitten before: a formatting error
        in this block once threw away a successful generation).
        """
        try:
            # Verify the server actually applied what we asked for. The SDK
            # drops unknown keys without a word, so this is the only way a
            # parameter that quietly does nothing becomes visible.
            try:
                applied = response.prediction_config.to_dict()
            except Exception:  # pragma: no cover - defensive
                applied = {}
            if applied:
                ignored = [
                    key for key in missing_config_keys(gen_config, applied)
                    if key not in _UNVERIFIABLE_CONFIG_KEYS
                ]
                if ignored:
                    troubleshooting_lines.append(
                        f"[WARNING] LM Studio did not apply: {', '.join(ignored)} - "
                        "your installed lmstudio package or LM Studio build may be too old for these"
                    )

            troubleshooting_lines.extend(EALMStudio._stats_lines(response.stats, elapsed))

            if structured:
                if response.structured:
                    troubleshooting_lines.append("[INFO] Structured output: valid JSON")
                else:
                    # "JSON (no schema)" does not constrain decoding, so models
                    # commonly answer with a ```json fenced block. Unwrapping it
                    # (only when the contents really parse) is the difference
                    # between a usable output and one no downstream node can read.
                    unfenced, stripped = strip_json_code_fence(response_text)
                    if stripped:
                        response_text = unfenced
                        troubleshooting_lines.append(
                            "[INFO] Structured output: removed a ```json code fence - "
                            "the response is valid JSON underneath"
                        )
                    else:
                        troubleshooting_lines.append(
                            "[WARNING] Structured output requested but the response did not parse as JSON"
                        )
                        if getattr(response.stats, "stop_reason", None) == MAX_TOKENS_STOP_REASON:
                            troubleshooting_lines.append(
                                "[HINT] The response hit max_tokens mid-object, so the JSON is "
                                "truncated. Raise max_tokens."
                            )
                        else:
                            troubleshooting_lines.append(
                                "[HINT] 'JSON (no schema)' asks for JSON but does not constrain "
                                "decoding. Use 'JSON (schema below)' with an explicit schema when "
                                "a downstream node has to parse the response."
                            )

            # Split reasoning from the answer.
            final_response, reasoning = EALMStudio._split_reasoning(
                response_text,
                native_reasoning,
                plain_content,
                reasoning_mode,
                custom_open_tag,
                custom_close_tag,
                troubleshooting_lines,
            )

            if reasoning:
                troubleshooting_lines.append(f"[INFO] Extracted reasoning: {len(reasoning)} chars")
                troubleshooting_lines.append(f"[INFO] Clean response: {len(final_response)} chars")
            return final_response, reasoning
        except Exception as e:
            troubleshooting_lines.append(
                f"[WARNING] Response post-processing failed ({type(e).__name__}: {e}) - "
                "returning the raw model output"
            )
            logger.exception("EA_LMStudio post-processing error")
            return response_text, ""

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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
        stop_strings: str = "",
        context_overflow: str = "Truncate middle",
        output_format: str = "Text",
        json_schema: str = "",
        reasoning_mode: str = "Auto-detect (recommended)",
        custom_open_tag: str = "<think>",
        custom_close_tag: str = "</think>",
        unload_llm: bool = True,
        unload_comfy_models: bool = False,
        refresh_models: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using LM Studio.

        Returns:
            A ComfyUI node result dict carrying both the (response, reasoning,
            troubleshooting) tuple and the ``ui`` payload rendered in the node.
        """
        troubleshooting_lines: List[str] = []

        # Get current config (read the file once and derive everything from it)
        config = _config_manager.get_config()
        server_url = _config_manager.get_server_url(config)
        timeout = _config_manager.get_timeout(config)
        excluded_patterns = _config_manager.get_excluded_patterns(config)
        api_token = _config_manager.get_api_token(config)

        troubleshooting_lines.append(f"[INFO] Server: {server_url}")
        troubleshooting_lines.append(f"[INFO] Cached models: {get_cached_model_count()}")

        # Handle model refresh request
        if refresh_models:
            success, message = refresh_model_cache(
                server_url,
                timeout,
                excluded_patterns=excluded_patterns,
                headers=auth_headers(api_token),
            )
            if success:
                troubleshooting_lines.append(f"[INFO] Model refresh: {message}")
            else:
                troubleshooting_lines.append(f"[WARNING] Model refresh failed: {message}")

        # Check for startup fetch errors
        last_error = get_last_fetch_error()
        if last_error and not get_last_fetch_success():
            troubleshooting_lines.append(f"[WARNING] Startup model fetch: {last_error}")

        # Models LM Studio offered but we could not accept: without this the
        # model just silently isn't in the dropdown and nobody knows why.
        # Informational, not a warning: it is a standing condition the user
        # usually cannot act on, and repeating it as a WARNING every single run
        # would just train people to ignore the warning prefix.
        rejected = get_last_rejected_models()
        if rejected:
            troubleshooting_lines.append(
                f"[INFO] {len(rejected)} model(s) hidden from the dropdown - "
                "LM Studio reported an identifier with unsupported characters: "
                + ", ".join(repr(m) for m in rejected)
            )

        # Resolve main model
        model_identifier, error = self._resolve_model_identifier(
            model_selection, custom_model_name, "model"
        )
        if error:
            troubleshooting_lines.append(f"[ERROR] {error}")
            return self._output("", "", troubleshooting_lines)

        if not model_identifier:
            error_msg = "No model selected. Choose a model from dropdown or enter a custom model name."
            troubleshooting_lines.append(f"[ERROR] {error_msg}")
            return self._output("", "", troubleshooting_lines)

        troubleshooting_lines.append(f"[INFO] Model: {model_identifier}")

        # Resolve draft model (optional)
        draft_model, error = self._resolve_model_identifier(
            draft_model_selection, custom_draft_model, "draft model"
        )
        if error:
            troubleshooting_lines.append(f"[WARNING] Draft model error: {error}")
            draft_model = None
        elif draft_model and draft_model == model_identifier:
            troubleshooting_lines.append(
                "[WARNING] Draft model is the same as the main model - speculative "
                "decoding disabled (it would only load the same weights twice)"
            )
            draft_model = None
        elif draft_model:
            troubleshooting_lines.append(f"[INFO] Draft model: {draft_model}")

        # Structured output must be resolved before any model work: a broken
        # schema should fail instantly, not after loading a model.
        structured, structured_error = build_structured_setting(output_format, json_schema)
        if structured_error:
            troubleshooting_lines.append(f"[ERROR] {structured_error}")
            return self._output("", "", troubleshooting_lines)

        stops = parse_stop_strings(stop_strings)

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
                pil_img = convert_image_to_pil(img_tensor, RESIZE_DIMENSIONS.get(image_resize))
                if pil_img:
                    pil_images.append(pil_img)
                    # Only claim "(resized)" when dimensions actually changed -
                    # images already under the target pass through untouched.
                    orig_w = int(img_tensor.shape[2])
                    orig_h = int(img_tensor.shape[1])
                    dims_changed = (pil_img.size[0], pil_img.size[1]) != (orig_w, orig_h)
                    note = " (resized)" if dims_changed and image_resize != "No Resize" else ""
                    troubleshooting_lines.append(
                        f"[INFO] Image {idx}: {pil_img.size[0]}x{pil_img.size[1]}{note}"
                    )
                else:
                    troubleshooting_lines.append(f"[WARNING] Failed to process image {idx}")

        if pil_images:
            troubleshooting_lines.append(f"[INFO] Total images for VLM: {len(pil_images)}")

        # Build inference request.
        # These are carried out of the ``with`` block: the troubleshooting lines
        # are still being appended to while the model is unloaded, so the node
        # result can only be built once the client is closed.
        result: Optional[Tuple[str, str]] = None
        loaded_identifier: Optional[str] = None
        loaded_key: Optional[str] = None
        try:
            troubleshooting_lines.append("[INFO] Connecting to LM Studio...")

            # Host:port from ConfigManager - the single source of truth.
            server_address = _config_manager.get_server_address(config)

            # Create LM Studio client (attaches the API token when configured
            # and the installed SDK supports it)
            with _create_client(server_address, api_token) as client:
                try:
                    # Load or get model
                    model = client.llm.model(model_identifier)
                    troubleshooting_lines.append(f"[INFO] Model loaded: {model_identifier}")

                    # Record both names now: once unloaded the handle no longer
                    # resolves, and the matcher needs whichever name LM Studio
                    # reports for this instance.
                    loaded_identifier, loaded_key = model_ids(model)

                    # Build chat
                    chat = lms.Chat(system_message)

                    # Add user message (with optional images)
                    if pil_images:
                        chat.add_user_message(prompt, images=self._prepare_images(client, pil_images))
                    else:
                        chat.add_user_message(prompt)

                    # Build generation config.
                    # Keys are validated against LlmPredictionConfigDict in the
                    # lmstudio SDK. Anything not in that type is DISCARDED SILENTLY
                    # by the SDK before the request leaves the machine, so a wrong
                    # key looks like it worked - the applied-config check below is
                    # what makes that visible.
                    gen_config: Dict[str, Any] = {
                        "temperature": temperature,
                        "maxTokens": max_tokens,
                        "contextOverflowPolicy": CONTEXT_OVERFLOW_POLICIES.get(
                            context_overflow, "truncateMiddle"
                        ),
                    }

                    # Add optional parameters only when set away from their disabled
                    # default, so default workflows send nothing extra.
                    if top_p < 1.0:
                        gen_config["topPSampling"] = top_p
                    if top_k > 0:
                        gen_config["topKSampling"] = top_k
                    if repeat_penalty != 1.0:
                        gen_config["repeatPenalty"] = repeat_penalty
                    if min_p > 0.0:
                        gen_config["minPSampling"] = min_p
                    if stops:
                        gen_config["stopStrings"] = stops
                    if structured:
                        gen_config["structured"] = structured
                    if draft_model:
                        gen_config["draftModel"] = draft_model

                    troubleshooting_lines.append(f"[INFO] Config: {_summarize_config(gen_config)}")
                    troubleshooting_lines.append("[INFO] Generating...")

                    start_time = time.time()
                    (
                        response,
                        native_reasoning,
                        plain_content,
                        interrupted,
                        stream_error,
                    ) = self._stream(model, chat, gen_config, max_tokens)
                    elapsed = time.time() - start_time

                    if interrupted:
                        # The queue reports a cancellation; nothing is returned
                        # to downstream nodes either way, so say exactly that.
                        troubleshooting_lines.append(
                            "[WARNING] Cancelled from ComfyUI - generation stopped early"
                        )
                    elif stream_error:
                        troubleshooting_lines.append(
                            f"[ERROR] Generation ended early: {stream_error}"
                        )
                    else:
                        troubleshooting_lines.append("[INFO] Generation complete")

                    if stream_error and response is None:
                        # The stream died mid-flight (LM Studio stalled or the
                        # socket dropped). Return what arrived instead of
                        # discarding it with a bare "Generation failed".
                        troubleshooting_lines.append(
                            "[WARNING] Returning the partial text received before the failure"
                        )
                        fallback_text = native_reasoning + plain_content
                        final_response, reasoning = self._split_reasoning(
                            fallback_text,
                            native_reasoning,
                            plain_content,
                            reasoning_mode,
                            custom_open_tag,
                            custom_close_tag,
                            troubleshooting_lines,
                        )
                        result = (final_response, reasoning)
                    else:
                        response_text = response.content
                        troubleshooting_lines.append(
                            f"[INFO] Raw response length: {len(response_text)} chars"
                        )
                        result = self._finalize_response(
                            response,
                            response_text,
                            gen_config,
                            elapsed,
                            structured,
                            native_reasoning,
                            plain_content,
                            reasoning_mode,
                            custom_open_tag,
                            custom_close_tag,
                            troubleshooting_lines,
                        )

                    if interrupted:
                        # Report the cancellation to ComfyUI from inside the try, so
                        # the finally below still frees the VRAM on the way out.
                        model_management.throw_exception_if_processing_interrupted()

                finally:
                    # Unload in a finally: a generation that failed (context
                    # overflow, dropped connection, an image the model cannot
                    # take) is exactly the run where the next node is about to
                    # ask for that VRAM.
                    if unload_llm:
                        unload_llm_instances(
                            client,
                            [model_identifier, draft_model, loaded_identifier, loaded_key],
                            troubleshooting_lines,
                        )
                        if loaded_key and loaded_identifier and loaded_key != loaded_identifier:
                            troubleshooting_lines.append(
                                f"[HINT] '{loaded_identifier}' was a serving identifier, not a "
                                f"model key. Once unloaded it no longer resolves - set this "
                                f"node's model to '{loaded_key}' so the next run can JIT-load it"
                            )

            if result is None:  # defensive; the try body always sets it
                result = ("", "")
            return self._output(result[0], result[1], troubleshooting_lines)

        except _INTERRUPT_EXCEPTIONS:
            raise  # user cancellation is not a node failure
        except Exception as e:
            error_msg = f"Generation failed: {type(e).__name__}: {e}"
            troubleshooting_lines.append(f"[ERROR] {error_msg}")
            troubleshooting_lines.extend(self._error_hints(str(e).lower()))

            logger.exception("EA_LMStudio generation error")
            return self._output("", "", troubleshooting_lines)

    def _stream(self, model, chat, gen_config: Dict[str, Any], max_tokens: int):
        """Run the prediction as a stream.

        Streaming (rather than a blocking ``respond``) buys three things:
        ComfyUI's cancel button can actually stop a runaway generation, the
        queue progress bar moves instead of the node looking hung, and LM
        Studio's own per-fragment ``reasoning_type`` tagging becomes available -
        which is more reliable than any tag regex when the model has Reasoning
        Parsing configured in LM Studio.

        Returns ``(result, native_reasoning, plain_content, interrupted, error)``
        with ``error`` None on a clean finish. If the stream dies mid-flight -
        the SDK's sync API raises LMStudioTimeoutError after ~60s without any
        server message; sockets can drop too - everything received so far is
        returned alongside the error description instead of being lost.
        """
        pbar = ProgressBar(max_tokens) if ProgressBar is not None else None
        interrupted = False
        error: Optional[str] = None
        reasoning_chunks: List[str] = []
        content_chunks: List[str] = []
        tokens_seen = 0

        stream = model.respond_stream(chat, config=gen_config)
        # Note: the stream must be drained rather than broken out of. Breaking
        # closes the underlying generator and .result() then raises GeneratorExit;
        # cancel() ends it promptly with stop_reason "userStopped" instead.
        try:
            for fragment in stream:
                reasoning_type = getattr(fragment, "reasoning_type", "none")
                if reasoning_type == "reasoning":
                    reasoning_chunks.append(fragment.content)
                elif reasoning_type == "none":
                    content_chunks.append(fragment.content)
                # reasoningStartTag / reasoningEndTag fragments are the delimiters
                # themselves and belong in neither output.

                tokens_seen += getattr(fragment, "tokens_count", 0) or 0
                if pbar is not None:
                    pbar.update_absolute(min(tokens_seen, max_tokens), max_tokens)

                if not interrupted and model_management.processing_interrupted():
                    stream.cancel()
                    interrupted = True
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"EA_LMStudio: prediction stream ended early - {error}")

        result = None if error is not None else stream.result()
        return result, "".join(reasoning_chunks), "".join(content_chunks), interrupted, error

    @staticmethod
    def _split_reasoning(
        response_text: str,
        native_reasoning: str,
        plain_content: str,
        reasoning_mode: str,
        custom_open_tag: str,
        custom_close_tag: str,
        troubleshooting_lines: List[str],
    ) -> Tuple[str, str]:
        """Separate thinking from the final answer.

        LM Studio's own reasoning parser wins when it fired, because it works
        off the model's configured delimiters rather than a guess. The tag
        regexes remain the fallback for the (very common) case of a model whose
        thinking LM Studio was never told how to parse.
        """
        if native_reasoning.strip():
            troubleshooting_lines.append(
                "[INFO] Reasoning separated by LM Studio's own parser (reasoning_type fragments)"
            )
            # plain_content is the same text minus the reasoning; prefer it, but
            # fall back to the full content if the backend sent no plain fragments.
            answer = plain_content.strip() or response_text.strip()
            return answer, native_reasoning.strip()

        if reasoning_mode == "Auto-detect (recommended)":
            final_response, reasoning, detected_pattern = extract_reasoning_auto(response_text)
            if detected_pattern:
                troubleshooting_lines.append(f"[INFO] Auto-detected reasoning format: {detected_pattern}")
            elif looks_like_leaked_thinking(response_text):
                # The model thought, but in a tagless plain-text format that
                # neither LM Studio's parser nor our tag-based extractor caught,
                # so the reasoning leaked into the response output.
                troubleshooting_lines.append("[WARNING] Output looks like tagless thinking that leaked into the response (no <think>-style tags found)")
                troubleshooting_lines.append("[HINT] To fix in LM Studio: set this model's Reasoning Parsing delimiters, or edit its Jinja template to hard-disable thinking ({%- set enable_thinking = false %})")
                troubleshooting_lines.append("[HINT] Or, if the model uses a consistent marker, switch reasoning_mode to 'Custom tags' and set the open/close tags")
            else:
                troubleshooting_lines.append("[INFO] No reasoning tags detected (model may not have used thinking for this query)")
            return final_response, reasoning

        if reasoning_mode == "Custom tags":
            return extract_reasoning_custom(response_text, custom_open_tag, custom_close_tag)

        # "Disabled" - no extraction
        return response_text, ""


# Node registration
NODE_CLASS_MAPPINGS = {
    "EA_LMStudio": EALMStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EA_LMStudio": "EA LM Studio"
}
