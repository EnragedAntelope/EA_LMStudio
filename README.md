# EA_LMStudio

 The most fully-featured [LM Studio](https://lmstudio.ai/) integration for ComfyUI — auto model discovery, multi-image vision, reasoning extraction, full sampler control, and built-in VRAM management. 

 <img width="1818" height="1285" alt="image" src="https://github.com/user-attachments/assets/2f7529c3-1be8-49e3-935f-db610129d990" />
 _Example of using as a Text-only LLM, with automated reasoning extraction_

<img width="2539" height="1328" alt="image" src="https://github.com/user-attachments/assets/bea0e4dc-5608-4378-a8b5-915ec7763c88" />
 _Example of using as a VLM, with multiple image inputs_

 This node is ready to be integrated into your workflows in dozens of ways!

## Features

- **Auto Model Discovery** - Models populate automatically from LM Studio at startup
- **Vision Support** - Up to 4 image inputs with smart auto-resize to prevent OOM
- **Reasoning Extraction** - Separates thinking from final response (DeepSeek R1, Qwen3, QwQ, etc.)
- **Advanced Controls** - Temperature, top-k/p, min-p, repetition & presence penalties, thinking toggle, speculative decoding
- **Smart Troubleshooting** - Helpful error messages with specific hints
- **Detailed Stats** - Tokens/sec, time to first token, stop reason, and token counts in troubleshooting output
- **VRAM Management** - Auto-unload after generation (enabled by default)

## Installation

**Via ComfyUI Manager** (recommended): Search for "EA_LMStudio"

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/EA_LMStudio.git
pip install -r EA_LMStudio/requirements.txt
```

## Requirements

- **LM Studio** with the local server enabled. Validated against **LM Studio 0.4.17**.
- The **`lmstudio`** Python package (installed via `requirements.txt`).

> **Keep LM Studio and the `lmstudio` package updated.** Newer sampling controls (e.g. min-p, presence penalty, thinking toggle) are applied by the backend only when it supports them — older backends silently ignore parameters they don't recognize rather than erroring, so updating ensures every setting actually takes effect.

## Quick Start

1. Start LM Studio with server enabled (default: `http://127.0.0.1:1234`)
2. Start ComfyUI
3. Find the node: **EA -> LMStudio**

## Tips

- **Models not showing?** LM Studio must be running before ComfyUI starts. Toggle the `refresh_models` checkbox to instantly re-fetch and update the dropdowns.
- **Context errors?** Increase context length in LM Studio settings (not max_tokens).
- **VLM issues?** Try a smaller image resize option or single image if multi-image fails.
- **Force thinking mode:** Set the `enable_thinking` toggle to `Enabled` for hybrid models like Qwen3 (no prompt hacks needed), or add `/think` to prompts / "Think step by step" for others.
- **`enable_thinking` not working?** The toggle sets the `enableThinking` flag, which only takes effect when the model's chat template honors the `enable_thinking` Jinja variable. Many community finetunes/merges (and models that emit a tagless `Thinking Process:` preamble instead of `<think>` tags) ignore it and keep thinking regardless. To actually disable or cleanly separate thinking for those, fix it in LM Studio: either edit the model's Jinja template (`{%- set enable_thinking = false %}`) or configure its **Reasoning Parsing** delimiters so LM Studio knows how to split reasoning from the answer.
- **Reducing repetition:** Raise `repeat_penalty` (1.1-1.3) and/or `presence_penalty` (0.3-0.8). These are distinct controls; LM Studio has no separate frequency penalty. `min_p` (0.05-0.1) is a modern alternative to `top_p` for keeping output coherent.

## Custom Server

Edit `lms_config/user_config.json`:
```json
{
    "server_host": "192.168.1.100",
    "server_port": 1234
}
```

## Model Exclusion Patterns

Exclude models from the dropdown by adding patterns to `lms_config/user_config.json`:
```json
{
    "excluded_model_patterns": ["embedding", "Qwen3-Coder", "codellama"]
}
```

- The default excludes all models containing **"embedding"**.
- Patterns are matched as case-insensitive substrings against the full model identifier.
- Use specific enough patterns to avoid accidentally excluding desired models (e.g., use `Qwen3-Coder` instead of just `coder`).
- To find exact model names, check LM Studio's API: `http://127.0.0.1:1234/v1/models` (look at the `id` field).
- The `user_config.json` file is gitignored so your settings survive updates.
- Restart ComfyUI or toggle the **refresh_models** checkbox after changing patterns.

## Outputs

| Output | Description |
|--------|-------------|
| response | Generated text (reasoning removed if extracted) |
| reasoning | Extracted thinking content |
| troubleshooting | Status messages, debug hints, and inference stats (tokens/sec, input/output tokens, time to first token, stop reason, total time) |

## License

[MIT License](LICENSE)

---

*Originally based on [YANC_LMStudio](https://github.com/ALatentPlace/YANC_LMStudio) by A Latent Place*
