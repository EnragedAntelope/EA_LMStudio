# EA_LMStudio

 The most fully-featured [LM Studio](https://lmstudio.ai/) integration for ComfyUI — auto model discovery, multi-image vision, reasoning extraction, structured JSON output, full sampler control, cancellable streaming, and built-in VRAM management.

![EA LM Studio turning a short idea into a full text-to-image prompt, with the response shown inside the node and inference stats in the troubleshooting output](docs/images/node-text-prompt-enhancer.png)
_Text-only: a prompt enhancer. The response renders inside the node, and `troubleshooting` reports exactly which parameters were applied plus full inference stats._

![EA LM Studio captioning an image with a vision model, constrained to a JSON schema](docs/images/node-vision-structured-json.png)
_Vision: captioning an image with a VLM into schema-constrained JSON that downstream nodes can parse directly._

 This node is ready to be integrated into your workflows in dozens of ways!

## Features

- **Auto Model Discovery** - Models populate automatically from LM Studio at startup
- **Vision Support** - Up to 4 image inputs with smart auto-resize to prevent OOM
- **Reasoning Extraction** - Uses LM Studio's own reasoning split when the model has Reasoning Parsing configured, and falls back to tag detection (DeepSeek R1, Qwen3, QwQ, GLM, GPT-OSS "harmony" channels, and similar) when it does not
- **Structured JSON Output** - Constrain decoding to a JSON Schema so downstream nodes can parse the response
- **Cancellable & Streamed** - ComfyUI's cancel button really stops a runaway generation, and the queue progress bar tracks tokens instead of the node looking frozen
- **In-Node Response Preview** - The generated text appears inside the node; no extra preview node required
- **Advanced Controls** - Temperature, top-k/p, min-p, repetition penalty, stop strings, context-overflow policy, speculative decoding
- **Honest Diagnostics** - Every parameter is checked against the config LM Studio reports back, so a setting that silently does nothing is reported instead of hidden
- **Detailed Stats** - Tokens/sec, time to first token, stop reason, token counts, and speculative-decoding acceptance rates
- **VRAM Management** - Auto-unload after generation (enabled by default), including the draft model, and confirmed against LM Studio rather than assumed — so the next node in the workflow really does see the memory back

## Installation

**Via ComfyUI Manager** (recommended): Search for "EA_LMStudio"

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/EA_LMStudio.git
pip install -r EA_LMStudio/requirements.txt
```

## Requirements

- **LM Studio** with the local server enabled. Validated against **LM Studio 0.4.19**.
- The **`lmstudio`** Python package (installed via `requirements.txt`). Requires Python 3.10+.

> **Keep LM Studio and the `lmstudio` package updated.** The SDK discards prediction parameters it does not recognise instead of raising, so an out-of-date package can turn a setting into a silent no-op. The node now compares what it sent against the config LM Studio echoes back and reports any parameter that was dropped, in the `troubleshooting` output.

## Quick Start

1. Start LM Studio with server enabled (default: `http://127.0.0.1:1234`)
2. Start ComfyUI
3. Find the node: **EA -> LMStudio**

Ready-made workflows live in [`example_workflows/`](example_workflows) — drag one onto the ComfyUI canvas:

| Workflow | What it shows |
|----------|---------------|
| `01-prompt-enhancer.json` | Turning a short idea into a rich t2i prompt, with reasoning and stats previews |
| `02-vision-caption-json.json` | Captioning an image with a VLM into schema-constrained JSON |

## Tips

- **Models not showing?** LM Studio must be running before ComfyUI starts. Toggle the `refresh_models` checkbox to instantly re-fetch and update the dropdowns.
- **Context errors?** Increase context length in LM Studio settings (not max_tokens). Set `context_overflow` to `Stop at limit (error)` if a silently shortened prompt would be worse than no answer.
- **VLM issues?** Try a smaller image resize option or a single image if multi-image fails.
- **Force thinking mode:** Add `/think` to the prompt (Qwen3 family) or "Think step by step" for others. LM Studio's API has no thinking on/off flag — see below.
- **Thinking leaking into the response?** LM Studio splits reasoning from the answer only when the model's **Reasoning Parsing** delimiters are configured. Set them in LM Studio (per model) and the node uses that split directly. Otherwise it falls back to tag detection, and reports it when a tagless `Thinking Process:` preamble appears to have leaked through.
- **Reducing repetition:** Raise `repeat_penalty` (1.1-1.3) and/or use `min_p` (0.05-0.1) as a modern alternative to `top_p`. LM Studio has no presence or frequency penalty.
- **Guaranteed parseable output:** Set `output_format` to `JSON (schema below)` and supply a schema. `JSON (no schema)` only *asks* for JSON — it does not constrain decoding, and many models answer with a ```` ```json ```` fenced block (the node unwraps that automatically when the contents are valid JSON).
- **Speculative decoding not helping?** The troubleshooting output reports the draft-token acceptance rate. Below ~30% the draft model is usually costing more than it saves.

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

A model whose identifier contains characters the node will not accept (LM Studio occasionally serves ids like `some-model@?`) is hidden from the dropdown and named in the `troubleshooting` output, so it is never a silent disappearance.

## Outputs

| Output | Description |
|--------|-------------|
| response | Generated text (reasoning removed if extracted) |
| reasoning | Extracted thinking content |
| troubleshooting | Status messages, debug hints, and inference stats (tokens/sec, input/output/total tokens, time to first token, stop reason, speculative-decoding acceptance, total time) |

The response also renders inside the node itself, so a preview node is optional.

## Upgrading from 1.x

**Version 2.0.0 removed the `presence_penalty` and `enable_thinking` widgets.** Neither ever did anything: the `lmstudio` SDK silently discards prediction-config keys it does not recognise, and LM Studio has no presence penalty and no thinking on/off flag. They were dropped before the request ever left your machine.

You do not need to rebuild anything. When a workflow saved by 1.x is loaded, the node detects the old layout and realigns every remaining setting to the right widget, showing a toast when it does. Without that, removing two mid-list widgets would have shifted every later value by one or two slots.

If you relied on `enable_thinking`, the equivalents that genuinely work are configuring **Reasoning Parsing** for the model in LM Studio, editing the model's Jinja template (`{%- set enable_thinking = false %}`), or a `/think` / `/no_think` marker in the prompt for Qwen3-family models.

## License

[MIT License](LICENSE)

---

*Originally based on [YANC_LMStudio](https://github.com/ALatentPlace/YANC_LMStudio) by A Latent Place*
