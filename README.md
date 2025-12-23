# EA_LMStudio

A ComfyUI custom node for local LLM/VLM inference via [LM Studio](https://lmstudio.ai/).

## Features

- **Auto Model Discovery** - Models populate automatically from LM Studio at startup
- **Vision Support** - Up to 4 image inputs with smart auto-resize to prevent OOM
- **Reasoning Extraction** - Separates thinking from final response (DeepSeek R1, Qwen3, QwQ, etc.)
- **Advanced Controls** - Temperature, top-k/p, repetition penalty, speculative decoding
- **Smart Troubleshooting** - Helpful error messages with specific hints
- **VRAM Management** - Auto-unload after generation (enabled by default)

## Installation

**Via ComfyUI Manager** (recommended): Search for "EA_LMStudio"

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/EA_LMStudio.git
pip install -r EA_LMStudio/requirements.txt
```

## Quick Start

1. Start LM Studio with server enabled (default: `http://127.0.0.1:1234`)
2. Start ComfyUI
3. Find the node: **EA -> LMStudio**

## Tips

- **Models not showing?** LM Studio must be running before ComfyUI starts. Use `refresh_models` checkbox to re-fetch.
- **Context errors?** Increase context length in LM Studio settings (not max_tokens).
- **VLM issues?** Try a smaller image resize option or single image if multi-image fails.
- **Force thinking mode:** Add `/think` to prompts for Qwen3, or "Think step by step" for others.

## Custom Server

Edit `lms_config/user_config.json`:
```json
{
    "server_host": "192.168.1.100",
    "server_port": 1234
}
```

## Outputs

| Output | Description |
|--------|-------------|
| response | Generated text (reasoning removed if extracted) |
| reasoning | Extracted thinking content |
| troubleshooting | Status messages and debug hints |

## License

[MIT License](LICENSE)

---

*Based on [YANC_LMStudio](https://github.com/ALatentPlace/YANC_LMStudio) by A Latent Place*
