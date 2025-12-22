# YANC_LMStudio

A ComfyUI custom node for integrating LM Studio's local LLM/VLM inference.

## Features

- **Automatic Model Discovery**: Models are fetched from LM Studio at ComfyUI startup
- **Vision Model Support**: Up to 4 image inputs for VLM models (LLaVA, Qwen-VL, etc.)
- **Image Resizing**: Automatic resize options to speed up VLM inference
- **Reasoning Extraction**: Separate thinking/reasoning from final response
- **Speculative Decoding**: Optional draft model support
- **VRAM Management**: Unload models after use (enabled by default)

## Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ALatentPlace/YANC_LMStudio.git
cd YANC_LMStudio
pip install -r requirements.txt
```

Or install via ComfyUI Manager.

## Setup

1. **Start LM Studio** with the server enabled (default: `http://127.0.0.1:1234`)
2. **Start ComfyUI** - models are auto-fetched at startup
3. Find the node under **YANC -> LMStudio**

## Model Selection

### Automatic (Recommended)
Models are fetched from LM Studio's `/v1/models` endpoint when ComfyUI starts. Select from the dropdown.

**Requirement:** LM Studio must be running before starting ComfyUI.

### Manual Entry
If LM Studio wasn't running at startup or you added models later:
1. Select `-- Custom (enter below) --` from dropdown
2. Enter model identifier in `custom_model_name`
3. Find identifiers in LM Studio's model list

### Refresh Models
Enable `refresh_models`, queue once, then disable. Updates take effect on next node load.

## Image Inputs (VLMs)

The node supports up to 4 image inputs for vision-language models:
- **image1-4**: Optional image inputs (connect any combination)
- **image_resize**: Resize images before processing to speed up inference
  - `No Resize` - Use original size
  - `Low (512px)` - Fast processing
  - `Medium (768px)` - Balanced (default)
  - `High (1024px)` - Better detail
  - `Ultra (1536px)` - Maximum detail

**Note:** Not all VLMs support multiple images. If you get errors with multiple images, try using only `image1`.

## Custom Server Address

Default: `http://127.0.0.1:1234`

To change, edit `lms_config/user_config.json`:
```json
{
    "server_host": "192.168.1.100",
    "server_port": 1234,
    "timeout_seconds": 5
}
```
This file survives git updates.

## Parameters

### Required
| Parameter | Default | Description |
|-----------|---------|-------------|
| system_message | "You are a helpful assistant." | System prompt defining LLM behavior |
| prompt | "" | User prompt / question |
| model_selection | - | Model dropdown or "Custom" |
| max_tokens | 1024 | Maximum OUTPUT tokens (see note below) |
| temperature | 0.7 | Randomness (0=deterministic, 1+=creative) |

### Optional
| Parameter | Default | Description |
|-----------|---------|-------------|
| image_resize | Medium (768px) | Resize images before VLM processing |
| top_p | 1.0 | Nucleus sampling (lower=more focused) |
| repeat_penalty | 1.0 | Reduce repetition (1.1-1.3 recommended) |
| unload_llm | True | Unload LLM after generation (recommended) |

### Understanding max_tokens vs Context Length

- **max_tokens**: Limits OUTPUT tokens (how long the response can be)
- **Context Length**: Total tokens for INPUT + OUTPUT combined (set in LM Studio when loading model)

If you see "Reached context length" errors, increase the model's context length in LM Studio's settings, not max_tokens.

## Outputs

- **response**: The generated text
- **reasoning**: Extracted thinking/reasoning (if reasoning_tag found)
- **troubleshooting**: Status messages, errors, and hints

## Troubleshooting

Check the `troubleshooting` output for detailed status information.

**Common Issues:**
- "Cannot connect": Ensure LM Studio server is running
- "Model not found": Verify model identifier matches LM Studio
- "Context length exceeded": Increase context length in LM Studio model settings
- Empty dropdown: Start LM Studio before ComfyUI, or use manual entry
- Multi-image errors: Model may only support single image, try just image1

## License

MIT License
