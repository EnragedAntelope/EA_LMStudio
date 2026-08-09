# EA_LMStudio — agent notes

A single ComfyUI custom node (`EA_LMStudio`, displayed as **EA LM Studio**) that
runs text and vision generation against a local [LM Studio](https://lmstudio.ai/)
server through the official `lmstudio` Python SDK. It returns three STRING
outputs — `response`, `reasoning`, `troubleshooting` — and renders the response
inside the node.

## Current state

_Last verified: 2026-08-09_

- **Status:** v2.0.0, published to the Comfy Registry from `main` on every
  `pyproject.toml` version change.
- **Works:** model discovery + refresh (startup fetch and a live API route),
  text generation, multi-image VLM input, reasoning extraction (LM Studio's own
  split, with tag-regex fallback), structured JSON output, stop strings,
  context-overflow policy, speculative decoding with acceptance stats,
  cancellable streaming with a queue progress bar, VRAM unload of both the LLM
  and ComfyUI's own models, and migration of workflows saved by 1.x.
- **In progress:** nothing outstanding.
- **Known gaps:** LM Studio's `ttl`, tool/function calling (`.act()`), GBNF
  grammars, and load-time `seed` (real determinism, but it only applies when the
  node actually loads the model) are all supported by the SDK and not exposed.
  There is no automated frontend test — `web/ea_lmstudio.js` is verified by hand
  in a browser.
- **Deep docs:** user-facing behaviour lives in `README.md`; nothing is
  duplicated here.

## Build / test / run

```bash
pip install -r requirements-dev.txt   # pulls requirements.txt too
pytest -q                              # 100+ tests, no LM Studio needed
```

The suite is dependency-light by design. `tests/conftest.py` stubs `lmstudio`
and `comfy.model_management` **only when they are genuinely missing**, so a real
install is never shadowed, and registers the repo root as a synthetic package so
`LMStudio.py` (which uses relative imports) is importable without the checkout
having to be named `EA_LMStudio`.

CI runs the same command on Python 3.10 and 3.13 (`.github/workflows/test.yml`).

To exercise the node for real, point a scratch script at a running LM Studio and
stub `comfy.model_management` / `comfy.utils` — the node only needs
`unload_all_models`, `soft_empty_cache`, `processing_interrupted`,
`throw_exception_if_processing_interrupted`, `InterruptProcessingException` and
`ProgressBar`.

## Layout

| File | Responsibility |
|------|----------------|
| `LMStudio.py` | The node: INPUT_TYPES, streaming, diagnostics, reasoning split |
| `lms_params.py` | Pure widget-string → SDK-value helpers (stop strings, schema, fences) |
| `lms_reasoning.py` | Tag/harmony reasoning regexes (fallback path) |
| `lms_image.py` | ComfyUI IMAGE tensor → JPEG-safe PIL |
| `model_fetcher.py` | `/v1/models` discovery, validation, cache |
| `lms_config/` | `default_config.json` + gitignored `user_config.json` |
| `web/ea_lmstudio.js` | Refresh toggle, in-node preview, 1.x workflow migration |
| `example_workflows/` | Shipped examples, loadable by drag-and-drop |

Everything except `LMStudio.py` is deliberately free of the `lmstudio` SDK and
`comfy` imports so it stays unit-testable.

## Things that will bite you

**The SDK silently discards unknown prediction-config keys.** `LlmPredictionConfig`
is built with msgspec and drops anything not in `LlmPredictionConfigDict` rather
than raising, so a wrong or wished-for key becomes a no-op that looks like it
worked. v1.x shipped `presencePenalty` and `enableThinking` this way for
releases. Before adding a parameter, check it exists in `LlmPredictionConfigDict`
in the installed SDK, then confirm it round-trips in
`PredictionResult.prediction_config`. `generate()` performs that diff on every
run and warns — do not remove it.

**LM Studio has no presence penalty, no frequency penalty, no inference-time
seed, and no thinking on/off flag.** `seed` exists only in the *load* config.
The `seed` widget is a ComfyUI cache-buster and nothing more.

**Removing or reordering a widget corrupts saved workflows.** ComfyUI serialises
`widgets_values` positionally, so a removal shifts every later value. The
migration in `web/ea_lmstudio.js` keys the old array by the v1.5.x widget-name
order (two variants, with and without the `control_after_generate` widget
ComfyUI inserts after an INT named `seed`). Any future removal needs the same
treatment, and the legacy order table must be kept.

**A streamed prediction must be drained, not broken out of.** Breaking the `for`
loop closes the generator and `stream.result()` then raises `GeneratorExit`.
Call `stream.cancel()` and keep iterating; it ends promptly with
`stop_reason == "userStopped"`.

**Every `LlmPredictionStats` field except `stop_reason` is Optional.** Formatting
one with `:.2f` without a None check raised `TypeError` *after* a successful
generation, which the outer handler then reported as a failure — throwing away
text the model had already produced.

**`{"type": "json"}` without a schema does not constrain decoding.** Models
routinely answer with a ```` ```json ```` fence. Only `jsonSchema` constrains the
sampler.

## Conventions

- Registry publishing is driven by the `version` in `pyproject.toml`; the
  workflow compares it against `HEAD^` and skips when unchanged. Bump it in the
  same commit as any change worth shipping.
- `lms_config/user_config.json` and `worklogs/` are gitignored. Never commit
  either, and never put a server address or token in a tracked file.
- Keep the `CUSTOM_MODEL_OPTION` literal in `web/ea_lmstudio.js` in sync with
  `model_fetcher.py`, which is the source of truth.
