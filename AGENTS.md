# EA_LMStudio — agent notes

A single ComfyUI custom node (`EA_LMStudio`, displayed as **EA LM Studio**) that
runs text and vision generation against a local [LM Studio](https://lmstudio.ai/)
server through the official `lmstudio` Python SDK. It returns three STRING
outputs — `response`, `reasoning`, `troubleshooting` — and renders the response
inside the node.

## Current state

_Last verified: 2026-08-24_

- **Status:** v2.1.0, published to the Comfy Registry from `main` on every
  `pyproject.toml` version change.
- **Works:** model discovery + refresh (background-thread startup fetch, a
  live refresh API route with a same-origin guard, and a GET status route
  the frontend uses to warn when LM Studio was unreachable at startup),
  text generation, multi-image VLM input, reasoning extraction (LM Studio's
  own split, with tag-regex fallback), structured JSON output, stop strings,
  context-overflow policy, speculative decoding with acceptance stats,
  cancellable streaming that preserves partial output if the stream dies,
  confirmed VRAM unload of the LLM, the draft model and ComfyUI's own
  models, silent migration of workflows saved by 1.x, and an optional
  `api_token` for authenticated servers (config file or env; REST always,
  SDK generation needs lmstudio>=1.6).
- **In progress:** nothing outstanding.
- **Known gaps:** uploaded image/file handles cannot be deleted through the
  SDK (`FileHandle` has no delete in 1.5.0 - LM Studio retains uploads until
  its own cleanup); tool/function calling (`.act()`), GBNF grammars and
  load-time `seed` are supported by the SDK and not exposed; the SDK's load
  config has no `ttl` field in 1.5.0. There is no automated frontend test —
  `web/ea_lmstudio.js` is verified by hand in a browser, though the
  JS/Python `CUSTOM_MODEL_OPTION` sync and route paths ARE unit-tested.
  Vision generation is exercised end-to-end only by hand against a real VLM;
  the suite covers `_prepare_images` (temp file written, uploaded, removed)
  but stops at the SDK boundary.
- **Deep docs:** user-facing behaviour lives in `README.md`; nothing is
  duplicated here.

## Build / test / run

```bash
pip install -r requirements-dev.txt   # pulls requirements.txt too
pyflakes .                             # undefined names / duplicate imports
pytest -q                              # whole suite; no LM Studio needed
```

The suite is dependency-light by design. `tests/conftest.py` stubs `lmstudio`
and `comfy.model_management` **only when they are genuinely missing**, so a real
install is never shadowed, and registers the repo root as a synthetic package so
`LMStudio.py` (which uses relative imports) is importable without the checkout
having to be named `EA_LMStudio`.

CI runs both commands on Python 3.10 and 3.13 (`.github/workflows/test.yml`).
The `pyflakes` step is not decoration: 2.1.0 development deleted
`from tempfile import NamedTemporaryFile` while leaving the call in place, and
the whole suite stayed green because nothing reached the image-upload path.
An undefined name only raises on the branch that uses it — tests alone cannot
be relied on to find that class.

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
| `lms_unload.py` | Match loaded instances, unload, wait for LM Studio to confirm |
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
`widgets_values` positionally, so a removal or a regroup shifts every later
value. The migration in `web/ea_lmstudio.js` keys the old array by the v1.5.x
widget-name order (two variants, with and without the `control_after_generate`
widget ComfyUI inserts after an INT named `seed`) and writes values onto current
widgets **by name**, which is why v2.0.0 could both drop two widgets and regroup
the rest. Any future removal or reorder needs the same treatment, and the legacy
order table must be kept.

**A stored node size is restored verbatim and is not re-checked against the
widgets.** Adding a widget therefore leaves every previously saved workflow too
short, and the overflow draws outside the node frame. `growToFitWidgets` in the
frontend extension grows (never shrinks) the node on configure and after
execution. Note that a stock ComfyUI `Note` node's textarea overhangs its own
frame by ~13 units at any size — that is upstream behaviour, not a symptom of
this, so don't chase it.

**A streamed prediction must be drained, not broken out of.** Breaking the `for`
loop closes the generator and `stream.result()` then raises `GeneratorExit`.
Call `stream.cancel()` and keep iterating; it ends promptly with
`stop_reason == "userStopped"`.

**Every `LlmPredictionStats` field except `stop_reason` is Optional.** Formatting
one with `:.2f` without a None check raised `TypeError` *after* a successful
generation, which the outer handler then reported as a failure — throwing away
text the model had already produced.

**LM Studio frees VRAM asynchronously, and a model answers to two names.**
`handle.unload()` returns as soon as the request is *accepted*, so a node that
returns immediately hands the next node a VRAM figure that is still shrinking —
which defeats the point of the `unload_llm` toggle. `lms_unload` polls
`client.llm.list_loaded()` until the model is gone. That listing is also the
only safe way to find the model: it is loaded under both a serving identifier
(`lms load --identifier`) and a model key (`publisher/repo`), and resolving
either through `client.llm.model(id)` would JIT-load a model purely in order to
unload it. Note that an identifier stops resolving once unloaded, so a workflow
pinned to one cannot JIT-load on its next run — the node warns about this.

**The startup model fetch is a daemon thread, so `INPUT_TYPES` races it.**
`/object_info` is what fills both model dropdowns, and it is served whenever the
browser asks — in principle before the fetch returns, which would render an
empty dropdown against a perfectly healthy LM Studio. Measured 2026-08-24
against a live local server with 14 models: the fetch takes **~5 ms**, while
ComfyUI still has the rest of its custom nodes to load and a server to start, so
the thread always wins. Do not add a join to "fix" this without measuring first
— a join at import time would reinstate exactly the startup stall the thread
removed. If LM Studio is genuinely slow or unreachable the dropdown is empty
*correctly*, and the `/ea_lmstudio/models` status route tells the frontend why.

**A stream failure is not the same as a pre-flight failure, and both need
hints.** `_error_hints` is called from two places: the outer `except` (errors
raised before or around generation) and the `stream_error` branch (the stream
died mid-flight). It also takes `has_images`, because LM Studio's most common
vision failure does not always mention images — sending a picture to a text-only
model can surface as `No engine protocol runtime is registered for '<id>'`,
which reads like an internal fault. Both spellings are covered; the observed
alternative is `image input is not supported ... you may need to provide the
mmproj`.

**`{"type": "json"}` without a schema does not constrain decoding.** Models
routinely answer with a ```` ```json ```` fence. Only `jsonSchema` constrains the
sampler.

## Conventions

- Registry publishing is driven by the `version` in `pyproject.toml`; the
  workflow compares it against `HEAD^` and skips when unchanged. Bump it in the
  same commit as any change worth shipping. **Check the Registry actually
  received it** — `https://api.comfy.org/nodes/EA_LMStudio` reports
  `latest_version`. A green "Publish to Comfy registry" run does not mean a
  publish happened: the version check is a separate step and a skip looks
  identical to a success from the runs list. That is not hypothetical — the
  guard's `sed 's/.*"//'` was greedy, parsed every version to an empty string,
  and silently skipped **every** publish from 2.0.0 through 2.1.0, leaving the
  Registry on 1.5.0 for the whole 2.x line. It now fails loudly instead of
  parsing to nothing.
- To publish a version the push path missed, run the workflow manually
  (`gh workflow run "Publish to Comfy registry" --ref main`). A
  `workflow_dispatch` run publishes unconditionally, because in that situation
  `HEAD^` already carries the same version and the change check can never pass.
- `lms_config/user_config.json` and `worklogs/` are gitignored. Never commit
  either, and never put a server address or token in a tracked file.
- Keep the `CUSTOM_MODEL_OPTION` literal in `web/ea_lmstudio.js` in sync with
  `model_fetcher.py`, which is the source of truth.
