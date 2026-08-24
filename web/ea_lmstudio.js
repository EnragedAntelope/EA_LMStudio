import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

/*
 * EA_LMStudio frontend extension
 *
 * Three jobs:
 *   1. Model refresh  - the refresh_models toggle re-fetches the model list and
 *                       updates both dropdowns in place, with a visible toast.
 *   2. Response preview - renders the generated text inside the node (the node
 *                       is an OUTPUT_NODE and sends a ui payload for this).
 *   3. Legacy migration - v1.x saved presence_penalty and enable_thinking into
 *                       widgets_values. Both were removed in v2.0.0 because the
 *                       LM Studio SDK silently discarded them, so every widget
 *                       after them would otherwise load one or two slots out of
 *                       alignment. See migrateLegacyWidgetValues below.
 */

const NODE_CLASS = "EA_LMStudio";
const CUSTOM_MODEL_OPTION = "-- Custom (enter below) --"; // keep in sync with model_fetcher.py
const PREVIEW_WIDGET_NAME = "response_preview";

/*
 * Widget order as serialised by v1.5.x, in LiteGraph order (required first,
 * then optional; IMAGE inputs are links, not widgets, so they never appear).
 * ComfyUI inserts a control_after_generate widget straight after an INT named
 * "seed" on some frontend versions, hence the two variants.
 */
const LEGACY_ORDER = [
    "system_message",
    "prompt",
    "model_selection",
    "custom_model_name",
    "max_tokens",
    "temperature",
    "seed",
    "image_resize",
    "draft_model_selection",
    "custom_draft_model",
    "top_p",
    "top_k",
    "repeat_penalty",
    "min_p",
    "presence_penalty",
    "enable_thinking",
    "reasoning_mode",
    "custom_open_tag",
    "custom_close_tag",
    "unload_llm",
    "unload_comfy_models",
    "refresh_models",
];
// v2.0.0 also regrouped the widgets, so the migration below cannot assume the
// old and new orders line up - it maps every legacy value onto the current
// widget of the same name, which is order-independent by construction.

const LEGACY_ORDER_WITH_SEED_CONTROL = [
    ...LEGACY_ORDER.slice(0, 7),
    "control_after_generate",
    ...LEGACY_ORDER.slice(7),
];

// v1.x values of the removed enable_thinking widget. Used to confirm an array of
// the right length really is a legacy layout before rewriting anything.
const LEGACY_ENABLE_THINKING_VALUES = ["Model default", "Enabled", "Disabled"];

function toast(severity, summary, detail) {
    const manager = app.extensionManager?.toast;
    if (manager?.add) {
        manager.add({ severity, summary, detail, life: 5000 });
    } else if (severity === "error") {
        console.error(`[EA_LMStudio] ${summary}: ${detail}`);
    } else {
        console.log(`[EA_LMStudio] ${summary}: ${detail}`);
    }
}

/**
 * Collect each input's declared default so widgets that did not exist in v1.x
 * can be reset after a legacy remap (LiteGraph will already have filled them
 * with whatever landed at their index in the old array).
 */
function collectDefaults(nodeData) {
    const defaults = {};
    for (const group of ["required", "optional"]) {
        const inputs = nodeData?.input?.[group] ?? {};
        for (const [name, spec] of Object.entries(inputs)) {
            const [type, options] = Array.isArray(spec) ? spec : [spec, undefined];
            if (options && Object.prototype.hasOwnProperty.call(options, "default")) {
                defaults[name] = options.default;
            } else if (Array.isArray(type)) {
                defaults[name] = type[0]; // combo: first entry
            }
        }
    }
    return defaults;
}

/**
 * Repair widget values loaded from a workflow saved by v1.x.
 *
 * Returns true when a remap happened. A v2 workflow has at least 24 widget
 * values, so the 22/23 length test cannot collide with a current save; the
 * enable_thinking value check guards against a coincidence anyway.
 */
function migrateLegacyWidgetValues(node, widgetValues, defaults) {
    if (!Array.isArray(widgetValues)) return false;

    let order = null;
    if (widgetValues.length === LEGACY_ORDER.length) {
        order = LEGACY_ORDER;
    } else if (widgetValues.length === LEGACY_ORDER_WITH_SEED_CONTROL.length) {
        order = LEGACY_ORDER_WITH_SEED_CONTROL;
    }
    if (!order) return false;

    const enableThinkingValue = widgetValues[order.indexOf("enable_thinking")];
    if (!LEGACY_ENABLE_THINKING_VALUES.includes(enableThinkingValue)) return false;

    const byName = {};
    order.forEach((name, index) => {
        byName[name] = widgetValues[index];
    });

    for (const widget of node.widgets ?? []) {
        if (widget.name === PREVIEW_WIDGET_NAME) continue;
        if (Object.prototype.hasOwnProperty.call(byName, widget.name)) {
            widget.value = byName[widget.name];
        } else if (Object.prototype.hasOwnProperty.call(defaults, widget.name)) {
            // New in v2.0.0 - it holds a shifted v1 value right now.
            widget.value = defaults[widget.name];
        }
    }

    // No toast on purpose: the realignment is silent housekeeping, and a
    // banner on every v1 workflow load was noise users just clicked away.
}

/**
 * Grow a node that is stored smaller than its widgets need.
 *
 * A workflow stores the node's size, and LiteGraph restores it verbatim - it
 * does not re-check that the widgets still fit. Any release that adds a widget
 * therefore leaves every previously saved workflow a row or two too short, and
 * the overflowing widgets draw outside the node's frame. v2.0.0 has a net two
 * more widgets than v1.5.x, so this affects every upgraded workflow, not just
 * an unlucky few. Only ever grows - a deliberately widened node is preserved.
 */
function growToFitWidgets(node) {
    try {
        const [minWidth, minHeight] = node.computeSize();
        if (node.size[0] < minWidth || node.size[1] < minHeight) {
            node.setSize([
                Math.max(node.size[0], minWidth),
                Math.max(node.size[1], minHeight),
            ]);
        }
    } catch (err) {
        console.error("[EA_LMStudio] Could not resize node to fit widgets:", err);
    }
}

function getPreviewWidget(node) {
    let widget = node.widgets?.find((w) => w.name === PREVIEW_WIDGET_NAME);
    if (widget) return widget;

    widget = ComfyWidgets["STRING"](
        node,
        PREVIEW_WIDGET_NAME,
        ["STRING", { multiline: true }],
        app
    ).widget;

    if (widget.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = "0.8";
        widget.inputEl.placeholder = "Response appears here after the node runs";
        // Long responses must not grow the node without bound - cap and scroll.
        widget.inputEl.style.maxHeight = "400px";
        widget.inputEl.style.overflowY = "auto";
    }
    // Never let the preview reach the prompt or the saved widget values - it is
    // display only, and a stray extra value is exactly the kind of drift this
    // extension has to migrate away from.
    widget.options = { ...(widget.options ?? {}), serialize: false };
    widget.serializeValue = () => undefined;
    return widget;
}

/**
 * Push a fresh model list into every EA_LMStudio instance on the graph.
 *
 * The combo widgets of all instances share ONE values array (each was filled
 * from the single /object_info node definition), so the array is mutated IN
 * PLACE - assigning w.options.values a new array would detach only the
 * widgets we touch and leave every other instance's dropdown stale. Returns
 * the selections that no longer exist and were therefore reset.
 */
function applyModelChoices(choices) {
    const dropped = [];
    let nodes = [];
    try {
        nodes = app.graph?.findNodesByType?.(NODE_CLASS) ?? [];
    } catch {
        nodes = [];
    }
    for (const n of nodes) {
        for (const widgetName of ["model_selection", "draft_model_selection"]) {
            const w = n.widgets?.find((ww) => ww.name === widgetName);
            if (!w || !w.options || !Array.isArray(w.options.values)) continue;
            w.options.values.length = 0;
            w.options.values.push(...choices);
            if (!choices.includes(w.value)) {
                dropped.push(`#${n.id}: ${String(w.value)}`);
                w.value = choices[0];
            }
        }
    }
    return dropped;
}

app.registerExtension({
    name: "EA_LMStudio.NodeExtras",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const defaults = collectDefaults(nodeData);

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            try {
                migrateLegacyWidgetValues(this, info?.widgets_values, defaults);
            } catch (err) {
                console.error("[EA_LMStudio] Legacy workflow migration failed:", err);
            }
            growToFitWidgets(this);
        };

        /*
         * Keep the preview out of the saved workflow.
         *
         * The preview is display-only and is excluded from the API prompt, but
         * LiteGraph's serialize() still writes its text into widgets_values -
         * which would bake the last generated response into every workflow file
         * the user shares. It is always the last widget, so dropping its entry
         * cannot disturb the index of any real widget.
         */
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            onSerialize?.apply(this, arguments);
            const index = this.widgets?.findIndex((w) => w.name === PREVIEW_WIDGET_NAME);
            if (index >= 0 && Array.isArray(info?.widgets_values) && info.widgets_values.length > index) {
                info.widgets_values.splice(index, 1);
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const text = Array.isArray(message?.text) ? message.text.join("") : "";
            const widget = getPreviewWidget(this);
            widget.value = text;

            growToFitWidgets(this);
            app.graph.setDirtyCanvas(true, false);
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        const refreshWidget = node.widgets?.find((w) => w.name === "refresh_models");
        if (!refreshWidget) return;

        const originalCallback = refreshWidget.callback;

        refreshWidget.callback = async function (value) {
            if (!value) {
                if (originalCallback) originalCallback.call(this, value);
                return;
            }

            try {
                const resp = await api.fetchApi("/ea_lmstudio/refresh_models", {
                    method: "POST",
                });
                const data = await resp.json();

                if (data.success && data.models) {
                    const choices = [CUSTOM_MODEL_OPTION, ...data.models];
                    const dropped = applyModelChoices(choices);

                    toast("success", "LM Studio models refreshed", data.message);
                    if (dropped.length > 0) {
                        // Selections LM Studio no longer offers cannot be kept;
                        // say so instead of silently resetting them.
                        toast(
                            "warn",
                            "Stale model selections reset to Custom",
                            `${dropped.join(", ")} - pick from the refreshed list.`
                        );
                    }
                } else {
                    toast(
                        "warn",
                        "LM Studio model refresh failed",
                        data.message || "Unknown error - is LM Studio running with the server enabled?"
                    );
                }
            } catch (err) {
                toast("error", "LM Studio model refresh failed", String(err));
            }

            // Toggle back off so it acts like a one-shot button
            refreshWidget.value = false;

            if (originalCallback) originalCallback.call(this, false);
        };
    },

    /*
     * One check per page load: if the startup model fetch failed (LM Studio
     * not running yet, wrong port...), say so now instead of leaving the user
     * a silent "-- Custom --" dropdown to decode. A later successful refresh
     * clears the stored error, so this fires only while the problem stands.
     */
    async setup() {
        try {
            const resp = await api.fetchApi("/ea_lmstudio/models");
            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.success && data.error) {
                toast(
                    "warn",
                    "LM Studio unreachable at startup",
                    `${data.error} The model dropdown stays empty until LM Studio is reachable - then toggle 'Refresh models'.`
                );
            }
        } catch (err) {
            console.debug("[EA_LMStudio] Startup model-status check skipped:", err);
        }
    },
});
