import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * EA_LMStudio Model Refresh Extension
 *
 * Intercepts the refresh_models toggle to fetch an updated model list from
 * the LM Studio server and update the dropdown widgets in-place.
 *
 * Compatibility:
 *   - Legacy LiteGraph frontend: full support via widget.options.values
 *   - Nodes 2.0 / Vue frontend: the server-side cache is always updated,
 *     so a browser refresh (F5) will pick up new models even if the
 *     in-place widget update doesn't propagate in a future Vue renderer.
 */

app.registerExtension({
    name: "EA_LMStudio.ModelRefresh",

    async nodeCreated(node) {
        if (node.comfyClass !== "EA_LMStudio") return;

        const refreshWidget = node.widgets?.find(w => w.name === "refresh_models");
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
                    const choices = ["-- Custom (enter below) --", ...data.models];

                    for (const widgetName of ["model_selection", "draft_model_selection"]) {
                        const w = node.widgets?.find(ww => ww.name === widgetName);
                        if (w && w.options) {
                            // Replace values array (breaks shared reference intentionally
                            // so other node instances also get the update)
                            w.options.values = choices;
                            if (!choices.includes(w.value)) {
                                w.value = choices[0];
                            }
                        }
                    }

                    console.log(
                        `[EA_LMStudio] Refreshed models: ${data.models.length} found`
                    );
                } else {
                    console.warn(
                        `[EA_LMStudio] Model refresh failed: ${data.message || "unknown error"}`
                    );
                }
            } catch (err) {
                console.error("[EA_LMStudio] Failed to refresh models:", err);
            }

            // Toggle back off so it acts like a one-shot button
            refreshWidget.value = false;

            if (originalCallback) originalCallback.call(this, false);
        };
    },
});
