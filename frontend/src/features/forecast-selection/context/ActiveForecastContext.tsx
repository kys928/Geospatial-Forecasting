import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import type { SelectedFeatureState } from "../../forecast/types/forecast.types";
import { sessionClient } from "../../sessions/api/sessionClient";
import type { ForecastFramesMetadata } from "../../sessions/types/session.types";
import { httpGet, httpPost } from "../../../services/api/http";
import { MODEL_BY_ID, type ActiveForecastKind, type ActiveModelId } from "../modelRegistry";

interface ActiveForecastBundle { summary: Record<string, unknown>; geojson: Record<string, unknown>; rasterMetadata: Record<string, unknown>; explanation?: Record<string, unknown>; }
interface AdoptActiveForecastInput { scenarioId: string | null; scenarioLabel: string | null; modelId: ActiveModelId; forecastKind: ActiveForecastKind; sessionId: string | null; persistedForecastId: string | null; bundle: ActiveForecastBundle | null; framesMetadata: ForecastFramesMetadata | null; }
interface Context {
  activeScenarioId: string | null; activeScenarioLabel: string | null; activeModelId: ActiveModelId; activeModelLabel: string; activeForecastKind: ActiveForecastKind; activeSessionId: string | null; activePersistedForecastId: string | null; activeForecastBundle: ActiveForecastBundle | null; activeFramesMetadata: ForecastFramesMetadata | null; status: "idle" | "loading" | "running" | "ready" | "error"; error: string | null; selectedFeature: SelectedFeatureState | null; selectedFrameIndex: number; lastSuccessfulRunKey: string;
  setActiveScenario: (scenarioId: string, label?: string | null) => void; setActiveModel: (modelId: ActiveModelId) => void; runActiveForecast: () => Promise<void>; clearActiveForecast: () => void; setSelectedFeature: (feature: SelectedFeatureState | null) => void; setSelectedFrameIndex: (idx: number) => void; adoptActiveForecast: (input: AdoptActiveForecastInput) => void;
}
const ActiveForecastContext = createContext<Context | undefined>(undefined);

export function ActiveForecastProvider({ children }: { children: ReactNode }) {
  const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null);
  const [activeScenarioLabel, setActiveScenarioLabel] = useState<string | null>(null);
  const [activeModelId, setActiveModelId] = useState<ActiveModelId>("convlstm_multistep");
  const [activeForecastKind, setActiveForecastKind] = useState<ActiveForecastKind>("none");
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activePersistedForecastId, setActivePersistedForecastId] = useState<string | null>(null);
  const [activeForecastBundle, setActiveForecastBundle] = useState<ActiveForecastBundle | null>(null);
  const [activeFramesMetadata, setActiveFramesMetadata] = useState<ForecastFramesMetadata | null>(null);
  const [status, setStatus] = useState<Context["status"]>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<SelectedFeatureState | null>(null);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const runIdRef = useRef(0);
  const inFlightRunKeyRef = useRef<string | null>(null);
  const [lastSuccessfulRunKey, setLastSuccessfulRunKey] = useState("");

  const adoptActiveForecast = useCallback((input: AdoptActiveForecastInput) => {
    setActiveScenarioId(input.scenarioId); setActiveScenarioLabel(input.scenarioLabel); setActiveModelId(input.modelId); setActiveForecastKind(input.forecastKind); setActiveSessionId(input.sessionId); setActivePersistedForecastId(input.persistedForecastId); setActiveForecastBundle(input.bundle); setActiveFramesMetadata(input.framesMetadata); setSelectedFrameIndex(input.framesMetadata?.default_frame_index ?? 0); setSelectedFeature(null);
  }, []);

  const runActiveForecast = useCallback(async () => {
    const model = MODEL_BY_ID[activeModelId];
    const scenarioIdForRun = activeScenarioId;
    const scenarioLabelForRun = activeScenarioLabel;
    const runKey = `${activeModelId}:${scenarioIdForRun ?? "none"}`;
    if (inFlightRunKeyRef.current === runKey) return;

    const requestId = ++runIdRef.current;
    inFlightRunKeyRef.current = runKey;
    setStatus("running"); setError(null);
    try {
      if (activeModelId === "convlstm_multistep") {
        const run = await sessionClient.runSessionForecast({ metadata: { selected_scenario_id: scenarioIdForRun, selected_model_id: activeModelId, scenario_mode: scenarioIdForRun ? "selected" : "degraded_default_input" } });
        const [bundle, frames] = await Promise.all([sessionClient.getLatestForecastBundle(run.sessionId, { includeExplanation: false }), sessionClient.getLatestForecastFrames(run.sessionId)]);
        if (requestId !== runIdRef.current) return;
        adoptActiveForecast({ scenarioId: scenarioIdForRun, scenarioLabel: scenarioLabelForRun, modelId: activeModelId, forecastKind: model.forecastKind, sessionId: run.sessionId, persistedForecastId: null, bundle, framesMetadata: frames });
      } else if (activeModelId === "ridge_baseline") {
        if (!scenarioIdForRun) throw new Error("Select a scenario first.");
        await httpPost(`/forecast-context/dataset-scenarios/${scenarioIdForRun}/activate`, {});
        await httpPost("/forecast-context/dataset-playback/state", { enabled: true, active_scenario_id: scenarioIdForRun, playback_running: false });
        const [ctx, overlay] = await Promise.all([httpGet<Record<string, unknown>>("/forecast-context/latest?source=dataset"), httpGet<Record<string, unknown>>(`/forecast-context/dataset-scenarios/${scenarioIdForRun}/overlay`).catch(() => ({ type: "FeatureCollection", features: [] }))]);
        if (requestId !== runIdRef.current) return;
        const summary: Record<string, unknown> = { ...((ctx.forecast as Record<string, unknown>) ?? {}), metadata: { model_id: "ridge_baseline", model_name: "ridge_plume_baseline", model_label: "Ridge baseline", forecast_kind: "dataset_ridge", input_source: "dataset_playback", frame_count: 1, selected_scenario_id: scenarioIdForRun } };
        adoptActiveForecast({ scenarioId: scenarioIdForRun, scenarioLabel: scenarioLabelForRun, modelId: activeModelId, forecastKind: model.forecastKind, sessionId: null, persistedForecastId: null, bundle: { summary, geojson: overlay, rasterMetadata: {} }, framesMetadata: { forecast_id: String(summary["forecast_id"] ?? scenarioIdForRun), model: "ridge_plume_baseline", model_version: null, frame_count: 1, frame_indices: [0], default_frame_index: 0, shape: [] } });
      } else {
        const create = await httpPost<{ forecast_id: string }>("/forecast", { metadata: { selected_scenario_id: scenarioIdForRun, selected_model_id: activeModelId } });
        const [summary, geojson, rasterMetadata] = await Promise.all([httpGet<Record<string, unknown>>(`/forecast/${create.forecast_id}/summary`), httpGet<Record<string, unknown>>(`/forecast/${create.forecast_id}/geojson`), httpGet<Record<string, unknown>>(`/forecast/${create.forecast_id}/raster-metadata`)]);
        if (requestId !== runIdRef.current) return;
        const patchedSummary = { ...summary, metadata: { ...((summary.metadata as Record<string, unknown>) ?? {}), model_id: "gaussian_baseline", model_name: String((summary as any).model ?? "gaussian_plume"), model_label: "Gaussian baseline", forecast_kind: "batch_gaussian", frame_count: 1, selected_scenario_id: scenarioIdForRun ?? "configured_default", scenario_usage: "not_supported_by_backend", scenario_note: "Gaussian baseline uses configured/default batch forecast; selected scenario is shown for comparison but not consumed by this backend path." } };
        adoptActiveForecast({ scenarioId: scenarioIdForRun, scenarioLabel: scenarioLabelForRun, modelId: activeModelId, forecastKind: model.forecastKind, sessionId: null, persistedForecastId: create.forecast_id, bundle: { summary: patchedSummary, geojson, rasterMetadata }, framesMetadata: { forecast_id: create.forecast_id, model: "gaussian_plume", model_version: null, frame_count: 1, frame_indices: [0], default_frame_index: 0, shape: [] } });
      }
      if (requestId === runIdRef.current) { setLastSuccessfulRunKey(runKey); setStatus("ready"); }
    } catch (e) {
      if (requestId === runIdRef.current) { setLastSuccessfulRunKey(""); setStatus("error"); setError(e instanceof Error ? e.message : String(e)); }
    } finally {
      if (requestId === runIdRef.current) inFlightRunKeyRef.current = null;
    }
  }, [activeModelId, activeScenarioId, activeScenarioLabel, adoptActiveForecast]);

  const clearActiveForecast = useCallback(() => { runIdRef.current += 1; setLastSuccessfulRunKey(""); setActiveForecastKind("none"); setActiveSessionId(null); setActivePersistedForecastId(null); setActiveForecastBundle(null); setActiveFramesMetadata(null); setSelectedFeature(null); setSelectedFrameIndex(0); setStatus("idle"); setError(null); inFlightRunKeyRef.current = null; }, []);
  const setActiveScenario = useCallback((id: string, label?: string | null) => { runIdRef.current += 1; setLastSuccessfulRunKey(""); setActiveScenarioId(id); setActiveScenarioLabel(label ?? null); setActiveForecastKind("none"); setActiveSessionId(null); setActivePersistedForecastId(null); setActiveForecastBundle(null); setActiveFramesMetadata(null); setSelectedFeature(null); setSelectedFrameIndex(0); setStatus("idle"); setError(null); }, []);
  const setActiveModel = useCallback((modelId: ActiveModelId) => { runIdRef.current += 1; setLastSuccessfulRunKey(""); setActiveModelId(modelId); setActiveForecastKind("none"); setActiveSessionId(null); setActivePersistedForecastId(null); setActiveForecastBundle(null); setActiveFramesMetadata(null); setSelectedFeature(null); setSelectedFrameIndex(0); setStatus("idle"); setError(null); }, []);

  const value = useMemo(() => ({ activeScenarioId, activeScenarioLabel, activeModelId, activeModelLabel: MODEL_BY_ID[activeModelId].label, activeForecastKind, activeSessionId, activePersistedForecastId, activeForecastBundle, activeFramesMetadata, status, error, selectedFeature, selectedFrameIndex, lastSuccessfulRunKey, setActiveScenario, setActiveModel, runActiveForecast, clearActiveForecast, setSelectedFeature, setSelectedFrameIndex, adoptActiveForecast }), [activeScenarioId, activeScenarioLabel, activeModelId, activeForecastKind, activeSessionId, activePersistedForecastId, activeForecastBundle, activeFramesMetadata, status, error, selectedFeature, selectedFrameIndex, lastSuccessfulRunKey, setActiveScenario, setActiveModel, runActiveForecast, clearActiveForecast, adoptActiveForecast]);
  return <ActiveForecastContext.Provider value={value}>{children}</ActiveForecastContext.Provider>;
}
export function useActiveForecast() { const ctx = useContext(ActiveForecastContext); if (!ctx) throw new Error("useActiveForecast must be used within ActiveForecastProvider"); return ctx; }
