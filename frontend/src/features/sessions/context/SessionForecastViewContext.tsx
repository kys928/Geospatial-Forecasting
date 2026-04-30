import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { SelectedFeatureState } from "../../forecast/types/forecast.types";
import type { SessionForecastBundle } from "../types/session.types";

type ForecastViewSource = "none" | "session" | "persisted";

interface SessionForecastViewContextValue {
  activeSessionId: string | null;
  latestForecastBundle: SessionForecastBundle | null;
  forecastViewSource: ForecastViewSource;
  activePersistedForecastId: string | null;
  selectedFeature: SelectedFeatureState | null;
  setActiveSessionId: (sessionId: string | null) => void;
  setLatestForecastBundle: (sessionId: string | null, bundle: SessionForecastBundle | null) => void;
  setPersistedForecastBundle: (forecastId: string, bundle: SessionForecastBundle) => void;
  setSelectedFeature: (feature: SelectedFeatureState | null) => void;
  clearSelectedFeature: () => void;
}

const SessionForecastViewContext = createContext<SessionForecastViewContextValue | undefined>(undefined);

export function SessionForecastViewProvider({ children }: { children: ReactNode }) {
  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(null);
  const [latestForecastBundle, setLatestForecastBundleState] = useState<SessionForecastBundle | null>(null);
  const [forecastViewSource, setForecastViewSource] = useState<ForecastViewSource>("none");
  const [activePersistedForecastId, setActivePersistedForecastId] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeatureState] = useState<SelectedFeatureState | null>(null);

  const setActiveSessionId = useCallback((sessionId: string | null) => {
    setActiveSessionIdState((previous) => {
      if (previous === sessionId) {
        return previous;
      }

      setLatestForecastBundleState(null);
      setForecastViewSource("none");
      setActivePersistedForecastId(null);
      setSelectedFeatureState(null);
      return sessionId;
    });
  }, []);

  const setLatestForecastBundle = useCallback(
    (sessionId: string | null, bundle: SessionForecastBundle | null) => {
      if (sessionId !== activeSessionId) {
        return;
      }
      setLatestForecastBundleState(bundle);
      setForecastViewSource(bundle ? "session" : "none");
      setActivePersistedForecastId(null);
      if (!bundle) {
        setSelectedFeatureState(null);
      }
    },
    [activeSessionId]
  );

  const setPersistedForecastBundle = useCallback((forecastId: string, bundle: SessionForecastBundle) => {
    setLatestForecastBundleState(bundle);
    setForecastViewSource("persisted");
    setActivePersistedForecastId(forecastId);
  }, []);

  const setSelectedFeature = useCallback((feature: SelectedFeatureState | null) => {
    setSelectedFeatureState(feature);
  }, []);

  const clearSelectedFeature = useCallback(() => {
    setSelectedFeatureState(null);
  }, []);

  const value = useMemo(
    () => ({
      activeSessionId,
      latestForecastBundle,
      forecastViewSource,
      activePersistedForecastId,
      selectedFeature,
      setActiveSessionId,
      setLatestForecastBundle,
      setPersistedForecastBundle,
      setSelectedFeature,
      clearSelectedFeature
    }),
    [
      activeSessionId,
      latestForecastBundle,
      forecastViewSource,
      activePersistedForecastId,
      selectedFeature,
      setActiveSessionId,
      setLatestForecastBundle,
      setPersistedForecastBundle,
      setSelectedFeature,
      clearSelectedFeature
    ]
  );

  return <SessionForecastViewContext.Provider value={value}>{children}</SessionForecastViewContext.Provider>;
}

export function useSessionForecastView() {
  const context = useContext(SessionForecastViewContext);
  if (!context) {
    throw new Error("useSessionForecastView must be used within SessionForecastViewProvider");
  }
  return context;
}
