import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { SelectedFeatureState } from "../../forecast/types/forecast.types";
import type { SessionForecastBundle } from "../types/session.types";

interface SessionForecastViewContextValue {
  activeSessionId: string | null;
  latestForecastBundle: SessionForecastBundle | null;
  selectedFeature: SelectedFeatureState | null;
  setActiveSessionId: (sessionId: string | null) => void;
  setLatestForecastBundle: (sessionId: string | null, bundle: SessionForecastBundle | null) => void;
  setSelectedFeature: (feature: SelectedFeatureState | null) => void;
  clearSelectedFeature: () => void;
}

const SessionForecastViewContext = createContext<SessionForecastViewContextValue | undefined>(undefined);

export function SessionForecastViewProvider({ children }: { children: ReactNode }) {
  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(null);
  const [latestForecastBundle, setLatestForecastBundleState] = useState<SessionForecastBundle | null>(null);
  const [selectedFeature, setSelectedFeatureState] = useState<SelectedFeatureState | null>(null);

  const setActiveSessionId = useCallback((sessionId: string | null) => {
    setActiveSessionIdState((previous) => {
      if (previous === sessionId) {
        return previous;
      }

      setLatestForecastBundleState(null);
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
      if (!bundle) {
        setSelectedFeatureState(null);
      }
    },
    [activeSessionId]
  );

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
      selectedFeature,
      setActiveSessionId,
      setLatestForecastBundle,
      setSelectedFeature,
      clearSelectedFeature
    }),
    [
      activeSessionId,
      latestForecastBundle,
      selectedFeature,
      setActiveSessionId,
      setLatestForecastBundle,
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
