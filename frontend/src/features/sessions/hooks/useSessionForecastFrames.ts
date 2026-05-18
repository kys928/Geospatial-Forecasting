import { useCallback, useRef, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type { ForecastFramesMetadata } from "../types/session.types";

interface UseSessionForecastFramesState {
  framesMetadata: ForecastFramesMetadata | null;
  selectedFrameIndex: number;
  selectedFrameSummary: Record<string, unknown> | null;
  selectedFrameGeoJson: Record<string, unknown> | null;
  frameLoading: boolean;
  frameError: string | null;
}

const INITIAL_STATE: UseSessionForecastFramesState = {
  framesMetadata: null,
  selectedFrameIndex: 0,
  selectedFrameSummary: null,
  selectedFrameGeoJson: null,
  frameLoading: false,
  frameError: null
};

export function useSessionForecastFrames(sessionId: string | null) {
  const [state, setState] = useState<UseSessionForecastFramesState>(INITIAL_STATE);
  const sequenceRef = useRef(0);

  const clearFrames = useCallback(() => {
    sequenceRef.current += 1;
    setState(INITIAL_STATE);
  }, []);

  const fetchFrame = useCallback(async (id: string, frameIndex: number) => {
    const requestId = ++sequenceRef.current;
    setState((previous) => ({ ...previous, selectedFrameIndex: frameIndex, frameLoading: true, frameError: null }));
    try {
      const [summary, geojson] = await Promise.all([
        sessionClient.getLatestForecastFrameSummary(id, frameIndex),
        sessionClient.getLatestForecastFrameGeoJson(id, frameIndex)
      ]);
      if (requestId !== sequenceRef.current) {
        return;
      }
      setState((previous) => ({
        ...previous,
        selectedFrameIndex: frameIndex,
        selectedFrameSummary: summary,
        selectedFrameGeoJson: geojson,
        frameLoading: false,
        frameError: null
      }));
    } catch (error) {
      if (requestId !== sequenceRef.current) {
        return;
      }
      setState((previous) => ({
        ...previous,
        selectedFrameIndex: frameIndex,
        frameLoading: false,
        frameError: error instanceof Error ? error.message : "Could not load selected frame"
      }));
    }
  }, []);

  const refreshFrames = useCallback(async () => {
    if (!sessionId) {
      clearFrames();
      return;
    }
    const requestId = ++sequenceRef.current;
    setState((previous) => ({ ...previous, frameLoading: true, frameError: null }));

    try {
      const metadata = await sessionClient.getLatestForecastFrames(sessionId);
      if (requestId !== sequenceRef.current) {
        return;
      }
      const hasFrames = metadata.frame_count > 0 && metadata.frame_indices.length > 0;
      if (!hasFrames) {
        setState((previous) => ({ ...previous, framesMetadata: metadata, frameLoading: false, frameError: "Frame sequence unavailable" }));
        return;
      }

      const boundedDefault = metadata.frame_indices.includes(metadata.default_frame_index)
        ? metadata.default_frame_index
        : metadata.frame_indices[0];
      setState((previous) => ({ ...previous, framesMetadata: metadata }));
      await fetchFrame(sessionId, boundedDefault);
    } catch (error) {
      if (requestId !== sequenceRef.current) {
        return;
      }
      const message = error instanceof Error ? error.message : "Frame sequence unavailable";
      setState((previous) => ({ ...previous, frameLoading: false, frameError: message }));
    }
  }, [clearFrames, fetchFrame, sessionId]);

  const setSelectedFrameIndex = useCallback((frameIndex: number) => {
    if (!sessionId) {
      return;
    }
    void fetchFrame(sessionId, frameIndex);
  }, [fetchFrame, sessionId]);

  return {
    ...state,
    refreshFrames,
    setSelectedFrameIndex,
    clearFrames
  };
}
