import { useCallback, useRef, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type { ForecastFrameRasterPayload, ForecastFramesMetadata } from "../types/session.types";

interface UseSessionForecastFramesState {
  framesMetadata: ForecastFramesMetadata | null;
  selectedFrameIndex: number;
  selectedFrameSummary: Record<string, unknown> | null;
  selectedFrameGeoJson: Record<string, unknown> | null;
  selectedFrameRaster: ForecastFrameRasterPayload | null;
  frameLoading: boolean;
  frameError: string | null;
}

export interface RefreshedFrameResult {
  framesMetadata: ForecastFramesMetadata;
  selectedFrameIndex: number;
  selectedFrameSummary: Record<string, unknown>;
  selectedFrameGeoJson: Record<string, unknown>;
  selectedFrameRaster: ForecastFrameRasterPayload | null;
}

const INITIAL_STATE: UseSessionForecastFramesState = {
  framesMetadata: null,
  selectedFrameIndex: 0,
  selectedFrameSummary: null,
  selectedFrameGeoJson: null,
  selectedFrameRaster: null,
  frameLoading: false,
  frameError: null
};

export function useSessionForecastFrames(sessionId: string | null, options?: { includeFrameSummary?: boolean }) {
  const includeFrameSummary = options?.includeFrameSummary ?? true;
  const [state, setState] = useState<UseSessionForecastFramesState>(INITIAL_STATE);
  const sequenceRef = useRef(0);

  const clearFrames = useCallback(() => {
    sequenceRef.current += 1;
    setState(INITIAL_STATE);
  }, []);

  const fetchFrame = useCallback(async (id: string, frameIndex: number): Promise<{ summary: Record<string, unknown>; geojson: Record<string, unknown>; raster: ForecastFrameRasterPayload | null }> => {
    const requestId = ++sequenceRef.current;
    setState((previous) => ({ ...previous, selectedFrameIndex: frameIndex, frameLoading: true, frameError: null }));
    try {
      const [summary, geojson] = await Promise.all([
        includeFrameSummary
          ? sessionClient.getLatestForecastFrameSummary(id, frameIndex)
          : Promise.resolve({}),
        sessionClient.getLatestForecastFrameGeoJson(id, frameIndex)
      ]);
      let raster: ForecastFrameRasterPayload | null = null;
      let rasterError: unknown = null;
      try {
        raster = await sessionClient.getLatestForecastFrameRaster(id, frameIndex);
      } catch (error) {
        rasterError = error;
        if (import.meta.env.DEV) {
          console.debug("[forecast-map] raster unavailable for frame", { sessionId: id, frameIndex, error });
        }
      }
      if (requestId !== sequenceRef.current) {
        throw new Error(`Frame request superseded before completion for /sessions/${id}/forecast/latest/frames/${frameIndex}`);
      }
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] frame payload", {
          frameIndex,
          hasGeojson: Boolean(geojson),
          hasRaster: Boolean(raster),
          rasterError: rasterError ? (rasterError instanceof Error ? rasterError.message : String(rasterError)) : null
        });
      }
      setState((previous) => ({
        ...previous,
        selectedFrameIndex: frameIndex,
        selectedFrameSummary: summary,
        selectedFrameGeoJson: geojson,
        selectedFrameRaster: raster,
        frameLoading: false,
        frameError: null
      }));
      return { summary, geojson, raster };
    } catch (error) {
      if (requestId !== sequenceRef.current) {
        throw new Error(`Frame request superseded for /sessions/${id}/forecast/latest/frames/${frameIndex}`);
      }
      setState((previous) => ({
        ...previous,
        selectedFrameIndex: frameIndex,
        frameLoading: false,
        frameError: `GET /sessions/${id}/forecast/latest/frames/${frameIndex}/summary|geojson failed: ${error instanceof Error ? error.message : "Could not load selected frame"}`
      }));
      throw error;
    }
  }, [includeFrameSummary]);

  const refreshFramesForSession = useCallback(async (sessionIdOverride: string | null): Promise<RefreshedFrameResult | null> => {
    if (!sessionIdOverride) {
      clearFrames();
      return null;
    }
    const requestId = ++sequenceRef.current;
    setState((previous) => ({ ...previous, frameLoading: true, frameError: null }));

    try {
      const metadata = await sessionClient.getLatestForecastFrames(sessionIdOverride);
      if (requestId !== sequenceRef.current) {
        return null;
      }
      const hasFrames = metadata.frame_count > 0 && metadata.frame_indices.length > 0;
      if (!hasFrames) {
        setState((previous) => ({ ...previous, framesMetadata: metadata, frameLoading: false, frameError: `GET /sessions/${sessionIdOverride}/forecast/latest/frames returned no frames` }));
        return null;
      }

      const boundedDefault = metadata.frame_indices.includes(metadata.default_frame_index)
        ? metadata.default_frame_index
        : (metadata.frame_indices[0] ?? 0);
      setState((previous) => ({ ...previous, framesMetadata: metadata }));
      const { summary, geojson, raster } = await fetchFrame(sessionIdOverride, boundedDefault);
      return {
        framesMetadata: metadata,
        selectedFrameIndex: boundedDefault,
        selectedFrameSummary: summary,
        selectedFrameGeoJson: geojson,
        selectedFrameRaster: raster
      };
    } catch (error) {
      if (requestId !== sequenceRef.current) {
        return null;
      }
      const message = `GET /sessions/${sessionIdOverride}/forecast/latest/frames failed: ${error instanceof Error ? error.message : "Frame sequence unavailable"}`;
      setState((previous) => ({ ...previous, frameLoading: false, frameError: message }));
      throw error;
    }
  }, [clearFrames, fetchFrame]);

  const refreshFrames = useCallback(async (sessionIdOverride?: string): Promise<RefreshedFrameResult | null> => {
    return refreshFramesForSession(sessionIdOverride ?? sessionId);
  }, [refreshFramesForSession, sessionId]);

  const setSelectedFrameIndex = useCallback((frameIndex: number) => {
    if (!sessionId) {
      return;
    }
    void fetchFrame(sessionId, frameIndex);
  }, [fetchFrame, sessionId]);

  return {
    ...state,
    refreshFrames,
    refreshFramesForSession,
    setSelectedFrameIndex,
    clearFrames
  };
}
