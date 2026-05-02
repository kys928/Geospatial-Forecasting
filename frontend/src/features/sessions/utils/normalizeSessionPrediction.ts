import type { SessionPredictionResponse } from "../types/session.types";

export interface SessionPredictionSummaryViewModel {
  explanation: string | null;
  source: string | null;
  recommendation: string | null;
  uncertaintyNote: string | null;
  riskLevel: string | null;
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value as Record<string, unknown>;
}

function getNested(record: Record<string, unknown> | null, path: string[]): unknown {
  let current: unknown = record;

  for (const segment of path) {
    const asObj = asRecord(current);
    if (!asObj || !(segment in asObj)) {
      return null;
    }
    current = asObj[segment];
  }

  return current;
}

function findStringByCandidatePaths(
  root: Record<string, unknown> | null,
  candidatePaths: string[][]
): string | null {
  for (const path of candidatePaths) {
    const value = asNonEmptyString(getNested(root, path));
    if (value) {
      return value;
    }
  }

  return null;
}

export function normalizeSessionPrediction(
  prediction: SessionPredictionResponse | null
): SessionPredictionSummaryViewModel | null {
  const root = asRecord(prediction);

  if (!root) {
    return null;
  }

  const explanation = findStringByCandidatePaths(root, [
    ["explanation", "summary"],
    ["summary", "explanation"],
    ["prediction", "explanation", "summary"],
    ["message"]
  ]);

  const source = findStringByCandidatePaths(root, [
    ["explanation_source"],
    ["source"],
    ["explanation", "source"],
    ["prediction", "source"],
    ["metadata", "source"]
  ]);

  const recommendation = findStringByCandidatePaths(root, [
    ["explanation", "recommendation"],
    ["recommendation"],
    ["prediction", "recommendation"]
  ]);

  const uncertaintyNote = findStringByCandidatePaths(root, [
    ["explanation", "uncertainty_note"],
    ["uncertainty_note"],
    ["prediction", "uncertainty_note"]
  ]);

  const riskLevel = findStringByCandidatePaths(root, [
    ["explanation", "risk_level"],
    ["risk_level"],
    ["prediction", "risk_level"],
    ["summary", "risk_level"]
  ]);

  return {
    explanation,
    source,
    recommendation,
    uncertaintyNote,
    riskLevel
  };
}
