import { safeText } from "./formatters";

export function hasMeaningfulPlume(params: {
  affectedAreaM2: unknown;
  affectedCellsAboveThreshold: unknown;
  maxConcentration: unknown;
  explanationSummary: unknown;
  riskLevel: string;
}): boolean {
  const toNumber = (value: unknown): number | null => {
    const parsed = typeof value === "string" ? Number(value) : value;
    return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
  };
  const affectedArea = toNumber(params.affectedAreaM2);
  const affectedCells = toNumber(params.affectedCellsAboveThreshold);
  const maxConcentration = toNumber(params.maxConcentration);
  const explanationText = safeText(params.explanationSummary, "").toLowerCase();

  if ((affectedArea != null && affectedArea > 0) || (affectedCells != null && affectedCells > 0) || (maxConcentration != null && maxConcentration > 0)) return true;
  if (affectedArea == 0 || affectedCells == 0 || maxConcentration == 0 || explanationText.includes("no meaningful plume")) return false;
  return params.riskLevel.toLowerCase() !== "low";
}
