export function cleanAssistantText(text: string): string {
  const withoutCellCounts = text
    .replace(/\b\d+[\d,]*\s+grid cells?\b/gi, "a broader part of the model grid")
    .replace(/grid cells?\s+(?:are|is)\s+above\s+threshold/gi, "a wider predicted plume area is above threshold")
    .replace(/affected_cells_above_threshold\s*[:=]\s*\d+[\d,]*/gi, "predicted plume extent is available in technical details");
  const thresholdOnly = /^\s*(?:the\s+)?threshold\s*(?:is|=|used)?\s*[0-9.]+\s*\.?\s*$/i;
  if (thresholdOnly.test(withoutCellCounts)) return "The threshold is only one technical indicator; use plume extent, risk, and wind context together.";
  return withoutCellCounts.trim();
}

export function safeText(value: unknown, fallback = "Unavailable"): string {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}
export function isPresentValue(value: unknown): boolean {
  if (value == null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}
export function getNestedValue(source: unknown, ...paths: string[]): unknown {
  for (const path of paths) {
    const parts = path.split(".");
    let current: unknown = source;
    let found = true;
    for (const part of parts) {
      if (current && typeof current === "object" && part in (current as Record<string, unknown>)) {
        current = (current as Record<string, unknown>)[part];
      } else {
        found = false;
        break;
      }
    }
    if (found && isPresentValue(current)) return current;
  }
  return undefined;
}

export function formatTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function formatNumber(value: unknown, digits = 2): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return parsed.toLocaleString(undefined, { maximumFractionDigits: digits });
}
export function formatArea(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  if (parsed <= 0) return "Estimated from plume grid";
  if (Math.abs(parsed) >= 10000) return `${(parsed / 10000).toLocaleString(undefined, { maximumFractionDigits: 1 })} ha`;
  return `${parsed.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²`;
}
export function formatCoordinate(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return parsed.toFixed(5);
}
export function formatSpeed(value: unknown): string {
  const n = formatNumber(value);
  return n === "Unavailable" ? n : `${n} m/s`;
}
export function formatDirection(value: unknown): string {
  if (!isPresentValue(value)) return "Unavailable";
  if (typeof value === "number") return `${value}°`;
  return String(value);
}
export function formatTemperature(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n} °C`;
}
export function formatPressure(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n} hPa`;
}
export function formatPercent(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n}%`;
}
export function formatDurationMinutes(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return `${formatNumber(parsed, 0)} min`;
}
export function formatGridSize(value: unknown): string {
  if (!isPresentValue(value)) return "Unavailable";
  if (Array.isArray(value)) return value.join(" × ");
  if (typeof value === "object") {
    const rows = getNestedValue(value, "rows");
    const cols = getNestedValue(value, "columns", "cols");
    if (isPresentValue(rows) && isPresentValue(cols)) return `${rows} × ${cols}`;
  }
  return formatUnknown(value);
}

export function formatUnknown(value: unknown): string {
  if (value == null) return "Unavailable";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.length ? value.map((item) => formatUnknown(item)).join(", ") : "Unavailable";
  return "Unavailable";
}

export function formatRiskLevel(value: unknown): string {
  const text = safeText(value, "Unknown").toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
}
