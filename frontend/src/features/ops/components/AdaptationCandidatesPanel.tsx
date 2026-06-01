import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { AdaptationCandidate, AdaptationPromotionDecision, AdaptationStorageWarning, CandidateDecisionRequest } from "../types/ops.types";

function fmt(value: unknown, fallback = "Not reported") {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim() === "") return fallback;
  return String(value);
}

function structured(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not reported";
  if (["string", "number", "boolean"].includes(typeof value)) return String(value);
  if (Array.isArray(value)) return value.map(structured).filter((item) => item !== "Not reported").join("; ") || "Not reported";
  if (typeof value === "object") return Object.entries(value as Record<string, unknown>).map(([key, val]) => `${key}: ${structured(val)}`).filter((item) => !item.endsWith("Not reported")).join("; ") || "Not reported";
  return "Not reported";
}

function decisionText(decision: AdaptationPromotionDecision | Record<string, unknown> | null | undefined): string {
  const source = "decision" in (decision ?? {}) ? (decision as AdaptationPromotionDecision).decision : decision;
  if (!source || typeof source !== "object") return "Not reported";
  const record = source as Record<string, unknown>;
  return fmt(record.classification ?? record.action ?? record.reason ?? structured(record));
}

function reasonText(candidate: AdaptationCandidate): string {
  const decision = candidate.last_adaptation_promotion_decision ?? candidate.last_promotion_result;
  if (!decision) return "Not reported";
  if (typeof decision.reason === "string") return decision.reason;
  if (Array.isArray(decision.reasons)) return decision.reasons.map(String).slice(0, 2).join("; ");
  if (Array.isArray(decision.warnings)) return decision.warnings.map(String).slice(0, 2).join("; ");
  return structured(decision);
}

function compactPath(value: unknown): string {
  const text = fmt(value);
  if (text === "Not reported" || text.length <= 34) return text;
  const parts = text.split("/").filter(Boolean);
  if (parts.length >= 2) return `.../${parts.slice(-2).join("/")}`;
  return `${text.slice(0, 14)}...${text.slice(-14)}`;
}

function isRejected(candidate: AdaptationCandidate): boolean {
  const status = String(candidate.status ?? "").toLowerCase();
  const approval = String(candidate.approval_status ?? "").toLowerCase();
  return status === "rejected" || approval === "rejected" || approval === "rejected_by_operator";
}

export function AdaptationCandidatesPanel({ activeModelId, onRegistryRefresh }: { activeModelId: string | null; onRegistryRefresh: () => Promise<void> }) {
  const [candidates, setCandidates] = useState<AdaptationCandidate[]>([]);
  const [storage, setStorage] = useState<AdaptationStorageWarning | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [candidatePayload, storagePayload] = await Promise.all([opsClient.getAdaptationCandidates(), opsClient.getAdaptationStorageWarnings()]);
      setCandidates(candidatePayload.candidates);
      setStorage(storagePayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load adaptation candidates.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  async function runAction(modelId: string, label: string, action: () => Promise<AdaptationPromotionDecision | unknown>, refreshStorage = false) {
    setActionId(`${modelId}:${label}`);
    setNotice(null);
    setError(null);
    try {
      const result = await action();
      const decision = result && typeof result === "object" ? decisionText(result as AdaptationPromotionDecision) : "Completed";
      setNotice(`${label} completed for ${modelId}: ${decision}`);
      await Promise.all([refresh(), onRegistryRefresh(), refreshStorage ? opsClient.getAdaptationStorageWarnings().then(setStorage) : Promise.resolve()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} failed for ${modelId}.`);
    } finally {
      setActionId(null);
    }
  }

  function decisionPayload(comment?: string): CandidateDecisionRequest {
    return { actor: "ops-ui", comment };
  }

  return <section className="panel" style={{ overflow: "visible" }}>
    <div className="button-row" style={{ justifyContent: "space-between", alignItems: "center" }}>
      <div><h3 style={{ margin: 0 }}>Adaptation Candidates</h3><p className="muted" style={{ margin: "4px 0 0" }}>Manual actions use backend compatibility and promotion checks as the source of truth.</p></div>
      <button className="secondary-button" onClick={() => void refresh()} disabled={loading}>Refresh</button>
    </div>

    {storage ? <div className={`ops-readiness-item ${storage.checkpoint_count_warning || storage.disk_usage_warning ? "ops-readiness-not_met" : "ops-readiness-met"}`} style={{ marginTop: 10 }}><span className="ops-readiness-dot" /><div><strong>Checkpoint storage</strong><p className="muted" style={{ margin: 0 }}>{storage.message}</p><p className="muted" style={{ margin: 0 }}>Checkpoints: {storage.checkpoint_count} · Registered adaptation models: {fmt(storage.registered_adaptation_model_count)} · Disk: {storage.disk_usage_percent.toFixed(1)}% · Automatic deletion: {storage.automatic_deletion ? "enabled" : "disabled"}</p></div></div> : null}
    {loading ? <p className="muted">Loading adaptation candidates...</p> : null}
    {error ? <p className="failure-text">{error}</p> : null}
    {notice ? <p className="muted">{notice}</p> : null}

    {!loading && !error && candidates.length === 0 ? <p className="muted">No adaptation candidates have been registered yet.</p> : null}
    {candidates.length > 0 ? <div style={{ overflowX: "auto", marginTop: 10 }}><table className="ops-model-table"><thead><tr><th>Model ID</th><th>Status</th><th>Approval</th><th>Checkpoint</th><th>Run</th><th>Best / Final</th><th>Decision</th><th>Reason / warning</th><th>Actions</th></tr></thead><tbody>{candidates.map((candidate) => {
      const isActive = candidate.model_id === activeModelId || candidate.status === "active";
      const rejected = isRejected(candidate);
      const canApprove = !isActive && !rejected;
      const canApplyPolicy = !isActive && !rejected;
      const busyPrefix = `${candidate.model_id}:`;
      const busy = actionId?.startsWith(busyPrefix) ?? false;
      const bestCheckpoint = compactPath(candidate.best_overall_checkpoint);
      const finalCheckpoint = compactPath(candidate.final_checkpoint);
      return <tr key={candidate.model_id}><td title={candidate.model_id}>{candidate.model_id}</td><td>{fmt(candidate.status)}</td><td>{fmt(candidate.approval_status)}</td><td>{candidate.checkpoint_file_exists ? "Yes" : "No"}</td><td title={fmt(candidate.run_id)}>{fmt(candidate.run_id)}</td><td title={`${fmt(candidate.best_overall_checkpoint)} / ${fmt(candidate.final_checkpoint)}`}>{bestCheckpoint} / {finalCheckpoint}</td><td>{decisionText(candidate.last_adaptation_promotion_decision)}</td><td title={reasonText(candidate)}>{reasonText(candidate)}</td><td><div className="button-row" style={{ gap: 6 }}>
        <button className="secondary-button" disabled={busy} onClick={() => void runAction(candidate.model_id, "Evaluate", () => opsClient.evaluateAdaptationCandidate(candidate.model_id))}>Evaluate</button>
        <button className="secondary-button" disabled={busy || !canApplyPolicy} onClick={() => void runAction(candidate.model_id, "Apply policy", () => opsClient.applyAdaptationPolicy(candidate.model_id))}>Apply policy</button>
        {canApprove ? <button className="secondary-button" disabled={busy} onClick={() => { if (window.confirm(`Approve and activate ${candidate.model_id}? Backend compatibility checks will run before activation.`)) void runAction(candidate.model_id, "Approve", () => opsClient.approveAdaptationCandidate(candidate.model_id, decisionPayload("Approved from Ops UI."))); }}>Approve</button> : null}
        {!isActive && !rejected ? <button className="secondary-button" disabled={busy} onClick={() => { if (window.confirm(`Reject ${candidate.model_id}? This keeps the checkpoint file and metadata.`)) void runAction(candidate.model_id, "Reject", () => opsClient.rejectAdaptationCandidate(candidate.model_id, decisionPayload("Rejected from Ops UI."))); }}>Reject</button> : null}
        {!isActive && candidate.checkpoint_file_exists ? <button className="secondary-button" disabled={busy} onClick={() => { if (window.confirm(`Delete checkpoint file for ${candidate.model_id}? Metadata/history remains and the row will stay visible.`)) void runAction(candidate.model_id, "Delete checkpoint file", () => opsClient.deleteAdaptationCheckpointFile(candidate.model_id, decisionPayload("Checkpoint file deleted from Ops UI.")), true); }}>Delete file</button> : null}
      </div></td></tr>;
    })}</tbody></table></div> : null}
  </section>;
}
