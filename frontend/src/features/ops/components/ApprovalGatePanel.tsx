import { useState } from "react";

interface ApprovalGatePanelProps {
  candidateId: string | null;
  disabled: boolean;
  onApprove: (actor: string, comment?: string) => Promise<void>;
  onReject: (actor: string, comment?: string) => Promise<void>;
}

export function ApprovalGatePanel({ candidateId, disabled, onApprove, onReject }: ApprovalGatePanelProps) {
  const [actor, setActor] = useState("ui_operator");
  const [comment, setComment] = useState("");

  return (
    <section className="panel">
      <h3>Approval gate</h3>
      <p><strong>candidate_id:</strong> {candidateId ?? "none"}</p>
      <div className="field"><span>actor</span><input value={actor} onChange={(e) => setActor(e.target.value)} /></div>
      <div className="field"><span>comment</span><input value={comment} onChange={(e) => setComment(e.target.value)} /></div>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="primary-button" disabled={disabled || !candidateId} onClick={() => void onApprove(actor, comment || undefined)}>Approve</button>
        <button className="primary-button" disabled={disabled || !candidateId} onClick={() => void onReject(actor, comment || undefined)}>Reject</button>
      </div>
    </section>
  );
}
