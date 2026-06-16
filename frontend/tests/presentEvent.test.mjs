import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";
import { Buffer } from "node:buffer";
import { transform } from "esbuild";

const source = await readFile(new URL("../src/features/events/utils/presentEvent.ts", import.meta.url), "utf8");
const { code } = await transform(source, { loader: "ts", format: "esm" });
const moduleUrl = `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
const { presentEvent } = await import(moduleUrl);

function assertPresented(event, expected) {
  const presented = presentEvent(event, 0);
  for (const [key, value] of Object.entries(expected)) {
    assert.equal(presented[key], value, `${key} mismatch`);
  }
  return presented;
}

assertPresented(
  {
    event_type: "model_activated",
    payload: {
      model_id: "candidate_retrain-job-000407",
      previous_active_model_id: "robust_pretrained_baseline_v3c_tiny_recall_lift"
    }
  },
  {
    title: "Model activated",
    summary:
      "Model candidate_retrain-job-000407 was activated; previous active model was robust_pretrained_baseline_v3c_tiny_recall_lift.",
    category: "model",
    severity: "success",
    objectLabel: "candidate_retrain-job-000407"
  }
);

assertPresented(
  {
    event_type: "candidate_approved_by_operator",
    payload: { candidate_model_id: "candidate_retrain-job-000407", actor: "ops-ui" }
  },
  {
    title: "Model candidate approved",
    summary: "Candidate candidate_retrain-job-000407 was approved by ops-ui.",
    category: "model",
    severity: "success",
    objectLabel: "candidate_retrain-job-000407"
  }
);

assertPresented(
  {
    event_type: "automatic_retraining_skipped_cooldown",
    payload: { cooldown_reason: "cooldown_active", cooldown_scope: "automatic_cadence", remaining_seconds: 30 }
  },
  {
    title: "Automatic retraining skipped",
    summary: "Automatic training skipped because automatic cadence cooldown is active (30 seconds remaining).",
    category: "training",
    severity: "warning"
  }
);

assertPresented(
  {
    event_type: "adaptation_checkpoint_file_deleted",
    payload: { model_id: "candidate_retrain-job-000399", checkpoint_file_exists: false }
  },
  {
    title: "Checkpoint file deleted",
    summary: "Checkpoint file for candidate_retrain-job-000399 was deleted; registry metadata was preserved.",
    category: "model",
    severity: "warning",
    objectLabel: "candidate_retrain-job-000399"
  }
);

assertPresented(
  { event_type: "unknown_backend_event", payload: { message: "backend emitted an unknown event" } },
  {
    title: "Workspace event",
    summary: "A workspace event was recorded. Open details for technical information.",
    category: "unknown",
    severity: "info"
  }
);

assertPresented(
  { event_type: "candidate_approved", payload: { candidate_model_id: "legacy-candidate", actor: "legacy-ui" } },
  {
    title: "Model candidate approved",
    summary: "Candidate legacy-candidate was approved by legacy-ui.",
    objectLabel: "legacy-candidate"
  }
);

assertPresented(
  { event_type: "candidate_rejected_by_operator", model_id: "top-level-candidate", actor: "ops-ui" },
  {
    title: "Model candidate rejected",
    summary: "Candidate top-level-candidate was rejected by ops-ui.",
    objectLabel: "top-level-candidate"
  }
);


assertPresented(
  {
    event_type: "model_activated",
    model_id: "candidate_123",
    payload: { model_id: "", previous_active_model_id: "active_456" }
  },
  {
    title: "Model activated",
    summary: "Model candidate_123 was activated; previous active model was active_456.",
    objectLabel: "candidate_123"
  }
);
