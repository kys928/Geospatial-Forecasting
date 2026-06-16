import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/features/ops/components/OpsRegistryTab.tsx", import.meta.url), "utf8");
const rowMenuSource = source.slice(source.indexOf("className=\"ops-row-menu\""), source.indexOf("document.body", source.indexOf("className=\"ops-row-menu\"")));
const deleteHandlerSource = source.slice(source.indexOf("async function handleDeleteCheckpointFile"), source.indexOf("return (", source.indexOf("async function handleDeleteCheckpointFile")));
const inspectModalSource = source.slice(source.indexOf("{inspectModel ? ("), source.indexOf("function CheckpointStatusSection"));

assert.match(rowMenuSource, /Activate model/, "row action menu still has Activate model");
assert.match(rowMenuSource, />\s*Inspect\s*<\/button>/, "row action menu labels inspection as Inspect");
assert.match(rowMenuSource, /: "Delete"/, "row action menu exposes Delete");
assert.doesNotMatch(rowMenuSource, /Inspect \/ manage/, "row action menu no longer says Inspect / manage");
assert.ok(
  rowMenuSource.indexOf("Activate model") < rowMenuSource.indexOf("Inspect") &&
    rowMenuSource.indexOf("Inspect") < rowMenuSource.indexOf("Delete"),
  "row action menu order is Activate model, Inspect, Delete",
);

assert.doesNotMatch(source, /function CheckpointDangerZone/, "Inspect modal no longer includes a danger zone component");
assert.doesNotMatch(inspectModalSource, /Danger zone/, "Inspect modal does not show a Danger zone heading");
assert.doesNotMatch(inspectModalSource, /Delete checkpoint file/, "Inspect modal does not show checkpoint delete controls");
assert.match(inspectModalSource, /<summary>Raw training log<\/summary>/, "Inspect modal preserves raw training log section");

assert.match(source, /if \(!modelId\) return "Model ID is missing\.";/, "delete disabled if model_id is missing");
assert.match(source, /Only eligible adaptation checkpoint records can be deleted/, "delete disabled for non-adaptation records");
assert.match(source, /Active model checkpoints cannot be deleted\./, "delete disabled for active models");
assert.match(source, /Checkpoint status is not reported; backend will verify before deletion\./, "delete tooltip explains unknown checkpoint state uses backend verification");
assert.match(source, /Checkpoint file is already missing; backend will remove the registry record\./, "delete tooltip explains already-missing checkpoints can still remove registry records");
assert.doesNotMatch(source, /if \(exists === null\) return "Checkpoint status is not reported\.";/, "unknown checkpoint state does not disable deletion");
assert.match(rowMenuSource, /disabled=\{\s*!canDeleteCheckpointFile\(model, activeModelId\)/s, "row-menu delete is disabled when deletion is unavailable");
assert.match(rowMenuSource, /checkpointDeleteDisabledReason\(model, activeModelId\)/, "row-menu delete shows the unavailable reason");
assert.match(rowMenuSource, /onClick=\{\(\) => void handleDeleteCheckpointFile\(model\)\}/, "row-menu delete calls the delete handler");


assert.match(source, /ARCHIVED_ACTIVATION_APPROVAL_STATUSES = new Set\(\[\s*"approved_for_activation",\s*"approved",\s*"not_required",/s, "archived activation allows previously approved approval states");
assert.match(source, /status === "candidate" \|\|\s*status === "approved" \|\|\s*\(status === "archived" && ARCHIVED_ACTIVATION_APPROVAL_STATUSES\.has\(approval\)\)/s, "activate gating allows candidate, approved, and previously approved archived models");
assert.match(source, /isAdaptationRecord\(model\) && status === "candidate"/, "only candidate adaptation records use candidate approval activation flow");
assert.match(rowMenuSource, /Only candidate, approved, or archived previously-approved models can be activated\./, "disabled activate tooltip mentions archived previously-approved models");

assert.match(deleteHandlerSource, /window\.confirm/, "delete handler requires explicit confirmation");
assert.match(deleteHandlerSource, /opsClient\.deleteAdaptationCheckpointFile/, "delete handler calls backend checkpoint-file delete API");
assert.match(deleteHandlerSource, /runAction\(modelId, "Delete checkpoint record"/, "delete handler refreshes through the existing action runner with updated label");
assert.match(deleteHandlerSource, /model version will disappear from the table/, "delete confirmation explains the model version disappears from the table");
assert.doesNotMatch(deleteHandlerSource, /setInspectModelId\(null\)/, "delete handler does not close Inspect Model modal");

assert.match(source, /if \(deleted === true\) return false;/, "checkpoint_file_deleted true makes checkpoint health missing");
assert.match(source, /checkpointFileDeletedMetadataPresent\(model\)\) return "Checkpoint file is already deleted\.";/, "checkpoint_file_deleted true keeps deletion unavailable");
assert.doesNotMatch(source, /if \(exists === false\) return "Checkpoint file is already missing\.";/, "missing checkpoint no longer disables deletion");
assert.match(source, /return exists \? "Exists" : "Missing";/, "checkpoint_file_exists true can present as Exists and false as Missing");
assert.match(source, /return checkpointDeleteDisabledReason\(model, activeModelId\) === null;/, "delete can be available only when all disabled reasons are clear");
assert.match(source, /Checkpoint file was previously deleted; legacy registry metadata is preserved\./, "Inspect Model explains legacy deleted checkpoint metadata state");
assert.match(source, /Checkpoint deletion is only available for adaptation records\./, "Inspect Model explains non-adaptation deletion unavailability");
