import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/features/ops/components/OpsRegistryTab.tsx", import.meta.url), "utf8");
const rowMenuSource = source.slice(source.indexOf("className=\"ops-row-menu\""), source.indexOf("document.body", source.indexOf("className=\"ops-row-menu\"")));
const deleteHandlerSource = source.slice(source.indexOf("async function handleDeleteCheckpointFile"), source.indexOf("return (", source.indexOf("async function handleDeleteCheckpointFile")));
const dangerZoneSource = source.slice(source.indexOf("function CheckpointDangerZone"), source.indexOf("function ModelDetailSection"));

assert.match(rowMenuSource, /Activate model/, "row action menu still has Activate model");
assert.match(rowMenuSource, /Inspect model/, "row action menu still has Inspect model");
assert.doesNotMatch(rowMenuSource, /Delete checkpoint file/, "row action menu no longer exposes checkpoint deletion");
assert.doesNotMatch(rowMenuSource, /Delete not available/, "row action menu does not show delete placeholders");

assert.match(source, /function CheckpointDangerZone/, "Inspect Model includes a danger zone component");
assert.match(dangerZoneSource, /<summary>Danger zone<\/summary>/, "danger zone is explicitly labeled");
assert.match(dangerZoneSource, /checkpoint \.pt file/, "danger zone explains only checkpoint file deletion");
assert.match(dangerZoneSource, /Registry\s+metadata and history remain visible/, "danger zone explains metadata/history remain");
assert.match(dangerZoneSource, /active model checkpoints cannot be\s+deleted/, "danger zone explains active checkpoint protection");
assert.match(dangerZoneSource, /cannot be undone/, "danger zone explains irreversibility");
assert.match(dangerZoneSource, /disabled=\{deleteDisabled\}/, "delete button is disabled when ineligible or busy");
assert.match(dangerZoneSource, /onClick=\{\(\) => void onDelete\(model\)\}/, "danger zone delete calls the delete handler");

assert.match(source, /if \(!modelId\) return "Model ID is missing\.";/, "delete disabled if model_id is missing");
assert.match(source, /Only adaptation checkpoint records can be deleted/, "delete disabled for non-adaptation records");
assert.match(source, /Active model checkpoints cannot be deleted\./, "delete disabled for active models");
assert.match(source, /Checkpoint file is already missing\./, "delete disabled when checkpoint file is missing");

assert.match(deleteHandlerSource, /window\.confirm/, "delete handler requires explicit confirmation");
assert.match(deleteHandlerSource, /opsClient\.deleteAdaptationCheckpointFile/, "delete handler calls backend checkpoint-file delete API");
assert.match(deleteHandlerSource, /runAction\(modelId, "Delete checkpoint file"/, "delete handler refreshes through the existing action runner");
assert.doesNotMatch(deleteHandlerSource, /setInspectModelId\(null\)/, "delete handler does not close Inspect Model modal");
assert.match(source, /\{actionNotice \? <p className="muted">\{actionNotice\}<\/p> : null\}/, "modal shows action notices while staying open");
