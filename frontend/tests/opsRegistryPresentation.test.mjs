import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/features/ops/components/OpsRegistryTab.tsx", import.meta.url), "utf8");

assert.match(source, /export function modelParentLabel/, "modelParentLabel helper is present");
assert.match(source, /\["parent_active_model_id"\]/, "parent helper reads parent_active_model_id");
assert.match(source, /\["selected_resume_checkpoint"\]/, "parent helper falls back to selected_resume_checkpoint");
assert.match(source, /export function modelGateLabel/, "modelGateLabel helper is present");
assert.match(source, /Stage 3 rejected; promoted Stage 2/, "gate helper reports stage-gate rejection with stage-2 promotion");
assert.match(source, /export function modelCheckpointHealthLabel/, "checkpoint health helper is present");
assert.match(source, /return exists \? "Yes" : "No";/, "checkpoint health helper reports yes/no");
assert.match(source, /export function compactPathLabel/, "compact path helper is present");

assert.match(source, /<th>Parent \/ Trained from<\/th>/, "table includes parent lineage column");
assert.match(source, /<th>Gate \/ Promotion<\/th>/, "table includes gate column");
assert.match(source, /<th>Checkpoint<\/th>/, "table includes checkpoint health column");
assert.doesNotMatch(source, /<th>Path<\/th>/, "main table no longer uses raw path as a primary column");
assert.match(source, /fieldRow\("Raw path", model\.path\)/, "raw path remains in Inspect Model technical details");
assert.match(source, /title="Lifecycle"/, "Inspect Model includes prominent lifecycle section");
assert.match(source, /rows=\{lifecycleRows\(inspectModel\)\}/, "Inspect Model lifecycle section uses lifecycle rows");
assert.match(source, /disabled=\{!canActivate \|\| busy\}/, "activation button availability remains guarded by existing canActivate logic");

assert.match(source, /<td>\{active \? "Yes" : "No"\}<\/td>/, "active model row still reports Active Yes/No");
