import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/features/ops/components/OpsRegistryTab.tsx", import.meta.url), "utf8");

assert.match(source, /No model records are currently registered\./, "empty registry shows honest empty state");
assert.match(source, /models\.length === 0/, "empty state is driven by the real registry model count");
assert.match(source, /models\.length > 0/, "table rendering is gated on real model records");
assert.match(source, /\{models\.map\(\(model\) => \{/, "table rows are rendered from real registry models only");
assert.doesNotMatch(source, /demo_convlstm_v0_1/, "demo model id is not rendered or defined");
assert.doesNotMatch(source, /demoRow/, "demo row object has been removed");
assert.doesNotMatch(source, /isDemo/, "demo-row branching has been removed");
assert.doesNotMatch(source, /Showing a demo row/, "old demo-row empty copy has been removed");
assert.doesNotMatch(source, /displayModels/, "display model fallback collection has been removed");
