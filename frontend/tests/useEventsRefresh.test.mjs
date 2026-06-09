import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";

const hookSource = await readFile(new URL("../src/features/events/hooks/useEvents.ts", import.meta.url), "utf8");
const tabSource = await readFile(new URL("../src/features/ops/components/OpsEventsTab.tsx", import.meta.url), "utf8");

assert.match(hookSource, /refresh = useCallback\(async \(options: EventsRefreshOptions = \{\}\)/, "refresh accepts options");
assert.match(hookSource, /opsClient\.getEvents\(200, \{ force: options\.force \}\)/, "refresh forwards force option to opsClient.getEvents");
assert.match(hookSource, /setRefreshing\(true\)/, "refreshing state is tracked separately from initial loading");
assert.match(hookSource, /return \{ events, loading, refreshing, error, lastUpdatedLabel, refresh \}/, "hook exposes refreshing and lastUpdatedLabel");

assert.match(tabSource, /Last updated:/, "Activity Log renders the last updated label");
assert.match(tabSource, /eventsState\.refresh\(\{ force: true \}\)/, "manual refresh forces a cache-bypassing events load");
assert.match(tabSource, /eventsState\.refreshing \? "Refreshing\.\.\." : "Refresh"/, "manual refresh button reflects refreshing state");
assert.match(tabSource, /hasRealEvents && filteredEvents\.length === 0/, "filtered empty state still requires real events");
