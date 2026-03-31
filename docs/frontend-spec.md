# Frontend Specification

## Purpose

This frontend is a **disposable demo shell** for the Geospatial Forecasting backend. Its job is to present forecast outputs clearly on a map and through lightweight operator-facing UI panels. It is **not** a second backend, **not** an OpenRemote fork, and **not** a place to reimplement plume logic.

The frontend must consume backend outputs only through HTTP/JSON contracts. During initial implementation, it must support a **mock-data mode first** so the UI can be built and stabilized without waiting on additional backend work.

---

## Scope

Phase 1 includes exactly one serious page:

* a top bar
* a left sidebar for scenario and display controls
* a large central MapLibre-based map canvas
* a right-hand detail drawer
* summary metric cards
* a lightweight status/timeline area

Phase 1 does **not** include:

* authentication
* multi-page routing complexity
* user accounts
* backend business logic in the frontend
* true uncertainty analytics
* real time streaming
* advanced time-series playback
* OpenRemote embedding or plugin packaging

---

## Design Principles

The frontend must follow these principles:

1. **HTTP boundary only**

   * No Python imports.
   * No direct reuse of backend internals.
   * No logic copied from forecast generation code.

2. **Mock-first, live-second**

   * UI should work fully against local mock JSON fixtures.
   * Switching to live API mode should only swap the data source layer.

3. **Single-page seriousness**

   * This should feel like a real operator dashboard, not a toy mockup.
   * But it should remain intentionally narrow in scope.

4. **Map-first experience**

   * The map is the center of the product.
   * Panels exist to support interpretation, not dominate the screen.

5. **OpenRemote-inspired, not OpenRemote-copied**

   * Mimic the feel: operational, spatial, restrained.
   * Do not reproduce OpenRemote structure, architecture, or product assumptions.

---

## Visual Direction

### Theme

* light theme
* restrained palette
* thin borders
* subtle shadows only where useful
* lots of whitespace
* compact labels
* operational rather than decorative styling

### UI Character

The page should feel like:

* a clean geospatial operations console
* readable at a glance
* sparse but not empty
* modern and neutral

### Color Guidance

Use color conservatively:

* neutral backgrounds for frame and panels
* blue/teal for informational state
* amber for cautionary placeholder state
* red only for high concentration or alert-like emphasis
* green only for healthy API/status indicators

Avoid loud saturation and gradient-heavy styling.

---

## Technical Stack

Recommended stack:

* **Vite**
* **React**
* **TypeScript**
* **MapLibre GL**
* plain CSS or simple modular styling

Optional but acceptable later:

* lightweight state management if the page grows
* component library only if it does not fight the visual direction

Avoid introducing large architectural frameworks unless needed.

---

## Proposed Frontend Structure

```text
frontend/
├── index.html
├── .gitignore
├── tsconfig.json
├── tsconfig.node.json
├── package.json
├── vite.config.ts
├── public/
├── src/
│   ├── main.tsx
│   ├── vite-env.d.ts
│   ├── app/
│   │   └── App.tsx
│   ├── pages/
│   │   └── MapPage.tsx
│   ├── components/
│   │   ├── TopBar.tsx
│   │   ├── Sidebar.tsx
│   │   ├── DetailDrawer.tsx
│   │   ├── SummaryCards.tsx
│   │   └── StatusBar.tsx
│   ├── features/
│   │   ├── map/
│   │   │   ├── ForecastMap.tsx
│   │   │   └── layers.ts
│   │   ├── forecast/
│   │   │   ├── forecast.types.ts
│   │   │   └── forecastApi.ts
│   │   ├── scenario/
│   │   │   └── ScenarioControls.tsx
│   │   └── timeline/
│   │       └── TimelineSlider.tsx
│   ├── services/api/
│   │   └── client.ts
│   ├── styles/
│   │   └── theme.css
│   └── mocks/
│       ├── forecast.json
│       ├── summary.json
│       ├── geojson.json
│       ├── raster-metadata.json
│       └── capabilities.json
```

This structure is intentionally small. It separates layout, map behavior, forecast data access, and styling without overengineering.

---

## Page Layout

### 1. Top Bar

Purpose:

* global status and context

Contents:

* project title: `Geospatial Forecasting Demo`
* scenario selector
* model badge: `Gaussian Baseline`
* API mode badge: `Mock` or `Live`
* API health indicator
* optional timestamp of currently loaded forecast

Behavior:

* does not contain deep settings
* must remain compact

---

### 2. Left Sidebar

Purpose:

* operator input and display controls

Contents:

* scenario controls
* release/source controls
* wind controls
* threshold toggle(s)
* overlay visibility toggle(s)
* `Run Forecast` action
* optional reset button

Important rule:

* controls may shape request payloads or local display state
* they must **not** compute plume outputs locally

Mock mode:

* controls may switch between prepared fixtures or mutate request payloads before a mock response is loaded

Live mode:

* controls submit request(s) to backend endpoints

---

### 3. Central Map Canvas

Purpose:

* primary spatial visualization area

Map requirements:

* MapLibre GL from the start
* navigable map with pan and zoom
* source marker visible
* forecast plume overlay visible
* threshold or concentration overlay toggleable
* placeholder timeline control anchored near map area

Layers should support:

* source point
* plume or affected cells overlay
* optional raster-inspired overlay later
* optional grid cell highlighting on hover/select

Phase 1 simplification:

* GeoJSON layer rendering is enough
* raster metadata may be shown in side panels before any richer map rendering is added

---

### 4. Right Detail Drawer

Purpose:

* focused interpretation of whatever the user selected or loaded

Contents:

* selected cell or overlay details
* summary explanation text
* forecast metadata
* source information
* wind information
* uncertainty placeholder section

Important note:

* uncertainty area should be explicitly labeled as placeholder until implemented for real
* do not fake probabilistic meaning

---

### 5. Summary Cards

Purpose:

* quick-glance operational metrics

Required cards:

* wind
* source
* max concentration
* affected cells
* timestamp

Optional later:

* forecast id
* threshold used
* API latency

---

### 6. Status / Timeline Area

Purpose:

* lightweight operational context

Contents:

* API mode
* last refresh/run status
* timeline placeholder or basic slider control

Important rule:

* timeline in phase 1 is UI scaffolding only
* do not imply real time-sequence forecasting unless implemented

---

## Data Modes

### Mode A: Mock Mode

This is the required first mode.

Mock files:

* `mocks/forecast.json`
* `mocks/summary.json`
* `mocks/geojson.json`
* `mocks/raster-metadata.json`
* `mocks/capabilities.json`

Goals of mock mode:

* unblock UI implementation
* lock component contracts
* validate screen structure and interactions
* keep frontend moving independently of backend changes

### Mode B: Live API Mode

This mode should be enabled once the shell is stable.

Expected initial backend usage:

* `GET /health`
* `GET /capabilities`
* `POST /forecast`
* `GET /forecast/{forecast_id}`
* `GET /forecast/{forecast_id}/summary`
* `GET /forecast/{forecast_id}/geojson`
* `GET /forecast/{forecast_id}/raster-metadata`

Design rule:

* switching between mock and live should happen in the API client layer, not across the whole component tree

---

## Frontend Domain Types

The frontend should define small TypeScript types that mirror consumed backend shapes where needed, but only for transport and rendering.

Examples:

* forecast reference
* forecast summary
* geojson feature collection
* raster metadata summary
* capabilities response
* scenario control state

Do not attempt to recreate full backend schema complexity unless the UI needs it.

---

## API Client Rules

Create a small `client.ts` and `forecastApi.ts` layer.

Responsibilities:

* encapsulate fetch calls
* switch between mock and live providers
* normalize transport errors into simple UI-safe messages

Must not:

* contain business logic
* compute forecast values
* transform the frontend into an orchestration engine

---

## Map Rendering Rules

Initial rendering target:

* render GeoJSON output on MapLibre cleanly
* show source marker clearly
* support selection of visible cells/features if practical

Initial behavior is allowed to be simple.

Do not block frontend progress on:

* contour polygon generation
* advanced interpolation
* animated plume playback
* probabilistic layer blending

Those are later enhancements.

---

## Interaction Model

### Core interactions for phase 1

1. user loads page
2. top bar shows mock/live status
3. map renders default base map
4. mock forecast or seeded scenario loads
5. summary cards populate
6. geojson overlay appears on map
7. selecting a feature updates the right-hand detail drawer
8. sidebar controls can trigger a new run or swap scenario fixture

That is enough for phase 1.

---

## Error and Empty States

The frontend must handle these visibly and simply:

* API unavailable
* failed forecast request
* missing GeoJSON
* empty selection state
* mock file load failure

Use plain operational language.

Examples:

* `API unavailable`
* `No forecast loaded`
* `GeoJSON layer missing`
* `Select a map cell to inspect details`

Avoid dramatic UX language.

---

## Non-Goals

The frontend must not become:

* a second forecasting engine
* an analytics notebook
* a workflow platform
* an OpenRemote clone
* a complex design system exercise

---

## Acceptance Criteria for Phase 1

The frontend shell is considered successful when all of the following are true:

1. `frontend/` exists with Vite + React + TypeScript structure.
2. One page renders cleanly.
3. MapLibre map loads successfully.
4. Mock mode works without backend dependency.
5. Summary cards display realistic forecast information.
6. Right-hand drawer displays selected feature/details.
7. Left sidebar drives request/display state without local forecast logic.
8. API client layer can be switched toward live backend mode.
9. Styling feels operational, sparse, and coherent.
10. No frontend code imports Python or duplicates backend plume logic.

---

## Recommended Build Order

1. Create frontend app shell and layout
2. Add theme and frame styling
3. Add TopBar, Sidebar, SummaryCards, DetailDrawer, StatusBar
4. Add MapLibre map container
5. Load mock JSON fixtures
6. Render GeoJSON overlay on the map
7. Wire mock selection and detail interactions
8. Add API client abstraction
9. Introduce live backend mode behind the same interface
10. Only then consider frontend CI/build checks

---

## Follow-On Docs After Frontend Spec

Once this spec is implemented, the next docs should be:

* `docs/openremote-integration.md`
* `docs/demo-scenarios.md`

Those documents must describe the real implemented state, not imagined future integration.

---

## Final Guardrails

Do not:

* re-architect the backend
* reimplement Gaussian plume logic in TypeScript
* claim uncertainty support before it exists
* claim OpenRemote compatibility beyond the provisional payload adapter

Do:

* keep the shell honest
* keep the layers thin
* keep the page spatially centered
* make the UI good enough to demonstrate the backend clearly
