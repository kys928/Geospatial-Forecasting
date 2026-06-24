# OpenRemote Integration Guide

## Purpose

This document explains how the Geospatial Forecasting system can be integrated with OpenRemote in a way that fits OpenRemote’s existing structure.

The goal is to describe a realistic integration path, not just a loose connection between two separate dashboards. OpenRemote already has a Manager application, reusable UI components, asset views, map components, dashboard tooling, REST services, and backend manager services. The Geospatial Forecasting system should connect into that structure as a forecasting and decision-support service.

The main idea:

```text
OpenRemote manages assets, users, dashboards, and operational context.
Geospatial Forecasting produces plume forecasts, forecast artifacts, and explanation context.
The integration connects those outputs back into OpenRemote assets and Manager UI views.
```

## OpenRemote Structure Relevant to This Project

OpenRemote is structured as a platform with a backend Manager service and a separate UI workspace.

The relevant parts for this project are:

```text
openremote/
  manager/
    src/
  ui/
    app/
      manager/
      insights/
      shared/
      storybook/
      swagger/
    component/
      or-map/
      or-dashboard-builder/
      or-services/
      rest/
      ...
```

This matters because the Geospatial Forecasting system should not be treated as a random external page. A better integration would use the same structure OpenRemote already uses:

- backend service communication through the Manager/API layer;
- asset and attribute updates through OpenRemote-compatible payloads;
- map display through OpenRemote map components;
- forecast panels through a Manager UI page or custom console;
- dashboard display through existing dashboard or attribute components.

The natural frontend target is the Manager UI. The natural backend target is the OpenRemote Manager/service API boundary.

## Proposed Integration Shape

The best integration shape is a hybrid approach:

```text
Geospatial Forecasting Service
        |
        | service registration / heartbeat
        v
OpenRemote Manager
        |
        | forecast assets / attributes / predicted datapoints
        v
OpenRemote asset model
        |
        | Manager UI page / map / dashboard components
        v
Operator view
```

This means the Geospatial Forecasting system remains responsible for forecast execution, model status, forecast artifacts, and explanation context.

OpenRemote remains responsible for users, realms, assets, dashboards, and operator access.

The integration should not move the ConvLSTM model into OpenRemote. Instead, OpenRemote should see the forecasting system as a service that can publish useful forecast information back into the OpenRemote asset model.

## Manager UI Integration Proposal

The cleanest UI direction is to add a Geospatial Forecasting page or console inside the OpenRemote Manager UI.

This page could live conceptually as a Manager UI feature, for example:

```text
ui/app/manager/src/pages/geospatial/
```

or as a project-specific Manager extension if OpenRemote prefers not to place it inside the core Manager pages.

The page would use OpenRemote’s existing UI style and component ecosystem instead of duplicating the whole Geospatial frontend. The likely UI structure would be:

```text
Left panel:
  site / asset / sensor selection

Center:
  map view with plume overlay

Right panel:
  forecast summary
  runtime status
  model status
  decision-support explanation
```

The map layer could use OpenRemote’s map components and load forecast GeoJSON or raster-derived overlays from the Geospatial Forecasting API.

The panels could read forecast attributes already published into OpenRemote assets, or they could call the Geospatial Forecasting API directly depending on the chosen integration depth.

## Service Registration and Heartbeat

The current Geospatial Forecasting repository contains an optional service registration path for OpenRemote.

The purpose of service registration is to let OpenRemote know that the Geospatial Forecasting service exists and is available.

The service registration payload contains information such as:

```text
serviceId
version
icon
label
homepageUrl
status
```

The registration can be global or non-global depending on configuration.

After registration, the service can send heartbeat-style updates. The heartbeat keeps the service lifecycle visible to OpenRemote. If the service disappears or stops responding, OpenRemote can treat the forecast service as unavailable instead of silently assuming forecast output is still valid.

The lifecycle is:

```text
Geospatial service starts
        |
        v
register with OpenRemote
        |
        v
store returned service instance id
        |
        v
send heartbeat while running
        |
        v
deregister or stop heartbeat on shutdown
```

## HTTP Publishing Path

The second part of the integration is HTTP publishing.

HTTP publishing is how forecast outputs can be written back into OpenRemote-compatible assets and attributes.

The current publishing path supports several kinds of OpenRemote-facing objects:

```text
Hazard source asset
Forecast run asset
Forecast asset attributes
Sensor asset
Sensor observation attributes
Forecast zone asset
Predicted concentration datapoints
```

This gives OpenRemote a way to receive forecast results without needing to run the forecasting model itself.

A typical publishing flow would look like this:

```text
ConvLSTM forecast result
        |
        v
Geospatial forecast summary and metadata
        |
        v
OpenRemote payload builder
        |
        v
HTTP result sink
        |
        v
OpenRemote asset / attribute / predicted datapoint update
```

This keeps forecast execution separate from OpenRemote asset display.

## Forecast Run Asset

A forecast run asset represents one forecast execution.

It can contain information such as:

- forecast run id;
- session id;
- backend name;
- model name;
- model version;
- run status;
- issued time;
- source asset id;
- summary statistics;
- alert level;
- plume footprint;
- centroid;
- bounding box;
- GeoJSON URL;
- raster metadata;
- grid specification;
- scenario snapshot;
- execution metadata.

This gives OpenRemote a structured object that can be displayed, inspected, or linked to other operational assets.

## Hazard Source Asset

A hazard source asset represents the source of the plume scenario.

It can contain:

- source id;
- location;
- pollutant type;
- release rate;
- release height;
- source status;
- last observation time;
- scenario metadata.

This allows the forecast to remain connected to the physical or simulated incident source that produced the plume.

## Forecast Attributes

In some deployments, creating a new forecast asset for every forecast may be too heavy. In that case, the integration can publish forecast attributes onto an existing configured forecast asset.

This path is useful when OpenRemote should keep one stable forecast asset and update its current state.

The published attributes can include:

- forecast id;
- issued time;
- summary;
- GeoJSON;
- raster metadata;
- runtime metadata;
- provenance information.

This is useful for dashboards because a dashboard can keep reading the same asset while the attribute values change over time.

## Forecast Zones and Predicted Datapoints

The integration can also represent affected areas as forecast zone assets.

A forecast zone asset can contain:

- zone id;
- zone name;
- zone geometry;
- zone type;
- latest forecast run id;
- risk level.

For time-series style forecast output, predicted concentration values can be written as predicted datapoints. This fits cases where OpenRemote should display a forecast trend over time instead of only a static map.

## How the Manager UI Could Display the Forecast

A Geospatial Forecasting page inside Manager UI could combine OpenRemote data and Geospatial forecast output in one view.

A practical layout would be:

```text
Asset/site selector
        |
        v
Forecast action panel
        |
        v
Map with plume overlay
        |
        v
Forecast summary and runtime status
        |
        v
Decision-support explanation
```

The map should display the plume footprint or forecast frames.

The side panel should show the forecast status and important context, such as:

- which forecast run is being shown;
- which backend produced it;
- which model or checkpoint was used;
- whether the output is active model output, fallback output, or dataset playback;
- when it was issued;
- what the main forecast summary says.

This keeps the operator view useful without requiring the operator to understand the model internals.

## Final Position

The Geospatial Forecasting system fits OpenRemote best as a forecasting service with a native Manager UI surface.

The backend service registers itself, keeps a heartbeat, produces plume forecast outputs, and publishes those outputs into OpenRemote-compatible assets and attributes.

The Manager UI can then display the result using OpenRemote’s own map, dashboard, asset, and service components.
