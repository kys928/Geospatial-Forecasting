# Final System Overview

## Purpose

This document gives a high-level overview of the Geospatial Forecasting system. It explains what the application does, how the main parts fit together, and how forecast data moves through the system.

The goal is to make the system understandable without needing to read the source code first.

## Project Challenge

The project focuses on hazardous plume forecasting. In an environmental incident, an operator needs to understand where a plume may spread, how serious the predicted area is, and what information supports the forecast.

A raw model output is not enough by itself. A grid of predicted concentration values needs to become something useful: a map overlay, forecast frames, summary information, runtime status, and decision-support context.

The system is built around that problem. It turns model and dataset outputs into a usable forecasting dashboard.

## System Purpose

The Geospatial Forecasting system is a web application for viewing and explaining plume forecasts.

At a high level, it can:

- run or serve ConvLSTM-based plume forecasts;
- display predicted plume spread on a map;
- show forecast frames over time;
- load dataset playback scenarios;
- generate forecast summaries;
- provide decision-support context;
- show model and runtime status;
- support local LLM-based explanation;
- manage model checkpoints and training operations.

The system is not just a model script. It is an application around the model, with a backend, frontend, runtime services, model operations, dataset support, and explanation support.

## System Architecture

The system is split into four main layers:

```text
Data and model assets
        |
        v
Backend runtime services
        |
        v
API layer
        |
        v
Frontend dashboard
```

The data and model asset layer contains the dataset, model checkpoint, local LLM file, and runtime configuration.

The backend runtime services load data, run forecasts, manage model state, prepare outputs, and create forecast context.

The API layer exposes forecast, session, dataset, runtime, and decision-support endpoints.

The frontend dashboard presents the forecast to the user through maps, panels, frames, summaries, and status views.

## Backend

The backend is the central runtime layer of the system. It is responsible for running the application logic and serving forecast data to the frontend.

The backend handles:

- configuration loading;
- forecast execution;
- session forecast storage;
- ConvLSTM backend access;
- dataset playback access;
- forecast summaries;
- raster and frame outputs;
- GeoJSON outputs;
- runtime metadata;
- model registry status;
- training and retraining operations;
- decision-support context;
- local LLM explanation requests.

The backend keeps the different parts of the system separated. Forecasting, dataset playback, explanation, and training are related, but they are not the same process.

## Frontend

The frontend is the user-facing part of the system. It turns backend outputs into a usable dashboard.

The frontend provides:

- a forecast map;
- plume overlays;
- forecast frame controls;
- forecast summaries;
- model and runtime status panels;
- decision-support panels;
- training and model operation views;
- model registry browsing.

The frontend does not create forecast data by itself. It asks the backend for forecast results, metadata, and status information, then presents that information in a readable way.

## Forecasting Workflow

The main forecasting workflow starts with input data and ends with forecast outputs that can be viewed by the user.

A simplified forecast flow is:

```text
Input window or runtime context
        |
        v
ConvLSTM forecast backend
        |
        v
Predicted plume frames
        |
        v
Raster, GeoJSON, summary, and metadata outputs
        |
        v
Frontend map and panels
```

The ConvLSTM model is used because plume movement is both spatial and temporal. The model needs to understand how values change across a grid over time.

The predicted output can then be converted into user-facing formats such as map overlays, raster frames, and summary statistics.

## Dataset Playback

The system includes dataset playback support. This allows the application to load prepared plume scenarios from a dataset and show them through the same dashboard style.

Dataset playback is useful for:

- demonstrations;
- development;
- testing;
- showing realistic forecast-like scenarios without live sensor input;
- validating frontend and backend behavior.

Dataset playback is part of the system, but it is not the same as active model forecasting. It is a controlled way to present known scenario data.

## Local LLM Explanation

The system includes a local LLM explanation layer. The LLM is used to explain forecast context in human-readable language.

The LLM can help produce:

- a short summary;
- a risk interpretation;
- a recommendation;
- an uncertainty note.

The LLM does not create the forecast. The forecast comes from the forecasting system. The LLM only explains structured forecast information that already exists.

This keeps the model output and the explanation layer separated.

## Model Operations

The system also includes model operation features. These features support the lifecycle around the forecasting model.

The model operation layer includes:

- model registry information;
- active model status;
- checkpoint paths;
- training jobs;
- automatic adaptation job handling;
- model version browsing;
- operational status views.

This makes the application more than a static forecast viewer. It can also show and manage the state of the model system around the forecast workflow.

Forecast serving and training are kept separate. A forecast run uses the currently available model state, while training operations prepare or evaluate future model candidates.

## Main System Flow

The system can be understood as one connected flow:

```text
Dataset / runtime input / model checkpoint
        |
        v
Backend forecast runtime
        |
        v
Forecast result
        |
        v
Stored session output
        |
        v
API response
        |
        v
Frontend dashboard
        |
        v
Operator-facing map, summary, status, and explanation
```

Each step has a clear responsibility. The backend creates and stores forecast outputs. The API exposes them. The frontend displays them. The LLM explains them. Training operations manage future model updates.

## Current Final System

The final system represents a cleaned and structured version of the project. It brings together the main application parts into one understandable architecture:

- backend API;
- frontend dashboard;
- ConvLSTM forecasting;
- dataset playback;
- forecast summaries;
- raster and frame outputs;
- local LLM explanation;
- model registry and training operations;
- runtime setup support.

The final system should be understood as a complete project architecture and demonstration platform. It shows how hazardous plume forecasting can be turned into an operator-facing application.

## Summary

The Geospatial Forecasting system is a forecasting and decision-support application for hazardous plume scenarios.

Its main purpose is to take model and dataset outputs and make them useful through a backend API, frontend dashboard, map visualization, forecast summaries, runtime status, model operations, and local explanation support.

The core idea is simple:

```text
The model produces forecast data.
The backend prepares and stores forecast outputs.
The API exposes the outputs.
The frontend presents the result.
The LLM explains existing forecast context.
Model operations manage the model lifecycle.
```
