# Model Modes, Training, and Provenance

## Purpose

This document explains how the Geospatial Forecasting system describes its forecast outputs and model operations.

The system can show different kinds of forecast-like results. Some results come from active ConvLSTM inference. Some come from dataset playback. Some may come from fallback logic. Some are explanation outputs produced after the forecast already exists.

These things must not be mixed together.

This document explains the difference between them and how the system keeps the source of each result visible.

## Why This Document Exists

A plume map can look convincing even when it comes from a demo scenario, a fallback model, or an active model using a prepared dataset window.

That is dangerous if the system does not explain what the user is actually seeing.

The main rule is:

```text
A forecast output must say where it came from.
```

The system uses runtime metadata and provenance fields to make that clear.

## The Main Model Modes

The system has several model and runtime modes. They can look similar in the frontend, but they mean different things.

```text
active_convlstm
dataset_window
dataset_playback
fallback
demo_backend
temporary_substitution
llm_explanation
```

These are not just labels for the UI. They describe how the result was produced.

## Active ConvLSTM

Active ConvLSTM means the configured ConvLSTM backend produced the forecast result.

This is the main model-backed forecast path.

In this mode, the system is using the ConvLSTM forecasting backend rather than a playback scenario or fallback model.

A forecast can be treated as active ConvLSTM only when the metadata supports it. The backend name, model family, prediction engine, and fallback status all matter.

A simple version:

```text
ConvLSTM backend is used
fallback is not used
demo backend is not used
dataset playback is not the output source
temporary substitution is not active
```

The important thing is that active ConvLSTM means the model actually produced the forecast. It does not automatically mean the input came from live sensors.

## ConvLSTM With Dataset-Window Input

The system can also run the ConvLSTM model using a prepared dataset window as the input seed.

This is an important middle case.

The ConvLSTM still runs, so it is not the same as pure dataset playback. But the input did not come from fresh live observations. It came from a prepared dataset window.

This should be described honestly as:

```text
ConvLSTM inference using a dataset-window input
```

It should not be described as:

```text
live active forecasting from real-time sensor input
```

This distinction matters because the model execution is real, but the input source is controlled.

## Dataset Playback

Dataset playback is used for demonstration, testing, and showing realistic plume scenarios from prepared data.

Dataset playback is not active live forecasting.

It is useful because it lets the frontend and backend show a realistic scenario even when no live sensor stream is available. It also helps test forecast panels, map overlays, frame controls, decision-support context, and raster/GeoJSON outputs.

The correct interpretation is:

```text
This is a prepared scenario being replayed through the system.
```

It is not:

```text
The model has predicted a new live event.
```

Dataset playback is valuable, but it must stay labeled as playback.

## Fallback Mode

Fallback mode is used when the normal model-backed path cannot be used or when the system chooses a fallback backend.

In this project, the fallback path exists so that the application can still return a usable result instead of failing completely.

A fallback result should not be presented as the same thing as active ConvLSTM output.

The correct interpretation is:

```text
The system produced a fallback result.
```

The wrong interpretation is:

```text
The active ConvLSTM model successfully generated this forecast.
```

Fallback output can be useful for robustness, but it has to be labeled clearly.

## Demo Backend and Temporary Substitution

The system also has development or substitution paths.

A demo backend is useful for testing and local development. It should not be treated as a real forecasting model.

Temporary substitution is a case where a simpler or replacement prediction engine is being used instead of the intended active model path.

These modes are useful during development, but they should be obvious in metadata so they do not get mistaken for the main forecast path.

## LLM Explanation Mode

The local LLM explanation layer is separate from forecasting.

The LLM does not create plume forecasts. It reads structured forecast context and turns it into a more understandable explanation.

It can help produce:

* a summary;
* a risk interpretation;
* a recommendation;
* an uncertainty note.

The LLM is therefore an explanation layer, not a forecasting layer.

The correct flow is:

```text
Forecast result already exists
        |
        v
Backend prepares forecast context
        |
        v
LLM explains that context
```

The wrong flow would be:

```text
LLM invents forecast
        |
        v
System treats it as model output
```

That is not how this system should work.

## Provenance

Provenance means the system records where a forecast result came from.

This is the part that keeps the application honest.

Important provenance fields include:

```text
forecast_source
model_id
model_family
model_backend
checkpoint_path
inference_mode
fallback_used
fallback_reason
temporary_model_substitution
prediction_engine
input_source
input_window_source
output_source
dataset_playback_enabled
active_registry_model_id
generated_at
stale_model
active_model_mismatch
current_active_model_id
artifact_model_id
```

These fields help answer practical questions:

```text
Who produced this output?
Which model was used?
Was this a fallback?
Was the input a dataset window?
Was dataset playback active?
Which checkpoint does this result belong to?
Is the artifact aligned with the active registry model?
```

Without provenance, a map overlay is just a picture. With provenance, the system can explain what kind of result the user is seeing.

## Runtime Mode Classification

Runtime mode is the system’s simplified interpretation of the metadata.

The backend can inspect metadata and classify the output into a practical mode such as:

```text
fallback
demo_backend
temporary_substitution
dataset_window
active_convlstm
unknown
```

This is useful because raw metadata can be noisy. The frontend and decision-support layer need a simpler way to understand the result.

A simplified classification order looks like this:

```text
If fallback is used:
    mode = fallback

Else if demo backend is used:
    mode = demo_backend

Else if temporary substitution is used:
    mode = temporary_substitution

Else if dataset-window input is used:
    mode = dataset_window

Else if ConvLSTM backend/model/engine is detected:
    mode = active_convlstm

Else:
    mode = unknown
```

This order matters. For example, a forecast should not be called active ConvLSTM if it was actually served by fallback logic.

## Model Registry

The model registry keeps track of model records and their operational state.

The registry is important because the application may have more than one checkpoint available:

* an active model;
* candidate models;
* older models;
* rejected candidates;
* models with missing checkpoint files;
* models that cannot be activated because of compatibility problems.

The active model is the one used for serving when the system resolves the active registry path.

Candidate models are not automatically the active model. They are possible future models that still need evaluation or approval.

This prevents a training run from silently replacing the serving model.

## Automatic Training and Adaptation

The system includes an automatic adaptation direction for ConvLSTM training.

This does not mean the system blindly trains and replaces itself whenever new data appears.

The adaptation workflow is controlled.

A simplified version looks like this:

```text
adaptation buffer
        |
        v
readiness checks
        |
        v
dataset manifest
        |
        v
training run
        |
        v
candidate checkpoint
        |
        v
promotion policy
        |
        v
operator / Ops decision
        |
        v
active model only if approved or policy allows it
```

The adaptation buffer stores accepted samples that may be useful for future retraining.

Readiness checks decide whether training should even be considered. They can check things like sample availability, checkpoint compatibility, disk space, and whether conflicting jobs are already running.

The dataset manifest combines reference data and accepted adaptation samples into a training set.

The trainer can produce a new candidate checkpoint.

That candidate is then evaluated before it becomes active.

## Readiness Checks

Readiness checks are not training.

They are a safety gate before training.

A readiness check can say whether the system appears ready for adaptation work, but it should not start a training job by itself.

This is useful because training can be expensive, slow, and risky if the input data or model state is not suitable.

The system should be able to say:

```text
The system is ready.
The system is not ready.
The system may be ready, but there are warnings.
```

That gives the operator a reason to investigate before launching training.

## Training Jobs

Training jobs are separate from forecast serving.

Forecast serving uses the current available model state. Training jobs prepare possible future model candidates.

That separation matters.

A forecast should not stop being served just because training is running. The current active model should continue to serve forecasts while adaptation work happens separately.

The system can support training through worker-style execution. The worker handles queued work and writes status, while the API/control service exposes status and job information.

## Candidate Checkpoints

A training run can produce a candidate checkpoint.

A candidate checkpoint is not automatically trusted.

It needs to be registered, evaluated, and checked before it can become active.

The candidate may be better than the active model, worse than the active model, incomplete, incompatible, or missing required metrics.

A candidate can be useful even if it is not promoted, because it records an experiment and gives the operator something to compare.

## Promotion Policy

Promotion policy is the rule layer between a candidate model and the active model.

The promotion policy should look at candidate quality before activation. For example, it can check whether the candidate improves important metrics and whether it avoids unacceptable regressions.

The key point is:

```text
training completion is not the same as model promotion
```

A model can finish training and still not become active.

This  prevents the system from replacing a stable serving model with a weaker candidate.

## Operator Control

Some actions are safe read-only checks. Other actions mutate model state.

Readiness checks are read-only.

Candidate evaluation is also read-only.

Approval, rejection, policy application, activation, and checkpoint-file deletion are state-changing actions.

That difference should be clear in the UI and API.

A practical mental model:

```text
Check readiness = look only
Evaluate candidate = look only
Apply policy = may change registry state
Approve candidate = may activate candidate
Reject candidate = changes candidate status
Delete checkpoint file = removes file but keeps metadata/history
```

This separation helps prevent accidental model changes.

## Final Position

The system should be understood as a forecast application with controlled model operations.

Forecasting, playback, fallback, explanation, training, and promotion are connected, but they are not the same thing.

The model can produce forecasts.

Dataset playback can show prepared scenarios.

Fallback can keep the app usable when the main model path is not available.

The LLM can explain forecast context.

Automatic adaptation can prepare candidate checkpoints.

Only a controlled promotion path should change which model is active.
