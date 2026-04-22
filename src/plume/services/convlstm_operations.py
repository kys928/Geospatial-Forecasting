from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION, CONVLSTM_NORMALIZATION_MODE
from plume.models.convlstm_training import load_best_checkpoint_summary, load_run_summary


OPERATIONAL_PHASES = {
    "idle",
    "collecting",
    "ready_for_retraining",
    "dataset_snapshotting",
    "training",
    "evaluating_candidate",
    "promotion_decision",
    "deploying_model",
    "candidate_rejected",
    "monitoring",
}
MODEL_STATUSES = {"candidate", "approved", "active", "rejected", "archived"}


@dataclass(frozen=True)
class OperationalState:
    phase: str = "idle"
    active_model_id: str | None = None
    active_model_path: str | None = None
    candidate_model_id: str | None = None
    candidate_model_path: str | None = None
    buffered_new_sample_count: int = 0
    last_retrain_time: str | None = None
    current_run_id: str | None = None
    last_promotion_result: dict[str, object] | None = None
    latest_warning_or_error: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in OPERATIONAL_PHASES:
            raise ValueError(f"Unsupported operational phase: {self.phase}")
        if self.buffered_new_sample_count < 0:
            raise ValueError("buffered_new_sample_count must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OperationalState":
        return cls(
            phase=str(payload.get("phase", "idle")),
            active_model_id=_optional_str(payload.get("active_model_id")),
            active_model_path=_optional_str(payload.get("active_model_path")),
            candidate_model_id=_optional_str(payload.get("candidate_model_id")),
            candidate_model_path=_optional_str(payload.get("candidate_model_path")),
            buffered_new_sample_count=int(payload.get("buffered_new_sample_count", 0)),
            last_retrain_time=_optional_str(payload.get("last_retrain_time")),
            current_run_id=_optional_str(payload.get("current_run_id")),
            last_promotion_result=_optional_dict(payload.get("last_promotion_result")),
            latest_warning_or_error=_optional_str(payload.get("latest_warning_or_error")),
        )


@dataclass(frozen=True)
class RetrainingPolicy:
    retraining_enabled: bool = True
    retraining_min_new_samples: int = 1
    retraining_manual_only: bool = False
    retraining_min_interval_seconds: int | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    promotion_enabled: bool = True
    promotion_require_contract_match: bool = True
    promotion_metric_name: str = "val_mse"
    promotion_metric_direction: str = "min"
    promotion_min_improvement: float = 0.0
    promotion_max_regression_support_iou: float | None = None
    promotion_max_regression_centroid: float | None = None
    promotion_manual_approval_required: bool = False


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {"active_model_id": None, "previous_active_model_id": None, "models": [], "events": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Model registry must decode to JSON object: {self.path}")
        payload.setdefault("models", [])
        payload.setdefault("events", [])
        payload.setdefault("active_model_id", None)
        payload.setdefault("previous_active_model_id", None)
        return payload

    def save(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)

    def find_record(self, model_id: str) -> dict[str, object] | None:
        payload = self.load()
        for record in payload["models"]:
            if record.get("model_id") == model_id:
                return dict(record)
        return None


def evaluate_retraining_readiness(
    *,
    state: OperationalState,
    policy: RetrainingPolicy,
    now: datetime | None = None,
    manual_trigger: bool = False,
) -> dict[str, object]:
    current_time = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    ready = True

    if not policy.retraining_enabled:
        ready = False
        reasons.append("retraining_disabled")
    if policy.retraining_manual_only and not manual_trigger:
        ready = False
        reasons.append("manual_only_requires_override")
    if state.buffered_new_sample_count < policy.retraining_min_new_samples:
        ready = False
        reasons.append("insufficient_new_samples")
    if policy.retraining_min_interval_seconds is not None and state.last_retrain_time:
        last_retrain = datetime.fromisoformat(state.last_retrain_time)
        elapsed = (current_time - last_retrain).total_seconds()
        if elapsed < policy.retraining_min_interval_seconds:
            ready = False
            reasons.append("min_interval_not_elapsed")

    if manual_trigger and policy.retraining_enabled:
        ready = True
        reasons = ["manual_override"]

    return {"should_trigger": ready, "reasons": reasons or ["ready"], "manual_trigger": manual_trigger}


def register_candidate_from_run(
    *,
    registry: ModelRegistry,
    run_dir: str | Path,
    run_id: str | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    run_path = Path(run_dir)
    run_summary = load_run_summary(run_path)
    checkpoint_summary = load_best_checkpoint_summary(run_path)

    checkpoint_path = checkpoint_summary.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("Best checkpoint summary missing non-empty checkpoint_path")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint does not exist: {checkpoint}")

    policy = run_summary.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Run summary missing policy object")

    metric_name = checkpoint_summary.get("best_metric_name")
    metric_value = checkpoint_summary.get("best_metric_value")
    if not isinstance(metric_name, str):
        raise ValueError("Best checkpoint summary missing best_metric_name")
    if not isinstance(metric_value, (float, int)):
        raise ValueError("Best checkpoint summary missing numeric best_metric_value")

    record_id = model_id or f"candidate_{run_path.name}"
    now = _utc_now_iso()
    record = {
        "model_id": record_id,
        "path": str(checkpoint),
        "created_from_run_dir": str(run_path),
        "run_id": run_id or run_path.name,
        "status": "candidate",
        "contract_version": policy.get("contract_version"),
        "target_policy": policy.get("target_policy"),
        "normalization_mode": policy.get("normalization_mode"),
        "checkpoint_metric": {"name": metric_name, "value": float(metric_value)},
        "plume_metrics": _extract_plume_metrics(run_summary.get("final_validation_metrics")),
        "timestamp": now,
        "parent_active_model_id": None,
    }

    payload = registry.load()
    if any(item.get("model_id") == record_id for item in payload["models"]):
        raise ValueError(f"Model id already exists in registry: {record_id}")
    payload["models"].append(record)
    payload["events"].append(
        {"timestamp": now, "event_type": "candidate_registered", "model_id": record_id, "run_id": record["run_id"]}
    )
    registry.save(payload)
    return record


def evaluate_promotion(
    *,
    candidate_record: dict[str, object],
    active_record: dict[str, object] | None,
    policy: PromotionPolicy,
) -> dict[str, object]:
    reasons: list[str] = []
    comparisons: dict[str, object] = {}

    if not policy.promotion_enabled:
        return {"approved": False, "reasons": ["promotion_disabled"], "comparisons": comparisons}
    if policy.promotion_manual_approval_required:
        return {"approved": False, "reasons": ["manual_approval_required"], "comparisons": comparisons}

    if candidate_record.get("status") not in {"candidate", "approved"}:
        reasons.append("candidate_status_invalid")
    metric = _checkpoint_metric(candidate_record)
    if metric["name"] != policy.promotion_metric_name:
        reasons.append("promotion_metric_name_mismatch")

    if policy.promotion_require_contract_match:
        if candidate_record.get("contract_version") != CONVLSTM_CONTRACT_VERSION:
            reasons.append("contract_version_mismatch")
        if candidate_record.get("target_policy") != "plume_only":
            reasons.append("target_policy_mismatch")
        if candidate_record.get("normalization_mode") != CONVLSTM_NORMALIZATION_MODE:
            reasons.append("normalization_mode_mismatch")

    candidate_value = metric["value"]
    if active_record is not None:
        active_metric = _checkpoint_metric(active_record)
        comparisons["active_metric"] = active_metric
        comparisons["candidate_metric"] = metric
        delta = float(candidate_value) - float(active_metric["value"])
        comparisons["metric_delta"] = delta
        if policy.promotion_metric_direction == "min":
            improvement = float(active_metric["value"]) - float(candidate_value)
            if improvement < policy.promotion_min_improvement:
                reasons.append("insufficient_improvement")
        elif policy.promotion_metric_direction == "max":
            improvement = float(candidate_value) - float(active_metric["value"])
            if improvement < policy.promotion_min_improvement:
                reasons.append("insufficient_improvement")
        else:
            reasons.append("unsupported_metric_direction")

        _validate_plume_regressions(
            reasons=reasons,
            candidate=candidate_record,
            active=active_record,
            max_support_iou=policy.promotion_max_regression_support_iou,
            max_centroid=policy.promotion_max_regression_centroid,
        )

    approved = not reasons
    return {
        "approved": approved,
        "reasons": reasons or ["approved"],
        "comparisons": comparisons,
        "candidate_model_id": candidate_record.get("model_id"),
        "active_model_id": None if active_record is None else active_record.get("model_id"),
    }


def activate_approved_model(*, registry: ModelRegistry, model_id: str) -> dict[str, object]:
    payload = registry.load()
    models = payload["models"]
    record = next((m for m in models if m.get("model_id") == model_id), None)
    if record is None:
        raise ValueError(f"Unknown model id: {model_id}")
    if record.get("status") != "approved":
        raise ValueError("Only approved candidate models may be activated")

    path = Path(str(record.get("path")))
    if not path.exists():
        raise FileNotFoundError(f"Approved model artifact missing: {path}")
    if record.get("contract_version") != CONVLSTM_CONTRACT_VERSION:
        raise ValueError("Approved model contract version is incompatible with serving contract")

    previous_active_id = payload.get("active_model_id")
    for item in models:
        if item.get("status") == "active":
            item["status"] = "archived"
    record["status"] = "active"
    payload["previous_active_model_id"] = previous_active_id
    payload["active_model_id"] = model_id
    payload["events"].append(
        {
            "timestamp": _utc_now_iso(),
            "event_type": "model_activated",
            "model_id": model_id,
            "previous_active_model_id": previous_active_id,
        }
    )
    registry.save(payload)
    return {"activated": True, "model_id": model_id, "previous_active_model_id": previous_active_id}


def rollback_to_previous_model(*, registry: ModelRegistry) -> dict[str, object]:
    payload = registry.load()
    previous_id = payload.get("previous_active_model_id")
    if not isinstance(previous_id, str):
        raise ValueError("No previous active model id is available for rollback")
    models = payload["models"]
    target = next((m for m in models if m.get("model_id") == previous_id), None)
    if target is None:
        raise ValueError(f"Previous active model record not found: {previous_id}")

    for item in models:
        if item.get("status") == "active":
            item["status"] = "archived"
    target["status"] = "active"
    payload["active_model_id"] = previous_id
    payload["events"].append({"timestamp": _utc_now_iso(), "event_type": "rollback_performed", "model_id": previous_id})
    registry.save(payload)
    return {"rolled_back": True, "active_model_id": previous_id}


def resolve_active_model_artifact(registry_path: str | Path) -> dict[str, object]:
    registry = ModelRegistry(registry_path)
    payload = registry.load()
    active_model_id = payload.get("active_model_id")
    if not isinstance(active_model_id, str):
        raise ValueError("Model registry has no active model id")
    active_record = next((m for m in payload["models"] if m.get("model_id") == active_model_id), None)
    if active_record is None:
        raise ValueError(f"Active model id not found in registry: {active_model_id}")
    if active_record.get("status") != "active":
        raise ValueError(f"Registry active model record must have status='active', got {active_record.get('status')}")
    checkpoint_path = Path(str(active_record.get("path")))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Active model artifact missing: {checkpoint_path}")
    return {"model_id": active_model_id, "checkpoint_path": str(checkpoint_path), "record": dict(active_record)}


@dataclass
class OperationalEventLog:
    path: Path

    def append(self, *, event_type: str, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": _utc_now_iso(), "event_type": event_type, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass
class OperationalOrchestrator:
    registry: ModelRegistry
    retraining_policy: RetrainingPolicy
    promotion_policy: PromotionPolicy
    event_log: OperationalEventLog

    def process_retraining_cycle(
        self,
        *,
        state: OperationalState,
        manual_trigger: bool,
        train_fn: Callable[[], dict[str, object]],
    ) -> OperationalState:
        readiness = evaluate_retraining_readiness(state=state, policy=self.retraining_policy, manual_trigger=manual_trigger)
        if not readiness["should_trigger"]:
            self.event_log.append(event_type="retraining_not_ready", payload=readiness)
            return OperationalState(**{**state.to_dict(), "phase": "collecting"})

        self.event_log.append(event_type="retraining_ready", payload=readiness)
        self.event_log.append(event_type="retraining_started", payload={"phase": "training"})

        run_payload = train_fn()
        run_dir = run_payload.get("run_dir")
        if not isinstance(run_dir, str):
            raise ValueError("train_fn must return payload with string run_dir")

        candidate = register_candidate_from_run(registry=self.registry, run_dir=run_dir, run_id=run_payload.get("run_id"))
        self.event_log.append(event_type="candidate_registered", payload={"model_id": candidate["model_id"]})

        registry_payload = self.registry.load()
        active_id = registry_payload.get("active_model_id")
        active_record = None
        if isinstance(active_id, str):
            active_record = next((m for m in registry_payload["models"] if m.get("model_id") == active_id), None)

        decision = evaluate_promotion(candidate_record=candidate, active_record=active_record, policy=self.promotion_policy)
        self.event_log.append(
            event_type="promotion_approved" if decision["approved"] else "promotion_rejected",
            payload=decision,
        )

        if decision["approved"]:
            registry_payload = self.registry.load()
            for item in registry_payload["models"]:
                if item.get("model_id") == candidate["model_id"]:
                    item["status"] = "approved"
            self.registry.save(registry_payload)
            self.event_log.append(event_type="deploying_model", payload={"model_id": candidate["model_id"]})
            activation = activate_approved_model(registry=self.registry, model_id=str(candidate["model_id"]))
            self.event_log.append(event_type="model_activated", payload=activation)
            return OperationalState(
                phase="monitoring",
                active_model_id=str(candidate["model_id"]),
                active_model_path=str(candidate["path"]),
                candidate_model_id=str(candidate["model_id"]),
                candidate_model_path=str(candidate["path"]),
                buffered_new_sample_count=0,
                last_retrain_time=_utc_now_iso(),
                current_run_id=_optional_str(run_payload.get("run_id")) or str(Path(run_dir).name),
                last_promotion_result=decision,
                latest_warning_or_error=None,
            )

        return OperationalState(
            phase="candidate_rejected",
            active_model_id=state.active_model_id,
            active_model_path=state.active_model_path,
            candidate_model_id=str(candidate["model_id"]),
            candidate_model_path=str(candidate["path"]),
            buffered_new_sample_count=state.buffered_new_sample_count,
            last_retrain_time=state.last_retrain_time,
            current_run_id=_optional_str(run_payload.get("run_id")) or str(Path(run_dir).name),
            last_promotion_result=decision,
            latest_warning_or_error=None,
        )


def summarize_operational_status(
    *,
    state: OperationalState,
    readiness: dict[str, object],
    latest_run_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "phase": state.phase,
        "active_model": {"model_id": state.active_model_id, "path": state.active_model_path},
        "candidate_model": {"model_id": state.candidate_model_id, "path": state.candidate_model_path},
        "retraining_readiness": readiness,
        "last_promotion_result": state.last_promotion_result,
        "latest_warning_or_error": state.latest_warning_or_error,
        "latest_run_summary_excerpt": _run_summary_excerpt(latest_run_summary),
    }


def _run_summary_excerpt(summary: dict[str, object] | None) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "final_epoch": summary.get("final_epoch"),
        "final_validation_metrics": summary.get("final_validation_metrics"),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def _extract_plume_metrics(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, object] = {}
    for key in (
        "val_support_iou_transformed",
        "val_centroid_distance_raster_transformed",
        "val_mass_abs_error_transformed",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _checkpoint_metric(record: dict[str, object]) -> dict[str, object]:
    metric = record.get("checkpoint_metric")
    if not isinstance(metric, dict):
        raise ValueError("Model record missing checkpoint_metric object")
    name = metric.get("name")
    value = metric.get("value")
    if not isinstance(name, str) or not isinstance(value, (float, int)):
        raise ValueError("checkpoint_metric must include string name and numeric value")
    return {"name": name, "value": float(value)}


def _validate_plume_regressions(
    *,
    reasons: list[str],
    candidate: dict[str, object],
    active: dict[str, object],
    max_support_iou: float | None,
    max_centroid: float | None,
) -> None:
    candidate_metrics = candidate.get("plume_metrics") or {}
    active_metrics = active.get("plume_metrics") or {}
    if not isinstance(candidate_metrics, dict) or not isinstance(active_metrics, dict):
        return

    if max_support_iou is not None:
        c_val = candidate_metrics.get("val_support_iou_transformed")
        a_val = active_metrics.get("val_support_iou_transformed")
        if isinstance(c_val, (float, int)) and isinstance(a_val, (float, int)) and (float(a_val) - float(c_val)) > max_support_iou:
            reasons.append("support_iou_regression_exceeds_tolerance")

    if max_centroid is not None:
        c_val = candidate_metrics.get("val_centroid_distance_raster_transformed")
        a_val = active_metrics.get("val_centroid_distance_raster_transformed")
        if isinstance(c_val, (float, int)) and isinstance(a_val, (float, int)) and (float(c_val) - float(a_val)) > max_centroid:
            reasons.append("centroid_regression_exceeds_tolerance")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected dictionary payload")
    return dict(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
