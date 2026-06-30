from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import json
from pathlib import Path
from typing import Any

ROBUST_MODEL_NAME = "RobustMultiStepConvLSTMForecaster"
EXPECTED_INPUT_SHAPE = [3, 10, 64, 64]
EXPECTED_OUTPUT_SHAPE = [4, 1, 64, 64]
PRIMARY_METRICS = (
    "val_rollout_weighted_mse",
    "val_rollout_weighted_mse_t3",
    "val_rollout_weighted_mse_t4",
    "val_rollout_mae",
    "val_rollout_mass_abs_error",
    "val_rollout_peak_location_error",
    "selection_score",
)


@dataclass(frozen=True)
class AdaptationPromotionThresholds:
    min_selection_score_improvement_percent: float = 2.0
    min_rollout_wmse_improvement_percent: float = 2.0
    max_allowed_t3_worse_percent: float = 3.0
    max_allowed_t4_worse_percent: float = 3.0
    max_allowed_mass_worse_percent: float = 25.0


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    auto_activation_allowed: bool
    checkpoint_path: str | None
    reasons: list[str] = field(default_factory=list)
    strict_torch_check_performed: bool = False
    contract: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationPromotionDecision:
    classification: str
    should_auto_activate: bool
    manual_approval_required: bool
    should_reject: bool
    reasons: list[str]
    comparisons: dict[str, object]
    compatibility: CompatibilityResult
    candidate_model_id: str | None = None
    active_model_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["compatibility"] = self.compatibility.to_dict()
        return payload


def evaluate_adaptation_candidate(
    *,
    candidate_record: dict[str, object],
    active_record: dict[str, object] | None,
    thresholds: AdaptationPromotionThresholds | None = None,
) -> AdaptationPromotionDecision:
    cfg = thresholds or AdaptationPromotionThresholds()
    reasons: list[str] = []
    comparisons: dict[str, object] = {}
    compatibility = check_adaptation_checkpoint_compatibility(candidate_record, require_strict_torch=True)

    summary = load_candidate_training_summary(candidate_record)
    if summary is None:
        reasons.append("training_summary_missing")
        return _decision("invalid", candidate_record, active_record, reasons, comparisons, compatibility)
    status = str(summary.get("status", "")).lower()
    if status and status not in {"completed", "succeeded", "success"}:
        reasons.append(f"training_summary_status_{status}")
        return _decision("worse", candidate_record, active_record, reasons, comparisons, compatibility)

    if not compatibility.compatible:
        reasons.extend(compatibility.reasons)
        return _decision("invalid", candidate_record, active_record, _dedupe(reasons), comparisons, compatibility)

    candidate_metrics = extract_promotion_metrics(candidate_record, training_summary=summary)
    active_metrics = extract_promotion_metrics(active_record) if active_record is not None else {}
    comparisons["candidate_metrics"] = candidate_metrics
    comparisons["active_metrics"] = active_metrics

    missing_candidate = [key for key in ("val_rollout_weighted_mse", "val_rollout_weighted_mse_t3", "val_rollout_weighted_mse_t4", "selection_score") if key not in candidate_metrics]
    if missing_candidate:
        reasons.append("candidate_comparison_metrics_missing:" + ",".join(missing_candidate))
        return _decision("worse", candidate_record, active_record, reasons, comparisons, compatibility)
    if not active_metrics:
        reasons.append("active_baseline_metrics_missing")
        if not compatibility.auto_activation_allowed:
            reasons.extend(compatibility.reasons)
        return _decision("uncertain", candidate_record, active_record, _dedupe(reasons), comparisons, compatibility)

    missing_active = [key for key in ("val_rollout_weighted_mse", "val_rollout_weighted_mse_t3", "val_rollout_weighted_mse_t4", "selection_score") if key not in active_metrics]
    if missing_active:
        reasons.append("active_baseline_metrics_missing:" + ",".join(missing_active))
        if not compatibility.auto_activation_allowed:
            reasons.extend(compatibility.reasons)
        return _decision("uncertain", candidate_record, active_record, _dedupe(reasons), comparisons, compatibility)

    percent = {
        key: _improvement_percent(active_metrics[key], candidate_metrics[key])
        for key in sorted(set(active_metrics).intersection(candidate_metrics))
        if key in PRIMARY_METRICS
    }
    comparisons["improvement_percent"] = percent

    rollout_improvement = percent.get("val_rollout_weighted_mse")
    selection_improvement = percent.get("selection_score")
    t3_worse = -percent.get("val_rollout_weighted_mse_t3", 0.0)
    t4_worse = -percent.get("val_rollout_weighted_mse_t4", 0.0)
    mass_worse = -percent.get("val_rollout_mass_abs_error", 0.0)

    if rollout_improvement is not None and rollout_improvement < -cfg.min_rollout_wmse_improvement_percent:
        reasons.append("rollout_wmse_regression_exceeds_tolerance")
    if t3_worse > cfg.max_allowed_t3_worse_percent:
        reasons.append("t3_regression_exceeds_tolerance")
    if t4_worse > cfg.max_allowed_t4_worse_percent:
        reasons.append("t4_regression_exceeds_tolerance")
    if mass_worse > cfg.max_allowed_mass_worse_percent:
        reasons.append("mass_regression_exceeds_tolerance")
    if reasons:
        return _decision("worse", candidate_record, active_record, reasons, comparisons, compatibility)

    meaningful_rollout = rollout_improvement is not None and rollout_improvement >= cfg.min_rollout_wmse_improvement_percent
    meaningful_selection = selection_improvement is not None and selection_improvement >= cfg.min_selection_score_improvement_percent
    if (meaningful_rollout or meaningful_selection) and compatibility.auto_activation_allowed:
        reasons.append("candidate_clearly_improves_primary_metric")
        return _decision("clearly_better", candidate_record, active_record, reasons, comparisons, compatibility)

    if not compatibility.auto_activation_allowed:
        reasons.extend(compatibility.reasons)
    if not meaningful_rollout and not meaningful_selection:
        reasons.append("improvement_below_clear_promotion_threshold")
    return _decision("uncertain", candidate_record, active_record, _dedupe(reasons), comparisons, compatibility)


def check_adaptation_checkpoint_compatibility(
    candidate_record: dict[str, object],
    *,
    require_strict_torch: bool,
) -> CompatibilityResult:
    checkpoint_value = candidate_record.get("path") or candidate_record.get("best_overall_checkpoint") or candidate_record.get("final_checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        return CompatibilityResult(False, False, None, ["checkpoint_path_missing"])
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        return CompatibilityResult(False, False, str(checkpoint_path), ["checkpoint_file_missing"])

    json_payload = _try_load_json_checkpoint(checkpoint_path)
    if json_payload is not None:
        contract_result = _validate_checkpoint_payload(json_payload)
        if not contract_result.compatible:
            return CompatibilityResult(False, False, str(checkpoint_path), contract_result.reasons, contract=contract_result.contract)
        reasons = ["cannot_perform_strict_torch_compatibility_check"] if require_strict_torch else []
        return CompatibilityResult(True, not require_strict_torch, str(checkpoint_path), reasons, False, contract_result.contract)

    if importlib.util.find_spec("torch") is None:
        if require_strict_torch:
            return CompatibilityResult(
                True,
                False,
                str(checkpoint_path),
                ["cannot_perform_strict_torch_compatibility_check", "cannot_inspect_torch_checkpoint_without_torch"],
            )
        return CompatibilityResult(False, False, str(checkpoint_path), ["torch_unavailable_contract_not_inspected"])

    try:
        import torch  # type: ignore

        raw = _load_torch_checkpoint_payload(torch, checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        return CompatibilityResult(False, False, str(checkpoint_path), [f"checkpoint_torch_load_failed:{exc}"])
    if not isinstance(raw, dict):
        return CompatibilityResult(False, False, str(checkpoint_path), ["checkpoint_payload_not_object"])
    contract_result = _validate_checkpoint_payload(raw)
    if not contract_result.compatible:
        return CompatibilityResult(False, False, str(checkpoint_path), contract_result.reasons, contract=contract_result.contract)
    if require_strict_torch:
        try:
            from plume.models.torch_robust_multistep_convlstm import RobustMultiStepConvLSTMCheckpoint

            RobustMultiStepConvLSTMCheckpoint(checkpoint_path, device="cpu")
        except Exception as exc:  # noqa: BLE001
            return CompatibilityResult(False, False, str(checkpoint_path), [f"strict_robust_loader_failed:{exc}"], contract=contract_result.contract)
    return CompatibilityResult(True, True, str(checkpoint_path), [], bool(require_strict_torch), contract_result.contract)


def validate_adaptation_checkpoint_for_activation(candidate_record: dict[str, object]) -> CompatibilityResult:
    return check_adaptation_checkpoint_compatibility(candidate_record, require_strict_torch=False)


def load_candidate_training_summary(candidate_record: dict[str, object]) -> dict[str, object] | None:
    adaptation_run = candidate_record.get("adaptation_run")
    if isinstance(adaptation_run, dict):
        summary = adaptation_run.get("training_summary")
        if isinstance(summary, dict) and summary:
            return dict(summary)
        summary_path = adaptation_run.get("training_summary_path")
        loaded = _load_json_object(summary_path)
        if loaded is not None:
            return loaded
    direct = candidate_record.get("training_summary")
    if isinstance(direct, dict) and direct:
        return dict(direct)
    loaded = _load_json_object(candidate_record.get("training_summary_path"))
    if loaded is not None:
        return loaded
    run_dir = candidate_record.get("created_from_run_dir") or candidate_record.get("output_dir")
    if isinstance(run_dir, str):
        return _load_json_object(str(Path(run_dir) / "training_summary.json"))
    return None


def extract_promotion_metrics(record: dict[str, object] | None, *, training_summary: dict[str, object] | None = None) -> dict[str, float]:
    if record is None:
        return {}
    metrics: dict[str, float] = {}
    for source_key in ("adaptation_promotion_metrics", "promotion_metrics", "metrics"):
        source = record.get(source_key)
        if isinstance(source, dict):
            _merge_numeric_metrics(metrics, source)
    adaptation_run = record.get("adaptation_run")
    if isinstance(adaptation_run, dict):
        summary = adaptation_run.get("training_summary")
        if isinstance(summary, dict):
            best = summary.get("best_metrics")
            if isinstance(best, dict):
                _merge_numeric_metrics(metrics, best)
    if training_summary is not None:
        best = training_summary.get("best_metrics")
        if isinstance(best, dict):
            _merge_numeric_metrics(metrics, best)
    checkpoint_metric = record.get("checkpoint_metric")
    if isinstance(checkpoint_metric, dict):
        name = checkpoint_metric.get("name")
        value = checkpoint_metric.get("value")
        if isinstance(name, str) and name in PRIMARY_METRICS and isinstance(value, (float, int)):
            metrics[name] = float(value)
    for key in PRIMARY_METRICS:
        value = record.get(key)
        if isinstance(value, (float, int)):
            metrics[key] = float(value)
    return metrics


def _load_torch_checkpoint_payload(torch_module: Any, checkpoint_path: Path) -> object:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch versions do not support weights_only.
        return torch_module.load(checkpoint_path, map_location="cpu")
    except Exception as weights_only_error:  # noqa: BLE001
        if not _is_weights_only_compatibility_error(weights_only_error):
            raise
        # Some legacy trusted checkpoints require pickle objects that are outside
        # torch's weights-only allowlist. Preserve existing compatibility only
        # for known weights-only compatibility failures.
        try:
            return torch_module.load(checkpoint_path, map_location="cpu")
        except Exception:  # noqa: BLE001
            raise weights_only_error


def _is_weights_only_compatibility_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "Weights only load failed",
            "weights_only",
            "Unsupported global",
            "add_safe_globals",
            "WeightsUnpickler",
        )
    )


def _validate_checkpoint_payload(payload: dict[str, object]) -> CompatibilityResult:
    reasons: list[str] = []
    if "model_state_dict" not in payload:
        reasons.append("model_state_dict_missing")
    contract = payload.get("model_contract")
    contract_dict = dict(contract) if isinstance(contract, dict) else None
    if contract_dict is None:
        reasons.append("model_contract_missing")
    else:
        if contract_dict.get("model_name") != ROBUST_MODEL_NAME:
            reasons.append("model_contract_model_name_mismatch")
        if list(contract_dict.get("input_shape") or []) != EXPECTED_INPUT_SHAPE:
            reasons.append("model_contract_input_shape_mismatch")
        if list(contract_dict.get("output_shape") or []) != EXPECTED_OUTPUT_SHAPE:
            reasons.append("model_contract_output_shape_mismatch")
    return CompatibilityResult(not reasons, not reasons, None, reasons, False, contract_dict)


def _try_load_json_checkpoint(path: Path) -> dict[str, object] | None:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return decoded if isinstance(decoded, dict) else None


def _load_json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.exists():
        return None
    decoded = json.loads(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else None


def _merge_numeric_metrics(target: dict[str, float], source: dict[str, object]) -> None:
    for key in PRIMARY_METRICS:
        value = source.get(key)
        if isinstance(value, (float, int)):
            target[key] = float(value)


def _improvement_percent(active_value: float, candidate_value: float) -> float:
    active = float(active_value)
    candidate = float(candidate_value)
    if active == 0.0:
        if candidate == 0.0:
            return 0.0
        return -100.0 if candidate > active else 100.0
    return ((active - candidate) / abs(active)) * 100.0


def _decision(
    classification: str,
    candidate_record: dict[str, object],
    active_record: dict[str, object] | None,
    reasons: list[str],
    comparisons: dict[str, object],
    compatibility: CompatibilityResult,
) -> AdaptationPromotionDecision:
    return AdaptationPromotionDecision(
        classification=classification,
        should_auto_activate=classification == "clearly_better",
        manual_approval_required=classification == "uncertain",
        should_reject=classification in {"worse", "invalid"},
        reasons=reasons or [classification],
        comparisons=comparisons,
        compatibility=compatibility,
        candidate_model_id=_optional_id(candidate_record),
        active_model_id=_optional_id(active_record),
    )


def _optional_id(record: dict[str, object] | None) -> str | None:
    if record is None:
        return None
    value = record.get("model_id")
    return value if isinstance(value, str) else None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
