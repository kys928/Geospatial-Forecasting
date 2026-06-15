from __future__ import annotations

from plume.services.runtime_mode import build_runtime_mode


EXPECTED_KEYS = [
    "mode",
    "is_active_convlstm",
    "is_fallback",
    "is_dataset_window",
    "is_demo_backend",
    "is_temporary_substitution",
    "backend_name",
    "model_family",
    "prediction_engine",
    "input_source",
    "output_source",
    "active_model_id",
    "checkpoint_path",
    "fallback_reason",
    "source_metadata_keys",
]


def test_none_metadata_returns_unknown_with_stable_keys():
    result = build_runtime_mode(None)

    assert list(result.keys()) == EXPECTED_KEYS
    assert result == {
        "mode": "unknown",
        "is_active_convlstm": False,
        "is_fallback": False,
        "is_dataset_window": False,
        "is_demo_backend": False,
        "is_temporary_substitution": False,
        "backend_name": None,
        "model_family": None,
        "prediction_engine": None,
        "input_source": None,
        "output_source": None,
        "active_model_id": None,
        "checkpoint_path": None,
        "fallback_reason": None,
        "source_metadata_keys": [],
    }


def test_empty_dict_returns_unknown():
    assert build_runtime_mode({})["mode"] == "unknown"


def test_active_convlstm_metadata_returns_mode_active_convlstm():
    result = build_runtime_mode({"backend_name": "convlstm_online"})

    assert result["mode"] == "active_convlstm"
    assert result["is_active_convlstm"] is True


def test_fallback_used_true_returns_fallback_even_if_backend_is_convlstm_online():
    result = build_runtime_mode({"fallback_used": True, "backend_name": "convlstm_online"})

    assert result["mode"] == "fallback"
    assert result["is_fallback"] is True
    assert result["is_active_convlstm"] is False


def test_fallback_backend_name_returns_fallback():
    result = build_runtime_mode({"fallback_backend_name": "gaussian_plume"})

    assert result["mode"] == "fallback"
    assert result["is_fallback"] is True


def test_mock_online_backend_returns_demo_backend():
    result = build_runtime_mode({"backend_name": "mock_online"})

    assert result["mode"] == "demo_backend"
    assert result["is_demo_backend"] is True


def test_ridge_baseline_prediction_engine_returns_temporary_substitution():
    result = build_runtime_mode({"prediction_engine": "ridge_baseline"})

    assert result["mode"] == "temporary_substitution"
    assert result["is_temporary_substitution"] is True


def test_dataset_window_input_source_returns_dataset_window():
    result = build_runtime_mode({"input_source": "dataset_window"})

    assert result["mode"] == "dataset_window"
    assert result["is_dataset_window"] is True


def test_dataset_window_used_true_returns_dataset_window():
    result = build_runtime_mode({"dataset_window_used": True})

    assert result["mode"] == "dataset_window"
    assert result["is_dataset_window"] is True


def test_function_does_not_mutate_input_metadata():
    metadata = {"backend_name": " convlstm_online ", "nested": {"value": 1}}
    original = metadata.copy()

    build_runtime_mode(metadata)

    assert metadata == original


def test_non_dict_input_does_not_crash_and_returns_unknown():
    result = build_runtime_mode([("backend_name", "convlstm_online")])  # type: ignore[arg-type]

    assert result["mode"] == "unknown"
    assert result["source_metadata_keys"] == []


def test_source_metadata_keys_is_sorted():
    result = build_runtime_mode({"z": 1, "a": 2, "m": 3})

    assert result["source_metadata_keys"] == ["a", "m", "z"]


def test_string_comparisons_are_case_insensitive_and_whitespace_tolerant():
    result = build_runtime_mode({"prediction_engine": "  TORCH_CONVLSTM  "})

    assert result["mode"] == "active_convlstm"
    assert result["prediction_engine"] == "TORCH_CONVLSTM"


def test_boolean_like_strings_are_handled_for_fallback_used_and_dataset_window_used():
    fallback = build_runtime_mode({"fallback_used": " yes "})
    dataset_window = build_runtime_mode({"dataset_window_used": "ON"})

    assert fallback["mode"] == "fallback"
    assert dataset_window["mode"] == "dataset_window"


def test_alias_fields_are_normalized_into_required_output_fields():
    result = build_runtime_mode(
        {
            "model_backend": " convlstm_online ",
            "model_name": " ConvLSTM ",
            "engine": " torch ",
            "source": " sensors ",
            "prediction_source": " grid ",
            "registry_model_id": " model-1 ",
            "model_path": " /tmp/model.pt ",
            "exception": " timeout ",
        }
    )

    assert result["backend_name"] == "convlstm_online"
    assert result["model_family"] == "ConvLSTM"
    assert result["prediction_engine"] == "torch"
    assert result["input_source"] == "sensors"
    assert result["output_source"] == "grid"
    assert result["active_model_id"] == "model-1"
    assert result["checkpoint_path"] == "/tmp/model.pt"
    assert result["fallback_reason"] == "timeout"
    assert result["mode"] == "fallback"
