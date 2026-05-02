from __future__ import annotations


def build_explanation_payload(result, explanation_result) -> dict[str, object]:
    return {
        "forecast_id": result.forecast_id,
        "issued_at": result.issued_at.isoformat(),
        "model": result.model_name,
        "used_llm": explanation_result.used_llm,
        "summary": {
            "source_latitude": explanation_result.summary.source_latitude,
            "source_longitude": explanation_result.summary.source_longitude,
            "grid_rows": explanation_result.summary.grid_rows,
            "grid_columns": explanation_result.summary.grid_columns,
            "projection": explanation_result.summary.projection,
            "max_concentration": explanation_result.summary.max_concentration,
            "mean_concentration": explanation_result.summary.mean_concentration,
            "affected_cells_above_threshold": explanation_result.summary.affected_cells_above_threshold,
            "affected_area_m2": explanation_result.summary.affected_area_m2,
            "affected_area_hectares": explanation_result.summary.affected_area_hectares,
            "dominant_spread_direction": explanation_result.summary.dominant_spread_direction,
            "threshold_used": explanation_result.summary.threshold_used,
            "note": explanation_result.summary.note,
        },
        "explanation": explanation_result.explanation,
    }
