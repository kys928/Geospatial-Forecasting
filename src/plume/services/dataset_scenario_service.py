from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetScenarioConfig:
    mode: str
    dataset_manifest_path: Path
    windows_manifest_enriched_path: Path
    windows_dir: Path
    scan_limit: int
    activation_state_path: Path
    online_subset_path: Path


class DatasetScenarioService:
    CHANNELS = [
        "plume_concentration", "u10m_ms", "v10m_ms", "wspd10_ms", "wdir_sin", "wdir_cos", "pblh_m", "sfcp_hpa", "rh2m_pct", "t02m_k"
    ]

    def __init__(self, config: DatasetScenarioConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "DatasetScenarioService":
        root = Path(os.getenv("PLUME_FULL_DATASET_PATH", ""))
        manifest = Path(os.getenv("PLUME_DATASET_MANIFEST_PATH", str(root / "dataset_manifest.csv")))
        windows_manifest = Path(os.getenv("PLUME_WINDOWS_MANIFEST_ENRICHED_PATH", str(root / "windows_manifest_enriched.csv")))
        windows_dir = Path(os.getenv("PLUME_WINDOWS_DIR", str(root / "windows")))
        mode = os.getenv("PLUME_DATASET_SCENARIO_MODE", "disabled").strip().lower()
        if mode not in {"enabled", "disabled"}:
            mode = "disabled"
        scan_limit = int(os.getenv("PLUME_DATASET_SCENARIO_SCAN_LIMIT", "500"))
        activation = Path(os.getenv("PLUME_DATASET_SCENARIO_STATE_PATH", "runtime_state/active_dataset_scenario.json"))
        online_subset_override = os.getenv("PLUME_ONLINE_SUBSET_PATH")
        if online_subset_override:
            online_subset = Path(online_subset_override)
        elif str(root).startswith("/workspace/Dataset"):
            online_subset = Path("/workspace/Dataset/online_learning_subset")
        else:
            online_subset = root / "online_learning_subset"
        return cls(DatasetScenarioConfig(mode, manifest, windows_manifest, windows_dir, scan_limit, activation, online_subset))

    def is_enabled(self) -> bool:
        return self.config.mode == "enabled"

    def availability(self) -> dict[str, Any]:
        windows_count = len(list(self.config.windows_dir.glob("*.npz"))) if self.config.windows_dir.exists() else 0
        return {
            "dataset_available": self.config.dataset_manifest_path.exists() and self.config.windows_manifest_enriched_path.exists() and self.config.windows_dir.exists(),
            "dataset_window_count": windows_count,
            "dataset_manifest_path": str(self.config.dataset_manifest_path),
            "windows_manifest_enriched_path": str(self.config.windows_manifest_enriched_path),
            "online_subset_path": str(self.config.online_subset_path),
            "online_subset_available": self.config.online_subset_path.exists(),
        }

    def list_scenarios(self) -> list[dict[str, Any]]:
        return [self._scenario_preview(k, v) for k, v in self._select_scenarios().items()]

    def get_scenario(self, scenario_key: str) -> dict[str, Any]:
        selected = self._select_scenarios()
        if scenario_key not in selected:
            raise KeyError(scenario_key)
        return selected[scenario_key]["payload"]

    def activate(self, scenario_key: str) -> None:
        _ = self.get_scenario(scenario_key)
        self.config.activation_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.activation_state_path.write_text(json.dumps({"scenario_id": scenario_key}), encoding="utf-8")

    def get_active(self) -> str | None:
        if not self.config.activation_state_path.exists():
            return None
        try:
            payload = json.loads(self.config.activation_state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = payload.get("scenario_id")
        return value if isinstance(value, str) else None

    def _select_scenarios(self) -> dict[str, dict[str, Any]]:
        if not self.is_enabled():
            return {}
        if not (self.config.dataset_manifest_path.exists() and self.config.windows_manifest_enriched_path.exists() and self.config.windows_dir.exists()):
            return {}
        manifests = {r["scenario_id"]: r for r in self._read_csv(self.config.dataset_manifest_path)}
        windows = [w for w in self._read_csv(self.config.windows_manifest_enriched_path) if str(w.get("ok", "")).lower() in {"true", "1", "yes"}]
        candidates = []
        for row in windows[: self.config.scan_limit]:
            if "(3, 10, 64, 64)" not in str(row.get("input_shape_new", "")):
                continue
            scenario_id = row.get("scenario_id")
            if scenario_id not in manifests:
                continue
            npz_path = self.config.windows_dir / Path(str(row.get("sample_path", ""))).name
            if not npz_path.exists():
                continue
            candidates.append((row, manifests[scenario_id], npz_path))

        scored = [self._score_candidate(w, m, p) for w, m, p in candidates]
        scored = [s for s in scored if s is not None]
        if not scored:
            return {}
        scored.sort(key=lambda x: x["plume_strength"])
        lowest = scored[0]
        nonzero = [s for s in scored if s["plume_strength"] > 0]
        small = nonzero[0] if nonzero else lowest
        large = max(scored, key=lambda x: x["plume_strength"])
        strong_wind = max(scored, key=lambda x: x["wind_speed"])
        picks = {
            "dataset_lowest_plume": lowest,
            "dataset_small_plume": small,
            "dataset_strong_wind": strong_wind,
            "dataset_large_plume": large,
        }
        strengths = np.array([float(s["plume_strength"]) for s in scored], dtype=float)
        top_quartile = float(np.nanpercentile(strengths, 75)) if strengths.size else 0.0
        for key, selected in picks.items():
            plume_strength = float(selected["plume_strength"])
            risk = "low"
            if key == "dataset_large_plume" or plume_strength >= top_quartile > 0:
                risk = "high"
            elif key == "dataset_small_plume":
                risk = "medium" if plume_strength > 0 else "low"
            elif key == "dataset_strong_wind":
                risk = "medium" if plume_strength > 0 else "low"
            selected["payload"]["forecast"]["risk_level"] = risk
        return {k: {"payload": v["payload"], "risk": v["payload"]["forecast"]["risk_level"], "status": v["payload"]["forecast"]["status"]} for k, v in picks.items()}

    def _score_candidate(self, window_row: dict[str, str], manifest_row: dict[str, str], npz_path: Path) -> dict[str, Any] | None:
        arr = np.load(npz_path)
        input_data = arr["input"]
        target = arr["target"]
        plume = self._extract_plume_channel(target)
        plume_strength = float(np.nanmean(plume))
        wind_speed = float(np.nanmedian(input_data[2, 3]))
        payload = self._build_payload(window_row, manifest_row, input_data, target, plume_strength)
        return {"plume_strength": plume_strength, "wind_speed": wind_speed, "payload": payload}

    def _build_payload(self, window_row, manifest_row, input_data, target, plume_strength):
        last = input_data[-1]
        u10 = float(np.nanmedian(last[1])); v10 = float(np.nanmedian(last[2])); wspd = float(np.nanmedian(last[3]))
        sin_v = float(np.nanmedian(last[4])); cos_v = float(np.nanmedian(last[5]))
        direction = (math.degrees(math.atan2(sin_v, cos_v)) + 360.0) % 360.0 if not math.isnan(sin_v) and not math.isnan(cos_v) else (math.degrees(math.atan2(u10, v10))+360.0)%360.0
        risk = "low"
        start = datetime.fromisoformat(str(manifest_row["start_time"]).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        run_hours = float(manifest_row.get("run_hours", 0) or 0)
        plume_channel = self._extract_plume_channel(target)
        threshold = float(np.nanmax(plume_channel)) * 0.1 if float(np.nanmax(plume_channel)) > 0 else 0.0
        affected = int(np.count_nonzero(plume_channel > threshold)) if threshold > 0 else int(np.count_nonzero(plume_channel > 0))
        if float(np.nanmax(plume_channel)) == 0.0 and affected == 0:
            risk = "low"
        elif plume_strength > 0:
            risk = "medium"
        return {
            "forecast": {"forecast_id": f"dataset_{window_row.get('window_id')}", "timestamp": start.isoformat(), "issued_at": start.isoformat(), "status": "plume detected above threshold" if float(np.nanmax(plume_channel)) > 0 else "no meaningful plume above threshold", "risk_level": risk, "input_source": "dataset_playback", "scenario_id": f"{manifest_row.get('scenario_id')}:{window_row.get('window_id')}"},
            "conditions": {"u10m_ms": u10, "v10m_ms": v10, "wind_speed_ms": wspd, "wind_direction_deg": direction, "wind_direction_label": self._direction_label(direction), "pbl_height_m": float(np.nanmedian(last[6])), "surface_pressure_hpa": float(np.nanmedian(last[7])), "humidity_pct": float(np.nanmedian(last[8])), "temperature_c": float(np.nanmedian(last[9])) - 273.15, "meteorology_source": "kaggle_hysplit_enriched_npz", "meteorology_timestamp": start.isoformat()},
            "source": {"latitude": float(manifest_row.get("lat", 0)), "longitude": float(manifest_row.get("lon", 0)), "pollutant": "demo_release", "emission_rate": float(manifest_row.get("emission_rate", 0)), "release_height_m": float(manifest_row.get("height_m", 0)), "duration_minutes": run_hours * 60, "start_time": start.isoformat(), "end_time": (start + timedelta(hours=run_hours)).isoformat()},
            "plume_metrics": {"max_concentration": float(np.nanmax(plume_channel)), "mean_concentration": float(np.nanmean(plume_channel)), "affected_cells_above_threshold": affected, "affected_area_m2": None, "affected_area_hectares": None, "dominant_spread_direction": None, "threshold_used": threshold, "grid_rows": int(plume_channel.shape[-2]), "grid_columns": int(plume_channel.shape[-1])},
            "runtime": {"backend": "dataset_playback", "model_name": "hysplit_convlstm_enriched_dataset", "model_source": "dataset_playback", "model_version": "kaggle_dataset", "output_space": "dataset_playback", "input_mode": "scenario_playback", "prediction_trust": "demo_only", "missing_channels": [], "missing_frame_indices": [], "meteorology_available": True, "observations_available": False, "limitations": ["Dataset playback from Kaggle HYSPLIT/ConvLSTM dataset; not live OpenRemote data.", "Values are for demo/development playback.", "Plume values may be transformed dataset values unless model/output-space metadata defines physical units."]},
            "raw": {"source_file": str(window_row.get("source_file")), "manifest_row": manifest_row, "window_row": window_row, "input_shape": list(input_data.shape), "target_shape": list(target.shape), "channel_order": self.CHANNELS},
        }

    def _scenario_preview(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        p = value["payload"]
        return {"scenario_id": key, "label": key.replace("dataset_", "").replace("_", " ").title(), "status": value["status"], "risk_level": value["risk"], "wind_speed_ms": p["conditions"]["wind_speed_ms"], "max_concentration": p["plume_metrics"]["max_concentration"]}

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _direction_label(degrees: float) -> str:
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        return labels[int((degrees + 22.5) // 45)]

    @staticmethod
    def _extract_plume_channel(target: np.ndarray) -> np.ndarray:
        if target.ndim == 4:
            return target[-1, 0]
        if target.ndim == 3:
            return target[0]
        if target.ndim == 2:
            return target
        raise ValueError(f"Expected target ndim to be 2, 3, or 4; got shape {target.shape}.")
