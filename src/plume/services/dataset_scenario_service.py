from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from plume.models.ridge_plume_baseline import load_ridge_artifact, predict_ridge_plume


@dataclass
class DatasetScenarioConfig:
    mode: str
    dataset_manifest_path: Path
    windows_manifest_enriched_path: Path
    windows_dir: Path
    scan_limit: int
    activation_state_path: Path
    online_subset_path: Path
    playback_state_path: Path
    ridge_model_path: Path


class DatasetScenarioService:
    SCENARIO_ALIASES = {
        "dataset_normal_stream": "dataset_normal",
        "dataset_lowest_plume": "dataset_low_plume",
        "dataset_small_plume": "dataset_medium_plume",
    }
    CHANNELS = [
        "plume_concentration", "u10m_ms", "v10m_ms", "wspd10_ms", "wdir_sin", "wdir_cos", "pblh_m", "sfcp_hpa", "rh2m_pct", "t02m_k"
    ]

    def __init__(self, config: DatasetScenarioConfig):
        self.config = config
        self._scenario_cache: dict[str, dict[str, Any]] | None = None
        self._cache_signature: tuple[str, ...] | None = None
        self._ridge_artifact: dict[str, Any] | None = None

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
        playback_state = Path(os.getenv("PLUME_DATASET_PLAYBACK_STATE_PATH", "runtime_state/dataset_playback_state.json"))
        ridge_model_path = Path(os.getenv("PLUME_DATASET_RIDGE_MODEL_PATH", "artifacts/models/ridge_plume_baseline.pkl"))
        return cls(DatasetScenarioConfig(mode, manifest, windows_manifest, windows_dir, scan_limit, activation, online_subset, playback_state, ridge_model_path))

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
        scenario_key = self.SCENARIO_ALIASES.get(scenario_key, scenario_key)
        if scenario_key not in selected:
            raise KeyError(scenario_key)
        return selected[scenario_key]["payload"]

    def activate(self, scenario_key: str) -> None:
        scenario_key = self.SCENARIO_ALIASES.get(scenario_key, scenario_key)
        _ = self.get_scenario(scenario_key)
        self.config.activation_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.activation_state_path.write_text(json.dumps({"scenario_id": scenario_key}), encoding="utf-8")
        self.update_playback_state(enabled=True, active_scenario_id=scenario_key)

    def get_active(self) -> str | None:
        if not self.config.activation_state_path.exists():
            return None
        try:
            payload = json.loads(self.config.activation_state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = payload.get("scenario_id")
        if not isinstance(value, str):
            return None
        return self.SCENARIO_ALIASES.get(value, value)


    def refresh_cache(self) -> None:
        self._scenario_cache = None
        self._cache_signature = None

    def get_playback_state(self) -> dict[str, Any]:
        default_speed = int(os.getenv("PLUME_DATASET_PLAYBACK_SPEED_SECONDS", "3"))
        defaults = {"enabled": False, "active_scenario_id": self.get_active(), "mode": "dataset_playback", "updated_at": datetime.now(timezone.utc).isoformat(), "playback_running": False, "playback_index": 0, "playback_speed_seconds": default_speed}
        path = self.config.playback_state_path
        if not path.exists():
            return defaults
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        if not isinstance(payload, dict):
            return defaults
        return {**defaults, **payload}

    def update_playback_state(self, *, enabled: bool, active_scenario_id: str | None = None, playback_running: bool | None = None, playback_index: int | None = None, playback_speed_seconds: int | None = None) -> dict[str, Any]:
        state = self.get_playback_state()
        state["enabled"] = bool(enabled)
        if active_scenario_id is not None:
            state["active_scenario_id"] = active_scenario_id
            self.config.activation_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.activation_state_path.write_text(json.dumps({"scenario_id": active_scenario_id}), encoding="utf-8")
        if playback_running is not None:
            state["playback_running"] = bool(playback_running)
        if playback_index is not None:
            state["playback_index"] = int(playback_index)
        if playback_speed_seconds is not None:
            state["playback_speed_seconds"] = int(playback_speed_seconds)
        state["mode"] = "dataset_playback"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.config.playback_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.playback_state_path.write_text(json.dumps(state), encoding="utf-8")
        return state

    def playback_next(self) -> dict[str, Any]:
        scenarios = self.list_scenarios()
        ids = [s.get("scenario_id") for s in scenarios if isinstance(s.get("scenario_id"), str)]
        state = self.get_playback_state()
        if not ids:
            return state
        current = state.get("active_scenario_id")
        idx = ids.index(current) + 1 if current in ids else 0
        idx = idx % len(ids)
        return self.update_playback_state(enabled=True, active_scenario_id=ids[idx], playback_index=idx)

    def resolve_current_playback_state(self) -> dict[str, Any]:
        state = self.get_playback_state()
        scenarios = self.list_scenarios()
        ids = [s.get("scenario_id") for s in scenarios if isinstance(s.get("scenario_id"), str)]
        if not ids:
            return state
        active = state.get("active_scenario_id")
        if active not in ids:
            active = ids[0]
            state = self.update_playback_state(
                enabled=bool(state.get("enabled", False)),
                active_scenario_id=active,
                playback_running=bool(state.get("playback_running", False)),
                playback_index=0,
                playback_speed_seconds=int(state.get("playback_speed_seconds") or 3),
            )
        return state

    def _select_scenarios(self) -> dict[str, dict[str, Any]]:
        if not self.is_enabled():
            return {}
        signature = (str(self.config.dataset_manifest_path), str(self.config.windows_manifest_enriched_path), str(self.config.windows_dir), str(self.config.scan_limit))
        if self._scenario_cache is not None and self._cache_signature == signature:
            return self._scenario_cache
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
        plume_candidates = [s for s in scored if float(s["plume_metrics"]["max_plume_score"]) > 0 and int(s["plume_metrics"]["plume_cell_count"]) > 0]
        sorted_plume = sorted(plume_candidates, key=lambda x: x["plume_strength"])
        if sorted_plume:
            low_idx = max(0, int(math.floor((len(sorted_plume) - 1) * 0.25)))
            med_idx = max(0, int(math.floor((len(sorted_plume) - 1) * 0.50)))
            high_idx = max(0, int(math.floor((len(sorted_plume) - 1) * 0.75)))
            lowest = sorted_plume[low_idx]
            medium = sorted_plume[med_idx]
            large_pool = sorted([s for s in sorted_plume if s.get("in_demo_bbox")], key=lambda x: x["plume_strength"]) or sorted_plume
            large = large_pool[high_idx if len(large_pool) > high_idx else -1]
            if len(sorted_plume) >= 3:
                picks_unique = []
                seen: set[tuple[str, str]] = set()
                for entry in (lowest, medium, large):
                    pair = entry["candidate_key"]
                    if pair in seen:
                        entry = next((c for c in sorted_plume if c["candidate_key"] not in seen), entry)
                        pair = entry["candidate_key"]
                    seen.add(pair)
                    picks_unique.append(entry)
                lowest, medium, large = picks_unique
        else:
            scored.sort(key=lambda x: x["plume_strength"])
            lowest = medium = large = scored[0]
        normal_stream = copy.deepcopy(sorted(scored, key=lambda x: x["window_sort_key"])[0])
        normal_stream["payload"]["forecast"]["status"] = "no meaningful plume above threshold"
        normal_stream["payload"]["forecast"]["risk_level"] = "low"
        normal_stream["payload"]["plume_metrics"].update({"max_plume_score": 0.0, "mean_plume_score": 0.0, "plume_cell_count": 0, "detection_threshold": 0.0, "plume_fraction": 0.0, "dominant_spread_direction": "No plume", "max_concentration": 0.0, "mean_concentration": 0.0, "affected_cells_above_threshold": 0, "threshold_used": 0.0})
        normal_stream["payload"]["runtime"]["normal_baseline"] = "deterministic_zero_plume_from_dataset_window"
        picks = {
            "dataset_normal": normal_stream,
            "dataset_low_plume": lowest,
            "dataset_medium_plume": medium,
            "dataset_large_plume": large,
        }
        strengths = np.array([float(s["plume_strength"]) for s in scored], dtype=float)
        top_quartile = float(np.nanpercentile(strengths, 75)) if strengths.size else 0.0
        for key, selected in picks.items():
            plume_strength = float(selected["plume_strength"])
            risk = "low"
            if key == "dataset_large_plume" or plume_strength >= top_quartile > 0:
                risk = "high"
            elif key == "dataset_medium_plume":
                risk = "medium" if plume_strength > 0 else "low"
            selected["payload"]["forecast"]["risk_level"] = risk
        if not large.get("in_demo_bbox"):
            large["payload"].setdefault("runtime", {}).setdefault("limitations", []).append("Large plume scenario fell back to a candidate outside the configured demo bounding box.")
        result = {k: {"payload": v["payload"], "risk": v["payload"]["forecast"]["risk_level"], "status": v["payload"]["forecast"]["status"]} for k, v in picks.items()}
        self._scenario_cache = result
        self._cache_signature = signature
        return result

    def _score_candidate(self, window_row: dict[str, str], manifest_row: dict[str, str], npz_path: Path) -> dict[str, Any] | None:
        arr = np.load(npz_path)
        input_data = arr["input"]
        target = arr["target"]
        prediction = predict_ridge_plume(input_data, self._get_ridge_artifact())
        threshold = float(np.nanmax(prediction)) * 0.1 if float(np.nanmax(prediction)) > 0 else 0.0
        plume_cell_count = int(np.count_nonzero(prediction > threshold)) if threshold > 0 else int(np.count_nonzero(prediction > 0))
        plume_fraction = plume_cell_count / float(prediction.size or 1)
        mean_score = float(np.nanmean(prediction))
        max_score = float(np.nanmax(prediction))
        plume_strength = max_score + mean_score + plume_fraction * 10.0
        wind_speed = float(np.nanmedian(input_data[2, 3]))
        payload = self._build_payload(window_row, manifest_row, input_data, target, prediction, plume_strength)
        source_lat = float(manifest_row.get("lat", 0) or 0)
        source_lon = float(manifest_row.get("lon", 0) or 0)
        in_demo_bbox = self._in_demo_bbox(source_lat, source_lon)
        forecast_id = str(payload["forecast"]["forecast_id"])
        window_id = str(window_row.get("window_id", ""))
        window_sort_key = f"{manifest_row.get('start_time', '')}_{window_row.get('window_id', '')}"
        return {"plume_strength": plume_strength, "wind_speed": wind_speed, "payload": payload, "window_sort_key": window_sort_key, "plume_metrics": payload["plume_metrics"], "in_demo_bbox": in_demo_bbox, "candidate_key": (forecast_id, window_id)}

    def _build_payload(self, window_row, manifest_row, input_data, target, prediction, plume_strength):
        last = input_data[-1]
        u10 = float(np.nanmedian(last[1])); v10 = float(np.nanmedian(last[2])); wspd = float(np.nanmedian(last[3]))
        sin_v = float(np.nanmedian(last[4])); cos_v = float(np.nanmedian(last[5]))
        direction = (math.degrees(math.atan2(sin_v, cos_v)) + 360.0) % 360.0 if not math.isnan(sin_v) and not math.isnan(cos_v) else (math.degrees(math.atan2(u10, v10))+360.0)%360.0
        risk = "low"
        start = datetime.fromisoformat(str(manifest_row["start_time"]).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        run_hours = float(manifest_row.get("run_hours", 0) or 0)
        plume_channel = prediction
        threshold = float(np.nanmax(plume_channel)) * 0.1 if float(np.nanmax(plume_channel)) > 0 else 0.0
        affected = int(np.count_nonzero(plume_channel > threshold)) if threshold > 0 else int(np.count_nonzero(plume_channel > 0))
        plume_fraction = affected / float(plume_channel.size or 1)
        spread_direction = self._compute_predicted_spread_direction(plume_channel, threshold)
        if spread_direction is None:
            spread_direction = self._direction_label(direction)
        if float(np.nanmax(plume_channel)) == 0.0 and affected == 0:
            risk = "low"
        elif plume_strength > 0:
            risk = "medium"
        return {
            "forecast": {"forecast_id": f"dataset_{window_row.get('window_id')}", "timestamp": start.isoformat(), "issued_at": start.isoformat(), "status": "plume detected above threshold" if float(np.nanmax(plume_channel)) > 0 else "no meaningful plume above threshold", "risk_level": risk, "input_source": "dataset_playback", "scenario_id": f"{manifest_row.get('scenario_id')}:{window_row.get('window_id')}"},
            "conditions": {"u10m_ms": u10, "v10m_ms": v10, "wind_speed_ms": wspd, "wind_direction_deg": direction, "wind_direction_label": self._direction_label(direction), "pbl_height_m": float(np.nanmedian(last[6])), "surface_pressure_hpa": float(np.nanmedian(last[7])), "humidity_pct": float(np.nanmedian(last[8])), "temperature_c": float(np.nanmedian(last[9])) - 273.15, "meteorology_source": "kaggle_hysplit_enriched_npz", "meteorology_timestamp": start.isoformat()},
            "source": {"latitude": float(manifest_row.get("lat", 0)), "longitude": float(manifest_row.get("lon", 0)), "pollutant": "demo_release", "emission_rate": float(manifest_row.get("emission_rate", 0)), "release_height_m": float(manifest_row.get("height_m", 0)), "duration_minutes": run_hours * 60, "start_time": start.isoformat(), "end_time": (start + timedelta(hours=run_hours)).isoformat()},
            "plume_metrics": {"max_plume_score": float(np.nanmax(plume_channel)), "mean_plume_score": float(np.nanmean(plume_channel)), "detection_threshold": threshold, "plume_cell_count": affected, "plume_fraction": plume_fraction, "dominant_spread_direction": spread_direction, "max_concentration": float(np.nanmax(plume_channel)), "mean_concentration": float(np.nanmean(plume_channel)), "affected_cells_above_threshold": affected, "affected_area_m2": None, "affected_area_hectares": None, "threshold_used": threshold, "grid_rows": int(plume_channel.shape[-2]), "grid_columns": int(plume_channel.shape[-1])},
            "runtime": {"backend": "dataset_playback", "model_name": "ridge_plume_baseline", "model_source": "dataset_input_inference", "model_version": str(self._get_ridge_artifact().get("model_version") or "ridge_baseline_pickle"), "output_space": "ridge_prediction", "input_mode": "dataset_stream_window", "missing_channels": [], "missing_frame_indices": [], "meteorology_available": True, "observations_available": False, "limitations": ["Dataset playback from Kaggle HYSPLIT/ConvLSTM dataset; not live OpenRemote data.", "Values are for demo/development playback.", "Plume values may be transformed dataset values unless model/output-space metadata defines physical units."]},
            "raw": {"source_file": str(window_row.get("source_file")), "manifest_row": manifest_row, "window_row": window_row, "input_shape": list(input_data.shape), "target_shape": list(target.shape), "prediction_shape": list(prediction.shape), "target_usage": "optional_reference_only", "channel_order": self.CHANNELS, "model_inference": {"model_name": "ridge_plume_baseline", "artifact_path": str(self._get_ridge_artifact().get("artifact_path", self.config.ridge_model_path)), "input_shape": list(input_data.shape), "prediction_shape": list(prediction.shape), "used_ridge_model": True, "output_space": "ridge_prediction", "threshold_method": "relative_to_prediction_peak", "threshold_description": "Cutoff used to decide which predicted grid cells are rendered as plume. This is a model-display threshold, not a physical safety threshold."}},
        }

    def _in_demo_bbox(self, lat: float, lon: float) -> bool:
        min_lat = float(os.getenv("PLUME_DATASET_DEMO_MIN_LAT", "50.7"))
        max_lat = float(os.getenv("PLUME_DATASET_DEMO_MAX_LAT", "53.5"))
        min_lon = float(os.getenv("PLUME_DATASET_DEMO_MIN_LON", "4.3"))
        max_lon = float(os.getenv("PLUME_DATASET_DEMO_MAX_LON", "7.3"))
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def _compute_predicted_spread_direction(self, prediction_grid: np.ndarray, threshold: float) -> str | None:
        mask = prediction_grid > threshold
        if not np.any(mask):
            return "No plume"
        rows, cols = np.where(mask)
        weights = prediction_grid[mask].astype(float)
        if weights.size == 0 or float(np.nansum(weights)) <= 0:
            return None
        centroid_row = float(np.average(rows, weights=weights))
        centroid_col = float(np.average(cols, weights=weights))
        center_row = (prediction_grid.shape[0] - 1) / 2.0
        center_col = (prediction_grid.shape[1] - 1) / 2.0
        dy = center_row - centroid_row
        dx = centroid_col - center_col
        if math.hypot(dx, dy) < 0.5:
            return None
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "E"]
        return labels[int((angle + 22.5) // 45)]


    def get_active_payload(self) -> dict[str, Any]:
        self.resolve_current_playback_state()
        scenarios = self.list_scenarios()
        available_ids = {item.get("scenario_id") for item in scenarios}
        active_id = self.get_active()
        selected_id = active_id if isinstance(active_id, str) and active_id in available_ids else None
        if selected_id is None and scenarios:
            selected_id = scenarios[0].get("scenario_id")
        selected_payload = None
        if isinstance(selected_id, str):
            try:
                selected_payload = self.get_scenario(selected_id)
            except KeyError:
                selected_payload = None
        return {
            "enabled": self.is_enabled(),
            "available": bool(scenarios),
            "active_scenario_id": active_id if isinstance(active_id, str) else None,
            "selected_scenario_id": selected_id,
            "scenario": selected_payload,
        }

    def overlay_geojson(self, scenario_key: str) -> dict[str, Any]:
        payload = self.get_scenario(scenario_key)
        if scenario_key == "dataset_normal":
            plume = np.zeros((64, 64), dtype=float)
        else:
            plume = self._load_plume_channel(payload)
        return self._build_overlay(payload, plume)

    def overlay_active_geojson(self) -> dict[str, Any]:
        self.resolve_current_playback_state()
        active = self.get_active_payload()
        scenario_id = active.get("selected_scenario_id")
        if not isinstance(scenario_id, str):
            raise KeyError("no active scenario")
        return self.overlay_geojson(scenario_id)
    def _scenario_preview(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        p = value["payload"]
        labels={"dataset_normal":"Normal","dataset_low_plume":"Low plume","dataset_medium_plume":"Medium plume","dataset_large_plume":"Large plume"}
        return {"scenario_id": key, "label": labels.get(key,key.replace("dataset_", "").replace("_", " ").title()), "status": p["forecast"]["status"], "risk_level": p["forecast"]["risk_level"], "wind_speed_ms": p["conditions"]["wind_speed_ms"], "max_concentration": p["plume_metrics"]["max_concentration"]}

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _direction_label(degrees: float) -> str:
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        return labels[int((degrees + 22.5) // 45)]


    def _load_plume_channel(self, payload: dict[str, Any]) -> np.ndarray:
        sample_path = str(payload.get("raw", {}).get("window_row", {}).get("sample_path", ""))
        if not sample_path:
            raise KeyError("missing sample path")
        npz_path = self.config.windows_dir / Path(sample_path).name
        if not npz_path.exists():
            raise KeyError("dataset sample file unavailable")
        arr = np.load(npz_path)
        return predict_ridge_plume(arr["input"], self._get_ridge_artifact())

    def _build_overlay(self, payload: dict[str, Any], plume: np.ndarray) -> dict[str, Any]:
        span_km = float(os.getenv("PLUME_DATASET_OVERLAY_SPAN_KM", "20"))
        max_features = int(os.getenv("PLUME_DATASET_OVERLAY_MAX_FEATURES", "1000"))
        src = payload.get("source", {})
        lat0 = float(src.get("latitude", 0.0)); lon0 = float(src.get("longitude", 0.0))
        rows, cols = plume.shape[-2], plume.shape[-1]
        positives = plume[plume > 0]
        threshold = payload.get("plume_metrics", {}).get("threshold_used")
        if not isinstance(threshold, (int, float)):
            threshold = float(np.nanpercentile(positives, 75)) if positives.size else 0.0
        lat_step = (span_km / max(rows, 1)) / 111.0
        lon_step = (span_km / max(cols, 1)) / max(111.0 * max(math.cos(math.radians(lat0)), 1e-6), 1e-6)
        half_r = rows / 2.0; half_c = cols / 2.0
        candidates=[]
        for r in range(rows):
            for c in range(cols):
                v=float(plume[r,c])
                if v < float(threshold):
                    continue
                if v <= 0:
                    continue
                candidates.append((v,r,c))
        candidates.sort(reverse=True, key=lambda x: x[0])
        if len(candidates)>max_features:
            candidates=candidates[:max_features]
        max_v = max([v for v,_,_ in candidates], default=0.0)
        feats=[]
        for v,r,c in candidates:
            lat_center = lat0 + (r - half_r + 0.5) * lat_step
            lon_center = lon0 + (c - half_c + 0.5) * lon_step
            lat_min,lat_max=lat_center-lat_step/2,lat_center+lat_step/2
            lon_min,lon_max=lon_center-lon_step/2,lon_center+lon_step/2
            nv=(v/max_v if max_v>0 else 0.0)
            kind = "plume_band_high" if nv >= 0.66 else ("plume_band_medium" if nv >= 0.33 else "plume_band_low")
            feats.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[lon_min,lat_min],[lon_max,lat_min],[lon_max,lat_max],[lon_min,lat_max],[lon_min,lat_min]]]},"properties":{"kind":kind,"value":v,"row":r,"col":c,"normalized_value":nv,"threshold":float(threshold),"source":"dataset_playback","model_source":"ridge_plume_baseline"}})
        feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[lon0,lat0]},"properties":{"kind":"source","title":"Release source"}})
        raw = payload.get("raw", {})
        plume_feature_count = len([f for f in feats if f.get("geometry", {}).get("type") == "Polygon"])
        return {"type":"FeatureCollection","features":feats,"metadata":{"source":"dataset_playback","feature_count":len(feats),"plume_polygon_count":plume_feature_count,"source_latitude":lat0,"source_longitude":lon0,"model_source":"dataset_input_inference","output_space":"ridge_prediction","prediction_shape":raw.get("prediction_shape"),"input_shape":raw.get("input_shape"),"active_window_id":raw.get("window_row",{}).get("window_id"),"active_scenario_id":payload.get("forecast",{}).get("scenario_id"),"georeferencing":"approximate_source_centered_grid"}}


    def _get_ridge_artifact(self) -> dict[str, Any]:
        if self._ridge_artifact is not None:
            return self._ridge_artifact
        model_path = self.config.ridge_model_path.expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        self._ridge_artifact = load_ridge_artifact(model_path)
        return self._ridge_artifact
    @staticmethod
    def _extract_plume_channel(target: np.ndarray) -> np.ndarray:
        if target.ndim == 4:
            return target[-1, 0]
        if target.ndim == 3:
            return target[0]
        if target.ndim == 2:
            return target
        raise ValueError(f"Expected target ndim to be 2, 3, or 4; got shape {target.shape}.")
