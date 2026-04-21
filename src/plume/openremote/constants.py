from __future__ import annotations

from plume.openremote.models import ProjectAssetType

ASSET_TYPE_SITE = ProjectAssetType.SITE.value
ASSET_TYPE_HAZARD_SOURCE = ProjectAssetType.HAZARD_SOURCE.value
ASSET_TYPE_SENSOR = ProjectAssetType.SENSOR.value
ASSET_TYPE_FORECAST_RUN = ProjectAssetType.FORECAST_RUN.value
ASSET_TYPE_FORECAST_ZONE = ProjectAssetType.FORECAST_ZONE.value

ATTR_SOURCE_ID = "sourceId"
ATTR_LOCATION = "location"
ATTR_POLLUTANT_TYPE = "pollutantType"
ATTR_RELEASE_RATE = "releaseRate"
ATTR_RELEASE_HEIGHT = "releaseHeight"
ATTR_SOURCE_STATUS = "sourceStatus"
ATTR_LAST_OBSERVATION_TIME = "lastObservationTime"
ATTR_SCENARIO_METADATA = "scenarioMetadata"

ATTR_WIND_U = "windU"
ATTR_WIND_V = "windV"
ATTR_WIND_SPEED = "windSpeed"
ATTR_WIND_DIRECTION = "windDirection"
ATTR_STABILITY_CLASS = "stabilityClass"
ATTR_BOUNDARY_LAYER_HEIGHT = "boundaryLayerHeight"

ATTR_SENSOR_ID = "sensorId"
ATTR_SENSOR_TYPE = "sensorType"
ATTR_OBSERVED_VALUE = "observedValue"
ATTR_OBSERVED_UNIT = "observedUnit"
ATTR_QUALITY_FLAG = "qualityFlag"
ATTR_OBSERVATION_METADATA = "observationMetadata"

ATTR_FORECAST_RUN_ID = "forecastRunId"
ATTR_SESSION_ID = "sessionId"
ATTR_BACKEND_NAME = "backendName"
ATTR_MODEL_NAME = "modelName"
ATTR_MODEL_VERSION = "modelVersion"
ATTR_RUN_STATUS = "runStatus"
ATTR_ISSUED_AT = "issuedAt"
ATTR_HORIZON_SECONDS = "horizonSeconds"
ATTR_SOURCE_ASSET_ID = "sourceAssetId"
ATTR_SITE_ASSET_ID = "siteAssetId"

ATTR_MAX_CONCENTRATION = "maxConcentration"
ATTR_MEAN_CONCENTRATION = "meanConcentration"
ATTR_AFFECTED_CELLS_ABOVE_THRESHOLD = "affectedCellsAboveThreshold"
ATTR_AFFECTED_AREA_M2 = "affectedAreaM2"
ATTR_AFFECTED_AREA_HECTARES = "affectedAreaHectares"
ATTR_DOMINANT_SPREAD_DIRECTION = "dominantSpreadDirection"
ATTR_THRESHOLD_USED = "thresholdUsed"
ATTR_CONFIDENCE_SCORE = "confidenceScore"
ATTR_EXPLANATION_TEXT = "explanationText"
ATTR_ALERT_LEVEL = "alertLevel"

ATTR_PLUME_FOOTPRINT_GEOJSON = "plumeFootprintGeojson"
ATTR_CENTROID = "centroid"
ATTR_BOUNDING_BOX = "boundingBox"
ATTR_GEOJSON_URL = "geojsonUrl"
ATTR_RASTER_METADATA = "rasterMetadata"
ATTR_GRID_SPEC = "gridSpec"
ATTR_SCENARIO_SNAPSHOT = "scenarioSnapshot"
ATTR_EXECUTION_METADATA = "executionMetadata"
ATTR_EXTERNAL_ARTIFACT_REF = "externalArtifactRef"

ATTR_ZONE_ID = "zoneId"
ATTR_ZONE_NAME = "zoneName"
ATTR_ZONE_GEOMETRY = "zoneGeometry"
ATTR_ZONE_TYPE = "zoneType"
ATTR_PREDICTED_MAX_CONCENTRATION = "predictedMaxConcentration"
ATTR_PREDICTED_MEAN_CONCENTRATION = "predictedMeanConcentration"
ATTR_PREDICTED_ARRIVAL_TIME = "predictedArrivalTime"
ATTR_PREDICTED_PEAK_TIME = "predictedPeakTime"
ATTR_PREDICTED_CLEAR_TIME = "predictedClearTime"
ATTR_RISK_LEVEL = "riskLevel"
ATTR_LATEST_FORECAST_RUN_ID = "latestForecastRunId"

META_STORE_DATA_POINTS = "storeDataPoints"
META_DATA_POINTS_MAX_AGE_DAYS = "dataPointsMaxAgeDays"
META_HAS_PREDICTED_DATA_POINTS = "hasPredictedDataPoints"
META_APPLY_PREDICTED_DATA_POINTS = "applyPredictedDataPoints"
