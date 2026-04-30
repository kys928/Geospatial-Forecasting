from __future__ import annotations

from typing import Any

from plume.openremote.models import (
    ORAssetPayload,
    ORAttributeWrite,
    ORPredictedDatapointWrite,
    ORTimestampedAttributeWrite,
)
from plume.openremote.sink import OpenRemoteResultSink


class InMemoryOpenRemoteResultSink(OpenRemoteResultSink):
    def __init__(self) -> None:
        self.assets: list[ORAssetPayload] = []
        self.attribute_writes: list[ORAttributeWrite] = []
        self.timestamped_attribute_writes: list[ORTimestampedAttributeWrite] = []
        self.predicted_datapoint_writes: list[ORPredictedDatapointWrite] = []
        self._next_asset_id = 1

    async def upsert_asset(self, asset: ORAssetPayload) -> dict[str, Any]:
        self.assets.append(asset)
        created_id = asset.id or f"test-asset-{self._next_asset_id}"
        self._next_asset_id += 1
        return {"id": created_id}

    async def write_attribute(self, write: ORAttributeWrite) -> dict[str, Any]:
        self.attribute_writes.append(write)
        return {"ok": True}

    async def write_attributes(self, writes: list[ORAttributeWrite]) -> list[dict[str, Any]]:
        self.attribute_writes.extend(writes)
        return [{"ok": True} for _ in writes]

    async def write_attributes_with_timestamps(
        self,
        writes: list[ORTimestampedAttributeWrite],
    ) -> list[dict[str, Any]]:
        self.timestamped_attribute_writes.extend(writes)
        return [{"ok": True} for _ in writes]

    async def write_predicted_datapoints(self, write: ORPredictedDatapointWrite) -> dict[str, Any]:
        self.predicted_datapoint_writes.append(write)
        return {"ok": True}

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "assets": [item.model_dump(exclude_none=True) for item in self.assets],
            "attribute_writes": [item.model_dump(exclude_none=True) for item in self.attribute_writes],
            "timestamped_attribute_writes": [
                item.model_dump(exclude_none=True) for item in self.timestamped_attribute_writes
            ],
            "predicted_datapoint_writes": [
                item.model_dump(exclude_none=True) for item in self.predicted_datapoint_writes
            ],
        }
