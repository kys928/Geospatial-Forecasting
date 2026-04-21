from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from src.plume.openremote.models import (
    ORAssetPayload,
    ORAttributeWrite,
    ORTimestampedAttributeWrite,
    ORPredictedDatapointWrite,
)


class OpenRemoteResultSink(ABC):
    """
    Project-facing interface.
    Keep it narrow and demoable.
    """

    @abstractmethod
    async def upsert_asset(self, asset: ORAssetPayload) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def write_attribute(self, write: ORAttributeWrite) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def write_attributes(self, writes: list[ORAttributeWrite]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def write_attributes_with_timestamps(
        self,
        writes: list[ORTimestampedAttributeWrite],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def write_predicted_datapoints(self, write: ORPredictedDatapointWrite) -> dict[str, Any]:
        raise NotImplementedError


class HttpOpenRemoteResultSink(OpenRemoteResultSink):
    """
    This is intentionally thin. You can harden it later.
    """

    def __init__(
        self,
        base_url: str,
        access_token: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        self.timeout_seconds = timeout_seconds

    async def upsert_asset(self, asset: ORAssetPayload) -> dict[str, Any]:
        """
        Simplest contract:
        - POST if no ID
        - PUT if ID exists
        """
        payload = asset.model_dump(exclude_none=True)

        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            if asset.id:
                response = await client.put(f"{self.base_url}/asset", json=payload)
            else:
                response = await client.post(f"{self.base_url}/asset", json=payload)

            response.raise_for_status()
            return response.json()

    async def write_attribute(self, write: ORAttributeWrite) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = await client.put(
                f"{self.base_url}/asset/{write.asset_id}/attribute/{write.attribute_name}",
                json=write.value,
            )
            response.raise_for_status()
            return response.json()

    async def write_attributes(self, writes: list[ORAttributeWrite]) -> list[dict[str, Any]]:
        payload = [
            {
                "id": w.asset_id,
                "name": w.attribute_name,
                "value": w.value,
            }
            for w in writes
        ]
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = await client.put(f"{self.base_url}/asset/attributes", json=payload)
            response.raise_for_status()
            return response.json()

    async def write_attributes_with_timestamps(
        self,
        writes: list[ORTimestampedAttributeWrite],
    ) -> list[dict[str, Any]]:
        """
        Endpoint name is based on the documented 'Update attribute values with timestamps' capability.
        Adjust path if your deployed OpenRemote version differs.
        """
        payload = [
            {
                "id": w.asset_id,
                "name": w.attribute_name,
                "value": w.value,
                "timestamp": int(w.timestamp.timestamp() * 1000),
            }
            for w in writes
        ]
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = await client.put(f"{self.base_url}/asset/attributes/timestamp", json=payload)
            response.raise_for_status()
            return response.json()

    async def write_predicted_datapoints(self, write: ORPredictedDatapointWrite) -> dict[str, Any]:
        payload = [
            {
                "timestamp": int(dp.timestamp.timestamp() * 1000),
                "value": dp.value,
            }
            for dp in write.datapoints
        ]
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = await client.put(
                f"{self.base_url}/asset/predicted/{write.asset_id}/{write.attribute_name}",
                json=payload,
            )
            response.raise_for_status()
            return response.json()