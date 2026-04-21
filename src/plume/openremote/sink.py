from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import quote

import httpx

from plume.openremote.models import (
    ORAssetPayload,
    ORAttributeWrite,
    ORPredictedDatapointWrite,
    ORTimestampedAttributeWrite,
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
    Thin HTTP sink with basic request/error hardening.
    """

    def __init__(
        self,
        base_url: str,
        access_token: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("HttpOpenRemoteResultSink requires a non-empty base_url")
        if not access_token.strip():
            raise ValueError("HttpOpenRemoteResultSink requires a non-empty access_token")

        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{path}"

    def _asset_payload(self, asset: ORAssetPayload) -> dict[str, Any]:
        """
        The deployed OpenRemote versions used by projects typically expect `parentId`
        rather than `parent_id`. We shape the payload accordingly while keeping this
        adapter thin and version-cautious.
        """
        payload = {
            "id": asset.id,
            "name": asset.name,
            "type": asset.type,
            "realm": asset.realm,
            "parentId": asset.parent_id,
            "attributes": [attr.model_dump(exclude_none=True) for attr in asset.attributes],
            "metadata": asset.metadata,
        }
        return {k: v for k, v in payload.items() if v is not None}

    @staticmethod
    def _decode_response_json(response: httpx.Response, *, fallback: Any) -> Any:
        if not response.content:
            return fallback
        try:
            return response.json()
        except ValueError:
            return fallback

    @staticmethod
    def _raise_http_error(operation: str, response: httpx.Response) -> None:
        body = (response.text or "").strip()
        body_snippet = body[:300] if body else "<empty body>"
        raise RuntimeError(
            f"OpenRemote {operation} failed: HTTP {response.status_code} at "
            f"{response.request.method} {response.request.url}; body={body_snippet}"
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any,
        operation: str,
    ) -> httpx.Response:
        url = self._url(path)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self.headers) as client:
            try:
                response = await client.request(method, url, json=json_payload)
            except httpx.RequestError as exc:
                raise RuntimeError(f"OpenRemote {operation} request failed: {exc}") from exc

        if response.status_code >= 400:
            self._raise_http_error(operation, response)
        return response

    async def upsert_asset(self, asset: ORAssetPayload) -> dict[str, Any]:
        """
        Simplest contract:
        - POST if no ID
        - PUT if ID exists
        """
        payload = self._asset_payload(asset)
        method = "PUT" if asset.id else "POST"
        response = await self._request(
            method,
            "/asset",
            json_payload=payload,
            operation="asset upsert",
        )
        return self._decode_response_json(response, fallback={})

    async def write_attribute(self, write: ORAttributeWrite) -> dict[str, Any]:
        response = await self._request(
            "PUT",
            f"/asset/{quote(write.asset_id, safe='')}/attribute/{quote(write.attribute_name, safe='')}",
            json_payload=write.value,
            operation="single attribute write",
        )
        return self._decode_response_json(response, fallback={})

    async def write_attributes(self, writes: list[ORAttributeWrite]) -> list[dict[str, Any]]:
        payload = [
            {
                "id": w.asset_id,
                "name": w.attribute_name,
                "value": w.value,
            }
            for w in writes
        ]
        response = await self._request(
            "PUT",
            "/asset/attributes",
            json_payload=payload,
            operation="bulk attribute write",
        )
        return self._decode_response_json(response, fallback=[])

    async def write_attributes_with_timestamps(
        self,
        writes: list[ORTimestampedAttributeWrite],
    ) -> list[dict[str, Any]]:
        """
        Endpoint name is based on the documented 'Update attribute values with timestamps' capability.
        Some OpenRemote deployments/version lines can vary endpoint naming.
        Adjust this path if your target deployment expects a different timestamped-write route.
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
        response = await self._request(
            "PUT",
            "/asset/attributes/timestamp",
            json_payload=payload,
            operation="timestamped attribute write",
        )
        return self._decode_response_json(response, fallback=[])

    async def write_predicted_datapoints(self, write: ORPredictedDatapointWrite) -> dict[str, Any]:
        payload = [
            {
                "timestamp": int(dp.timestamp.timestamp() * 1000),
                "value": dp.value,
            }
            for dp in write.datapoints
        ]
        response = await self._request(
            "PUT",
            (
                f"/asset/predicted/{quote(write.asset_id, safe='')}/"
                f"{quote(write.attribute_name, safe='')}"
            ),
            json_payload=payload,
            operation="predicted datapoint write",
        )
        return self._decode_response_json(response, fallback={})
