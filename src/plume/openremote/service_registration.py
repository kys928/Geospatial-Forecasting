from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx


@dataclass(frozen=True)
class OpenRemoteServiceRegistrationSettings:
    enabled: bool
    manager_api_url: str
    service_id: str
    label: str
    version: str
    icon: str
    homepage_url: str
    global_service: bool
    heartbeat_interval_seconds: int
    access_token: str | None


class OpenRemoteServiceRegistrar:
    def __init__(
        self,
        settings: OpenRemoteServiceRegistrationSettings,
        *,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.settings = settings
        self.registered = False
        self.instance_id: int | None = None
        self.last_registered_at: str | None = None
        self.last_heartbeat_at: str | None = None
        self.last_error: str | None = None
        self.heartbeat_task: asyncio.Task | None = None
        self._client_factory = client_factory or httpx.AsyncClient

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _auth_headers(self) -> dict[str, str]:
        if not self.settings.access_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.access_token}"}

    def _validate(self) -> bool:
        if not self.settings.enabled:
            return False
        missing = []
        if not self.settings.manager_api_url.strip():
            missing.append("manager_api_url")
        if not self.settings.homepage_url.strip():
            missing.append("homepage_url")
        if not self.settings.service_id.strip():
            missing.append("service_id")
        if not self.settings.label.strip():
            missing.append("label")
        if missing:
            self.last_error = "Missing required settings: " + ", ".join(missing)
            return False
        return True

    async def register(self) -> dict[str, object] | None:
        if not self._validate():
            return None
        endpoint = "/service/global" if self.settings.global_service else "/service"
        payload = {
            "serviceId": self.settings.service_id,
            "version": self.settings.version,
            "icon": self.settings.icon,
            "label": self.settings.label,
            "homepageUrl": self.settings.homepage_url,
            "status": "AVAILABLE",
        }
        try:
            async with self._client_factory() as client:
                response = await client.post(
                    f"{self.settings.manager_api_url.rstrip('/')}{endpoint}",
                    json=payload,
                    headers=self._auth_headers(),
                )
            response.raise_for_status()
            data = response.json()
            self.instance_id = int(data.get("instanceId")) if data.get("instanceId") is not None else None
            self.registered = self.instance_id is not None
            self.last_registered_at = self._now_iso() if self.registered else None
            self.last_error = None
            return data
        except Exception as exc:
            self.last_error = str(exc)
            return None

    async def heartbeat(self) -> bool:
        if not self.settings.enabled:
            return False
        if not self.registered or self.instance_id is None:
            return False
        path = f"/service/{self.settings.service_id}/{self.instance_id}"
        try:
            async with self._client_factory() as client:
                response = await client.put(
                    f"{self.settings.manager_api_url.rstrip('/')}{path}",
                    headers=self._auth_headers(),
                )
            if response.status_code == 404:
                self.registered = False
                self.instance_id = None
                self.last_error = "Service registration not found (404)"
                return False
            response.raise_for_status()
            self.last_heartbeat_at = self._now_iso()
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    async def deregister(self) -> bool:
        if not self.settings.enabled:
            return False
        if self.instance_id is None:
            return False
        path = f"/service/{self.settings.service_id}/{self.instance_id}"
        try:
            async with self._client_factory() as client:
                response = await client.delete(
                    f"{self.settings.manager_api_url.rstrip('/')}{path}",
                    headers=self._auth_headers(),
                )
            response.raise_for_status()
            self.registered = False
            self.instance_id = None
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    async def _heartbeat_loop(self) -> None:
        while True:
            if self.settings.enabled:
                if not self.registered:
                    await self.register()
                else:
                    await self.heartbeat()
            await asyncio.sleep(max(1, int(self.settings.heartbeat_interval_seconds)))

    def start_background_heartbeat(self) -> None:
        if not self.settings.enabled or self.heartbeat_task is not None:
            return
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_background_heartbeat(self) -> None:
        if self.heartbeat_task is None:
            return
        self.heartbeat_task.cancel()
        try:
            await self.heartbeat_task
        except asyncio.CancelledError:
            pass
        finally:
            self.heartbeat_task = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.settings.enabled,
            "registered": self.registered,
            "service_id": self.settings.service_id,
            "instance_id": self.instance_id,
            "last_registered_at": self.last_registered_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_error": self.last_error,
            "global_service": self.settings.global_service,
            "manager_api_url": self.settings.manager_api_url,
            "homepage_url": self.settings.homepage_url,
            "heartbeat_interval_seconds": self.settings.heartbeat_interval_seconds,
        }
