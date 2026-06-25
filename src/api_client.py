# Adjuntar la API key en cada request.
# Reintentar en errores transitorios (429 y 5XX) con backoff exponencial.
# Respetar el gap mínimo entre requests para no gatillar rate limiting.
# Devolver el JSON ya parseado o lanzar ApiFootballError.

from __future__ import annotations

import time
from typing import Any

import requests

from .config import API_BASE_URL, API_FOOTBALL_KEY


class ApiFootballError(RuntimeError):
    """Error al comunicarse con la API (auth, cuota, red, etc.)."""


class ApiFootballClient:
    """
    Cliente reutilizable para API-Football v3.

    Uso:
        client = ApiFootballClient()
        data = client.get("status")
        data = client.get("fixtures", {"league": 39, "season": 2023})
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        timeout: int = 20,
        max_retries: int = 4,
        min_seconds_between_requests: float = 0.2,
    ) -> None:
        key = (api_key or API_FOOTBALL_KEY).strip()
        if not key:
            raise ApiFootballError(
                "No se encontró API_FOOTBALL_KEY. "
                "Verifica el archivo .env en la raíz del proyecto."
            )
        self._key      = key
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        self._retries  = max_retries
        self._min_gap  = min_seconds_between_requests
        self._last_ts  = 0.0
        self._session  = requests.Session()
        self._session.headers.update({"x-apisports-key": self._key})

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """GET /<path> con reintentos y backoff exponencial."""
        url     = f"{self._base_url}/{path.lstrip('/')}"
        backoff = 2.0

        for attempt in range(1, self._retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params or {}, timeout=self._timeout)
            except requests.RequestException as exc:
                if attempt == self._retries:
                    raise ApiFootballError(f"Error de red persistente: {exc}") from exc
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                payload = resp.json()
                self._check_api_errors(payload)
                return payload

            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue

            raise ApiFootballError(
                f"HTTP {resp.status_code} en {path}: {resp.text[:300]}"
            )

        raise ApiFootballError(f"Se agotaron los reintentos para {path}")

    def remaining_requests(self) -> int:
        """Devuelve los requests disponibles para hoy."""
        data = self.get("status")
        req  = data["response"]["requests"]
        return req["limit_day"] - req["current"]

    def _throttle(self) -> None:
        delta = time.time() - self._last_ts
        if delta < self._min_gap:
            time.sleep(self._min_gap - delta)
        self._last_ts = time.time()

    @staticmethod
    def _check_api_errors(payload: dict) -> None:
        errors = payload.get("errors")
        if errors and not (isinstance(errors, list) and len(errors) == 0):
            raise ApiFootballError(f"La API devolvió errores: {errors}")
