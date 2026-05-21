"""HTTP middleware for request correlation and latency."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.observability.logging import log_event, reset_request_id, set_request_id

REQUEST_HEADER = "x-request-id"
RESPONSE_HEADER = "X-Request-ID"

http_access_logger = logging.getLogger("app.observability.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add or propagate a request ID and emit an access-style JSON line per request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_HEADER)
        request_id = incoming or str(uuid.uuid4())
        token = set_request_id(request_id)

        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            status_code = response.status_code if response is not None else 500

            log_event(
                http_access_logger,
                event="http.request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=elapsed_ms,
                client_host=request.client.host if request.client else None,
            )

            if response is not None:
                response.headers[RESPONSE_HEADER] = request_id

            reset_request_id(token)


def add_observability_middleware(app: FastAPI) -> None:
    """Register middleware; add last so it runs first on incoming requests (outermost)."""

    app.add_middleware(RequestContextMiddleware)
