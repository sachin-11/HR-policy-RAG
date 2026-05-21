"""FastAPI application entrypoint."""

import json
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth_routes import router as auth_router
from app.api.chat_routes import router as chat_router
from app.api.document_routes import router as document_router
from app.api.session_routes import router as session_router
from app.api.schemas import ApiError, ErrorResponse, HealthResponse, MessageResponse
from app.config import Settings, get_settings
from app.observability.logging import configure_logging
from app.observability.middleware import add_observability_middleware

APP_VERSION = "0.1.0"


class AppError(Exception):
    """Application-level exception with a stable error code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


def build_error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""

    payload = ErrorResponse(error=ApiError(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)

    app = FastAPI(
        title=resolved_settings.app_name,
        version=APP_VERSION,
        debug=resolved_settings.app_debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    add_observability_middleware(app)
    app.include_router(auth_router)
    app.include_router(auth_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(chat_router)
    app.include_router(chat_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(document_router)
    app.include_router(document_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(session_router)
    app.include_router(session_router, prefix=resolved_settings.api_v1_prefix)

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return build_error_response(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        safe_errors = json.loads(json.dumps(exc.errors(), default=str))
        return build_error_response(
            code="validation_error",
            message="Request validation failed.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"errors": safe_errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        details = {"error": str(exc)} if resolved_settings.app_debug else None
        return build_error_response(
            code="internal_server_error",
            message="An unexpected error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )

    @app.get("/", response_model=MessageResponse, tags=["system"])
    async def root() -> MessageResponse:
        return MessageResponse(message=f"{resolved_settings.app_name} API is running.")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=resolved_settings.app_name,
            environment=resolved_settings.app_env,
            version=APP_VERSION,
        )

    return app


app = create_app()
