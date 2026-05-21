from fastapi import status
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import AppError, create_app


def build_client() -> TestClient:
    app = create_app(
        Settings(
            app_name="Test HR Assistant",
            app_env="test",
            app_debug=False,
            frontend_origin="http://localhost:3000",
        )
    )
    return TestClient(app)


def test_health_endpoint_returns_service_metadata() -> None:
    client = build_client()

    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "Test HR Assistant"
    assert body["environment"] == "test"
    assert body["version"] == "0.1.0"
    assert "timestamp" in body


def test_root_endpoint_returns_message() -> None:
    client = build_client()

    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Test HR Assistant API is running."}


def test_cors_allows_configured_frontend_origin() -> None:
    client = build_client()

    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_app_error_handler_returns_standard_error_shape() -> None:
    app = create_app(Settings(app_env="test", app_debug=False))

    @app.get("/raise-app-error")
    async def raise_app_error() -> None:
        raise AppError(code="example_error", message="Example failure.")

    client = TestClient(app)

    response = client.get("/raise-app-error")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {
        "error": {
            "code": "example_error",
            "message": "Example failure.",
            "details": None,
        }
    }
