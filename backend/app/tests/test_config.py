from app.config import Settings


def test_cors_origins_splits_comma_separated_values() -> None:
    settings = Settings(frontend_origin="http://localhost:3000, http://localhost:3001")

    assert settings.cors_origins == ["http://localhost:3000", "http://localhost:3001"]
