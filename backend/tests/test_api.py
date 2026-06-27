from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import allowed_origins, app, service


def test_health_returns_200_when_model_is_ready(monkeypatch):
    monkeypatch.setattr(service, "load", lambda: None)
    service.interpreter = object()
    service.load_error = None

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(service, "load", lambda: None)
    service.interpreter = None
    service.load_error = "model failed"

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_predict_rejects_non_image_upload(monkeypatch):
    monkeypatch.setattr(service, "load", lambda: None)

    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415


def test_local_cors_preflight_is_allowed(monkeypatch):
    monkeypatch.setattr(service, "load", lambda: None)

    with TestClient(app) as client:
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_allowed_origins_reads_comma_separated_environment(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://oct.vercel.app/, https://oct-preview.vercel.app",
    )

    assert allowed_origins() == [
        "https://oct.vercel.app",
        "https://oct-preview.vercel.app",
    ]
