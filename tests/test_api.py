
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_prediction_rejects_invalid_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"This is not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
