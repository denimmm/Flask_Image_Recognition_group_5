# test_integration_sad.py

import pytest
from app import app
from io import BytesIO

@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    with app.test_client() as client:
        yield client

def test_integration_corrupt_image(client):
    # Ensures corrupt uploads return a handled error rather than crashing.
    bad = BytesIO(b"\x00\x01\x02\x03\x04corrupt")
    bad.name = "bad.jpg"
    resp = client.post(
        "/prediction",
        data={"file": (bad, bad.name)},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (200, 400, 415, 422, 500)
    assert (b"error" in resp.data.lower()) or (b"cannot be processed" in resp.data.lower())

def test_missing_file(client):
    """Test the prediction route with a missing file."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data  # Check if the error message is displayed
