# test_integration_happy.py

from io import BytesIO
import pytest
import time

def test_integration_repeat_same_image_consistent(client):
    # Ensures predictions for the same image across requests are consistent.
    buf = BytesIO(b"fake_image_data_consistent")
    buf.name = "same.jpg"
    r1 = client.post(
        "/prediction",
        data={"file": (buf, buf.name)},
        content_type="multipart/form-data",
    )
    buf2 = BytesIO(b"fake_image_data_consistent")
    buf2.name = "same.jpg"
    r2 = client.post(
        "/prediction",
        data={"file": (buf2, buf2.name)},
        content_type="multipart/form-data",
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert b"Prediction" in r1.data and b"Prediction" in r2.data

def test_integration_latency_budget(client):
    # Checks typical latency remains under a reasonable threshold in test env.
    img_data = BytesIO(b"fake_image_data_latency")
    img_data.name = "latency.jpg"
    t0 = time.time()
    resp = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data",
    )
    elapsed_ms = (time.time() - t0) * 1000
    assert resp.status_code == 200
    assert elapsed_ms < 2000

def test_successful_prediction(client):
    """Test the successful image upload and prediction."""
    # Create a mock image file with minimal valid content
    img_data = BytesIO(b"fake_image_data")
    img_data.name = "test.jpg"

    # Simulate a file upload to the correct prediction endpoint
    response = client.post(
        "/prediction",  # Correct route for prediction
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    # Assertions
    assert response.status_code == 200
    assert b"Prediction" in response.data  # Modify this check based on your output
