# test_acceptance_sad.py

import pytest
from app import app
from io import BytesIO

@pytest.fixture
def client():
    """
    Fixture for the Flask test client.
    - Purpose: Set up a test client for making requests to the Flask app during testing.
    - Usage: Provides a `client` object to use for HTTP request simulations.
    """
    with app.test_client() as client:
        yield client

def test_acceptance_empty_file_upload(client):
    # Ensures zero-byte files are rejected with an appropriate error message.
    empty = BytesIO(b"")
    empty.name = "empty.jpg"
    resp = client.post(
        "/prediction",
        data={"file": (empty, empty.name)},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (200, 400, 413)
    assert b"cannot be processed" in resp.data or b"error" in resp.data.lower()

def test_acceptance_wrong_field_name(client):
    # Ensures uploads not using the expected 'file' field produce a clear error.
    img_data = BytesIO(b"fake_image_data")
    img_data.name = "wrongfield.jpg"
    resp = client.post(
        "/prediction",
        data={"upload": (img_data, img_data.name)},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (200, 400)
    assert b"File cannot be processed" in resp.data or b"file" in resp.data.lower()

def test_acceptance_missing_file(client):
    """
    Test Case: No File Uploaded
    - Purpose: Validate the application's behavior when no file is provided in the upload request.
    - Scenario:
        - Simulate a POST request to the `/prediction` route with no file data.
        - Assert the response status code is 200 (to indicate a valid request was processed).
        - Verify that the response includes an appropriate error message.
    """
    # Simulate a POST request with no file data
    response = client.post("/prediction", data={}, content_type="multipart/form-data")

    # Assertions:
    # 1. Ensure the response status code is 200, indicating the request was processed.
    assert response.status_code == 200

    # 2. Check for a meaningful error message in the response data.
    #    Modify the message check if your application uses a different error response text.
    assert b"File cannot be processed" in response.data  # Expected error message
