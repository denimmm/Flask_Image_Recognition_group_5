# -----------------------------------------------------------
# CSCN73010 – Assignment 4
# Automated Acceptance Tests (Step 1)
#
# This file contains two acceptance tests following the
# GIVEN / WHEN / THEN format from Week 11.
# Tests are automated using pytest and use Flask's
# test client as the application driver layer.
# ---------------------------------------------------------

import pytest
from app import app
from io import BytesIO

#------------------------------------------------------------
# Application Driver Layer:
# The Flask test client acts as the driver that interacts with the running app.
# This keeps the acceptance test implementation decoupled from the UI.
#------------------------------------------------------------------

@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    with app.test_client() as client:
        yield client

# -------------------------------------------------------------
# Acceptance Test ACCEPT_01
# Title: Successful Image Upload and Prediction Display
#
# Acceptance Criteria (GIVEN / WHEN / THEN):
#   GIVEN the user is on the prediction upload page
#   WHEN the user uploads a valid image file (.jpg, .png)
#   THEN the system MUST return a 200 response
#   AND display a prediction result in the output.
#
# Test Steps:
#   1. Create a mock valid image file in memory.
#   2. POST the file to the /prediction route.
#   3. Capture the server response.
#
# Expected Result:
#   - HTTP 200 status
#   - Response contains the token "Prediction"
# ----------------------------------------------------------------

def test_at_01_successful_image_upload_and_prediction_display(client):
    """Acceptance Test AT-01: Successful Image Upload and Prediction Display"""

    # Create a small valid RGB PNG in memory using PIL so PIL.Image.open succeeds.
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(73, 109, 137))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "valid_image.png"

    response = client.post(
        "/prediction",
        data={"file": (buf, buf.name)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data or b"prediction" in response.data

# ---------------------------------------------------------
# Acceptance Test ACCEPT_02
# Title: Error Handling When No File Is Uploaded
#
# Acceptance Criteria (GIVEN / WHEN / THEN):
#   GIVEN the user is on the upload page
#   WHEN the user submits the form WITHOUT selecting a file
#   THEN the system MUST return a clear error message
#   AND must NOT crash or return an unhandled exception.
#
# Test Steps:
#   1. Submit a POST request to /prediction with no file.
#   2. Capture the server response.
#
# Expected Result:
#   - HTTP 200 or 400
#   - Response contains "File cannot be processed" or meaningful error text.
# ---------------------------------------------------------

def test_at_02_error_handling_when_no_file_uploaded(client):
    """Acceptance Test AT-02: Error Handling When No File Is Uploaded"""

    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    

    assert response.status_code in (200, 400)
    assert b"File cannot be processed." in response.data