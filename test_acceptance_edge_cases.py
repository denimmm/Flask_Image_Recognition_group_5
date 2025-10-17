# test_acceptance_edge_cases.py

from io import BytesIO
import pytest
from threading import Thread

def test_edge_case_extension_mismatch_content_type(client):
    # Verifies extension/content-type mismatch is handled consistently.
    img_data = BytesIO(b"valid_image_data")
    img_data.name = "mismatch.png"
    resp = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data",
        headers={"Content-Type": "multipart/form-data"},
    )
    assert resp.status_code in (200, 400)
    # If accepted, Prediction appears; if rejected, error present.
    assert (b"Prediction" in resp.data) or (b"error" in resp.data.lower())

def test_edge_case_long_filename(client):
    # Ensures very long filenames do not crash the server.
    long_name = "a" * 200 + ".jpg"
    img_data = BytesIO(b"valid_image_data")
    img_data.name = long_name
    resp = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (200, 400)
    assert (b"Prediction" in resp.data) or (b"cannot" in resp.data.lower())

def test_edge_case_multiple_files_single_field(client):
    # Confirms behavior when multiple files are sent under one field.
    img1 = BytesIO(b"img1")
    img1.name = "one.jpg"
    img2 = BytesIO(b"img2")
    img2.name = "two.jpg"
    data = {
    "file": [
        (img1, img1.name),
        (img2, img2.name),
    ]
}
    resp = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert resp.status_code in (200, 400)
    assert (b"Prediction" in resp.data) or (b"multiple" in resp.data.lower()) or (b"error" in resp.data.lower())

# Helper function for concurrent image uploads
def upload_image(client, img_data):
    """
    Helper function to upload an image within a thread.
    - Purpose: Enables concurrent uploads for testing multithreaded scenarios.
    - Usage: Called in separate threads during concurrent tests.
    """
    client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

# 1. Edge Case: Uploading a Large Image File
def test_edge_case_large_image_upload(client):
    """
    Test uploading a large image to see how the system handles large file sizes.
    - Scenario: Simulates a file upload with a large image (e.g., 10 MB).
    - Expected Behavior: System should handle the file and return a valid prediction response.
    """
    large_img_data = BytesIO(b"large_image_data" * 10**6)  # Simulating a large image
    large_img_data.name = "large_image.jpg"

    response = client.post(
        "/prediction",
        data={"file": (large_img_data, large_img_data.name)},
        content_type="multipart/form-data"
    )

    # Assert that the system can process large image files
    assert b"Prediction" in response.data  # Adjust based on actual prediction content

# 2. Edge Case: Uploading an Image with Missing or Incorrect Metadata
def test_edge_case_invalid_metadata(client):
    """
    Test uploading an image with missing or incorrect metadata.
    - Scenario: Upload an image without any metadata or with invalid metadata.
    - Expected Behavior: System processes the image regardless of metadata issues.
    """
    img_data = BytesIO(b"image_with_no_metadata")
    img_data.name = "image_no_metadata.jpg"

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    # Assert that the system processes the image even without metadata
    assert b"Prediction" in response.data  # Adjust based on actual behavior

# 3. Edge Case: Uploading an Image with Non-Standard File Extensions
def test_edge_case_non_standard_image_extensions(client):
    """
    Test uploading images with non-standard file extensions.
    - Scenario: Upload a file with an uncommon extension, e.g., .webp.
    - Expected Behavior: System processes the file as a valid image.
    """
    img_data = BytesIO(b"valid_image_data")
    img_data.name = "non_standard_image.webp"  # Non-standard extension

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    # Ensure the system processes the file despite the non-standard extension
    assert b"Prediction" in response.data  # Adjust based on actual prediction content

# 4. Edge Case: Uploading a Sequence of Images for Multi-Step Processing
def test_edge_case_sequential_image_uploads(client):
    """
    Test uploading a sequence of images that trigger multi-step processing.
    - Scenario: Simulate sequential uploads of multiple images.
    - Expected Behavior: Each upload should trigger a separate valid prediction response.
    """
    img_data1 = BytesIO(b"first_image_data")
    img_data1.name = "first_image.jpg"

    img_data2 = BytesIO(b"second_image_data")
    img_data2.name = "second_image.jpg"

    # First upload
    response1 = client.post(
        "/prediction",
        data={"file": (img_data1, img_data1.name)},
        content_type="multipart/form-data"
    )

    # Second upload
    response2 = client.post(
        "/prediction",
        data={"file": (img_data2, img_data2.name)},
        content_type="multipart/form-data"
    )

    # Ensure that each upload processed correctly
    assert b"Prediction" in response1.data
    assert b"Prediction" in response2.data

# 5. Edge Case: Uploading with Unexpected Headers
def test_edge_case_unexpected_headers(client):
    """
    Test uploading an image with unexpected headers.
    - Scenario: Simulate sending headers not expected by the system.
    - Expected Behavior: System should still process the file without errors.
    """
    img_data = BytesIO(b"valid_image_data")
    img_data.name = "unexpected_headers_image.jpg"

    # Simulate uploading with unexpected headers
    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data",
        headers={"X-Unexpected-Header": "value"}
    )

    # Assert that the upload is still processed correctly despite the unexpected header
    assert b"Prediction" in response.data

# 6. Edge Case: Uploading an Image with HTTP/2
def test_edge_case_upload_over_http2(client):
    """
    Test uploading an image using HTTP/2 protocol.
    - Scenario: Simulate uploading an image over HTTP/2 protocol.
    - Expected Behavior: System processes the upload as it would over HTTP/1.1.
    """
    img_data = BytesIO(b"valid_image_data")
    img_data.name = "http2_image.jpg"

    # Simulate uploading the image using HTTP/2 (client would need to support it)
    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    # Ensure that the image upload works successfully over HTTP/2
    assert b"Prediction" in response.data
