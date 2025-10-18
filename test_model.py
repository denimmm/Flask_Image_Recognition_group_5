"""Unit tests for model loading and prediction logic."""

# Third-party
import pytest
from io import BytesIO
from PIL import Image
import numpy as np

# Your own modules
from model import load_model, preprocess_image, predict_result


# Load the model once for all tests
@pytest.fixture(scope="module")
def model_instance():
    """Load the ML model once before running tests."""
    return load_model("digit_model.h5")  # Adjust path as needed


def test_preprocess_img_accepts_pil_image():
    # Verifies preprocess_img can accept a PIL Image object and produce the expected normalized shape.
    img = Image.new("RGB", (300, 200), color=(123, 111, 222))
    arr = preprocess_img(img)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0

def test_preprocess_img_accepts_numpy_array():
    # Verifies preprocess_img can accept a NumPy array and produce the expected normalized shape.
    np_img = np.random.randint(0, 256, size=(180, 240, 3), dtype=np.uint8)
    arr = preprocess_img(np_img)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0

def test_preprocess_img_grayscale_promotes_to_rgb():
    # Ensures grayscale input becomes 3-channel RGB after preprocessing.
    gray = Image.new("L", (128, 128), color=200)
    arr = preprocess_img(gray)
    assert arr.shape == (1, 224, 224, 3)

def test_preprocess_img_rgba_drops_alpha():
    # Ensures RGBA input drops alpha channel and becomes 3 channels.
    rgba = Image.new("RGBA", (256, 256), color=(10, 20, 30, 40))
    arr = preprocess_img(rgba)
    assert arr.shape == (1, 224, 224, 3)

def test_preprocess_img_invalid_bytes_raises():
    # Confirms corrupt bytes raise a meaningful exception in preprocessing.
    with pytest.raises(Exception):
        bad = BytesIO(b"\x00\x01notanimage")
        bad.name = "corrupt.jpg"
        preprocess_img(bad)

def test_preprocess_img_deterministic_output(tmp_path):
    # Ensures repeated preprocessing on the same file yields identical arrays.
    p = tmp_path / "const.jpg"
    Image.new("RGB", (100, 100), color=(50, 60, 70)).save(p)
    a = preprocess_img(str(p))
    b = preprocess_img(str(p))
    np.testing.assert_allclose(a, b, rtol=0, atol=0)

def test_predict_result_rejects_wrong_shape(model):
    # Verifies predict_result raises an error for input missing the batch dimension.
    bad = np.random.rand(224, 224, 3).astype(np.float32)
    with pytest.raises(Exception):
        predict_result(bad)

def test_predict_result_output_type(model, tmp_path):
    # Ensures predict_result returns an integer class index.
    p = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(p)
    arr = preprocess_img(str(p))
    pred = predict_result(arr)
    assert isinstance(pred, (int, np.integer))

# Basic Tests

def test_preprocess_image():
    """Test the preprocess_image function for shape and normalization."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img_local = preprocess_image(img_path)

    assert processed_img_local.shape == (1, 224, 224, 3), (
        "Processed image shape should be (1, 224, 224, 3)"
    )

    assert np.min(processed_img_local) >= 0 and np.max(processed_img_local) <= 1, (
        "Image pixel values should be normalized"
    )


def test_predict_result(model_instance):
    """Test the predict_result function for correct output type."""
    img_path = "test_images/4/Sign 4 (92).jpeg"
    processed_img_local = preprocess_image(img_path)

    prediction_local = predict_result(model_instance, processed_img_local)
    assert isinstance(prediction_local, (int, np.integer)), (
        "Prediction should be an integer class index"
    )


# Advanced Tests

def test_invalid_image_path():
    """Test preprocess_image with an invalid image path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_image("invalid/path/to/image.jpeg")


def test_image_shape_on_prediction(model_instance):
    """Test that predict_result returns an integer for a valid image."""
    img_path = "test_images/5/Sign 5 (86).jpeg"
    processed_img_local = preprocess_image(img_path)
    prediction_local = predict_result(model_instance, processed_img_local)
    assert isinstance(prediction_local, (int, np.integer)), "Prediction should be an integer"


def test_model_predictions_consistency(model_instance):
    """Test that predictions for the same input are consistent across multiple runs."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img_local = preprocess_image(img_path)

    predictions_local = [predict_result(model_instance, processed_img_local) for _ in range(5)]
    assert all(p == predictions_local[0] for p in predictions_local), (
        "Predictions should be consistent for the same input"
    )
