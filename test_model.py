import os
import pytest
from io import BytesIO
from PIL import Image
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result  # Adjust based on your structure

# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model = load_model("digit_model.h5")  # Adjust path as needed
    return model

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

def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3), "Processed image shape should be (1, 224, 224, 3)"

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, "Image pixel values should be normalized between 0 and 1"


def test_predict_result(model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"


# Advanced Tests

def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(model):
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_model_predictions_consistency(model):
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions), "Predictions for the same input should be consistent"
