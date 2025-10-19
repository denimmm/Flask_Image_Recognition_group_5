"""Model loading and prediction logic."""

# Third-party
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

model = load_model("digit_model.h5")

def preprocess_img(image):
    """Prepare an image for model prediction.

    Args:
        image: A file-like object or path to the image.

    Returns:
        np.ndarray: Preprocessed image array ready for model input.
    """
    op_img = Image.open(image)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(image):
    """Predict the class label of a preprocessed image using the given model.

    Args:
        model (keras.Model): Loaded Keras model.
        image (np.ndarray): Preprocessed image array (from preprocess_img).

    Returns:
        int: Predicted class label as an integer.
    """
    pred = model.predict(image)
    return np.argmax(pred[0], axis=-1)
