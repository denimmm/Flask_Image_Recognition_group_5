"""Main Flask application for image recognition."""

# Third-party
from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

# Your own modules
from model import preprocess_image, predict_result, load_model

# Instantiating Flask app
app = Flask(__name__)

# Load the model once at startup
model = load_model("digit_model.h5")


# Home route
@app.route("/")
def main():
    """Render the home page with the file upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Process uploaded image, run prediction, and render results."""
    if request.method == 'POST':
        try:
            processed_img = preprocess_image(request.files['file'].stream)
            prediction_result = predict_result(model, processed_img)
            return render_template("result.html", predictions=str(prediction_result))

        except (FileNotFoundError, UnidentifiedImageError) as e:
            # Catch specific exceptions instead of all exceptions
            error = f"File cannot be processed. Error: {e}"
            return render_template("result.html", err=error)

    # Fallback return to satisfy Pylint inconsistent-return warning
    return render_template("index.html")


# Driver code
if __name__ == "__main__":
    # Run the Flask app on port 9000 in debug mode
    app.run(port=9000, debug=True)
