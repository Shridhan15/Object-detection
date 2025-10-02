from flask import Flask, render_template, request  # Flask framework for web app
import os  # For file path operations
from image_analysis import analyze_image  # Our module to analyze images 

# Initialize Flask app
app = Flask(__name__)

# Folder where uploaded images will be saved
app.config['UPLOAD_FOLDER'] = 'static'

@app.route("/", methods=["GET", "POST"])
def index():
    """
    this function handles the main page:
    GET request: shows the upload form
    POST request: processes the uploaded image and shows analysis results
    """
    report = {}       # store image analysis results
    img_filename = None  # Name of the uploaded image file

    if request.method == "POST":
        # Get uploaded image from the form
        img_file = request.files["image"]
        if img_file:
            img_filename = img_file.filename  # Get the original filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)  # Full path to save the file
            img_file.save(img_path)  # Save uploaded image to the static folder

            # Analyze the saved image using our custom image analysis module
            report = analyze_image(img_path)

    # Render the HTML template and pass the report and image filename
    return render_template("index.html", report=report, img_filename=img_filename)

if __name__ == "__main__":
    # Run the Flask app in debug mode for development
    app.run(debug=True)
