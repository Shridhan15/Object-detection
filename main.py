from flask import Flask, render_template, request
import os
from image_analysis import analyze_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route("/", methods=["GET", "POST"])
def index():
    report = {}
    img_filename = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img_filename = img_file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            img_file.save(img_path)

            # Analyze image
            report = analyze_image(img_path)

    return render_template(
        "index.html",
        report=report,
        img_filename=img_filename
    )

if __name__ == "__main__":
    app.run(debug=True)
