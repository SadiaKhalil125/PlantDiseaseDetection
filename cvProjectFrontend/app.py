# from flask import Flask, render_template, request
# from yolov_model import detect_with_model
# import os
# from treatments import TREATMENTS

# app = Flask(__name__)


# @app.route("/", methods=["GET", "POST"])
# def index():
#     result_img = None
#     label = None
#     if request.method == "POST":
#         file = request.files["image"]
#         model_type = request.form.get("model_type", "yolov5")
        
#         if file:
#             img_path = os.path.join("static", "upload.jpg")
#             file.save(img_path)
#             result_img, label = detect_with_model(img_path, model_type)

#     return render_template("index.html", result_img=result_img, label=label)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request
from yolov_model import detect_with_model
import os
from treatments import TREATMENTS

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    label = None
    description = None
    recommendations = None

    if request.method == "POST":
        file = request.files["image"]
        model_type = request.form.get("model_type", "yolov5")
        
        if file:
            img_path = os.path.join("static", "upload.jpg")
            file.save(img_path)
            result_img, label = detect_with_model(img_path, model_type)

            # Fetch treatment data
            if label in TREATMENTS:
                description = TREATMENTS[label]["description"]
                recommendations = TREATMENTS[label]["recommendations"]

    return render_template(
        "index.html",
        result_img=result_img,
        label=label,
        description=description,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)
