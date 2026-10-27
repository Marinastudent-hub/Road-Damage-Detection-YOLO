
from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_IMAGES = Path(
    "/kaggle/input/datasets/lorenzoarcioni/"
    "road-damage-dataset-potholes-cracks-and-manholes/data/images"
)

MODEL_PATH = BASE_DIR / "best.pt"
model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []
    error = None
    selected_image = ""

    sample_images = (
        sorted(
            p.name for p in DATASET_IMAGES.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if DATASET_IMAGES.exists()
        else []
    )

    if request.method == "POST":
        selected_image = request.form.get("sample_image", "").strip()
        uploaded_file = request.files.get("image")
        image = None

        if selected_image:
            if selected_image not in sample_images:
                error = "Invalid dataset image selection."
            else:
                image = cv2.imread(
                    str(DATASET_IMAGES / selected_image)
                )

        elif uploaded_file and uploaded_file.filename:
            file_bytes = np.frombuffer(
                uploaded_file.read(), dtype=np.uint8
            )
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        else:
            error = "Choose a dataset image or upload your own."

        try:
            confidence = float(
                request.form.get("confidence", 0.25)
            )
            confidence = max(0.1, min(0.9, confidence))
        except (ValueError, TypeError):
            confidence = 0.25

        if not error:
            if image is None:
                error = "Could not read this image."
            else:
                result = model.predict(
                    image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )[0]

                output_name = f"{uuid.uuid4().hex}.jpg"
                output_path = RESULTS_DIR / output_name
                cv2.imwrite(str(output_path), result.plot())

                result_image = "results/" + output_name

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        detections.append({
                            "name": CLASS_NAMES.get(
                                cls_id, str(cls_id)
                            ),
                            "confidence": round(
                                float(box.conf.item()) * 100, 2
                            )
                        })

    return render_template(
        "index.html",
        sample_images=sample_images,
        selected_image=selected_image,
        result_image=result_image,
        detections=detections,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
