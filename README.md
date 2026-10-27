# Road-Damage-Detection-YOLO
Road damage detection using YOLOv8n, YOLO11n, and YOLO12n to detect potholes, cracks, and manholes, with model comparison, evaluation, and a Flask web application.

# Road Damage Detection Using YOLO

A computer vision project for detecting road damage using YOLO object detection models. The system identifies three types of road damage: potholes, cracks, and manholes.

The project compares YOLOv8n, YOLO11n, and YOLO12n, evaluates their performance, experiments with hyperparameter tuning, and includes a Flask web application for image-based predictions.

## Project Overview

Road damage can affect road safety and driving conditions. This project explores deep learning-based object detection to identify road damage from images.

### Detection Classes

- **Pothole** — Detects potholes on road surfaces.
- **Crack** — Detects visible cracks in roads.
- **Manhole** — Detects manholes in road images.

## Dataset

- **Source:** [Road Damage Dataset — Kaggle](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
- **Images:** 2,009
- **Classes:** 3
- **Split:** 70% training, 20% validation, and 10% testing.

The dataset was organized into YOLO-compatible image and label directories.

## Models Evaluated

Three lightweight YOLO models were trained and compared:

- YOLOv8n
- YOLO11n
- YOLO12n

The comparison considers detection accuracy, precision, recall, inference speed, model size, and training time.

## Validation Results

The following results were obtained during baseline validation:

| Metric | YOLOv8n | YOLO11n | YOLO12n |
|---|---:|---:|---:|
| mAP@50 | 0.4594 | 0.4487 | **0.4629** |
| mAP@50–95 | **0.1971** | 0.1891 | 0.1964 |
| Precision | 0.5082 | **0.5330** | 0.4967 |
| Recall | 0.4844 | 0.4224 | **0.4854** |
| Inference time (ms/image) | 2.24 | **2.04** | 2.73 |
| Approx. FPS | 447 | **489** | 367 |
| Model size (MB) | 6.18 | **5.23** | 5.28 |
| Training time (minutes) | 5.87 | 6.74 | 9.02 |

YOLO12n achieved the highest validation mAP@50 and recall, while YOLOv8n achieved the highest mAP@50–95. YOLO11n achieved the highest precision and fastest measured inference time.

Based on the overall validation results, YOLO12n was selected for further evaluation.

*Note: Speed and FPS are approximate measurements and may vary depending on the hardware and runtime environment.*

## Test Set Evaluation

The selected YOLO12n baseline model was evaluated separately on the held-out test set.

| Metric | Result |
|---|---:|
| Test images | 201 |
| Ground-truth instances | 492 |
| mAP@50 | 0.4913 |
| mAP@50–95 | 0.2134 |
| Precision | 0.5337 |
| Recall | 0.4740 |

The evaluation reported approximately 0.7 ms preprocessing, 4.9 ms inference, and 1.0 ms postprocessing per image.

A separate mean IoU calculation produced **0.5941 (59.41%)** across 259 matched prediction–ground-truth pairs. This value is calculated only over matched boxes and should not be interpreted as an overall detection score, since missed ground-truth objects are not penalized in that calculation.

## Hyperparameter Tuning

Two additional YOLO12n experiments were conducted using 15 training epochs:

- Learning rate `lr0 = 0.001`
- Learning rate `lr0 = 0.0005`, with `lrf = 0.01` and `weight_decay = 0.0005`

Both experiments reported validation mAP@50 of 0.4088 and mAP@50–95 of 0.1693, below the baseline model's validation results. These experiments did not improve performance under the tested settings.

## Flask Web Application

The project includes a Flask web interface that allows users to upload road images and run predictions using the trained YOLO model.

The application displays detected road damage and prediction confidence scores.

![Flask Web Application](flask_app.png)

## Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Flask
- NumPy
- Matplotlib
- Kaggle Notebooks

## Repository Contents

- `road-damage-detection.ipynb` — Notebook containing dataset preparation, model training, evaluation, tuning, and visualizations.
- `README.md` — Project documentation.
- `LICENSE` — MIT License.

The Flask application files and model weights should also be included in the repository if they are intended to be distributed with the project.

## How to Run

1. Open the notebook in Kaggle.
2. Add the required Kaggle dataset to the notebook.
3. Run the setup and dataset preparation cells.
4. Train or load the required YOLO model.
5. Run the evaluation and visualization cells.
6. To use the Flask application, install its dependencies, place the trained model weights at the path expected by the application, and start the Flask server.

**Note:** The notebook uses Kaggle-specific input and working directories. These paths may need to be updated when running in another environment.

## Future Improvements

- Improve detection accuracy through additional training and hyperparameter experiments.
- Evaluate performance on more diverse road conditions and lighting scenarios.
- Improve deployment and inference efficiency.
- Expand testing across different environments.

## Author

**Marina Essam**

GitHub: [Marinastudent-hub](https://github.com/Marinastudent-hub)
