from pathlib import Path
from ultralytics import YOLO


def train_model():
    """
    Fine-tunes a pre-trained YOLOv8n model on the custom desk object dataset v2.

    Dataset classes:
    - bottle
    - cell_phone
    - cup
    - laptop
    - mouse
    """

    project_root = Path(__file__).resolve().parents[1]

    data_yaml = project_root / "data" / "processed" / "desk_object_detection_v3" / "data.yaml"
    output_dir = project_root / "results" / "fine_tuned"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("Starting fine-tuning on Dataset v3...")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        project=str(output_dir),
        name="yolov8n_desk_objects_v3",
        exist_ok=True,
        patience=10
    )

    print("Training completed.")
    print(f"Results saved in: {output_dir / 'yolov8n_desk_objects_v3'}")


if __name__ == "__main__":
    train_model()