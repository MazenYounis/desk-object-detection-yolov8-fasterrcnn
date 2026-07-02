from pathlib import Path
from ultralytics import YOLO


def compare_models():
    """
    Runs both the pretrained YOLOv8n model and the fine-tuned YOLOv8n model
    on the same demo test images.

    This is used to compare the baseline model with the custom fine-tuned model.
    """

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data" / "test_images"

    fine_tuned_model_path = (
        project_root
        / "results"
        / "fine_tuned"
        / "yolov8n_desk_objects_v1"
        / "weights"
        / "best.pt"
    )

    output_dir = project_root / "results" / "comparison"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if not fine_tuned_model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {fine_tuned_model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained YOLOv8n model...")
    baseline_model = YOLO("yolov8n.pt")

    print("Running pretrained baseline model...")
    baseline_model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="pretrained_yolov8n",
        exist_ok=True,
        conf=0.25
    )

    print("Loading fine-tuned YOLOv8n model...")
    fine_tuned_model = YOLO(str(fine_tuned_model_path))

    print("Running fine-tuned model...")
    fine_tuned_model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="fine_tuned_yolov8n",
        exist_ok=True,
        conf=0.25
    )

    print("Comparison completed.")
    print(f"Pretrained results saved in: {output_dir / 'pretrained_yolov8n'}")
    print(f"Fine-tuned results saved in: {output_dir / 'fine_tuned_yolov8n'}")


if __name__ == "__main__":
    compare_models()