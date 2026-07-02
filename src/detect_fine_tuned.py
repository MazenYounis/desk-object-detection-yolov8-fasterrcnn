from pathlib import Path
from ultralytics import YOLO


def run_fine_tuned_detection():
    """
    Runs the fine-tuned YOLO model on the same baseline test images.
    This allows visual comparison against the original pre-trained YOLO baseline.
    """

    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "results"
        / "fine_tuned"
        / "yolov8n_desk_objects_v3"
        / "weights"
        / "best.pt"
    )

    input_dir = project_root / "data" / "test_images"
    output_dir = project_root / "results" / "fine_tuned"

    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    print("Loading fine-tuned YOLO model...")
    model = YOLO(str(model_path))

    print("Running fine-tuned detection on baseline test images...")

    model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="predictions_on_test_images_v3",
        exist_ok=True,
        conf=0.25
    )

    print("Fine-tuned detection completed.")
    print(f"Results saved in: {output_dir / 'predictions_on_test_images'}")


if __name__ == "__main__":
    run_fine_tuned_detection()