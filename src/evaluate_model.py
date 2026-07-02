from pathlib import Path
from ultralytics import YOLO


def evaluate_model():
    """
    Evaluates the fine-tuned YOLO v2 model on the test split of Dataset v2.
    This gives a final performance check on images that were not used for training.
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

    data_yaml = (
        project_root
        / "data"
        / "processed"
        / "desk_object_detection_v3"
        / "data.yaml"
    )

    output_dir = project_root / "results" / "evaluation"

    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fine-tuned YOLO v3 model...")
    model = YOLO(str(model_path))

    print("Evaluating model on Dataset v3 test set...")

    model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=8,
        project=str(output_dir),
        name="test_set_evaluation_v3",
        exist_ok=True
    )

    print("Evaluation completed.")
    print(f"Results saved in: {output_dir / 'test_set_evaluation_v3'}")


if __name__ == "__main__":
    evaluate_model()