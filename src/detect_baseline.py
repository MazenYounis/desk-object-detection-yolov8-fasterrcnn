from pathlib import Path
from ultralytics import YOLO


def run_baseline_detection():
    """
    Runs a pre-trained YOLO model on all images inside data/test_images
    and saves the detection results in results/baseline.

    This is our first baseline step:
    We want to see how well a pre-trained YOLO model detects our desk objects
    before any fine-tuning.
    """

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data" / "test_images"
    output_dir = project_root / "results" / "baseline"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}\n"
            "Please create the folder and add some test images."
        )

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []

    for extension in image_extensions:
        image_files.extend(input_dir.glob(extension))

    if not image_files:
        raise FileNotFoundError(
            f"No images found in: {input_dir}\n"
            "Please add .jpg, .jpeg, or .png images."
        )

    print("Loading pre-trained YOLO model...")
    model = YOLO("yolov8n.pt")

    print(f"Running baseline detection on {len(image_files)} image(s)...")

    model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        conf=0.25
    )

    print("Baseline detection completed.")
    print(f"Results saved in: {output_dir / 'predictions'}")


if __name__ == "__main__":
    run_baseline_detection()