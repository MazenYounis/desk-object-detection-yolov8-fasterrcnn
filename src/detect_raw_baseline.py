from pathlib import Path
from ultralytics import YOLO


def run_raw_dataset_baseline():
    """
    Runs a pre-trained YOLO model on all raw dataset images inside data/raw
    and saves the detection results separately in results/baseline/raw_batch_01.

    This helps us understand how well the pre-trained model performs on our
    first collected dataset batch before annotation and fine-tuning.
    """

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "results" / "baseline"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}\n"
            "Please add raw dataset images to data/raw."
        )

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
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

    print(f"Running baseline detection on {len(image_files)} raw image(s)...")

    model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="raw_batch_01",
        exist_ok=True,
        conf=0.25
    )

    print("Raw dataset baseline detection completed.")
    print(f"Results saved in: {output_dir / 'raw_batch_01'}")


if __name__ == "__main__":
    run_raw_dataset_baseline()