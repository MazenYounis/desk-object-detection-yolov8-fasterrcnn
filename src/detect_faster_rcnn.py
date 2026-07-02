from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAMES = ["background", "bottle", "cell_phone", "cup", "laptop", "mouse"]
NUM_CLASSES = len(CLASS_NAMES)


def create_model(num_classes: int):
    """
    Creates the same Faster R-CNN architecture used during training.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def draw_predictions(image, prediction, confidence_threshold=0.5):
    """
    Draws predicted bounding boxes, class names and confidence scores.
    """
    draw = ImageDraw.Draw(image)

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue

        x_min, y_min, x_max, y_max = box.tolist()
        class_name = CLASS_NAMES[label.item()]
        text = f"{class_name} {score:.2f}"

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=4
        )

        text_background = draw.textbbox((x_min, y_min), text, font=font)
        draw.rectangle(text_background, fill="red")
        draw.text((x_min, y_min), text, fill="white", font=font)

    return image


def detect_faster_rcnn():
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "results"
        / "faster_rcnn"
        / "faster_rcnn_desk_objects_v1.pth"
    )

    input_dir = project_root / "data" / "test_images"
    output_dir = project_root / "results" / "faster_rcnn" / "predictions_on_test_images_v1"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Faster R-CNN model not found: {model_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Loading Faster R-CNN model...")

    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_paths = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
        + list(input_dir.glob("*.JPG"))
        + list(input_dir.glob("*.JPEG"))
        + list(input_dir.glob("*.PNG"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Running Faster R-CNN detection on {len(image_paths)} images...")

    with torch.no_grad():
        for image_path in image_paths:
            original_image = Image.open(image_path).convert("RGB")
            image_tensor = F.to_tensor(original_image).to(device)

            prediction = model([image_tensor])[0]

            result_image = draw_predictions(
                image=original_image.copy(),
                prediction=prediction,
                confidence_threshold=0.5
            )

            output_path = output_dir / image_path.name
            result_image.save(output_path)

            kept_predictions = (
                prediction["scores"].detach().cpu() >= 0.5
            ).sum().item()

            print(f"{image_path.name}: saved with {kept_predictions} predictions")

    print("Faster R-CNN detection completed.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    detect_faster_rcnn()