from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "desk_object_detection_v3"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test" / "labels"

MODEL_PATH = PROJECT_ROOT / "results" / "faster_rcnn" / "faster_rcnn_desk_objects_v1.pth"

CLASS_NAMES = [
    "background",
    "bottle",
    "cell_phone",
    "cup",
    "laptop",
    "mouse"
]

NUM_CLASSES = len(CLASS_NAMES)


class DeskObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png"))
        )

        self.transforms = T.Compose([
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = self.labels_dir / f"{image_path.stem}.txt"

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as file:
                for line in file:
                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id, x_center, y_center, box_width, box_height = map(float, values)

                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2
                    x_max = x_center + box_width / 2
                    y_max = y_center + box_height / 2

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)

                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])

                        # YOLO labels start at 0.
                        # Faster R-CNN uses 0 as background.
                        # Therefore object labels are shifted by +1.
                        labels.append(int(class_id) + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index])
        }

        image = self.transforms(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading test dataset...")
    test_dataset = DeskObjectDetectionDataset(
        images_dir=TEST_IMAGES_DIR,
        labels_dir=TEST_LABELS_DIR
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Number of test images: {len(test_dataset)}")

    print("Loading Faster R-CNN model...")
    model = create_model()

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )

    print("Running evaluation...")

    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]

            predictions = model(images)

            predictions_for_metric = []
            targets_for_metric = []

            for prediction, target in zip(predictions, targets):
                predictions_for_metric.append({
                    "boxes": prediction["boxes"].cpu(),
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu()
                })

                targets_for_metric.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                })

            metric.update(predictions_for_metric, targets_for_metric)

    results = metric.compute()

    print("\nFaster R-CNN Evaluation Results")
    print("--------------------------------")
    print(f"mAP50-95: {results['map']:.4f}")
    print(f"mAP50:    {results['map_50']:.4f}")
    print(f"mAP75:    {results['map_75']:.4f}")
    print(f"mAR100:   {results['mar_100']:.4f}")

    output_dir = PROJECT_ROOT / "results" / "faster_rcnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "evaluation_metrics.txt"

    with open(output_file, "w") as file:
        file.write("Faster R-CNN Evaluation Results\n")
        file.write("--------------------------------\n")
        file.write(f"mAP50-95: {results['map']:.4f}\n")
        file.write(f"mAP50:    {results['map_50']:.4f}\n")
        file.write(f"mAP75:    {results['map_75']:.4f}\n")
        file.write(f"mAR100:   {results['mar_100']:.4f}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()