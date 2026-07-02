from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAMES = ["background", "bottle", "cell_phone", "cup", "laptop", "mouse"]
NUM_CLASSES = len(CLASS_NAMES)


class YoloDetectionDataset(Dataset):
    """
    Loads a YOLOv8 dataset and converts YOLO labels into Faster R-CNN format.

    YOLO format:
    class_id x_center y_center width height

    Faster R-CNN format:
    boxes: [x_min, y_min, x_max, y_max]
    labels: class ids

    Faster R-CNN reserves label 0 for background.
    Therefore, YOLO class ids 0 to 4 are shifted to labels 1 to 5.
    """

    def __init__(self, dataset_dir: Path, split: str):
        self.dataset_dir = dataset_dir
        self.split = split

        self.images_dir = dataset_dir / split / "images"
        self.labels_dir = dataset_dir / split / "labels"

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.png"))
            + list(self.images_dir.glob("*.JPG"))
            + list(self.images_dir.glob("*.JPEG"))
            + list(self.images_dir.glob("*.PNG"))
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as file:
                lines = file.read().strip().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                class_id, x_center, y_center, width, height = map(float, line.split())

                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height

                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }

        image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    """
    Creates a pretrained Faster R-CNN model and replaces the classification head
    so it predicts our custom classes.
    """

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def train_faster_rcnn():
    project_root = Path(__file__).resolve().parents[1]

    dataset_dir = project_root / "data" / "processed" / "desk_object_detection_v3"
    output_dir = project_root / "results" / "faster_rcnn"

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = YoloDetectionDataset(dataset_dir=dataset_dir, split="train")
    valid_dataset = YoloDetectionDataset(dataset_dir=dataset_dir, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(valid_dataset)}")

    print("Creating Faster R-CNN model...")
    model = create_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    num_epochs = 10

    print("Starting Faster R-CNN training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_index, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if batch_index % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{batch_index}/{len(train_loader)}], "
                    f"Loss: {losses.item():.4f}"
                )

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average loss: {average_loss:.4f}")

    model_path = output_dir / "faster_rcnn_desk_objects_v1.pth"
    torch.save(model.state_dict(), model_path)

    print("Faster R-CNN training completed.")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_faster_rcnn()