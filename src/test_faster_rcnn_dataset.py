from pathlib import Path
from PIL import Image


CLASS_NAMES = ["bottle", "cell_phone", "cup", "laptop", "mouse"]


def convert_yolo_to_xyxy(label_line, image_width, image_height):
    class_id, x_center, y_center, width, height = map(float, label_line.split())

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return int(class_id), [x_min, y_min, x_max, y_max]


def main():
    project_root = Path(__file__).resolve().parents[1]

    image_path = (
        project_root
        / "data"
        / "processed"
        / "desk_object_detection_v3"
        / "train"
        / "images"
        / "desk_003_jpg.rf.6cf80e88e759b636ab9209fd2b281503.jpg"
    )

    label_path = (
        project_root
        / "data"
        / "processed"
        / "desk_object_detection_v3"
        / "train"
        / "labels"
        / "desk_003_jpg.rf.6cf80e88e759b636ab9209fd2b281503.txt"
    )

    image = Image.open(image_path)
    image_width, image_height = image.size

    print(f"Image size: {image_width} x {image_height}")

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.read().strip().splitlines()

    for line in lines:
        class_id, box = convert_yolo_to_xyxy(line, image_width, image_height)
        print(f"Class: {CLASS_NAMES[class_id]} | Box xyxy: {box}")


if __name__ == "__main__":
    main()