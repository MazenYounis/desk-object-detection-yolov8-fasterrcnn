from pathlib import Path

import matplotlib.pyplot as plt


def create_output_folder():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_yolo_version_metrics(output_dir):
    """
    Creates a grouped bar chart comparing YOLOv8n v1, v2 and v3
    on the test set.
    """

    versions = ["YOLOv8n v1", "YOLOv8n v2", "YOLOv8n v3"]

    precision = [0.853, 0.834, 0.890]
    recall = [0.791, 0.879, 0.906]
    map50 = [0.903, 0.877, 0.965]
    map50_95 = [0.800, 0.699, 0.870]

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }

    x_positions = range(len(versions))
    bar_width = 0.18

    plt.figure(figsize=(10, 6))

    for index, (metric_name, values) in enumerate(metrics.items()):
        positions = [
            x + (index - 1.5) * bar_width
            for x in x_positions
        ]

        plt.bar(
            positions,
            values,
            width=bar_width,
            label=metric_name
        )

    plt.xticks(list(x_positions), versions)
    plt.ylim(0.6, 1.0)
    plt.ylabel("Score")
    plt.title("YOLOv8n Test Metrics Across Dataset Versions")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "yolo_test_metrics_versions.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_final_yolo_v3_metrics(output_dir):
    """
    Creates a bar chart showing the final YOLOv8n v3 test results.
    """

    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    scores = [0.890, 0.906, 0.965, 0.870]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, scores)

    plt.ylim(0.7, 1.0)
    plt.ylabel("Score")
    plt.title("Final YOLOv8n v3 Test Performance")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "final_yolov8n_v3_test_performance.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_faster_rcnn_training_loss(output_dir):
    """
    Creates a line chart showing Faster R-CNN training loss over 10 epochs.
    """

    epochs = list(range(1, 11))

    average_losses = [
        0.4982,
        0.2240,
        0.1524,
        0.1063,
        0.0917,
        0.0840,
        0.0779,
        0.0685,
        0.0659,
        0.0622,
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, average_losses, marker="o")

    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Faster R-CNN Training Loss Over 10 Epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "faster_rcnn_training_loss.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    output_dir = create_output_folder()

    plot_yolo_version_metrics(output_dir)
    plot_final_yolo_v3_metrics(output_dir)
    plot_faster_rcnn_training_loss(output_dir)

    print("All result graphs generated successfully.")


if __name__ == "__main__":
    main()