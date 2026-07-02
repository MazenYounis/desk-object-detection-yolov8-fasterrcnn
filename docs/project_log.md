# Project Log

## Project Title

**Real-Time Object Detection of Everyday Items Using YOLO and Transfer Learning**

## Purpose of This Log

This file documents the main project steps from setup to first training and evaluation.  
It is used to keep the workflow organized and to make the project easier to explain later.

## 1. Project Setup

A new PyCharm project was created for the object detection project.

Project folder:

```text
desk-object-detection-yolo/
```

A Python virtual environment was created:

```text
.venv/
```

Python version used:

```text
Python 3.12.10
```

The following main packages were installed:

- ultralytics
- opencv-python
- matplotlib
- pandas
- numpy
- torch
- torchvision

The installed package versions were saved in:

```text
requirements.txt
```

## 2. Initial Project Structure

The following folder structure was created:

```text
data/
├── raw/
├── raw_mert/
├── test_images/
└── processed/

docs/
results/
src/
models/
notebooks/
```

The purpose of the main folders:

| Folder | Purpose |
|---|---|
| `data/test_images/` | Small demo/baseline test images |
| `data/raw/` | First raw image batch collected by Mazen |
| `data/raw_mert/` | Second raw image batch collected by Mert |
| `data/processed/` | Exported YOLO dataset from Roboflow |
| `src/` | Python scripts |
| `docs/` | Project documentation |
| `results/` | Baseline, training and evaluation outputs |

## 3. Baseline Test with Pretrained YOLO

A pretrained YOLOv8n model was tested first before any fine-tuning.

Script:

```text
src/detect_baseline.py
```

Input folder:

```text
data/test_images/
```

Output folder:

```text
results/baseline/predictions/
```

The baseline test showed that YOLOv8n could already detect several desk objects, such as laptop, mouse, bottle, cup and cell phone.  
However, it also produced irrelevant detections such as keyboard and dining table.

The baseline observations were documented in:

```text
docs/baseline_observations.md
```

## 4. Raw Dataset Collection

A first dataset batch was collected by Mazen.

Location:

```text
data/raw/
```

Number of images:

```text
48
```

A second dataset batch was collected by Mert.

Location:

```text
data/raw_mert/
```

Number of images:

```text
32
```

Total raw images collected:

```text
80
```

Target classes:

- bottle
- cell_phone
- cup
- laptop
- mouse

## 5. Annotation in Roboflow

The 80 images were uploaded to Roboflow and annotated manually using bounding boxes.

Only the five target classes were annotated:

- bottle
- cell_phone
- cup
- laptop
- mouse

Other objects such as keyboard, monitor, desk, cables and table were not annotated because they are outside the project scope.

## 6. Dataset Export

A first dataset version was generated in Roboflow.

Dataset settings:

| Setting | Value |
|---|---|
| Dataset version | v1 |
| Format | YOLOv8 |
| Images | 80 |
| Classes | 5 |
| Train split | 56 images |
| Validation split | 16 images |
| Test split | 8 images |
| Resize | 640 x 640 |
| Augmentation | None |

The exported dataset was saved locally in:

```text
data/processed/desk_object_detection_v1/
```

The dataset configuration file is:

```text
data/processed/desk_object_detection_v1/data.yaml
```

The `data.yaml` paths were adjusted to:

```yaml
train: train/images
val: valid/images
test: test/images
```

## 7. Fine-Tuning

The model was fine-tuned locally using YOLOv8n pretrained weights.

Training script:

```text
src/train_yolo.py
```

Training setup:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Pretrained weights | yolov8n.pt |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 8 |
| Device | CPU, Apple M4 |

The training completed successfully.

The best model was saved as:

```text
results/fine_tuned/yolov8n_desk_objects_v1/weights/best.pt
```

## 8. Validation Results

After training, the model was validated on the validation set.

Overall validation results:

| Metric | Value |
|---|---:|
| Precision | 0.839 |
| Recall | 0.811 |
| mAP50 | 0.891 |
| mAP50-95 | 0.797 |

The strongest classes were:

- bottle
- cup
- laptop

The weakest class was:

- mouse

## 9. Fine-Tuned Prediction Test

The fine-tuned model was tested on the same demo images used in the original baseline.

Script:

```text
src/detect_fine_tuned.py
```

Output folder:

```text
results/fine_tuned/predictions_on_test_images/
```

The visual comparison showed that the fine-tuned model improved over the pretrained baseline.  
It focused only on the five target classes and removed irrelevant detections such as keyboard and dining table.

## 10. Test Set Evaluation

The fine-tuned model was evaluated on the independent test set.

Evaluation script:

```text
src/evaluate_model.py
```

Output folder:

```text
results/evaluation/test_set_evaluation/
```

Overall test results:

| Metric | Value |
|---|---:|
| Precision | 0.853 |
| Recall | 0.791 |
| mAP50 | 0.903 |
| mAP50-95 | 0.800 |

Class-level test results:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 2 | 2 | 1.000 | 0.784 | 0.995 | 0.821 |
| cell_phone | 4 | 6 | 0.771 | 0.571 | 0.766 | 0.672 |
| cup | 2 | 2 | 0.965 | 1.000 | 0.995 | 0.895 |
| laptop | 3 | 4 | 0.661 | 1.000 | 0.995 | 0.932 |
| mouse | 5 | 5 | 0.866 | 0.600 | 0.765 | 0.677 |

## 11. Current Conclusion

The first fine-tuning experiment was successful.

The model learned the five custom desk-object classes and achieved strong results on the test set, even with only 80 images.

The project already has:

- working baseline detection
- custom annotated dataset
- successful fine-tuning
- validation results
- test evaluation results
- visual before/after comparison
- project documentation

## 12. Next Possible Improvements

Possible next steps:

- collect more images for `cell_phone` and `mouse`
- add more difficult lighting conditions
- add more varied desks and backgrounds
- train a second dataset version
- compare YOLOv8n with another YOLO model size
- prepare final poster visuals and result tables
