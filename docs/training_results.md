# Training Results

## Project

**Title:** Real-Time Object Detection of Everyday Items Using YOLO and Transfer Learning

This document records the first fine-tuning run of the YOLO model on our custom desk-object dataset.

## Dataset Version

The first training dataset was created using Roboflow.

**Dataset location in project:**

`data/processed/desk_object_detection_v1/`

**Number of images:** 80

**Classes:** 5

- bottle
- cell_phone
- cup
- laptop
- mouse

**Train / Validation / Test split:**

- Training set: 56 images
- Validation set: 16 images
- Test set: 8 images

**Preprocessing:**

- Auto-orient: applied
- Resize: 640 x 640
- Augmentation: none

The dataset was exported from Roboflow in YOLOv8 format and then used locally in PyCharm for fine-tuning.

## Training Setup

**Model:** YOLOv8n  
**Pretrained weights:** `yolov8n.pt`  
**Training script:** `src/train_yolo.py`  
**Epochs:** 50  
**Image size:** 640  
**Batch size:** 8  
**Training device:** CPU on Apple M4  
**Output folder:**

`results/fine_tuned/yolov8n_desk_objects_v1/`

The best trained model was saved as:

`results/fine_tuned/yolov8n_desk_objects_v1/weights/best.pt`

## Validation Results

After 50 epochs, the model was validated on the validation set.

Overall validation results:

| Metric | Value |
|---|---:|
| Precision | 0.839 |
| Recall | 0.811 |
| mAP50 | 0.891 |
| mAP50-95 | 0.797 |

Class-level results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bottle | 0.902 | 1.000 | 0.995 | 0.852 |
| cell_phone | 1.000 | 0.554 | 0.904 | 0.793 |
| cup | 0.984 | 1.000 | 0.995 | 0.945 |
| laptop | 0.862 | 1.000 | 0.973 | 0.877 |
| mouse | 0.446 | 0.500 | 0.586 | 0.518 |

## Interpretation

The first fine-tuning run was successful. The model learned the five custom classes and produced strong overall results, especially considering the small dataset size of 80 images.

The strongest classes were:

- cup
- bottle
- laptop

The weakest class was:

- mouse

The weaker mouse result is expected because the validation set contained only a small number of mouse examples. This means the result is less stable and more data is needed for this class.

## Visual Comparison Against Baseline

After training, the fine-tuned model was tested on the same baseline test images that were previously used with the pretrained YOLO model.

**Baseline results folder:**

`results/baseline/predictions/`

**Fine-tuned results folder:**

`results/fine_tuned/predictions_on_test_images/`

The comparison showed that the fine-tuned model performed better overall.

Main improvements:

- The model focused only on the five project classes.
- Irrelevant detections such as `keyboard` and `dining table` disappeared.
- Bottle, cup, laptop, and phone detections became more confident.
- The model detected the dark phone more reliably than the pretrained baseline.

Example improvements:

- Bottle confidence improved strongly in multiple images.
- Cup detection became more stable.
- Laptop detection confidence improved in several images.
- Phone detection improved from weak or missing detections to clearer `cell_phone` detections.

## Remaining Issues

The model is not perfect yet. Some issues remain:

- In one image, the fine-tuned model incorrectly detected part of the keyboard area as a second laptop.
- Mouse detection is still weaker than the other classes.
- The dataset is still relatively small.
- More varied images are needed to improve generalization.

Possible reasons for these issues:

- Some objects appear only a few times in the dataset.
- Several images contain similar desk layouts.
- Some objects are dark or partially hidden.
- The model may confuse rectangular shapes such as laptops, keyboards, and dark phones.

## Conclusion

The first fine-tuning run shows a clear improvement over the pretrained YOLOv8n baseline. The model is now more adapted to the selected desk-object environment and focuses on the five selected classes.

The results confirm that transfer learning is suitable for this project. Even with a small dataset, the model improved compared to the generic pretrained baseline.

The next improvement step would be to collect more images, especially for the weaker classes such as mouse and cell_phone, and then train a second version of the model.

## Test Set Evaluation

After training, the fine-tuned model was also evaluated on the independent test set.

**Evaluation script:** `src/evaluate_model.py`  
**Model used:** `results/fine_tuned/yolov8n_desk_objects_v1/weights/best.pt`  
**Test set:** `data/processed/desk_object_detection_v1/test/`  
**Number of test images:** 8  
**Number of test instances:** 19  

### Overall Test Results

| Metric | Value |
|---|---:|
| Precision | 0.853 |
| Recall | 0.791 |
| mAP50 | 0.903 |
| mAP50-95 | 0.800 |

### Class-Level Test Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 2 | 2 | 1.000 | 0.784 | 0.995 | 0.821 |
| cell_phone | 4 | 6 | 0.771 | 0.571 | 0.766 | 0.672 |
| cup | 2 | 2 | 0.965 | 1.000 | 0.995 | 0.895 |
| laptop | 3 | 4 | 0.661 | 1.000 | 0.995 | 0.932 |
| mouse | 5 | 5 | 0.866 | 0.600 | 0.765 | 0.677 |

### Interpretation of Test Results

The test results confirm that the fine-tuned model generalizes reasonably well to unseen images. The overall mAP50 of 0.903 shows strong detection performance for the first dataset version.

The model performs especially well on `bottle`, `cup`, and `laptop`. These classes achieved high mAP50 values and were detected reliably in the test set.

The weaker classes are `cell_phone` and `mouse`. Both have lower recall, which means the model does not always detect every instance. This is expected because these objects can be small, dark, reflective, partially hidden, or visually similar to the desk background.

The confusion matrix shows that most main object classes were recognized correctly, but some objects were still confused with background or other nearby desk objects.

### Conclusion from Test Evaluation

The test evaluation supports the result from the visual comparison: fine-tuning improved the YOLO model for our specific desk-object scenario. The model is not perfect yet, but it shows clear learning behavior and useful detection performance after transfer learning.

For the next dataset version, more diverse images should be collected, especially for `cell_phone` and `mouse`, to improve recall and reduce missed detections.

## Baseline vs Fine-Tuned Comparison

A comparison script was created to run both the pretrained YOLOv8n model and the fine-tuned YOLOv8n model on the same five demo images.

**Comparison script:** `src/compare_models.py`  
**Input images:** `data/test_images/`

**Output folders:**

```text
results/comparison/pretrained_yolov8n/
results/comparison/fine_tuned_yolov8n/
```

### Pretrained YOLOv8n Results

The pretrained YOLOv8n model detected several correct objects, such as laptop, mouse, bottle, cup and cell phone.

However, it also predicted irrelevant classes that are outside the scope of this project:

- keyboard
- dining table

These detections show that the pretrained model is a general object detection model. It can recognize many everyday objects, but it is not specifically adapted to the selected desk-object task.

Example pretrained detections:

- Image 1: laptop, mouse, keyboard
- Image 2: laptop, keyboard, cell phone
- Image 3: bottle, cup, dining table
- Image 4: cup, mouse, keyboard
- Image 5: bottle, cup, laptop, mouse, keyboard, cell phone

### Fine-Tuned YOLOv8n Results

The fine-tuned model detected only the five selected project classes:

- bottle
- cell_phone
- cup
- laptop
- mouse

This shows that the model became more task-specific after fine-tuning.

Example fine-tuned detections:

- Image 1: 2 laptops
- Image 2: 1 cell_phone, 1 laptop
- Image 3: 1 bottle, 1 cup
- Image 4: 1 cell_phone, 1 cup, 1 mouse
- Image 5: 1 bottle, 1 cell_phone, 1 cup, 3 laptops, 1 mouse

### Main Improvement

The fine-tuned model is better aligned with the project goal because it no longer predicts irrelevant classes such as `keyboard` and `dining table`.

The comparison confirms that transfer learning helped adapt the model from a general object detector to a detector focused on the custom desk-object classes.

### Remaining Issues in the Comparison

The fine-tuned model still produced some false positives.

The main issue is laptop over-detection. In some images, the model predicted more than one laptop, even though only one laptop was actually present.

This likely happened because some rectangular desk regions, keyboard areas, or flat dark objects visually resemble laptop shapes.

### Conclusion from the Comparison

The pretrained model was more general and detected extra classes that were not needed for this project.

The fine-tuned model became more focused and task-specific. It detected only the five selected classes and therefore better matched the goal of the project.

However, the model is not fully perfect yet. More diverse images, especially with different laptop positions, keyboard areas, phones and mice, would help reduce false positives and improve reliability.


## YOLOv8n Dataset v2 Results

A second YOLO fine-tuning run was performed using Dataset v2. Dataset v2 contains 120 original images and was generated in Roboflow with brightness augmentation, resulting in 216 images in the exported version.

The goal of Dataset v2 was to improve the weaker classes from the first training run, especially `cell_phone` and `mouse`.

### Validation Results v2

| Metric | Value |
|---|---:|
| Precision | 0.799 |
| Recall | 0.871 |
| mAP50 | 0.896 |
| mAP50-95 | 0.755 |

### Test Results v2

| Metric | Value |
|---|---:|
| Precision | 0.834 |
| Recall | 0.879 |
| mAP50 | 0.877 |
| mAP50-95 | 0.699 |

### Comparison with Dataset v1 Test Results

| Version | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n v1 | 0.853 | 0.791 | 0.903 | 0.800 |
| YOLOv8n v2 | 0.834 | 0.879 | 0.877 | 0.699 |

### Interpretation

Dataset v2 improved the overall recall compared to Dataset v1. This means that the model found more of the objects present in the test images. The improvement is especially visible for the `mouse` class, where recall increased from 0.600 to 0.800.

However, precision, mAP50 and especially mAP50-95 decreased compared to v1. This suggests that the second model detects more objects, but its bounding boxes and prediction quality are less precise. One possible reason is that Dataset v2 contains more difficult examples and brightness augmentation, which makes the task harder.

Overall, Dataset v2 is useful because it improves object coverage, but Dataset v1 still produced better localization quality. This trade-off should be considered when comparing model versions.

## YOLOv8n Dataset v3 Results

A third YOLO fine-tuning run was performed using Dataset v3. Dataset v3 uses the same 120 source images as Dataset v2, but without brightness augmentation. The goal was to check whether the false positive detections from Dataset v2 were caused by the augmentation.

### Validation Results v3

| Metric | Value |
|---|---:|
| Precision | 0.911 |
| Recall | 0.814 |
| mAP50 | 0.942 |
| mAP50-95 | 0.865 |

### Test Results v3

| Metric | Value |
|---|---:|
| Precision | 0.890 |
| Recall | 0.906 |
| mAP50 | 0.965 |
| mAP50-95 | 0.870 |

### Comparison of YOLO Versions on the Test Set

| Version | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n v1 | 80 images, no augmentation | 0.853 | 0.791 | 0.903 | 0.800 |
| YOLOv8n v2 | 120 images, brightness augmentation | 0.834 | 0.879 | 0.877 | 0.699 |
| YOLOv8n v3 | 120 images, no augmentation | 0.890 | 0.906 | 0.965 | 0.870 |

### Interpretation

Dataset v3 produced the best overall YOLO results. Compared to Dataset v1, it improved Precision, Recall, mAP50 and mAP50-95. Compared to Dataset v2, it also strongly reduced the false positive detections that appeared in the visual prediction examples.

The results suggest that adding more targeted images for `cell_phone` and `mouse` was useful, but the brightness augmentation in Dataset v2 made the model less stable. Without augmentation, Dataset v3 achieved better localization quality and cleaner visual predictions.

For this reason, YOLOv8n v3 is currently considered the best YOLO model version in the project.

## Faster R-CNN v1 Results

A Faster R-CNN model was trained as a second object detection method for comparison with YOLOv8n. The model was trained on Dataset v3, which contains 120 source images without augmentation.

The model was trained for 10 epochs on CPU. The average training loss decreased clearly during training:

| Epoch | Average Loss |
|---|---:|
| 1 | 0.4982 |
| 2 | 0.2240 |
| 3 | 0.1524 |
| 4 | 0.1063 |
| 5 | 0.0917 |
| 6 | 0.0840 |
| 7 | 0.0779 |
| 8 | 0.0685 |
| 9 | 0.0659 |
| 10 | 0.0622 |

The decreasing loss shows that the model learned from the training data. After training, Faster R-CNN was tested visually on the same five example images used for the YOLO comparison.

The visual predictions were clean overall. The model detected the main desk objects such as laptop, bottle, cup, cell phone and mouse with high confidence. Compared to YOLOv8n v3, Faster R-CNN also produced good bounding boxes, but it is expected to be slower because it uses a two-stage detection approach.

The trained Faster R-CNN model was saved as:

`results/faster_rcnn/faster_rcnn_desk_objects_v1.pth`