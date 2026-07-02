# Baseline Observations

## Project

**Title:** Real-Time Object Detection of Everyday Items Using YOLO and Transfer Learning

This document records the first baseline test using a pre-trained YOLO model before any fine-tuning.

## Baseline Setup

**Model:** YOLOv8n  
**Model file:** `yolov8n.pt`  
**Input folder:** `data/test_images/`  
**Output folder:** `results/baseline/predictions/`  
**Number of test images:** 5  
**Confidence threshold:** 0.25  
**Target project classes:**

- laptop
- smartphone / cell phone
- mouse
- bottle
- cup / mug

## Purpose of the Baseline

The purpose of this first baseline test is to evaluate how well a pre-trained YOLO model can detect the selected desk objects without any project-specific fine-tuning.

This step helps us understand:

- which target objects are already detected correctly,
- which objects are missed,
- which objects are detected with low confidence,
- which wrong or irrelevant classes appear,
- where fine-tuning could improve the model.

## Baseline Results

| Image | Objects visible in the image | Correct detections | Missed or weak detections | Extra / irrelevant detections | Notes |
|---|---|---|---|---|---|
| `desk_01.jpg` | laptop, smartphone, mouse, bottle, cup, keyboard in background | laptop, mouse, bottle, cup, cell phone | cell phone confidence is moderate | keyboard | Strong baseline result. Most project objects are detected. Keyboard is detected although it is not part of the final target classes. |
| `desk_02.jpg` | laptop, smartphone, keyboard in background | laptop, cell phone | cell phone confidence is low | keyboard | Laptop is detected well. Smartphone is detected but with low confidence. |
| `desk_03.jpg` | laptop, keyboard in background | laptop | mouse appears only partially / background object | keyboard, mouse | Laptop is detected very well. Keyboard is again detected as an extra class. |
| `desk_04.jpg` | mouse, cup, smartphone, keyboard in background | mouse, cup | smartphone is not detected | keyboard | Mouse and cup are detected well. Phone is dark and blends into the desk, which may explain why it was missed. |
| `desk_05.jpg` | bottle, cup, keyboard edge in background | bottle, cup | bottle confidence is lower than cup | dining table | Cup is detected well. Bottle is detected with lower confidence. The desk/table is incorrectly detected as dining table. |

## General Observations

The pre-trained YOLO model already performs well on several of the selected target objects. Laptop, mouse, bottle and cup are detected in multiple images. Smartphone is detected in some cases as `cell phone`, which is the class name used by YOLO.

However, the baseline is not fully stable. The smartphone is sometimes missed or detected with low confidence, especially when it is dark and placed on a dark desk. The bottle is detected, but in some images with lower confidence. The model also detects irrelevant classes such as `keyboard` and `dining table`, which are not part of our final five project classes.

## Initial Conclusion

The baseline confirms that a pre-trained YOLO model can already recognize many relevant desk objects. At the same time, the results show clear room for improvement in our specific desk environment.

This supports the motivation for the next project phase: creating or collecting a small task-specific dataset and fine-tuning the model on the selected object classes.

The expected benefit of fine-tuning is improved detection stability for the five project classes:

- laptop
- smartphone / cell phone
- mouse
- bottle
- cup / mug

In the final project, the fine-tuned model can be compared against this baseline using both visual examples and quantitative metrics such as confidence values, precision, recall and mAP.

## Raw Dataset Batch 01 Baseline

A second baseline test was performed on the first raw dataset batch containing 48 images.

The results were mixed. The pretrained YOLOv8n model detected some target objects correctly, but it also produced several wrong or irrelevant detections. In particular, some objects were detected as `remote control`, `keyboard` or `dining table`, although these classes are not part of the final project scope.

This confirms that the pretrained model is not fully adapted to our specific desk-object environment. The result supports the need for a custom annotated dataset and fine-tuning on the five selected target classes:

- laptop
- cell_phone
- mouse
- bottle
- cup

These baseline errors will be useful later when comparing the pretrained model with the fine-tuned model.