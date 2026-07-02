# Dataset Plan

## Purpose

This document defines the dataset plan for the fine-tuning phase of the project.

The goal is to create a small, task-specific dataset for detecting typical desk objects using YOLO.

## Target Classes

The final project focuses on five object classes:

| Class ID | Class Name | German Description |
|---:|---|---|
| 0 | laptop | Laptop |
| 1 | cell_phone | Smartphone |
| 2 | mouse | Maus |
| 3 | bottle | Flasche |
| 4 | cup | Tasse / Becher |

The English class names will be used in the dataset because YOLO workflows and annotation exports usually use English labels.

## Dataset Strategy

The dataset will be created mainly from our own desk photos.

This is useful because the project focuses on a realistic desk environment. Own photos also make the dataset more specific to the selected application and allow us to control object placement, lighting and camera angles.

A public dataset may be used later as additional support, but the main fine-tuning dataset should come from our own images.

## Planned Dataset Size

For the first fine-tuning attempt, the target is:

| Class | Minimum Target Images |
|---|---:|
| laptop | 50 |
| cell_phone | 50 |
| mouse | 50 |
| bottle | 50 |
| cup | 50 |

This does not mean 250 separate photos are strictly required. One image can contain multiple objects. For example, one desk photo can include laptop, phone, mouse, bottle and cup at the same time.

The practical target is:

- Minimum: 100 total images
- Better target: 150 to 250 total images
- Each class should appear in at least 50 images

## Image Collection Rules

Images should show realistic desk or workplace situations.

The dataset should include variation in:

- lighting conditions
- camera angles
- object positions
- object distances
- single-object scenes
- multi-object scenes
- partial occlusion
- different backgrounds
- landscape and portrait images

## Required Image Types

The dataset should include:

| Image Type | Purpose |
|---|---|
| Single-object images | Help the model learn each object clearly |
| Multi-object desk images | Represent realistic project scenarios |
| Angled images | Test robustness to perspective changes |
| Close-up images | Improve object detail learning |
| Wider desk images | Test detection in larger scenes |
| Slightly darker images | Test difficult lighting conditions |
| Partially occluded objects | Test robustness when objects are not fully visible |

## Recommended Photo Distribution

A possible collection plan:

| Type | Approx. Number of Images |
|---|---:|
| Laptop-focused images | 20 |
| Smartphone-focused images | 20 |
| Mouse-focused images | 20 |
| Bottle-focused images | 20 |
| Cup-focused images | 20 |
| Mixed desk scenes with 2 to 5 objects | 50 to 100 |

This gives enough variation while keeping the dataset realistic for a course project.

## What to Avoid

Avoid images where:

- the object is extremely blurry
- the object is almost fully hidden
- the image is too dark to understand
- private information is visible on screens or documents
- the target object is too small to annotate clearly
- the same photo is repeated many times with almost no variation

## Annotation Plan

Each visible target object will be annotated with a bounding box.

Only the five selected classes will be annotated:

- laptop
- cell_phone
- mouse
- bottle
- cup

Other objects, such as keyboard, monitor, cables, notebooks or docking stations, will remain unlabelled unless the project scope changes.

## Annotation Tool

The preferred annotation tool is:

**Roboflow**

Reason:

- easy browser-based workflow
- supports bounding box annotation
- exports datasets in YOLO format
- creates train/validation/test split
- useful for small object detection projects

Alternative tools:

- LabelImg
- CVAT

## Dataset Split

The dataset should be split into:

| Split | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Used for fine-tuning |
| Validation | 20% | Used during training to monitor performance |
| Test | 10% | Used for final evaluation |

The test images should be different from the training images and should include realistic desk scenes.

## File Naming Convention

Images should be named clearly and consistently.

Suggested format:

```text
desk_001.jpg
desk_002.jpg
desk_003.jpg
...
```

## Dataset Batch 01

The first image collection batch has been created.

**Location:** `data/raw/`  
**Number of images:** 48  
**Naming format:** `desk_001.jpg` to `desk_048.jpg`

This first batch contains desk-object images for the five target classes:

- laptop
- cell_phone
- mouse
- bottle
- cup

The images include a mix of single-object and multi-object desk scenes. This batch will be used for the first annotation and fine-tuning workflow test.

The current purpose of this batch is not to create the final model yet, but to test the full project pipeline:

1. collect raw images,
2. annotate target objects,
3. export the dataset in YOLO format,
4. fine-tune the model,
5. compare results against the baseline.