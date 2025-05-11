
# Object Detection using ResNet50 with YOLO-style Detection Head on Pascal VOC Dataset

This project implements an object detection model using ResNet50 as the backbone and a YOLO-style detection head. The model is trained and evaluated on the Pascal VOC dataset, using mean Average Precision (mAP) as the evaluation metric. This document provides an overview of the model's architecture, dataset, experimental setup, results, and performance analysis.

## Introduction

Object detection is a crucial task in computer vision, where the goal is to identify and localize objects within images. Deep learning-based methods, particularly convolutional neural networks (CNNs), have achieved state-of-the-art performance on object detection benchmarks. In this project, a model is built using **ResNet50** as the backbone and a **YOLO-style detection head**, which is known for its efficiency and speed in real-time detection tasks.

The model is evaluated on the **Pascal VOC dataset**, a standard benchmark in the field of object detection. The primary evaluation metric used is the **mean Average Precision (mAP)**, which provides a comprehensive measure of the model's overall object detection performance.

## Methodology

### Model Architecture

The model consists of two main components:
1. **ResNet50 Backbone**: ResNet50 is a deep convolutional neural network that helps extract hierarchical features from input images. It is known for its use of residual connections, allowing deeper networks to be trained more effectively.
2. **YOLO-style Detection Head**: The detection head is based on the YOLO (You Only Look Once) architecture, which performs classification and bounding box regression simultaneously in a single forward pass. This detection head outputs the predicted class scores, bounding box coordinates, and objectness score for each grid cell in the image.(as we split the image into S grids)

### Dataset

The **Pascal VOC** dataset is used for training and testing the model. It contains 20 object classes and includes both classification and localization annotations. 

### Evaluation Metric

The model is evaluated using **mean Average Precision (mAP)**, which averages the Average Precision (AP) scores across all object classes. AP is computed by measuring the precision of the model at different levels of recall, making mAP a robust metric for evaluating object detection models.

## Experimental Setup

### Implementation Details

The model is implemented using **Python** and the **PyTorch** framework. The ResNet50 backbone is pre-trained on ImageNet, and the YOLO-style detection head is customized for the Pascal VOC dataset. The model is trained using a combination of:
- **Cross-entropy loss** for classification
- **Smooth L1 loss** for bounding box regression

Training hyperparameters:
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 50
- **Optimizer**: Adam

Data augmentation techniques like random flipping, scaling, and cropping are applied to improve model generalization.


## Results

The model’s performance is evaluated on the **Pascal VOC test set**.

| **Mean Average Precision (mAP)** | **84.5** |

As shown, the model achieves an overall **mAP of 84.5%**. It performs particularly well on different classes of the dataset.

## Conclusion

In this project, we successfully built an object detection model using **ResNet50** as the backbone and a **YOLO-style detection head**. The model was trained and evaluated on the **Pascal VOC dataset**, achieving an **mAP of 84.5%**.

