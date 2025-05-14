
# Object Detection using ResNet50 with YOLO-style Detection Head on Pascal VOC Dataset



## Introduction

Object detection is a crucial task in computer vision, where the goal is to identify and localize objects within images. Deep learning-based methods, particularly convolutional neural networks (CNNs), have achieved state-of-the-art performance on object detection benchmarks. In this project, a model is built using **ResNet50** as the backbone and a **YOLO-style detection head**, which is known for its efficiency and speed in real-time detection tasks.

The model is evaluated on the **Pascal VOC dataset**, a standard benchmark in the field of object detection. The primary evaluation metric used is the **mean Average Precision (mAP)**, which provides a comprehensive measure of the model's overall object detection performance.

## Methodology

### Model Architecture

The model consists of two main components:
1. **ResNet50 Backbone**: ResNet50 is a deep convolutional neural network that helps extract hierarchical features from input images. It is known for its use of residual connections, allowing deeper networks to be trained more effectively.
2. **YOLO-style Detection Head**: The detection head is based on the YOLO (You Only Look Once) architecture, which performs classification and bounding box regression simultaneously in a single forward pass. This detection head outputs the predicted class scores, bounding box coordinates, and objectness score for each grid cell in the image.(as we split the image into S grids)

### Dataset

The **Pascal VOC** dataset is used for training and testing the model. It contains 20 object classes and includes both classification and localization annotations. The model was trained on a subset of 100 samples from the dataset. Link to the dataset: https://www.kaggle.com/datasets/aladdinpersson/pascalvoc-yolo

### Evaluation Metric

The model is evaluated using **mean Average Precision (mAP)**, which averages the Average Precision (AP) scores across all object classes. AP is computed by measuring the precision of the model at different levels of recall, making mAP a robust metric for evaluating object detection models.

## Experimental Setup

### Implementation Details

The model is implemented using **Python** and the **PyTorch** framework. The ResNet50 backbone is pre-trained on ImageNet, and the YOLO-style detection head is customized for the Pascal VOC dataset. 

**Loss Components**:

- box_loss: Loss related to bounding box predictions (both coordinates and dimensions).

- object_loss: Loss related to predicting object confidence (whether there is an object in a grid cell).

- no_object_loss: Loss related to predicting no object in grid cells.

- class_loss: Loss related to classification of object classes

Training hyperparameters:
- **Learning Rate**: 2e-5
- **Batch Size**: 8
- **Epochs**: 1000
- **Optimizer**: Adam


## Results

The model’s performance is evaluated on the **Pascal VOC test set**.

| **Mean Average Precision (mAP)** | **58.73** |
Here are the few test data images with the visualization of bounding box :
![image](https://github.com/user-attachments/assets/cf9ef8e2-80c0-4a1a-b951-3ce39af53936)
![Screenshot 2025-05-12 163303](https://github.com/user-attachments/assets/46123419-d416-415c-9a41-0aa8af350158)
![Screenshot 2025-05-12 163229](https://github.com/user-attachments/assets/15ab72fb-cdfe-4fce-aa9a-2b5666bd6079)

Model will be more generalizable if trained on more data with proper hypertuning.

**Trained model weights (drive link)** : https://drive.google.com/file/d/135XI-5wnvg98-7WHR5EEgRHfE8H87pek/view?usp=sharing



## Conclusion

In this project, we successfully built an object detection model using **ResNet50** as the backbone and a **YOLO-style detection head**. The model was trained and evaluated on the **Pascal VOC dataset**, achieving an **mAP of 58.73%**.

