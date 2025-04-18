# Bone Fracture Detection using Deep Learning  



## 📌 Project Description  
This project focuses on **bone fracture detection** using **X-ray images** and deep learning. We implemented a **ResNet-50** model to classify X-ray images as fractured or normal. Our approach includes **data augmentation**, **transfer learning**, and **5-fold cross-validation** to enhance model performance and generalization.



## 📌 Contribution Table
| Name   | Task Contribution |
|--------|-------------------|
| **Farhan** | Implemented the training pipeline, trained the model, and optimized hyperparameters. |
| **Yeash** | Handled the testing phase, including model evaluation, inference, and prediction on test data. |
| **Jim** | Implemented data augmentation techniques to enhance dataset variability and model robustness. |



## 📌 Data Augmentation Methods Used
To improve the diversity and robustness of the dataset, we applied various data augmentation techniques using Albumentations. These augmentations were applied to the training images to help the model generalize better and reduce overfitting. The following transformations were used:  

- **Resize**: All images were resized to 224×224 pixels.
- **Random Horizontal Flip**: Images were flipped horizontally with a 50% probability to simulate left/right limb orientation.
- **Random Rotation**: Applied a small random rotation (±10 degrees) to account for slight misalignment in X-ray captures.
- **Normalization**: Images were normalized using the standard ImageNet mean and standard deviation:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

These augmentation techniques help the model learn to recognize bone fractures under different conditions, improving its overall accuracy and reliability.  

---
