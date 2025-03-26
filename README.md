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

- **Horizontal Flip (50%)** → Randomly flips images horizontally to introduce left-right variations.  
- **Vertical Flip (50%)** → Randomly flips images vertically to simulate different perspectives.  
- **Random Rotation (90°) (50%)** → Rotates images by 90 degrees randomly to add orientation diversity.  
- **Shift-Scale-Rotate (50%)** →  
  - Randomly shifts images up/down or left/right within 5% of their size.  
  - Randomly scales images within a 5% range.  
  - Rotates images within a ±15-degree range.  
- **Gaussian Noise (30%)** → Adds random noise to images, making the model more robust against real-world imperfections.  
- **Random Brightness & Contrast (30%)** → Randomly adjusts brightness and contrast to simulate lighting variations.  

These augmentation techniques help the model learn to recognize bone fractures under different conditions, improving its overall accuracy and reliability.  

---
