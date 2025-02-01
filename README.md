# 🚗 **CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)** 🐦

This project demonstrates an image classification model built using Convolutional Neural Networks (CNNs) to classify images from the **CIFAR-10 dataset**. Achieved an impressive **test accuracy of 83.11%!**

---

## 🚀 **Project Overview**

**CIFAR-10** is a dataset consisting of **60,000 32x32 color images** in **10 different classes**, with **6,000 images per class**. The objective of this project is to build a CNN model to classify these images into categories such as airplane, automobile, cat, dog, etc.

### **Key Highlights**:

- **Dataset**: CIFAR-10 (10 classes)
- **Accuracy**: **83.11%** test accuracy
- **Framework**: TensorFlow/Keras, Python

### **Optimizations**:

- Data Augmentation for better generalization
- Batch Normalization for improved training stability
- Dropout for preventing overfitting
- GlobalAveragePooling2D for efficient feature extraction

---

## 📚 **Model Architecture**

The CNN model has been designed with the following architecture:

### **Conv Block 1**:

- 32 filters, 3x3 kernels
- BatchNormalization
- MaxPooling
- Dropout (0.25)

### **Conv Block 2**:

- 64 filters, 3x3 kernels
- BatchNormalization
- MaxPooling
- Dropout (0.25)

### **Conv Block 3**:

- 128 filters, 3x3 kernels
- BatchNormalization
- MaxPooling
- Dropout (0.4)

### **Fully Connected Layers**:

- GlobalAveragePooling2D
- Dense layer with 128 neurons and ReLU activation
- Dropout (0.5)
- Output layer with softmax activation (10 classes)

### **Optimizer**:

- Adam Optimizer used for faster convergence

---

## 🛠️ **Technologies Used**

- **Python**: The programming language used.
- **TensorFlow/Keras**: Deep learning framework for building the CNN model.
- **Matplotlib**: For visualizing training progress and results.
- **Scikit-Learn**: For splitting the dataset into training, validation, and testing sets.
- **NumPy**: For efficient array manipulation.

---
