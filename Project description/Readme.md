# Computer Vision Based Disease Classification using Chest X-Ray Images

## Project Overview
This project focuses on building a **deep learning based computer vision system** that automatically classifies chest X-ray images into different disease categories. The model is trained using a **Convolutional Neural Network (CNN)** implemented in TensorFlow/Keras.

The goal of the project is to assist in **early detection of respiratory diseases** by analyzing medical images and predicting the presence of conditions such as **COVID-19, Viral Pneumonia, Lung Opacity, or Normal lungs**.

Medical image classification using deep learning can help healthcare professionals by providing **fast and automated diagnostic support**.

---

## Problem Statement
Manual diagnosis of chest X-ray images requires expert radiologists and can be time-consuming. During large outbreaks such as COVID-19, rapid screening becomes critical.

This project aims to develop a **computer vision model that can automatically analyze chest X-ray images and classify them into disease categories**, helping reduce diagnostic workload and enabling faster medical decisions.

---

## Dataset

The dataset used in this project is the **COVID-19 Radiography Database**, which contains chest X-ray images for multiple respiratory conditions.

Dataset Source: Kaggle

Dataset Link  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

### Dataset Statistics

Total Images: **42,330**

Classes in the dataset:

| Class | Description |
|------|-------------|
| COVID | Chest X-ray images of COVID-19 infected lungs |
| Normal | Healthy lungs |
| Viral Pneumonia | Viral pneumonia infected lungs |
| Lung Opacity | Lung abnormalities and opacity regions |

Each class contains thousands of X-ray images used to train and evaluate the model.

---

## Technologies Used

### Programming Language
- Python

### Libraries and Frameworks
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub

### Development Environment
- Google Colab

---

## Project Pipeline

The project follows a complete deep learning workflow consisting of several stages.

### 1. Dataset Download
The dataset is automatically downloaded using **KaggleHub**, which allows direct access to Kaggle datasets within the notebook environment.

The dataset contains separate folders for each disease class.

---

### 2. Data Loading

Images are loaded using the TensorFlow function: image_dataset_from_directory()

Dataset split:

- **80% Training Data**
- **20% Validation Data**

---

### 3. Image Preprocessing

Before feeding images into the neural network, several preprocessing steps are applied.

#### Image Resizing
All images are resized to: 224*224 pixels


This ensures a consistent input size for the CNN model.

#### Grayscale Conversion
Chest X-ray images are loaded in **grayscale format** since color information is not necessary for medical X-ray analysis.

#### Normalization

Pixel values are scaled from: 0 – 255 → 0 – 1

## Model Architecture

The model used in this project is a **Convolutional Neural Network (CNN)** designed for image classification.

CNNs are particularly effective for computer vision tasks because they automatically learn spatial features from images.

### Network Structure

Input Image  

224 × 224 × 1


### Convolution Layer 1
- Filters: 32
- Kernel Size: 3×3
- Activation: ReLU

This layer extracts low-level features such as edges and textures.

---

### Max Pooling Layer 1
Reduces spatial dimensions and retains the most important features.

---

### Convolution Layer 2
- Filters: 64
- Kernel Size: 3×3
- Activation: ReLU

Learns more complex patterns such as lung structures.

---

### Max Pooling Layer 2
Further reduces feature map size and computational cost.

---

### Convolution Layer 3
- Filters: 128
- Kernel Size: 3×3
- Activation: ReLU

Extracts deeper features related to lung abnormalities.

---

### Max Pooling Layer 3
Reduces dimensionality before fully connected layers.

---

### Flatten Layer
Converts the 2D feature maps into a 1D vector so it can be processed by dense layers.

---

### Dense Layer
- 128 neurons
- Activation: ReLU

This layer learns complex relationships between extracted features.

---

### Dropout Layer
Dropout rate: **0.5**

Dropout randomly disables neurons during training to reduce **overfitting**, which is especially important for medical datasets.

---

### Output Layer
- 4 neurons
- Activation: Softmax

Each neuron represents a class:

- COVID
- Normal
- Viral Pneumonia
- Lung Opacity

The softmax function outputs probabilities for each class.

---

## Model Training

The model is compiled with the following configuration.

### Optimizer
Adam optimizer with learning rate:


0.001


### Loss Function

Sparse Categorical Crossentropy


This loss function is suitable for **multi-class classification problems**.

### Evaluation Metric

Accuracy


Training is performed for **10 epochs**, where the model iteratively learns patterns from the dataset.

---

## Model Evaluation

After training, the model performance can be evaluated using:

- Validation Accuracy
- Confusion Matrix
- Classification Report

These metrics help measure how well the model predicts each disease class.

---

## Expected Output

Given a chest X-ray image, the model predicts the probability of each disease category.

Example prediction:


Input: Chest X-ray image

Output:
COVID → 0.03
Normal → 0.92
Viral Pneumonia → 0.04
Lung Opacity → 0.01


Predicted Class: **Normal**

## Conclusion

This project demonstrates how **deep learning and computer vision techniques can be applied to medical imaging** for disease classification. By training a CNN on chest X-ray images, the system can automatically identify different lung conditions.

AI-powered medical image analysis has the potential to assist healthcare professionals by providing faster screening and improving diagnostic efficiency.
