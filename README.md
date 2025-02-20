# 🤖 Machine Learning - Comprehensive Model Suite

## 📌 Project Overview
This project implements various **Machine Learning** models to solve multiple real-world problems using supervised learning techniques. The goal is to develop an integrated system capable of handling different datasets and applying the appropriate model for each task.

## 🚀 Features
- **Simple Linear Regression**: Predicts concrete strength based on key ingredients.
- **Multiple Linear Regression**: Enhances predictions by using multiple input variables.
- **Logistic Regression**: Classifies industrial machinery failures using sensor data.
- **K-Nearest Neighbors (KNN)**: Estimates room occupancy based on environmental sensors.
- **Data Visualization**: Displays model predictions and comparisons.
- **Dataset Splitting**: Uses `train_test_split()` for training and testing datasets.
- **Performance Metrics**: Evaluates models using accuracy, confusion matrices, and visual comparisons.

## 📂 Implemented Models & Datasets

📂 **All CSV datasets are included in the repository**, allowing anyone to test and replicate the results.
📄 **The original PDF document (in Spanish) is also included**, detailing all program components and methodology.

### 1️⃣ Simple Linear Regression
**Dataset:** `ConcreteStrengthData.csv`
- Predicts concrete strength based on cement, water, and aggregates.
- Splits dataset into **80% training** and **20% testing**.
- Evaluates model performance with visual plots.

### 2️⃣ Multiple Linear Regression
**Dataset:** `ConcreteStrengthData.csv`
- Uses multiple features to improve prediction accuracy.
- Selects relevant features using a **correlation matrix**.
- Compares the impact of multiple variables on prediction.

### 3️⃣ Logistic Regression
**Dataset:** `ai4i2020.csv`
- Classifies machine failures (0 = No failure, 1 = Failure) using industrial sensor data.
- Implements separate models for each failure type.
- Evaluates performance using:
  - **Confusion matrices**
  - **Precision & accuracy metrics**
  - **Comparative plots**

### 4️⃣ K-Nearest Neighbors (KNN)
**Dataset:** `Occupancy_Estimation.csv`
- Predicts room occupancy using temperature, light, sound, CO2, and motion sensors.
- Implements **KNN classifier with K=7**.
- Compares different weightings of neighbors.
- Uses **accuracy** as the primary evaluation metric.

## 🛠️ Libraries Used
The project is built using the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
```

## 📊 Results & Visualizations
Each model generates:
- **Prediction graphs** comparing test vs predicted values.
- **Confusion matrices** for classification tasks.
- **Performance metrics** for accuracy and precision evaluation.

## 📜 License
This project is licensed under the **MIT License**.

---
✉️ **Author**: Angel Gabriel Cenciarini & Yasimel Joaquin Cabello  
📍 **Developed for Facultad de Ciencias de la Alimentación, UNER**
