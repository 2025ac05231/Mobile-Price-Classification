

# Mobile Price Classification using Machine Learning

## Problem Statement

The aim of this project is to classify the price range of a mobile phone based on its specifications using different machine learning algorithms. The objective is to compare the performance of multiple classification models and identify the model that performs best on the given dataset. A Streamlit application is also developed to allow users to upload test data, select a trained model and view the prediction results along with evaluation metrics.

## Dataset Description

Dataset Name: Mobile Price Classification

Source:
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

The dataset contains 2000 mobile phone records with 20 input features and one target variable (`price_range`). The target consists of four price categories.

There are no missing values or duplicate records in the dataset, making it suitable for training classification models without extensive preprocessing.

Some of the important features include:

- Battery Power
- RAM
- Internal Memory
- Mobile Depth
- Mobile Weight
- Front Camera
- Primary Camera
- Screen Height
- Screen Width
- Battery Talk Time
- WiFi
- Bluetooth
- 4G Support
- 3G Support

## GitHub Repository

Repository Link:

https://github.com/2025ac05231/Mobile-Price-Classification

## Streamlit Application

Application Link:

https://mobile-price-classification-pawzhknhjohcs6nxxpxgf8.streamlit.app/

## Models Implemented

The following machine learning models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest Classifier

## Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|---------:|---------:|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Gaussian Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest | 0.8800 | 0.9769 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |

## Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Achieved the highest overall performance with **96.5% accuracy** and the best values for Precision, Recall, F1 Score, MCC, and AUC. It was the most reliable model for this dataset. |
| Decision Tree | Delivered moderate performance (83.0% accuracy). It was able to classify the data reasonably well but was less accurate than Logistic Regression and Random Forest. |
| K-Nearest Neighbors (KNN) | Achieved the lowest performance (50.0% accuracy) on the current train-test split. The model may benefit from hyperparameter tuning and further optimization. |
| Gaussian Naive Bayes | Produced acceptable performance (81.0% accuracy) while maintaining a simple and computationally efficient model. |
| Random Forest (Ensemble) | Achieved strong overall performance (88.0% accuracy) and ranked second among all models. The ensemble approach improved classification compared to a single Decision Tree. |
| **Overall Winner** | **Logistic Regression** achieved the best overall performance with an accuracy of **96.5%**, AUC of **0.9987**, and the highest MCC (**0.9534**). Based on these evaluation metrics, it was selected as the best model for this dataset. |


## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```