# Disease Diagnosis Prediction: Heart Disease 🫀

## 📌 Project Overview
This project aims to build a robust machine learning classification model capable of accurately predicting the likelihood of a patient having heart disease based on their medical history and symptoms. 

Cardiovascular diseases are the leading cause of death globally. Early detection through machine learning can significantly enhance diagnosis capabilities, assisting healthcare professionals in making data-driven decisions. In this repository, we compare two widely used machine learning algorithms—**Logistic Regression** and **Random Forest Classifier**—to determine the most reliable method for this critical task.

## 📊 Dataset Information
The project utilizes a Heart Disease Dataset consisting of **1190 instances** and **11 features**. 

**Target Variable:**
* `0`: Disease not present
* `1`: Disease present

**Key Features:**
* `Age` & `Sex`
* `Chest Pain Type` (Typical angina, atypical angina, non-anginal, asymptomatic)
* `Resting Blood Pressure` (mm Hg)
* `Serum Cholesterol` (mg/dl)
* `Fasting Blood Sugar`
* `Resting ECG Results`
* `Maximum Heart Rate Achieved`
* `Exercise-Induced Angina`
* `Oldpeak` (ST Depression)
* `ST Segment Slope`

## ⚙️ Methodology & Data Preprocessing
To ensure optimal model performance, the following feature engineering and preprocessing steps were applied:
1. **Handling Categorical Data:** Applied `OneHotEncoder` to features like `sex`, `chest pain type`, `resting ecg`, `exercise angina`, and `ST slope`.
2. **Feature Scaling:** Applied `StandardScaler` to numerical features (`age`, `resting bp s`, `cholesterol`, `max heart rate`, `oldpeak`) since models like Logistic Regression are sensitive to data scale.
3. **Data Splitting:** The dataset was split into 80% training and 20% testing sets.

## 🚀 Models Implemented
1. **Logistic Regression:** Used as a strong baseline for binary classification to find linear decision boundaries.
2. **Random Forest Classifier:** An ensemble learning method utilizing 100 decision trees (`n_estimators=100`) to capture complex, non-linear relationships and interactions between medical features.

## 📈 Results & Performance Comparison

The **Random Forest Classifier** significantly outperformed the Logistic Regression model, particularly in capturing the non-linear relationships present in the medical data. It effectively reduced both false positives and false negatives, which is crucial in healthcare predictions.

### Evaluation Metrics
| Metric | Logistic Regression (Class 0) | Logistic Regression (Class 1) | Random Forest (Class 0) | Random Forest (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.88 | 0.87 | **0.94** | **0.94** |
| **Recall** | 0.83 | 0.91 | **0.93** | **0.95** |
| **F1-Score** | 0.86 | 0.89 | **0.93** | **0.95** |
| **Overall Accuracy** | \- | **87.39%** | \- | **94.12%** |

### Confusion Matrix Summary
| Model | True Positives | True Negatives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 119 | 89 | 18 | 12 |
| **Random Forest** | **125** | **99** | **8** | **6** |

*Note: While Logistic Regression offered faster training and prediction times, the Random Forest model is the preferred choice for this task due to its superior accuracy and lower error rates.*

## 🛠️ Tools & Libraries
* **Python**
* **NumPy & Pandas** (Data manipulation)
* **Scikit-learn** (Machine learning algorithms & preprocessing)
* **Matplotlib & Seaborn** (Data visualization & correlation matrix)

## 💻 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/disease-diagnosis-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script (or Jupyter Notebook) to see the data preprocessing, model training, and evaluation results.