# Telco Customer Churn Prediction

## Project Overview

An end-to-end Machine Learning project that predicts customer churn in the telecommunications industry.

The goal is to identify customers who are likely to leave the company and understand the factors influencing their decisions, helping businesses develop effective customer retention strategies.

## Dataset

- Dataset: Telco Customer Churn
- Number of records: 7,043 customers
- Number of columns: 21
- Target variable: `Churn`
- Problem type: Binary Classification

## Project Workflow

### 1. Data Understanding
- Dataset exploration and structure analysis.
- Numerical and categorical feature identification.
- Target variable distribution analysis.

### 2. Data Cleaning & Preprocessing
- Handled missing values and hidden blanks.
- Converted data types.
- Removed duplicate records.
- Standardized categorical values.
- Checked numerical features for outliers.

### 3. Exploratory Data Analysis (EDA)
- Customer demographics analysis.
- Services and billing behavior analysis.
- Contract and tenure analysis.
- Churn rate visualization.
- Correlation analysis.
- Identification of key churn indicators.

### 4. Feature Engineering & Selection

Created additional features:
- `num_addons`
- `avg_monthly_charge`
- `charge_change`
- `has_family`

Applied:
- Label Encoding
- One-Hot Encoding
- StandardScaler
- Mutual Information
- Random Forest Feature Importance

### 5. Model Building

Trained and compared 12 Machine Learning models.

**Individual Models:**
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

**Ensemble Models:**
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- Hist Gradient Boosting
- Voting Classifier
- Stacking Classifier

### 6. Model Evaluation

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curves

## Model Performance

Random Forest achieved the highest F1-score among the 12 evaluated models.

| Metric | Score |
|---|---:|
| Accuracy | 76.7% |
| Precision | 54.3% |
| Recall | 77.8% |
| F1-Score | 64.0% |
| ROC-AUC | 84.4% |

## Key Business Insights

- Month-to-month contracts have a significantly higher churn rate (42.7%) compared to two-year contracts (2.8%).
- Customers with shorter tenure are more likely to churn.
- Fiber-optic customers have a higher churn rate (41.9%).
- Electronic-check payment users show a churn rate of 45.3%.
- Online Security and Tech Support services are associated with significantly lower churn rates.

## Streamlit Application

An interactive Streamlit application was developed to:
- Enter customer information.
- Predict whether a customer is likely to churn or stay.
- Display churn probability.
- Visualize the prediction probability.

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
telco-customer-churn-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
├── Telco_Churn_Project.ipynb
├── app.py
└── churn_model.joblib
```

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/sohaelzelaky/telco-customer-churn-prediction.git
```

2. Navigate to the project directory:

```bash
cd telco-customer-churn-prediction
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Future Improvements

- Hyperparameter tuning using cross-validation.
- Decision threshold optimization.
- Experimenting with XGBoost, LightGBM, and CatBoost.
- SHAP-based model explainability.
- Model monitoring and probability calibration.

## Author

**Soha Elzelaky**
