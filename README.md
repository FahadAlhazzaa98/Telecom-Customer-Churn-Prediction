# Telecom Customer Churn Prediction

An end-to-end Machine Learning project designed to predict customer churn in the telecommunications industry using interpretable classification models.

**[Live Streamlit Application](https://telecom-customer-churn-prediction-g3vbghivucmfk47mxbcxxz.streamlit.app/)**

## Project Overview

Customer churn is an important challenge for telecommunications companies. This project explores historical customer characteristics and develops machine learning models to identify customers who may be at risk of discontinuing their services.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Methodology

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Logistic Regression and Random Forest training
4. Model comparison using classification metrics
5. Decision threshold optimization on validation data
6. Feature importance analysis
7. Final model evaluation and Streamlit deployment

The dataset contains 7,032 customer records after cleaning. The revised experiment used a 60% training, 20% validation, and 20% test split.

## Final Model Performance

The selected model is Logistic Regression, with a classification threshold of 0.30.

| Metric | Test Result |
|---|---|
| Accuracy | 77.83% |
| Precision | 56.13% |
| Recall | 75.94% |
| F1 Score | 0.6455 |
| ROC-AUC | 0.8497 |

The model correctly identified 284 out of 374 customers who churned in the test set.

## Key Findings

- Customer tenure was the most influential feature according to permutation importance.
- Month-to-month contracts were associated with higher churn rates.
- Contract duration and internet service type provided useful predictive information.
- Lowering the classification threshold improved recall while increasing false-positive predictions.

## Interactive Application

The Streamlit application allows users to enter customer information, estimate churn probability, and view the resulting risk classification.

## Limitations

This project uses a public historical telecommunications dataset and is intended for educational demonstration. Its results do not establish causal relationships or guarantee performance on future customers.

Earlier exploratory experiments used the same underlying dataset, so the final evaluation should not be interpreted as completely independent external validation.

## Future Improvements

- Cross-validation and hyperparameter optimization
- Probability calibration
- SHAP-based model explanations
- Evaluation on external or time-separated datasets
- Cost-sensitive churn intervention analysis

## Author

Fahad Alhazzaa

BSc Computer Science

[LinkedIn](https://www.linkedin.com/in/fahad-alhazzaa-798732231/)
