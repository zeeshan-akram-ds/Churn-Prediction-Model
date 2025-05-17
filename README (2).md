
#  Customer Churn Prediction

This repository contains two comprehensive Jupyter notebooks developed to predict customer churn using various machine learning models and detailed exploratory data analysis. The project is focused on understanding churn patterns and building robust predictive models using both classical and boosting algorithms.

---

##  Project Overview

Customer churn is one of the key challenges faced by telecom companies. Reducing churn rate can lead to significantly higher profitability. This project demonstrates the end-to-end process of churn prediction — from data preprocessing to model building and evaluation.

---

##  Project Structure

```
.
├── customer_voting_churn.ipynb     # Main modeling file with VotingClassifier
├── Customer Churn Analysis.ipynb     # In-depth EDA and LightGBM model
├── requirements.txt             # Required packages
└── README.md                    # Project documentation
```

---

##  Highlights

### 1. `customer_voting_churn` (Modeling-Focused)

- **Data Cleaning**:
  - Handled incorrect `TotalCharges` data type.
  - Removed missing values.
  - Converted binary columns to 0/1.
  - Cleaned multi-class categorical variables.

- **Feature Engineering**:
  - Created `AvgMonthlySpend` from `TotalCharges / Tenure`.

- **Modeling Pipeline**:
  - Built pipeline with imputation, encoding, and scaling.
  - Applied several models: RandomForest, LogisticRegression, GradientBoosting, AdaBoost, XGBoost.
  - Tuned models using `Optuna`.
  - Final ensemble: **VotingClassifier (soft)** using `LogisticRegression`, `GradientBoosting`, and `AdaBoost`.

- **Threshold Optimization**:
  - Precision-Recall curve plotted to choose best threshold (0.4).
  - Best model performance at threshold = 0.4:

    ```
    precision    recall  f1-score   support

        0       0.88      0.81      0.84      1035
        1       0.57      0.70      0.63       374

    accuracy                           0.78      1409
    macro avg       0.73      0.75      0.74      1409
    ```

- **Evaluation Metrics**:
  - ROC AUC Score: **1.0**
  - Average Precision Score: **1.0**
  - Confusion Matrix and Classification Report included.

---

### 2. `Customer Churn Analysis.ipynb` (Exploratory Data Analysis & LightGBM)

- **EDA & Visualizations**:
  - Barplots, countplots, pie charts, boxplots, histograms.
  - Churn analysis by:
    - Contract type
    - Internet service
    - Payment method
    - Tenure group
    - Paperless billing
    - Senior citizen status
  - Feature correlation and churn heatmaps.

- **Feature Engineering**:
  - `StreamingBundle`, `HasBoth`, `TenureBins`, `NoSupportInternet`, `TotalBin`
  - Advanced engineered features:
    - `CostTenureRatio`, `ContractTenure`, `EarlyHighCost`
    - `MissingServicesCount`, `MonthlyCostZ_Risk`
    - `ChurnRiskScore`, `ChurnRiskLevel`, `ChurnRiskScoreBin`

- **Modeling**:
  - Used **LightGBM** with SMOTENC (resampled categorical-aware balancing).
  - Applied categorical type conversion for LightGBM optimization.
  - Tuned model and evaluated on original test set.

- **Model Results**:

    ```
    precision    recall  f1-score   support

        0       0.90      0.76      0.82      1035
        1       0.53      0.76      0.62       374

    accuracy                           0.76      1409
    ROC AUC Score: 0.8312
    Average Precision Score: 0.598
    ```

---

##  Final Notes

- The VotingClassifier in `customer_churn_voting.ipynb` provided the best balance of precision and recall, especially when using a custom threshold of 0.4.
- The LightGBM notebook offers valuable domain insights and detailed visualizations — ideal for understanding churn patterns and presenting data storytelling.

---

##  Installation

Install required packages:

```
pip install -r requirements.txt
```

---

##  Technologies Used

- Python 3.11
- Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Matplotlib, Seaborn
- Optuna for hyperparameter tuning
- SMOTENC for imbalanced classification

---

##  Contributing

Contributions are welcome! Feel free to fork the repository and open a pull request.

---

## 📧 Contact

For queries or collaborations, reach out via [GitHub Issues](https://github.com/) or connect on LinkedIn.
