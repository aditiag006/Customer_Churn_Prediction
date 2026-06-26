\# Customer Churn Prediction



Predicting customer churn for a telecom provider using classical ML and gradient boosting, with a focus on handling class imbalance and model interpretability (SHAP).



\## Problem Statement

Customer churn — when a subscriber cancels or stops using a service — directly impacts recurring revenue. This project builds a binary classification model to predict which customers are likely to churn, so retention efforts can be targeted proactively.



\## Dataset

\[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7,043 customers, 21 features including demographics, account info, and subscribed services.



> Dataset is not included in this repo due to size/licensing. Download the CSV from the link above and place it in `data/raw/`.



\## Tech Stack

\- Python, pandas, NumPy

\- scikit-learn, XGBoost, imbalanced-learn (SMOTE)

\- SHAP (model interpretability)

\- Matplotlib, Seaborn



\## Setup

```bash

python -m venv churn\_env

churn\_env\Scripts\activate

pip install -r requirements.txt

```



\## Key Findings

\*- Target is moderately imbalanced: 73.5% No-churn vs 26.5% Churn
- `TotalCharges` loaded as string due to 11 blank values, all corresponding to new customers with `tenure = 0`; imputed as 0
- Several service columns (`OnlineSecurity`, `OnlineBackup`, etc.) had a redundant `"No internet service"` category tied to `InternetService`; consolidated into binary Yes/No
- Engineered `TotalServices` (count of subscribed add-ons) and `TenureGroup` (lifecycle-stage buckets) as additional features
- Baseline Logistic Regression: 76.2% accuracy; untuned XGBoost: 76.3% accuracy
- Accuracy alone was misleading: Logistic Regression and untuned XGBoost both scored ~76% accuracy, but Logistic Regression caught significantly more actual churners (70% recall vs 60%)
- GridSearchCV tuning (max_depth=7, learning_rate=0.01, n_estimators=200) improved XGBoost recall to 77%, F1 to 0.62, and ROC-AUC to 0.836 — surpassing Logistic Regression on every metric except accuracy and precision
- Tuned XGBoost selected as the final model, prioritizing recall over accuracy given the business cost of missing actual churners
- SHAP analysis identified the strongest churn drivers: two-year contracts strongly reduce churn risk; fiber optic internet, electronic check payment, low tenure, and high monthly charges increase it
- Local SHAP explanations (force plots) allow individual customer risk scores to be explained in plain terms, suitable for retention team use\*



## Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.762 | 0.54 | 0.70 | 0.61 | 0.832 |
| XGBoost (untuned) | 0.763 | 0.55 | 0.60 | 0.58 | 0.805 |
| **XGBoost (tuned)** | 0.75 | 0.52 | **0.77** | **0.62** | **0.836** |

**Final model:** Tuned XGBoost, selected for highest recall and ROC-AUC — prioritizing catching actual churners over raw accuracy, given the asymmetric business cost of missed churners vs. false alarms.









