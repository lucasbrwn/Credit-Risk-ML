# Credit Default Prediction

Beginner machine learning project that predicts whether a borrower will experience **serious delinquency within 2 years** using the *Give Me Some Credit* dataset.

## Overview
- Data cleaning (missing value handling and outlier clipping)
- Stratified train/test split for imbalanced data
- Models:
  - Random Forest (main model)
  - Logistic Regression (baseline)
- Evaluation using ROC-AUC and classification metrics
- Threshold tuning to improve recall for high-risk borrowers
- Random Forest feature importance
- Simple business impact estimation

## Dataset
This project uses the Kaggle dataset:  
https://www.kaggle.com/c/GiveMeSomeCredit/data  

Download `cs-training.csv` and place it in:

```
data/
```

(The dataset is not included due to Kaggle licensing.)

## How to Run
From the project root:

```
pip install -r requirements.txt
python src/creditml.py
```

## Results
- Random Forest ROC-AUC ≈ **0.84**
- Logistic Regression ROC-AUC ≈ **0.80**
- Random Forest performed better overall

## Interpretation
The Random Forest model showed stronger performance in distinguishing between high-risk and low-risk borrowers.

Lowering the decision threshold to 0.25 increased recall for default cases, allowing the model to identify more risky borrowers. This introduces more false positives, reflecting a real-world tradeoff between risk prevention and approving good customers.

Feature importance analysis indicated that credit utilization, age, debt ratio, and income were among the strongest predictors of default risk.

## Tech Stack
Python, Pandas, Scikit-learn