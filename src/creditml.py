
# Run this script from the project root:
# python src/creditml.py

"""
Credit Default Prediction - Beginner Machine Learning Project

Goal:
    Predict whether a borrower will experience serious delinquency within 2 years.

What this script does:
    1) Loads and cleans the dataset using clean() from data_cleaning.py
       - Fills missing MonthlyIncome and NumberOfDependents with the median
       - Clips extreme outliers for MonthlyIncome and DebtRatio
    2) Splits data into training and testing sets (stratified to keep class balance similar)
    3) Trains two models:
       - Random Forest (main model)
       - Logistic Regression (baseline model)
    4) Compares model performance using:
       - ROC-AUC
       - Classification report (precision, recall, F1)
       - Confusion matrix
    5) Tests a lower probability threshold (0.25) to increase recall for risky borrowers
       - Threshold can be changed
    6) Prints the top 10 feature importances (Random Forest)
    7) Estimates a simple business impact using the confusion matrix:
       - FN: approved bad loans (missed defaults)
       - FP: rejected good customers (lost profit)
       - TP: prevented losses by correctly flagging risky borrowers

Notes:
    - This is an educational project and the business impact section is a simplified estimate.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from data_cleaning import clean

# Load and clean the dataset
df = clean()

# SeriousDlqin2yrs: 1 means the borrower had serious delinquency
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# Split data into training and testing sets
# Stratify keeps the distribution the same since the data is imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create Random Forest Model
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Train the model
rf.fit(X_train, y_train)

# Predicted probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# Predictions using the default threshold (0.50)
y_pred = rf.predict(X_test)

print("\n=== Random Forest (default threshold = 0.50) ===")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Lower threshold to improve recall for defaults, but it introduces more false positives
threshold = 0.25
y_pred_custom = (y_prob >= threshold).astype(int)

print("\n=== Random Forest (threshold = 0.25) ===")
print(classification_report(y_test, y_pred_custom))

# TN = True Negatives, FP = False Positives
# FN = False Negatives, TP = True Positives
c_matrix = confusion_matrix(y_test, y_pred_custom)
print("Confusion Matrix: ")
print(c_matrix)

TN, FP, FN, TP = c_matrix.ravel()

# Feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n=== Random Forest Feature Importances (Top 10) ===")
print(importance.head(10))

# Example business assumptions (simplified for illustration)
loan_amount = 10000
loss_rate = 0.8
profit_per_good = 1000

# Approved bad loans (False Negatives)
loss_from_defaults = FN * loan_amount * loss_rate

# Rejected good loans (False Positives)
lost_profit = FP * profit_per_good

# Correctly flagged defaulters (True Positives) → losses prevented
loss_prevented = TP * loan_amount * loss_rate

print("\n=== Business Impact Estimate ===")
print("Loss from approved bad loans: $", loss_from_defaults)
print("Profit lost from rejecting good customers: $", lost_profit)
print("Loss prevented by model: $", loss_prevented)

# Train Logistic Regression as a simpler baseline model
log_model = LogisticRegression(max_iter=5000, class_weight='balanced')
log_model.fit(X_train, y_train)

# Predictions
y_prob_log = log_model.predict_proba(X_test)[:, 1]
y_pred_log = log_model.predict(X_test)

print("\n=== Logistic Regression (default threshold = 0.50) ===")
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
print(classification_report(y_test, y_pred_log))

# Compares model performance using ROC-AUC
print("\n=== Model Comparison ===")
rf_auc = roc_auc_score(y_test, y_prob)
log_auc = roc_auc_score(y_test, y_prob_log)

print("Random Forest ROC-AUC:", rf_auc)
print("Logistic Regression ROC-AUC:", log_auc)

if rf_auc > log_auc:
    print("Random Forest performs better overall (higher ROC-AUC)")
else:
    print("Logistic Regression performs better overall (higher ROC-AUC)")