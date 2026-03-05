import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

from data_cleaning import clean


def business_impact(cm, model_name):

    TN, FP, FN, TP = cm.ravel()

    loan_amount = 11700
    loss_rate = 0.85
    interest_rate = 0.13

    profit_per_good = loan_amount * interest_rate

    loss_from_defaults = FN * loan_amount * loss_rate
    lost_profit = FP * profit_per_good
    loss_prevented = TP * loan_amount * loss_rate

    net_impact = loss_prevented - loss_from_defaults - lost_profit

    print(f"\n=== Business Impact: {model_name} ===")
    print(f"Loss from approved bad loans: ${loss_from_defaults:,.0f}")
    print(f"Profit lost from rejecting good borrowers: ${lost_profit:,.0f}")
    print(f"Loss prevented by detecting risky borrowers: ${loss_prevented:,.0f}")
    print(f"Net Financial Impact: ${net_impact:,.0f}")

def print_confusion_results(cm, model_name):
    TN, FP, FN, TP = cm.ravel()

    print(f"\n=== {model_name} ===")
    print(f"Correctly identified good loans (TN): {TN}")
    print(f"Incorrectly flagged good loans (FP): {FP}")
    print(f"Incorrectly approved bad loans (FN): {FN}")
    print(f"Correctly identified bad loans (TP): {TP}")

# Load and clean the dataset
df = clean()

# SeriousDlqin2yrs: 1 means the borrower had serious delinquency
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# Split data into training and testing sets
# Stratify keeps the distribution the same since the data is imbalanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
lr_model = LogisticRegression(max_iter=10000, random_state=42, class_weight="balanced")
lr_model.fit(X_train, y_train)

y_pred_logistic = lr_model.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_logistic)
print_confusion_results(cm_lr, "Logistic Regression")


# DECISION TREE MODEL
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
dt_model.fit(X_train, y_train)

y_pred_tree = dt_model.predict(X_test)

cm_tree = confusion_matrix(y_test, y_pred_tree)
print_confusion_results(cm_tree, "Decision Tree")

# KNN MODEL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print_confusion_results(cm_knn, "KNN (k=7)")

precision_lr = precision_score(y_test, y_pred_logistic)
precision_tree = precision_score(y_test, y_pred_tree)
precision_knn = precision_score(y_test, y_pred_knn)

print("\n=== Precision Comparison (Predicting Serious Delinquency) ===")
print("Logistic Regression Precision:", precision_lr)
print("Decision Tree Precision:", precision_tree)
print("KNN Precision:", precision_knn)

print("\n=== Best Model for Predicting Serious Delinquency (Highest Precision) ===")

if precision_lr >= precision_tree and precision_lr >= precision_knn:
    print("Logistic Regression most accurately predicts serious delinquency.")
elif precision_tree >= precision_lr and precision_tree >= precision_knn:
    print("Decision Tree most accurately predicts serious delinquency.")
else:
    print("KNN most accurately predicts serious delinquency.")

recall_lr = recall_score(y_test, y_pred_logistic)
recall_tree = recall_score(y_test, y_pred_tree)
recall_knn = recall_score(y_test, y_pred_knn)

print("\n=== Recall Comparison (Catching Serious Delinquency) ===")
print("Logistic Regression Recall:", recall_lr)
print("Decision Tree Recall:", recall_tree)
print("KNN Recall:", recall_knn)

print("\n=== Best Model for Catching Serious Delinquency (Highest Recall) ===")

if recall_lr >= recall_tree and recall_lr >= recall_knn:
    print("Logistic Regression catches the most serious delinquency cases.")
elif recall_tree >= recall_lr and recall_tree >= recall_knn:
    print("Decision Tree catches the most serious delinquency cases.")
else:
    print("KNN catches the most serious delinquency cases.")

f1_lr = f1_score(y_test, y_pred_logistic)
f1_tree = f1_score(y_test, y_pred_tree)
f1_knn = f1_score(y_test, y_pred_knn)

print("\n=== Model Comparison (F1 Score for Catching Serious Delinquency) ===")
print("Logistic Regression F1:", f1_lr)
print("Decision Tree F1:", f1_tree)
print("KNN F1:", f1_knn)

print("\n=== Best Model for Catching Serious Delinquency (Highest F1) ===")

if f1_lr >= f1_tree and f1_lr >= f1_knn:
    print("Logistic Regression performed best based on F1-score for catching serious delinquency.")

elif f1_tree >= f1_lr and f1_tree >= f1_knn:
    print("Decision Tree performed best based on F1-score for catching serious delinquency.")

else:
    print("KNN performed best based on F1-score for catching serious delinquency.")


print("\n=== Model Performance Summary ===")

summary = {
    "Model": ["Logistic Regression", "Decision Tree", "KNN (k=7)"],
    "Precision (Serious Delinquency)": [precision_lr, precision_tree, precision_knn],
    "Recall (Catching Serious Delinquency)": [recall_lr, recall_tree, recall_knn],
    "F1 (Catching Serious Delinquency)": [f1_lr, f1_tree, f1_knn]
}

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

business_impact(cm_lr, "Logistic Regression")
business_impact(cm_tree, "Decision Tree")
business_impact(cm_knn, "KNN (k=7)")