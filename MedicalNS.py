import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("KaggleV2-May-2016.csv")

# Drop missing values
df.dropna(inplace=True)

# Extract relevant features
features = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
X = df[features]
y = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)  # Encode target: 1 for No-show, 0 for Show

# Encode categorical features
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

# Scale numerical features
scaler = StandardScaler()
X[['Age']] = scaler.fit_transform(X[['Age']])

# Split the data into train (80%), validation (10%), and test (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)

# Train Decision Tree classifiers with different criteria
criteria = ['gini', 'entropy']
dt_results = {}
for criterion in criteria:
    dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_val = dt.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    dt_results[criterion] = (acc, dt)

# Select best Decision Tree model
best_criterion = max(dt_results, key=lambda k: dt_results[k][0])
best_dt_model = dt_results[best_criterion][1]

# Evaluate best Decision Tree on test set
y_pred_test_dt = best_dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_test_dt)
dt_conf_matrix = confusion_matrix(y_test, y_pred_test_dt)

# Train Random Forest classifiers with different number of estimators
estimators = [10, 50, 100]
rf_results = {}
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_val = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    rf_results[n] = (acc, rf)

# Select best Random Forest model
best_n_estimators = max(rf_results, key=lambda k: rf_results[k][0])
best_rf_model = rf_results[best_n_estimators][1]

# Evaluate best Random Forest on test set
y_pred_test_rf = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_test_rf)
rf_conf_matrix = confusion_matrix(y_test, y_pred_test_rf)

# Print results
print(f"Decision Tree (criterion={best_criterion}) Test Accuracy: {dt_accuracy:.4f}")
print("Decision Tree Confusion Matrix:")
print(dt_conf_matrix)

print(f"\nRandom Forest (n_estimators={best_n_estimators}) Test Accuracy: {rf_accuracy:.4f}")
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)

print("\nClassification Report for Best Random Forest Model:")
print(classification_report(y_test, y_pred_test_rf))
