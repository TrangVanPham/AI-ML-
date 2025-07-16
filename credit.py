import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("your_file 2.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Select numeric and nominal features
numeric_features = ['duration', 'credit_amount', 'installment_commitment', 'age']
nominal_features = ['checking_status', 'credit_history', 'purpose']

# Target variable
target = 'class'

# Encode nominal features
df_encoded = df[numeric_features + nominal_features + [target]].copy()
for col in nominal_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Encode target variable
df_encoded[target] = LabelEncoder().fit_transform(df_encoded[target].astype(str))

# Scale numeric features
scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Split the data into features and target
X = df_encoded[numeric_features + nominal_features]
y = df_encoded[target]

# Split into training (80%), validation (10%), and test (10%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)

# Train KNN classifiers with different k values and evaluate on validation set
k_values = range(1, 21)
accuracies = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    accuracies[k] = acc
    print(f"Validation Accuracy for k={k}: {acc:.4f}")

# Select the best k
best_k = max(accuracies, key=accuracies.get)
print(f"\nBest k value: {best_k}")

# Train final model with best k on combined training and validation set
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_val, y_train_val)
y_pred_test = final_knn.predict(X_test)

# Evaluate on test set
final_accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
