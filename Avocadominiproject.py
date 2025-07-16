import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("avocado.csv")

# Drop missing values
df.dropna(inplace=True)

# Drop 'region' and 'Date' columns
df.drop(columns=['region', 'Date'], inplace=True)

# Encode categorical variables
df['type'] = LabelEncoder().fit_transform(df['type'])

# Separate features and target
X = df.drop(columns=['AveragePrice'])
y = df['AveragePrice']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train (80%), validation (10%), and test (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 ≈ 0.1

# Train KNN regressors with different k values and evaluate R-squared on validation set
r2_scores_knn = {}
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_val)
    r2_scores_knn[k] = r2_score(y_val, y_pred_val)

# Train a Linear Regression model and evaluate R-squared on validation set
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
r2_score_lr = r2_score(y_val, y_pred_lr)

# Print R-squared scores
print("R-squared scores for KNN with different k values:")
for k, score in r2_scores_knn.items():
    print(f"k={k}: R^2={score:.4f}")

print(f"\nR-squared score for Linear Regression: R^2={r2_score_lr:.4f}")

# Optional: Plot R-squared scores
plt.figure(figsize=(10, 6))
plt.plot(list(r2_scores_knn.keys()), list(r2_scores_knn.values()), marker='o', label='KNN R²')
plt.axhline(y=r2_score_lr, color='r', linestyle='--', label='Linear Regression R²')
plt.title('R² Scores for KNN and Linear Regression')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('R² Score')
plt.legend()
plt.grid(True)
plt.show()
