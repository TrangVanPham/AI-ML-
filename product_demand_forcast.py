import pandas as pd

# Load the dataset
file_path = "Historical_Product_Demand 1.csv"
df = pd.read_csv(file_path)

# Display column names and data types
column_info = df.dtypes
print("Column Names and Data Types:\n", column_info)

# Preview the first few rows
preview = df.head()
print("\nFirst 5 Rows of the Dataset:\n", preview)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values)

# Count unique values in key columns
unique_counts = {
    "Product_Code": df['Product_Code'].nunique(),
    "Warehouse": df['Warehouse'].nunique(),
    "Product_Category": df['Product_Category'].nunique()
}
print("\nUnique Value Counts:\n", unique_counts)
