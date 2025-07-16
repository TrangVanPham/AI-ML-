import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('flavors_of_cacao.csv')

# 1. Count the number of tuples in the dataset
num_tuples = len(df)
print(f"Number of tuples in the dataset: {num_tuples}")

# 2. Count the number of unique company names
num_unique_companies = df['Company\n(Maker-if known)'].nunique()
print(f"Number of unique company names: {num_unique_companies}")

# 3. Count the number of reviews in 2013
num_reviews_2013 = df[df['Review\nDate'] == 2013].shape[0]
print(f"Number of reviews in 2013: {num_reviews_2013}")

# 4. Count the number of missing values in the 'Bean\nType' column
num_missing_bean_type = df['Bean\nType'].isnull().sum()
print(f"Number of missing values in 'Bean\nType' column: {num_missing_bean_type}")

# 5. Plot a histogram of the 'Rating' column
plt.hist(df['Rating'], bins=20, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Ratings')
plt.show()

# 6. Convert 'Cocoa\nPercent' from string to numerical values and plot scatter plot against 'Rating'
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('%', '').astype(float)
plt.scatter(df['Cocoa\nPercent'], df['Rating'], alpha=0.1)
plt.xlabel('Cocoa Percent')
plt.ylabel('Rating')
plt.title('Scatter Plot of Cocoa Percent vs Rating')
plt.show()

# 7. Normalize the 'Rating' column
df['NormalizedRating'] = (df['Rating'] - df['Rating'].min()) / (df['Rating'].max() - df['Rating'].min())
print("Normalized Ratings:")
print(df['NormalizedRating'])

# List companies ordered by their average score
avg_score_by_company = df.groupby('Company\n(Maker-if known)')['Rating'].mean().sort_values(ascending=False)
print("Companies ordered by their average score:")
print(avg_score_by_company)

# 8. Encode categorical columns 'Company\n(Maker-if known)' and 'Company\nLocation'
df_encoded = pd.get_dummies(df, columns=['Company\n(Maker-if known)', 'Company\nLocation'])
print("Encoded DataFrame:")
print(df_encoded.head())