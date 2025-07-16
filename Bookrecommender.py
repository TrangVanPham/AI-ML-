
import pandas as pd

# Load the dataset, skipping malformed lines
df = pd.read_csv("books.csv", on_bad_lines='skip')

# Display initial shape
print(f"Initial dataset shape: {df.shape}")

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows with missing values in key columns
key_columns = ['title', 'authors', 'average_rating', 'ratings_count']
df.dropna(subset=key_columns, inplace=True)

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Display cleaned shape
print(f"Cleaned dataset shape: {df.shape}")

# Save the cleaned dataset
df.to_csv("books_cleaned.csv", index=False)
print("Cleaned dataset saved as 'books_cleaned.csv'")

def popularity_recommender(data, top_n=10):
    C = data['average_rating'].mean()
    m = data['ratings_count'].quantile(0.90)
    qualified = data[data['ratings_count'] >= m].copy()
    qualified['weighted_rating'] = qualified.apply(
        lambda x: (x['ratings_count'] / (x['ratings_count'] + m) * x['average_rating']) +
                  (m / (m + x['ratings_count']) * C), axis=1)
    return qualified.sort_values('weighted_rating', ascending=False)[['title', 'authors', 'average_rating', 'ratings_count']].head(top_n)

def content_based_recommender(data, book_title, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['authors'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title'].str.lower())
    
    idx = indices.get(book_title.lower())
    if idx is None:
        return f"Book titled '{book_title}' not found in the dataset."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    
    return data.iloc[book_indices][['title', 'authors', 'average_rating']]
