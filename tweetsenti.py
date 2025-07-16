import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("text_emotion (2) 1.csv")

# Count the number of tweets per sentiment
sentiment_counts = df['sentiment'].value_counts()

# Plot the distribution of sentiments
plt.figure(figsize=(12, 6))
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title("Distribution of Emotions in Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig("sentiment_distribution.png")

# Print the sentiment counts
print("Sentiment Counts:")
print(sentiment_counts)
