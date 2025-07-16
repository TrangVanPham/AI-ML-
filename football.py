# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("results.csv")

# Drop missing values
df.dropna(inplace=True)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# 1. Count of tuples (rows)
tuple_count = len(df)
print(f"Total number of tuples: {tuple_count}")

# 2. Count of unique tournament names
unique_tournaments = df['tournament'].nunique()
print(f"Number of unique tournaments: {unique_tournaments}")

# 3. Count of matches in 2018
matches_2018 = df[df['date'].dt.year == 2018].shape[0]
print(f"Number of matches in 2018: {matches_2018}")

# 4. Count of home wins, away wins, and draws
home_wins = (df['home_score'] > df['away_score']).sum()
away_wins = (df['home_score'] < df['away_score']).sum()
draws = (df['home_score'] == df['away_score']).sum()
print(f"Home Wins: {home_wins}, Away Wins: {away_wins}, Draws: {draws}")

# 5. Pie chart of match outcomes
outcomes = [home_wins, away_wins, draws]
labels = ['Home Wins', 'Away Wins', 'Draws']
plt.figure(figsize=(6, 6))
plt.pie(outcomes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Match Outcomes')
plt.show()

# 6. Pie chart of the 'neutral' column
df['neutral'].value_counts().plot.pie(
    autopct='%1.1f%%',
    labels=['Not Neutral', 'Neutral'],
    title='Neutral Venue Distribution'
)
plt.ylabel('')
plt.show()

# 7. Count of unique team names from home and away teams
unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel()).size
print(f"Number of unique team names: {unique_teams}")
