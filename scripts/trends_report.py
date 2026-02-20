import pandas as pd
import numpy as np


df = pd.read_csv('asian_cinema_stats_CLEAN.csv')

# Decade-level rating trends
df['decade'] = (df['year'] // 10) * 10
decade_trends = df.groupby('decade')['lb_rating'].agg(['mean', 'count']).rename(columns={'mean': 'Avg Rating', 'count': 'Films'})

print("--- 🏆 Average Letterboxd Rating by Decade ---")
print(decade_trends)
print("\n")

# Distribution of languages in top popularity slices
language_counts = df['original_language'].value_counts()
print("--- 🌍 Frequency in Popularity Lists by Language ---")
print(language_counts)
print("\n")

# Identify 'Cult Classics' where cinephile (LB) ratings exceed general (TMDb) ratings
# Aligning scales by dividing TMDb (0-10) by 2
df['rating_diff'] = df['lb_rating'] - (df['tmdb_rating'] / 2)
df['cult_status'] = np.where(df['rating_diff'] > 0.5, "Cult Classic", "Mainstream")

print("--- 📽️ Top 5 'Cult Classics' (LB Rating > TMDb Rating) ---")
cult_classics = df[df['lb_rating'].notnull()].sort_values(by='rating_diff', ascending=False)
print(cult_classics[['title', 'year', 'lb_rating', 'tmdb_rating', 'rating_diff']].head())
print("\n")

# Return on investment for films with known financials
df['roi'] = df['revenue'] / df['budget']
print("--- 💰 Top 5 Highest Return on Investment (ROI) ---")
print(df[['title', 'year', 'roi']].sort_values(by='roi', ascending=False).head())