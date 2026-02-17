import pandas as pd
import numpy as np

# Load the clean data
df = pd.read_csv('asian_cinema_stats_CLEAN.csv')

# 1. THE "GOLDEN ERA" FINDER
# Which decade had the highest average Letterboxd ratings?
df['decade'] = (df['year'] // 10) * 10
decade_trends = df.groupby('decade')['lb_rating'].agg(['mean', 'count']).rename(columns={'mean': 'Avg Rating', 'count': 'Films'})

print("--- 🏆 Average Letterboxd Rating by Decade ---")
print(decade_trends)
print("\n")

# 2. INDUSTRY DOMINANCE
# Which language (ja, ko, zh, th) appears most often in the "Top 10 Popularity" lists?
language_counts = df['original_language'].value_counts()

print("--- 🌍 Frequency in Popularity Lists by Language ---")
print(language_counts)
print("\n")

# 3. CULT CLASSIC DETECTOR
# Find films where Letterboxd fans (cinephiles) rate the movie much 
# higher than the general TMDb public.
# Note: TMDb is 1-10, LB is 1-5, so we divide TMDb by 2 to compare.
df['rating_diff'] = df['lb_rating'] - (df['tmdb_rating'] / 2)

# Use numpy to create a "Status" label
df['cult_status'] = np.where(df['rating_diff'] > 0.5, "Cult Classic", "Mainstream")

print("--- 📽️ Top 5 'Cult Classics' (LB Rating > TMDb Rating) ---")
cult_classics = df[df['lb_rating'].notnull()].sort_values(by='rating_diff', ascending=False)
print(cult_classics[['title', 'year', 'lb_rating', 'tmdb_rating', 'rating_diff']].head())
print("\n")

# 4. BIGGEST FINANCIAL SURPRISES
# Best ROI (Return on Investment) for films that have budget data
# We use ROI = Revenue / Budget
df['roi'] = df['revenue'] / df['budget']
print("--- 💰 Top 5 Highest Return on Investment (ROI) ---")
print(df[['title', 'year', 'roi']].sort_values(by='roi', ascending=False).head())