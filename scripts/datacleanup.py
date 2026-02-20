import pandas as pd
import numpy as np


def clean_data():
    try:
        df = pd.read_csv('asian_cinema_stats_ja_ko_zh_th.csv')
        print("✅ File loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: The CSV file was not found.")
        return

    # Handle missing ratings and 0s in financial columns
    df['lb_rating'] = pd.to_numeric(df['lb_rating'], errors='coerce')
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['profit'] = df['revenue'] - df['budget']

    print("\n--- Summary Stats ---")
    print(f"Total Movies: {len(df)}")
    print(f"Movies with LB Ratings: {df['lb_rating'].notnull().sum()}")
    print(f"Average LB Rating: {df['lb_rating'].mean():.2f}")

    print("\n--- Top 5 Profitable Movies ---")
    top_profit = df[['title', 'year', 'profit']].sort_values(by='profit', ascending=False).head()
    print(top_profit)

    df.to_csv('asian_cinema_stats_CLEAN.csv', index=False)
    print("\n✅ Cleaned data saved to 'asian_cinema_stats_CLEAN.csv'")

if __name__ == "__main__":
    clean_data()