import pandas as pd
import numpy as np

def clean_data():
    # 1. Load the data
    try:
        df = pd.read_csv('asian_cinema_stats_ja_ko_zh_th.csv')
        print("✅ File loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: The CSV file was not found.")
        return

    # 2. Convert Letterboxd Rating from string "None" to actual numbers
    # errors='coerce' turns "None" into NaN (Not a Number)
    df['lb_rating'] = pd.to_numeric(df['lb_rating'], errors='coerce')

    # 3. Replace 0s with NaN in financial columns
    # This prevents the 0s from skewing your averages
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)

    # 4. Convert release_date to actual dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # 5. Create a Profit column
    df['profit'] = df['revenue'] - df['budget']

    # --- QUICK ANALYSIS ---
    print("\n--- Summary Stats ---")
    print(f"Total Movies: {len(df)}")
    print(f"Movies with LB Ratings: {df['lb_rating'].notnull().sum()}")
    print(f"Average LB Rating: {df['lb_rating'].mean():.2f}")

    # Show the top 5 most profitable movies
    print("\n--- Top 5 Profitable Movies ---")
    top_profit = df[['title', 'year', 'profit']].sort_values(by='profit', ascending=False).head()
    print(top_profit)

    # 6. Save the cleaned data to a new file
    df.to_csv('asian_cinema_stats_CLEAN.csv', index=False)
    print("\n✅ Cleaned data saved to 'asian_cinema_stats_CLEAN.csv'")

if __name__ == "__main__":
    clean_data()