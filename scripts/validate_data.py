import pandas as pd
import numpy as np
import os

def validate_data(file_path):
    print(f"--- 🔍 Data Validation Report for {os.path.basename(file_path)} ---")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    total_rows = len(df)
    print(f"Total Rows: {total_rows}\n")

    # 1. Schema Integrity
    expected_columns = [
        'year', 'title', 'lb_rating', 'tmdb_rating', 'genres', 'runtime_min',
        'budget', 'revenue', 'original_language', 'production_companies',
        'imdb_id', 'tmdb_id', 'original_title', 'tmdb_popularity', 'vote_count',
        'release_date', 'overview', 'production_countries', 'status', 'tagline',
        'profit'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Schema Integrity: FAILED (Missing columns: {', '.join(missing_cols)})")
    else:
        print("✅ Schema Integrity: PASSED (All expected columns found)")

    # 2. Boundary Checks
    print("\n--- Boundary Checks ---")
    
    # lb_rating: 0-5
    if 'lb_rating' in df.columns:
        # Handle NaN values for boundary checks
        valid_lb = df['lb_rating'].dropna()
        lb_outliers = valid_lb[(valid_lb < 0) | (valid_lb > 5)]
        if not lb_outliers.empty:
            print(f"❌ lb_rating: FAILED ({len(lb_outliers)} values outside [0, 5])")
        else:
            print(f"✅ lb_rating: PASSED (All valid values within [0, 5])")
    
    # tmdb_rating: 0-10
    if 'tmdb_rating' in df.columns:
        valid_tmdb = df['tmdb_rating'].dropna()
        tmdb_outliers = valid_tmdb[(valid_tmdb < 0) | (valid_tmdb > 10)]
        if not tmdb_outliers.empty:
            print(f"❌ tmdb_rating: FAILED ({len(tmdb_outliers)} values outside [0, 10])")
        else:
            print(f"✅ tmdb_rating: PASSED (All valid values within [0, 10])")

    # year: 1945-2025
    if 'year' in df.columns:
        year_outliers = df[(df['year'] < 1945) | (df['year'] > 2025)]
        if not year_outliers.empty:
            print(f"❌ year: FAILED ({len(year_outliers)} values outside [1945, 2025])")
        else:
            print(f"✅ year: PASSED (All years within [1945, 2025])")

    # 3. Consistency Check
    print("\n--- Consistency Checks ---")
    if 'profit' in df.columns and 'revenue' in df.columns and 'budget' in df.columns:
        # Check if profit == revenue - budget
        # We handle NaNs by comparing only rows where all three are present
        valid_mask = df[['profit', 'revenue', 'budget']].notnull().all(axis=1)
        mismatch = df[valid_mask & (abs(df['profit'] - (df['revenue'] - df['budget'])) > 0.01)]
        
        if not mismatch.empty:
            print(f"❌ Profit Calculation Integrity: FAILED ({len(mismatch)} rows have profit != revenue - budget)")
        else:
            print("✅ Profit Calculation Integrity: PASSED (profit == revenue - budget)")
    else:
        print("⚠️ Profit Consistency: SKIPPED (Missing required columns: 'profit', 'revenue', or 'budget')")

    # 4. Completeness Check
    print("\n--- Completeness Check (Missing Values %) ---")
    missing_report = df.isnull().mean() * 100
    for col, pct in missing_report.items():
        status = "✅" if pct == 0 else "⚠️"
        print(f"{status} {col}: {pct:.2f}% missing")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'asian_cinema_stats_CLEAN.csv')
    validate_data(os.path.abspath(path))
