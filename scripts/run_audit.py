import pandas as pd
import os

# Paths
original_path = 'data/asian_cinema_stats_CLEAN.csv'
final_path = 'data/asian_cinema_FINAL.csv'
audit_path = 'data/missing_ratings_audit.csv'

if not os.path.exists(original_path):
    print(f"❌ Error: {original_path} not found.")
    exit(1)

df_orig = pd.read_csv(original_path)

# Initialize final dataset if it doesn't exist by dropping rows without ratings
if not os.path.exists(final_path):
    print(f"ℹ️ Creating {final_path} by dropping rows with missing 'lb_rating'...")
    df_orig['lb_rating'] = pd.to_numeric(df_orig['lb_rating'], errors='coerce')
    df_final = df_orig.dropna(subset=['lb_rating'])
    df_final.to_csv(final_path, index=False)
    print(f"✅ {final_path} created ({len(df_final)} rows).")
else:
    df_final = pd.read_csv(final_path)

# Identify missing ratings in original for audit tracking
missing_titles = df_orig[df_orig['lb_rating'].isnull()]
missing_titles.to_csv(audit_path, index=False)

print(f"✅ Audit file created at {audit_path}")
print(f"Total movies in 'backlog': {len(missing_titles)}")
