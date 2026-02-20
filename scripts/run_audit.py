import pandas as pd
import os

# Paths
original_path = 'data/asian_cinema_stats_CLEAN.csv'
final_path = 'data/asian_cinema_FINAL.csv'
audit_path = 'data/missing_ratings_audit.csv'

# Load original
if not os.path.exists(original_path):
    print(f"❌ Error: {original_path} not found.")
    exit(1)

df_orig = pd.read_csv(original_path)

# If final doesn't exist, create it by dropping missing ratings (Remediation Logic)
if not os.path.exists(final_path):
    print(f"ℹ️ Creating {final_path} by dropping rows with missing 'lb_rating'...")
    # Convert lb_rating to numeric just in case (to find NaNs)
    df_orig['lb_rating'] = pd.to_numeric(df_orig['lb_rating'], errors='coerce')
    df_final = df_orig.dropna(subset=['lb_rating'])
    df_final.to_csv(final_path, index=False)
    print(f"✅ {final_path} created ({len(df_final)} rows).")
else:
    df_final = pd.read_csv(final_path)

# Find rows that exist in Original but NOT in Final
# These are the movies we "dropped" during remediation
missing_titles = df_orig[~df_orig['tmdb_id'].isin(df_final['tmdb_id'])]

# Save the audit list
missing_titles[['title', 'year', 'original_language']].to_csv(audit_path, index=False)

print(f"✅ Audit file created at {audit_path}")
print(f"Total movies in 'backlog': {len(missing_titles)}")
