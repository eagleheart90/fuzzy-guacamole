import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the clean data
df = pd.read_csv('asian_cinema_stats_CLEAN.csv')

# Pre-calculate decade for the charts
df['decade'] = (df['year'] // 10) * 10

# Set a professional style
plt.style.use('ggplot')

# --- GRAPH 1: RATING TRENDS OVER TIME ---
plt.figure(figsize=(10, 6))

# Aggregate ratings by decade
decade_ratings = df.groupby('decade')[['lb_rating', 'tmdb_rating']].mean()
# Normalize TMDb to a 5-star scale for comparison
decade_ratings['tmdb_normalized'] = decade_ratings['tmdb_rating'] / 2

plt.plot(decade_ratings.index, decade_ratings['lb_rating'], marker='o', label='Letterboxd (Cinephiles)', color='#ff8000', linewidth=2)
plt.plot(decade_ratings.index, decade_ratings['tmdb_normalized'], marker='s', label='TMDb (General Public)', color='#00d2ff', linewidth=2)

plt.title('Asian Cinema Quality Trends (1946-2025)', fontsize=14)
plt.xlabel('Decade')
plt.ylabel('Average Rating (out of 5)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show() # This opens the window on your Mac


# --- GRAPH 2: LANGUAGE DOMINANCE BY DECADE ---
plt.figure(figsize=(10, 6))

# Count how many of each language appear per decade
lang_decade = df.groupby(['decade', 'original_language']).size().unstack().fillna(0)

lang_decade.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

plt.title('Top 10 Popular Movies: Language Distribution by Decade', fontsize=14)
plt.xlabel('Decade')
plt.ylabel('Number of Films in Top 10 Lists')
plt.legend(title='Language Code')
plt.tight_layout()
plt.show()


# --- GRAPH 3: PROFIT VS RATING (SCATTER PLOT) ---
# We only plot movies that have both a rating and profit data
plot_df = df.dropna(subset=['lb_rating', 'profit']).copy()

# 2. ELIMINATE OUTLIERS: Only keep profits below $2,000 million
# This prevents one or two massive hits from ruining the scale
limit_m = 2000 
plot_df = plot_df[plot_df['profit'] < (limit_m * 1_000_000)]

# 3. CONVERT PROFIT TO MILLIONS for easier reading
plot_df['profit_m'] = plot_df['profit'] / 1_000_000

# --- GRAPH 3: REFINED SCATTER PLOT ---
plt.figure(figsize=(12, 7))

# Create the scatter plot
# Using 'viridis' colormap: yellow is newest, dark purple is oldest
scatter = plt.scatter(
    plot_df['lb_rating'], 
    plot_df['profit_m'], 
    c=plot_df['year'], 
    cmap='viridis', 
    alpha=0.7, 
    edgecolors='w', 
    s=80 # size of dots
)

# ADD BEST FIT LINE (Linear Regression)
# Calculate the slope (m) and intercept (b)
m, b = np.polyfit(plot_df['lb_rating'], plot_df['profit_m'], 1)
# Create the line coordinates
line_x = np.array([plot_df['lb_rating'].min(), plot_df['lb_rating'].max()])
line_y = m * line_x + b

plt.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, label=f'Trend Line (Slope: {m:.2f})')

# Aesthetics
plt.title(f'Relationship Between Letterboxd Rating and Profit (Films < ${limit_m}M Profit)', fontsize=15)
plt.xlabel('Letterboxd Rating (1-5 Stars)', fontsize=12)
plt.ylabel('Profit (Millions of USD)', fontsize=12)

# Colorbar for Years
cbar = plt.colorbar(scatter)
cbar.set_label('Release Year', fontsize=12)

plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()