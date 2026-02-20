import pandas as pd
import numpy as np
import os
import time
from letterboxdpy.search import Search
from letterboxdpy.movie import Movie


# Paths
AUDIT_PATH = 'data/missing_ratings_audit.csv'
CLEAN_PATH = 'data/asian_cinema_stats_CLEAN.csv'
OUTPUT_PATH = 'data/asian_cinema_RECOVERED.csv'

def get_decade(year):
    return (year // 10) * 10

def fetch_lb_rating(title, year):
    """Attempt to scrape Letterboxd rating by guessing slug from title/year search."""
    try:
        print(f"🔍 Searching Letterboxd for: {title} ({year})...")
        search = Search(title)
        films = search.results.get('films', [])
        
        for film in films:
            film_year = film.get('year')
            # Handle string year or None
            try:
                film_year = int(film_year) if film_year else None
            except:
                film_year = None
                
            if film_year == year:
                slug = film.get('url').split('/')[-2]
                movie = Movie(slug)
                rating = movie.rating
                if rating:
                    print(f"  ✅ Found rating: {rating}")
                    return float(rating)
        
        print("  ⚠️ No matching film found in search results.")
        return None
    except Exception as e:
        print(f"  ❌ Error fetching rating: {e}")
        return None

def main():
    if not os.path.exists(AUDIT_PATH) or not os.path.exists(CLEAN_PATH):
        print("❌ Required files missing.")
        return

    df_audit = pd.read_csv(AUDIT_PATH)
    df_clean = pd.read_csv(CLEAN_PATH)

    # Calculate genre-based medians as fallback for missing ratings
    print("📊 Calculating genre medians...")
    df_with_ratings = df_clean[df_clean['lb_rating'].notnull()].copy()
    df_with_ratings['decade'] = df_with_ratings['year'].apply(get_decade)
    
    df_genres = df_with_ratings.copy()
    df_genres['primary_genre'] = df_genres['genres'].str.split(',').str[0].str.strip()
    
    median_map = df_genres.groupby(['primary_genre', 'decade'])['lb_rating'].median().to_dict()
    global_median = df_with_ratings['lb_rating'].median()

    recovered_ratings = []
    total = len(df_audit)
    
    for i, row in df_audit.iterrows():
        title = row['title']
        year = int(row['year'])
        primary_genre = str(row['genres']).split(',')[0].strip() if pd.notnull(row['genres']) else 'Unknown'
        decade = get_decade(year)
        
        print(f"[{i+1}/{total}] Processing: {title}")
        rating = fetch_lb_rating(title, year)
        
        if rating is None:
            # Hierarchical fallback: genre/decade -> genre -> global
            rating = median_map.get((primary_genre, decade))
            if rating is None:
                genre_median = df_genres[df_genres['primary_genre'] == primary_genre]['lb_rating'].median()
                rating = genre_median if not np.isnan(genre_median) else global_median
            print(f"  💡 Fallback to median: {rating:.2f}")
            method = 'median'
        else:
            method = 'fetched'
            
        recovered_ratings.append({'tmdb_id': row['tmdb_id'], 'new_lb_rating': rating, 'method': method})
        time.sleep(0.5)

    # Merge recovered data back into main dataset
    df_recovered_map = pd.DataFrame(recovered_ratings)
    df_final = df_clean.merge(df_recovered_map, on='tmdb_id', how='left')
    df_final['lb_rating'] = df_final['lb_rating'].combine_first(df_final['new_lb_rating'])
    
    df_final = df_final.drop(columns=['new_lb_rating', 'method'], errors='ignore')
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n🎉 Recovery complete! Saved to {OUTPUT_PATH}")
    print(f"Total rows updated: {len(df_audit)}")

if __name__ == "__main__":
    main()
