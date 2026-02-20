import requests
import pandas as pd
import time
import re
from letterboxdpy import movie as lb_movie

# API Configuration
API_KEY = "TMDB_API_KEY_REMOVED"
BASE_URL = "https://api.themoviedb.org/3"
START_YEAR = 1946
END_YEAR = 2025
MOVIES_PER_YEAR = 10

# Map user choice to TMDb ISO 639-1 codes
LANGUAGE_MAP = {
    "1": {"name": "Japanese", "code": "ja"},
    "2": {"name": "Korean", "code": "ko"},
    "3": {"name": "Chinese", "code": "zh"},
    "4": {"name": "Thai", "code": "th"},
    "5": {"name": "All Combined", "code": "ja|ko|zh|th"}
}

def clean_slug(title):
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower()).strip()
    slug = re.sub(r'\s+', '-', slug)
    return slug

def get_full_tmdb_details(movie_id):
    """Fetches secondary financial and meta data from TMDb."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY}
    
    try:
        data = requests.get(url, params=params).json()
        
        # safely extract nested lists
        genres = ", ".join([g['name'] for g in data.get('genres', [])])
        companies = ", ".join([c['name'] for c in data.get('production_companies', [])])
        countries = ", ".join([c['name'] for c in data.get('production_countries', [])])
        
        return {
            "tmdb_id": data.get('id'),
            "imdb_id": data.get('imdb_id'),
            "budget": data.get('budget', 0),
            "revenue": data.get('revenue', 0),
            "runtime_min": data.get('runtime', 0),
            "genres": genres,
            "production_companies": companies,
            "production_countries": countries,
            "status": data.get('status'),
            "tagline": data.get('tagline', "")
        }
    except:
        return {}

def get_letterboxd_rating(slug):
    """Scrapes Letterboxd scores using local slug guessing."""
    try:
        movie_instance = lb_movie.Movie(slug)
        if hasattr(movie_instance, 'rating') and movie_instance.rating:
            return str(movie_instance.rating)
    except Exception:
        pass
    return "None"

# --- MAIN EXECUTION ---

print("--- 🌏 Asian Cinema Data Collector 🌏 ---")
print("Select the language scope:")
print("1. Japanese (ja)")
print("2. Korean (ko)")
print("3. Chinese (zh)")
print("4. Thai (th)")
print("5. Combine All 4 (Top Asian Films)")

choice = input("\nEnter number (1-5): ").strip()
target = LANGUAGE_MAP.get(choice, LANGUAGE_MAP["1"]) # Default to Japanese if invalid

print(f"\n🚀 Starting collection for {target['name']} films ({START_YEAR}-{END_YEAR})...")
print(f"   Fetching Top {MOVIES_PER_YEAR} per year. This will take time.\n")

all_movies = []
total_requests = (END_YEAR - START_YEAR + 1) * MOVIES_PER_YEAR
counter = 0

for year in range(START_YEAR, END_YEAR + 1):
    discover_url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_original_language": target['code'],
        "primary_release_year": year,
        "sort_by": "popularity.desc",
        "page": 1
    }
    
    # fetch slightly more than needed in case some are invalid
    res = requests.get(discover_url, params=params).json()
    candidates = res.get('results', [])[:MOVIES_PER_YEAR]
    
    print(f"📅 {year} | Processing {len(candidates)} films...", end="\r")
    
    for film in candidates:
        details = get_full_tmdb_details(film['id'])
        slug = clean_slug(film['title'])
        lb_rating = get_letterboxd_rating(slug)
        
        row = {
            "year": year,
            "title": film['title'],
            "original_title": film['original_title'],
            "original_language": film['original_language'],
            "lb_rating": lb_rating,
            "tmdb_rating": film['vote_average'],
            "tmdb_popularity": film['popularity'],
            "vote_count": film['vote_count'],
            "release_date": film['release_date'],
            "overview": film['overview'],
            **details
        }
        all_movies.append(row)
        time.sleep(0.5)

# --- SAVE TO CSV ---
filename = f"asian_cinema_stats_{target['code'].replace('|','_')}.csv"
df = pd.DataFrame(all_movies)

# Reorder columns for logical reading
cols = [
    'year', 'title', 'lb_rating', 'tmdb_rating', 'genres', 'director', 
    'runtime_min', 'budget', 'revenue', 'original_language', 
    'production_companies', 'imdb_id', 'tmdb_id'
]
# Ensure we only use columns that actually exist (ignoring 'director' if not fetched above)
available_cols = [c for c in cols if c in df.columns] 
remaining_cols = [c for c in df.columns if c not in available_cols]
df = df[available_cols + remaining_cols]

df.to_csv(filename, index=False)

print(f"\n\n✅ COMPLETED! Saved {len(df)} films to '{filename}'.")
print(f"   You can import this directly into SQL or open in Excel.")