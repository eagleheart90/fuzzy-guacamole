import requests
import pandas as pd
import time
import re
from letterboxdpy import movie as lb_movie

# --- CONFIGURATION ---
API_KEY = "93a1728123a8625023a361d994ce2021"  # <--- PASTE YOUR KEY HERE
BASE_URL = "https://api.themoviedb.org/3"
START_YEAR = 1946
END_YEAR = 2025
MOVIES_PER_YEAR = 10  # Top 10 by popularity

# Map user choice to TMDb ISO 639-1 codes
LANGUAGE_MAP = {
    "1": {"name": "Japanese", "code": "ja"},
    "2": {"name": "Korean", "code": "ko"},
    "3": {"name": "Chinese", "code": "zh"},
    "4": {"name": "Thai", "code": "th"},
    "5": {"name": "All Combined", "code": "ja|ko|zh|th"}
}

def clean_slug(title):
    """
    Converts 'Seven Samurai' -> 'seven-samurai' for Letterboxd URL guessing.
    """
    # Lowercase, remove special chars (keep spaces/hyphens), strip whitespace
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower()).strip()
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    return slug

def get_full_tmdb_details(movie_id):
    """
    Fetches the 'Deep' data: Budget, Revenue, Production Companies, etc.
    """
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
    """
    Uses letterboxdpy to scrape the specific Letterboxd rating.
    Returns the string "None" if missing, for SQL friendliness.
    """
    try:
        # Initialize the library's Movie class
        movie_instance = lb_movie.Movie(slug)
        
        # Check if the rating attribute exists and is populated
        if hasattr(movie_instance, 'rating') and movie_instance.rating:
            return str(movie_instance.rating)
            
    except Exception:
        # If page not found or library error, fail silently
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
    # 1. Discover the hits of the year
    discover_url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_original_language": target['code'],
        "primary_release_year": year,
        "sort_by": "popularity.desc",
        "page": 1
    }
    
    # We fetch slightly more than needed in case some are invalid
    res = requests.get(discover_url, params=params).json()
    candidates = res.get('results', [])[:MOVIES_PER_YEAR]
    
    print(f"📅 {year} | Processing {len(candidates)} films...", end="\r")
    
    for film in candidates:
        # 2. Get the "Deep" details from TMDb
        details = get_full_tmdb_details(film['id'])
        
        # 3. Get the Rating from Letterboxd
        slug = clean_slug(film['title'])
        lb_rating = get_letterboxd_rating(slug)
        
        # 4. Build the Master Row
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
            # Merged details:
            **details
        }
        
        all_movies.append(row)
        
        # Progress counter
        counter += 1
        
        # Rate Limiting (Crucial for 800+ requests)
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