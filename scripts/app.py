import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- 1. SMART PATH FINDING ---
# This finds the absolute path to the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to find the CSV
csv_path = os.path.join(os.path.dirname(BASE_DIR), 'asian_cinema_stats_CLEAN.csv')

model_path = os.path.join(BASE_DIR, 'asian_cinema_model.joblib')
features_path = os.path.join(BASE_DIR, 'feature_cols.joblib')

# --- 2. LOAD DATA & MODEL ---
# Check if the files exist before trying to load them
if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("❌ Model files not found!")
    st.write(f"I looked in: `{BASE_DIR}`")
    st.info("Make sure you moved 'asian_cinema_model.joblib' and 'feature_cols.joblib' into the /scripts folder.")
    st.stop()

model = joblib.load(model_path)
feature_cols = joblib.load(features_path)


@st.cache_data
def load_data():
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.warning("⚠️ Data file not found! The dashboard will have limited functionality.")
        st.write(f"I looked for: `{csv_path}`")
        return None

df = load_data()

# --- 2. SIDEBAR INPUTS ---
st.set_page_config(page_title="Asian Cinema AI", layout="wide")
st.sidebar.title("🎬 Movie Specs")

# Dynamic Director Search (Feature for the UI)
if df is not None and 'director' in df.columns:
    directors = sorted(df['director'].dropna().unique())
    selected_dir = st.sidebar.selectbox("Director Search (Historical Reference)", ["None"] + directors)
    if selected_dir != "None":
        dir_avg = df[df['director'] == selected_dir]['lb_rating'].mean()
        st.sidebar.info(f"💡 {selected_dir}'s average rating: {dir_avg:.2f}")

# Main Predictor Inputs
year = st.sidebar.slider("Release Year", 1950, 2026, 2024)
runtime = st.sidebar.number_input("Runtime (Minutes)", 1, 300, 105)
popularity = st.sidebar.slider("TMDb Popularity", 0.0, 500.0, 45.0)

# Genre Picker
genre_cols = [c for c in feature_cols if c not in ['year', 'tmdb_popularity', 'runtime_min', 'budget', 'revenue']]
selected_genre = st.sidebar.selectbox("Primary Genre", sorted(genre_cols))

# --- 3. MAIN DASHBOARD ---
st.title("🏮 Asian Cinema Intelligent Dashboard")

# Top Section: The Predictor
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🤖 AI Rating Predictor")
    if st.button("Calculate Score"):
        input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
        input_df[['year', 'runtime_min', 'tmdb_popularity']] = [year, runtime, popularity]
        if selected_genre in input_df.columns: input_df[selected_genre] = 1
        
        prediction = model.predict(input_df)[0]
        st.metric("Predicted Score", f"{prediction:.2f} / 5")
        st.progress(min(prediction/5, 1.0))

with col2:
    st.subheader("📊 Genre Performance")
    if df is not None:
        # 1. Handle movies with multiple genres (e.g., 'Action, Drama')
        # This splits the strings and creates a separate row for each genre
        df_exploded = df.assign(genre_list=df['genres'].str.split(', ')).explode('genre_list')
        
        # 2. Calculate average ratings for each individual genre
        genre_stats = df_exploded.groupby('genre_list')['lb_rating'].mean().sort_values(ascending=False).head(10).reset_index()
        
        # 3. Create the interactive bar chart
        fig = px.bar(
            genre_stats, 
            x='genre_list', 
            y='lb_rating', 
            color='lb_rating',
            color_continuous_scale='Viridis',
            labels={'lb_rating': 'Avg Rating', 'genre_list': 'Genre'}
        )
        
        # Adjust the layout for better visibility in the dashboard
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- 5. HISTORICAL CONTEXT ---
if df is not None:
    st.divider()
    st.subheader(f"📜 Historical Context: Movies from {year}")
    
    # Filter movies for the selected year
    year_movies = df[df['year'] == year]
    
    if not year_movies.empty:
        # Dynamically choose which columns to show based on what's in the CSV
        columns_to_show = ['title', 'lb_rating']
        if 'director' in df.columns:
            columns_to_show.insert(1, 'director')
        elif 'production_companies' in df.columns:
            # Optionally show studio if director is missing
            columns_to_show.insert(1, 'production_companies')
            
        # Select and sort the available columns
        display_df = year_movies[columns_to_show].sort_values('lb_rating', ascending=False)
        
        #Rename columns for better display
        #Create dictionary for renaming
        display_df = display_df.rename(columns={
            'title': 'Title',
            'lb_rating': 'Letterboxd Rating',
            'production_companies': 'Studio',
        })

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.write(f"No historical data found for {year}.")