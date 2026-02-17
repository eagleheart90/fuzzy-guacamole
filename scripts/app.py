import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px # type: ignore

# --- 1. SMART PATH FINDING ---
# This finds the absolute path to the folder where THIS script (app.py) is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Load Data for Insights
csv_path = os.path.join(BASE_DIR, 'asian_cinema_data.csv')

@st.cache_data
def load_data():
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else None

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
        # Show top genres by rating
        genre_stats = df.groupby('genres')['lb_rating'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(genre_stats, x='genres', y='lb_rating', color='lb_rating', 
                     labels={'lb_rating': 'Avg Rating', 'genres': 'Genre'})
        st.plotly_chart(fig, use_container_width=True)

# Bottom Section: Historical Context
if df is not None:
    st.divider()
    st.subheader("📜 Historical Context: Movies from " + str(year))
    year_movies = df[df['year'] == year][['title', 'director', 'lb_rating']].sort_values('lb_rating', ascending=False)
    if not year_movies.empty:
        st.table(year_movies.head(5))
    else:
        st.write("No historical data found for this exact year.")