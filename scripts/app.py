import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- 1. SMART PATH FINDING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)

#File Paths
model_path = os.path.join(ROOT_DIR, 'models', 'asian_cinema_model.joblib')
features_path = os.path.join(ROOT_DIR, 'models', 'feature_cols.joblib')
csv_path = os.path.join(ROOT_DIR, 'data', 'asian_cinema_stats_CLEAN.csv')

# Load Model Files
if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("❌ Model files missing from /scripts!")
    st.stop()

model = joblib.load(model_path)
feature_cols = joblib.load(features_path)

# Language Mapping Dictionary
LANG_MAP = {
    'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 
    'th': 'Thai', 'vi': 'Vietnamese', 'en': 'English'
}

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else None

df = load_data()

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Asian Cinema AI", layout="wide", page_icon="🏮")
st.title("🏮 Asian Cinema Intelligent Predictor")

# Sidebar for Inputs
st.sidebar.header("🎬 Movie Configuration")

# Updated range: 1945 - 2025
year = st.sidebar.slider("Release Year", 1945, 2025, 2024)
runtime = st.sidebar.number_input("Runtime (Minutes)", 1, 300, 105)

# (Popularity removed from sidebar as requested)

# Genre Picker
genre_list = [c for c in feature_cols if not c.startswith('lang') and c not in ['year', 'tmdb_popularity', 'runtime_min', 'budget', 'revenue']]
selected_genre = st.sidebar.selectbox("Primary Genre", sorted(genre_list))

# Language Picker
lang_cols = [c for c in feature_cols if c.startswith('lang_')]
lang_options = {LANG_MAP.get(c.replace('lang_', ''), c.replace('lang_', '')): c for c in lang_cols}
selected_lang_name = st.sidebar.selectbox("Original Language", sorted(lang_options.keys()))
selected_lang_col = lang_options[selected_lang_name]

# --- 4. PREDICTOR TOOL ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🤖 AI Rating Guess")
    if st.button("Generate Prediction"):
        # Create input row
        input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
        
        # Fill inputs (Popularity defaults to 50 internally)
        input_df['year'] = year
        input_df['runtime_min'] = runtime
        input_df['tmdb_popularity'] = 50.0 
        
        # Set Genre and Language Dummies
        if selected_genre in input_df.columns: input_df[selected_genre] = 1
        if selected_lang_col in input_df.columns: input_df[selected_lang_col] = 1
        
        prediction = model.predict(input_df)[0]
        st.metric("Predicted Letterboxd Score", f"{prediction:.2f} ⭐")
        
        stars = "⭐" * int(round(prediction))
        st.markdown(f"### Visual Rating: {stars}")
        st.progress(min(prediction/5.0, 1.0))

with col2:
    st.subheader("📊 Genre Performance")
    if df is not None:
        df_exploded = df.assign(genres=df['genres'].str.split(', ')).explode('genres')
        genre_stats = df_exploded.groupby('genres')['lb_rating'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(genre_stats, x='genres', y='lb_rating', color='lb_rating', 
                     color_continuous_scale='Viridis', labels={'genres': 'Genre', 'lb_rating': 'Avg Rating'})
        st.plotly_chart(fig, use_container_width=True)

# --- 5. CLEAN HISTORICAL TABLE ---
if df is not None:
    st.divider()
    st.subheader(f"📜 Top Movies from {year}")
    year_movies = df[df['year'] == year]
    
    if not year_movies.empty:
        cols = ['title', 'original_language', 'lb_rating']
        display_df = year_movies[cols].sort_values('lb_rating', ascending=False).head(10)
        
        display_df = display_df.rename(columns={
            'title': 'Title',
            'original_language': 'Language',
            'lb_rating': 'Letterboxd Rating'
        })
        
        display_df['Language'] = display_df['Language'].replace(LANG_MAP)
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.write(f"No historical data found for {year} in the dataset.")