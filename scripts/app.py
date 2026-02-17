import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. SMART PATH FINDING ---
# This finds the absolute path to the folder where THIS script (app.py) is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the full paths to your model files
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

# --- 3. DASHBOARD UI ---
st.set_page_config(page_title="Asian Cinema AI", page_icon="🏮")

st.title("🏮 Asian Cinema Rating Predictor")
st.markdown("""
This AI was trained on decades of Asian cinema data. 
Adjust the sliders to see what rating the 'brain' predicts for a movie with these stats.
""")

# Sidebar for inputs
st.sidebar.header("Movie Configuration")

# Year and Runtime are major predictors from our ML analysis
year = st.sidebar.slider("Release Year", 1950, 2026, 2024)
runtime = st.sidebar.number_input("Runtime (Minutes)", 1, 300, 105)
popularity = st.sidebar.slider("TMDb Popularity", 0.0, 500.0, 45.0)

# Financials
budget = st.sidebar.number_input("Budget (USD)", 0, 300000000, 5000000)
revenue = st.sidebar.number_input("Revenue (USD)", 0, 1000000000, 12000000)

# Identify genre columns from our feature list
genre_list = [c for c in feature_cols if c not in ['year', 'tmdb_popularity', 'runtime_min', 'budget', 'revenue']]
selected_genre = st.sidebar.selectbox("Primary Genre", sorted(genre_list))

# --- 4. PREDICTION LOGIC ---
if st.button("Generate AI Prediction"):
    # Create a DataFrame with 1 row, all zeros
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # Fill in the basics
    input_df['year'] = year
    input_df['runtime_min'] = runtime
    input_df['tmdb_popularity'] = popularity
    input_df['budget'] = budget
    input_df['revenue'] = revenue
    
    # Switch on the selected genre (One-Hot Encoding)
    if selected_genre in input_df.columns:
        input_df[selected_genre] = 1
        
    # Get the prediction
    prediction = model.predict(input_df)[0]
    
    # Display Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Letterboxd Score", f"{prediction:.2f} ⭐")
        
    with col2:
        if prediction >= 4.0:
            st.success("Verdict: A Potential Masterpiece")
        elif prediction >= 3.3:
            st.info("Verdict: Strong Critical Reception")
        else:
            st.warning("Verdict: Mixed or Niche Appeal")

    st.progress(min(prediction / 5.0, 1.0))

# --- 5. VISUAL ANALYTICS ---
st.divider()
st.subheader("📊 Global Insights: Genre vs. Rating")

# We can load a small sample of the clean data to show trends
@st.cache_data # This keeps the app fast by 'remembering' the data
def load_stats_data():
    # Look for the CSV in the parent directory since app.py is in /scripts
    csv_path = os.path.join(os.path.dirname(BASE_DIR), 'asian_cinema_stats_CLEAN.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df_stats = load_stats_data()

if df_stats is not None:
    # Create a simple bar chart of average ratings by Genre
    # Note: This is simplified for the dashboard
    avg_ratings = df_stats.groupby('genres')['lb_rating'].mean().sort_values(ascending=False).head(10)
    
    st.bar_chart(avg_ratings)
    st.caption("Average Letterboxd Rating by Genre (Top 10)")
else:
    st.info("💡 Tip: Place 'asian_cinema_stats_CLEAN.csv' in your main folder to see global trends here.")