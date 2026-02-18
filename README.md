# 🏮 East Asian Cinema Rating Predictor

A dashboard using compiled film meta data from East Asian film to predict letterboxd ratings using AI and machine learning
## Project Goal

This project uses machine learning (Random Forest) to predict Letterboxd ratings for East Asian films based on historical data from TMDB. It was built to explore how factors like language, genre, and runtime influence critical reception across different eras of cinema.

## Link

[Streamlit Link](https://asian-cinema-lbdpredict.streamlit.app/)
## Features

Interactive Predictor: Input movie stats to see an AI-generated rating.

Historical Analysis: Explore top-rated films by year (1945–2025).

Visual Analytics: Dynamic charts showing genre performance.

## Tools used

Python: Core logic and data processing.

Scikit-Learn: Random Forest Regressor for the prediction engine.

Streamlit: Interactive web dashboard.

Plotly: Interactive data visualizations.

SQL: Initial data extraction and cleaning.

## How to Run Locally

Clone the repo: `git clone https://github.com/eagleheart90/fuzzy-guacamole.git`
    
Install requirements: `pip install -r requirements.txt`

Launch app: `streamlit run scripts/app.py`

## Attributions
letterboxdpy

themoviedb.org
