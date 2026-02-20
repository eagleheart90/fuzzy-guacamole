import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import os

# 1. SMART PATH FINDING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 1. Load Data
data_path = os.path.join(ROOT_DIR, 'data', 'asian_cinema_RECOVERED.csv')
df = pd.read_csv(data_path)
df = df.dropna(subset=['lb_rating', 'runtime_min', 'tmdb_popularity'])

# 2. Encoding: Genres AND Languages
# We expand the brain to understand 'ja', 'ko', 'zh', etc.
genres_dummies = df['genres'].str.get_dummies(sep=', ')
lang_dummies = pd.get_dummies(df['original_language'], prefix='lang')

df_model = pd.concat([df, genres_dummies, lang_dummies], axis=1)

# 3. Define the New Feature Set
feature_cols = ['year', 'tmdb_popularity', 'runtime_min', 'budget', 'revenue'] 
feature_cols += list(genres_dummies.columns)
feature_cols += list(lang_dummies.columns)

X = df_model[feature_cols]
y = df_model['lb_rating']

# 4. Train the "Smarter" Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the new "Brain"
model_path = os.path.join(ROOT_DIR, 'models', 'asian_cinema_model.joblib')
features_path = os.path.join(ROOT_DIR, 'models', 'feature_cols.joblib')

joblib.dump(model, model_path)
joblib.dump(feature_cols, features_path)

print(f"✅ Success! Model updated with {len(lang_dummies.columns)} languages.")
print(f"New Error Rate: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")