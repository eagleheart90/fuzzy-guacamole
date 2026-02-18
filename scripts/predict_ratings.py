import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load Data
df = pd.read_csv('asian_cinema_stats_CLEAN.csv')
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
# This will overwrite your old files in the main folder
joblib.dump(model, 'asian_cinema_model.joblib')
joblib.dump(feature_cols, 'feature_cols.joblib')

print(f"✅ Success! Model updated with {len(lang_dummies.columns)} languages.")
print(f"New Error Rate: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")