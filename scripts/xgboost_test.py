import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('asian_cinema_stats_CLEAN.csv')
df = df.dropna(subset=['lb_rating', 'runtime_min', 'tmdb_popularity'])

# Prep Features
genres_dummies = df['genres'].str.get_dummies(sep=', ')
df_model = pd.concat([df, genres_dummies], axis=1)
feature_cols = ['year', 'tmdb_popularity', 'runtime_min'] + list(genres_dummies.columns)

X = df_model[feature_cols]
y = df_model['lb_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training XGBoost as an alternative to Random Forest
print("🚀 Launching XGBoost...")
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate performance
predictions = xgb_model.predict(X_test)
xgb_error = mean_absolute_error(y_test, predictions)

print(f"\n📊 XGBoost Error: {xgb_error:.4f}")
print(f"Comparison: {'XGBoost Wins!' if xgb_error < 0.32 else 'Random Forest is still King.'}")