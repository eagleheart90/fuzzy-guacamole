import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('asian_cinema_stats_CLEAN.csv')
df = df.dropna(subset=['lb_rating', 'runtime_min', 'tmdb_popularity'])

genres_dummies = df['genres'].str.get_dummies(sep=', ')
df_model = pd.concat([df, genres_dummies], axis=1)

feature_cols = ['year', 'tmdb_popularity', 'runtime_min'] + list(genres_dummies.columns)
X = df_model[feature_cols]
y = df_model['lb_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random search over typical forest hyperparameters
param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("🏎️ Starting Hyperparameter Tuning... this might take a minute.")
rf = RandomForestRegressor(random_state=42)

search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_distributions, 
    n_iter=20, 
    cv=3, 
    verbose=1, 
    random_state=42, 
    n_jobs=-1
)

search.fit(X_train, y_train)

best_rf = search.best_estimator_
predictions = best_rf.predict(X_test)
new_error = mean_absolute_error(y_test, predictions)

print("\n🏆 TUNING COMPLETE")
print(f"Best Parameters Found: {search.best_params_}")
print(f"Original Error: 0.32")
print(f"New Error: {new_error:.4f}")

if new_error < 0.32:
    print(f"🎉 Success! You reduced the error by {0.32 - new_error:.4f} stars.")
else:
    print("📈 The model is already very optimized, or we need a larger search space!")