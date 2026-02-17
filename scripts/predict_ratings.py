import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load the cleaned data
filename = 'asian_cinema_stats_CLEAN.csv'
df = pd.read_csv(filename)

# 2. Data Cleaning for Machine Learning
# We must remove any rows where our Target (lb_rating) is missing
# We also remove rows where key features like runtime or popularity are missing
df = df.dropna(subset=['lb_rating', 'runtime_min', 'tmdb_popularity'])

# 3. Feature Engineering: Handling Genres
# Convert strings like "Action, Drama" into separate 0 and 1 columns
genres_dummies = df['genres'].str.get_dummies(sep=', ')
df_model = pd.concat([df, genres_dummies], axis=1)

# 4. Define our Features (X) and our Target (y)
# X = The variables we use to guess (Year, Popularity, Runtime + Genres)
feature_cols = ['year', 'tmdb_popularity', 'runtime_min'] + list(genres_dummies.columns)

# Fill missing budget/revenue with 0 so the math doesn't break
df_model['budget'] = df_model['budget'].fillna(0)
df_model['revenue'] = df_model['revenue'].fillna(0)
feature_cols += ['budget', 'revenue']

X = df_model[feature_cols]
y = df_model['lb_rating']

# 5. Split the data
# 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train the Model
print("🤖 Training the Rating Predictor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Make Predictions and Calculate Error
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"\n📊 Model Results:")
print(f"Average Error: {error:.2f} stars")

# 8. Show Sample Predictions
print("\n--- Top 10 Test Results (Actual vs Predicted) ---")
results = pd.DataFrame({
    'Title': df.loc[X_test.index, 'title'], 
    'Actual': y_test, 
    'Predicted': predictions.round(2)
})
print(results.head(10))

# 9. Feature Importance
# This tells us which factors actually drive a high rating
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Top 5 Predictors of a High Rating ---")
print(importances.head(5))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='#ff8000')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
plt.title(f'Machine Learning: Actual vs. Predicted Ratings (Error: {error:.2f})')
plt.xlabel('Actual Letterboxd Rating')
plt.ylabel('Model Predicted Rating')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()