import pandas as pd
import numpy as np

# Load datasets
movies = pd.read_csv('dataset/movies.csv')
ratings = pd.read_csv('dataset/ratings.csv')

# Count number of movies rated per user
user_counts = ratings["userId"].value_counts()

# Filter users who rated at least 300 movies
active_users = user_counts[user_counts >= 300].index
filtered_ratings = ratings[ratings["userId"].isin(active_users)]

# Count number of ratings per movie
movie_counts = filtered_ratings["movieId"].value_counts()

# Filter movies that received at least 700 ratings
popular_movies = movie_counts[movie_counts >= 700].index
filtered_ratings = filtered_ratings[filtered_ratings["movieId"].isin(popular_movies)]

# Drop unnecessary column
filtered_ratings = filtered_ratings.drop(columns=["timestamp"])

# Convert data types to optimize memory usage
filtered_ratings["userId"] = filtered_ratings["userId"].astype("int32")
filtered_ratings["movieId"] = filtered_ratings["movieId"].astype("int32")
filtered_ratings["rating"] = filtered_ratings["rating"].astype("float32")

from scipy.sparse import csr_matrix

# Create user-movie matrix (pivot table)
user_movie_matrix = filtered_ratings.pivot(index="userId", columns="movieId", values="rating")

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Convert to sparse matrix to optimize memory
user_movie_sparse = csr_matrix(user_movie_matrix.values)

print("Matrix shape: ", user_movie_matrix.shape)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode genres
movies["genres_encoded"] = LabelEncoder().fit_transform(movies["genres"])

# Merge features and target
X = filtered_ratings.merge(movies[["movieId", "genres_encoded"]], on="movieId", how="left")[["userId", "movieId", "genres_encoded"]]
y = filtered_ratings["rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

print("XGBoost Model Trained!")

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Evaluate with RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.4f}")

# Evaluate with MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {mae:.4f}")

def recommend_movies_xgb(user_id, movies, model, top_n=10):
    if user_id not in filtered_ratings["userId"].unique():
        return f"Error: User ID {user_id} not found in dataset!"

    watched_movies = filtered_ratings[filtered_ratings["userId"] == user_id]["movieId"].unique()
    all_movies = movies["movieId"].unique()
    unwatched_movies = np.setdiff1d(all_movies, watched_movies)

    user_movie_pairs = pd.DataFrame({"userId": [user_id] * len(unwatched_movies), "movieId": unwatched_movies})
    user_movie_pairs = user_movie_pairs.merge(movies[["movieId", "genres_encoded"]], on="movieId", how="left")

    user_movie_pairs = user_movie_pairs.astype({"userId": "int32", "movieId": "int32", "genres_encoded": "int32"})
    user_movie_pairs["predicted_rating"] = model.predict(user_movie_pairs)

    recommended_movies = user_movie_pairs.sort_values(by="predicted_rating", ascending=False).head(top_n)
    recommended_movies = recommended_movies.merge(movies, on="movieId", how="left")

    return recommended_movies[["title", "genres", "predicted_rating"]]

import streamlit as st

st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=100000, value=10)
top_n = st.slider("Number of Recommendations:", 1, 20, 10)

if st.button("Get Recommendations"):
    recommendations = recommend_movies_xgb(user_id, movies, xgb_model, top_n)
    st.write(recommendations)
