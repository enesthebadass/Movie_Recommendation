# 🎬 Movie Recommendation System using XGBoost

This project builds a movie recommendation system using the MovieLens dataset. It filters users and movies based on activity levels and uses the XGBoost Regressor to predict movie ratings for individual users.

## Features
- Filters active users (rated ≥ 300 movies)
- Filters popular movies (rated ≥ 700 times)
- Creates user-movie pivot table for memory-efficient modeling
- Uses `XGBoost` for rating prediction
- Deploys a recommendation interface via `Streamlit`

## Dataset
- `movies.csv` and `ratings.csv` from the [MovieLens dataset](https://grouplens.org/datasets/movielens/)

## Installation
```bash
pip install -r requirements.txt
