# 🎬 XGBoost-Powered Movie Recommendation System

A personalized movie recommendation system built using the [MovieLens dataset](https://grouplens.org/datasets/movielens/), XGBoost for regression-based rating prediction, and Streamlit for an interactive user interface.

---

## 🚀 Project Overview

This project provides tailored movie suggestions based on a user’s previous ratings and the genre of each movie. It filters out inactive users and unpopular movies to improve model efficiency and prediction quality.

The core of the recommendation engine is powered by an `XGBoost Regressor`, which learns the patterns between users, movies, and their genres to predict ratings for unseen content.

---

## 📂 Dataset

- **ratings.csv** – Contains userId, movieId, rating, timestamp
- **movies.csv** – Contains movieId, title, genres

> Source: [MovieLens 100K/1M/10M](https://grouplens.org/datasets/movielens/)

---

## ⚙️ Features

✅ Filters:
- Users who rated at least **300** movies  
- Movies with at least **700** ratings  

✅ Machine Learning:
- Encodes movie genres using `LabelEncoder`
- Predicts ratings using `XGBoost Regressor`
- Evaluates model with **RMSE** and **MAE**

✅ Frontend:
- Built with **Streamlit**
- Interactive user ID input
- Adjustable number of recommended movies

---

## 🧠 Model Architecture

- Features:
  - `userId` (int32)
  - `movieId` (int32)
  - `genres_encoded` (int32)
- Target:
  - `rating` (float32)

Model: `XGBRegressor`  
Parameters:
- `max_depth=6`
- `learning_rate=0.05`
- `n_estimators=500`
- `subsample=0.8`
- `colsample_bytree=0.8`

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error on the test set |
| **MAE** | Mean Absolute Error for additional robustness |

---

## 💻 Installation & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/xgb-movie-recommender.git
cd xgb-movie-recommender
