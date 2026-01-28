import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================
# BUILD MODELS
# =============================
def build_models(movies, ratings):
    # -------- CONTENT-BASED --------
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    content_similarity = cosine_similarity(tfidf_matrix)

    # -------- COLLABORATIVE --------
    user_item = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    user_similarity = cosine_similarity(user_item)

    return content_similarity, user_similarity, user_item


# =============================
# CONTENT-BASED
# =============================
def content_recommend(movie_title, movies, similarity, top_n=10):
    if movie_title not in movies["title"].values:
        return pd.DataFrame()

    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in scores]
    return movies.iloc[movie_indices]


# =============================
# COLLABORATIVE
# =============================
def collaborative_recommend(user_id, movies, ratings, user_similarity, user_item, top_n=10):
    if user_id not in user_item.index:
        return pd.DataFrame()

    user_idx = list(user_item.index).index(user_id)
    similar_users = user_similarity[user_idx].argsort()[::-1][1:11]

    mean_ratings = user_item.iloc[similar_users].mean(axis=0)

    watched = ratings[ratings["userId"] == user_id]["movieId"]
    recommendations = mean_ratings.drop(watched, errors="ignore")

    top_movies = recommendations.sort_values(ascending=False).head(top_n).index
    return movies[movies["movieId"].isin(top_movies)]


# =============================
# HYBRID
# =============================
def hybrid_recommend(
    user_id,
    movie_title,
    movies,
    ratings,
    content_similarity,
    user_similarity,
    user_item,
    top_n=10
):
    content_df = content_recommend(movie_title, movies, content_similarity, top_n * 2)
    collab_df = collaborative_recommend(
        user_id, movies, ratings, user_similarity, user_item, top_n * 2
    )

    if content_df.empty:
        return collab_df.head(top_n)
    if collab_df.empty:
        return content_df.head(top_n)

    hybrid = pd.concat([content_df, collab_df]).drop_duplicates("movieId")
    return hybrid.head(top_n)
