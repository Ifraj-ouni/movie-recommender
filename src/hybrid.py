import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


# =============================
# BUILD MODELS
# =============================
def build_models(movies, ratings):
    # CONTENT-BASED
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(tfidf_matrix)

    # COLLABORATIVE
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    return similarity, svd


# =============================
# CONTENT-BASED
# =============================
def content_recommend(movie_title, movies, similarity, top_n=10):
    if movie_title not in movies["title"].values:
        return pd.DataFrame()

    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


# =============================
# COLLABORATIVE
# =============================
def collaborative_recommend(user_id, movies, ratings, svd, top_n=10):
    watched = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    candidates = movies[~movies["movieId"].isin(watched)]

    preds = []
    for _, row in candidates.iterrows():
        est = svd.predict(user_id, row["movieId"]).est
        preds.append((row["movieId"], est))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    movie_ids = [p[0] for p in preds]

    return movies[movies["movieId"].isin(movie_ids)]


# =============================
# HYBRID
# =============================
def hybrid_recommend(
    user_id,
    movie_title,
    movies,
    ratings,
    similarity,
    svd,
    top_n=10
):
    content_df = content_recommend(movie_title, movies, similarity, top_n * 2)
    collab_df = collaborative_recommend(user_id, movies, ratings, svd, top_n * 2)

    if collab_df.empty:
        return content_df.head(top_n)

    if content_df.empty:
        return collab_df.head(top_n)

    hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates("movieId")
    return hybrid_df.head(top_n)
