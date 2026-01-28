import sys
import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# PATH
# =============================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =============================
# CONFIG PAGE
# =============================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎥",
    layout="wide"
)

st.title("🎬 Movie Recommender System")
st.write("Recommandation hybride : Content-Based + Collaborative Filtering")

# =============================
# LOAD DATA (CACHED)
# =============================
@st.cache_data
def load_data():
    # Garde uniquement des fichiers légers pour GitHub/Streamlit
    movies = pd.read_csv("data/movies_clean.csv")
    ratings = pd.read_csv("data/ratings_clean.csv")
    return movies, ratings

movies, ratings = load_data()

# =============================
# BUILD MODELS (CACHED)
# =============================
@st.cache_resource
def build_models(movies, ratings):
    # CONTENT-BASED
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)

    # SIMPLE COLLABORATIVE (user-item matrix)
    user_item = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    return similarity, user_item

similarity, user_item = build_models(movies, ratings)

# =============================
# CONTENT-BASED RECOMMENDATION
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
# SIMPLE COLLABORATIVE RECOMMENDATION
# =============================
def collaborative_recommend(user_id, movies, user_item, top_n=10):
    if user_id not in user_item.index:
        return pd.DataFrame()
    
    # Scores = moyenne des notes des autres utilisateurs pondérée par similarité simple
    user_ratings = user_item.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    movie_scores = user_item[unrated_movies].mean(axis=0)
    top_movies = movie_scores.sort_values(ascending=False).head(top_n).index
    return movies[movies["movieId"].isin(top_movies)]

# =============================
# HYBRID RECOMMENDATION
# =============================
def hybrid_recommend(user_id, movie_title, movies, user_item, similarity, top_n=10):
    content_df = content_recommend(movie_title, movies, similarity, top_n*2)
    collab_df = collaborative_recommend(user_id, movies, user_item, top_n*2)

    if collab_df.empty:
        return content_df.head(top_n)
    if content_df.empty:
        return collab_df.head(top_n)

    hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates("movieId")
    return hybrid_df.head(top_n)

# =============================
# SIDEBAR
# =============================
st.sidebar.header("🎯 Paramètres")

user_ids = sorted(ratings["userId"].unique())
selected_user = st.sidebar.selectbox("Utilisateur", user_ids)

movie_titles = sorted(movies["title"].unique())
selected_movie = st.sidebar.selectbox("Film de référence", movie_titles)

top_n = st.sidebar.slider("Nombre de recommandations", 5, 20, 10)

# =============================
# GENERATE RECOMMENDATIONS
# =============================
if st.sidebar.button("🎬 Générer recommandations"):
    with st.spinner("Calcul des recommandations..."):
        recs = hybrid_recommend(
            user_id=selected_user,
            movie_title=selected_movie,
            movies=movies,
            user_item=user_item,
            similarity=similarity,
            top_n=top_n
        )

    if recs.empty:
        st.warning("Aucune recommandation trouvée 😢")
    else:
        st.success(f"Top {top_n} recommandations")
        st.dataframe(recs[["title", "genres"]])
