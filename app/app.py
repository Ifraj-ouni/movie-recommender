import sys
import os
import streamlit as st
import pandas as pd

# =============================
# PATH
# =============================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hybrid import build_models, hybrid_recommend

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
    movies = pd.read_csv("data/movies_clean.csv")
    ratings = pd.read_csv("data/ratings_clean.csv")
    return movies, ratings


movies, ratings = load_data()

# =============================
# BUILD MODELS (CACHED)
# =============================
@st.cache_resource
def load_models():
    similarity, svd = build_models(movies, ratings)
    return similarity, svd


similarity, svd = load_models()

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
# RECOMMEND
# =============================
if st.sidebar.button("🎬 Générer recommandations"):
    with st.spinner("Calcul des recommandations..."):
        recs = hybrid_recommend(
            user_id=selected_user,
            movie_title=selected_movie,
            movies=movies,
            ratings=ratings,
            similarity=similarity,
            svd=svd,
            top_n=top_n
        )

    if recs.empty:
        st.warning("Aucune recommandation trouvée 😢")
    else:
        st.success(f"Top {top_n} recommandations")
        st.dataframe(recs[["title", "genres"]])
