import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "movies_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

def load_data():
    return pd.read_csv(DATA_PATH)
def prepare_features(movies):
    movies["text"] = movies["title"] + " " + movies["genres"]
    return movies
def train_content_model(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["text"])

    similarity = cosine_similarity(tfidf_matrix)

    return tfidf, similarity
def save_model(tfidf, similarity, movies):
    with open(os.path.join(MODEL_PATH, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    with open(os.path.join(MODEL_PATH, "similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)

    movies.to_csv(os.path.join(MODEL_PATH, "movies_index.csv"), index=False)
if __name__ == "__main__":
    movies = load_data()
    movies = prepare_features(movies)

    tfidf, similarity = train_content_model(movies)
    save_model(tfidf, similarity, movies)

    print("✅ Content-based model trained and saved")
