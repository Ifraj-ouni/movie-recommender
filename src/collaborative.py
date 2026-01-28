import os
import pandas as pd
import pickle

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "ratings_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)
def load_data():
    return pd.read_csv(DATA_PATH)
def prepare_surprise_data(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )
    return data
def train_svd(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )

    model.fit(trainset)

    predictions = model.test(testset)
    print("📊 RMSE :", rmse(predictions))

    return model
def save_model(model):
    with open(os.path.join(MODEL_PATH, "svd_model.pkl"), "wb") as f:
        pickle.dump(model, f)
if __name__ == "__main__":
    ratings = load_data()
    data = prepare_surprise_data(ratings)

    model = train_svd(data)
    save_model(model)

    print("✅ Collaborative filtering model trained and saved")
