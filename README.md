# 🎬 Movie Recommender System

A **Hybrid Movie Recommendation System** built using **Machine Learning and Streamlit**.  
This application recommends movies to users by combining **Content-Based Filtering** and **Collaborative Filtering**.

🔗 **Live Demo:**  
https://movie-recommender-ajnaazpumzu7mpna4xqvfh.streamlit.app/

---

# 📌 Project Overview

Recommendation systems are widely used in platforms like **Netflix, Amazon, and Spotify** to suggest relevant content to users.

This project implements a **hybrid recommendation system** that leverages both:

- Movie content (genres)
- User behavior (ratings)

By combining these two approaches, the system generates more accurate and personalized movie recommendations.

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Cosine Similarity

---

# 🧠 Recommendation Approaches

## Content-Based Filtering

This method recommends movies based on **movie attributes** such as genres.

Steps:
- Extract movie genres
- Convert genres into numerical vectors using **TF-IDF**
- Compute similarity between movies using **Cosine Similarity**
- Recommend movies that are most similar to the selected movie

---

## Collaborative Filtering

This method recommends movies based on **user ratings and preferences**.

Steps:
- Analyze user ratings
- Identify patterns in user preferences
- Suggest movies liked by users with similar tastes

---

## Hybrid Recommendation

The final recommendation combines both methods:

Hybrid Recommendation = Content-Based + Collaborative Filtering

Benefits:
- Better recommendation accuracy
- Reduced cold-start problem
- More personalized suggestions

---

# 📊 Dataset

This project uses the **MovieLens dataset**, which contains:

- Movie titles
- Movie genres
- User ratings

Main dataset files:

```
movies_clean.csv
ratings_clean.csv
```

---

# 🖥️ Application Interface

The application is built with **Streamlit** and allows users to:

- Select a user ID
- Select a reference movie
- Choose the number of recommendations
- Generate personalized movie recommendations

---

# 📂 Project Structure

```
movie-recommender
│
├── app
│   └── app.py
│
├── src
│   └── hybrid.py
│
├── data
│   ├── movies_clean.csv
│   └── ratings_clean.csv
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Ifraj-ouni/movie-recommender.git
cd movie-recommender
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

# 🌐 Deployment

The application is deployed using **Streamlit Cloud**.

Steps:
1. Push the project to GitHub
2. Connect the repository to Streamlit Cloud
3. Select `app/app.py` as the main file

# 🔮 Future Improvements

This is the **first version (V0)** of the project and several improvements are planned:

- Add movie posters
- Improve recommendation accuracy
- Implement advanced recommendation models
- Improve the user interface
- Use a larger dataset
- Add user accounts and personalization

GitHub: https://github.com/Ifraj-ouni
LinkedIn: https://www.linkedin.com
