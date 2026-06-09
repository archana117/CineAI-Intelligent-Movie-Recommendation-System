# CineAI-Intelligent-Movie-Recommendation-System

A Flask-based Movie Recommendation System that provides personalized movie suggestions based on language, genre, and content similarity. The system leverages machine learning techniques such as TF-IDF Vectorization and Cosine Similarity to deliver intelligent recommendations using TMDB movie metadata.

🚀 Features
🎭 Filter movies by Language and Genre
🤖 Content-Based Recommendation using TF-IDF and Cosine Similarity
🔍 Movie Title Autocomplete Search
📊 Ratings, Popularity, Vote Count, and Movie Overview Display
📁 Export Recommendations to CSV
🌐 REST API Support with JSON Responses
🖥️ Interactive Flask Web Interface
📈 Recommendation Ranking based on Ratings and Popularity
🛠️ Tech Stack
Python
Flask
Pandas
NumPy
Scikit-Learn
TF-IDF Vectorizer
Cosine Similarity
TMDB Movie Dataset

Dependencies are listed in the requirements file.

📂 Project Structure
Movie-Recommendation-System/
│
├── app.py                    # Flask Web Application
├── main.py                   # Command-Line Interface
├── movie.py                  # Genre Extraction Utility
├── requirements.txt
├── README.md
│
├── src/
│   ├── recommender.py
│   ├── utils.py
│   └── __init__.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   └── movies_metadata.csv
│
├── ratings_small.csv
├── credits.csv
├── keywords.csv
├── links.csv
└── links_small.csv
📊 Dataset

The project uses TMDB Movie Metadata Dataset containing:

Movie Titles
Genres
Language Information
Ratings & Vote Counts
Popularity Scores
Movie Overviews
Release Dates

Additional datasets:

credits.csv
keywords.csv
ratings_small.csv
links.csv
links_small.csv
