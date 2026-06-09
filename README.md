# AI-Powered Movie Recommendation System

A robust, production-ready, and object-oriented Movie Recommendation System implemented in Python. It ingests and cleans the Kaggle dataset **"The Movies Dataset"** (compiled by Rounak Banik) and recommends the top 10 movies based on user-selected criteria.

This system is engineered for developers seeking to demonstrate strong data engineering, machine learning preprocessing, and object-oriented programming (OOP) practices in an AIML portfolio.

---

## 🚀 Features

1. **Robust Data Cleaning & Ingestion**:
   - Safely parses Kaggle's JSON-like `genres` column string using Python's AST literal evaluation.
   - Cleans and casts numerical fields (`vote_average`, `vote_count`, `popularity`, `id`) while dropping corrupt or malformed rows.
   - Deduplicates movie entries by their unique database ID.

2. **Strict Metadata Filter (Demographic Recommendation)**:
   - Filters movies strictly by target language (e.g. Telugu, Hindi, English) and target genre.
   - Returns recommendations sorted by rating (`vote_average`) in descending order, with popularity (`popularity`) acting as a secondary tie-breaker.

3. **Advanced AI Content-Based Engine (Scikit-Learn Integration)**:
   - Uses **TF-IDF Vectorization** (`TfidfVectorizer`) to translate movie metadata (Title + Overview + Genres display text) into numeric representations.
   - Computes **Cosine Similarity** (`cosine_similarity`) between a reference movie title and the filtered subset of candidates to suggest contextually similar movies.

4. **Production Utilities**:
   - **Case-Insensitive Input Validation**: Correctly handles input variants like `"tElUgU"` or `"sci-fi"` and maps them to clean ISO codes (`te`) and standard genre strings (`Science Fiction`).
   - **Execution Ingestion & Search Timers**: Benchmarks the execution of loading, preprocessing, and finding recommendations.
   - **CSV Export**: Automatically exports recommended results into `sample_outputs/` with structured schemas.

---

## 📁 Directory Structure

```
movie_recomm/
├── src/
│   ├── __init__.py
│   ├── recommender.py      # Core OOP logic (MovieRecommender class)
│   └── utils.py            # Helpers for input validation, text formatting, and timers
├── main.py                 # Interactive command-line interface (CLI)
├── requirements.txt        # Python dependency list
├── README.md               # Project documentation (this file)
└── sample_outputs/         # Folder where generated CSV recommendations are stored
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+ (This project was validated on Python 3.13)
- Kaggle Dataset `movies_metadata.csv` downloaded and placed in the project root directory.

### Installation
1. Clone this repository or open the project folder in your shell.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

Start the recommendation system by running the main CLI:
```bash
python main.py
```

### Demonstration Flow:
1. **Startup**: The recommender loads the raw dataset, sanitizes rows, and builds the TF-IDF feature matrix.
2. **Inputs**: Enter your desired Language (e.g., `Telugu` or `te`) and Genre (e.g., `Action`).
3. **Select Mode**:
   - Choose `1` for standard rating sorting.
   - Choose `2` for Content similarity, then provide a reference movie title (e.g., `Baahubali: The Beginning`).
4. **Execution & Export**: View the Top 10 recommendations directly in the console (ratings, genre names, and language details). The recommendations are also exported as a CSV under `sample_outputs/`.

---

## 📊 Core Algorithms Explained

### 1. Strict Metadata Filter
```python
recommended = candidates.sort_values(
    by=['vote_average', 'popularity'], 
    ascending=[False, False]
).head(10)
```
Ties are common in average ratings (e.g., multiple movies scored 8.0). To ensure the highest quality suggestions, we break ties by using TMDB's `popularity` score.

### 2. Content-Based AI Engine
```
Metadata Text Soup = [Title] + [Overview Summary] + [Genre Text]
                   ↓
         TfidfVectorizer (Stop words removed)
                   ↓
         Cosine Similarity computation
                   ↓
         Sorted by Similarity Score (descending)
```
By vectorizing the text soup, we can measure the cosine angle between movies. If a user likes a specific movie, the system returns movies within the requested genre and language that share similar semantic profiles.

---

## 📈 Sample Output Format (Console)

```
============================================================
================== Get Recommendations ===================
============================================================
Enter Language (e.g. English, Hindi, Telugu): Telugu
Enter Genre (e.g. Action, Comedy, Drama, Sci-Fi): Action

→ Total movies found in category 'Telugu Action': 11

Choose recommendation algorithm:
1. Rating & Popularity Sort (Strict Filter - Default)
2. Content Similarity (AI-powered TF-IDF search based on a reference movie)
Enter choice (1 or 2): 1

Generating recommendations...

Top 10 Telugu Action Movies

1. Baahubali: The Beginning
   Rating: 7.5
   Genre: Action, Adventure, Drama, Fantasy
   Language: Telugu

2. Magadheera
   Rating: 7.5
   Genre: Action, Drama, Fantasy, Romance
   Language: Telugu

3. 1 - Nenokkadine
   Rating: 7.1
   Genre: Action, Comedy, Thriller
   Language: Telugu

...

Execution time for recommendation: 0.0012 seconds
✓ Recommendations saved successfully to: C:\...\sample_outputs\recommendations_telugu_action.csv
```

---

## 💻 Tech Stack
- **pandas**: Used for data manipulation, cleaning, and filtering.
- **numpy**: Used for optimized vector representations.
- **scikit-learn**: Powers the content similarity engine (`TfidfVectorizer`, `cosine_similarity`).
- **ast (Standard Library)**: Used for secure evaluation of list/dict literal structures within CSV fields.
