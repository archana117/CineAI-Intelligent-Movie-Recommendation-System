import os
import io
from flask import Flask, render_template, request, jsonify, make_response
from src.recommender import MovieRecommender
from src.utils import get_pretty_language, normalize_language, normalize_genre

app = Flask(__name__)

# Dataset Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "movies_metadata.csv")

# Load Dataset and Recommender
try:
    recommender = MovieRecommender(DATASET_PATH)
    recommender.load_data()
    recommender.preprocess_data()
    print("[OK] Dataset loaded successfully")
except Exception as e:
    print("[ERROR] Dataset Loading Error:", e)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# JSON Recommendation API
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        data = request.get_json() or {}
        language_input = data.get("language", "").strip()
        genre_input = data.get("genre", "").strip()
        method = data.get("method", "strict").strip()
        query_title = data.get("query_title", "").strip()
        top_n = int(data.get("top_n", 10))

        # Normalize inputs
        norm_lang = normalize_language(language_input)
        norm_genre = normalize_genre(genre_input)

        if not norm_lang or not norm_genre:
            return jsonify({
                "success": False,
                "error": f"Invalid language ('{language_input}') or genre ('{genre_input}')."
            }), 400

        # Execute recommender
        recommendations = recommender.recommend_movies(
            language_code=norm_lang,
            genre_name=norm_genre,
            top_n=top_n,
            method=method,
            query_title=query_title if method == 'content' else None
        )

        if recommendations is None or recommendations.empty:
            return jsonify({
                "success": True,
                "movies": [],
                "message": "No movies found matching the selected filters."
            })

        # Format output records
        movies_list = []
        for _, row in recommendations.iterrows():
            year = ""
            if 'release_date' in row and row['release_date'] and len(row['release_date']) >= 4:
                year = row['release_date'][:4]
                
            movies_list.append({
                "id": int(row['id']),
                "title": row['title'],
                "genres": row['genres_display'],
                "language": get_pretty_language(row['original_language']),
                "vote_average": float(row['vote_average']),
                "vote_count": int(row['vote_count']),
                "popularity": float(row['popularity']),
                "overview": row['overview'],
                "poster_path": row.get('poster_path', ''),
                "release_year": year,
                "tagline": row.get('tagline', ''),
                "similarity_score": float(row.get('similarity_score', 0.0)) if 'similarity_score' in row else None
            })

        return jsonify({
            "success": True,
            "movies": movies_list,
            "language_display": get_pretty_language(norm_lang),
            "genre_display": norm_genre
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Autocomplete Title Search API
@app.route("/api/autocomplete", methods=["GET"])
def api_autocomplete():
    query = request.args.get("query", "").strip().lower()
    if not query or len(query) < 2:
        return jsonify([])
    
    if recommender.processed_df is None:
        return jsonify([])
    
    # Find matching movies by title (substring search)
    matches = recommender.processed_df[
        recommender.processed_df['title'].str.lower().str.contains(query, na=False)
    ]
    
    # Sort matches by popularity to show most relevant movies first
    matches = matches.sort_values(by='popularity', ascending=False).head(10)
    
    results = []
    for _, row in matches.iterrows():
        year = ""
        if 'release_date' in row and row['release_date'] and len(row['release_date']) >= 4:
            year = f" ({row['release_date'][:4]})"
        
        results.append({
            "title": row['title'],
            "year": year,
            "id": int(row['id']),
            "language": row['original_language']
        })
        
    return jsonify(results)

# Export Recommendations CSV API
@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    try:
        language_input = request.args.get("language", "").strip()
        genre_input = request.args.get("genre", "").strip()
        method = request.args.get("method", "strict").strip()
        query_title = request.args.get("query_title", "").strip()

        norm_lang = normalize_language(language_input)
        norm_genre = normalize_genre(genre_input)

        if not norm_lang or not norm_genre:
            return "Invalid language or genre", 400

        recommendations = recommender.recommend_movies(
            language_code=norm_lang,
            genre_name=norm_genre,
            top_n=10,
            method=method,
            query_title=query_title if method == 'content' else None
        )

        if recommendations is None or recommendations.empty:
            return "No recommendations found to export", 404

        # Generate CSV in memory
        export_df = recommendations.copy()
        export_df['language_display'] = export_df['original_language'].apply(get_pretty_language)
        
        columns_to_save = ['title', 'vote_average', 'vote_count', 'popularity', 'genres_display', 'language_display']
        if 'similarity_score' in export_df:
            columns_to_save.append('similarity_score')
            
        save_df = export_df[columns_to_save].rename(columns={
            'title': 'Movie Name',
            'vote_average': 'Rating',
            'vote_count': 'Vote Count',
            'popularity': 'Popularity',
            'genres_display': 'Genres',
            'language_display': 'Language',
            'similarity_score': 'Similarity Score'
        })

        csv_buffer = io.StringIO()
        save_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Create download response
        pretty_lang = get_pretty_language(norm_lang)
        filename = f"recommendations_{pretty_lang.lower()}_{norm_genre.lower().replace(' ', '_')}.csv"
        
        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/csv"
        return response

    except Exception as e:
        return f"Error exporting CSV: {str(e)}", 500

# Legacy HTML Post Recommendation Page
@app.route("/recommend", methods=["POST"])
def legacy_recommend():
    try:
        language = request.form.get("language")
        genre = request.form.get("genre")
        method = request.form.get("method", "strict")
        query_title = request.form.get("query_title", "")

        norm_lang = normalize_language(language)
        norm_genre = normalize_genre(genre)

        recommendations = recommender.recommend_movies(
            language_code=norm_lang,
            genre_name=norm_genre,
            top_n=10,
            method=method,
            query_title=query_title if method == "content" else None
        )

        if recommendations is None or recommendations.empty:
            return render_template(
                "results.html",
                movies=[],
                language=language,
                genre=genre,
                message="No movies found for the selected language and genre."
            )

        movies = recommendations.to_dict("records")
        for m in movies:
            if 'release_date' in m and m['release_date'] and len(m['release_date']) >= 4:
                m['release_year'] = m['release_date'][:4]
            else:
                m['release_year'] = ''
            m['genres'] = m['genres_display']
            m['language'] = get_pretty_language(m['original_language'])

        return render_template(
            "results.html",
            movies=movies,
            language=language,
            genre=genre,
            message=None
        )

    except Exception as e:
        return f"<h2>Application Error</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    print("Starting Flask...")
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)
