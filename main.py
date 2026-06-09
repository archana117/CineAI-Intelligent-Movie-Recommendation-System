import os
import sys
from src.recommender import MovieRecommender
from src.utils import Timer, validate_inputs, get_pretty_language, normalize_language, normalize_genre

def print_header(title: str):
    print("=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def print_footer():
    print("=" * 60)

def main():
    # Setup dataset path (relative to workspace root)
    dataset_path = "movies_metadata.csv"
    
    # Header block
    print_header("AI-powered Movie Recommendation System")
    print("Loading and preprocessing TMDB dataset. Please wait...\n")

    # Initialize recommender
    recommender = MovieRecommender(dataset_path)

    # 1. Ingest Data
    try:
        with Timer() as load_timer:
            recommender.load_data()
        print(f"[OK] Raw dataset loaded in {load_timer.get_formatted_duration()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure 'movies_metadata.csv' is placed in the project root directory.")
        sys.exit(1)

    # 2. Preprocess Data
    try:
        with Timer() as prep_timer:
            recommender.preprocess_data()
        total_movies = len(recommender.processed_df)
        print(f"[OK] Data preprocessed and cleaned in {prep_timer.get_formatted_duration()}")
        print(f"[OK] Total clean unique movies indexed: {total_movies}\n")
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        sys.exit(1)

    # Main interaction loop
    while True:
        print_header("Get Recommendations")
        
        # User input for language
        lang_input = input("Enter Language (e.g. English, Hindi, Telugu): ").strip()
        # User input for genre
        genre_input = input("Enter Genre (e.g. Action, Comedy, Drama, Sci-Fi): ").strip()

        # Validate inputs
        norm_lang, norm_genre, errors = validate_inputs(lang_input, genre_input)
        
        if errors:
            print("\n" + "!" * 40)
            for err in errors:
                print(err)
            print("!" * 40 + "\n")
            continue
        
        # Display candidates found matching criteria
        candidates = recommender.filter_movies(norm_lang, norm_genre)
        total_found = len(candidates)
        print(f"\n[Info] Total movies found in category '{get_pretty_language(norm_lang)} {norm_genre}': {total_found}")

        if total_found == 0:
            print("No movies match these filters in our dataset. Please try a different combination.\n")
            continue

        # Choose recommendation mode
        print("\nChoose recommendation algorithm:")
        print("1. Rating & Popularity Sort (Strict Filter - Default)")
        print("2. Content Similarity (AI-powered TF-IDF search based on a reference movie)")
        choice = input("Enter choice (1 or 2): ").strip()

        method = 'strict'
        query_title = None
        
        if choice == '2':
            method = 'content'
            query_title = input("Enter a reference movie title you like (e.g. Toy Story, Baahubali): ").strip()

        # Execute recommendation engine
        print("\nGenerating recommendations...")
        with Timer() as rec_timer:
            recommendations = recommender.recommend_movies(
                language_code=norm_lang,
                genre_name=norm_genre,
                top_n=10,
                method=method,
                query_title=query_title
            )

        # Output results
        pretty_lang_name = get_pretty_language(norm_lang)
        recommender.display_results(recommendations, pretty_lang_name, norm_genre)
        print(f"Execution time for recommendation: {rec_timer.get_formatted_duration()}\n")

        # Save to CSV
        if not recommendations.empty:
            save_name = f"recommendations_{pretty_lang_name.lower()}_{norm_genre.lower().replace(' ', '_')}.csv"
            output_path = os.path.join("sample_outputs", save_name)
            
            try:
                saved_file = recommender.save_recommendations(recommendations, output_path)
                if saved_file:
                    print(f"[OK] Recommendations saved successfully to: {os.path.abspath(saved_file)}")
            except Exception as e:
                print(f"Error saving recommendations to CSV: {e}")

        # Continue or Exit
        print_footer()
        again = input("Do you want another recommendation? (yes/no): ").strip().lower()
        if again not in ['y', 'yes']:
            print("\nThank you for using the Movie Recommendation System! Goodbye.")
            break
        print("\n")

if __name__ == "__main__":
    main()
