import os
import ast
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_pretty_language

class MovieRecommender:
    def __init__(self, filepath: str):
        """
        Initialize the Movie Recommender.
        
        Args:
            filepath (str): Absolute or relative path to the movies_metadata.csv file.
        """
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        
        # Scikit-learn attributes for content-based filtering
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the movies metadata CSV dataset.
        Checks multiple fallback paths if the primary path is not found.
        
        Returns:
            pd.DataFrame: Raw loaded pandas DataFrame.
        """
        resolved_path = self.filepath
        
        # If the direct filepath doesn't exist, search common locations
        if not os.path.exists(resolved_path):
            filename = os.path.basename(self.filepath)
            parent_dir = os.path.dirname(self.filepath)
            
            fallbacks = [
                os.path.join(parent_dir, "static", filename) if parent_dir else None,
                os.path.join("static", filename),
                os.path.join("backend", "static", filename),
                os.path.join("..", "backend", "static", filename),
                filename
            ]
            
            found = False
            for fb in fallbacks:
                if fb and os.path.exists(fb):
                    resolved_path = fb
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"Dataset file '{filename}' not found at: '{self.filepath}' "
                    f"or any standard fallbacks (e.g. 'static/{filename}')."
                )

        print(f"[INFO] Loading dataset from: {os.path.abspath(resolved_path)}")
        # Read the CSV file. Use low_memory=False to avoid DtypeWarnings with mixed types
        self.df = pd.read_csv(resolved_path, low_memory=False)
        return self.df


    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean, preprocess, and normalize the loaded dataset.
        Handles missing values, parses JSON genre lists, casts types, and removes duplicates.
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed pandas DataFrame.
        """
        if self.df is None:
            raise ValueError("Dataset has not been loaded. Please call load_data() first.")

        df_clean = self.df.copy()

        # 1. Clean corrupt rows by converting IDs and ratings to numeric and dropping NaNs
        df_clean['id'] = pd.to_numeric(df_clean['id'], errors='coerce')
        df_clean['vote_average'] = pd.to_numeric(df_clean['vote_average'], errors='coerce')
        df_clean['vote_count'] = pd.to_numeric(df_clean['vote_count'], errors='coerce')
        df_clean['popularity'] = pd.to_numeric(df_clean['popularity'], errors='coerce')

        # Drop rows missing crucial values
        df_clean = df_clean.dropna(subset=['id', 'title', 'vote_average', 'vote_count', 'popularity'])

        # Cast types
        df_clean['id'] = df_clean['id'].astype(int)
        df_clean['vote_count'] = df_clean['vote_count'].astype(int)
        df_clean['vote_average'] = df_clean['vote_average'].astype(float)
        df_clean['popularity'] = df_clean['popularity'].astype(float)
        df_clean['title'] = df_clean['title'].astype(str).str.strip()
        df_clean['original_language'] = df_clean['original_language'].fillna('unknown').astype(str).str.strip().str.lower()
        df_clean['overview'] = df_clean['overview'].fillna('').astype(str).str.strip()
        df_clean['poster_path'] = df_clean['poster_path'].fillna('').astype(str).str.strip()
        df_clean['release_date'] = df_clean['release_date'].fillna('').astype(str).str.strip()
        df_clean['tagline'] = df_clean['tagline'].fillna('').astype(str).str.strip()

        # 2. Parse JSON-like genres string
        def parse_genres(genre_str: str) -> List[str]:
            if pd.isna(genre_str) or not isinstance(genre_str, str):
                return []
            try:
                # Safely parse python literal structure
                genres_list = ast.literal_eval(genre_str)
                if isinstance(genres_list, list):
                    return [g['name'] for g in genres_list if isinstance(g, dict) and 'name' in g]
            except Exception:
                pass
            return []

        df_clean['genres_list'] = df_clean['genres'].apply(parse_genres)
        df_clean['genres_display'] = df_clean['genres_list'].apply(lambda x: ", ".join(x))

        # 3. Remove duplicate movie entries by unique ID
        df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')

        # Keep only required columns to conserve memory
        columns_to_keep = [
            'id', 'title', 'genres_list', 'genres_display', 
            'original_language', 'vote_average', 'vote_count', 
            'popularity', 'overview', 'poster_path', 'release_date', 'tagline'
        ]
        self.processed_df = df_clean[columns_to_keep].reset_index(drop=True)
        
        # 4. Prepare TF-IDF matrix for Content-Based Filtering
        self._initialize_tfidf()
        
        return self.processed_df

    def _initialize_tfidf(self):
        """Prepare the TF-IDF representation of movie overviews and genres for AI search."""
        if self.processed_df is None or len(self.processed_df) == 0:
            return

        # Create a text soup: Title + Overview + Genres
        metadata_soup = (
            self.processed_df['title'] + " " + 
            self.processed_df['overview'] + " " + 
            self.processed_df['genres_display']
        )
        
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(metadata_soup)

    def filter_movies(self, language_code: str, genre_name: str) -> pd.DataFrame:
        """
        Filter movies strictly by language code and genre name.
        
        Args:
            language_code (str): ISO 639-1 language code (e.g. 'te', 'hi', 'en').
            genre_name (str): Standard genre name (e.g. 'Action', 'Comedy').
            
        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if self.processed_df is None:
            raise ValueError("Dataset has not been preprocessed. Please call preprocess_data() first.")

        # Strict language filter
        lang_filtered = self.processed_df[self.processed_df['original_language'] == language_code]

        # Strict genre filter
        genre_filtered = lang_filtered[lang_filtered['genres_list'].apply(lambda x: genre_name in x)]

        return genre_filtered

    def recommend_movies(
        self, 
        language_code: str, 
        genre_name: str, 
        top_n: int = 10,
        method: str = 'strict',
        query_title: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recommend top N movies based on input criteria.
        
        Args:
            language_code (str): ISO 639-1 language code.
            genre_name (str): Standard genre name.
            top_n (int): Number of recommendations to return (default 10).
            method (str): 'strict' (sort by rating & popularity) or 'content' (use TF-IDF & similarity).
            query_title (str): The reference movie title to find similarities for (used in 'content' mode).
            
        Returns:
            pd.DataFrame: Top N recommendations.
        """
        # Get candidates matching the filter
        candidates = self.filter_movies(language_code, genre_name)
        if len(candidates) == 0:
            return pd.DataFrame() # Return empty DataFrame if no candidates

        if method == 'strict' or not query_title:
            # Sort by vote_average (descending), then popularity (descending)
            recommended = candidates.sort_values(
                by=['vote_average', 'popularity'], 
                ascending=[False, False]
            )
            return recommended.head(top_n)
            
        elif method == 'content':
            # Content-based recommendations using TF-IDF & Cosine Similarity
            # Find the query movie in the dataset to get its vector
            query_matches = self.processed_df[self.processed_df['title'].str.lower() == query_title.strip().lower()]
            if len(query_matches) == 0:
                # If target movie is not found, fallback to strict rating sort and log a warning
                print(f"Warning: Movie '{query_title}' not found in database. Falling back to rating sort.")
                recommended = candidates.sort_values(by=['vote_average', 'popularity'], ascending=[False, False])
                return recommended.head(top_n)

            # Get the TF-IDF vector of the query movie (take the first match if multiple exist)
            query_idx = query_matches.index[0]
            query_vector = self.tfidf_matrix[query_idx]
            
            # Get vectors of the candidate movies
            candidate_indices = candidates.index
            candidate_vectors = self.tfidf_matrix[candidate_indices]
            
            # Compute cosine similarities between query movie and candidates
            similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
            
            # Assign similarity score to candidates
            candidates_with_score = candidates.copy()
            candidates_with_score['similarity_score'] = similarities
            
            # Filter out the query movie itself from recommendations, if it is in candidates
            candidates_with_score = candidates_with_score[candidates_with_score['id'] != query_matches.iloc[0]['id']]
            
            # Sort by similarity score (descending), then vote_average (descending)
            recommended = candidates_with_score.sort_values(
                by=['similarity_score', 'vote_average'],
                ascending=[False, False]
            )
            return recommended.head(top_n)
        
        else:
            raise ValueError(f"Unknown recommendation method: {method}")

    def display_results(self, recommended_df: pd.DataFrame, language_name: str, genre_name: str):
        """
        Display recommended movies in the formatted output required.
        
        Args:
            recommended_df (pd.DataFrame): DataFrame containing recommendations.
            language_name (str): Original language name entered by the user.
            genre_name (str): Original genre name entered by the user.
        """
        print(f"\nTop {len(recommended_df)} {language_name} {genre_name} Movies\n")
        
        if recommended_df.empty:
            print("No movies found matching the selected language and genre.")
            return

        for idx, (_, row) in enumerate(recommended_df.iterrows(), 1):
            pretty_lang = get_pretty_language(row['original_language'])
            # Extract standard display fields
            title = row['title']
            rating = row['vote_average']
            genres = row['genres_display']
            
            print(f"{idx}. {title}")
            print(f"   Rating: {rating:.1f}")
            print(f"   Genre: {genres}")
            print(f"   Language: {pretty_lang}\n")

    def save_recommendations(self, recommended_df: pd.DataFrame, output_filepath: str) -> str:
        """
        Save recommendations to a CSV file. Creates directory if needed.
        
        Args:
            recommended_df (pd.DataFrame): DataFrame to save.
            output_filepath (str): Target CSV file path.
            
        Returns:
            str: Path to the saved file.
        """
        if recommended_df.empty:
            return ""
            
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Select columns to export
        export_df = recommended_df.copy()
        export_df['language_display'] = export_df['original_language'].apply(get_pretty_language)
        
        columns_to_save = ['title', 'vote_average', 'vote_count', 'popularity', 'genres_display', 'language_display']
        # Map columns to user-friendly titles
        save_df = export_df[columns_to_save].rename(columns={
            'title': 'Movie Name',
            'vote_average': 'Rating',
            'vote_count': 'Vote Count',
            'popularity': 'Popularity',
            'genres_display': 'Genres',
            'language_display': 'Language'
        })
        
        save_df.to_csv(output_filepath, index=False)
        return output_filepath
