import time
from typing import Tuple, List, Optional

# Predefined language name to ISO 639-1 code mappings
LANGUAGE_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi",
    "telugu": "te", "te": "te",
    "tamil": "ta", "ta": "ta",
    "malayalam": "ml", "ml": "ml",
    "kannada": "kn", "kn": "kn",
    "bengali": "bn", "bn": "bn",
    "spanish": "es", "es": "es",
    "french": "fr", "fr": "fr",
    "japanese": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "chinese": "zh", "zh": "zh", "cn": "cn",
    "german": "de", "de": "de",
    "italian": "it", "it": "it",
    "russian": "ru", "ru": "ru",
    "portuguese": "pt", "pt": "pt",
}

# Reverse mapping for displaying user-friendly language names
REVERSE_LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "bn": "Bengali",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "cn": "Chinese",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "pt": "Portuguese"
}

# Mapping user input genres to standard dataset genre titles
GENRE_MAP = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "family": "Family",
    "fantasy": "Fantasy",
    "foreign": "Foreign",
    "history": "History",
    "horror": "Horror",
    "music": "Music",
    "mystery": "Mystery",
    "romance": "Romance",
    "science fiction": "Science Fiction",
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "tv movie": "TV Movie",
    "tv": "TV Movie",
    "thriller": "Thriller",
    "war": "War",
    "western": "Western"
}

class Timer:
    """Context manager to measure and format execution execution time."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

    def get_formatted_duration(self) -> str:
        """Return execution duration formatted to 4 decimal places."""
        return f"{self.duration:.4f} seconds"

def normalize_language(lang_input: str) -> Optional[str]:
    """Normalize input language string to ISO 639-1 code."""
    if not lang_input or not isinstance(lang_input, str):
        return None
    lang_clean = lang_input.strip().lower()
    if lang_clean in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang_clean]
    # Fallback to the code itself if it is two characters
    if len(lang_clean) == 2:
        return lang_clean
    return None

def normalize_genre(genre_input: str) -> Optional[str]:
    """Normalize input genre string to match dataset genre names."""
    if not genre_input or not isinstance(genre_input, str):
        return None
    genre_clean = genre_input.strip().lower()
    return GENRE_MAP.get(genre_clean, None)

def get_pretty_language(lang_code: str) -> str:
    """Get the capitalized display name of a language code, fallback to uppercase code."""
    if not lang_code:
        return "Unknown"
    return REVERSE_LANGUAGE_MAP.get(lang_code.lower(), lang_code.upper())

def validate_inputs(lang_input: str, genre_input: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Validate and normalize language and genre inputs.
    Returns:
        tuple: (normalized_lang_code, normalized_genre_name, list_of_error_messages)
    """
    errors = []
    
    norm_lang = normalize_language(lang_input)
    if not norm_lang:
        supported_langs = sorted(list(set(REVERSE_LANGUAGE_MAP.values())))
        errors.append(
            f"Error: '{lang_input}' is not a supported language.\n"
            f"Supported languages include: {', '.join(supported_langs)}"
        )
        
    norm_genre = normalize_genre(genre_input)
    if not norm_genre:
        supported_genres = sorted(list(set(GENRE_MAP.values())))
        errors.append(
            f"Error: '{genre_input}' is not a supported genre.\n"
            f"Supported genres include: {', '.join(supported_genres)}"
        )
        
    return norm_lang, norm_genre, errors
