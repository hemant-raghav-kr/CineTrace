"""
CineTrace - Language Personalisation & Recommendation Service
Inspired by streaming platforms such as Hotstar.
Handles language metadata parsing, classification, multi-language matching,
and fair-mixture recommendation personalisation across Home and Discover.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

# Exactly the six supported language options
SUPPORTED_LANGUAGES: List[str] = [
    "English",
    "Tamil",
    "Telugu",
    "Hindi",
    "Malayalam",
    "Others",
]

# Set of canonical named languages (used to define and identify "Others")
NAMED_LANGUAGES: Set[str] = {
    "English",
    "Tamil",
    "Telugu",
    "Hindi",
    "Malayalam",
}

# Rich UI presentation metadata for onboarding and preferences dialogs
LANGUAGE_METADATA: Dict[str, Dict[str, str]] = {
    "English": {
        "native": "English",
        "subtitle": "Hollywood & Global Hits",
        "icon": "🌐",
    },
    "Tamil": {
        "native": "தமிழ்",
        "subtitle": "Kollywood Classics",
        "icon": "🎭",
    },
    "Telugu": {
        "native": "తెలుగు",
        "subtitle": "Tollywood Blockbusters",
        "icon": "⚡",
    },
    "Hindi": {
        "native": "हिन्दी",
        "subtitle": "Bollywood & Cinema",
        "icon": "✨",
    },
    "Malayalam": {
        "native": "മലയാളം",
        "subtitle": "Mollywood Masterpieces",
        "icon": "🌿",
    },
    "Others": {
        "native": "World",
        "subtitle": "International Cinema",
        "icon": "🎬",
    },
}


def parse_movie_languages(lang_raw: Any) -> List[str]:
    """
    Parse a comma-separated language string into a clean list of capitalized language names.
    Returns [] for missing, None, empty, or 'N/A' languages, indicating Unknown status.
    """
    if not lang_raw or not isinstance(lang_raw, str):
        return []
    s = lang_raw.strip()
    if not s or s.upper() == "N/A":
        return []
    tokens = [t.strip().capitalize() for t in s.split(",") if t.strip()]
    return [t for t in tokens if t.upper() != "N/A"]


def is_language_unknown(movie: Dict[str, Any]) -> bool:
    """
    Return True if movie has no verified language metadata.
    Unknown languages must NEVER automatically be classified as 'Others'.
    """
    langs = parse_movie_languages(movie.get("language") or movie.get("Language"))
    return len(langs) == 0


def movie_matches_language(movie: Dict[str, Any], target_language: str) -> bool:
    """
    Check if a movie matches a specific language preference.

    Matching Rules:
    1. If movie language is Unknown/missing, it NEVER matches (returns False).
    2. If target_language is one of the five named languages (English, Tamil, Telugu, Hindi, Malayalam):
       Matches if ANY of the movie's verified languages equals target_language (case-insensitive).
    3. If target_language == 'Others':
       Matches if and only if at least one verified language is outside the five named languages
       (e.g., Japanese, Korean, French, German, Spanish, Mandarin).
    4. For movies with multiple verified languages (e.g. 'Hindi, English' or 'Telugu, Tamil'):
       The movie matches each of its constituent languages individually when that preference is selected.
    """
    langs = parse_movie_languages(movie.get("language") or movie.get("Language"))
    if not langs:
        return False

    t_clean = target_language.strip().capitalize()
    if t_clean in NAMED_LANGUAGES:
        return any(l.lower() == t_clean.lower() for l in langs)
    elif t_clean == "Others":
        # Matches if the movie's primary language is outside NAMED_LANGUAGES,
        # or if none of its languages belong to NAMED_LANGUAGES.
        # This properly classifies World Cinema (e.g. Spirited Away [Japanese, English],
        # Parasite [Korean, English]) as 'Others' while preventing primarily English
        # films with incidental foreign dialogue (e.g. The Dark Knight [English, Mandarin])
        # from being misclassified as Others.
        primary_lang = langs[0].capitalize()
        if primary_lang not in NAMED_LANGUAGES:
            return True
        return all(l.capitalize() not in NAMED_LANGUAGES for l in langs)
    return False


def movie_matches_preferences(movie: Dict[str, Any], selected_languages: List[str]) -> bool:
    """
    Check if a movie matches any of the user's selected language preferences.
    If no languages are selected (e.g. default / skipped onboarding), returns True.
    """
    if not selected_languages:
        return True
    return any(movie_matches_language(movie, lang) for lang in selected_languages)


def personalize_recommendations(
    featured: Optional[Dict[str, Any]],
    picks: List[Dict[str, Any]],
    selected_languages: List[str],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Personalise the featured hero film and curated picks based on selected language preferences.

    Behaviour:
    - If selected_languages is empty: returns original (featured, picks) unaltered.
    - If English is in selected_languages and featured is an English classic (e.g. Inception):
      preserves Inception as the hero banner.
    - If English is NOT selected: selects the highest-rated movie from the user's preferred
      languages as the featured hero card.
    - Popular Picks grid: fair-mixture round-robin interleaving across selected languages.
      Prevents any single language from monopolizing the screen while ensuring available
      films from each selected language are presented without empty cards.
    - Deduplicates strictly by IMDb ID so the featured movie does not reappear in the grid.
    """
    if not selected_languages:
        return featured, list(picks)

    # 1. Collect all available curated movies without duplicates
    all_movies: List[Dict[str, Any]] = []
    seen_all_ids: Set[str] = set()

    for m in ([featured] if featured else []) + list(picks):
        if not isinstance(m, dict):
            continue
        mid = (m.get("imdb_id") or m.get("imdbID") or "").strip()
        if mid and mid not in seen_all_ids:
            seen_all_ids.add(mid)
            all_movies.append(m)

    # 2. Filter matching movies
    matching_movies = [m for m in all_movies if movie_matches_preferences(m, selected_languages)]
    if not matching_movies:
        # Graceful fallback: preserve original if no movies match
        return featured, list(picks)

    # 3. Determine Personalized Featured Movie
    def _rating_val(m: Dict[str, Any]) -> float:
        try:
            return float(m.get("imdb_rating") or m.get("imdbRating") or 0.0)
        except Exception:
            return 0.0

    chosen_featured: Optional[Dict[str, Any]] = None

    # If English is selected and default featured is English, keep it
    if "English" in selected_languages and featured and movie_matches_language(featured, "English"):
        chosen_featured = featured
    else:
        # Find highest-rated matching movie with valid poster
        candidates_with_poster = [
            m for m in matching_movies
            if str(m.get("poster_url") or m.get("poster") or "N/A").strip().upper() != "N/A"
        ]
        pool_for_hero = candidates_with_poster if candidates_with_poster else matching_movies
        sorted_hero_candidates = sorted(pool_for_hero, key=_rating_val, reverse=True)
        chosen_featured = sorted_hero_candidates[0] if sorted_hero_candidates else matching_movies[0]

    chosen_hero_id = (chosen_featured.get("imdb_id") or chosen_featured.get("imdbID") or "").strip() if chosen_featured else ""

    # 4. Candidate pool for picks (exclude chosen hero)
    pool_for_picks = [
        m for m in matching_movies
        if (m.get("imdb_id") or m.get("imdbID") or "").strip() != chosen_hero_id
    ]

    # 5. Fair Round-Robin Interleaving across selected languages
    # Create language buckets
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for lang in selected_languages:
        # Sort each bucket by rating descending so top titles appear first in rotation
        lang_movies = [m for m in pool_for_picks if movie_matches_language(m, lang)]
        buckets[lang] = sorted(lang_movies, key=_rating_val, reverse=True)

    result_picks: List[Dict[str, Any]] = []
    seen_pick_ids: Set[str] = set()

    active = True
    while active:
        any_added = False
        for lang in selected_languages:
            b = buckets[lang]
            while b:
                candidate = b.pop(0)
                cid = (candidate.get("imdb_id") or candidate.get("imdbID") or "").strip()
                if cid and cid not in seen_pick_ids:
                    seen_pick_ids.add(cid)
                    result_picks.append(candidate)
                    any_added = True
                    break  # Move to next language in rotation
        if not any_added:
            active = False

    return chosen_featured, result_picks
