"""
CineTrace - Curated Discovery & Popular Picks Service
Manages verified, iconic cinematic classics across genres and eras for the
Home and Discover screens. Retrieves and verifies real metadata and posters
via OMDb API and local SQLite cache, with persistent JSON caching to ensure
fast startup, zero GUI latency, and respect for API limits.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omdb_client import OMDbClient, OMDbError
import config
import database

CACHE_FILE = Path(__file__).resolve().parent / ".omdb_cache.json"
CACHE_DURATION_SECONDS = 86400 * 7  # 7 days cache validity

# Configurable list of universally recognized, iconic films across genres and eras (1942 - 2021)
CURATED_POPULAR_PICKS = [
    # Featured Hero Film
    {"imdb_id": "tt1375666", "title": "Inception"},

    # International Cinema Classics
    {"imdb_id": "tt0468569", "title": "The Dark Knight"},
    {"imdb_id": "tt0816692", "title": "Interstellar"},
    {"imdb_id": "tt0111161", "title": "The Shawshank Redemption"},
    {"imdb_id": "tt0110912", "title": "Pulp Fiction"},
    {"imdb_id": "tt0245429", "title": "Spirited Away"},
    {"imdb_id": "tt6751668", "title": "Parasite"},
    {"imdb_id": "tt0133093", "title": "The Matrix"},
    {"imdb_id": "tt0172495", "title": "Gladiator"},
    {"imdb_id": "tt0107290", "title": "Jurassic Park"},
    {"imdb_id": "tt0088763", "title": "Back to the Future"},
    {"imdb_id": "tt4633694", "title": "Spider-Man: Into the Spider-Verse"},
    {"imdb_id": "tt0034583", "title": "Casablanca"},
    {"imdb_id": "tt2582802", "title": "Whiplash"},
    {"imdb_id": "tt0078748", "title": "Alien"},
    {"imdb_id": "tt0109830", "title": "Forrest Gump"},

    # Indian Cinema — Hindi
    {"imdb_id": "tt1187043", "title": "3 Idiots"},
    {"imdb_id": "tt5074352", "title": "Dangal"},
    {"imdb_id": "tt1562872", "title": "Zindagi Na Milegi Dobara"},
    {"imdb_id": "tt8108198", "title": "Andhadhun"},
    {"imdb_id": "tt0169102", "title": "Lagaan: Once Upon a Time in India"},

    # Indian Cinema — Tamil
    {"imdb_id": "tt0367495", "title": "Anbe Sivam"},
    {"imdb_id": "tt15097216", "title": "Jai Bhim"},
    {"imdb_id": "tt6148156", "title": "Vikram Vedha"},
    {"imdb_id": "tt10189514", "title": "Soorarai Pottru"},
    {"imdb_id": "tt7019942", "title": "Super Deluxe"},

    # Indian Cinema — Malayalam
    {"imdb_id": "tt3417422", "title": "Drishyam"},
    {"imdb_id": "tt0214915", "title": "Manichitrathazhu"},
    {"imdb_id": "tt8413338", "title": "Kumbalangi Nights"},
    {"imdb_id": "tt3668162", "title": "Bangalore Days"},
    {"imdb_id": "tt4679210", "title": "Premam"},

    # Indian Cinema — Telugu
    {"imdb_id": "tt0249795", "title": "Mayabazar"},
    {"imdb_id": "tt7391996", "title": "C/o Kancharapalem"},
    {"imdb_id": "tt20850406", "title": "Sita Ramam"},
    {"imdb_id": "tt4849438", "title": "Baahubali 2: The Conclusion"},
    {"imdb_id": "tt8178634", "title": "RRR"},
]


def _is_valid_movie(m: Any) -> bool:
    """Validate that movie dictionary has required fields and non-empty values."""
    if not isinstance(m, dict):
        return False
    title = (m.get("title") or m.get("Title") or "").strip()
    imdb_id = (m.get("imdb_id") or m.get("imdbID") or "").strip()
    return bool(title and imdb_id)


def _normalize_movie(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure both 'poster' and 'poster_url' fields are populated if either exists."""
    if not isinstance(m, dict):
        return m
    d = dict(m)
    p = d.get("poster") or d.get("poster_url") or d.get("Poster")
    if p and str(p).strip().upper() != "N/A":
        clean_p = str(p).strip()
        d["poster"] = clean_p
        d["poster_url"] = clean_p
    return d


class DiscoveryService:
    """Service for managing curated, verified popular picks for Home and Discover."""

    def __init__(self, client: Optional[OMDbClient] = None):
        self.client = client or OMDbClient()

    def get_cached_home_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve cached curated data if valid and intact."""
        if not CACHE_FILE.exists():
            return None
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = data.get("timestamp", 0)
            featured = _normalize_movie(data.get("featured"))
            raw_picks = data.get("picks") or data.get("trending", [])
            picks = [_normalize_movie(p) for p in raw_picks] if isinstance(raw_picks, list) else []

            # Check validity: must have featured movie and at least 3 verified picks
            if _is_valid_movie(featured) and len(picks) >= 3:
                # Cache validity window check
                if time.time() - cached_time < CACHE_DURATION_SECONDS:
                    return {
                        "timestamp": cached_time,
                        "featured": featured,
                        "picks": picks,
                        "trending": picks,  # alias for backwards compatibility
                    }
        except Exception:
            pass
        return None

    def fetch_home_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve curated Popular Picks for Home and Discover.
        1. Check and return local JSON cache if fresh.
        2. Verify each curated film: check SQLite first, then OMDb API.
        3. Persist verified movies to SQLite.
        4. Display only successfully verified movies, strictly deduplicated by IMDb ID.
        5. Maintain consistent, intentional curated ordering.
        6. Cache result locally to protect API limits and support offline launches.
        """
        if not force_refresh:
            cached = self.get_cached_home_data()
            if cached:
                return cached

        verified_movies: List[Dict[str, Any]] = []
        seen_ids = set()

        for item in CURATED_POPULAR_PICKS:
            imdb_id = item["imdb_id"]
            if imdb_id in seen_ids:
                continue

            movie_data = None

            # Step A: Check local SQLite database cache first
            try:
                db_movie = database.get_movie_by_id(imdb_id)
                if db_movie and db_movie.get("title") and db_movie.get("year"):
                    movie_data = _normalize_movie(db_movie)
            except Exception:
                pass

            # Step B: If not in SQLite or incomplete, fetch via OMDb API
            if not movie_data and config.is_api_key_configured():
                try:
                    detail = self.client.get_movie_details(imdb_id)
                    if _is_valid_movie(detail):
                        movie_data = _normalize_movie(detail)
                        try:
                            database.save_movie(detail)
                        except Exception:
                            pass
                except Exception:
                    # Individual lookup failure (rate limit, timeout, or network blip)
                    pass

            if movie_data and _is_valid_movie(movie_data):
                seen_ids.add(imdb_id)
                verified_movies.append(movie_data)

        # If lookups failed (e.g. offline with empty cache), attempt reading any previous cache file
        if not verified_movies and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                if _is_valid_movie(old_data.get("featured")):
                    return old_data
            except Exception:
                pass

        if not verified_movies:
            if not config.is_api_key_configured():
                raise OMDbError(
                    "OMDb API key is not configured. Please enter your API key to load Popular Picks."
                )
            raise OMDbError(
                "Unable to retrieve Popular Picks from OMDb at this time. Please check your network connection."
            )

        featured_movie = verified_movies[0]
        picks = verified_movies[1:] if len(verified_movies) > 1 else []

        payload = {
            "timestamp": time.time(),
            "featured": featured_movie,
            "picks": picks,
            "trending": picks,  # alias for backwards compatibility
        }

        # Save to cache file safely
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

        return payload
