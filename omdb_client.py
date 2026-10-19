"""
CineTrace - OMDb API Client
Handles communication with the OMDb API for movie search and detail retrieval.
Crucially uses the 's' parameter for general title searches (partial matching)
and the 'i' parameter for detailed movie metadata by IMDb ID.
"""

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import requests
from config import get_api_key

OMDB_BASE_URL = "https://www.omdbapi.com/"
DEFAULT_TIMEOUT = 10  # seconds


class OMDbError(Exception):
    """Base exception for all OMDb client errors."""
    pass


class OMDbApiKeyMissingError(OMDbError):
    """Raised when no OMDb API key has been configured."""
    pass


class OMDbApiKeyInvalidError(OMDbError):
    """Raised when the configured OMDb API key is invalid."""
    pass


class OMDbMovieNotFoundError(OMDbError):
    """Raised when no movies match the search criteria."""
    pass


class OMDbConnectionError(OMDbError):
    """Raised when a network or timeout error occurs."""
    pass


def rank_search_results(movies: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Rank search results sensibly:
    1. Exact title matches (e.g. 'The Avengers', 'Avengers') -> Tier 1
    2. Franchise title prefixes (e.g. 'Avengers: Endgame', 'Avengers: Doomsday') -> Tier 2
    3. Sequel dash prefixes (e.g. 'Avengers - Infinity War') -> Tier 3
    4. Query prefix (e.g. 'Avengers Assemble') -> Tier 4
    5. Query whole word in title (e.g. 'Lego Marvel Avengers') -> Tier 5
    6. General substring match -> Tier 6
    Penalizes obvious non-feature entries (parodies, fan films, making-ofs, reviews).
    Prioritizes feature films ('movie') slightly over games/series.
    Secondary sort by release year (newer/canonical first) and original order.
    """
    if not movies:
        return []

    q_clean = query.strip().lower()
    q_words = [w for w in re.findall(r"\w+", q_clean) if len(w) > 1]

    def _score(item_tuple):
        idx, m = item_tuple
        title = (m.get("title") or "").strip()
        t_low = title.lower()
        m_type = (m.get("type") or "").lower()

        score = 0.0

        # Exact title match
        if t_low == q_clean or t_low == "the " + q_clean:
            score += 100.0
        # Franchise colon prefix: e.g. "Avengers: ..." or "The Hunger Games: ..."
        elif t_low.startswith(q_clean + ":") or t_low.startswith("the " + q_clean + ":"):
            score += 85.0
        # Franchise dash prefix: e.g. "Avengers - ..." or "The Hunger Games - ..."
        elif t_low.startswith(q_clean + " -") or t_low.startswith("the " + q_clean + " -"):
            score += 80.0
        # Query prefix: e.g. "Avengers Assemble"
        elif t_low.startswith(q_clean) or t_low.startswith("the " + q_clean):
            score += 65.0
        # Whole word match: e.g. "\bAvengers\b"
        elif re.search(r"\b" + re.escape(q_clean) + r"\b", t_low):
            score += 45.0
        elif q_words and all(re.search(r"\b" + re.escape(w) + r"\b", t_low) for w in q_words):
            score += 35.0
        else:
            score += 20.0

        # Penalty for parodies, fan films, behind the scenes, reviews
        low_quality = ("parody", "spoof", "fan film", "making of", "behind the", "demastered", "recreation", "review", "reaction")
        if any(k in t_low for k in low_quality):
            score -= 30.0

        # Feature films preference
        if m_type == "movie":
            score += 10.0

        # Extract 4-digit release year safely
        year_str = str(m.get("year") or "")
        clean_year = "".join(filter(str.isdigit, year_str[:4]))
        year_int = int(clean_year) if clean_year else 0

        # Secondary sort: score desc, year desc, original_index asc
        return (-score, -year_int, idx)

    sorted_items = sorted(enumerate(movies), key=_score)
    return [m for _, m in sorted_items]


class OMDbClient:
    """Client for querying the OMDb API."""

    def __init__(self, api_key: Optional[str] = None):
        self._custom_key = api_key

    def _resolve_key(self) -> str:
        """Retrieve API key securely, prioritizing custom key then config."""
        key = self._custom_key if self._custom_key else get_api_key()
        if not key:
            raise OMDbApiKeyMissingError(
                "OMDb API key is not configured. Please add your key to '.env' "
                "or enter it via the application settings."
            )
        return key

    def _fetch_single_page(
        self,
        cleaned_query: str,
        page: int = 1,
        movie_type: Optional[str] = None,
        year: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Fetch a single page (up to 10 items) directly from OMDb API."""
        api_key = self._resolve_key()

        params = {
            "apikey": api_key,
            "s": cleaned_query,
            "page": max(1, page),
        }
        if movie_type:
            params["type"] = movie_type
        if year is not None and str(year).strip():
            params["y"] = str(year).strip()

        try:
            response = requests.get(OMDB_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.Timeout:
            raise OMDbConnectionError("Request timed out while connecting to OMDb. Please check your internet connection.")
        except requests.exceptions.ConnectionError:
            raise OMDbConnectionError("Network connection failed. Unable to connect to OMDb. Please check your internet connection.")
        except requests.exceptions.RequestException:
            raise OMDbConnectionError("Network error occurred while contacting OMDb.")

        try:
            data = response.json()
        except Exception:
            raise OMDbError(f"Received invalid response format from OMDb (HTTP {response.status_code}).")

        # Check for API error response
        if data.get("Response") == "False":
            err_msg = data.get("Error", "Unknown OMDb error.")
            if "Invalid API key" in err_msg:
                raise OMDbApiKeyInvalidError("The provided OMDb API key is invalid. Please verify your API key.")
            elif "No API key" in err_msg:
                raise OMDbApiKeyMissingError("No OMDb API key provided. Please configure your API key.")
            elif "Movie not found" in err_msg:
                return {
                    "movies": [],
                    "total_results": 0,
                    "total_pages": 0,
                    "current_page": page,
                    "query": cleaned_query,
                    "message": f"No movies found matching '{cleaned_query}'."
                }
            elif "Too many results" in err_msg:
                raise OMDbError(f"Too many results found for '{cleaned_query}'. Please enter a more specific title.")
            elif "Request limit reached" in err_msg:
                raise OMDbError("OMDb daily request limit reached. Please try again later or use an upgraded key.")
            else:
                raise OMDbError(f"OMDb Error: {err_msg}")

        # Successfully received results
        raw_results = data.get("Search", [])
        try:
            total_results = int(data.get("totalResults", len(raw_results)))
        except (ValueError, TypeError):
            total_results = len(raw_results)

        total_pages = max(1, math.ceil(total_results / 10)) if total_results > 0 else 0

        movies = []
        for item in raw_results:
            movies.append({
                "title": item.get("Title", "Untitled"),
                "year": item.get("Year", "N/A"),
                "imdb_id": item.get("imdbID", ""),
                "type": item.get("Type", "movie"),
                "poster": item.get("Poster", "N/A"),
            })

        return {
            "movies": movies,
            "total_results": total_results,
            "total_pages": total_pages,
            "current_page": page,
            "query": cleaned_query,
            "message": f"Found {total_results} matching result{'s' if total_results != 1 else ''}."
        }

    def search_movies(
        self,
        query: str,
        page: int = 1,
        movie_type: Optional[str] = None,
        year: Optional[Any] = None,
        max_pages: int = 1
    ) -> Dict[str, Any]:
        """
        Search for movies using OMDb's 's' parameter.
        Supports single-page search (max_pages=1) or multi-page franchise search (max_pages > 1).
        When max_pages > 1:
            - Fetches multiple pages concurrently up to min(total_pages, max_pages)
            - Deduplicates strictly by IMDb ID
            - Preserves results if an additional page fails
            - Ranks results sensibly by franchise title relevance and canonical sequels
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Please enter a movie title to search.")

        # Single page requested: return directly without extra overhead
        if max_pages <= 1:
            single_page = self._fetch_single_page(cleaned_query, page=page, movie_type=movie_type, year=year)
            if single_page.get("movies"):
                single_page["movies"] = rank_search_results(single_page["movies"], cleaned_query)
            return single_page

        # Multi-page search: always start with page 1 to discover totalResults
        first_page = self._fetch_single_page(cleaned_query, page=1, movie_type=movie_type, year=year)
        total_pages = first_page.get("total_pages", 0)
        total_results = first_page.get("total_results", 0)
        first_movies = first_page.get("movies", [])

        if not first_movies or total_pages <= 1:
            if first_movies:
                first_page["movies"] = rank_search_results(first_movies, cleaned_query)
            return first_page

        pages_to_fetch = list(range(2, min(total_pages + 1, max_pages + 1)))
        additional_results: List[Tuple[int, List[Dict[str, Any]]]] = []

        if pages_to_fetch:
            workers = min(5, len(pages_to_fetch))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._fetch_single_page, cleaned_query, p, movie_type, year): p
                    for p in pages_to_fetch
                }
                for fut in as_completed(futures):
                    p_num = futures[fut]
                    try:
                        p_res = fut.result()
                        additional_results.append((p_num, p_res.get("movies", [])))
                    except Exception:
                        # Acceptance criteria: A failed additional page does not discard already retrieved results.
                        pass

        # Sort additional results in page order
        additional_results.sort(key=lambda x: x[0])

        # Combine all movies in initial retrieval order
        all_movies = list(first_movies)
        for _, m_list in additional_results:
            all_movies.extend(m_list)

        # Deduplicate strictly by IMDb ID
        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for m in all_movies:
            mid = (m.get("imdb_id") or "").strip().lower()
            if mid:
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    deduped.append(m)
            else:
                deduped.append(m)

        # Rank franchise-title matches sensibly
        ranked_movies = rank_search_results(deduped, cleaned_query)

        return {
            "movies": ranked_movies,
            "total_results": total_results,
            "total_pages": total_pages,
            "current_page": 1,
            "pages_fetched": 1 + len(additional_results),
            "query": cleaned_query,
            "message": f"Found {len(ranked_movies)} matching result{'s' if len(ranked_movies) != 1 else ''}."
        }

    def get_movie_details(self, imdb_id: str, plot: str = "full") -> Dict[str, Any]:
        """
        Retrieve comprehensive movie details using OMDb's 'i' parameter (IMDb ID).
        """
        cleaned_id = (imdb_id or "").strip()
        if not cleaned_id:
            raise ValueError("A valid IMDb ID is required to retrieve movie details.")

        api_key = self._resolve_key()

        params = {
            "apikey": api_key,
            "i": cleaned_id,
            "plot": plot,
        }

        try:
            response = requests.get(OMDB_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.Timeout:
            raise OMDbConnectionError("Request timed out while fetching movie details.")
        except requests.exceptions.ConnectionError:
            raise OMDbConnectionError("Network connection failed. Unable to fetch movie details.")
        except requests.exceptions.RequestException:
            raise OMDbConnectionError("Network error occurred while fetching movie details.")

        try:
            data = response.json()
        except Exception:
            raise OMDbError(f"Received invalid response format from OMDb (HTTP {response.status_code}).")

        if data.get("Response") == "False":
            err_msg = data.get("Error", "Failed to retrieve movie details.")
            if "Invalid API key" in err_msg:
                raise OMDbApiKeyInvalidError("The provided OMDb API key is invalid.")
            raise OMDbError(f"OMDb Error: {err_msg}")

        return {
            "title": data.get("Title", "N/A"),
            "year": data.get("Year", "N/A"),
            "rated": data.get("Rated", "N/A"),
            "released": data.get("Released", "N/A"),
            "runtime": data.get("Runtime", "N/A"),
            "genre": data.get("Genre", "N/A"),
            "director": data.get("Director", "N/A"),
            "writer": data.get("Writer", "N/A"),
            "actors": data.get("Actors", "N/A"),
            "plot": data.get("Plot", "No plot summary available."),
            "language": data.get("Language", "N/A"),
            "country": data.get("Country", "N/A"),
            "awards": data.get("Awards", "N/A"),
            "poster": data.get("Poster", "N/A"),
            "ratings": data.get("Ratings", []),
            "metascore": data.get("Metascore", "N/A"),
            "imdb_rating": data.get("imdbRating", "N/A"),
            "imdb_votes": data.get("imdbVotes", "N/A"),
            "imdb_id": data.get("imdbID", cleaned_id),
            "type": data.get("Type", "movie"),
            "box_office": data.get("BoxOffice", "N/A"),
        }
