"""
CineTrace - SQLite Database Layer
Handles local persistence of movie records, user Wishlist, and Backlog
using Python's built-in sqlite3 module. Enforces foreign key constraints
and ensures all connections are closed properly to prevent resource locking.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import config

# Default database location in application directory
DEFAULT_DB_PATH = config.DB_FILE

# Global in-memory cache for catalogue statistics
_CATALOGUE_STATS_CACHE: Dict[str, Dict[str, Any]] = {}
_CATALOGUE_STATS_LOCK = threading.Lock()


def invalidate_catalogue_cache():
    """Invalidate in-memory catalogue statistics and lookup caches."""
    with _CATALOGUE_STATS_LOCK:
        _CATALOGUE_STATS_CACHE.clear()


VALID_PRIORITIES = ("High", "Medium", "Low")
VALID_STATUSES = ("Want to Watch", "Watched")


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a new SQLite connection with row factory and foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def is_fts5_available() -> bool:
    """Return True if SQLite build supports the FTS5 full-text search extension."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts_check USING fts5(x);")
        conn.close()
        return True
    except Exception:
        return False


def init_db(db_path: Optional[Path] = None) -> bool:
    """
    Initialize SQLite database and create tables if they do not exist.
    Creates 'movies', 'wishlist', and 'backlog' tables automatically.
    Preserves all existing data.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            # 1. Main Movies Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS movies (
                    imdb_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    year TEXT,
                    rated TEXT,
                    released TEXT,
                    runtime TEXT,
                    genre TEXT,
                    director TEXT,
                    writer TEXT,
                    actors TEXT,
                    plot TEXT,
                    language TEXT,
                    country TEXT,
                    awards TEXT,
                    poster_url TEXT,
                    metascore TEXT,
                    imdb_rating TEXT,
                    imdb_votes TEXT,
                    movie_type TEXT,
                    box_office TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Wishlist Table (references movies, unique per movie)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wishlist (
                    imdb_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (imdb_id) REFERENCES movies(imdb_id) ON DELETE CASCADE
                );
            """)

            # 3. Backlog Table (references movies, priority & status)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backlog (
                    imdb_id TEXT PRIMARY KEY,
                    priority TEXT NOT NULL DEFAULT 'Medium',
                    status TEXT NOT NULL DEFAULT 'Want to Watch',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (imdb_id) REFERENCES movies(imdb_id) ON DELETE CASCADE
                );
            """)

            # 4. Large-Scale Movie Catalogue Table (Phase 5.2)
            # Decoupled from user movies/wishlist/backlog tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS catalogue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imdb_id TEXT UNIQUE,
                    title TEXT NOT NULL,
                    alternate_titles TEXT DEFAULT '',
                    year TEXT DEFAULT '',
                    plot TEXT DEFAULT '',
                    actors TEXT DEFAULT '',
                    director TEXT DEFAULT '',
                    genre TEXT DEFAULT '',
                    origin_ethnicity TEXT DEFAULT '',
                    wiki_url TEXT DEFAULT '',
                    poster_url TEXT DEFAULT '',
                    source_provenance TEXT DEFAULT '',
                    license_info TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_catalogue_title ON catalogue(title);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_catalogue_year ON catalogue(year);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_catalogue_lower_title ON catalogue(LOWER(title));")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_catalogue_title_year ON catalogue(LOWER(title), year);")

            # Safe migration for existing catalogue databases
            for col in ("origin_ethnicity", "wiki_url"):
                try:
                    cursor.execute(f"ALTER TABLE catalogue ADD COLUMN {col} TEXT DEFAULT '';")
                except sqlite3.OperationalError:
                    pass

            # 5. SQLite FTS5 Full-Text Indexing (if supported by SQLite build)
            if is_fts5_available():
                try:
                    cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS catalogue_fts USING fts5(
                            title,
                            alternate_titles,
                            actors,
                            director,
                            plot,
                            genre,
                            content='catalogue',
                            content_rowid='id'
                        );
                    """)
                    # Automatic sync triggers for FTS5
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS catalogue_ai AFTER INSERT ON catalogue BEGIN
                            INSERT INTO catalogue_fts(rowid, title, alternate_titles, actors, director, plot, genre)
                            VALUES (new.id, new.title, new.alternate_titles, new.actors, new.director, new.plot, new.genre);
                        END;
                    """)
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS catalogue_ad AFTER DELETE ON catalogue BEGIN
                            INSERT INTO catalogue_fts(catalogue_fts, rowid, title, alternate_titles, actors, director, plot, genre)
                            VALUES('delete', old.id, old.title, old.alternate_titles, old.actors, old.director, old.plot, old.genre);
                        END;
                    """)
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS catalogue_au AFTER UPDATE ON catalogue BEGIN
                            INSERT INTO catalogue_fts(catalogue_fts, rowid, title, alternate_titles, actors, director, plot, genre)
                            VALUES('delete', old.id, old.title, old.alternate_titles, old.actors, old.director, old.plot, old.genre);
                            INSERT INTO catalogue_fts(rowid, title, alternate_titles, actors, director, plot, genre)
                            VALUES (new.id, new.title, new.alternate_titles, new.actors, new.director, new.plot, new.genre);
                        END;
                    """)
                except sqlite3.Error as fts_err:
                    print(f"[CineTrace Database Warning] FTS5 setup skipped: {fts_err}")

            # 6. Persistent Dense Embeddings Table (Phase 5.4 Semantic Search)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS catalogue_embeddings (
                    catalogue_id INTEGER PRIMARY KEY,
                    plot_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (catalogue_id) REFERENCES catalogue(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cat_emb_hash ON catalogue_embeddings(plot_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cat_emb_model ON catalogue_embeddings(model_name);")

            # 7. Persistent Embedding Provenance & Metadata Table (Phase 5.4.1)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS catalogue_embeddings_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 8. User Preferences & Settings Table (Phase 6.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to initialize database: {e}")
        return False
    finally:
        conn.close()


# =====================================================================
# Movies Table Operations
# =====================================================================

def save_movie(movie_data: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """
    Insert a movie record or update its details if it already exists.
    Uses IMDb ID as unique key to prevent duplicate records.
    If no valid IMDb ID is present, skips insertion safely without inventing IDs.
    """
    imdb_id = (movie_data.get("imdb_id") or movie_data.get("imdbID") or "").strip()
    if not imdb_id or imdb_id.upper() == "N/A":
        return False

    title = (movie_data.get("title") or movie_data.get("Title") or "Untitled").strip()
    year = movie_data.get("year") or movie_data.get("Year") or "N/A"
    rated = movie_data.get("rated") or movie_data.get("Rated") or "N/A"
    released = movie_data.get("released") or movie_data.get("Released") or "N/A"
    runtime = movie_data.get("runtime") or movie_data.get("Runtime") or "N/A"
    genre = movie_data.get("genre") or movie_data.get("Genre") or "N/A"
    director = movie_data.get("director") or movie_data.get("Director") or "N/A"
    writer = movie_data.get("writer") or movie_data.get("Writer") or "N/A"
    actors = movie_data.get("actors") or movie_data.get("Actors") or "N/A"
    plot = movie_data.get("plot") or movie_data.get("Plot") or "No plot summary available."
    language = movie_data.get("language") or movie_data.get("Language") or "N/A"
    country = movie_data.get("country") or movie_data.get("Country") or "N/A"
    awards = movie_data.get("awards") or movie_data.get("Awards") or "N/A"
    poster_url = movie_data.get("poster") or movie_data.get("Poster") or movie_data.get("poster_url") or "N/A"
    metascore = movie_data.get("metascore") or movie_data.get("Metascore") or "N/A"
    imdb_rating = movie_data.get("imdb_rating") or movie_data.get("imdbRating") or "N/A"
    imdb_votes = movie_data.get("imdb_votes") or movie_data.get("imdbVotes") or "N/A"
    movie_type = movie_data.get("type") or movie_data.get("Type") or movie_data.get("movie_type") or "movie"
    box_office = movie_data.get("box_office") or movie_data.get("BoxOffice") or "N/A"

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO movies (
                    imdb_id, title, year, rated, released, runtime, genre,
                    director, writer, actors, plot, language, country, awards,
                    poster_url, metascore, imdb_rating, imdb_votes, movie_type,
                    box_office, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, CURRENT_TIMESTAMP
                )
                ON CONFLICT(imdb_id) DO UPDATE SET
                    title = excluded.title,
                    year = excluded.year,
                    rated = excluded.rated,
                    released = excluded.released,
                    runtime = excluded.runtime,
                    genre = excluded.genre,
                    director = excluded.director,
                    writer = excluded.writer,
                    actors = excluded.actors,
                    plot = excluded.plot,
                    language = excluded.language,
                    country = excluded.country,
                    awards = excluded.awards,
                    poster_url = excluded.poster_url,
                    metascore = excluded.metascore,
                    imdb_rating = excluded.imdb_rating,
                    imdb_votes = excluded.imdb_votes,
                    movie_type = excluded.movie_type,
                    box_office = excluded.box_office,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                imdb_id, title, year, rated, released, runtime, genre,
                director, writer, actors, plot, language, country, awards,
                poster_url, metascore, imdb_rating, imdb_votes, movie_type,
                box_office
            ))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to save movie '{imdb_id}': {e}")
        return False
    finally:
        conn.close()


def get_movie_by_id(imdb_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a locally stored movie record by its IMDb ID."""
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return None

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE imdb_id = ?", (cleaned_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Query failed for '{imdb_id}': {e}")
        return None
    finally:
        conn.close()


get_movie_by_imdb_id = get_movie_by_id


def movie_exists(imdb_id: str, db_path: Optional[Path] = None) -> bool:
    """Check whether a movie with the given IMDb ID exists in the local database."""
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM movies WHERE imdb_id = ? LIMIT 1", (cleaned_id,))
        return cursor.fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def get_all_movies(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all stored movie records ordered by insertion date descending."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movies ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to retrieve movies: {e}")
        return []
    finally:
        conn.close()


def get_movie_count(db_path: Optional[Path] = None) -> int:
    """Return total number of movies stored in SQLite."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM movies")
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# =====================================================================
# Wishlist Operations
# =====================================================================

def is_in_wishlist(imdb_id: str, db_path: Optional[Path] = None) -> bool:
    """Check whether a movie is currently in the Wishlist."""
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM wishlist WHERE imdb_id = ? LIMIT 1", (cleaned_id,))
        return cursor.fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def add_to_wishlist(imdb_id: str, movie_data: Optional[Dict[str, Any]] = None, db_path: Optional[Path] = None) -> bool:
    """
    Add a movie to the Wishlist.
    If movie_data is provided and movie does not exist in 'movies' table, saves it first.
    Enforces uniqueness using primary key.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id or cleaned_id.upper() == "N/A":
        return False

    path = db_path or DEFAULT_DB_PATH

    # Ensure foreign key target exists in movies table
    if movie_data and not movie_exists(cleaned_id, path):
        save_movie(movie_data, path)

    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO wishlist (imdb_id, created_at)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(imdb_id) DO NOTHING;
            """, (cleaned_id,))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to add '{cleaned_id}' to wishlist: {e}")
        return False
    finally:
        conn.close()


def remove_from_wishlist(imdb_id: str, db_path: Optional[Path] = None) -> bool:
    """
    Remove a movie from the Wishlist.
    Preserves the movie record in the 'movies' table.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM wishlist WHERE imdb_id = ?", (cleaned_id,))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to remove '{cleaned_id}' from wishlist: {e}")
        return False
    finally:
        conn.close()


def get_wishlist_movies(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all movies saved in the Wishlist ordered by added date descending."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.*, w.created_at AS wishlist_added_at
            FROM wishlist w
            JOIN movies m ON w.imdb_id = m.imdb_id
            ORDER BY w.created_at DESC;
        """)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to retrieve wishlist: {e}")
        return []
    finally:
        conn.close()


def get_wishlist_ids(db_path: Optional[Path] = None) -> Set[str]:
    """Retrieve all IMDb IDs currently saved in Wishlist as a set for instant O(1) checks."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT imdb_id FROM wishlist;")
        return {r[0] for r in cursor.fetchall() if r[0]}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def move_wishlist_to_backlog(
    imdb_id: str,
    priority: str = "Medium",
    status: str = "Want to Watch",
    movie_data: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> bool:
    """
    Atomic transaction: Adds movie to Backlog and removes it from Wishlist.
    Preserves the movie record in the 'movies' table.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    cleaned_priority = priority if priority in VALID_PRIORITIES else "Medium"
    cleaned_status = status if status in VALID_STATUSES else "Want to Watch"

    path = db_path or DEFAULT_DB_PATH

    # Ensure movie exists in movies table first
    if movie_data and not movie_exists(cleaned_id, path):
        save_movie(movie_data, path)

    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            # 1. Add to Backlog (or update if already queued)
            cursor.execute("""
                INSERT INTO backlog (imdb_id, priority, status, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(imdb_id) DO UPDATE SET
                    priority = excluded.priority,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP;
            """, (cleaned_id, cleaned_priority, cleaned_status))

            # 2. Remove from Wishlist
            cursor.execute("DELETE FROM wishlist WHERE imdb_id = ?", (cleaned_id,))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to move '{cleaned_id}' from wishlist to backlog: {e}")
        return False
    finally:
        conn.close()


# =====================================================================
# Backlog Operations
# =====================================================================

def is_in_backlog(imdb_id: str, active_only: bool = False, db_path: Optional[Path] = None) -> bool:
    """
    Check whether a movie is in the Backlog.
    If active_only is True, returns True only if status != 'Watched'.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    if isinstance(active_only, (str, Path)):
        db_path = Path(active_only)
        active_only = False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT 1 FROM backlog WHERE imdb_id = ? AND status != 'Watched' LIMIT 1", (cleaned_id,))
        else:
            cursor.execute("SELECT 1 FROM backlog WHERE imdb_id = ? LIMIT 1", (cleaned_id,))
        return cursor.fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def get_backlog_ids(active_only: bool = True, db_path: Optional[Path] = None) -> Set[str]:
    """Retrieve all IMDb IDs currently saved in Backlog as a set for instant O(1) checks."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT imdb_id FROM backlog WHERE status != 'Watched';")
        else:
            cursor.execute("SELECT imdb_id FROM backlog;")
        return {r[0] for r in cursor.fetchall() if r[0]}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def add_to_backlog(
    imdb_id: str,
    priority: str = "Medium",
    status: str = "Want to Watch",
    movie_data: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> bool:
    """
    Add a movie to the Backlog.
    Defaults to Medium priority and 'Want to Watch' status.
    If movie_data is provided and movie does not exist in 'movies', saves it first.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id or cleaned_id.upper() == "N/A":
        return False

    cleaned_priority = priority if priority in VALID_PRIORITIES else "Medium"
    cleaned_status = status if status in VALID_STATUSES else "Want to Watch"

    path = db_path or DEFAULT_DB_PATH

    if movie_data and not movie_exists(cleaned_id, path):
        save_movie(movie_data, path)

    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backlog (imdb_id, priority, status, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(imdb_id) DO UPDATE SET
                    priority = excluded.priority,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP;
            """, (cleaned_id, cleaned_priority, cleaned_status))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to add '{cleaned_id}' to backlog: {e}")
        return False
    finally:
        conn.close()


def remove_from_backlog(imdb_id: str, db_path: Optional[Path] = None) -> bool:
    """
    Remove a movie from the Backlog.
    Preserves the movie record in the 'movies' table.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM backlog WHERE imdb_id = ?", (cleaned_id,))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to remove '{cleaned_id}' from backlog: {e}")
        return False
    finally:
        conn.close()


def get_backlog_movies(include_watched: bool = False, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Retrieve queued movies from Backlog.
    By default (include_watched=False), returns only active movies (status != 'Watched').
    """
    if isinstance(include_watched, (str, Path)):
        db_path = Path(include_watched)
        include_watched = False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        if include_watched:
            cursor.execute("""
                SELECT m.*, b.priority, b.status, b.created_at AS backlog_added_at, b.updated_at AS backlog_updated_at
                FROM backlog b
                JOIN movies m ON b.imdb_id = m.imdb_id
                ORDER BY b.created_at DESC;
            """)
        else:
            cursor.execute("""
                SELECT m.*, b.priority, b.status, b.created_at AS backlog_added_at, b.updated_at AS backlog_updated_at
                FROM backlog b
                JOIN movies m ON b.imdb_id = m.imdb_id
                WHERE b.status != 'Watched'
                ORDER BY b.created_at DESC;
            """)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to retrieve backlog: {e}")
        return []
    finally:
        conn.close()


def get_backlog_entry(imdb_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve backlog entry details (priority, status) for a specific movie."""
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return None

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM backlog WHERE imdb_id = ?", (cleaned_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def update_backlog_priority(imdb_id: str, priority: str, db_path: Optional[Path] = None) -> bool:
    """Update priority for a movie in the Backlog ('High', 'Medium', 'Low')."""
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    cleaned_priority = priority if priority in VALID_PRIORITIES else "Medium"
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE backlog
                SET priority = ?, updated_at = CURRENT_TIMESTAMP
                WHERE imdb_id = ?;
            """, (cleaned_priority, cleaned_id))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to update priority for '{cleaned_id}': {e}")
        return False
    finally:
        conn.close()


def mark_backlog_watched(imdb_id: str, db_path: Optional[Path] = None) -> bool:
    """
    Mark a movie in the Backlog as 'Watched'.
    Updates its status in SQLite, which removes it from the active Backlog view.
    Preserves the movie record in the 'movies' table.
    """
    cleaned_id = (imdb_id or "").strip()
    if not cleaned_id:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE backlog
                SET status = 'Watched', updated_at = CURRENT_TIMESTAMP
                WHERE imdb_id = ?;
            """, (cleaned_id,))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to mark '{cleaned_id}' as watched: {e}")
        return False
    finally:
        conn.close()


# =====================================================================
# Large-Scale Movie Catalogue Operations (Phase 5.2)
# =====================================================================

def _merge_catalogue_fields(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge incoming record into existing record without overwriting richer metadata.
    Preserves non-empty existing values if incoming values are empty, N/A, or unknown.
    """
    merged = dict(existing)
    for field in (
        "title", "alternate_titles", "year", "genre", "actors", "director",
        "origin_ethnicity", "wiki_url", "poster_url", "source_provenance", "license_info"
    ):
        inc_val = incoming.get(field)
        if inc_val is not None:
            s_val = str(inc_val).strip()
            if s_val and s_val != "N/A" and s_val.lower() != "unknown":
                merged[field] = s_val

    # Adopt valid IMDb ID if existing is missing
    inc_id = (incoming.get("imdb_id") or incoming.get("imdbID") or "").strip()
    if inc_id and inc_id.upper() != "N/A" and not merged.get("imdb_id"):
        merged["imdb_id"] = inc_id

    # For plot: preserve existing richer plot unless incoming is non-empty and longer
    inc_plot = (incoming.get("plot") or incoming.get("Plot") or "").strip()
    ext_plot = (existing.get("plot") or "").strip()
    if inc_plot and inc_plot != "N/A":
        if not ext_plot or len(inc_plot) >= len(ext_plot):
            merged["plot"] = inc_plot

    return merged


def save_catalogue_movie(movie_data: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """
    Insert or update a single movie record in the catalogue.
    Deduplicates using IMDb ID when available, or (title, year) fallback.
    Preserves richer existing fields when incoming fields are empty.
    """
    title = (movie_data.get("title") or movie_data.get("Title") or "").strip()
    plot = (movie_data.get("plot") or movie_data.get("Plot") or "").strip()
    imdb_id = (movie_data.get("imdb_id") or movie_data.get("imdbID") or "").strip()
    imdb_id_val = imdb_id if imdb_id and imdb_id.upper() != "N/A" else None

    if not title and not plot and not imdb_id_val:
        return False

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            existing_row = None
            if imdb_id_val:
                cursor.execute("SELECT * FROM catalogue WHERE imdb_id = ?", (imdb_id_val,))
                existing_row = cursor.fetchone()
            if not existing_row and title:
                year = str(movie_data.get("year") or movie_data.get("Year") or "").strip()
                cursor.execute(
                    "SELECT * FROM catalogue WHERE LOWER(title) = LOWER(?) AND year = ?",
                    (title, year)
                )
                existing_row = cursor.fetchone()

            if existing_row:
                existing_dict = dict(existing_row)
                merged = _merge_catalogue_fields(existing_dict, movie_data)
                cursor.execute("""
                    UPDATE catalogue
                    SET imdb_id = ?, title = ?, alternate_titles = ?, year = ?, plot = ?,
                        actors = ?, director = ?, genre = ?, origin_ethnicity = ?, wiki_url = ?,
                        poster_url = ?, source_provenance = ?, license_info = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                """, (
                    merged.get("imdb_id"),
                    merged.get("title") or title or "Untitled",
                    merged.get("alternate_titles", ""),
                    merged.get("year", ""),
                    merged.get("plot", ""),
                    merged.get("actors", ""),
                    merged.get("director", ""),
                    merged.get("genre", ""),
                    merged.get("origin_ethnicity", ""),
                    merged.get("wiki_url", ""),
                    merged.get("poster_url", ""),
                    merged.get("source_provenance", ""),
                    merged.get("license_info", ""),
                    existing_dict["id"]
                ))
            else:
                cursor.execute("""
                    INSERT INTO catalogue (
                        imdb_id, title, alternate_titles, year, plot, actors,
                        director, genre, origin_ethnicity, wiki_url, poster_url, source_provenance, license_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    imdb_id_val,
                    title or "Untitled",
                    str(movie_data.get("alternate_titles") or "").strip(),
                    str(movie_data.get("year") or movie_data.get("Year") or "").strip(),
                    plot,
                    str(movie_data.get("actors") or movie_data.get("Cast") or movie_data.get("cast") or "").strip(),
                    str(movie_data.get("director") or movie_data.get("Director") or "").strip(),
                    str(movie_data.get("genre") or movie_data.get("Genre") or "").strip(),
                    str(movie_data.get("origin_ethnicity") or "").strip(),
                    str(movie_data.get("wiki_url") or "").strip(),
                    str(movie_data.get("poster_url") or movie_data.get("poster") or "").strip(),
                    str(movie_data.get("source_provenance") or "").strip(),
                    str(movie_data.get("license_info") or "").strip()
                ))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to save catalogue movie: {e}")
        return False
    finally:
        conn.close()


def bulk_upsert_catalogue(
    records: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
    rebuild_fts: bool = True,
    existing_lookup: Optional[Tuple[Dict[str, int], Dict[Tuple[str, str], int]]] = None
) -> Tuple[int, int, int]:
    """
    Efficient batch upsert for catalogue records inside a single transaction.
    Returns (inserted_count, updated_count, skipped_count).
    Deduplicates using IMDb ID and (title, year) fallback.
    Preserves richer existing metadata without silent overwrite.
    Accepts optional pre-populated existing_lookup (existing_by_imdb, existing_by_title_year)
    to avoid re-querying all catalogue rows across multiple streaming batches.
    """
    if not records:
        return 0, 0, 0

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    inserted = 0
    updated = 0
    skipped = 0

    try:
        with conn:
            cursor = conn.cursor()
            if existing_lookup is not None:
                existing_by_imdb, existing_by_title_year = existing_lookup
            else:
                # Cache existing records in memory for rapid lookup during bulk ingest
                cursor.execute("SELECT id, imdb_id, LOWER(title) AS lower_title, year FROM catalogue;")
                existing_by_imdb = {}
                existing_by_title_year = {}
                for r in cursor.fetchall():
                    row_id = r["id"]
                    if r["imdb_id"]:
                        existing_by_imdb[r["imdb_id"]] = row_id
                    if r["lower_title"]:
                        existing_by_title_year[(r["lower_title"], str(r["year"] or ""))] = row_id

            for item in records:
                if not isinstance(item, dict):
                    skipped += 1
                    continue

                title = (item.get("title") or item.get("Title") or "").strip()
                plot = (item.get("plot") or item.get("Plot") or "").strip()
                imdb_id = (item.get("imdb_id") or item.get("imdbID") or "").strip()
                imdb_id_val = imdb_id if imdb_id and imdb_id.upper() != "N/A" else None
                year = str(item.get("year") or item.get("Year") or "").strip()

                if not title and not plot and not imdb_id_val:
                    skipped += 1
                    continue

                existing_id = None
                if imdb_id_val and imdb_id_val in existing_by_imdb:
                    existing_id = existing_by_imdb[imdb_id_val]
                elif title and (title.lower(), year) in existing_by_title_year:
                    existing_id = existing_by_title_year[(title.lower(), year)]

                if existing_id is not None:
                    # Update existing record safely
                    cursor.execute("SELECT * FROM catalogue WHERE id = ?", (existing_id,))
                    row = cursor.fetchone()
                    if row:
                        merged = _merge_catalogue_fields(dict(row), item)
                        cursor.execute("""
                            UPDATE catalogue
                            SET imdb_id = ?, title = ?, alternate_titles = ?, year = ?, plot = ?,
                                actors = ?, director = ?, genre = ?, origin_ethnicity = ?, wiki_url = ?,
                                poster_url = ?, source_provenance = ?, license_info = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?;
                        """, (
                            merged.get("imdb_id"),
                            merged.get("title") or title or "Untitled",
                            merged.get("alternate_titles", ""),
                            merged.get("year", ""),
                            merged.get("plot", ""),
                            merged.get("actors", ""),
                            merged.get("director", ""),
                            merged.get("genre", ""),
                            merged.get("origin_ethnicity", ""),
                            merged.get("wiki_url", ""),
                            merged.get("poster_url", ""),
                            merged.get("source_provenance", ""),
                            merged.get("license_info", ""),
                            existing_id
                        ))
                        updated += 1
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO catalogue (
                            imdb_id, title, alternate_titles, year, plot, actors,
                            director, genre, origin_ethnicity, wiki_url, poster_url, source_provenance, license_info
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        imdb_id_val,
                        title or "Untitled",
                        str(item.get("alternate_titles") or "").strip(),
                        year,
                        plot,
                        str(item.get("actors") or item.get("Cast") or item.get("cast") or "").strip(),
                        str(item.get("director") or item.get("Director") or "").strip(),
                        str(item.get("genre") or item.get("Genre") or "").strip(),
                        str(item.get("origin_ethnicity") or "").strip(),
                        str(item.get("wiki_url") or "").strip(),
                        str(item.get("poster_url") or item.get("poster") or "").strip(),
                        str(item.get("source_provenance") or "").strip(),
                        str(item.get("license_info") or "").strip()
                    ))
                    new_id = cursor.lastrowid
                    if imdb_id_val:
                        existing_by_imdb[imdb_id_val] = new_id
                    if title:
                        existing_by_title_year[(title.lower(), year)] = new_id
                    inserted += 1

        # Rebuild FTS5 index if requested to guarantee index integrity after bulk changes
        if rebuild_fts:
            rebuild_catalogue_fts(path)

        invalidate_catalogue_cache()
        return inserted, updated, skipped
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Bulk upsert catalogue failed: {e}")
        return inserted, updated, skipped
    finally:
        conn.close()


def get_catalogue_movies(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all movie records from the catalogue table."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM catalogue ORDER BY id ASC;")
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to retrieve catalogue: {e}")
        return []
    finally:
        conn.close()


def get_catalogue_movie_by_id(imdb_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single catalogue movie record by IMDb ID."""
    cleaned = str(imdb_id or "").strip()
    if not cleaned:
        return None
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM catalogue WHERE imdb_id = ?;", (cleaned,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def get_catalogue_movie_by_title_year(title: str, year: Optional[str] = None, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single catalogue movie record by title and optional release year."""
    cleaned = (title or "").strip()
    if not cleaned:
        return None
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        if year:
            cursor.execute("SELECT * FROM catalogue WHERE LOWER(title) = LOWER(?) AND year = ? LIMIT 1;", (cleaned, str(year)))
        else:
            cursor.execute("SELECT * FROM catalogue WHERE LOWER(title) = LOWER(?) LIMIT 1;", (cleaned,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()



def get_catalogue_count(db_path: Optional[Path] = None) -> int:
    """Return total number of movies currently stored in the catalogue."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM catalogue;")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def get_catalogue_stats(db_path: Optional[Path] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Return comprehensive statistics on catalogue coverage:
    Total records, records with plots, records with cast, records with IMDb IDs,
    records with genres, and source provenance breakdown.
    Caches results in memory to avoid repeated table scans on un-mutated data.
    """
    path = Path(db_path or DEFAULT_DB_PATH)
    cache_key = str(path.resolve())

    with _CATALOGUE_STATS_LOCK:
        if not force_refresh and cache_key in _CATALOGUE_STATS_CACHE:
            return dict(_CATALOGUE_STATS_CACHE[cache_key])

    conn = get_db_connection(path)
    stats: Dict[str, Any] = {
        "total_movies": 0,
        "movies_with_plots": 0,
        "movies_with_actors": 0,
        "movies_with_directors": 0,
        "movies_with_imdb_id": 0,
        "movies_with_genres": 0,
        "movies_with_years": 0,
        "provenances": {},
        "fts5_enabled": is_fts5_available()
    }
    try:
        cursor = conn.cursor()
        # Execute single aggregation pass across catalogue table instead of 7 table scans
        cursor.execute("""
            SELECT
                COUNT(*) AS total_movies,
                COUNT(CASE WHEN plot IS NOT NULL AND TRIM(plot) != '' AND plot != 'N/A' THEN 1 END) AS movies_with_plots,
                COUNT(CASE WHEN actors IS NOT NULL AND TRIM(actors) != '' AND actors != 'N/A' THEN 1 END) AS movies_with_actors,
                COUNT(CASE WHEN director IS NOT NULL AND TRIM(director) != '' AND director != 'N/A' THEN 1 END) AS movies_with_directors,
                COUNT(CASE WHEN imdb_id IS NOT NULL AND TRIM(imdb_id) != '' AND imdb_id != 'N/A' THEN 1 END) AS movies_with_imdb_id,
                COUNT(CASE WHEN genre IS NOT NULL AND TRIM(genre) != '' AND genre != 'N/A' THEN 1 END) AS movies_with_genres,
                COUNT(CASE WHEN year IS NOT NULL AND TRIM(year) != '' AND year != 'N/A' THEN 1 END) AS movies_with_years
            FROM catalogue;
        """)
        row = cursor.fetchone()
        if row:
            stats["total_movies"] = row[0] or 0
            stats["movies_with_plots"] = row[1] or 0
            stats["movies_with_actors"] = row[2] or 0
            stats["movies_with_directors"] = row[3] or 0
            stats["movies_with_imdb_id"] = row[4] or 0
            stats["movies_with_genres"] = row[5] or 0
            stats["movies_with_years"] = row[6] or 0

        cursor.execute("SELECT source_provenance, COUNT(*) FROM catalogue GROUP BY source_provenance;")
        for r in cursor.fetchall():
            prov_name = r[0] or "Unspecified"
            stats["provenances"][prov_name] = r[1]

        # Convenience & descriptive metrics aliases (Phase 5.3)
        stats["genres"] = stats["movies_with_genres"]
        stats["provenance_counts"] = stats["provenances"]
        stats["total_indexed_titles"] = stats["total_movies"]
        stats["titles_with_plots"] = stats["movies_with_plots"]
        stats["titles_with_cast"] = stats["movies_with_actors"]
        stats["titles_with_genres"] = stats["movies_with_genres"]
        stats["titles_with_release_years"] = stats["movies_with_years"]
        stats["records_with_verified_imdb_ids"] = stats["movies_with_imdb_id"]

        with _CATALOGUE_STATS_LOCK:
            _CATALOGUE_STATS_CACHE[cache_key] = dict(stats)

        return stats
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to compute catalogue stats: {e}")
        return stats
    finally:
        conn.close()


def update_catalogue_enrichment(
    title: str,
    year: Optional[str] = None,
    imdb_id: Optional[str] = None,
    poster_url: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """
    Update an existing catalogue movie record with verified IMDb ID and poster from OMDb.
    Persists enrichment so future sessions immediately have the poster and IMDb ID without querying OMDb again.
    """
    cleaned_title = (title or "").strip()
    if not cleaned_title:
        return False
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            if imdb_id and poster_url and poster_url != "N/A":
                cursor.execute("""
                    UPDATE catalogue
                    SET imdb_id = COALESCE(NULLIF(imdb_id, ''), ?),
                        poster_url = COALESCE(NULLIF(poster_url, ''), ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE LOWER(title) = LOWER(?) AND (year = ? OR ? IS NULL OR ? = '');
                """, (imdb_id, poster_url, cleaned_title, str(year or ""), str(year or ""), str(year or "")))
            elif imdb_id:
                cursor.execute("""
                    UPDATE catalogue
                    SET imdb_id = COALESCE(NULLIF(imdb_id, ''), ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE LOWER(title) = LOWER(?) AND (year = ? OR ? IS NULL OR ? = '');
                """, (imdb_id, cleaned_title, str(year or ""), str(year or ""), str(year or "")))
            elif poster_url and poster_url != "N/A":
                cursor.execute("""
                    UPDATE catalogue
                    SET poster_url = COALESCE(NULLIF(poster_url, ''), ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE LOWER(title) = LOWER(?) AND (year = ? OR ? IS NULL OR ? = '');
                """, (poster_url, cleaned_title, str(year or ""), str(year or ""), str(year or "")))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to update catalogue enrichment: {e}")
        return False
    finally:
        conn.close()


FTS_COMMON_STOPWORDS = {
    "a", "about", "after", "again", "all", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours"
}


def search_catalogue_fts(
    query: str,
    mode: str = "all",
    limit: int = 50,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Search catalogue using SQLite FTS5 index where available.
    Supports specific target columns ('title', 'actor', 'plot', or 'all').
    Gracefully falls back to SQL LIKE if FTS5 is not available or query syntax is invalid.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)

    # 1. Attempt FTS5 search if available
    if is_fts5_available():
        try:
            cursor = conn.cursor()
            # Sanitize query for FTS5 (strip punctuation that breaks FTS5 expression parser)
            sanitized = "".join(ch if ch.isalnum() or ch in " _-" else " " for ch in cleaned_query).strip()
            if sanitized:
                tokens = sanitized.split()
                # Only uppercase OR, AND, NOT represent explicit user boolean operators
                has_bool = any(t in ("OR", "AND", "NOT") for t in tokens)

                if has_bool:
                    # User-supplied boolean query
                    expr_parts = []
                    has_term = False
                    for t in tokens:
                        if t.upper() in ("OR", "AND", "NOT"):
                            expr_parts.append(t.upper())
                        else:
                            expr_parts.append(f'"{t}"*')
                            has_term = True
                    fts_expr = " ".join(expr_parts)
                elif len(tokens) >= 3 and mode in ("plot", "scene", "all", "hybrid"):
                    # Natural language sentence: filter common stopwords and combine keywords with OR for BM25 ranking
                    content_tokens = [t for t in tokens if len(t) >= 3 and t.lower() not in FTS_COMMON_STOPWORDS]
                    if content_tokens:
                        fts_expr = " OR ".join(f'"{t}"*' for t in content_tokens)
                    else:
                        fts_expr = " OR ".join(f'"{t}"*' for t in tokens)
                    has_term = True
                else:
                    # Short 1-2 token query: match all with prefix
                    fts_expr = " ".join(f'"{t}"*' for t in tokens)
                    has_term = True

                if has_term:
                    if mode == "title":
                        match_clause = f"title: ({fts_expr}) OR alternate_titles: ({fts_expr})"
                    elif mode in ("actor", "cast"):
                        match_clause = f"actors: ({fts_expr}) OR director: ({fts_expr})"
                    elif mode in ("plot", "scene"):
                        match_clause = f"plot: ({fts_expr})"
                    else:
                        match_clause = fts_expr


                cursor.execute(f"""
                    SELECT c.*, bm25(catalogue_fts) AS rank
                    FROM catalogue c
                    JOIN catalogue_fts f ON c.id = f.rowid
                    WHERE catalogue_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?;
                """, (match_clause, limit))
                rows = cursor.fetchall()
                if rows:
                    return [dict(r) for r in rows]
        except Exception:
            pass

    # 2. Documented Fallback: SQL LIKE pattern matching
    return search_catalogue_like(cleaned_query, mode=mode, limit=limit, db_path=path)


def search_catalogue_like(
    query: str,
    mode: str = "all",
    limit: int = 50,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Search catalogue using SQL LIKE pattern matching.
    Documented fallback when FTS5 is unavailable or for exact substring queries.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        pattern = f"%{cleaned_query}%"
        if mode == "title":
            cursor.execute("""
                SELECT * FROM catalogue
                WHERE title LIKE ? OR alternate_titles LIKE ?
                ORDER BY (title = ?) DESC, title ASC
                LIMIT ?;
            """, (pattern, pattern, cleaned_query, limit))
        elif mode in ("actor", "cast"):
            cursor.execute("""
                SELECT * FROM catalogue
                WHERE actors LIKE ? OR director LIKE ?
                ORDER BY title ASC
                LIMIT ?;
            """, (pattern, pattern, limit))
        elif mode == "plot":
            cursor.execute("""
                SELECT * FROM catalogue
                WHERE plot LIKE ?
                ORDER BY title ASC
                LIMIT ?;
            """, (pattern, limit))
        else:
            cursor.execute("""
                SELECT * FROM catalogue
                WHERE title LIKE ? OR alternate_titles LIKE ? OR actors LIKE ? OR plot LIKE ?
                ORDER BY (title = ?) DESC, title ASC
                LIMIT ?;
            """, (pattern, pattern, pattern, pattern, cleaned_query, limit))
        return [dict(r) for r in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] SQL LIKE search failed: {e}")
        return []
    finally:
        conn.close()


def rebuild_catalogue_fts(db_path: Optional[Path] = None) -> bool:
    """Rebuild the SQLite FTS5 index for the catalogue table."""
    if not is_fts5_available():
        return False
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO catalogue_fts(catalogue_fts) VALUES('rebuild');")
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Warning] Failed to rebuild FTS5 index: {e}")
        return False
    finally:
        conn.close()


def clear_catalogue(db_path: Optional[Path] = None) -> bool:
    """Clear all records from the catalogue table and FTS5 index."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM catalogue;")
            if is_fts5_available():
                try:
                    cursor.execute("INSERT INTO catalogue_fts(catalogue_fts) VALUES('delete-all');")
                except sqlite3.Error:
                    pass
        invalidate_catalogue_cache()
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to clear catalogue: {e}")
        return False
    finally:
        conn.close()


# =====================================================================
# Catalogue Dense Embeddings Operations (Phase 5.4 Semantic Search)
# =====================================================================

def compute_plot_hash(plot: str, title: str = "") -> str:
    """Generate a deterministic 16-hex-char SHA256 digest of plot text and title."""
    content = f"{title.strip().lower()}::{plot.strip().lower()}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def save_catalogue_embeddings_batch(
    records: List[Dict[str, Any]],
    db_path: Optional[Path] = None
) -> int:
    """
    Insert or update a batch of dense embeddings in catalogue_embeddings table.
    Each record must have:
      - 'catalogue_id': int
      - 'plot_hash': str
      - 'embedding': bytes (raw float32 buffer)
      - 'dimension': int
      - 'model_name': str
    Executed inside a single atomic transaction.
    Returns count of saved records.
    """
    if not records:
        return 0
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            rows_to_insert = [
                (
                    r["catalogue_id"],
                    r["plot_hash"],
                    sqlite3.Binary(r["embedding"]) if not isinstance(r["embedding"], sqlite3.Binary) else r["embedding"],
                    r["dimension"],
                    r["model_name"]
                )
                for r in records
            ]
            cursor.executemany("""
                INSERT INTO catalogue_embeddings (
                    catalogue_id, plot_hash, embedding, dimension, model_name, updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(catalogue_id) DO UPDATE SET
                    plot_hash = excluded.plot_hash,
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP;
            """, rows_to_insert)
            return len(rows_to_insert)
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to save embeddings batch: {e}")
        return 0
    finally:
        conn.close()


def get_catalogue_embeddings(
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve all persisted catalogue embeddings.
    Returns list of dicts with 'catalogue_id', 'plot_hash', 'embedding' (bytes), 'dimension', 'model_name'.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        if model_name:
            cursor.execute("""
                SELECT catalogue_id, plot_hash, embedding, dimension, model_name
                FROM catalogue_embeddings
                WHERE model_name = ?
                ORDER BY catalogue_id ASC;
            """, (model_name,))
        else:
            cursor.execute("""
                SELECT catalogue_id, plot_hash, embedding, dimension, model_name
                FROM catalogue_embeddings
                ORDER BY catalogue_id ASC;
            """)
        return [dict(r) for r in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to fetch embeddings: {e}")
        return []
    finally:
        conn.close()


def get_catalogue_embedding_stats(
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Return comprehensive statistics on catalogue embeddings:
    - total_plots: count of catalogue movies with non-empty plot descriptions
    - indexed_count: count of stored embeddings matching model
    - pending_count: count of catalogue plots that have no embedding or a stale hash
    - dimension: vector dimension
    - model_name: model identifier
    """
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        # Total plots in catalogue
        cursor.execute("SELECT COUNT(*) FROM catalogue WHERE trim(plot) != '';")
        total_plots = cursor.fetchone()[0]

        # Indexed embeddings
        target_model = model_name or config.DEFAULT_MODEL_NAME
        cursor.execute("""
            SELECT COUNT(*), dimension FROM catalogue_embeddings
            WHERE model_name = ?
            GROUP BY dimension;
        """, (target_model,))
        row = cursor.fetchone()
        indexed_count = row[0] if row else 0
        dim = row[1] if row else config.SEMANTIC_DIMENSION

        # Stale or pending count: plots without an embedding in catalogue_embeddings
        cursor.execute("""
            SELECT COUNT(*) FROM catalogue c
            LEFT JOIN catalogue_embeddings e ON c.id = e.catalogue_id AND e.model_name = ?
            WHERE trim(c.plot) != '' AND e.catalogue_id IS NULL;
        """, (target_model,))
        missing_count = cursor.fetchone()[0]

        # Provenance metadata
        meta = get_all_catalogue_embedding_meta(db_path=path)
        provenance_status = meta.get("provenance_status", "unverified")
        embedding_type = meta.get("embedding_type", "unknown")
        model_revision = meta.get("model_revision", "")
        schema_version = int(meta.get("schema_version", "1"))

        return {
            "total_plots": total_plots,
            "indexed_count": indexed_count,
            "pending_count": missing_count,
            "dimension": dim,
            "model_name": target_model,
            "provenance_status": provenance_status,
            "embedding_type": embedding_type,
            "model_revision": model_revision,
            "schema_version": schema_version
        }
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to get embedding stats: {e}")
        return {
            "total_plots": 0,
            "indexed_count": 0,
            "pending_count": 0,
            "dimension": config.SEMANTIC_DIMENSION,
            "model_name": model_name or config.DEFAULT_MODEL_NAME,
            "provenance_status": "unverified",
            "embedding_type": "unknown",
            "model_revision": "",
            "schema_version": 1
        }
    finally:
        conn.close()


def get_pending_catalogue_plots(
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve catalogue records that have a usable plot description and either:
    1. Do not have a corresponding embedding in catalogue_embeddings, or
    2. Have an embedding whose plot_hash does not match the current plot text.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    target_model = model_name or config.DEFAULT_MODEL_NAME
    try:
        cursor = conn.cursor()
        query = """
            SELECT c.id, c.title, c.plot, e.plot_hash AS existing_hash
            FROM catalogue c
            LEFT JOIN catalogue_embeddings e ON c.id = e.catalogue_id AND e.model_name = ?
            WHERE trim(c.plot) != '' AND (e.catalogue_id IS NULL OR e.plot_hash IS NULL)
            ORDER BY c.id ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        cursor.execute(query, (target_model,))
        pending = [dict(r) for r in cursor.fetchall()]

        # Also check for stale embeddings where text changed
        cursor.execute("""
            SELECT c.id, c.title, c.plot, e.plot_hash AS existing_hash
            FROM catalogue c
            INNER JOIN catalogue_embeddings e ON c.id = e.catalogue_id AND e.model_name = ?
            WHERE trim(c.plot) != ''
            ORDER BY c.id ASC
        """, (target_model,))
        stale_check = cursor.fetchall()
        for r in stale_check:
            cur_hash = compute_plot_hash(r["plot"], r["title"])
            if r["existing_hash"] != cur_hash:
                pending.append({
                    "id": r["id"],
                    "title": r["title"],
                    "plot": r["plot"],
                    "existing_hash": r["existing_hash"]
                })

        return pending
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to fetch pending plots: {e}")
        return []
    finally:
        conn.close()


def get_catalogue_plots_by_ids(
    ids: List[int],
    db_path: Optional[Path] = None
) -> Dict[int, str]:
    """Retrieve plot descriptions for a list of catalogue IDs."""
    if not ids:
        return {}
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        clean_ids = [int(i) for i in ids]
        placeholders = ",".join("?" for _ in clean_ids)
        cursor.execute(f"SELECT id, plot FROM catalogue WHERE id IN ({placeholders});", clean_ids)
        return {r[0]: r[1] for r in cursor.fetchall()}
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to fetch plots by ids: {e}")
        return {}
    finally:
        conn.close()


def delete_stale_catalogue_embeddings(db_path: Optional[Path] = None) -> int:
    """
    Delete embeddings whose catalogue records no longer exist in catalogue table,
    or whose plots have become empty.
    Returns number of rows deleted.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM catalogue_embeddings
                WHERE catalogue_id NOT IN (
                    SELECT id FROM catalogue WHERE trim(plot) != ''
                );
            """)
            return cursor.rowcount
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to delete stale embeddings: {e}")
        return 0
    finally:
        conn.close()


def clear_catalogue_embeddings(
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None
) -> bool:
    """Clear all embeddings, optionally filtered by model name."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            if model_name:
                cursor.execute("DELETE FROM catalogue_embeddings WHERE model_name = ?;", (model_name,))
            else:
                cursor.execute("DELETE FROM catalogue_embeddings;")
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to clear embeddings: {e}")
        return False
    finally:
        conn.close()


def set_catalogue_embedding_meta(key: str, value: str, db_path: Optional[Path] = None) -> bool:
    """Store or update a key-value pair in catalogue_embeddings_meta table."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO catalogue_embeddings_meta (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP;
            """, (str(key), str(value)))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to set embedding meta: {e}")
        return False
    finally:
        conn.close()


def get_catalogue_embedding_meta(key: str, db_path: Optional[Path] = None) -> Optional[str]:
    """Retrieve a single metadata value by key."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM catalogue_embeddings_meta WHERE key = ?;", (str(key),))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to get embedding meta: {e}")
        return None
    finally:
        conn.close()


def get_all_catalogue_embedding_meta(db_path: Optional[Path] = None) -> Dict[str, str]:
    """Retrieve all metadata key-value pairs as a dictionary."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM catalogue_embeddings_meta;")
        return {r[0]: r[1] for r in cursor.fetchall()}
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to get all embedding meta: {e}")
        return {}
    finally:
        conn.close()


def clear_catalogue_embedding_meta(db_path: Optional[Path] = None) -> bool:
    """Clear all entries from catalogue_embeddings_meta."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM catalogue_embeddings_meta;")
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to clear embedding meta: {e}")
        return False
    finally:
        conn.close()


# =====================================================================
# User Preferences Operations (Phase 6.2 Language Personalisation)
# =====================================================================

def get_preference(key: str, default: Optional[str] = None, db_path: Optional[Path] = None) -> Optional[str]:
    """Retrieve a user preference string by key from the database."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM user_preferences WHERE key = ?;", (key,))
        row = cursor.fetchone()
        if row:
            return str(row["value"])
        return default
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to get preference '{key}': {e}")
        return default
    finally:
        conn.close()


def set_preference(key: str, value: str, db_path: Optional[Path] = None) -> bool:
    """Set a user preference key-value pair in the database."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP;
            """, (key, str(value)))
        return True
    except sqlite3.Error as e:
        print(f"[CineTrace Database Error] Failed to set preference '{key}': {e}")
        return False
    finally:
        conn.close()


def get_preferred_languages(db_path: Optional[Path] = None) -> List[str]:
    """Retrieve list of user preferred languages. Returns [] if none set or cleared."""
    import json
    val = get_preference("preferred_languages", default=None, db_path=db_path)
    if not val:
        return []
    try:
        langs = json.loads(val)
        if isinstance(langs, list):
            return [str(l).strip() for l in langs if str(l).strip()]
    except Exception:
        pass
    return []


def set_preferred_languages(languages: List[str], db_path: Optional[Path] = None) -> bool:
    """Save list of user preferred languages and mark onboarding completed."""
    import json
    clean_langs = [str(l).strip() for l in languages if str(l).strip()]
    ok1 = set_preference("preferred_languages", json.dumps(clean_langs), db_path=db_path)
    ok2 = set_preference("onboarding_completed", "true", db_path=db_path)
    return ok1 and ok2


def clear_preferred_languages(db_path: Optional[Path] = None) -> bool:
    """Clear preferred languages back to default without forcing onboarding again."""
    import json
    return set_preference("preferred_languages", json.dumps([]), db_path=db_path)


def is_onboarding_completed(db_path: Optional[Path] = None) -> bool:
    """Return True if the user has completed or skipped the first-launch onboarding."""
    val = get_preference("onboarding_completed", default="false", db_path=db_path)
    return str(val).lower() == "true"


def set_onboarding_completed(completed: bool = True, db_path: Optional[Path] = None) -> bool:
    """Record whether first-launch onboarding has been completed or skipped."""
    return set_preference("onboarding_completed", "true" if completed else "false", db_path=db_path)
