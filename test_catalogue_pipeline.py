"""
Comprehensive Phase 5.2 Automated Test Suite for Large-Scale Movie Index & Hybrid Search
Tests:
1. Ingestion pipeline (Wikipedia Movie Plots CSV, CineTrace JSON/JSONL, CMU schemas)
2. Deduplication by IMDb ID and (title, year)
3. Non-destructive metadata preservation (empty/None fields do not overwrite rich metadata)
4. Malformed record skipping and logging
5. Multi-signal hybrid matching engine:
   - Title matching (exact and partial)
   - Actor/cast matching (exact and partial)
   - Plot description matching (TF-IDF ranking)
   - Search modes: "hybrid", "plot", "title", "actor"
6. Filtering by genre and release year min/max
7. Configurable similarity score thresholding
8. SQLite FTS5 full-text indexing and LIKE fallback
9. Catalogue stats and coverage breakdown
10. Empty catalogue error handling
11. Offline OMDb resilience (enrichment failures preserve candidate records)
12. Zero regression for Phases 1-4 (movies, wishlist, backlog tables decouple cleanly)
All tests run completely offline without requiring network access or external services.
"""

import os
import sys
import tempfile
import json
import csv
import sqlite3
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import database
import catalogue_service
from catalogue_service import (
    parse_dataset_stream,
    import_dataset,
    ImportStats,
    normalize_record_keys,
    FIELD_MAPPINGS
)
import scene_matcher
from scene_matcher import (
    SceneMatcher,
    enrich_candidate,
    normalize_text,
    tokenize,
    DatasetNotFoundError,
    DatasetError
)


def run_pipeline_tests():
    print("=" * 70)
    print("CineTrace Phase 5.2 - Large-Scale Index & Hybrid Search Verification")
    print("=" * 70)

    # Use an isolated temporary database for test suite execution
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    test_db_path = str(temp_path / "test_catalogue.db")

    # Patch database.DEFAULT_DB_PATH to isolated test DB
    orig_db_path = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = Path(test_db_path)
    database.init_db(test_db_path)

    try:
        # -----------------------------------------------------------------
        # Test 1: Ingestion of Wikipedia Movie Plots CSV Schema
        # -----------------------------------------------------------------
        print("\n--- Test 1: Ingestion of Wikipedia Movie Plots CSV Schema ---")
        wiki_csv = temp_path / "wiki_sample.csv"
        with open(wiki_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"])
            writer.writerow([
                "2017",
                "Gifted",
                "American",
                "Marc Webb",
                "Chris Evans, Mckenna Grace, Lindsay Duncan, Jenny Slate, Octavia Spencer",
                "drama",
                "https://en.wikipedia.org/wiki/Gifted_(2017_film)",
                "In a coastal Florida town, seven-year-old Mary Adler shows extraordinary mathematical talent on her first day of first grade. Her uncle and de facto guardian, Frank, wants her to have a normal childhood."
            ])
            writer.writerow([
                "2010",
                "Inception",
                "American",
                "Christopher Nolan",
                "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
                "sci-fi, thriller",
                "https://en.wikipedia.org/wiki/Inception",
                "Dom Cobb and Arthur are extractors who perform corporate espionage using experimental dream-sharing technology to infiltrate the subconscious of their targets."
            ])

        stats = import_dataset(wiki_csv, source_provenance="Wikipedia Movie Plots", db_path=test_db_path)
        assert stats.imported == 2, f"Expected 2 imported records, got {stats.imported}"
        assert stats.skipped == 0, f"Expected 0 skipped records, got {stats.skipped}"

        movies = database.get_catalogue_movies(db_path=test_db_path)
        assert len(movies) == 2, f"Expected 2 catalogue rows in DB, got {len(movies)}"
        gifted = next((m for m in movies if m["title"] == "Gifted"), None)
        assert gifted is not None, "Gifted not found in catalogue"
        assert gifted["year"] == "2017"
        assert "Chris Evans" in gifted["actors"]
        assert "Marc Webb" in gifted["director"]
        assert "mathematical talent" in gifted["plot"]
        assert gifted["source_provenance"] == "Wikipedia Movie Plots"
        print("[PASS] Wikipedia Movie Plots CSV ingested with exact column mappings.")

        # -----------------------------------------------------------------
        # Test 2: Ingestion of JSON and JSONL datasets
        # -----------------------------------------------------------------
        print("\n--- Test 2: Ingestion of JSON and JSONL datasets ---")
        json_file = temp_path / "custom_sample.json"
        json_data = [
            {
                "imdb_id": "tt1375666",
                "title": "Inception",
                "year": "2010",
                "genre": "Action, Sci-Fi",
                "actors": "Leonardo DiCaprio, Joseph Gordon-Levitt",
                "plot": "Dom Cobb and Arthur are extractors who perform corporate espionage using experimental dream-sharing technology.",
                "poster_url": "https://example.com/inception.jpg",
                "provenance": "OMDb Verified"
            },
            {
                "imdb_id": "tt0110912",
                "title": "Pulp Fiction",
                "year": "1994",
                "genre": "Crime, Drama",
                "actors": "John Travolta, Samuel L. Jackson, Uma Thurman",
                "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
                "provenance": "CineTrace Canonical"
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        json_stats = import_dataset(json_file, source_provenance="JSON Import", db_path=test_db_path)
        # Inception should be updated (deduplicated by title+year), Pulp Fiction is newly imported
        assert json_stats.imported == 1, f"Expected 1 new import (Pulp Fiction), got {json_stats.imported}"
        assert json_stats.updated == 1, f"Expected 1 update (Inception), got {json_stats.updated}"

        pulp = database.get_catalogue_movie_by_id("tt0110912", db_path=test_db_path)
        assert pulp is not None, "Pulp Fiction should be retrieved by IMDb ID"
        assert "John Travolta" in pulp["actors"]
        print("[PASS] JSON import succeeded with updates and new records.")

        # -----------------------------------------------------------------
        # Test 3: Deduplication by IMDb ID and (title, year)
        # -----------------------------------------------------------------
        print("\n--- Test 3: Deduplication by IMDb ID and (title, year) ---")
        count_before = database.get_catalogue_count(db_path=test_db_path)

        # Ingest duplicate of Pulp Fiction by IMDb ID with added director and sparse fields
        dup_csv = temp_path / "dup_sample.csv"
        with open(dup_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["imdb_id", "title", "year", "director", "plot"])
            writer.writerow(["tt0110912", "Pulp Fiction", "1994", "Quentin Tarantino", "Short synopsis"])

        dup_stats = import_dataset(dup_csv, source_provenance="Dup Test", db_path=test_db_path)
        assert dup_stats.imported == 0, f"Expected 0 new imports for duplicate, got {dup_stats.imported}"
        assert dup_stats.updated == 1, f"Expected 1 update for duplicate, got {dup_stats.updated}"
        count_after = database.get_catalogue_count(db_path=test_db_path)
        assert count_after == count_before, "Database record count must not grow when updating existing movie"

        pulp_updated = database.get_catalogue_movie_by_id("tt0110912", db_path=test_db_path)
        # Newly supplied director is merged
        assert pulp_updated["director"] == "Quentin Tarantino"
        # Non-destructive test: actors and richer plot from previous import must NOT be erased!
        assert "John Travolta" in pulp_updated["actors"], "Rich metadata (actors) must be preserved across updates"
        assert "mob hitmen" in pulp_updated["plot"], "Richer existing plot must be preserved over shorter synopsis"
        print("[PASS] Deduplication and non-destructive metadata merge verified.")

        # -----------------------------------------------------------------
        # Test 4: Skipping of malformed and empty records
        # -----------------------------------------------------------------
        print("\n--- Test 4: Skipping of malformed and empty records ---")
        malformed_csv = temp_path / "malformed.csv"
        with open(malformed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["title", "year", "plot"])
            writer.writerow(["", "2020", "A movie with no title"]) # missing title
            writer.writerow(["No Plot Movie", "2020", ""]) # missing plot
            writer.writerow(["Valid Movie", "2020", "A valid plot about space exploration."])

        mal_stats = import_dataset(malformed_csv, db_path=test_db_path)
        assert mal_stats.skipped == 2, f"Expected 2 skipped records, got {mal_stats.skipped}"
        assert mal_stats.imported == 1, f"Expected 1 valid imported record, got {mal_stats.imported}"
        print("[PASS] Malformed rows safely skipped with error tracking.")

        # -----------------------------------------------------------------
        # Test 5: Multi-Signal Hybrid Search (Title, Cast, Plot)
        # -----------------------------------------------------------------
        print("\n--- Test 5: Multi-Signal Hybrid Search ---")
        # Initialize SceneMatcher pointing to our test database
        matcher = SceneMatcher(db_path=test_db_path)
        assert matcher.is_ready(), "SceneMatcher should be ready with indexed test records"
        assert matcher.get_indexed_count() >= 4, f"Indexed count is {matcher.get_indexed_count()}"

        # 5a. Title Match (Exact and partial)
        res_title = matcher.search("Gifted", mode="title", top_k=5)
        assert len(res_title) >= 1, "Expected match for 'Gifted' title search"
        assert res_title[0]["title"] == "Gifted"
        assert res_title[0]["similarity_pct"] == "100%"
        assert "Title Match: 100%" in res_title[0]["score_label"]

        # 5b. Cast Match (Actor search)
        res_actor = matcher.search("Chris Evans", mode="actor", top_k=5)
        assert len(res_actor) >= 1, "Expected match for 'Chris Evans' actor search"
        assert res_actor[0]["title"] == "Gifted"
        assert "Cast Match" in res_actor[0]["score_label"]
        assert "Chris Evans" in res_actor[0]["score_label"]

        # 5c. Plot Description Match (TF-IDF ranking)
        res_plot = matcher.search("mathematical talent first grade normal childhood", mode="plot", top_k=5)
        assert len(res_plot) >= 1, "Expected match for plot query"
        assert res_plot[0]["title"] == "Gifted"
        assert "Plot Match" in res_plot[0]["score_label"]

        # 5d. Hybrid Search (All signals blended)
        res_hybrid = matcher.search("Chris Evans mathematical prodigy", mode="hybrid", top_k=5)
        assert len(res_hybrid) >= 1, "Expected match for hybrid query"
        assert res_hybrid[0]["title"] == "Gifted"
        print("[PASS] Multi-signal search correctly handles Title, Actor, Plot, and Hybrid modes.")

        # -----------------------------------------------------------------
        # Test 6: Filtering by Genre and Year
        # -----------------------------------------------------------------
        print("\n--- Test 6: Filtering by Genre and Year ---")
        # Inception is 2010 Sci-Fi; Gifted is 2017 Drama; Pulp Fiction is 1994 Crime
        res_filter_genre = matcher.search("espionage dream technology", genre="Sci-Fi", top_k=5)
        assert len(res_filter_genre) >= 1
        assert res_filter_genre[0]["title"] == "Inception"

        res_wrong_genre = matcher.search("espionage dream technology", genre="Animation", top_k=5)
        assert len(res_wrong_genre) == 0, "Wrong genre filter must return 0 results"

        # Year range filter
        res_year = matcher.search("mob hitmen diner bandits", year_min=1990, year_max=1999, top_k=5)
        assert len(res_year) >= 1
        assert res_year[0]["title"] == "Pulp Fiction"

        res_year_out = matcher.search("mob hitmen diner bandits", year_min=2000, year_max=2020, top_k=5)
        assert len(res_year_out) == 0, "Out of range year filter must exclude candidate"
        print("[PASS] Genre and Year min/max filters operate accurately.")

        # -----------------------------------------------------------------
        # Test 7: Configurable Minimum Score Rejection
        # -----------------------------------------------------------------
        print("\n--- Test 7: Configurable Minimum Score Rejection ---")
        # Unrelated query should be rejected under standard thresholds
        res_unrelated = matcher.search("Alien spaceships conquering mars with laser weapons", min_threshold=0.10)
        assert len(res_unrelated) == 0, "Unrelated query must not produce false positive matches"

        # Lower threshold allows fringe potential matches if needed
        res_low = matcher.search("space", min_threshold=0.01)
        # Should match Valid Movie or Inception without crashing
        assert isinstance(res_low, list)
        print("[PASS] Configurable threshold rejects low-confidence false positives.")

        # -----------------------------------------------------------------
        # Test 8: SQLite FTS5 Full-Text Indexing and LIKE Fallback
        # -----------------------------------------------------------------
        print("\n--- Test 8: SQLite FTS5 Full-Text Indexing & Fallback ---")
        has_fts5 = database.is_fts5_available()
        print(f"  SQLite FTS5 compile support detected: {has_fts5}")

        fts_results = database.search_catalogue_fts("espionage OR corporate", db_path=test_db_path)
        assert len(fts_results) >= 1, "FTS search should find Inception"
        assert fts_results[0]["title"] == "Inception"

        # Test LIKE fallback explicitly by calling search_catalogue_like directly
        like_results = database.search_catalogue_like("subconscious", db_path=test_db_path)
        assert len(like_results) >= 1, "LIKE fallback should find Inception"
        assert like_results[0]["title"] == "Inception"
        print("[PASS] Full-text search and LIKE fallback verified.")

        # -----------------------------------------------------------------
        # Test 9: Catalogue Stats and Coverage Reporting
        # -----------------------------------------------------------------
        print("\n--- Test 9: Catalogue Stats and Coverage Reporting ---")
        cat_stats = database.get_catalogue_stats(db_path=test_db_path)
        assert cat_stats["total_movies"] >= 4
        assert cat_stats["movies_with_plots"] >= 4
        assert cat_stats["movies_with_actors"] >= 2
        assert "genres" in cat_stats
        assert "provenance_counts" in cat_stats
        print(f"  Catalogue Stats: {cat_stats['total_movies']} movies, {cat_stats['movies_with_plots']} plots, {cat_stats['movies_with_actors']} cast indexed.")
        print("[PASS] Catalogue statistics and provenance breakdown verified.")

        # -----------------------------------------------------------------
        # Test 10: Empty Catalogue Error Handling
        # -----------------------------------------------------------------
        print("\n--- Test 10: Empty Catalogue Error Handling ---")
        empty_db_path = str(temp_path / "empty_catalogue.db")
        database.init_db(empty_db_path)

        empty_matcher = SceneMatcher(db_path=empty_db_path, auto_seed=False)
        assert not empty_matcher.is_ready()
        assert empty_matcher.get_indexed_count() == 0
        empty_results = empty_matcher.search("any query")
        assert empty_results == [], "Searching empty catalogue must return empty list safely"
        print("[PASS] Empty catalogue handled gracefully without runtime exceptions.")

        # -----------------------------------------------------------------
        # Test 11: Offline OMDb Resilience (No Candidate Loss on Error)
        # -----------------------------------------------------------------
        print("\n--- Test 11: Offline OMDb Resilience ---")
        mock_candidate = {
            "title": "Gifted",
            "year": "2017",
            "plot": "A young math prodigy raised by her uncle.",
            "similarity": 0.45,
            "similarity_pct": "45%",
            "score_label": "Plot Match: 45%"
        }
        # Passing None client simulates network down or offline mode
        enriched = enrich_candidate(mock_candidate, omdb_client=None)
        assert enriched["title"] == "Gifted"
        assert enriched["similarity_pct"] == "45%"
        assert enriched["score_label"] == "Plot Match: 45%"
        print("[PASS] Offline enrichment preserves candidate data without data loss.")

        # -----------------------------------------------------------------
        # Test 12: Decoupled Architecture & Regression Verification (Phases 1-4)
        # -----------------------------------------------------------------
        print("\n--- Test 12: Decoupled Architecture & Phase 1-4 Regression ---")
        # Ensure user movies, wishlist, and backlog tables are intact and independent
        test_movie = {
            "imdb_id": "tt1375666",
            "title": "Inception",
            "year": "2010",
            "rated": "PG-13",
            "released": "16 Jul 2010",
            "runtime": "148 min",
            "genre": "Action, Sci-Fi",
            "director": "Christopher Nolan",
            "writer": "Christopher Nolan",
            "actors": "Leonardo DiCaprio, Joseph Gordon-Levitt",
            "plot": "Subconscious extraction.",
            "language": "English",
            "country": "USA",
            "awards": "Won 4 Oscars",
            "poster": "https://example.com/poster.jpg",
            "imdb_rating": "8.8",
            "imdb_votes": "2,000,000",
            "type": "movie"
        }

        # Save to user movies table
        database.save_movie(test_movie, db_path=test_db_path)
        user_movie = database.get_movie_by_imdb_id("tt1375666", db_path=test_db_path)
        assert user_movie is not None
        assert user_movie["title"] == "Inception"

        # Wishlist management
        database.add_to_wishlist("tt1375666", test_movie, db_path=test_db_path)
        assert database.is_in_wishlist("tt1375666", db_path=test_db_path)
        wishlist_items = database.get_wishlist_movies(db_path=test_db_path)
        assert len(wishlist_items) == 1

        # Backlog management
        database.add_to_backlog("tt1375666", priority="High", status="Want to Watch", movie_data=test_movie, db_path=test_db_path)
        assert database.is_in_backlog("tt1375666", db_path=test_db_path)
        backlog_items = database.get_backlog_movies(include_watched=False, db_path=test_db_path)
        assert len(backlog_items) == 1
        assert backlog_items[0]["priority"] == "High"

        # Removing from wishlist/backlog must NOT affect catalogue
        database.remove_from_wishlist("tt1375666", db_path=test_db_path)
        assert not database.is_in_wishlist("tt1375666", db_path=test_db_path)
        cat_inception = database.get_catalogue_movie_by_id("tt1375666", db_path=test_db_path)
        assert cat_inception is not None, "Catalogue movie must remain untouched when user removes from wishlist"

        print("[PASS] User tables (movies, wishlist, backlog) completely decoupled from catalogue.")

        # -----------------------------------------------------------------
        # Test 13: Canonical Seed Dataset Verification
        # -----------------------------------------------------------------
        print("\n--- Test 13: Canonical Seed Dataset Verification ---")
        canonical_json = PROJECT_ROOT / "data" / "movies_plots.json"
        assert canonical_json.exists(), f"Canonical dataset not found at {canonical_json}"
        with open(canonical_json, "r", encoding="utf-8") as f:
            canonical_records = json.load(f)

        assert len(canonical_records) >= 50, f"Expected at least 50 canonical records, got {len(canonical_records)}"
        # Check that Gifted is in canonical dataset and has actors
        gifted_rec = next((m for m in canonical_records if m.get("title") == "Gifted"), None)
        assert gifted_rec is not None, "Gifted must be present in canonical dataset"
        assert "Chris Evans" in gifted_rec.get("actors", ""), "Gifted must have Chris Evans in actors"
        assert "mathematical" in gifted_rec.get("plot", "").lower()
        print(f"[PASS] Canonical dataset contains {len(canonical_records)} fully enriched films.")

        # -----------------------------------------------------------------
        # Test 14: Sample Import CSV Verification
        # -----------------------------------------------------------------
        print("\n--- Test 14: Sample Import CSV Verification ---")
        sample_csv = PROJECT_ROOT / "data" / "sample_import_movies.csv"
        assert sample_csv.exists(), f"Sample import CSV not found at {sample_csv}"
        sample_stats = import_dataset(sample_csv, source_provenance="Sample Test CSV", db_path=test_db_path)
        assert (sample_stats.imported + sample_stats.updated) >= 5, "Sample CSV should import or update records"
        print(f"[PASS] Sample import dataset verified: {sample_stats.imported} imported, {sample_stats.updated} updated.")

        print("\n" + "=" * 70)
        print("ALL 14 PHASE 5.2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        # Restore original database path and clean up temporary files
        database.DEFAULT_DB_PATH = orig_db_path
        try:
            for file in temp_path.iterdir():
                file.unlink(missing_ok=True)
            temp_path.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    run_pipeline_tests()
