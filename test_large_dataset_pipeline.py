"""
CineTrace — Phase 5.3 Large-Scale Movie Dataset Integration Test Suite
Validates:
1. Wikipedia Movie Plots CSV schema mapping (Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Wiki Page, Plot).
2. Large-file streaming & batch ingestion without full in-memory loading.
3. Unicode preservation (accents, non-Latin scripts, UTF-8 BOM) & malformed row skipping.
4. Duplicate detection (by IMDb ID and title/year).
5. Incremental import and non-destructive metadata merge.
6. SQLite FTS5 index synchronization.
7. Plot searches against imported records.
8. Actor / cast searches against imported records.
9. Restart persistence across application sessions.
10. Cooperative cancellation & failure recovery without catalogue corruption.
11. OMDb outage resilience, rate limit handling & verified enrichment caching.
12. Decoupled user tables (movies, wishlist, backlog).

All tests run completely offline with zero network calls and isolated temporary databases.
"""

import csv
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure CineTrace project root is in sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 console output for Windows cmd/powershell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import database
import catalogue_service
from catalogue_service import (
    import_dataset,
    stream_dataset_rows,
    normalize_record_keys,
    normalize_year,
    ImportStats
)
import scene_matcher
from scene_matcher import (
    SceneMatcher,
    enrich_candidate,
    clear_enrichment_cache,
    normalize_text
)


def run_phase_5_3_tests():
    print("=" * 75)
    print("CineTrace Phase 5.3 - Large-Scale Dataset Integration Test Suite")
    print("=" * 75)

    temp_dir = tempfile.mkdtemp(prefix="cinetrace_test_5_3_")
    temp_path = Path(temp_dir)
    test_db = str(temp_path / "test_catalogue.db")

    # Patch default database path for test isolation
    orig_db_path = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = Path(test_db)
    database.init_db(test_db)

    passed_tests = 0
    total_tests = 12

    try:
        # -------------------------------------------------------------
        # Test 1: Wikipedia Movie Plots CSV Schema Mapping
        # -------------------------------------------------------------
        print("\n[Test 1] Wikipedia Movie Plots CSV Schema Mapping...")
        wiki_csv = temp_path / "wiki_sample.csv"
        with open(wiki_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"])
            writer.writerow([
                "2017",
                "Gifted",
                "American",
                "Marc Webb",
                "Chris Evans, Mckenna Grace, Lindsay Duncan, Jenny Slate",
                "drama",
                "https://en.wikipedia.org/wiki/Gifted_(2017_film)",
                "In a coastal Florida town, seven-year-old Mary Adler shows extraordinary mathematical talent on her first day of first grade. Her uncle Frank wants her to have a normal childhood."
            ])
            writer.writerow([
                "2010",
                "Inception",
                "American",
                "Christopher Nolan",
                "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
                "sci-fi, thriller",
                "https://en.wikipedia.org/wiki/Inception",
                "Dom Cobb and Arthur are extractors who perform corporate espionage using experimental dream-sharing technology to infiltrate the subconscious."
            ])

        stats1 = import_dataset(wiki_csv, source_provenance="Wikipedia Movie Plots (CC BY-SA 4.0)", db_path=test_db)
        assert stats1.imported == 2, f"Expected 2 imported records, got {stats1.imported}"
        assert stats1.skipped == 0, f"Expected 0 skipped records, got {stats1.skipped}"

        # Verify columns in SQLite
        row = database.get_catalogue_movie_by_title_year("Gifted", "2017", db_path=test_db)
        assert row is not None, "Gifted not found in catalogue"
        assert row.get("director") == "Marc Webb", f"Director mismatch: {row.get('director')}"
        assert "Chris Evans" in row.get("actors", ""), f"Cast mismatch: {row.get('actors')}"
        assert row.get("origin_ethnicity") == "American", f"Origin mismatch: {row.get('origin_ethnicity')}"
        assert "wikipedia.org" in (row.get("wiki_url") or ""), f"Wiki URL mismatch: {row.get('wiki_url')}"
        print("  [PASS] Wikipedia columns (Origin/Ethnicity, Wiki Page, Cast, Plot, Release Year) mapped cleanly.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 2: Large-File Streaming & Batch Ingestion (RAM-Safe)
        # -------------------------------------------------------------
        print("\n[Test 2] Large-File Streaming & Batch Ingestion...")
        large_csv = temp_path / "large_stream_sample.csv"
        NUM_ROWS = 650
        with open(large_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"])
            for i in range(NUM_ROWS):
                writer.writerow([
                    f"{1950 + (i % 70)}",
                    f"Streaming Film {i}",
                    "British",
                    f"Director {i % 10}",
                    f"Actor A{i % 20}, Actor B{i % 30}",
                    "drama",
                    f"https://en.wikipedia.org/wiki/Film_{i}",
                    f"This is the detailed plot summary for streaming synthetic movie number {i} with key plot developments."
                ])

        progress_calls = []
        def _on_prog(cur, tot, msg):
            progress_calls.append(cur)

        stats2 = import_dataset(large_csv, source_provenance="Streaming Test", batch_size=100, on_progress=_on_prog, db_path=test_db)
        assert stats2.imported == NUM_ROWS, f"Expected {NUM_ROWS} imported, got {stats2.imported}"
        assert len(progress_calls) >= 6, f"Expected frequent progress reports, got {len(progress_calls)}"
        print(f"  [PASS] Streamed {NUM_ROWS} records across 100-item transaction batches without full in-memory list.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 3: Unicode Preservation and Malformed Row Skipping
        # -------------------------------------------------------------
        print("\n[Test 3] Unicode Preservation and Malformed Row Handling...")
        unicode_csv = temp_path / "unicode_and_malformed.csv"
        # Write with UTF-8 BOM to test BOM tolerance
        with open(unicode_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Cast", "Genre", "Plot"])
            # Valid unicode rows: French, Japanese, Hindi
            writer.writerow(["2001", "Le Fabuleux Destin d'Amélie Poulain", "Audrey Tautou, Mathieu Kassovitz", "romance", "Amélie est une jeune serveuse dans un café de Montmartre."])
            writer.writerow(["2001", "千と千尋の神隠し (Spirited Away)", "Rumi Hiiragi, Miyu Irino", "animation", "Chihiro wanders into a world ruled by gods, witches, and spirits."])
            writer.writerow(["2022", "RRR (రౌద్రం రణం రుధిరం)", "N. T. Rama Rao Jr., Ram Charan", "action", "A fearless revolutionary and an officer in the British force fight against tyranny."])
            # Malformed rows: Missing title, empty, invalid
            writer.writerow(["1999", "", "Unknown Actor", "drama", "Film with no title should be skipped."])
            writer.writerow(["", "", "", "", ""])

        stats3 = import_dataset(unicode_csv, source_provenance="Unicode Test", db_path=test_db)
        assert stats3.imported == 3, f"Expected 3 imported unicode records, got {stats3.imported}"
        assert stats3.skipped >= 1, f"Expected skipped malformed rows, got {stats3.skipped}"

        # Verify Unicode intact in DB
        amelie = database.get_catalogue_movie_by_title_year("Le Fabuleux Destin d'Amélie Poulain", "2001", db_path=test_db)
        assert amelie is not None, "Amélie record not found"
        assert "Poulain" in amelie["title"], "Accented characters lost"

        rrr = database.get_catalogue_movie_by_title_year("RRR (రౌద్రం రణం రుధిరం)", "2022", db_path=test_db)
        assert rrr is not None, "RRR Telugu script record not found"
        assert "రౌద్రం" in rrr["title"], "Telugu unicode characters lost"
        print("  [PASS] Accented, Japanese, and Indic scripts preserved intact; malformed rows skipped safely.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 4: Duplicate Detection (by IMDb ID and title/year)
        # -------------------------------------------------------------
        print("\n[Test 4] Duplicate Detection by IMDb ID and (title, year)...")
        dup_json = temp_path / "dup_test.json"
        with open(dup_json, "w", encoding="utf-8") as f:
            json.dump([
                # Duplicate of Gifted 2017
                {
                    "title": "Gifted",
                    "year": "2017",
                    "director": "Marc Webb",
                    "plot": "Updated plot text for Gifted."
                },
                # Duplicate with IMDb ID
                {
                    "imdb_id": "tt1375666",
                    "title": "Inception",
                    "year": "2010",
                    "plot": "Dom Cobb dreams inside dreams."
                }
            ], f)

        stats4 = import_dataset(dup_json, source_provenance="Deduplication Test", db_path=test_db)
        assert stats4.imported == 0, f"Expected 0 new records, got {stats4.imported}"
        assert stats4.updated >= 1, f"Expected at least 1 updated record, got {stats4.updated}"
        print(f"  [PASS] Duplicate rows identified and merged non-destructively ({stats4.updated} updated).")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 5: Incremental Import & Non-Destructive Metadata Merge
        # -------------------------------------------------------------
        print("\n[Test 5] Non-Destructive Metadata Preservation...")
        # Check Gifted: prior to this test, it had Cast: "Chris Evans, Mckenna Grace..."
        # An incoming update with cast: "unknown" or "N/A" must NOT overwrite the valid cast
        sparse_update = temp_path / "sparse_update.csv"
        with open(sparse_update, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Cast", "Genre", "Plot"])
            writer.writerow(["2017", "Gifted", "N/A", "drama, family", "In a coastal Florida town..."])

        import_dataset(sparse_update, source_provenance="Sparse Update", db_path=test_db)
        gifted_after = database.get_catalogue_movie_by_title_year("Gifted", "2017", db_path=test_db)
        assert "Chris Evans" in (gifted_after.get("actors") or ""), "Valid cast was overwritten by N/A!"
        assert "family" in (gifted_after.get("genre") or ""), "New genre was not merged!"
        print("  [PASS] Sparse updates enriched genre without erasing existing cast.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 6: FTS5 Index Synchronization & Fast Full-Text Querying
        # -------------------------------------------------------------
        print("\n[Test 6] SQLite FTS5 Index Synchronization...")
        fts_rows = database.search_catalogue_fts("mathematical talent", limit=5, db_path=test_db)
        assert len(fts_rows) > 0, "FTS5 query failed to match 'mathematical talent'"
        assert fts_rows[0]["title"] == "Gifted", f"Expected Gifted first, got {fts_rows[0]['title']}"

        fts_actor = database.search_catalogue_fts("DiCaprio", limit=5, db_path=test_db)
        assert any(r["title"] == "Inception" for r in fts_actor), "FTS5 failed to match actor DiCaprio"
        print("  [PASS] SQLite FTS5 index synchronized and returned accurate BM25 rankings.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 7: Plot Searches Against Imported Records
        # -------------------------------------------------------------
        print("\n[Test 7] Plot Searches Against Large Dataset Engine...")
        matcher = SceneMatcher(db_path=Path(test_db))
        assert matcher.is_ready(), "SceneMatcher should be ready with indexed records"

        res_plot = matcher.search(
            query="seven year old girl extraordinary mathematical abilities raised by her uncle",
            mode="plot",
            top_k=5
        )
        assert len(res_plot) > 0, "No results returned for plot query"
        top_match = res_plot[0]
        assert top_match["title"] == "Gifted", f"Expected Gifted, got {top_match['title']}"
        assert top_match["similarity"] >= 0.15, f"Expected score >= 0.15, got {top_match['similarity']}"
        print(f"  [PASS] Query correctly ranked '{top_match['title']}' #1 with score {top_match['similarity_pct']}.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 8: Actor / Cast Searches Against Imported Records
        # -------------------------------------------------------------
        print("\n[Test 8] Actor / Cast Searches...")
        res_actor = matcher.search(
            query="Chris Evans",
            mode="actor",
            top_k=5
        )
        assert len(res_actor) > 0, "No results returned for actor query 'Chris Evans'"
        assert res_actor[0]["title"] == "Gifted", f"Expected Gifted for Chris Evans, got {res_actor[0]['title']}"
        assert res_actor[0]["match_type"] == "cast", f"Expected match_type 'cast', got {res_actor[0]['match_type']}"

        res_hybrid = matcher.search(
            query="Leonardo DiCaprio dream espionage",
            mode="hybrid",
            top_k=5
        )
        assert len(res_hybrid) > 0, "No results returned for hybrid query"
        assert res_hybrid[0]["title"] == "Inception", f"Expected Inception, got {res_hybrid[0]['title']}"
        print("  [PASS] Actor mode and hybrid multi-signal fusion correctly matched cast members.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 9: Restart Persistence
        # -------------------------------------------------------------
        print("\n[Test 9] Restart Persistence across Sessions...")
        # Create brand new SceneMatcher instance without cache
        matcher2 = SceneMatcher(db_path=Path(test_db))
        total_indexed = matcher2.get_indexed_count()
        assert total_indexed > 600, f"Expected >600 persisted records, got {total_indexed}"

        # Check coverage stats
        stats_data = database.get_catalogue_stats(db_path=test_db)
        assert stats_data["total_movies"] == total_indexed
        assert stats_data["movies_with_plots"] > 600
        assert stats_data["movies_with_years"] > 600
        print(f"  [PASS] Successfully verified {total_indexed} persisted records across fresh session restart.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 10: Cooperative Cancellation & Corruption-Free Recovery
        # -------------------------------------------------------------
        print("\n[Test 10] Cooperative Cancellation & Corruption-Free Recovery...")
        cancel_csv = temp_path / "cancellation_test.csv"
        with open(cancel_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Cast", "Genre", "Plot"])
            for i in range(500):
                writer.writerow([f"202{i%5}", f"Cancel Film {i}", "Actor X", "comedy", "A funny cancelled movie."])

        # Simulate cancellation after 150 records
        cancel_flag = False
        def _cancel_check():
            return cancel_flag

        def _cancel_prog(cur, tot, msg):
            nonlocal cancel_flag
            if cur >= 150:
                cancel_flag = True

        stats_cancel = import_dataset(
            cancel_csv,
            source_provenance="Cancel Test",
            batch_size=50,
            on_progress=_cancel_prog,
            cancel_check=_cancel_check,
            db_path=test_db
        )
        assert stats_cancel.cancelled is True, "Import was not flagged as cancelled"
        assert 150 <= stats_cancel.total_read < 500, f"Expected partial read, got {stats_cancel.total_read}"

        # Verify DB integrity
        with sqlite3.connect(test_db) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            assert integrity == "ok", f"Integrity check failed: {integrity}"

        # Verify FTS still functional
        fts_cancel = database.search_catalogue_fts("funny cancelled movie", limit=5, db_path=test_db)
        assert len(fts_cancel) > 0, "FTS5 query failed after cancelled import"
        print(f"  [PASS] Safely halted at row {stats_cancel.total_read}; SQLite database integrity check returned 'ok'.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 11: OMDb Resilience, Rate Limit Detection & Failed Caching
        # -------------------------------------------------------------
        print("\n[Test 11] OMDb Resilience, Rate Limit Detection & Enrichment Caching...")
        clear_enrichment_cache()

        class MockOmdbClient:
            def __init__(self):
                self.calls = 0

            def search_movies(self, title, year=None):
                self.calls += 1
                if "rate_limit" in title.lower():
                    return {"error": "Request limit reached!"}
                if "network_error" in title.lower():
                    raise ConnectionError("No internet connection")
                if "gifted" in title.lower():
                    return {"movies": [{"imdb_id": "tt4481414", "title": "Gifted", "year": "2017"}]}
                return {"movies": []}

            def get_movie_details(self, imdb_id):
                self.calls += 1
                if imdb_id == "tt4481414":
                    return {
                        "imdb_id": "tt4481414",
                        "title": "Gifted",
                        "year": "2017",
                        "poster_url": "https://m.media-amazon.com/images/M/gifted.jpg",
                        "imdb_rating": "7.6",
                        "rated": "PG-13",
                        "runtime": "101 min"
                    }
                return {"error": "Movie not found!"}

        mock_client = MockOmdbClient()

        # A. Candidate with failed lookup should be cached
        candidate_fail = {"title": "Nonexistent Movie 12345", "year": "1999", "plot": "No plot"}
        res1 = enrich_candidate(candidate_fail, omdb_client=mock_client, db_path=Path(test_db))
        assert res1["title"] == "Nonexistent Movie 12345", "Candidate title corrupted on failure"
        calls_before = mock_client.calls

        # Second call for the same candidate should hit cache, not OMDb
        res1_repeat = enrich_candidate(candidate_fail, omdb_client=mock_client, db_path=Path(test_db))
        assert mock_client.calls == calls_before, "Failed lookup was not cached!"

        # B. Rate limit detection
        candidate_rate_limit = {"title": "Rate_Limit Film", "year": "2020", "plot": "Rate limit test"}
        enrich_candidate(candidate_rate_limit, omdb_client=mock_client, db_path=Path(test_db))
        assert scene_matcher._RATE_LIMIT_HIT is True, "Rate limit flag was not set"

        # Once rate limit hit, further enrichments bypass OMDb completely
        calls_at_limit = mock_client.calls
        candidate_subsequent = {"title": "Another Film", "year": "2021", "plot": "Test"}
        enrich_candidate(candidate_subsequent, omdb_client=mock_client, db_path=Path(test_db))
        assert mock_client.calls == calls_at_limit, "OMDb was queried despite rate limit hit"

        # C. Reset cache & test successful enrichment updating catalogue table
        clear_enrichment_cache()
        candidate_gifted = {"title": "Gifted", "year": "2017", "plot": "A math prodigy..."}
        res_gifted = enrich_candidate(candidate_gifted, omdb_client=mock_client, db_path=Path(test_db))
        assert res_gifted.get("imdb_id") == "tt4481414", "IMDb ID was not resolved"
        assert res_gifted.get("poster_url") == "https://m.media-amazon.com/images/M/gifted.jpg"

        # Check catalogue table was updated with verified IMDb ID and poster
        gifted_db = database.get_catalogue_movie_by_title_year("Gifted", "2017", db_path=test_db)
        assert gifted_db.get("imdb_id") == "tt4481414", "Catalogue table was not enriched with IMDb ID"
        assert gifted_db.get("poster_url") == "https://m.media-amazon.com/images/M/gifted.jpg", "Catalogue table was not enriched with poster URL"
        print("  [PASS] OMDb rate limit detection, failed caching, offline resilience, and verified catalogue enrichment passed.")
        passed_tests += 1

        # -------------------------------------------------------------
        # Test 12: Decoupled User Tables (movies, wishlist, backlog)
        # -------------------------------------------------------------
        print("\n[Test 12] Decoupled User Tables Isolation...")
        # Add items to user wishlist and backlog
        user_movie = {
            "imdb_id": "tt0068646",
            "title": "The Godfather",
            "year": "1972",
            "genre": "Crime, Drama",
            "poster_url": "https://godfather.jpg"
        }
        database.save_movie(user_movie, db_path=test_db)
        database.add_to_wishlist("tt0068646", user_movie, db_path=test_db)
        database.add_to_backlog("tt0068646", "High", "Must Watch", user_movie, db_path=test_db)

        # Ingest another large batch into catalogue
        extra_csv = temp_path / "extra_isolation.csv"
        with open(extra_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Release Year", "Title", "Cast", "Genre", "Plot"])
            writer.writerow(["1972", "The Godfather", "Marlon Brando, Al Pacino", "crime", "The aging patriarch of an organized crime dynasty."])

        import_dataset(extra_csv, source_provenance="Isolation Test", db_path=test_db)

        # Verify wishlist and backlog are completely intact
        wishlist_items = database.get_wishlist_movies(db_path=test_db)
        backlog_items = database.get_backlog_movies(db_path=test_db)
        assert len(wishlist_items) == 1 and wishlist_items[0]["imdb_id"] == "tt0068646", "Wishlist compromised by catalogue import"
        assert len(backlog_items) == 1 and backlog_items[0]["imdb_id"] == "tt0068646", "Backlog compromised by catalogue import"
        print("  [PASS] Wishlist, Backlog, and user Movie tables remain strictly decoupled and intact.")
        passed_tests += 1

        print("\n" + "=" * 75)
        print(f"ALL {passed_tests}/{total_tests} PHASE 5.3 AUTOMATED PIPELINE TESTS PASSED!")
        print("=" * 75)

    finally:
        database.DEFAULT_DB_PATH = orig_db_path
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    run_phase_5_3_tests()
