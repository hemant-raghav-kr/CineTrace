"""
Comprehensive Phase 5 Automated Test Suite for Scene-to-Movie Finder
Tests dataset loading, text normalization, matching engine, filters,
offline enrichment, thresholding, and regression checks for Phases 1-4.
All tests run completely offline without requiring an active OMDb key or network access.
"""

import os
import sys
import tempfile
import json
import csv
from pathlib import Path

# Ensure application modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

import scene_matcher
from scene_matcher import (
    SceneMatcher,
    load_plot_dataset,
    normalize_text,
    tokenize,
    enrich_candidate,
    DatasetNotFoundError,
    DatasetError,
)
import database

# Synthetic test records for isolated unit tests (not production data)
SYNTHETIC_DATASET = [
    {
        "imdb_id": "tt9990001",
        "title": "Time Traveler's Loop",
        "year": "2015",
        "genre": "Sci-Fi, Thriller",
        "plot": "A physicist is trapped in a temporal anomaly where he relives the same day repeatedly, desperately trying to escape the endless time loop before time unravels."
    },
    {
        "imdb_id": "tt9990002",
        "title": "Deep Ocean Voyage",
        "year": "1998",
        "genre": "Adventure, Drama",
        "plot": "Marine biologists embark on a submarine expedition into the Mariana Trench to catalog deep-sea hydrothermal vents and bioluminescent creatures."
    },
    {
        "imdb_id": "tt9990003",
        "title": "Medieval Kingdom",
        "year": "2005",
        "genre": "Action, History",
        "plot": "Knights and archers defend a stone fortress during a prolonged siege in fourteenth-century Europe, forging strategic alliances between rival dukes."
    },
    {
        "imdb_id": "tt9990004",
        "title": "Jazz Club Romance",
        "year": "2021",
        "genre": "Comedy, Romance",
        "plot": "A passionate trumpet player and a vintage boutique owner fall in love against the backdrop of a bustling subterranean jazz club in New Orleans."
    }
]


def run_tests():
    print("=" * 65)
    print("CineTrace Phase 5 - Scene Finder Automated Verification Suite")
    print("=" * 65)

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    try:
        # -----------------------------------------------------------
        # Test 1: Dataset loading with valid records (JSON and CSV)
        # -----------------------------------------------------------
        print("\n--- Test 1: Dataset loading with valid records ---")
        json_file = temp_path / "valid_dataset.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(SYNTHETIC_DATASET, f)

        loaded_json = load_plot_dataset(json_file)
        assert len(loaded_json) == 4, f"Expected 4 records, got {len(loaded_json)}"
        assert loaded_json[0]["title"] == "Time Traveler's Loop"

        # CSV format test
        csv_file = temp_path / "valid_dataset.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["imdb_id", "title", "year", "genre", "plot"])
            writer.writeheader()
            for row in SYNTHETIC_DATASET:
                writer.writerow(row)

        loaded_csv = load_plot_dataset(csv_file)
        assert len(loaded_csv) == 4, f"Expected 4 records from CSV, got {len(loaded_csv)}"
        assert loaded_csv[1]["title"] == "Deep Ocean Voyage"
        print("[PASS] Valid dataset loaded cleanly from both JSON and CSV files.")

        # -----------------------------------------------------------
        # Test 2: Missing or empty dataset handling
        # -----------------------------------------------------------
        print("\n--- Test 2: Missing or empty dataset handling ---")
        non_existent = temp_path / "does_not_exist.json"
        try:
            load_plot_dataset(non_existent)
            assert False, "Should have raised DatasetNotFoundError for missing file"
        except DatasetNotFoundError:
            print("  -> Handled non-existent file cleanly with DatasetNotFoundError.")

        empty_file = temp_path / "empty_dataset.json"
        with open(empty_file, "w", encoding="utf-8") as f:
            f.write("[]")
        empty_records = load_plot_dataset(empty_file)
        assert len(empty_records) == 0, "Empty dataset should return empty list"
        matcher_empty = SceneMatcher(records=[])
        assert not matcher_empty.is_ready()
        assert matcher_empty.search("any query") == []
        print("[PASS] Missing and empty datasets handled gracefully.")

        # -----------------------------------------------------------
        # Test 3: Malformed and incomplete records
        # -----------------------------------------------------------
        print("\n--- Test 3: Malformed and incomplete records ---")
        malformed_json = temp_path / "malformed.json"
        malformed_data = [
            {"title": "Valid Movie", "plot": "A valid plot description.", "year": "2020"},
            {"title": "", "plot": "Missing title record"},  # Invalid
            {"title": "Missing Plot", "plot": ""},          # Invalid
            {"plot": "No title key at all"},                # Invalid
            "Not a dict record",                            # Invalid
            {"title": "   ", "plot": "Whitespace title"}    # Invalid
        ]
        with open(malformed_json, "w", encoding="utf-8") as f:
            json.dump(malformed_data, f)

        sanitized = load_plot_dataset(malformed_json)
        assert len(sanitized) == 1, f"Expected 1 valid record, got {len(sanitized)}"
        assert sanitized[0]["title"] == "Valid Movie"
        print("[PASS] Malformed, missing, and whitespace records excluded safely.")

        # -----------------------------------------------------------
        # Test 4: Ranking description against relevant & irrelevant plots
        # -----------------------------------------------------------
        print("\n--- Test 4: Ranking relevant vs irrelevant plots ---")
        matcher = SceneMatcher(records=SYNTHETIC_DATASET)
        query = "A man caught in an endless time loop reliving the same day"
        results = matcher.search(query, min_threshold=0.05)
        assert len(results) > 0, "Expected at least 1 match"
        top_match = results[0]
        assert top_match["title"] == "Time Traveler's Loop", f"Expected 'Time Traveler's Loop' #1, got {top_match['title']}"
        assert top_match["similarity"] > 0.15, f"Expected high similarity, got {top_match['similarity']}"
        print(f"  -> Top match: '{top_match['title']}' with score {top_match['similarity_pct']}.")

        # Test pure python engine specifically
        matcher_pure = SceneMatcher(records=SYNTHETIC_DATASET, force_pure_python=True)
        results_pure = matcher_pure.search(query, min_threshold=0.05)
        assert len(results_pure) > 0
        assert results_pure[0]["title"] == "Time Traveler's Loop"
        print("[PASS] Ranking successfully prioritized relevant plot above all others.")

        # -----------------------------------------------------------
        # Test 5: Case and punctuation normalization
        # -----------------------------------------------------------
        print("\n--- Test 5: Case and punctuation normalization ---")
        q1 = "time loop escape"
        q2 = "TIME LOOP, ESCAPE!!!"
        q3 = "  ...TiMe   lOoP??   eScApE---  "
        res1 = matcher.search(q1)
        res2 = matcher.search(q2)
        res3 = matcher.search(q3)
        assert len(res1) > 0 and len(res2) > 0 and len(res3) > 0
        assert res1[0]["title"] == res2[0]["title"] == res3[0]["title"]
        assert abs(res1[0]["similarity"] - res2[0]["similarity"]) < 1e-4
        assert abs(res1[0]["similarity"] - res3[0]["similarity"]) < 1e-4
        print("[PASS] Normalization produces identical ranking regardless of case and punctuation.")

        # -----------------------------------------------------------
        # Test 6: Genre filtering
        # -----------------------------------------------------------
        print("\n--- Test 6: Genre filtering ---")
        # Query matches time traveler (Sci-Fi) and Jazz Club (Romance)
        # Search with Romance genre filter
        genre_results = matcher.search("love music romance", genre="Romance")
        for r in genre_results:
            assert "romance" in r["genre"].lower(), f"Genre filter failed for: {r}"
        assert len(genre_results) == 1
        assert genre_results[0]["title"] == "Jazz Club Romance"
        print("[PASS] Genre filter accurately restricted candidates to matching genres.")

        # -----------------------------------------------------------
        # Test 7: Year filtering
        # -----------------------------------------------------------
        print("\n--- Test 7: Year filtering ---")
        # Time Traveler's Loop is 2015. Filter for 2010 to 2018
        year_match = matcher.search("time loop", year_min=2010, year_max=2018)
        assert len(year_match) == 1
        assert year_match[0]["title"] == "Time Traveler's Loop"

        # Filter out 2015 by specifying range 1990-2000
        year_nomatch = matcher.search("time loop", year_min=1990, year_max=2000)
        assert len(year_nomatch) == 0
        print("[PASS] Year range filtering verified (inclusive matches, excluded non-matches).")

        # -----------------------------------------------------------
        # Test 8: Minimum similarity threshold and no-match handling
        # -----------------------------------------------------------
        print("\n--- Test 8: Minimum similarity threshold and no-match ---")
        unrelated_query = "quantum spacecraft cooking lasagna with parmesan cheese"
        # High threshold should filter out noise completely
        no_matches = matcher.search(unrelated_query, min_threshold=0.25)
        assert len(no_matches) == 0, f"Expected 0 matches for unrelated query, got {len(no_matches)}"
        print("[PASS] Unrelated queries rejected by minimum similarity threshold.")

        # -----------------------------------------------------------
        # Test 9: Missing OMDb metadata without losing local results
        # -----------------------------------------------------------
        print("\n--- Test 9: Offline / Missing OMDb metadata retention ---")
        candidate = {
            "imdb_id": "tt9990001",
            "title": "Time Traveler's Loop",
            "year": "2015",
            "genre": "Sci-Fi, Thriller",
            "plot": "A physicist is trapped in a temporal anomaly.",
            "similarity": 0.42,
            "similarity_pct": "42%"
        }
        # Simulate null or failing OMDb client (offline simulation)
        enriched = enrich_candidate(candidate, omdb_client=None)
        assert enriched["title"] == "Time Traveler's Loop"
        assert enriched["similarity"] == 0.42
        assert enriched["plot"] == "A physicist is trapped in a temporal anomaly."
        assert enriched["poster_url"] == "N/A"
        assert enriched["imdb_rating"] == "N/A"
        print("[PASS] Candidate results fully preserved during missing/offline OMDb enrichment.")

        # -----------------------------------------------------------
        # Test 10: Regression checks for Phases 1–4
        # -----------------------------------------------------------
        print("\n--- Test 10: Regression checks for Phases 1–4 ---")
        test_db = temp_path / "regression.db"
        assert database.init_db(test_db), "Database init failed"

        # Wishlist test
        m = {"imdb_id": "tt9990001", "title": "Time Traveler's Loop", "year": "2015", "genre": "Sci-Fi"}
        assert database.add_to_wishlist("tt9990001", m, db_path=test_db)
        assert database.is_in_wishlist("tt9990001", db_path=test_db)

        # Move to Backlog
        assert database.move_wishlist_to_backlog("tt9990001", "High", "Want to Watch", m, db_path=test_db)
        assert not database.is_in_wishlist("tt9990001", db_path=test_db)
        assert database.is_in_backlog("tt9990001", db_path=test_db)

        # Watched status
        assert database.mark_backlog_watched("tt9990001", db_path=test_db)
        assert len(database.get_backlog_movies(include_watched=False, db_path=test_db)) == 0
        assert len(database.get_backlog_movies(include_watched=True, db_path=test_db)) == 1

        # Remove without deleting from movies table
        assert database.remove_from_backlog("tt9990001", db_path=test_db)
        assert database.movie_exists("tt9990001", db_path=test_db)
        print("[PASS] Regression check: Phases 1-4 database, wishlist, and backlog fully functional.")

        # -----------------------------------------------------------
        # Test 11: Phase 5.1 Verification — Gifted (2017) Search Query
        # -----------------------------------------------------------
        print("\n--- Test 11: Phase 5.1 Gifted (2017) Plot Matching ---")
        prod_dataset_file = Path(__file__).parent / "data" / "movies_plots.json"
        assert prod_dataset_file.exists(), f"Production dataset {prod_dataset_file} not found"
        real_matcher = SceneMatcher(file_path=prod_dataset_file)
        assert real_matcher.is_ready()

        gifted_query = (
            "A young girl discovers she has extraordinary mathematical abilities and is "
            "raised by her uncle, who wants her to have a normal childhood."
        )
        gifted_results = real_matcher.search(gifted_query, min_threshold=0.10)
        assert len(gifted_results) >= 1, f"Expected Gifted to match, got {len(gifted_results)} results"
        top_gifted = gifted_results[0]
        assert top_gifted["title"] == "Gifted", f"Expected Gifted #1, got {top_gifted['title']}"
        assert top_gifted["year"] == "2017", f"Expected year 2017, got {top_gifted['year']}"
        assert top_gifted["imdb_id"] == "tt4481414", f"Expected tt4481414, got {top_gifted['imdb_id']}"
        assert top_gifted["similarity"] >= 0.25, f"Expected score >= 0.25, got {top_gifted['similarity']}"
        assert top_gifted["poster_url"].startswith("http"), f"Expected real poster URL, got {top_gifted['poster_url']}"

        # Check pure Python engine with Gifted query
        real_matcher_pure = SceneMatcher(file_path=prod_dataset_file, force_pure_python=True)
        gifted_results_pure = real_matcher_pure.search(gifted_query, min_threshold=0.10)
        assert len(gifted_results_pure) >= 1
        assert gifted_results_pure[0]["title"] == "Gifted"
        assert gifted_results_pure[0]["similarity"] >= 0.25
        print(f"  -> Gifted (2017) ranked #1 with {top_gifted['similarity_pct']} similarity and valid poster.")
        print("[PASS] Gifted query correctly matches Gifted (2017) as top result in both engines.")

        # -----------------------------------------------------------
        # Test 12: Absent movie query and no-confident-match state
        # -----------------------------------------------------------
        print("\n--- Test 12: Absent movie query thresholding ---")
        absent_query = "A samurai warrior in 14th century Kyoto invents a bicycle to deliver noodle soup."
        absent_results = real_matcher.search(absent_query, min_threshold=0.10)
        assert len(absent_results) == 0, f"Expected 0 results for absent movie, got {len(absent_results)}"
        print("  -> Unrelated/absent movie query yielded 0 results under 10% threshold.")
        print("[PASS] Application safely identifies absent movies and triggers no-confident-match state.")

        # -----------------------------------------------------------
        # Test 13: Configurable threshold filtering
        # -----------------------------------------------------------
        print("\n--- Test 13: Configurable threshold levels ---")
        # High precision (15%): Gifted (30%) passes, low incidental matches excluded
        high_res = real_matcher.search(gifted_query, min_threshold=0.15)
        assert len(high_res) == 1
        assert high_res[0]["title"] == "Gifted"

        # Broad threshold (8%): Gifted is still #1, allows broader candidate pool
        broad_res = real_matcher.search(gifted_query, min_threshold=0.08)
        assert len(broad_res) >= 1
        assert broad_res[0]["title"] == "Gifted"
        assert all(r["similarity"] >= 0.08 for r in broad_res)
        print("[PASS] Configurable thresholds (15%, 10%, 8%) verified.")

        # -----------------------------------------------------------
        # Test 14: Poster URL and rating enrichment via SQLite / OMDb
        # -----------------------------------------------------------
        print("\n--- Test 14: Metadata and poster enrichment ---")
        # Candidate with missing poster and rating
        mock_candidate = {
            "imdb_id": "tt0111161",
            "title": "The Shawshank Redemption",
            "year": "1994",
            "poster_url": "N/A",
            "imdb_rating": "N/A",
            "plot": "Two imprisoned men bond over a number of years.",
            "similarity": 0.35,
            "similarity_pct": "35%"
        }
        # Pre-populate SQLite with cached poster and rating
        database.save_movie({
            "imdb_id": "tt0111161",
            "title": "The Shawshank Redemption",
            "year": "1994",
            "imdb_rating": "9.3",
            "poster_url": "https://example.com/shawshank.jpg",
            "plot": "Two imprisoned men bond over a number of years."
        }, db_path=test_db)

        # Enrich candidate from SQLite cache
        enriched_cached = enrich_candidate(mock_candidate, omdb_client=None, db_path=test_db)
        assert enriched_cached["imdb_rating"] == "9.3", f"Expected rating 9.3, got {enriched_cached.get('imdb_rating')}"
        assert enriched_cached["poster_url"] == "https://example.com/shawshank.jpg"
        print("  -> Candidate without poster enriched cleanly from local cache.")
        print("[PASS] Poster and rating enrichment verified.")

        print("\n" + "=" * 65)
        print("ALL 14 PHASE 5.1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 65)

    finally:
        # Cleanup temporary files
        try:
            for p in temp_path.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            temp_path.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    run_tests()
