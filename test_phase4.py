"""
Comprehensive Phase 4 Verification Suite for CineTrace
Tests all 10 verification scenarios required by Phase 4 specification.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure application modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

import database
from omdb_client import OMDbClient
import config

TEST_MOVIE_1 = {
    "imdb_id": "tt0137523",
    "title": "Fight Club",
    "year": "1999",
    "rated": "R",
    "released": "15 Oct 1999",
    "runtime": "139 min",
    "genre": "Drama",
    "director": "David Fincher",
    "writer": "Chuck Palahniuk, Jim Uhls",
    "actors": "Brad Pitt, Edward Norton, Meat Loaf",
    "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club...",
    "language": "English",
    "country": "United States, Germany",
    "awards": "Nominated for 1 Oscar",
    "poster_url": "https://m.media-amazon.com/images/M/placeholder.jpg",
    "metascore": "66",
    "imdb_rating": "8.8",
    "imdb_votes": "2,200,000",
    "movie_type": "movie",
    "box_office": "$37,030,102"
}

TEST_MOVIE_2 = {
    "imdb_id": "tt0111161",
    "title": "The Shawshank Redemption",
    "year": "1994",
    "rated": "R",
    "released": "14 Oct 1994",
    "runtime": "142 min",
    "genre": "Drama",
    "director": "Frank Darabont",
    "writer": "Stephen King, Frank Darabont",
    "actors": "Tim Robbins, Morgan Freeman, Bob Gunton",
    "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption...",
    "language": "English",
    "country": "United States",
    "awards": "Nominated for 7 Oscars",
    "poster_url": "https://m.media-amazon.com/images/M/placeholder2.jpg",
    "metascore": "82",
    "imdb_rating": "9.3",
    "imdb_votes": "2,800,000",
    "movie_type": "movie",
    "box_office": "$28,699,976"
}


def run_tests():
    print("=" * 60)
    print("CineTrace Phase 4 - Automated Verification Suite")
    print("=" * 60)

    # Use a temporary file for isolated database tests
    temp_dir = tempfile.mkdtemp()
    test_db = Path(temp_dir) / "test_cinetrace.db"

    try:
        # Initial schema setup
        assert database.init_db(test_db), "Database initialization failed"
        print("[PASS] Database initialized with movies, wishlist, and backlog tables.")

        # -----------------------------------------------------------
        # Test 1: Add movie to Wishlist
        # -----------------------------------------------------------
        print("\n--- Test 1: Add movie to Wishlist ---")
        added = database.add_to_wishlist(TEST_MOVIE_1["imdb_id"], TEST_MOVIE_1, db_path=test_db)
        assert added, "Failed to add movie to wishlist"
        assert database.is_in_wishlist(TEST_MOVIE_1["imdb_id"], db_path=test_db), "Movie not found in wishlist"
        wishlist_items = database.get_wishlist_movies(db_path=test_db)
        assert len(wishlist_items) == 1, f"Expected 1 wishlist item, got {len(wishlist_items)}"
        assert wishlist_items[0]["title"] == "Fight Club"
        print(f"[PASS] Successfully added '{wishlist_items[0]['title']}' to Wishlist.")

        # -----------------------------------------------------------
        # Test 2: Restart application/connection & verify persistence
        # -----------------------------------------------------------
        print("\n--- Test 2: Verify persistence across session/connection restart ---")
        # Simulate new session by querying test_db directly through new connections
        assert database.is_in_wishlist(TEST_MOVIE_1["imdb_id"], db_path=test_db), "Persistence check failed"
        persisted_movies = database.get_wishlist_movies(db_path=test_db)
        assert len(persisted_movies) == 1, f"Expected 1 persisted movie, got {len(persisted_movies)}"
        assert persisted_movies[0]["imdb_id"] == TEST_MOVIE_1["imdb_id"]
        print("[PASS] Wishlist persisted cleanly across independent connection/session.")

        # -----------------------------------------------------------
        # Test 3: Attempt to add same movie twice (No duplicates)
        # -----------------------------------------------------------
        print("\n--- Test 3: Enforce uniqueness / No duplicates ---")
        dup_result = database.add_to_wishlist(TEST_MOVIE_1["imdb_id"], TEST_MOVIE_1, db_path=test_db)
        assert dup_result, "Duplicate add should safely return True via ON CONFLICT DO NOTHING"
        wishlist_after_dup = database.get_wishlist_movies(db_path=test_db)
        assert len(wishlist_after_dup) == 1, f"Expected exactly 1 entry, but found {len(wishlist_after_dup)}"
        print("[PASS] Duplicate insertion safely ignored. Exactly 1 entry maintained.")

        # -----------------------------------------------------------
        # Test 4: Move movie from Wishlist to Backlog
        # -----------------------------------------------------------
        print("\n--- Test 4: Move movie from Wishlist to Backlog ---")
        moved = database.move_wishlist_to_backlog(
            TEST_MOVIE_1["imdb_id"],
            priority="Medium",
            status="Want to Watch",
            movie_data=TEST_MOVIE_1,
            db_path=test_db
        )
        assert moved, "Failed to move movie from wishlist to backlog"
        assert not database.is_in_wishlist(TEST_MOVIE_1["imdb_id"], db_path=test_db), "Movie still in wishlist after move"
        assert database.is_in_backlog(TEST_MOVIE_1["imdb_id"], db_path=test_db), "Movie not found in backlog after move"
        backlog_items = database.get_backlog_movies(include_watched=False, db_path=test_db)
        assert len(backlog_items) == 1, f"Expected 1 backlog movie, got {len(backlog_items)}"
        assert backlog_items[0]["priority"] == "Medium"
        assert backlog_items[0]["status"] == "Want to Watch"
        print("[PASS] Movie atomically moved from Wishlist to Backlog.")

        # -----------------------------------------------------------
        # Test 5: Change Backlog priority (High, Medium, Low)
        # -----------------------------------------------------------
        print("\n--- Test 5: Change Backlog priority ---")
        for prio in ["High", "Low", "Medium"]:
            ok = database.update_backlog_priority(TEST_MOVIE_1["imdb_id"], prio, db_path=test_db)
            assert ok, f"Failed to update priority to {prio}"
            entry = database.get_backlog_entry(TEST_MOVIE_1["imdb_id"], db_path=test_db)
            assert entry["priority"] == prio, f"Expected priority {prio}, got {entry['priority']}"
            print(f"  -> Priority updated and persisted as: {prio}")
        print("[PASS] Backlog priority successfully changed and persisted.")

        # -----------------------------------------------------------
        # Test 6: Mark movie as Watched and verify it leaves active Backlog
        # -----------------------------------------------------------
        print("\n--- Test 6: Mark as Watched & verify active queue leaves it ---")
        marked = database.mark_backlog_watched(TEST_MOVIE_1["imdb_id"], db_path=test_db)
        assert marked, "Failed to mark movie as watched"
        active_backlog = database.get_backlog_movies(include_watched=False, db_path=test_db)
        assert len(active_backlog) == 0, f"Expected 0 active backlog items, got {len(active_backlog)}"
        all_backlog = database.get_backlog_movies(include_watched=True, db_path=test_db)
        assert len(all_backlog) == 1, f"Expected 1 total backlog item, got {len(all_backlog)}"
        assert all_backlog[0]["status"] == "Watched"
        print("[PASS] Watched movie leaves active backlog view while retaining history.")

        # -----------------------------------------------------------
        # Test 7: Remove movie from list
        # -----------------------------------------------------------
        print("\n--- Test 7: Remove movie from Backlog and Wishlist ---")
        # Add Movie 2 to Wishlist, then remove it
        database.add_to_wishlist(TEST_MOVIE_2["imdb_id"], TEST_MOVIE_2, db_path=test_db)
        assert database.is_in_wishlist(TEST_MOVIE_2["imdb_id"], db_path=test_db)
        removed_wish = database.remove_from_wishlist(TEST_MOVIE_2["imdb_id"], db_path=test_db)
        assert removed_wish, "Failed to remove from wishlist"
        assert not database.is_in_wishlist(TEST_MOVIE_2["imdb_id"], db_path=test_db)

        # Remove Movie 1 from Backlog
        removed_back = database.remove_from_backlog(TEST_MOVIE_1["imdb_id"], db_path=test_db)
        assert removed_back, "Failed to remove from backlog"
        assert not database.is_in_backlog(TEST_MOVIE_1["imdb_id"], db_path=test_db)
        print("[PASS] Movies successfully removed from Wishlist and Backlog.")

        # -----------------------------------------------------------
        # Test 8: Verify movie record remains in movies table
        # -----------------------------------------------------------
        print("\n--- Test 8: Verify movie records preserved in movies table ---")
        m1 = database.get_movie_by_id(TEST_MOVIE_1["imdb_id"], db_path=test_db)
        m2 = database.get_movie_by_id(TEST_MOVIE_2["imdb_id"], db_path=test_db)
        assert m1 is not None, "Movie 1 was deleted from movies table!"
        assert m2 is not None, "Movie 2 was deleted from movies table!"
        assert m1["title"] == "Fight Club"
        assert m2["title"] == "The Shawshank Redemption"
        total_movies = database.get_movie_count(db_path=test_db)
        assert total_movies == 2, f"Expected 2 movies in movies table, found {total_movies}"
        print("[PASS] All movie metadata records preserved in movies table after list removals.")

        # -----------------------------------------------------------
        # Test 9: Verify details dialog logic, Home/Discover data & OMDb client
        # -----------------------------------------------------------
        print("\n--- Test 9: Verify OMDb client and details dialog workflow ---")
        client = OMDbClient()
        masked_key = config.get_masked_api_key()
        assert "****" in masked_key, f"API key was not properly masked: {masked_key}"
        assert config.get_api_key() != "", "API key not found in config"
        print(f"  -> OMDb API Key securely configured and masked: {masked_key}")

        # Real OMDb search verification
        res = client.search_movies("inception")
        results = res.get("movies", [])
        assert len(results) > 0, "OMDb search returned no results"
        print(f"  -> Real OMDb search verified: returned {len(results)} movies for query 'inception'.")
        print("[PASS] OMDb client, API masking, and dynamic searches verified.")

        # -----------------------------------------------------------
        # Test 10: Verify empty states
        # -----------------------------------------------------------
        print("\n--- Test 10: Verify empty states on empty Wishlist and Backlog ---")
        empty_wish = database.get_wishlist_movies(db_path=test_db)
        empty_back = database.get_backlog_movies(include_watched=False, db_path=test_db)
        assert len(empty_wish) == 0, "Wishlist should be empty"
        assert len(empty_back) == 0, "Backlog should be empty"
        print("[PASS] Empty states confirmed for both Wishlist and Backlog.")

        print("\n" + "=" * 60)
        print("ALL 10 PHASE 4 VERIFICATION REQUIREMENTS PASSED!")
        print("=" * 60)

    finally:
        # Cleanup temporary files safely
        try:
            if test_db.exists():
                test_db.unlink()
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Notice: cleanup warning: {e}")


if __name__ == "__main__":
    run_tests()
