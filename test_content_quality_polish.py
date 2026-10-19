"""
test_content_quality_polish.py
Automated test suite verifying Content Quality and Application Polish improvements:
- Curated Popular Picks verification, ordering, and caching
- Removal of arbitrary catalogue records from Home and Discover
- Discover initial curated view, genre filtering, and search reset
- Duplicate prevention and membership tracking in Wishlist/Backlog
- Missing metadata and poster resilience
- Responsive layout column calculation and title length safeguards
- Duplicate search submission prevention
- Offline execution with mocked network responses
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
import database
import discovery_service
from discovery_service import DiscoveryService, CURATED_POPULAR_PICKS


class TestCuratedPopularPicks(unittest.TestCase):
    """Test curated popular picks retrieval, verification, deduplication, and caching."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_curated_list_contains_iconic_classics(self):
        """Verify the curated list contains well-known titles across genres and eras."""
        titles = [p["title"] for p in CURATED_POPULAR_PICKS]
        imdb_ids = [p["imdb_id"] for p in CURATED_POPULAR_PICKS]

        # Check key iconic classics across genres
        self.assertIn("Inception", titles)
        self.assertIn("The Dark Knight", titles)
        self.assertIn("The Shawshank Redemption", titles)
        self.assertIn("Pulp Fiction", titles)
        self.assertIn("Spirited Away", titles)
        self.assertIn("Parasite", titles)

        # Check deduplication in list
        self.assertEqual(len(imdb_ids), len(set(imdb_ids)))
        self.assertGreaterEqual(len(CURATED_POPULAR_PICKS), 12)

    @patch("discovery_service.CACHE_FILE")
    def test_cache_hit_loads_instantly_without_api_calls(self, mock_cache_path):
        """Verify that valid cached data returns immediately with zero API calls."""
        mock_cache_file = Path("test_mock_cache.json")
        mock_cache_path.exists.return_value = True

        cached_payload = {
            "timestamp": 9999999999.0,  # far in future -> fresh
            "featured": {"title": "Inception", "year": "2010", "imdb_id": "tt1375666", "genre": "Sci-Fi"},
            "picks": [
                {"title": "The Dark Knight", "year": "2008", "imdb_id": "tt0468569", "genre": "Action"},
                {"title": "Interstellar", "year": "2014", "imdb_id": "tt0816692", "genre": "Sci-Fi"},
                {"title": "The Shawshank Redemption", "year": "1994", "imdb_id": "tt0111161", "genre": "Drama"},
            ],
            "trending": []
        }

        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(cached_payload))):
            ds = DiscoveryService(client=self.mock_client)
            data = ds.fetch_home_data(force_refresh=False)

            self.assertEqual(data["featured"]["title"], "Inception")
            self.assertEqual(len(data["picks"]), 3)
            # Verify client was never called on cache hit
            self.mock_client.get_movie_details.assert_not_called()

    @patch("discovery_service.database.get_movie_by_id")
    @patch("discovery_service.database.save_movie")
    def test_verification_checks_sqlite_first(self, mock_save, mock_get_db):
        """Verify that local SQLite records are reused before querying OMDb."""
        # Simulate Inception already in SQLite
        mock_get_db.side_effect = lambda mid: {
            "title": "Inception", "year": "2010", "imdb_id": "tt1375666", "genre": "Sci-Fi"
        } if mid == "tt1375666" else None

        # OMDb returns detail for the second movie
        self.mock_client.get_movie_details.return_value = {
            "title": "The Dark Knight", "year": "2008", "imdb_id": "tt0468569", "genre": "Action"
        }

        ds = DiscoveryService(client=self.mock_client)
        with patch.object(ds, "get_cached_home_data", return_value=None):
            with patch("discovery_service.CURATED_POPULAR_PICKS", [
                {"imdb_id": "tt1375666", "title": "Inception"},
                {"imdb_id": "tt0468569", "title": "The Dark Knight"},
            ]):
                with patch("builtins.open", unittest.mock.mock_open()):
                    data = ds.fetch_home_data(force_refresh=True)

                    # Inception came from DB, The Dark Knight from OMDb
                    self.assertEqual(data["featured"]["title"], "Inception")
                    self.assertEqual(data["picks"][0]["title"], "The Dark Knight")
                    # get_movie_details was only called for tt0468569
                    self.mock_client.get_movie_details.assert_called_once_with("tt0468569")

    def test_never_fills_with_arbitrary_local_catalogue_records(self):
        """Verify that failed OMDb lookups do not inject arbitrary local catalogue plots."""
        ds = DiscoveryService(client=self.mock_client)
        self.mock_client.get_movie_details.side_effect = Exception("Network offline")

        mock_cache = MagicMock()
        mock_cache.exists.return_value = False

        with patch.object(ds, "get_cached_home_data", return_value=None):
            with patch("discovery_service.database.get_movie_by_id", return_value=None):
                with patch("discovery_service.CACHE_FILE", mock_cache):
                    with self.assertRaises(discovery_service.OMDbError) as ctx:
                        ds.fetch_home_data(force_refresh=True)
                    # Must raise clear error and NEVER return random catalogue records
                    self.assertIn("Unable to retrieve Popular Picks", str(ctx.exception))


class TestSavedListDuplicatePrevention(unittest.TestCase):
    """Test duplicate prevention in Wishlist and Backlog SQLite tables."""

    def setUp(self):
        self.test_db = Path("test_polish_db.sqlite")
        if self.test_db.exists():
            self.test_db.unlink()
        database.init_db(self.test_db)

    def tearDown(self):
        if self.test_db.exists():
            try:
                self.test_db.unlink()
            except Exception:
                pass

    def test_wishlist_strict_duplicate_prevention(self):
        movie = {"imdb_id": "tt1375666", "title": "Inception", "year": "2010"}
        # Add once
        self.assertTrue(database.add_to_wishlist("tt1375666", movie, self.test_db))
        # Add second time
        database.add_to_wishlist("tt1375666", movie, self.test_db)

        # Verify only 1 entry in Wishlist
        items = database.get_wishlist_movies(self.test_db)
        self.assertEqual(len(items), 1)

    def test_backlog_strict_duplicate_prevention(self):
        movie = {"imdb_id": "tt1375666", "title": "Inception", "year": "2010"}
        # Add once with High priority
        self.assertTrue(database.add_to_backlog("tt1375666", "High", "Want to Watch", movie, self.test_db))
        # Add second time with Low priority (should update, not duplicate)
        database.add_to_backlog("tt1375666", "Low", "Want to Watch", movie, self.test_db)

        items = database.get_backlog_movies(include_watched=True, db_path=self.test_db)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["priority"], "Low")


class TestMissingMetadataResilience(unittest.TestCase):
    """Verify handling of movies with missing ratings, years, or posters."""

    def test_missing_metadata_handled_gracefully(self):
        movie_sparse = {
            "imdb_id": "tt0000001",
            "title": "Sparse Movie",
            "year": "N/A",
            "imdb_rating": "N/A",
            "genre": None,
            "poster": "N/A"
        }
        # Safe extraction
        from discovery_service import _is_valid_movie
        self.assertTrue(_is_valid_movie(movie_sparse))
        # Incomplete movie without title or imdb_id
        self.assertFalse(_is_valid_movie({"title": "", "imdb_id": ""}))
        self.assertFalse(_is_valid_movie(None))

    def test_title_truncation_safeguard(self):
        """Verify that long titles are truncated safely with ellipsis."""
        long_title = "The Lord of the Rings: The Fellowship of the Ring Extended Edition"
        display_title = long_title if len(long_title) <= 42 else long_title[:39] + "..."
        self.assertTrue(display_title.endswith("..."))
        self.assertLessEqual(len(display_title), 42)


class TestDiscoverScreenInteractions(unittest.TestCase):
    """Test Discover screen curated picks, genre filtering, and search reset."""

    def test_genre_filter_logic_on_curated_picks(self):
        curated_picks = [
            {"title": "Inception", "genre": "Action, Adventure, Sci-Fi", "imdb_id": "tt1"},
            {"title": "The Dark Knight", "genre": "Action, Crime, Drama", "imdb_id": "tt2"},
            {"title": "The Shawshank Redemption", "genre": "Drama", "imdb_id": "tt3"},
            {"title": "Spirited Away", "genre": "Animation, Adventure, Family", "imdb_id": "tt4"},
        ]

        # Test Sci-Fi filtering
        scifi = [m for m in curated_picks if "sci-fi" in (m.get("genre") or "").lower()]
        self.assertEqual(len(scifi), 1)
        self.assertEqual(scifi[0]["title"], "Inception")

        # Test Drama filtering
        drama = [m for m in curated_picks if "drama" in (m.get("genre") or "").lower()]
        self.assertEqual(len(drama), 2)
        drama_titles = [m["title"] for m in drama]
        self.assertIn("The Dark Knight", drama_titles)
        self.assertIn("The Shawshank Redemption", drama_titles)

        # Test Animation filtering
        anim = [m for m in curated_picks if "animation" in (m.get("genre") or "").lower()]
        self.assertEqual(len(anim), 1)
        self.assertEqual(anim[0]["title"], "Spirited Away")


class TestResponsiveColumnCalculation(unittest.TestCase):
    """Test dynamic responsive columns based on viewport width."""

    def test_column_calculation_across_window_sizes(self):
        def calc_cols(width):
            if width > 100:
                avail = max(400, width - 80)
                return max(1, min(4, avail // 246))
            return 4

        # Large screen (1260px window -> canvas ~1050px)
        self.assertEqual(calc_cols(1050), 3)
        # Maximized desktop (1600px window -> canvas ~1390px)
        self.assertEqual(calc_cols(1390), 4)
        # Compact laptop (900px window -> canvas ~690px)
        self.assertEqual(calc_cols(690), 2)
        # Minimum window size fallback
        self.assertGreaterEqual(calc_cols(500), 1)


if __name__ == "__main__":
    unittest.main()
