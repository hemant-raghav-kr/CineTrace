"""
test_discover_search.py
Automated test suite verifying CineTrace Discover search improvements:
- Multi-page pagination for broad franchise queries (max_pages > 1)
- Strict deduplication by IMDb ID (no duplicate cards)
- Partial failure resilience (failed additional page preserves successful results)
- Result limits respecting max_pages ceiling
- Franchise relevance ranking (canonical sequels & franchise titles ranked above parodies/reviews)
- Exact title search preservation
- Offline test execution with mocked OMDb API responses
"""

import unittest
from unittest.mock import MagicMock, patch
import requests

from omdb_client import (
    OMDbClient,
    rank_search_results,
    OMDbError,
    OMDbConnectionError,
)
import config


class TestFranchiseRanking(unittest.TestCase):
    """Test ranking logic for franchise titles, sequels, parodies, and recency."""

    def test_exact_match_ranked_highest(self):
        movies = [
            {"title": "The Avengers: Earth's Mightiest Heroes", "year": "2010", "imdb_id": "tt1", "type": "series"},
            {"title": "The Avengers", "year": "2012", "imdb_id": "tt2", "type": "movie"},
            {"title": "Avengers: Endgame", "year": "2019", "imdb_id": "tt3", "type": "movie"},
        ]
        ranked = rank_search_results(movies, "The Avengers")
        self.assertEqual(ranked[0]["imdb_id"], "tt2")

        # Query without "The"
        ranked2 = rank_search_results(movies, "Avengers")
        # "The Avengers" matches "the " + query exact rule
        self.assertEqual(ranked2[0]["imdb_id"], "tt2")

    def test_canonical_sequels_ranked_above_parodies_and_reviews(self):
        movies = [
            {"title": "Avengers: Grimm - Time Wars (Parody)", "year": "2018", "imdb_id": "tt_parody", "type": "movie"},
            {"title": "Avengers: Doomsday", "year": "2026", "imdb_id": "tt_doom", "type": "movie"},
            {"title": "The Avengers: Behind The Scenes Review", "year": "2020", "imdb_id": "tt_review", "type": "movie"},
            {"title": "Avengers: Endgame", "year": "2019", "imdb_id": "tt_endgame", "type": "movie"},
        ]
        ranked = rank_search_results(movies, "Avengers")
        top_two_ids = [m["imdb_id"] for m in ranked[:2]]
        self.assertIn("tt_doom", top_two_ids)
        self.assertIn("tt_endgame", top_two_ids)
        # Parody and review should be demoted to bottom
        bottom_two_ids = [m["imdb_id"] for m in ranked[2:]]
        self.assertIn("tt_parody", bottom_two_ids)
        self.assertIn("tt_review", bottom_two_ids)

    def test_hunger_games_franchise_ranking(self):
        movies = [
            {"title": "The Hunger Games Spoof", "year": "2013", "imdb_id": "tt_spoof", "type": "movie"},
            {"title": "The Hunger Games", "year": "2012", "imdb_id": "tt_hg1", "type": "movie"},
            {"title": "The Hunger Games: Catching Fire", "year": "2013", "imdb_id": "tt_hg2", "type": "movie"},
            {"title": "The Hunger Games: Sunrise on the Reaping", "year": "2026", "imdb_id": "tt_hg_prequel", "type": "movie"},
        ]
        ranked = rank_search_results(movies, "Hunger Games")
        self.assertEqual(ranked[0]["imdb_id"], "tt_hg1")  # Exact prefix "The Hunger Games"
        # Sequels with colon prefix should come next
        sequel_ids = [m["imdb_id"] for m in ranked[1:3]]
        self.assertIn("tt_hg_prequel", sequel_ids)
        self.assertIn("tt_hg2", sequel_ids)
        # Spoof is last
        self.assertEqual(ranked[-1]["imdb_id"], "tt_spoof")

    def test_movie_type_preference_over_series_and_games(self):
        movies = [
            {"title": "Avengers: United", "year": "2020", "imdb_id": "tt_game", "type": "game"},
            {"title": "Avengers: United", "year": "2020", "imdb_id": "tt_movie", "type": "movie"},
        ]
        ranked = rank_search_results(movies, "Avengers")
        self.assertEqual(ranked[0]["imdb_id"], "tt_movie")


class TestMultiPageSearch(unittest.TestCase):
    """Test multi-page pagination, deduplication, and resilience against partial failures."""

    def setUp(self):
        self.client = OMDbClient(api_key="mock_key_12345")

    @patch("omdb_client.requests.get")
    def test_single_page_search_when_max_pages_is_one(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Response": "True",
            "totalResults": "150",
            "Search": [
                {"Title": f"Movie {i}", "Year": "2020", "imdbID": f"tt{i}", "Type": "movie", "Poster": "N/A"}
                for i in range(10)
            ]
        }
        mock_get.return_value = mock_response

        res = self.client.search_movies("Test Movie", page=1, max_pages=1)
        self.assertEqual(len(res["movies"]), 10)
        self.assertEqual(mock_get.call_count, 1)

    @patch("omdb_client.requests.get")
    def test_multi_page_pagination_fetches_multiple_pages(self, mock_get):
        def side_effect(url, params, timeout):
            page = params.get("page", 1)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "Response": "True",
                "totalResults": "25",
                "Search": [
                    {"Title": f"Movie P{page}_{i}", "Year": "2020", "imdbID": f"tt_{page}_{i}", "Type": "movie", "Poster": "N/A"}
                    for i in range(10 if page < 3 else 5)
                ]
            }
            return mock_resp

        mock_get.side_effect = side_effect

        res = self.client.search_movies("Test Movie", max_pages=5)
        # Total results = 25 (3 pages: 10 + 10 + 5)
        self.assertEqual(res["total_results"], 25)
        self.assertEqual(res["total_pages"], 3)
        self.assertEqual(len(res["movies"]), 25)
        # Verify 3 requests were made (page 1, 2, 3)
        self.assertEqual(mock_get.call_count, 3)

    @patch("omdb_client.requests.get")
    def test_max_pages_ceiling_enforced(self, mock_get):
        def side_effect(url, params, timeout):
            page = params.get("page", 1)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "Response": "True",
                "totalResults": "200",  # 20 pages available
                "Search": [
                    {"Title": f"Movie P{page}_{i}", "Year": "2020", "imdbID": f"tt_{page}_{i}", "Type": "movie", "Poster": "N/A"}
                    for i in range(10)
                ]
            }
            return mock_resp

        mock_get.side_effect = side_effect

        # Limit to max_pages=3
        res = self.client.search_movies("Avengers", max_pages=3)
        self.assertEqual(len(res["movies"]), 30)
        self.assertEqual(mock_get.call_count, 3)

    @patch("omdb_client.requests.get")
    def test_strict_deduplication_by_imdb_id(self, mock_get):
        def side_effect(url, params, timeout):
            page = params.get("page", 1)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            # Page 1 and Page 2 both return tt_duplicate
            mock_resp.json.return_value = {
                "Response": "True",
                "totalResults": "20",
                "Search": [
                    {"Title": f"Duplicate Movie", "Year": "2020", "imdbID": "tt_duplicate", "Type": "movie", "Poster": "N/A"},
                    {"Title": f"Unique Movie P{page}", "Year": "2020", "imdbID": f"tt_unique_{page}", "Type": "movie", "Poster": "N/A"}
                ]
            }
            return mock_resp

        mock_get.side_effect = side_effect

        res = self.client.search_movies("Duplicate Test", max_pages=2)
        movie_ids = [m["imdb_id"] for m in res["movies"]]
        # tt_duplicate should only appear once
        self.assertEqual(movie_ids.count("tt_duplicate"), 1)
        self.assertEqual(len(movie_ids), 3)  # 1 duplicate + 2 unique

    @patch("omdb_client.requests.get")
    def test_partial_failure_preserves_retrieved_results(self, mock_get):
        def side_effect(url, params, timeout):
            page = params.get("page", 1)
            if page == 1:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "Response": "True",
                    "totalResults": "30",
                    "Search": [
                        {"Title": f"Page 1 Movie {i}", "Year": "2020", "imdbID": f"tt_p1_{i}", "Type": "movie", "Poster": "N/A"}
                        for i in range(10)
                    ]
                }
                return mock_resp
            elif page == 2:
                # Simulating a network timeout on page 2
                raise requests.exceptions.Timeout("Connection timed out on page 2")
            else:
                # Page 3 succeeds
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "Response": "True",
                    "totalResults": "30",
                    "Search": [
                        {"Title": f"Page 3 Movie {i}", "Year": "2020", "imdbID": f"tt_p3_{i}", "Type": "movie", "Poster": "N/A"}
                        for i in range(10)
                    ]
                }
                return mock_resp

        mock_get.side_effect = side_effect

        res = self.client.search_movies("Resilience Test", max_pages=3)
        # Successful results from page 1 (10) and page 3 (10) should be preserved
        self.assertEqual(len(res["movies"]), 20)
        retrieved_ids = {m["imdb_id"] for m in res["movies"]}
        self.assertIn("tt_p1_0", retrieved_ids)
        self.assertIn("tt_p3_0", retrieved_ids)

    @patch("omdb_client.requests.get")
    def test_exact_title_search_works_and_ranks_first(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Response": "True",
            "totalResults": "5",
            "Search": [
                {"Title": "Inception: The Shooting Script", "Year": "2010", "imdbID": "tt_script", "Type": "movie", "Poster": "N/A"},
                {"Title": "Inception", "Year": "2010", "imdbID": "tt1375666", "Type": "movie", "Poster": "N/A"},
                {"Title": "Inception Explained", "Year": "2014", "imdbID": "tt_explained", "Type": "movie", "Poster": "N/A"},
            ]
        }
        mock_get.return_value = mock_response

        res = self.client.search_movies("Inception", max_pages=1)
        self.assertEqual(res["movies"][0]["imdb_id"], "tt1375666")
        self.assertEqual(res["movies"][0]["title"], "Inception")


class TestConfigAndDefaults(unittest.TestCase):
    """Test configuration settings for Discover search pagination."""

    def test_default_max_pages_configured(self):
        self.assertTrue(hasattr(config, "OMDB_SEARCH_MAX_PAGES"))
        self.assertGreaterEqual(config.OMDB_SEARCH_MAX_PAGES, 10)


if __name__ == "__main__":
    unittest.main()
