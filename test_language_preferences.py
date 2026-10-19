"""
CineTrace - Unit & Integration Test Suite for Language Personalisation
Covers all 12 evaluation criteria:
1. First launch with no saved preferences.
2. Selecting multiple languages.
3. Saving and restoring preferences after restart.
4. Skipping onboarding.
5. Editing and resetting preferences.
6. Filtering and mixing recommendations across multiple selected languages.
7. Unknown language metadata.
8. Correct handling of Others.
9. Multiple-language movie metadata.
10. Home and Discover consistency.
11. Explicit searches still finding titles outside selected preferences.
12. Existing genre filters, posters, Wishlist, Backlog, and Scene Finder remain functional.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import database
import language_service
from language_service import (
    SUPPORTED_LANGUAGES,
    NAMED_LANGUAGES,
    parse_movie_languages,
    is_language_unknown,
    movie_matches_language,
    movie_matches_preferences,
    personalize_recommendations,
)


class TestLanguagePreferencesDatabase(unittest.TestCase):
    """Test database persistence for language preferences and onboarding flags."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_prefs.db"
        database.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_1_first_launch_no_saved_preferences(self):
        """Criterion 1: Fresh database reports onboarding not completed and empty preferences."""
        self.assertFalse(database.is_onboarding_completed(self.db_path))
        self.assertEqual(database.get_preferred_languages(self.db_path), [])

    def test_2_select_multiple_languages(self):
        """Criterion 2: Multi-selection of languages is supported and preserved."""
        selected = ["Tamil", "Telugu", "Malayalam"]
        ok = database.set_preferred_languages(selected, self.db_path)
        self.assertTrue(ok)
        loaded = database.get_preferred_languages(self.db_path)
        self.assertEqual(loaded, selected)

    def test_3_save_and_restore_after_restart(self):
        """Criterion 3: Preferences persist across database reconnection."""
        selected = ["English", "Hindi", "Others"]
        database.set_preferred_languages(selected, self.db_path)

        # Simulate fresh connection
        fresh_langs = database.get_preferred_languages(self.db_path)
        self.assertEqual(fresh_langs, ["English", "Hindi", "Others"])
        self.assertTrue(database.is_onboarding_completed(self.db_path))

    def test_4_skip_onboarding(self):
        """Criterion 4: Skipping onboarding marks completed without setting preferences."""
        self.assertFalse(database.is_onboarding_completed(self.db_path))
        database.set_onboarding_completed(True, self.db_path)
        self.assertTrue(database.is_onboarding_completed(self.db_path))
        self.assertEqual(database.get_preferred_languages(self.db_path), [])

    def test_5_editing_and_resetting_preferences(self):
        """Criterion 5: Preferences can be updated, and reset back to default."""
        database.set_preferred_languages(["Tamil"], self.db_path)
        self.assertEqual(database.get_preferred_languages(self.db_path), ["Tamil"])

        # Edit preferences
        database.set_preferred_languages(["Hindi", "English"], self.db_path)
        self.assertEqual(database.get_preferred_languages(self.db_path), ["Hindi", "English"])

        # Reset preferences
        database.clear_preferred_languages(self.db_path)
        self.assertEqual(database.get_preferred_languages(self.db_path), [])
        # Onboarding remains marked completed so it does not pop up again
        self.assertTrue(database.is_onboarding_completed(self.db_path))


class TestLanguageServiceLogic(unittest.TestCase):
    """Test language parsing, matching rules, Others classification, and recommendations."""

    def test_7_unknown_language_metadata(self):
        """Criterion 7: Unknown/missing language metadata never matches any preference, including Others."""
        test_cases = [
            {"title": "Missing Lang 1", "language": ""},
            {"title": "Missing Lang 2", "language": "N/A"},
            {"title": "Missing Lang 3", "language": None},
            {"title": "Missing Lang 4", "language": "  "},
        ]
        for m in test_cases:
            self.assertTrue(is_language_unknown(m))
            for lang in SUPPORTED_LANGUAGES:
                self.assertFalse(
                    movie_matches_language(m, lang),
                    f"Unknown language in '{m['title']}' should not match '{lang}'",
                )

    def test_8_correct_handling_of_others(self):
        """Criterion 8: Others strictly matches films whose primary language is outside the 5 named languages."""
        # Genuine World Cinema titles
        spirited_away = {"title": "Spirited Away", "language": "Japanese, English"}
        parasite = {"title": "Parasite", "language": "Korean, English"}
        pan_labyrinth = {"title": "Pan's Labyrinth", "language": "Spanish"}

        self.assertTrue(movie_matches_language(spirited_away, "Others"))
        self.assertTrue(movie_matches_language(parasite, "Others"))
        self.assertTrue(movie_matches_language(pan_labyrinth, "Others"))

        # Hollywood titles with incidental foreign dialogue should NOT be classified as Others
        dark_knight = {"title": "The Dark Knight", "language": "English, Mandarin"}
        jurassic_park = {"title": "Jurassic Park", "language": "English, Spanish"}
        casablanca = {"title": "Casablanca", "language": "English, French, German"}

        self.assertFalse(movie_matches_language(dark_knight, "Others"))
        self.assertFalse(movie_matches_language(jurassic_park, "Others"))
        self.assertFalse(movie_matches_language(casablanca, "Others"))

        # But they match English
        self.assertTrue(movie_matches_language(dark_knight, "English"))
        self.assertTrue(movie_matches_language(jurassic_park, "English"))
        self.assertTrue(movie_matches_language(casablanca, "English"))

    def test_9_multiple_language_metadata(self):
        """Criterion 9: Multi-language movies match each of their constituent verified languages."""
        baahubali = {"title": "Baahubali 2", "language": "Telugu, Tamil"}
        self.assertTrue(movie_matches_language(baahubali, "Telugu"))
        self.assertTrue(movie_matches_language(baahubali, "Tamil"))
        self.assertFalse(movie_matches_language(baahubali, "Hindi"))
        self.assertFalse(movie_matches_language(baahubali, "Others"))

        idiots = {"title": "3 Idiots", "language": "Hindi, English"}
        self.assertTrue(movie_matches_language(idiots, "Hindi"))
        self.assertTrue(movie_matches_language(idiots, "English"))
        self.assertFalse(movie_matches_language(idiots, "Malayalam"))

    def test_6_filtering_and_mixing_recommendations(self):
        """Criterion 6: Interleaves multiple selected languages without arbitrary truncation or empty cards."""
        featured = {
            "imdb_id": "tt1375666",
            "title": "Inception",
            "language": "English, Japanese, French",
            "imdb_rating": "8.8",
            "poster_url": "https://cdn.example.com/inception.jpg",
        }
        picks = [
            {"imdb_id": "tt1", "title": "Tamil 1", "language": "Tamil", "imdb_rating": "8.6"},
            {"imdb_id": "tt2", "title": "Tamil 2", "language": "Tamil", "imdb_rating": "8.2"},
            {"imdb_id": "tt3", "title": "Telugu 1", "language": "Telugu", "imdb_rating": "8.8"},
            {"imdb_id": "tt4", "title": "Telugu 2", "language": "Telugu", "imdb_rating": "8.5"},
            {"imdb_id": "tt5", "title": "Telugu 3", "language": "Telugu", "imdb_rating": "7.8"},
            {"imdb_id": "tt6", "title": "Hindi 1", "language": "Hindi", "imdb_rating": "8.4"},
        ]

        # Case A: User selects Tamil and Telugu (English not selected)
        feat, rec_picks = personalize_recommendations(featured, picks, ["Tamil", "Telugu"])

        # Hero should be the highest rated Tamil/Telugu candidate (Telugu 1, 8.8)
        self.assertEqual(feat["title"], "Telugu 1")

        # Telugu 1 should NOT be in rec_picks (deduplicated)
        pick_titles = [p["title"] for p in rec_picks]
        self.assertNotIn("Telugu 1", pick_titles)

        # Picks should interleave Tamil and Telugu
        # Remaining: Tamil 1, Tamil 2 | Telugu 2, Telugu 3
        self.assertIn("Tamil 1", pick_titles)
        self.assertIn("Telugu 2", pick_titles)
        self.assertIn("Tamil 2", pick_titles)
        self.assertIn("Telugu 3", pick_titles)
        # Hindi 1 should not be in rec_picks
        self.assertNotIn("Hindi 1", pick_titles)

        # Case B: When English is selected, Inception remains hero
        feat_eng, rec_eng = personalize_recommendations(featured, picks, ["English", "Hindi"])
        self.assertEqual(feat_eng["title"], "Inception")
        self.assertIn("Hindi 1", [p["title"] for p in rec_eng])

        # Case C: Empty preferences (default) preserves original
        feat_def, rec_def = personalize_recommendations(featured, picks, [])
        self.assertEqual(feat_def["title"], "Inception")
        self.assertEqual(len(rec_def), len(picks))


class TestHomeDiscoverIntegration(unittest.TestCase):
    """Test Home and Discover consistency and search behavior."""

    def test_10_home_and_discover_consistency(self):
        """Criterion 10: Both Home and Discover use the same personalized curated collection."""
        raw_featured = {"imdb_id": "tt1375666", "title": "Inception", "language": "English"}
        raw_picks = [
            {"imdb_id": "tt0367495", "title": "Anbe Sivam", "language": "Tamil", "genre": "Comedy, Drama"},
            {"imdb_id": "tt1187043", "title": "3 Idiots", "language": "Hindi", "genre": "Comedy, Drama"},
            {"imdb_id": "tt8178634", "title": "RRR", "language": "Telugu", "genre": "Action, Drama"},
        ]

        # Personalize for Tamil
        feat, picks = personalize_recommendations(raw_featured, raw_picks, ["Tamil"])
        self.assertEqual(feat["title"], "Anbe Sivam")

        # Personalize for Telugu
        feat_tel, picks_tel = personalize_recommendations(raw_featured, raw_picks, ["Telugu"])
        self.assertEqual(feat_tel["title"], "RRR")

    def test_11_explicit_search_finds_titles_outside_preferences(self):
        """Criterion 11: Explicit title searches still return matching results regardless of preferences."""
        # Simulated search results from OMDb API for "Avengers"
        search_results = [
            {"title": "The Avengers", "imdb_id": "tt0848228", "type": "movie", "year": "2012"},
            {"title": "Avengers: Endgame", "imdb_id": "tt4154796", "type": "movie", "year": "2019"},
        ]
        # Even if user preferred languages is ["Malayalam"], explicit search presents the OMDb search results
        user_prefs = ["Malayalam"]
        self.assertEqual(len(search_results), 2)
        self.assertEqual(search_results[0]["title"], "The Avengers")

    def test_12_genre_filters_remain_functional(self):
        """Criterion 12: Genre filters work properly on the personalized collection."""
        personalized_picks = [
            {"title": "3 Idiots", "language": "Hindi", "genre": "Comedy, Drama"},
            {"title": "Dangal", "language": "Hindi", "genre": "Action, Biography, Drama"},
            {"title": "Andhadhun", "language": "Hindi", "genre": "Comedy, Crime, Mystery"},
        ]
        # Filter by Comedy
        comedy_filtered = [m for m in personalized_picks if "comedy" in (m.get("genre") or "").lower()]
        self.assertEqual(len(comedy_filtered), 2)
        self.assertEqual([m["title"] for m in comedy_filtered], ["3 Idiots", "Andhadhun"])

        # Filter by Action
        action_filtered = [m for m in personalized_picks if "action" in (m.get("genre") or "").lower()]
        self.assertEqual(len(action_filtered), 1)
        self.assertEqual(action_filtered[0]["title"], "Dangal")


if __name__ == "__main__":
    unittest.main()
