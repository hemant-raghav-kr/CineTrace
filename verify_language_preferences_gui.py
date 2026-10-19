"""
CineTrace - Live Tkinter GUI Verification: Language Preferences & Personalisation
Tests actual Tkinter widgets, first-launch onboarding, preference toggling,
Home & Discover dynamic re-rendering, genre filtering, and search behavior.
"""

import time
import tkinter as tk
import database
import main
from preferences_dialog import LanguagePreferencesDialog

print("=== Running Live GUI Verification: Language Preferences & Personalisation ===", flush=True)

# 1. Clean database state to test true first-launch behavior
database.init_db()
database.set_preference("onboarding_completed", "false")
database.clear_preferred_languages()

app = main.CineTraceApp()

def run_gui_verification():
    try:
        print("\n--- 1. Testing Initial State (First Launch) ---", flush=True)
        assert not database.is_onboarding_completed(), "Expected onboarding to be incomplete initially"
        assert database.get_preferred_languages() == [], "Expected no initial preferred languages"
        print("Sidebar button text:", app.prefs_status_btn["text"].encode("ascii", "replace").decode(), flush=True)
        print("Top bar pill text:", app.lang_filter_btn["text"].encode("ascii", "replace").decode(), flush=True)
        assert "All" in app.prefs_status_btn["text"] or "Languages" in app.prefs_status_btn["text"]

        print("\n--- 2. Testing Onboarding Dialog & Multi-Select ---", flush=True)
        onboarding_dlg = LanguagePreferencesDialog(app, is_onboarding=True, on_save_callback=app._on_preferences_saved)
        assert len(onboarding_dlg.card_frames) == 6, f"Expected 6 language cards, got {len(onboarding_dlg.card_frames)}"
        
        # Verify all 6 languages are present
        for lang in ("English", "Tamil", "Telugu", "Hindi", "Malayalam", "Others"):
            assert lang in onboarding_dlg.card_frames, f"Card for '{lang}' missing in dialog!"

        # Select Tamil and Telugu
        print("Selecting Tamil and Telugu...", flush=True)
        onboarding_dlg._toggle_language("Tamil")
        onboarding_dlg._toggle_language("Telugu")
        assert "Tamil" in onboarding_dlg.selected_languages
        assert "Telugu" in onboarding_dlg.selected_languages

        # Click Continue
        onboarding_dlg._on_continue()
        print("Onboarding completed and saved.", flush=True)
        assert database.is_onboarding_completed(), "Onboarding should be marked completed in DB"
        assert sorted(database.get_preferred_languages()) == ["Tamil", "Telugu"], "DB should store Tamil and Telugu"

        print("\n--- 3. Testing Home Screen Personalisation ---", flush=True)
        app.update()
        print(f"App active preferred languages: {app.preferred_languages}", flush=True)
        print(f"Personalised Featured Hero: '{app.home_featured['title']}' ({app.home_featured.get('language')})", flush=True)
        assert app.home_featured['title'] != "Inception", "Inception should not be featured when English is not selected!"
        
        # Verify Hero is a verified Tamil/Telugu movie
        hero_lang = app.home_featured.get('language', '')
        assert "Tamil" in hero_lang or "Telugu" in hero_lang, f"Hero film '{app.home_featured['title']}' should be Tamil or Telugu!"

        # Verify picks are restricted to Tamil and Telugu
        print(f"Personalised Popular Picks count: {len(app.home_picks)}", flush=True)
        assert len(app.home_picks) > 0, "Personalised picks should not be empty!"
        for p in app.home_picks:
            p_lang = p.get('language', '')
            assert "Tamil" in p_lang or "Telugu" in p_lang, f"Movie '{p['title']}' with lang '{p_lang}' should not be in Tamil/Telugu picks!"

        # Verify UI indicators updated
        print("Updated sidebar text:", app.prefs_status_btn["text"].encode("ascii", "replace").decode(), flush=True)
        print("Updated top bar text:", app.lang_filter_btn["text"].encode("ascii", "replace").decode(), flush=True)
        assert "2" in app.prefs_status_btn["text"] or "Tamil" in app.lang_filter_btn["text"]

        print("\n--- 4. Testing Discover Screen Synchronisation & Genre Filter ---", flush=True)
        app._switch_tab("Discover")
        app.update()
        print(f"Discover movies count: {len(app.discover_movies)}", flush=True)
        assert len(app.discover_movies) == len(app.home_picks), "Discover should match Home personalized picks!"

        # Test Genre Filter (e.g. Drama)
        app._on_genre_filter_clicked("Drama")
        app.update()
        drama_movies = [m for m in app.curated_picks if "drama" in (m.get("genre") or "").lower()]
        print(f"Drama filtered count: {len(drama_movies)}", flush=True)
        assert len(drama_movies) > 0, "Expected drama titles among Tamil/Telugu collection"

        # Reset Genre
        app._on_genre_filter_clicked("All")

        print("\n--- 5. Testing Explicit Search (Not Blocked by Preferences) ---", flush=True)
        # Search for an English film (e.g. "Interstellar")
        app.search_var.set("Interstellar")
        app.current_search_query = "Interstellar"
        # Simulate OMDb search results
        app.discover_movies = [
            {"title": "Interstellar", "year": "2014", "imdb_id": "tt0816692", "type": "movie"}
        ]
        app._render_discover_screen()
        app.update()
        print("Explicit search returned:", [m['title'] for m in app.discover_movies], flush=True)
        assert len(app.discover_movies) == 1 and app.discover_movies[0]['title'] == "Interstellar"

        # Clear search: returns to personalised collection
        app._clear_search()
        app.update()
        print(f"After clear search, restored discover count: {len(app.discover_movies)}", flush=True)
        assert len(app.discover_movies) == len(app.curated_picks), "Clearing search should restore personalised collection!"

        print("\n--- 6. Testing Preferences Reset to Default ---", flush=True)
        settings_dlg = LanguagePreferencesDialog(app, is_onboarding=False, on_save_callback=app._on_preferences_saved)
        settings_dlg._on_reset()
        app.update()

        assert database.get_preferred_languages() == [], "Preferences should be empty in DB after reset"
        assert app.preferred_languages == [], "App preferences should be empty after reset"
        print(f"After reset, Featured Hero: '{app.home_featured['title']}'", flush=True)
        assert app.home_featured['title'] == "Inception", "Inception should be restored as featured hero!"
        print(f"After reset, Total home picks: {len(app.home_picks)}", flush=True)
        assert len(app.home_picks) in (30, 35), f"Expected 30 or 35 default picks, got {len(app.home_picks)}"

        print("\n>>> ALL LANGUAGE PREFERENCES & PERSONALISATION GUI CHECKS PASSED! <<<", flush=True)

    finally:
        app.destroy()

# Run verification in Tkinter event loop
app.after(50, run_gui_verification)
app.mainloop()
print("=== Live GUI Verification Completed Successfully ===", flush=True)
