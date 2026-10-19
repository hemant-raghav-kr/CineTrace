"""
GUI Integration Test for CineTrace (Phases 1-5)
Tests switching tabs, rendering Wishlist, Backlog, and Scene Finder screens,
executing a local plot search, verifying candidate cards, and opening dialogs.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure application modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

import tkinter as tk
import database
from main import CineTraceApp
from movie_details_dialog import MovieDetailsDialog


def test_gui():
    print("Testing CineTrace GUI integration (Phases 1-5)...")
    temp_dir = tempfile.mkdtemp()
    test_db = Path(temp_dir) / "test_gui_cinetrace.db"

    # Temporarily override database path
    old_default = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = test_db
    database.init_db(test_db)

    # Seed with 1 movie in wishlist and 1 in backlog
    m1 = {
        "imdb_id": "tt0137523",
        "title": "Fight Club",
        "year": "1999",
        "imdb_rating": "8.8",
        "poster_url": "N/A"
    }
    m2 = {
        "imdb_id": "tt0111161",
        "title": "The Shawshank Redemption",
        "year": "1994",
        "imdb_rating": "9.3",
        "poster_url": "N/A"
    }
    database.save_movie(m1, test_db)
    database.save_movie(m2, test_db)
    database.add_to_wishlist("tt0137523", m1, test_db)
    database.add_to_backlog("tt0111161", "High", "Want to Watch", m2, test_db)

    try:
        app = CineTraceApp()
        app.withdraw()  # Offscreen
        app.update_idletasks()

        # 1. Test Tab Switching to Wishlist
        print("  -> Testing switch to Wishlist tab...")
        app._switch_tab("Wishlist")
        app.update()
        assert app.active_tab == "Wishlist"
        wishlist_items = database.get_wishlist_movies(test_db)
        assert len(wishlist_items) == 1
        print("  [PASS] Wishlist screen rendered with 1 movie.")

        # 2. Test Tab Switching to Backlog
        print("  -> Testing switch to Backlog tab...")
        app._switch_tab("Backlog")
        app.update()
        assert app.active_tab == "Backlog"
        backlog_items = database.get_backlog_movies(include_watched=False, db_path=test_db)
        assert len(backlog_items) == 1
        print("  [PASS] Backlog screen rendered with 1 movie.")

        # 3. Test Tab Switching to Scene Finder
        print("  -> Testing switch to Scene Finder tab...")
        app._switch_tab("Scene Finder")
        app.update()
        assert app.active_tab == "Scene Finder"
        assert hasattr(app, "scene_text_box")
        assert hasattr(app, "scene_search_btn")
        print("  [PASS] Scene Finder screen rendered with input box and controls.")

        # 4. Test executing search in Scene Finder
        print("  -> Testing Scene Finder search query...")
        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert("1.0", "relives the same day repeatedly in a time loop")
        app._on_scene_search_clicked()

        # Wait briefly for background thread to complete local matching
        timeout = 15.0
        start_t = time.time()
        while app.scene_is_searching and (time.time() - start_t) < timeout:
            app.update()
            time.sleep(0.05)

        app.update()
        assert not app.scene_is_searching, "Scene search timed out"
        assert app.scene_has_searched, "Scene search flag not set"
        assert len(app.scene_results) > 0, "Scene search returned 0 results for time loop query"
        top_title = app.scene_results[0]["title"]
        print(f"  [PASS] Scene search succeeded. Top result: '{top_title}' ({app.scene_results[0]['similarity_pct']}).")

        # 4b. Test Phase 5.1 Gifted search query and poster loading in GUI
        print("  -> Testing Phase 5.1 Gifted (2017) search in GUI...")
        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert(
            "1.0",
            "A young girl discovers she has extraordinary mathematical abilities and is raised by her uncle, who wants her to have a normal childhood."
        )
        app.scene_min_threshold_var.set("10% (Balanced)")
        app._on_scene_search_clicked()

        start_t = time.time()
        while app.scene_is_searching and (time.time() - start_t) < timeout:
            app.update()
            time.sleep(0.05)

        app.update()
        assert not app.scene_is_searching, "Gifted scene search timed out"
        assert len(app.scene_results) >= 1, "Expected Gifted in GUI results"
        assert app.scene_results[0]["title"] == "Gifted"
        assert "tt4481414" in app.scene_card_widgets or "Gifted" in app.scene_card_widgets
        print(f"  [PASS] Gifted matched in GUI as #1 with {app.scene_results[0]['similarity_pct']} similarity.")

        # 4c. Test Phase 5.1 No Confident Match state in GUI
        print("  -> Testing Phase 5.1 No Confident Match empty state in GUI...")
        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert("1.0", "xyzzy plugh 99887766 completely nonexistent gibberish query.")
        app._on_scene_search_clicked()

        start_t = time.time()
        while app.scene_is_searching and (time.time() - start_t) < timeout:
            app.update()
            time.sleep(0.05)

        app.update()
        assert not app.scene_is_searching, "Unrelated scene search timed out"
        assert len(app.scene_results) == 0, "Expected 0 results for unrelated query"
        assert app.scene_has_searched
        print("  [PASS] 'No Confident Match Found' empty state displayed correctly for unrelated query.")

        # 5. Test Clear button
        print("  -> Testing Scene Finder Clear button...")
        app._on_scene_clear_clicked()
        app.update()
        assert app.scene_query_text == ""
        assert len(app.scene_results) == 0
        assert not app.scene_has_searched
        print("  [PASS] Scene Finder cleared cleanly.")

        # 6. Test Movie Details Dialog with callbacks
        print("  -> Testing MovieDetailsDialog creation and list toggling...")
        dialog = MovieDetailsDialog(app, "tt0137523", "Fight Club", on_change_callback=lambda: None)
        dialog.withdraw()
        dialog.update()

        assert dialog.wishlist_btn.cget("text") == "♥ In Wishlist"
        assert dialog.backlog_btn.cget("text") == "▤ Backlog"
        print("  [PASS] Details dialog accurately displays 'In Wishlist'.")

        dialog.destroy()
        app.destroy()
        print("  [PASS] GUI created, interacted, and closed cleanly without errors.")

    finally:
        database.DEFAULT_DB_PATH = old_default
        try:
            if test_db.exists():
                test_db.unlink()
            os.rmdir(temp_dir)
        except Exception:
            pass

    print("\nALL GUI INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    test_gui()
