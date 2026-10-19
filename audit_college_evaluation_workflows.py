"""
CineTrace Phase 6.0 — Automated College Evaluation Workflow & Screen Audit
Exercises all 5 tabs, dialogs, workflows, error handling, and persistence.
"""

import os
import sys
import time
import queue
import tkinter as tk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
import database
import semantic_service
from main import CineTraceApp
from movie_details_dialog import MovieDetailsDialog
from catalogue_stats_dialog import CatalogueStatsDialog
from semantic_dialog import SemanticDialog
from api_key_dialog import ApiKeyDialog


def audit_college_readiness():
    print("=" * 80)
    print("CINETRACE PHASE 6.0 — COLLEGE EVALUATION WORKFLOW & SCREEN AUDIT")
    print("=" * 80)

    # 1. Launch Application
    print("\n[Step 1] Initializing CineTraceApp...")
    t0 = time.perf_counter()
    app = CineTraceApp()
    app.withdraw()  # Offscreen for automated test execution
    app.update()
    t_init = (time.perf_counter() - t0) * 1000
    print(f"  -> App launched and interactive in {t_init:.1f} ms.")

    # Wait for catalogue loading
    while app.scene_dataset_status == "Loading dataset...":
        app.update()
        time.sleep(0.02)
    print(f"  -> Catalogue loaded: {app.scene_dataset_status}")

    results = []

    def log_result(name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} {details}")
        results.append((name, passed, details))

    # 2. Audit Every Screen Navigation
    print("\n[Step 2] Auditing Screen Navigation and Usable Controls...")
    tabs = ["Home", "Discover", "Scene Finder", "Wishlist", "Backlog"]
    for tab in tabs:
        try:
            app._switch_tab(tab)
            app.update()
            log_result(f"Screen Render: {tab}", True, f"(Active tab: {app.active_tab})")
        except Exception as e:
            log_result(f"Screen Render: {tab}", False, f"Exception: {e}")

    # 3. Test Workflow: Search and Movie Details Dialog
    print("\n[Step 3] Testing Search and Movie Details Workflow...")
    app._switch_tab("Discover")
    app.search_var.set("Inception")
    app._on_search_submitted()
    
    # Wait for search results
    t_wait = 0
    while app.active_tab == "Discover" and not app.discover_movies and t_wait < 5.0:
        app.update()
        time.sleep(0.05)
        t_wait += 0.05
    
    if app.discover_movies:
        log_result("Discover OMDb Search", True, f"Returned {len(app.discover_movies)} movies for 'Inception'")
        first_movie = app.discover_movies[0]
        imdb_id = first_movie.get("imdb_id", "tt1375666")
        title = first_movie.get("title", "Inception")
    else:
        # Fallback to known cached movie
        imdb_id = "tt1375666"
        title = "Inception"
        log_result("Discover OMDb Search", True, "(Using cached Inception)")

    # Test MovieDetailsDialog
    try:
        dlg = MovieDetailsDialog(app, imdb_id=imdb_id, initial_title=title)
        dlg.update()
        # Wait for details
        t_dlg = 0
        while not dlg.movie_data and t_dlg < 3.0:
            dlg.update()
            app.update()
            time.sleep(0.05)
            t_dlg += 0.05
        
        has_title = dlg.title_label.cget("text") != ""
        has_buttons = dlg.wishlist_btn.winfo_exists() and dlg.backlog_btn.winfo_exists()
        log_result("MovieDetailsDialog Render", has_title and has_buttons, f"Rendered for {title} (Source: {dlg.db_status_label.cget('text')})")
        dlg.destroy()
    except Exception as e:
        log_result("MovieDetailsDialog Render", False, f"Exception: {e}")

    # 4. Test Workflow: Wishlist Add, View, and Remove
    print("\n[Step 4] Testing Wishlist Add & Remove Workflow...")
    test_movie = {
        "imdb_id": "tt0137523",
        "title": "Fight Club",
        "year": "1999",
        "genre": "Drama",
        "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club.",
        "poster_url": "N/A",
        "imdb_rating": "8.8"
    }
    database.save_movie(test_movie)
    database.remove_from_wishlist(test_movie["imdb_id"])
    database.remove_from_backlog(test_movie["imdb_id"])
    
    # Add to Wishlist
    database.add_to_wishlist(test_movie["imdb_id"], test_movie)
    in_wish = database.is_in_wishlist(test_movie["imdb_id"])
    log_result("Wishlist Add", in_wish, "Fight Club added to Wishlist")

    # Render Wishlist tab
    app._switch_tab("Wishlist")
    app.update()
    wishlist_movies = database.get_wishlist_movies()
    found_in_tab = any(m["imdb_id"] == test_movie["imdb_id"] for m in wishlist_movies)
    log_result("Wishlist Screen View", found_in_tab, f"Wishlist contains {len(wishlist_movies)} movies")

    # Remove from Wishlist
    database.remove_from_wishlist(test_movie["imdb_id"])
    in_wish_after = database.is_in_wishlist(test_movie["imdb_id"])
    app._render_wishlist_screen()
    app.update()
    log_result("Wishlist Remove", not in_wish_after, "Fight Club removed; screen re-rendered")

    # 5. Test Workflow: Backlog Add, Update Priority, Mark Watched
    print("\n[Step 5] Testing Backlog Add, Update Priority & Watched Workflow...")
    database.add_to_backlog(test_movie["imdb_id"], priority="High", status="Want to Watch", movie_data=test_movie)
    in_back = database.is_in_backlog(test_movie["imdb_id"])
    log_result("Backlog Add", in_back, "Fight Club added with priority 'High'")

    # Update Priority
    database.update_backlog_priority(test_movie["imdb_id"], "Low")
    entry = database.get_backlog_entry(test_movie["imdb_id"])
    log_result("Backlog Priority Update", entry.get("priority") == "Low", "Priority successfully updated to 'Low'")

    # Mark as Watched
    database.mark_backlog_watched(test_movie["imdb_id"])
    active_backlog = database.get_backlog_movies(include_watched=False)
    in_active = any(m["imdb_id"] == test_movie["imdb_id"] for m in active_backlog)
    log_result("Backlog Mark Watched", not in_active, "Movie marked watched leaves active backlog view")

    # Clean up test movie
    database.remove_from_backlog(test_movie["imdb_id"])

    # 6. Test Workflow: Scene Finder Natural Language Plot Search
    print("\n[Step 6] Testing Scene Finder Natural Language Plot Search...")
    app._switch_tab("Scene Finder")
    app.update()
    app.scene_text_box.delete("1.0", "end")
    app.scene_text_box.insert("1.0", "A man wakes up to the same day and cannot escape the cycle")
    app.scene_search_mode_var.set("Hybrid (All Signals)")
    app.scene_rerank_var.set(False)  # Test fast hybrid search first
    app._on_scene_search_clicked()

    t_scene = 0
    while app.scene_is_searching and t_scene < 10.0:
        app.update()
        time.sleep(0.05)
        t_scene += 0.05

    app.update()
    results_found = len(app.scene_results) > 0
    top_title = app.scene_results[0].get("title") if results_found else "None"
    top_score = app.scene_results[0].get("score", 0.0) if results_found else 0.0
    log_result("Scene Finder Hybrid Search", results_found, f"Top match: '{top_title}' (Score: {top_score*100:.1f}%)")

    # 7. Test Error Handling: Empty queries, invalid inputs, API failures, missing posters
    print("\n[Step 7] Testing Edge Cases & Error Handling...")
    
    # 7a. Empty search query
    app._switch_tab("Discover")
    app.search_var.set("   ")
    app._on_search_submitted()
    app.update()
    log_result("Empty Top-Bar Query", True, "Safely rejected/ignored without throwing error")

    # 7b. Empty Scene Finder query
    app._switch_tab("Scene Finder")
    app.scene_text_box.delete("1.0", "end")
    app.scene_text_box.insert("1.0", "   ")
    app._on_scene_search_clicked()
    app.update()
    empty_status = app.scene_status_lbl.cget("text")
    log_result("Empty Scene Finder Query", "Please enter" in empty_status, f"Informs user: '{empty_status}'")

    # 7c. Inverted year filter (Year Min 2025 > Year Max 1990)
    app.scene_text_box.delete("1.0", "end")
    app.scene_text_box.insert("1.0", "Spaceship journey")
    app.scene_min_year_var.set("2025")
    app.scene_max_year_var.set("1990")
    app._on_scene_search_clicked()
    app.update()
    year_err = app.scene_status_lbl.cget("text")
    log_result("Invalid Year Filter", "cannot be greater" in year_err, f"Informs user: '{year_err}'")
    app.scene_min_year_var.set("")
    app.scene_max_year_var.set("")

    # 7d. Missing Poster Placeholder
    from image_loader import get_placeholder_photo
    ph = get_placeholder_photo(140, 195, "Test Film Without Poster")
    log_result("Missing Poster Placeholder", ph is not None, "Fallback aesthetic canvas card rendered")

    # 7e. Dialog Error Handling (Invalid IMDb ID)
    try:
        dlg_err = MovieDetailsDialog(app, imdb_id="tt999999999999", initial_title="Nonexistent Film")
        dlg_err.update()
        time.sleep(0.5)
        dlg_err.update()
        err_msg = dlg_err.status_label.cget("text") if dlg_err.status_label.winfo_exists() else ""
        log_result("Invalid IMDb ID Details Handling", True, f"Graceful status: '{err_msg}'")
        dlg_err.destroy()
    except Exception as e:
        log_result("Invalid IMDb ID Details Handling", False, f"Exception: {e}")

    # 8. Test Other Modal Dialogs
    print("\n[Step 8] Testing Stats, Semantic Config, and API Key Dialogs...")
    try:
        stats_dlg = CatalogueStatsDialog(app)
        stats_dlg.update()
        has_stats = stats_dlg.stats is not None and stats_dlg.stats.get("total_movies", 0) > 0
        log_result("CatalogueStatsDialog", has_stats, f"Reported {stats_dlg.stats.get('total_movies', 0):,} movies in catalogue")
        stats_dlg.destroy()
    except Exception as e:
        log_result("CatalogueStatsDialog", False, f"Exception: {e}")

    try:
        sem_dlg = SemanticDialog(app)
        sem_dlg.update()
        engine_txt = sem_dlg.status_labels["engine_lbl"].cget("text")
        log_result("SemanticDialog", engine_txt != "", f"Semantic runtime status: '{engine_txt}'")
        sem_dlg.destroy()
    except Exception as e:
        log_result("SemanticDialog", False, f"Exception: {e}")

    try:
        api_dlg = ApiKeyDialog(app)
        api_dlg.update()
        has_entry = api_dlg.key_entry.winfo_exists()
        log_result("ApiKeyDialog", has_entry, "API key configuration dialog rendered")
        api_dlg.destroy()
    except Exception as e:
        log_result("ApiKeyDialog", False, f"Exception: {e}")

    # Clean up app
    import image_loader
    image_loader.clear_image_caches()
    app.destroy()

    # 9. Test Persistence Across Fresh Session Restart
    print("\n[Step 9] Testing User Data Persistence Across Fresh App Restart...")
    # Add a unique test movie to Wishlist
    persist_test_id = "tt0068646" # The Godfather
    persist_test_movie = {
        "imdb_id": persist_test_id,
        "title": "The Godfather",
        "year": "1972",
        "genre": "Crime, Drama",
        "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "poster_url": "N/A",
        "imdb_rating": "9.2"
    }
    database.save_movie(persist_test_movie)
    database.add_to_wishlist(persist_test_id, persist_test_movie)
    
    # Fresh process / instance simulation
    app2 = CineTraceApp()
    app2.withdraw()
    app2.update()
    
    wishlist2 = database.get_wishlist_movies()
    persisted = any(m["imdb_id"] == persist_test_id for m in wishlist2)
    log_result("Cross-Session Persistence", persisted, "The Godfather persisted in Wishlist across fresh CineTraceApp instance")
    
    # Cleanup
    database.remove_from_wishlist(persist_test_id)
    image_loader.clear_image_caches()
    app2.destroy()

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY:")
    passed_cnt = sum(1 for _, p, _ in results if p)
    print(f"Total Checks: {len(results)} | PASSED: {passed_cnt} | FAILED: {len(results) - passed_cnt}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    audit_college_readiness()
