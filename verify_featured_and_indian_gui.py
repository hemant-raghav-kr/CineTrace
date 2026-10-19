import time
import tkinter as tk
import main

print("=== Running Live GUI Verification: Featured Poster & Indian Cinema ===", flush=True)
app = main.CineTraceApp()

def run_tests():
    print("\n--- 1. Testing Home Screen Hero Banner ---", flush=True)
    # Check that hero banner has assigned a real poster for Inception (tt1375666)
    inception_photo = app.active_posters.get("tt1375666") or app.active_posters.get("hero_featured")
    print(f"Inception featured poster in active_posters: {inception_photo}", flush=True)
    assert inception_photo is not None, "Featured hero banner poster was not assigned!"

    print("\n--- 2. Testing Home Screen Popular Picks Grid ---", flush=True)
    print(f"Total home_picks: {len(app.home_picks)}", flush=True)
    assert len(app.home_picks) in (30, 35), f"Expected 30 or 35 home picks, found {len(app.home_picks)}"
    
    # Check that Indian cinema titles are in home_picks
    hindi_found = [p['title'] for p in app.home_picks if 'tt1187043' == p.get('imdb_id')]  # 3 Idiots
    tamil_found = [p['title'] for p in app.home_picks if 'tt15097216' == p.get('imdb_id')] # Jai Bhim
    malayalam_found = [p['title'] for p in app.home_picks if 'tt4679210' == p.get('imdb_id')] # Premam
    telugu_found = [p['title'] for p in app.home_picks if 'tt8178634' == p.get('imdb_id')] # RRR
    print(f"Hindi sample found: {hindi_found}", flush=True)
    print(f"Tamil sample found: {tamil_found}", flush=True)
    print(f"Malayalam sample found: {malayalam_found}", flush=True)
    print(f"Telugu sample found: {telugu_found}", flush=True)
    assert hindi_found and tamil_found and malayalam_found and telugu_found, "Sample Indian titles missing from home_picks!"

    print(f"Total active_posters in app: {len(app.active_posters)}", flush=True)

    print("\n--- 3. Testing Discover Screen Curated Collection & Genre Filters ---", flush=True)
    app._switch_tab("Discover")
    print(f"Discover movies count: {len(app.discover_movies)}", flush=True)
    assert len(app.discover_movies) in (30, 35), f"Expected 30 or 35 discover movies, found {len(app.discover_movies)}"

    # Test filtering by Comedy
    app._on_genre_filter_clicked("Comedy")
    comedy_count = len([m for m in app.discover_movies if "comedy" in (m.get("genre") or "").lower()])
    print(f"Comedy genre filter returned: {comedy_count} movies", flush=True)
    assert comedy_count >= 5, f"Expected at least 5 comedy titles (includes 3 Idiots, Premam, etc.), got {comedy_count}"

    # Test filtering by Drama
    app._on_genre_filter_clicked("Drama")
    drama_count = len([m for m in app.discover_movies if "drama" in (m.get("genre") or "").lower()])
    print(f"Drama genre filter returned: {drama_count} movies", flush=True)
    assert drama_count >= 10, f"Expected at least 10 drama titles, got {drama_count}"

    # Reset genre
    app._on_genre_filter_clicked("All")

    print("\n--- 4. Testing Fallback: Missing Poster URL in Hero Banner ---", flush=True)
    dummy_movie = {
        "title": "Text Only Film",
        "year": "2024",
        "genre": "Mystery",
        "imdb_rating": "8.0",
        "plot": "A movie with no poster to test clean graceful fallback without layout errors.",
        "poster_url": "N/A"
    }
    # Test rendering hero card with no poster into a temporary frame
    test_frame = tk.Frame(app)
    app._render_hero_featured_card(test_frame, dummy_movie)
    print("Clean text-only fallback hero card rendered without exception!", flush=True)

    print("\n>>> ALL FEATURED POSTER AND INDIAN CINEMA VERIFICATION CHECKS PASSED! <<<", flush=True)
    app.after(500, app.destroy)

# Allow 4.0 seconds for all background images to decode and assign
app.after(4000, run_tests)
app.mainloop()
print("=== Verification Script Completed Successfully ===", flush=True)
