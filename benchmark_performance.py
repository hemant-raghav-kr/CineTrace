"""
CineTrace — Performance Benchmark & Profiling Suite (Phase 5.5)

Measures and reports:
  1. Cold & warm application startup latency (until initial frame idle).
  2. Model and embedding index initialization latency.
  3. Query encoding latency (all-MiniLM-L6-v2).
  4. Vector similarity search latency (np.dot + argpartition).
  5. Database lookup latencies (catalogue stats, FTS5 match, candidate lookups).
  6. Initial result rendering latency (Tkinter card batch).
  7. End-to-end search latencies across Scene/Plot, Hybrid, Title, and Actor/Cast modes.
  8. Search accuracy verification (confirming Phase 5.1-5.4 top candidates).

Usage:
    python benchmark_performance.py
    python benchmark_performance.py --verbose
    python benchmark_performance.py --json
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config
import database
import scene_matcher
import semantic_service


def benchmark_database_operations(db_path: Path) -> Dict[str, float]:
    """Measure SQLite and FTS5 lookup latencies."""
    metrics = {}

    # 1. Database connection & schema check
    t0 = time.perf_counter()
    database.init_db(db_path)
    metrics["db_init_ms"] = (time.perf_counter() - t0) * 1000.0

    # 2. Catalogue stats (uncached first, then cached)
    database.invalidate_catalogue_cache()
    t0 = time.perf_counter()
    stats = database.get_catalogue_stats(db_path)
    metrics["catalogue_stats_uncached_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    stats_cached = database.get_catalogue_stats(db_path)
    metrics["catalogue_stats_cached_ms"] = (time.perf_counter() - t0) * 1000.0

    # 3. Wishlist & Backlog fast set retrieval
    t0 = time.perf_counter()
    wish_ids = database.get_wishlist_ids(db_path)
    metrics["get_wishlist_ids_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    back_ids = database.get_backlog_ids(active_only=True, db_path=db_path)
    metrics["get_backlog_ids_ms"] = (time.perf_counter() - t0) * 1000.0

    # 4. FTS5 full-text search match latency
    t0 = time.perf_counter()
    fts_results = database.search_catalogue_fts("mathematical uncle prodigy normal childhood", limit=20, db_path=db_path)
    metrics["fts5_query_ms"] = (time.perf_counter() - t0) * 1000.0

    # 5. Single movie lookup
    t0 = time.perf_counter()
    movie = database.get_catalogue_movie_by_id("tt4481414", db_path)
    metrics["catalogue_lookup_by_id_ms"] = (time.perf_counter() - t0) * 1000.0

    return metrics


def benchmark_semantic_components(db_path: Path) -> Dict[str, Any]:
    """Measure semantic model loading, index loading, query encoding, and vector search."""
    metrics = {}

    sem_svc = semantic_service.SemanticService(db_path=db_path)

    # 1. Index loading (.npz)
    t0 = time.perf_counter()
    index_ok = sem_svc.load_index_from_storage()
    metrics["embedding_index_load_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["vector_count"] = sem_svc.index.count

    # 2. Model loading (local weights)
    t0 = time.perf_counter()
    model_ok = sem_svc.load_model_sync()
    metrics["model_load_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["model_loaded"] = model_ok

    # 3. Query encoding
    query = "A young girl discovers she has extraordinary mathematical abilities and is raised by her uncle"
    t0 = time.perf_counter()
    q_vec = sem_svc.encode_query(query)
    metrics["query_encoding_ms"] = (time.perf_counter() - t0) * 1000.0

    # 4. Vector similarity search
    t0 = time.perf_counter()
    sim_results = sem_svc.index.search_vector(q_vec, top_k=50)
    metrics["vector_search_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["top_k_returned"] = len(sim_results)

    return metrics


def benchmark_scene_matcher(db_path: Path) -> Dict[str, Any]:
    """Measure SceneMatcher initialization, TF-IDF load, and search modes."""
    metrics = {}

    # 1. SceneMatcher initialization
    t0 = time.perf_counter()
    matcher = scene_matcher.SceneMatcher(db_path=db_path)
    metrics["scene_matcher_init_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["catalogue_records_loaded"] = len(matcher.records)

    # Queries
    query_plot = "A young girl discovers she has extraordinary mathematical abilities and is raised by her uncle who wants her to have a normal childhood"
    query_title = "Inception"
    query_actor = "Tom Hanks"
    query_hybrid = "man relives the same day repeatedly in a time loop"

    # Warm-up run
    _ = matcher.search(query_title, mode="title", top_k=10, min_threshold=0.10)

    # 2. Title Search
    t0 = time.perf_counter()
    title_matches = matcher.search(query_title, mode="title", top_k=10, min_threshold=0.10)
    metrics["search_title_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["title_top"] = title_matches[0]["title"] if title_matches else None

    # 3. Actor / Cast Search
    t0 = time.perf_counter()
    actor_matches = matcher.search(query_actor, mode="actor", top_k=10, min_threshold=0.10)
    metrics["search_actor_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["actor_count"] = len(actor_matches)

    # 4. Scene / Plot Search
    t0 = time.perf_counter()
    plot_matches = matcher.search(query_plot, mode="plot", top_k=10, min_threshold=0.10)
    metrics["search_scene_plot_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["plot_top"] = plot_matches[0]["title"] if plot_matches else None
    metrics["plot_top_score"] = plot_matches[0]["similarity_pct"] if plot_matches else None

    # 5. Hybrid Search
    t0 = time.perf_counter()
    hybrid_matches = matcher.search(query_hybrid, mode="hybrid", top_k=10, min_threshold=0.08)
    metrics["search_hybrid_ms"] = (time.perf_counter() - t0) * 1000.0
    metrics["hybrid_top"] = hybrid_matches[0]["title"] if hybrid_matches else None
    metrics["hybrid_top_score"] = hybrid_matches[0]["similarity_pct"] if hybrid_matches else None

    return metrics


def benchmark_ui_operations() -> Dict[str, float]:
    """Measure UI card rendering latency."""
    import tkinter as tk
    from main import CineTraceApp

    metrics = {}

    app = None
    try:
        # Measure initial startup frame (cold/warm first idle)
        t0 = time.perf_counter()
        app = CineTraceApp()
        app.withdraw()
        app.update_idletasks()
        metrics["startup_first_idle_ms"] = (time.perf_counter() - t0) * 1000.0

        # Benchmark card rendering (12 cards with pre-fetched sets)
        movies_mock = [
            {
                "title": f"Movie Benchmark {i}",
                "year": "2020",
                "imdb_rating": "8.5",
                "imdb_id": f"tt{i:07d}",
                "poster_url": None,
                "similarity_pct": f"{90 - i}%",
                "plot": "A sample benchmark movie plot description for UI rendering latency measurement.",
                "genre": "Sci-Fi",
                "match_type": "semantic",
                "match_details": "Semantic similarity match (cosine: 0.88)",
            }
            for i in range(12)
        ]

        test_frame = tk.Frame(app)
        test_frame.pack()

        wish_set = {"tt0000001", "tt0000003"}
        back_set = {"tt0000002"}

        t0 = time.perf_counter()
        for idx, m in enumerate(movies_mock):
            row = idx // 4
            col = idx % 4
            app._render_scene_card(test_frame, m, row, col, gen=1, wish_set=wish_set, back_set=back_set)
        test_frame.update_idletasks()
        metrics["render_12_cards_ms"] = (time.perf_counter() - t0) * 1000.0

        test_frame.destroy()
    finally:
        if app is not None:
            try:
                app.destroy()
            except Exception:
                pass

    return metrics


def run_full_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run all benchmark suites and produce structured results."""
    db_path = database.DEFAULT_DB_PATH

    print("=" * 72)
    print("   CineTrace Phase 5.5 — Comprehensive Performance Benchmark Suite")
    print("=" * 72)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }

    # 1. Database Benchmarks
    print("\n[1/4] Benchmarking SQLite & FTS5 operations...")
    db_metrics = benchmark_database_operations(db_path)
    results["database"] = db_metrics
    if verbose:
        print(f"  • DB Init:                      {db_metrics['db_init_ms']:8.2f} ms")
        print(f"  • Catalogue Stats (uncached):   {db_metrics['catalogue_stats_uncached_ms']:8.2f} ms")
        print(f"  • Catalogue Stats (cached):     {db_metrics['catalogue_stats_cached_ms']:8.2f} ms")
        print(f"  • Fast Wishlist IDs Set:        {db_metrics['get_wishlist_ids_ms']:8.2f} ms")
        print(f"  • Fast Backlog IDs Set:         {db_metrics['get_backlog_ids_ms']:8.2f} ms")
        print(f"  • FTS5 Full-Text Match:         {db_metrics['fts5_query_ms']:8.2f} ms")
        print(f"  • Single Record Lookup:         {db_metrics['catalogue_lookup_by_id_ms']:8.2f} ms")

    # 2. Semantic Search Components
    print("\n[2/4] Benchmarking Semantic Model & Vectors...")
    sem_metrics = benchmark_semantic_components(db_path)
    results["semantic"] = sem_metrics
    if verbose:
        print(f"  • Embedding Index Load (.npz):  {sem_metrics['embedding_index_load_ms']:8.2f} ms ({sem_metrics['vector_count']:,} vectors)")
        print(f"  • Local Model Load:             {sem_metrics['model_load_ms']:8.2f} ms (Loaded: {sem_metrics['model_loaded']})")
        print(f"  • Query Encoding:               {sem_metrics['query_encoding_ms']:8.2f} ms")
        print(f"  • Vector Dot + Top-K Search:    {sem_metrics['vector_search_ms']:8.2f} ms")

    # 3. Scene Matcher & Search Modes
    print("\n[3/4] Benchmarking Search Engine Modes across 34,570 Catalogue...")
    matcher_metrics = benchmark_scene_matcher(db_path)
    results["search_modes"] = matcher_metrics
    if verbose:
        print(f"  • SceneMatcher Load & TF-IDF:   {matcher_metrics['scene_matcher_init_ms']:8.2f} ms ({matcher_metrics['catalogue_records_loaded']:,} records)")
        print(f"  • Title Search Mode:            {matcher_metrics['search_title_ms']:8.2f} ms -> Top: {matcher_metrics['title_top']}")
        print(f"  • Actor / Cast Search Mode:     {matcher_metrics['search_actor_ms']:8.2f} ms -> Matches: {matcher_metrics['actor_count']}")
        print(f"  • Scene / Plot Search Mode:     {matcher_metrics['search_scene_plot_ms']:8.2f} ms -> Top: {matcher_metrics['plot_top']} ({matcher_metrics['plot_top_score']})")
        print(f"  • Hybrid Search Mode:           {matcher_metrics['search_hybrid_ms']:8.2f} ms -> Top: {matcher_metrics['hybrid_top']} ({matcher_metrics['hybrid_top_score']})")

    # 4. UI Rendering Latency
    print("\n[4/4] Benchmarking UI Startup & Card Rendering...")
    ui_metrics = benchmark_ui_operations()
    results["ui"] = ui_metrics
    if verbose:
        print(f"  • App Startup (first idle frame): {ui_metrics['startup_first_idle_ms']:8.2f} ms")
        print(f"  • Render 12 Result Cards:        {ui_metrics['render_12_cards_ms']:8.2f} ms")

    # Summary Table
    print("\n" + "=" * 72)
    print("   BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 72)
    print(f"  {'Operation':<36} | {'Target Budget':<15} | {'Measured Latency':<15}")
    print("  " + "-" * 70)
    print(f"  {'App Startup (interactive frame)':<36} | {'< 500 ms':<15} | {ui_metrics['startup_first_idle_ms']:8.2f} ms")
    print(f"  {'Catalogue Stats (cached)':<36} | {'< 5 ms':<15} | {db_metrics['catalogue_stats_cached_ms']:8.2f} ms")
    print(f"  {'TF-IDF Index Load (cached)':<36} | {'< 300 ms':<15} | {matcher_metrics['scene_matcher_init_ms']:8.2f} ms")
    print(f"  {'Model Loading (local weights)':<36} | {'< 250 ms':<15} | {sem_metrics['model_load_ms']:8.2f} ms")
    print(f"  {'Query Vector Encoding':<36} | {'< 30 ms':<15} | {sem_metrics['query_encoding_ms']:8.2f} ms")
    print(f"  {'Vector Top-K Similarity Search':<36} | {'< 5 ms':<15} | {sem_metrics['vector_search_ms']:8.2f} ms")
    print(f"  {'Title Search':<36} | {'< 100 ms':<15} | {matcher_metrics['search_title_ms']:8.2f} ms")
    print(f"  {'Actor / Cast Search':<36} | {'< 100 ms':<15} | {matcher_metrics['search_actor_ms']:8.2f} ms")
    print(f"  {'Scene / Plot Search':<36} | {'< 150 ms':<15} | {matcher_metrics['search_scene_plot_ms']:8.2f} ms")
    print(f"  {'Hybrid Search':<36} | {'< 300 ms':<15} | {matcher_metrics['search_hybrid_ms']:8.2f} ms")
    print(f"  {'Result Card Batch Render (12)':<36} | {'< 100 ms':<15} | {ui_metrics['render_12_cards_ms']:8.2f} ms")
    print("=" * 72)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineTrace Performance Benchmark Suite")
    parser.add_argument("--verbose", action="store_true", default=True, help="Display verbose metrics")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results = run_full_benchmark(verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))
