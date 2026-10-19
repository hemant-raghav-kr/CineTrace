"""
CineTrace Phase 5.9 — Memory Lifecycle Profiler

Profiles the application across the 10 explicit lifecycle stages defined in Phase 5.9:
1. Process launch and first usable window.
2. Catalogue and embedding initialization.
3. Semantic model loaded.
4. Cross-encoder loaded.
5. First semantic search.
6. First Top-25 reranked search.
7. First Top-50 reranked search.
8. Twenty consecutive searches using varied queries.
9. Repeated navigation across all screens.
10. Closing dialogs and returning to Home.

Measures:
- Process Resident Set Size (RSS via psutil)
- Python heap allocations (current and peak via tracemalloc)
- Stage latency (ms / s)
- Retained memory vs temporary peak
"""

import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure CineTrace root is on path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import psutil
import config
import database
import semantic_service
from scene_matcher import SceneMatcher
import tkinter as tk
from main import CineTraceApp
from movie_details_dialog import MovieDetailsDialog
from semantic_dialog import SemanticDialog
from catalogue_stats_dialog import CatalogueStatsDialog

VARIED_QUERIES = [
    "A man wakes up to the same day and cannot escape the cycle",
    "Astronaut stranded on Mars struggles to survive using science",
    "Young math genius raised by her uncle who wants a normal childhood",
    "Corporate espionage through shared dreams inside subconscious",
    "Cybernetic assassin sent back in time to kill future resistance leader",
    "Patient in psychiatric ward leads rebellion against tyrannical nurse",
    "Lonely writer falls in love with an artificial intelligence operating system",
    "Two detectives track a serial killer staging the seven deadly sins",
    "Child psychologist works with a boy who sees dead people",
    "Musicians disguise as women to escape mobsters after witnessing massacre",
    "Dying tycoon whispers final word as journalist investigates life",
    "Nightclub owner in Casablanca must help former lover escape authorities",
    "Detective with acrophobia becomes obsessed with a friend's wife",
    "Screenwriter drawn into delusions of faded silent movie star",
    "Photographer in wheelchair suspects neighbor of murdering his wife",
    "Middleweight boxer's rage and jealousy destroys family relationships",
    "Captain journeys up Cambodian river to assassinate rogue colonel",
    "Soldiers sent behind enemy lines to rescue paratrooper after D-Day",
    "Oil prospector amasses fortune through ruthless ambition in California",
    "Harvard student creates social networking website amidst legal disputes"
]


def get_mem() -> Tuple[float, float, float]:
    """Return (RSS_MB, tracemalloc_current_MB, tracemalloc_peak_MB)."""
    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    cur, peak = tracemalloc.get_traced_memory()
    return round(rss, 2), round(cur / (1024 * 1024), 2), round(peak / (1024 * 1024), 2)


def run_stage_profiling(run_id: int = 1) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"RUNNING 10-STAGE MEMORY PROFILING (RUN {run_id})")
    print(f"{'=' * 80}")

    tracemalloc.start()
    t_start_proc = time.perf_counter()
    stage_results: Dict[str, Any] = {}

    def record_stage(stage_num: int, name: str, lat_sec: float, extra: str = ""):
        rss, cur_heap, peak_heap = get_mem()
        stage_key = f"stage_{stage_num:02d}_{name}"
        stage_results[stage_key] = {
            "stage_number": stage_num,
            "name": name,
            "latency_ms": round(lat_sec * 1000.0, 2),
            "rss_mb": rss,
            "heap_current_mb": cur_heap,
            "heap_peak_mb": peak_heap,
            "extra": extra
        }
        print(f"[{stage_num:02d}] {name:<38} | RSS: {rss:>8.1f} MB | Heap: {cur_heap:>7.1f} MB (Peak: {peak_heap:>7.1f} MB) | Lat: {lat_sec*1000:>7.1f} ms {extra}")

    # --- Stage 1: Process launch and first usable window ---
    t0 = time.perf_counter()
    app = CineTraceApp()
    app.withdraw()  # Offscreen
    app.update()
    t1 = time.perf_counter() - t0
    record_stage(1, "process_launch_first_window", t1)

    # --- Stage 2: Catalogue and embedding initialization ---
    t0 = time.perf_counter()
    while app.scene_dataset_status == "Loading dataset...":
        app.update()
        time.sleep(0.02)
    t2 = time.perf_counter() - t0
    record_stage(2, "catalogue_and_embeddings_init", t2, f"({app.scene_dataset_status})")

    # --- Stage 3: Semantic model loaded ---
    sem_svc = app.scene_matcher.semantic_service or semantic_service.get_semantic_service()
    t0 = time.perf_counter()
    bi_ok, bi_msg = sem_svc.load_model_sync(allow_download=False)
    t3 = time.perf_counter() - t0
    record_stage(3, "semantic_bi_encoder_loaded", t3, f"({sem_svc.model_name})")

    # --- Stage 4: Cross-encoder loaded ---
    ce_svc = semantic_service.get_cross_encoder_service()
    t0 = time.perf_counter()
    ce_ok, ce_msg = ce_svc.load_model_sync()
    t4 = time.perf_counter() - t0
    record_stage(4, "cross_encoder_loaded", t4, f"({config.CROSS_ENCODER_MODEL_NAME})")

    # --- Stage 5: First semantic search ---
    t0 = time.perf_counter()
    res_sem = app.scene_matcher.search(
        query=VARIED_QUERIES[0],
        mode="plot",
        top_k=12,
        min_threshold=0.10,
        rerank=False
    )
    t5 = time.perf_counter() - t0
    record_stage(5, "first_semantic_search", t5, f"({len(res_sem)} results)")

    # --- Stage 6: First Top-25 reranked search ---
    t0 = time.perf_counter()
    res_r25 = app.scene_matcher.search(
        query=VARIED_QUERIES[0],
        mode="hybrid",
        top_k=12,
        min_threshold=0.10,
        rerank=True,
        rerank_top_k=25
    )
    t6 = time.perf_counter() - t0
    record_stage(6, "first_top25_reranked_search", t6, f"({len(res_r25)} results)")

    # --- Stage 7: First Top-50 reranked search ---
    t0 = time.perf_counter()
    res_r50 = app.scene_matcher.search(
        query=VARIED_QUERIES[0],
        mode="hybrid",
        top_k=12,
        min_threshold=0.10,
        rerank=True,
        rerank_top_k=50
    )
    t7 = time.perf_counter() - t0
    record_stage(7, "first_top50_reranked_search", t7, f"({len(res_r50)} results)")

    # --- Stage 8: 20 consecutive searches using varied queries ---
    t0 = time.perf_counter()
    consecutive_rss = []
    for q_idx, q in enumerate(VARIED_QUERIES, 1):
        t_sub = time.perf_counter()
        # Alternate between Top-25 and Top-50 to simulate realistic heavy usage
        rk = 50 if (q_idx % 2 == 0) else 25
        _ = app.scene_matcher.search(
            query=q,
            mode="hybrid",
            top_k=12,
            min_threshold=0.08,
            rerank=True,
            rerank_top_k=rk
        )
        app.update()
        cur_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        consecutive_rss.append(cur_rss)
    t8 = time.perf_counter() - t0
    rss_delta_20 = consecutive_rss[-1] - consecutive_rss[0]
    record_stage(8, "twenty_consecutive_searches", t8, f"(Net RSS delta: {rss_delta_20:+.1f} MB)")

    # --- Stage 9: Repeated navigation across all screens ---
    t0 = time.perf_counter()
    screens = ["Home", "Discover", "Wishlist", "Backlog", "Scene Finder"]
    for cycle in range(5):
        for s in screens:
            app._switch_tab(s)
            app.update()
    t9 = time.perf_counter() - t0
    record_stage(9, "repeated_screen_navigation", t9, "(5 full cycles x 5 tabs = 25 switches)")

    # --- Stage 10: Closing dialogs and returning to Home ---
    t0 = time.perf_counter()
    # Open and close details dialog
    m_sample = {
        "imdb_id": "tt0137523",
        "title": "Fight Club",
        "year": "1999",
        "genre": "Drama",
        "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club.",
        "poster_url": "N/A",
        "imdb_rating": "8.8"
    }
    dlg1 = MovieDetailsDialog(app, "tt0137523", initial_title="Fight Club")
    dlg1.update()
    dlg1.destroy()
    app.update()

    # Open and close stats dialog
    dlg2 = CatalogueStatsDialog(app)
    dlg2.update()
    dlg2.destroy()
    app.update()

    # Return to Home tab
    app._switch_tab("Home")
    app.update()
    t10 = time.perf_counter() - t0
    record_stage(10, "closing_dialogs_return_home", t10)

    total_time = time.perf_counter() - t_start_proc
    current_heap, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_sec": round(total_time, 2),
        "peak_heap_mb": round(peak_heap / (1024 * 1024), 2),
        "final_rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
        "consecutive_search_rss_history": [round(r, 1) for r in consecutive_rss],
        "stages": stage_results
    }

    import image_loader
    image_loader.clear_image_caches()
    app.destroy()
    image_loader.clear_image_caches()
    return summary


def run_full_baseline_profile(runs: int = 1, output_path: str = "baseline_memory_lifecycle.json"):
    import subprocess
    # If runs > 1 and this is the orchestrator, spawn separate processes for each run
    # so every run measures a true fresh-process launch RSS and heap baseline.
    if runs > 1 and "--sub-run" not in sys.argv:
        all_runs = []
        for r in range(1, runs + 1):
            tmp_out = f"_tmp_mem_run_{r}.json"
            print(f"\nSpawning subprocess for clean run {r}/{runs}...")
            cmd = [sys.executable, str(BASE_DIR / "profile_memory_lifecycle.py"), "--sub-run", "--run-id", str(r), "--output", tmp_out]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"Error in subprocess run {r}")
                continue
            if os.path.exists(tmp_out):
                with open(tmp_out, "r", encoding="utf-8") as f:
                    all_runs.append(json.load(f))
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass

        if not all_runs:
            print("No runs succeeded.")
            return {}

        # Aggregate across runs
        stage_keys = list(all_runs[0]["stages"].keys())
        aggregated = {}
        for k in stage_keys:
            rss_vals = [r["stages"][k]["rss_mb"] for r in all_runs]
            lat_vals = [r["stages"][k]["latency_ms"] for r in all_runs]
            heap_vals = [r["stages"][k]["heap_current_mb"] for r in all_runs]
            aggregated[k] = {
                "name": all_runs[0]["stages"][k]["name"],
                "rss_mean_mb": round(sum(rss_vals) / len(rss_vals), 1),
                "rss_min_mb": round(min(rss_vals), 1),
                "rss_max_mb": round(max(rss_vals), 1),
                "latency_mean_ms": round(sum(lat_vals) / len(lat_vals), 1),
                "heap_mean_mb": round(sum(heap_vals) / len(heap_vals), 1)
            }

        report = {
            "profile_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs_count": len(all_runs),
            "stages_aggregate": aggregated,
            "runs": all_runs
        }

        out_file = BASE_DIR / output_path
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Done] Multi-run memory profile aggregated and saved to: {out_file}")
        return report

    # Single run (either directly or via sub-run)
    run_id = 1
    if "--run-id" in sys.argv:
        idx = sys.argv.index("--run-id")
        run_id = int(sys.argv[idx + 1])
    res = run_stage_profiling(run_id)

    if "--sub-run" in sys.argv:
        out_file = BASE_DIR / output_path
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        return res

    stage_keys = list(res["stages"].keys())
    aggregated = {}
    for k in stage_keys:
        st = res["stages"][k]
        aggregated[k] = {
            "name": st["name"],
            "rss_mean_mb": st["rss_mb"],
            "rss_min_mb": st["rss_mb"],
            "rss_max_mb": st["rss_mb"],
            "latency_mean_ms": st["latency_ms"],
            "heap_mean_mb": st["heap_current_mb"]
        }

    report = {
        "profile_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_count": 1,
        "stages_aggregate": aggregated,
        "runs": [res]
    }
    out_file = BASE_DIR / output_path
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Done] Single-run memory profile saved to: {out_file}")
    return report


if __name__ == "__main__":
    runs_arg = 2
    out_arg = "optimized_memory_lifecycle.json"
    if "--runs" in sys.argv:
        runs_arg = int(sys.argv[sys.argv.index("--runs") + 1])
    if "--output" in sys.argv:
        out_arg = sys.argv[sys.argv.index("--output") + 1]
    run_full_baseline_profile(runs=runs_arg, output_path=out_arg)

