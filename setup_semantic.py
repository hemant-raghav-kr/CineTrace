"""
CineTrace — Semantic Search Setup & Management CLI (Phase 5.4)
Provides explicit commands to inspect, download, index, benchmark, and clear
the local sentence-embedding model and dense vector index.

Usage:
    python setup_semantic.py --status
    python setup_semantic.py --download-model
    python setup_semantic.py --build-index [--batch-size 128] [--mock]
    python setup_semantic.py --benchmark [--mock]
    python setup_semantic.py --clear-index
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import config
import database
import semantic_service


def print_header(title: str):
    print("=" * 68)
    print(f" {title}")
    print("=" * 68)


def show_status():
    print_header("CineTrace Semantic Search Status (Phase 5.4)")

    svc = semantic_service.get_semantic_service()

    # Check local weights using robust layout inspection
    model_info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.DEFAULT_MODEL_NAME)
    weights_exist = model_info["found"]

    # If weights exist on disk and model is not yet loaded, load offline and verify provenance
    if weights_exist and not svc.is_model_loaded() and svc.is_available():
        svc.load_model_sync(allow_download=False)

    stats = database.get_catalogue_embedding_stats()

    print(f"Embedding Model:        {config.DEFAULT_MODEL_NAME}")
    print(f"Model Provenance:       Hugging Face (sentence-transformers/{config.DEFAULT_MODEL_NAME})")
    print(f"Model License:          Apache 2.0 (Open Source)")
    print(f"Approx. Download Size:  ~80 MB (Disk Footprint: ~90 MB weights + ~53 MB vectors)")
    print(f"Vector Dimension:       {config.SEMANTIC_DIMENSION} floats per plot")
    print(f"Model Cache Location:   {config.MODEL_CACHE_DIR.resolve()}")
    print(f"Embedding Cache File:   {config.EMBEDDING_INDEX_FILE.resolve()}")
    print("-" * 68)

    # Library check
    lib_ok = svc.is_available()
    print(f"sentence-transformers:  {'[INSTALLED]' if lib_ok else '[NOT INSTALLED] (pip install sentence-transformers)'}")

    # Weights check
    if weights_exist:
        rev = model_info.get("revision") or ""
        rev_str = f" (Revision: {rev[:12]}...)" if rev else ""
        size_mb = model_info.get("size_bytes", 0) / (1024 * 1024)
        print(f"Local Model Weights:    [DOWNLOADED & CACHED]{rev_str} ({size_mb:.1f} MB)")
    else:
        print(f"Local Model Weights:    [NOT DOWNLOADED]")

    # Runtime status
    print(f"Service Runtime Status: {svc.model_status.value.upper()}")

    # Provenance status
    prov_stat = stats.get("provenance_status", "unverified")
    if svc.is_model_loaded() and prov_stat in ("verified", "mock_verified"):
        rev = stats.get("model_revision", "") or getattr(svc.model, "revision", "")
        rev_tag = f" (rev: {rev[:10]}...)" if rev else ""
        print(f"Index Provenance:       [VERIFIED: Compatible with {stats.get('model_name')}{rev_tag}]")
    elif prov_stat == "incompatible":
        print(f"Index Provenance:       [INCOMPATIBLE: Vectors do not match active model! Run --build-index to fix]")
    elif not weights_exist:
        print(f"Index Provenance:       [UNVERIFIED: Model weights not found]")
    else:
        print(f"Index Provenance:       [{prov_stat.upper()}]")

    print("-" * 68)

    # Database & Index counts
    total_plots = stats["total_plots"]
    indexed_cnt = stats["indexed_count"]
    pending_cnt = stats["pending_count"]
    in_mem_cnt = svc.index.count
    pct = (indexed_cnt / max(1, total_plots)) * 100

    print(f"Catalogue Plots:        {total_plots:,} usable plot descriptions")
    print(f"Persisted Embeddings:   {indexed_cnt:,} / {total_plots:,} ({pct:.1f}%)")
    print(f"Pending Embeddings:     {pending_cnt:,} un-indexed or stale records")
    print(f"In-Memory Vector Index: {in_mem_cnt:,} active vectors")

    if svc.is_ready():
        print("\n-> FIRST-STAGE SEMANTIC SEARCH: ACTIVE & OPERATIONAL")
    else:
        print("\n-> FIRST-STAGE SEMANTIC SEARCH: FALLBACK MODE (Lexical TF-IDF + FTS5)")

    # Cross-Encoder Reranker Status (Phase 5.7)
    print("-" * 68)
    print(" Stage 2: Cross-Encoder Reranker (Phase 5.7)")
    print("-" * 68)
    ce_svc = semantic_service.get_cross_encoder_service()
    ce_info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.CROSS_ENCODER_MODEL_NAME)
    ce_weights_exist = ce_info["found"]

    if ce_weights_exist and not ce_svc.is_model_loaded() and ce_svc.is_available():
        ce_svc.load_model_sync(allow_download=False)

    print(f"Cross-Encoder Model:    {config.CROSS_ENCODER_MODEL_NAME}")
    print(f"Model Provenance:       Hugging Face ({config.CROSS_ENCODER_MODEL_NAME})")
    print(f"Model License:          Apache 2.0 (Open Source)")
    print(f"Max Context Length:     512 tokens (Head + Tail Passage Selection)")

    if ce_weights_exist:
        ce_rev = ce_info.get("revision") or ""
        ce_rev_str = f" (Revision: {ce_rev[:12]}...)" if ce_rev else ""
        ce_size_mb = ce_info.get("size_bytes", 0) / (1024 * 1024)
        print(f"Local Model Weights:    [DOWNLOADED & CACHED]{ce_rev_str} ({ce_size_mb:.1f} MB)")
    else:
        print(f"Local Model Weights:    [NOT DOWNLOADED] (Run --download-cross-encoder)")

    print(f"Reranker Runtime:       {ce_svc.status.value.upper()}")
    if ce_svc.is_ready():
        print(f"Reranker Status:        ACTIVE & OPERATIONAL (Default Top {config.CROSS_ENCODER_TOP_K} reranked)")
    else:
        print(f"Reranker Status:        DISABLED / INACTIVE (Clean first-stage fallback)")
    print("=" * 68)



def download_model():
    print_header("Download Semantic Embedding Model Weights")
    svc = semantic_service.get_semantic_service()

    if not svc.is_available():
        print("[ERROR] 'sentence-transformers' is not installed in the current Python environment.")
        print("Please install it first using:")
        print("    pip install sentence-transformers")
        sys.exit(1)

    print(f"Model: {config.DEFAULT_MODEL_NAME}")
    print(f"Target Directory: {config.MODEL_CACHE_DIR}")
    print("Downloading weights explicitly (approx. 80 MB)...")

    t0 = time.perf_counter()
    success, msg = svc.download_and_load_model_sync()
    dt = time.perf_counter() - t0

    if success:
        info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.DEFAULT_MODEL_NAME)
        rev = info.get("revision", "")
        rev_str = f" [Revision: {rev[:12]}...]" if rev else ""
        print(f"[SUCCESS] {msg}{rev_str} ({dt:.2f} seconds)")
        if svc.index.count > 0:
            print("Verifying embedding provenance for existing catalogue...")
            ok, vmsg = svc.verify_provenance()
            if ok:
                print(f"[PROVENANCE] {vmsg}")
    else:
        print(f"[ERROR] {msg}")
        sys.exit(1)


def download_cross_encoder():
    print_header("Download Cross-Encoder Reranker Weights (Phase 5.7)")
    ce_svc = semantic_service.get_cross_encoder_service()

    if not ce_svc.is_available():
        print("[ERROR] 'sentence-transformers' is not installed in the current Python environment.")
        print("Please install it first using:")
        print("    pip install sentence-transformers")
        sys.exit(1)

    print(f"Model: {config.CROSS_ENCODER_MODEL_NAME}")
    print(f"Target Directory: {config.MODEL_CACHE_DIR}")
    print("Downloading weights explicitly (approx. 87.5 MB)...")

    t0 = time.perf_counter()
    success, msg = ce_svc.download_and_load_model_sync()
    dt = time.perf_counter() - t0

    if success:
        info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.CROSS_ENCODER_MODEL_NAME)
        rev = info.get("revision", "")
        rev_str = f" [Revision: {rev[:12]}...]" if rev else ""
        print(f"[SUCCESS] {msg}{rev_str} ({dt:.2f} seconds)")
    else:
        print(f"[ERROR] {msg}")
        sys.exit(1)


def verify_provenance_cli(use_mock: bool = False):
    print_header("Verify Embedding Index Provenance")
    svc = semantic_service.get_semantic_service()

    if use_mock:
        mock_model = semantic_service.MockEmbeddingModel(dimension=config.SEMANTIC_DIMENSION)
        svc.set_custom_model(mock_model)
    elif not svc.is_model_loaded():
        print(f"Loading local model '{svc.model_name}' offline...")
        ok, msg = svc.load_model_sync(allow_download=False)
        if not ok:
            print(f"[ERROR] Could not load model offline: {msg}")
            sys.exit(1)

    print("Verifying sample vectors against model inference...")
    ok, msg = svc.verify_provenance(sample_size=10, min_similarity=0.999)
    if ok:
        print(f"[SUCCESS] Provenance Verified: {msg}")
        print("Embeddings in database and cache are 100% compatible with active model.")
    else:
        print(f"[FAILED] Provenance Incompatible: {msg}")
        print("Run 'python setup_semantic.py --build-index' to regenerate embeddings.")
        sys.exit(1)


def build_index(batch_size: int = config.SEMANTIC_BATCH_SIZE, use_mock: bool = False):
    print_header("Build / Refresh Catalogue Embedding Index")
    svc = semantic_service.get_semantic_service()

    if use_mock:
        print("[INFO] Using deterministic MockEmbeddingModel for offline test/benchmark.")
        mock_model = semantic_service.MockEmbeddingModel(dimension=config.SEMANTIC_DIMENSION)
        svc.set_custom_model(mock_model)
    elif not svc.is_model_loaded():
        # Attempt loading without download
        print(f"Loading local model '{svc.model_name}' offline...")
        success, msg = svc.load_model_sync(allow_download=False)
        if not svc.is_model_loaded():
            print(f"[ERROR] Model weights are not loaded ({msg}). Run:")
            print("    python setup_semantic.py --download-model")
            print("or pass --mock to build with the deterministic mock model.")
            sys.exit(1)

    stats = database.get_catalogue_embedding_stats(svc.db_path, svc.model_name)
    total_plots = stats["total_plots"]
    pending = database.get_pending_catalogue_plots(svc.db_path, svc.model_name)

    print(f"Total Catalogue Plots: {total_plots:,}")
    print(f"Pending to Index:     {len(pending):,}")
    print(f"Batch Size:           {batch_size}")
    print("-" * 68)

    if not pending:
        svc.load_index_from_storage()
        if svc.index.count > 0 and svc.index.provenance_status not in ("verified", "mock_verified"):
            print("Verifying embedding provenance against active model...")
            ok, vmsg = svc.verify_provenance()
            if ok:
                print(f"[SUCCESS] {vmsg}")
            else:
                print(f"[WARNING] {vmsg}")
        print("[INFO] All catalogue plots are already indexed and verified. No work needed.")
        print(f"In-memory index loaded with {svc.index.count:,} active vectors.")
        return

    import threading
    done_event = threading.Event()
    results_holder = {}

    def _prog(cur, tot, msg):
        pct = (cur / max(1, tot)) * 100
        bar_len = 30
        filled = int(bar_len * cur / max(1, tot))
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {cur:,}/{tot:,} ({pct:5.1f}%) | {msg}")
        sys.stdout.flush()

    def _done(count, indexed, duration, cancelled):
        results_holder["count"] = count
        results_holder["indexed"] = indexed
        results_holder["duration"] = duration
        results_holder["cancelled"] = cancelled
        done_event.set()

    started = svc.build_index_async(
        batch_size=batch_size,
        on_progress=_prog,
        on_complete=_done
    )

    if not started:
        print("[ERROR] Could not start indexing worker (already running or model error).")
        return

    try:
        while not done_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[WARNING] Keyboard interrupt received. Cancelling indexing safely...")
        svc.cancel_indexing()
        done_event.wait(timeout=10.0)

    print("\n" + "-" * 68)
    if results_holder.get("cancelled"):
        print(f"[CANCELLED] Indexing safely stopped without catalogue corruption.")
        print(f"Indexed in this run: {results_holder.get('indexed', 0):,} plots in {results_holder.get('duration', 0.0):.2f}s.")
        print(f"Total persisted embeddings now: {results_holder.get('count', 0):,}.")
    else:
        print(f"[SUCCESS] Indexing complete in {results_holder.get('duration', 0.0):.2f} seconds.")
        print(f"Total active catalogue embeddings: {results_holder.get('count', 0):,}.")


def run_benchmark(use_mock: bool = False):
    print_header("Semantic Search Benchmark")
    svc = semantic_service.get_semantic_service()

    if use_mock:
        print("[INFO] Using deterministic MockEmbeddingModel...")
        mock_model = semantic_service.MockEmbeddingModel(dimension=config.SEMANTIC_DIMENSION)
        svc.set_custom_model(mock_model)
    elif not svc.is_model_loaded():
        info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.DEFAULT_MODEL_NAME)
        if info["found"]:
            print(f"Loading local model '{svc.model_name}' offline for real benchmark...")
            svc.load_model_sync(allow_download=False)
        else:
            print("[INFO] Local model weights not found. Using deterministic benchmark model...")
            mock_model = semantic_service.MockEmbeddingModel(dimension=config.SEMANTIC_DIMENSION)
            svc.set_custom_model(mock_model)

    # Ensure index is loaded
    if svc.index.is_empty():
        # Load from storage or synthesize benchmark catalogue
        loaded = svc.load_index_from_storage()
        if loaded == 0:
            print("[INFO] Index is empty. Generating test vector matrix for 34,570 records...")
            synth_ids = list(range(1, 34571))
            rng = np.random.RandomState(42)
            synth_mat = rng.randn(34570, config.SEMANTIC_DIMENSION).astype(np.float32)
            norms = np.linalg.norm(synth_mat, axis=1, keepdims=True)
            synth_mat /= norms
            svc.index.set_data(synth_ids, synth_mat)

    print(f"Indexed Vector Count: {svc.index.count:,} plots")
    print(f"Vector Dimension:     {config.SEMANTIC_DIMENSION}")
    print("-" * 68)

    queries = [
        "A man keeps waking up to the same day and cannot escape the cycle.",
        "A young girl discovers extraordinary mathematical abilities and is raised by her uncle.",
        "A team travels through a wormhole near Saturn in search of a new habitable planet.",
        "A thief who steals corporate secrets through dream-sharing technology.",
        "A samurai warrior in Kyoto invents a bicycle to deliver noodle soup."
    ]

    latencies = []
    print("Running sample queries:")
    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        matches = svc.search_by_query(q, top_k=10)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        top_match = list(matches.items())[0] if matches else ("None", 0.0)
        print(f"  [{idx}] \"{q[:45]}...\" -> Top ID: {top_match[0]} (Score: {top_match[1]:.2f}) in {dt_ms:.2f} ms")

    avg_lat = sum(latencies) / len(latencies)
    print("-" * 68)
    print(f"Average Search Latency: {avg_lat:.2f} ms across {svc.index.count:,} plot vectors.")
    print("=" * 68)


def clear_index():
    print_header("Clear Semantic Embedding Index")
    confirm = input("Are you sure you want to delete all generated plot embeddings? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Operation cancelled.")
        return

    svc = semantic_service.get_semantic_service()
    success = svc.clear_index()
    if success:
        print("[SUCCESS] All plot embeddings cleared from SQLite and disk cache.")
    else:
        print("[ERROR] Failed to clear embeddings.")


def main():
    parser = argparse.ArgumentParser(
        description="CineTrace Semantic Search Setup & Management CLI (Phase 5.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Display model, index, and library status")
    group.add_argument("--download-model", action="store_true", help="Download embedding model weights explicitly")
    group.add_argument("--download-cross-encoder", action="store_true", help="Download cross-encoder reranker weights explicitly (Phase 5.7)")
    group.add_argument("--build-index", action="store_true", help="Build or refresh embedding index for catalogue plots")
    group.add_argument("--verify-provenance", action="store_true", help="Verify embedding provenance against active model")
    group.add_argument("--benchmark", action="store_true", help="Benchmark search latency across indexed vectors")
    group.add_argument("--clear-index", action="store_true", help="Clear all stored embeddings")

    parser.add_argument("--batch-size", type=int, default=config.SEMANTIC_BATCH_SIZE, help="Batch size for index generation")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model for offline testing")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.download_model:
        download_model()
    elif args.download_cross_encoder:
        download_cross_encoder()
    elif args.build_index:
        build_index(batch_size=args.batch_size, use_mock=args.mock)
    elif args.verify_provenance:
        verify_provenance_cli(use_mock=args.mock)
    elif args.benchmark:
        run_benchmark(use_mock=args.mock)
    elif args.clear_index:
        clear_index()
    else:
        show_status()



if __name__ == "__main__":
    main()
