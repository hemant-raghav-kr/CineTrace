"""
CineTrace Phase 5.7 — Cross-Encoder Reranking Validation Benchmark

Benchmarks the second-stage Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
against the 55 labelled queries from Phase 5.6.2.

Compares four configurations:
1. Pure Semantic (all-MiniLM-L6-v2 dense cosine similarity)
2. Hybrid (Dense Semantic + Calibrated Lexical BM25/TF-IDF)
3. Hybrid + Cross-Encoder Rerank (Top 25 candidates)
4. Hybrid + Cross-Encoder Rerank (Top 50 candidates)

Measures and Reports:
- Hit Rate@1 (Recall@1)
- Recall@5
- Recall@10
- MRR (Mean Reciprocal Rank)
- Candidate Recall Before Reranking
- Zero-Result Rate
- Cold load time and warm inference latency per query
- Long plot descriptions, ambiguous queries, and negative queries
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure CineTrace root is on path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
import database
import semantic_service
from benchmark_semantic_validation import BENCHMARK_QUERIES
from scene_matcher import SceneMatcher


# Edge Case Test Sets
LONG_PLOT_TEST_QUERIES = [
    {
        "id": "L1",
        "title": "The Sixth Sense",
        "expected_id": 16,
        "query": "A child psychologist works with a frightened boy who sees dead people and only realizes at the end that he himself was dead all along.",
        "clue_location": "tail"  # twist is at the very end
    },
    {
        "id": "L2",
        "title": "Fight Club",
        "expected_id": 13,
        "query": "An insomniac office worker founds an underground fight society with a soap salesman who turns out to be his own split personality.",
        "clue_location": "tail"
    },
    {
        "id": "L3",
        "title": "Inception",
        "expected_id": 4,
        "query": "Corporate espionage agents use multi-level shared dreams and a spinning top totem to plant an inception idea into a business heir's subconscious.",
        "clue_location": "head_and_tail"
    }
]

AMBIGUOUS_TEST_QUERIES = [
    {
        "id": "A1",
        "query": "A masked vigilante protects the city from corrupt criminals during the night",
        "expected_genre": "action / crime"
    },
    {
        "id": "A2",
        "query": "Two childhood friends go on an unexpected cross-country road trip",
        "expected_genre": "comedy / drama"
    }
]

NEGATIVE_TEST_QUERIES = [
    {
        "id": "N1",
        "query": "Quantum mechanical density functional theory calculations of high temperature superconducting cuprates",
        "expected_behavior": "Low confidence or negative logits; no confident false positive"
    },
    {
        "id": "N2",
        "query": "Step by step instructions for replacing an engine oil filter on a 2012 Honda Civic sedan",
        "expected_behavior": "Low confidence or negative logits; no confident false positive"
    }
]


def run_reranker_benchmark(
    top_k_eval: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    print("=" * 84)
    print(" CINETRACE PHASE 5.7 — CROSS-ENCODER RERANKING BENCHMARK")
    print("=" * 84)
    print(f"Total labelled test queries:       {len(BENCHMARK_QUERIES)}")
    print(f"Evaluated configurations:          4 (Pure Semantic, Hybrid, Rerank Top 25, Rerank Top 50)")
    print(f"Catalogue size:                    Wikipedia Movie Plots (~34,570 films)")
    print(f"Hardware & Execution:              CPU Inference (sentence-transformers batching)")
    print("-" * 84)

    # Measure cold load time
    print("Measuring model load latency...")
    t_cold_sem_0 = time.perf_counter()
    sem_svc = semantic_service.get_semantic_service()
    if not sem_svc.is_model_loaded():
        sem_svc.load_model_sync(allow_download=False)
    cold_load_sem_sec = time.perf_counter() - t_cold_sem_0

    t_cold_ce_0 = time.perf_counter()
    ce_svc = semantic_service.get_cross_encoder_service()
    if not ce_svc.is_model_loaded():
        ce_svc.load_model_sync(allow_download=False)
    cold_load_ce_sec = time.perf_counter() - t_cold_ce_0

    print(f"Embedding Model Cold Load:         {cold_load_sem_sec:.3f} s ({sem_svc.model_name})")
    print(f"Cross-Encoder Cold Load:           {cold_load_ce_sec:.3f} s ({ce_svc.model_name})")

    matcher = SceneMatcher(semantic_service_instance=sem_svc)
    print(f"Search Engine Index Count:         {len(matcher.records):,} movies")
    print("-" * 84)
    print("Starting evaluation across all 55 labelled queries...\n")

    # Accumulators for the 4 configurations
    configs = ["pure_semantic", "hybrid", "rerank_25", "rerank_50"]
    config_labels = {
        "pure_semantic": "Pure Semantic (Dense Only)",
        "hybrid": "Hybrid (Semantic + Lexical)",
        "rerank_25": "Hybrid + Cross-Encoder (Top 25)",
        "rerank_50": "Hybrid + Cross-Encoder (Top 50)"
    }

    ranks: Dict[str, List[Optional[int]]] = {c: [] for c in configs}
    latencies: Dict[str, List[float]] = {c: [] for c in configs}
    zero_results: Dict[str, int] = {c: 0 for c in configs}
    candidate_recalls: Dict[str, List[bool]] = {"rerank_25": [], "rerank_50": []}

    query_details: List[Dict[str, Any]] = []

    for idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q_id = item["id"]
        q_text = item["query"]
        exp_id = item["expected_id"]
        exp_title = item["expected_title"]
        exp_year = item["expected_year"]
        cat = item["category"]

        q_detail: Dict[str, Any] = {
            "id": q_id,
            "category": cat,
            "query": q_text,
            "expected_id": exp_id,
            "expected_title": exp_title,
            "expected_year": exp_year,
            "configs": {}
        }

        # -------------------------------------------------------------
        # 1. Pure Semantic
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        sem_map = sem_svc.search_by_query(q_text, top_k=500)
        pure_lat = (time.perf_counter() - t0) * 1000.0
        latencies["pure_semantic"].append(pure_lat)

        pure_rank = None
        pure_score = 0.0
        for rank_pos, (m_id, sim) in enumerate(sem_map.items(), 1):
            if int(m_id) == exp_id:
                pure_rank = rank_pos
                pure_score = float(sim)
                break
        ranks["pure_semantic"].append(pure_rank)
        if not sem_map:
            zero_results["pure_semantic"] += 1

        q_detail["configs"]["pure_semantic"] = {
            "rank": pure_rank,
            "score": pure_score,
            "latency_ms": round(pure_lat, 2)
        }

        # -------------------------------------------------------------
        # 2. Hybrid (First Stage)
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        hybrid_res = matcher.search(
            query=q_text,
            mode="hybrid",
            top_k=100,
            min_threshold=0.01,
            semantic_top_k=500,
            rerank=False
        )
        hybrid_lat = (time.perf_counter() - t1) * 1000.0
        latencies["hybrid"].append(hybrid_lat)

        hybrid_rank = None
        hybrid_score = 0.0
        raw_lex = 0.0
        cal_lex = 0.0
        raw_sem = 0.0
        for rank_pos, cand in enumerate(hybrid_res, 1):
            c_id = cand.get("id")
            c_title = cand.get("title", "")
            c_yr = str(cand.get("year", ""))
            if (c_id is not None and int(c_id) == exp_id) or (c_title.lower() == exp_title.lower() and c_yr == exp_year):
                hybrid_rank = rank_pos
                hybrid_score = float(cand.get("similarity", 0.0))
                raw_lex = float(cand.get("raw_lexical_score", 0.0))
                cal_lex = float(cand.get("calibrated_lexical_score", 0.0))
                raw_sem = float(cand.get("raw_semantic_score", 0.0))
                break
        ranks["hybrid"].append(hybrid_rank)
        if not hybrid_res:
            zero_results["hybrid"] += 1

        q_detail["configs"]["hybrid"] = {
            "rank": hybrid_rank,
            "score": hybrid_score,
            "raw_semantic_score": raw_sem,
            "raw_lexical_score": raw_lex,
            "calibrated_lexical_score": cal_lex,
            "latency_ms": round(hybrid_lat, 2)
        }

        # Check candidate recall before reranking (was target in top 25 / top 50?)
        in_top_25_pool = (hybrid_rank is not None and hybrid_rank <= 25)
        in_top_50_pool = (hybrid_rank is not None and hybrid_rank <= 50)
        candidate_recalls["rerank_25"].append(in_top_25_pool)
        candidate_recalls["rerank_50"].append(in_top_50_pool)

        # -------------------------------------------------------------
        # 3. Hybrid + Cross-Encoder Rerank (Top 25)
        # -------------------------------------------------------------
        t2 = time.perf_counter()
        r25_res = matcher.search(
            query=q_text,
            mode="hybrid",
            top_k=100,
            min_threshold=0.01,
            semantic_top_k=500,
            rerank=True,
            rerank_top_k=25,
            cross_encoder_service=ce_svc
        )
        r25_lat = (time.perf_counter() - t2) * 1000.0
        latencies["rerank_25"].append(r25_lat)

        r25_rank = None
        r25_score = 0.0
        for rank_pos, cand in enumerate(r25_res, 1):
            c_id = cand.get("id")
            c_title = cand.get("title", "")
            c_yr = str(cand.get("year", ""))
            if (c_id is not None and int(c_id) == exp_id) or (c_title.lower() == exp_title.lower() and c_yr == exp_year):
                r25_rank = rank_pos
                r25_score = float(cand.get("raw_cross_encoder_score") or cand.get("similarity", 0.0))
                break
        ranks["rerank_25"].append(r25_rank)
        if not r25_res:
            zero_results["rerank_25"] += 1

        q_detail["configs"]["rerank_25"] = {
            "rank": r25_rank,
            "score": r25_score,
            "latency_ms": round(r25_lat, 2),
            "candidate_in_top_k": in_top_25_pool
        }

        # -------------------------------------------------------------
        # 4. Hybrid + Cross-Encoder Rerank (Top 50)
        # -------------------------------------------------------------
        t3 = time.perf_counter()
        r50_res = matcher.search(
            query=q_text,
            mode="hybrid",
            top_k=100,
            min_threshold=0.01,
            semantic_top_k=500,
            rerank=True,
            rerank_top_k=50,
            cross_encoder_service=ce_svc
        )
        r50_lat = (time.perf_counter() - t3) * 1000.0
        latencies["rerank_50"].append(r50_lat)

        r50_rank = None
        r50_score = 0.0
        for rank_pos, cand in enumerate(r50_res, 1):
            c_id = cand.get("id")
            c_title = cand.get("title", "")
            c_yr = str(cand.get("year", ""))
            if (c_id is not None and int(c_id) == exp_id) or (c_title.lower() == exp_title.lower() and c_yr == exp_year):
                r50_rank = rank_pos
                r50_score = float(cand.get("raw_cross_encoder_score") or cand.get("similarity", 0.0))
                break
        ranks["rerank_50"].append(r50_rank)
        if not r50_res:
            zero_results["rerank_50"] += 1

        # Store top-10 retrieved candidate summaries from Top-25 config
        top_candidates = []
        for cand in r25_res[:10]:
            top_candidates.append({
                "id": cand.get("id"),
                "title": cand.get("title"),
                "year": cand.get("year"),
                "similarity": cand.get("similarity"),
                "score_label": cand.get("score_label"),
                "match_type": cand.get("match_type"),
                "raw_cross_encoder_score": cand.get("raw_cross_encoder_score")
            })
        q_detail["top_10_reranked_candidates"] = top_candidates

        q_detail["configs"]["rerank_50"] = {
            "rank": r50_rank,
            "score": r50_score,
            "latency_ms": round(r50_lat, 2),
            "candidate_in_top_k": in_top_50_pool
        }

        query_details.append(q_detail)

        if verbose:
            p_str = f"#{pure_rank}" if pure_rank else " >500"
            h_str = f"#{hybrid_rank}" if hybrid_rank else " >100"
            r25_str = f"#{r25_rank}" if r25_rank else " >100"
            r50_str = f"#{r50_rank}" if r50_rank else " >100"
            print(f"[{idx:02d}/55] {exp_title[:24]:<24} ({exp_year}) | Pure: {p_str:>5} | Hyb: {h_str:>5} | Rerank25: {r25_str:>5} | Rerank50: {r50_str:>5}")

    # -------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -------------------------------------------------------------
    total_q = len(BENCHMARK_QUERIES)
    metrics: Dict[str, Dict[str, Any]] = {}

    for c in configs:
        c_ranks = ranks[c]
        hit_1 = sum(1 for r in c_ranks if r is not None and r == 1)
        rec_5 = sum(1 for r in c_ranks if r is not None and r <= 5)
        rec_10 = sum(1 for r in c_ranks if r is not None and r <= 10)
        mrr = sum((1.0 / r) for r in c_ranks if r is not None and r <= 10) / float(total_q)
        zero_r = zero_results[c]
        avg_lat = sum(latencies[c]) / float(total_q)

        metrics[c] = {
            "label": config_labels[c],
            "total_queries": total_q,
            "hit_rate_at_1": round(hit_1 / total_q, 4),
            "hit_count_at_1": hit_1,
            "recall_at_5": round(rec_5 / total_q, 4),
            "recall_count_at_5": rec_5,
            "recall_at_10": round(rec_10 / total_q, 4),
            "recall_count_at_10": rec_10,
            "mrr": round(mrr, 4),
            "zero_result_rate": round(zero_r / total_q, 4),
            "avg_latency_ms": round(avg_lat, 2)
        }

    # Add candidate recall before reranking
    metrics["rerank_25"]["candidate_recall_before_rerank"] = round(sum(candidate_recalls["rerank_25"]) / float(total_q), 4)
    metrics["rerank_50"]["candidate_recall_before_rerank"] = round(sum(candidate_recalls["rerank_50"]) / float(total_q), 4)

    # -------------------------------------------------------------
    # Edge Cases & Special Tests
    # -------------------------------------------------------------
    print("\n" + "-" * 84)
    print(" Running Long Plot, Ambiguous & Negative Query Verification...")
    print("-" * 84)

    long_plot_results = []
    for test in LONG_PLOT_TEST_QUERIES:
        t_start = time.perf_counter()
        res = matcher.search(test["query"], mode="hybrid", top_k=10, rerank=True, rerank_top_k=25, cross_encoder_service=ce_svc)
        dt = (time.perf_counter() - t_start) * 1000.0
        exp_id = test["expected_id"]
        found_rank = None
        for r_idx, c in enumerate(res, 1):
            if c.get("id") == exp_id:
                found_rank = r_idx
                break
        long_plot_results.append({
            "id": test["id"],
            "title": test["title"],
            "query": test["query"],
            "clue_location": test["clue_location"],
            "rank": found_rank,
            "latency_ms": round(dt, 2),
            "top_match": res[0]["title"] if res else "None",
            "top_score_label": res[0].get("score_label", "") if res else ""
        })
        print(f"  [Long Plot] {test['title']}: Rank #{found_rank} in {dt:.1f}ms (Top: {res[0]['title'] if res else 'None'})")

    ambiguous_results = []
    for test in AMBIGUOUS_TEST_QUERIES:
        res = matcher.search(test["query"], mode="hybrid", top_k=5, rerank=True, rerank_top_k=25, cross_encoder_service=ce_svc)
        ambiguous_results.append({
            "id": test["id"],
            "query": test["query"],
            "top_matches": [f"{r['title']} ({r.get('score_label')})" for r in res[:3]]
        })
        print(f"  [Ambiguous] '{test['query'][:38]}...': Top match '{res[0]['title'] if res else 'None'}'")

    negative_results = []
    for test in NEGATIVE_TEST_QUERIES:
        res = matcher.search(test["query"], mode="hybrid", top_k=5, min_threshold=0.10, rerank=True, rerank_top_k=25, cross_encoder_service=ce_svc)
        negative_results.append({
            "id": test["id"],
            "query": test["query"],
            "results_count": len(res),
            "top_score": res[0].get("similarity") if res else None,
            "expected_behavior": test["expected_behavior"]
        })
        print(f"  [Negative] '{test['query'][:38]}...': {len(res)} results above threshold")

    # -------------------------------------------------------------
    # Output Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 84)
    print(" BENCHMARK RESULTS SUMMARY (55 LABELLED QUERIES)")
    print("=" * 84)
    header = f"{'Configuration':<34} | {'Hit@1':<8} | {'Rec@5':<8} | {'Rec@10':<8} | {'MRR':<8} | {'Latency':<9}"
    print(header)
    print("-" * 84)
    for c in configs:
        m = metrics[c]
        row = f"{m['label']:<34} | {m['hit_rate_at_1']*100:5.1f}%  | {m['recall_at_5']*100:5.1f}%  | {m['recall_at_10']*100:5.1f}%  | {m['mrr']:6.4f} | {m['avg_latency_ms']:6.1f} ms"
        print(row)
    print("-" * 84)
    print(f"Candidate Recall Before Rerank (Top 25): {metrics['rerank_25']['candidate_recall_before_rerank']*100:.1f}%")
    print(f"Candidate Recall Before Rerank (Top 50): {metrics['rerank_50']['candidate_recall_before_rerank']*100:.1f}%")
    print("=" * 84)

    # -------------------------------------------------------------
    # Save Structured Results to JSON
    # -------------------------------------------------------------
    output_data = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_queries": total_q,
            "catalogue_size": len(matcher.records),
            "embedding_model": config.DEFAULT_MODEL_NAME,
            "cross_encoder_model": config.CROSS_ENCODER_MODEL_NAME,
            "cold_load_embedding_sec": round(cold_load_sem_sec, 3),
            "cold_load_cross_encoder_sec": round(cold_load_ce_sec, 3),
        },
        "aggregate_metrics": metrics,
        "query_evaluations": query_details,
        "long_plot_evaluations": long_plot_results,
        "ambiguous_query_evaluations": ambiguous_results,
        "negative_query_evaluations": negative_results
    }

    out_file = BASE_DIR / "benchmark_reranker_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[SAVED] Benchmark results successfully written to: {out_file.resolve()}\n")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineTrace Phase 5.7 Cross-Encoder Benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K evaluation window")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query stdout logs")
    args = parser.parse_args()

    run_reranker_benchmark(top_k_eval=args.top_k, verbose=not args.quiet)
