"""
CineTrace Phase 5.8 — Independent Held-Out Search Validation Benchmark

Evaluates 55 brand-new, unseen natural-language plot queries that have ZERO overlap
with the 55 tuning queries used in Phase 5.6.2 and Phase 5.7.

Measures and compares four retrieval pipelines against the 34,570-record catalogue:
1. Pure Semantic (all-MiniLM-L6-v2 dense cosine similarity)
2. Hybrid Retrieval (Dense Semantic + Calibrated Lexical BM25/TF-IDF)
3. Hybrid + Cross-Encoder Rerank (Top 25 candidates, ms-marco-MiniLM-L-6-v2)
4. Hybrid + Cross-Encoder Rerank (Top 50 candidates, ms-marco-MiniLM-L-6-v2)

Computes:
- Hit Rate@1 (Recall@1)
- Recall@5
- Recall@10
- MRR (Mean Reciprocal Rank)
- Candidate Recall Before Rerank
- Candidate Retrieval Failures vs Reranking Failures
- Zero-Result Rate
- Latency statistics (mean, median, p95)
"""

import os
import sys
import time
import json
import argparse
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
from scene_matcher import SceneMatcher

# 55 Fresh Held-Out Queries with ZERO overlap with the 55 tuning queries
HELDOUT_QUERIES = [
    # --- Category: Golden Age & Canonical Classics ---
    {
        "id": "H01",
        "category": "classic",
        "genre": "drama",
        "expected_id": 3077,
        "expected_title": "Citizen Kane",
        "expected_year": "1941",
        "query": "A dying newspaper tycoon whispers his final mysterious word, prompting a journalist to interview his acquaintances and uncover the story of his rise to immense wealth and lonely death."
    },
    {
        "id": "H02",
        "category": "classic",
        "genre": "drama",
        "expected_id": 16,
        "expected_title": "Casablanca",
        "expected_year": "1942",
        "query": "An expatriate American nightclub owner in Morocco must decide whether to help his former lover and her fugitive resistance leader husband escape from authorities."
    },
    {
        "id": "H03",
        "category": "classic",
        "genre": "thriller",
        "expected_id": 7202,
        "expected_title": "Vertigo",
        "expected_year": "1958",
        "query": "A retired police detective with extreme acrophobia is hired to follow an acquaintance's wife, but becomes obsessively infatuated with her before an apparent tragedy."
    },
    {
        "id": "H04",
        "category": "classic",
        "genre": "drama",
        "expected_id": 5301,
        "expected_title": "Sunset Boulevard",
        "expected_year": "1950",
        "query": "A struggling Hollywood screenwriter gets drawn into the delusions of a faded silent movie star who dreams of making a triumphant comeback to the screen."
    },
    {
        "id": "H05",
        "category": "classic",
        "genre": "thriller",
        "expected_id": 53,
        "expected_title": "Rear Window",
        "expected_year": "1954",
        "query": "A recuperating photographer confined to a wheelchair passes the time by spying on his courtyard neighbors and becomes convinced that a salesman in the opposite apartment has murdered his wife."
    },
    {
        "id": "H06",
        "category": "classic",
        "genre": "musical",
        "expected_id": 5790,
        "expected_title": "Singin' in the Rain",
        "expected_year": "1952",
        "query": "A silent screen star and his comedic dancing partner navigate the tumultuous transition of Hollywood from silent pictures into talking musical cinema."
    },
    {
        "id": "H07",
        "category": "classic",
        "genre": "crime",
        "expected_id": 3834,
        "expected_title": "Double Indemnity",
        "expected_year": "1944",
        "query": "An insurance salesman gets seduced by a cunning married woman into plotting the fraudulent murder of her husband to collect a lucrative accidental death payout."
    },
    {
        "id": "H08",
        "category": "classic",
        "genre": "comedy",
        "expected_id": 7337,
        "expected_title": "Some Like It Hot",
        "expected_year": "1959",
        "query": "Two struggling male musicians witness a deadly mob massacre and disguise themselves in drag as female instrumentalists to join an all-women jazz band traveling to Florida."
    },
    {
        "id": "H09",
        "category": "classic",
        "genre": "drama",
        "expected_id": 4320,
        "expected_title": "It's a Wonderful Life",
        "expected_year": "1946",
        "query": "A despondent small-town building and loan manager on the verge of financial ruin is visited on Christmas Eve by a guardian angel who shows him what the town would be like if he had never been born."
    },
    {
        "id": "H10",
        "category": "classic",
        "genre": "fantasy",
        "expected_id": 2769,
        "expected_title": "The Wizard of Oz",
        "expected_year": "1939",
        "query": "A farm girl swept away by a Kansas tornado finds herself in a magical realm and journeys down a yellow brick road accompanied by a scarecrow, a tin woodsman, and a cowardly lion."
    },
    {
        "id": "H11",
        "category": "classic",
        "genre": "adventure",
        "expected_id": 7679,
        "expected_title": "Lawrence of Arabia",
        "expected_year": "1962",
        "query": "A flamboyant British Army lieutenant unites disparate warring Arabian tribes to launch guerrilla warfare against the Ottoman Empire during the First World War."
    },
    {
        "id": "H12",
        "category": "classic",
        "genre": "thriller",
        "expected_id": 7310,
        "expected_title": "North by Northwest",
        "expected_year": "1959",
        "query": "An innocent advertising executive is mistaken for a non-existent government agent by ruthless foreign spies and chased across the United States, culminating in a chase on Mount Rushmore."
    },

    # --- Category: Crime, Mystery & Neo-Noir ---
    {
        "id": "H13",
        "category": "crime",
        "genre": "mystery",
        "expected_id": 9072,
        "expected_title": "Chinatown",
        "expected_year": "1974",
        "query": "A private investigator specializing in marital infidelity is hired by an impostor and stumbles into a complex web of municipal corruption, land swindles, and incestuous family secrets in 1930s Los Angeles."
    },
    {
        "id": "H14",
        "category": "crime",
        "genre": "drama",
        "expected_id": 9361,
        "expected_title": "Taxi Driver",
        "expected_year": "1976",
        "query": "A lonely, mentally unstable Vietnam War veteran drives a cab through the decaying streets of New York at night and plots violent retribution against urban filth and a corrupt politician."
    },
    {
        "id": "H15",
        "category": "crime",
        "genre": "crime",
        "expected_id": 10172,
        "expected_title": "Scarface",
        "expected_year": "1983",
        "query": "A Cuban refugee arrives in Miami during the Mariel boatlift and ruthlessly murders his way to the top of an enormous cocaine empire before succumbing to paranoia and greed."
    },
    {
        "id": "H16",
        "category": "crime",
        "genre": "crime",
        "expected_id": 12490,
        "expected_title": "Heat",
        "expected_year": "1995",
        "query": "A dedicated Los Angeles robbery-homicide detective relentlessly pursues a consummate professional master thief and his disciplined crew planning one final bank heist."
    },
    {
        "id": "H17",
        "category": "crime",
        "genre": "mystery",
        "expected_id": 12619,
        "expected_title": "The Usual Suspects",
        "expected_year": "1995",
        "query": "A crippled con artist being interrogated by customs agents spins a convoluted tale about five career criminals brought together on a police lineup by an elusive, legendary criminal mastermind."
    },
    {
        "id": "H18",
        "category": "crime",
        "genre": "mystery",
        "expected_id": 13073,
        "expected_title": "L.A. Confidential",
        "expected_year": "1997",
        "query": "Three very different LAPD detectives investigate a mass shooting at a downtown diner, uncovering widespread institutional police corruption, prostitution rings, and heroin distribution in 1950s Los Angeles."
    },
    {
        "id": "H19",
        "category": "crime",
        "genre": "thriller",
        "expected_id": 50,
        "expected_title": "Seven",
        "expected_year": "1995",
        "query": "A retiring detective and his impetuous young partner track a methodical serial killer who designs gruesome murders themed around the seven deadly sins."
    },
    {
        "id": "H20",
        "category": "crime",
        "genre": "crime",
        "expected_id": 55,
        "expected_title": "The Departed",
        "expected_year": "2006",
        "query": "An undercover state trooper infiltrates an organized crime syndicate in Boston while a ruthless gang boss plants a corrupt mole within the Massachusetts state police department."
    },
    {
        "id": "H21",
        "category": "crime",
        "genre": "thriller",
        "expected_id": 15311,
        "expected_title": "No Country for Old Men",
        "expected_year": "2007",
        "query": "A hunter in the Texas desert stumbles upon the aftermath of a botched drug deal and takes a satchel filled with millions, pursued by a psychopathic hitman wielding a cattle bolt pistol."
    },
    {
        "id": "H22",
        "category": "crime",
        "genre": "mystery",
        "expected_id": 15397,
        "expected_title": "Zodiac",
        "expected_year": "2007",
        "query": "A political cartoonist and an ambitious crime reporter become consumed with deciphering cryptic ciphers sent to San Francisco newspapers by an unidentified Bay Area serial murderer."
    },
    {
        "id": "H23",
        "category": "crime",
        "genre": "thriller",
        "expected_id": 16661,
        "expected_title": "Prisoners",
        "expected_year": "2013",
        "query": "When two young girls vanish on Thanksgiving Day, a distraught father kidnaps and tortures the prime suspect after police are forced to release him due to lack of evidence."
    },
    {
        "id": "H24",
        "category": "crime",
        "genre": "thriller",
        "expected_id": 16879,
        "expected_title": "Gone Girl",
        "expected_year": "2014",
        "query": "A seemingly devoted husband becomes the center of a frenzied media circus and prime murder suspect when his immaculate wife disappears on their fifth wedding anniversary."
    },
    {
        "id": "H25",
        "category": "crime",
        "genre": "thriller",
        "expected_id": 16893,
        "expected_title": "Nightcrawler",
        "expected_year": "2014",
        "query": "A desperate unemployed sociopath buys a police scanner and camcorder to film gruesome nocturnal car crashes and violent crime scenes to sell to local TV news producers."
    },
    {
        "id": "H26",
        "category": "crime",
        "genre": "action",
        "expected_id": 16029,
        "expected_title": "Drive",
        "expected_year": "2011",
        "query": "A taciturn Hollywood stuntman who works moonlighting night shifts as a getaway driver tries to protect his vulnerable neighbor and her young son from violent mobsters."
    },
    {
        "id": "H27",
        "category": "crime",
        "genre": "crime",
        "expected_id": 42,
        "expected_title": "Fargo",
        "expected_year": "1996",
        "query": "A heavily indebted Minneapolis car salesman hires two incompetent criminals to kidnap his wealthy wife for ransom, but a heavily pregnant Minnesota police chief unravels the bloody conspiracy."
    },

    # --- Category: Science Fiction & Future Worlds ---
    {
        "id": "H28",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 24,
        "expected_title": "2001: A Space Odyssey",
        "expected_year": "1968",
        "query": "Astronauts on a mission to Jupiter to investigate a mysterious black monolith encounter a deadly existential crisis when their sentient onboard computer HAL 9000 malfunctions."
    },
    {
        "id": "H29",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 19,
        "expected_title": "The Terminator",
        "expected_year": "1984",
        "query": "An indestructible cybernetic cyborg assassin is sent backwards through time from a post-apocalyptic future to execute an unsuspecting waitress whose unborn son will lead human resistance."
    },
    {
        "id": "H30",
        "category": "scifi",
        "genre": "horror",
        "expected_id": 10073,
        "expected_title": "The Thing",
        "expected_year": "1982",
        "query": "A team of American researchers stationed at an isolated Antarctic base battle a shape-shifting extraterrestrial organism capable of perfectly imitating any living creature it assimilates."
    },
    {
        "id": "H31",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 9,
        "expected_title": "Blade Runner",
        "expected_year": "1982",
        "query": "A burned-out detective is tasked with hunting down and terminating four synthetic humanoid replicants who escaped from an off-world colony to find their creator in futuristic Los Angeles."
    },
    {
        "id": "H32",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 15570,
        "expected_title": "WALL-E",
        "expected_year": "2008",
        "query": "A solitary waste-collecting robot left on an abandoned, garbage-strewn Earth falls in love with a sleek probe sent by humans living on a luxury starship in deep space."
    },
    {
        "id": "H33",
        "category": "scifi",
        "genre": "drama",
        "expected_id": 16561,
        "expected_title": "Her",
        "expected_year": "2013",
        "query": "A lonely introverted greeting card writer develops deep feelings and an emotional romantic relationship with an intuitive and evolving artificial intelligence operating system."
    },
    {
        "id": "H34",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 16978,
        "expected_title": "Ex Machina",
        "expected_year": "2015",
        "query": "A young software programmer is selected to evaluate the human qualities and genuine consciousness of a beautiful humanoid gynoid during a week-long Turing test at a remote estate."
    },
    {
        "id": "H35",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 40,
        "expected_title": "Arrival",
        "expected_year": "2016",
        "query": "A linguistics professor is recruited by the military to decode the non-linear circular language of enigmatic extraterrestrial visitors hovering across global locations."
    },
    {
        "id": "H36",
        "category": "scifi",
        "genre": "scifi",
        "expected_id": 6,
        "expected_title": "Interstellar",
        "expected_year": "2014",
        "query": "A former NASA pilot leads an exploratory spacecraft through a newly discovered wormhole near Saturn to find a habitable exoplanet for humanity as Earth faces agricultural collapse."
    },

    # --- Category: Drama, Character Study & Historical ---
    {
        "id": "H37",
        "category": "drama",
        "genre": "drama",
        "expected_id": 9226,
        "expected_title": "One Flew Over the Cuckoo's Nest",
        "expected_year": "1975",
        "query": "A rebellious criminal pleads insanity to serve his sentence in a mental institution, where he rallies the suppressed patients against a tyrannical and cold head nurse."
    },
    {
        "id": "H38",
        "category": "drama",
        "genre": "biography",
        "expected_id": 9805,
        "expected_title": "Raging Bull",
        "expected_year": "1980",
        "query": "An aggressive Italian-American middleweight boxer achieves championship glory inside the ring while his uncontrollable jealousy, rage, and paranoia destroy his marriage and family."
    },
    {
        "id": "H39",
        "category": "drama",
        "genre": "war",
        "expected_id": 9613,
        "expected_title": "Apocalypse Now",
        "expected_year": "1979",
        "query": "An Army captain is dispatched on a patrol boat along a remote Cambodian river on a classified mission to assassinate a renegade Special Forces colonel who has gone mad."
    },
    {
        "id": "H40",
        "category": "drama",
        "genre": "war",
        "expected_id": 13378,
        "expected_title": "Saving Private Ryan",
        "expected_year": "1998",
        "query": "Following the Normandy landings, a squad of Army Rangers journeys deep behind enemy lines in occupied France to retrieve a paratrooper whose three brothers were killed in action."
    },
    {
        "id": "H41",
        "category": "drama",
        "genre": "drama",
        "expected_id": 15374,
        "expected_title": "There Will Be Blood",
        "expected_year": "2007",
        "query": "A ruthless silver prospector turned oil magnate amasses vast power and wealth in California while engaging in a bitter rivalry with a young charismatic preacher."
    },
    {
        "id": "H42",
        "category": "drama",
        "genre": "drama",
        "expected_id": 35,
        "expected_title": "The Social Network",
        "expected_year": "2010",
        "query": "A Harvard undergraduate programmer creates a revolutionary college campus networking website, only to be sued by twin rowing athletes claiming intellectual theft and his co-founder."
    },
    {
        "id": "H43",
        "category": "drama",
        "genre": "drama",
        "expected_id": 32,
        "expected_title": "Good Will Hunting",
        "expected_year": "1997",
        "query": "A brilliant but troubled working-class janitor at MIT solves an impossible math theorem and agrees to undergo therapy with a compassionate counselor to avoid prison."
    },
    {
        "id": "H44",
        "category": "drama",
        "genre": "drama",
        "expected_id": 10214,
        "expected_title": "Amadeus",
        "expected_year": "1984",
        "query": "Court composer Antonio Salieri is consumed by agonizing jealousy and religious resentment against the vulgar, childish young genius Wolfgang Amadeus Mozart in 18th-century Vienna."
    },
    {
        "id": "H45",
        "category": "drama",
        "genre": "drama",
        "expected_id": 27,
        "expected_title": "Parasite",
        "expected_year": "2019",
        "query": "An impoverished family schemes to infiltrate the household of a wealthy clan by securing employment one by one through forged credentials and deceit."
    },
    {
        "id": "H46",
        "category": "drama",
        "genre": "comedy",
        "expected_id": 46,
        "expected_title": "The Grand Budapest Hotel",
        "expected_year": "2014",
        "query": "A fastidious European hotel concierge and his loyal lobby boy become embroiled in the theft and recovery of a priceless Renaissance painting amid a family fortune dispute."
    },
    {
        "id": "H47",
        "category": "drama",
        "genre": "drama",
        "expected_id": 16888,
        "expected_title": "Birdman or (The Unexpected Virtue of Ignorance)",
        "expected_year": "2014",
        "query": "A washed-up Hollywood actor famous for playing an iconic cinema superhero attempts to revive his career by directing and starring in a serious Broadway stage production."
    },
    {
        "id": "H48",
        "category": "drama",
        "genre": "drama",
        "expected_id": 21473,
        "expected_title": "Room",
        "expected_year": "2015",
        "query": "A young woman held captive for seven years in an enclosed garden shed devises a daring escape plan for her five-year-old son who has never seen the outside world."
    },
    {
        "id": "H49",
        "category": "drama",
        "genre": "fantasy",
        "expected_id": 45,
        "expected_title": "Pan's Labyrinth",
        "expected_year": "2006",
        "query": "In post-Civil War Francoist Spain, a young stepdaughter of a sadistic military captain discovers a crumbling ancient maze and encounters a mystical faun who assigns her three dangerous tasks."
    },

    # --- Category: Comedy, Adventure & Animation ---
    {
        "id": "H50",
        "category": "adventure",
        "genre": "action",
        "expected_id": 56,
        "expected_title": "Raiders of the Lost Ark",
        "expected_year": "1981",
        "query": "An intrepid archaeology professor armed with a bullwhip is contracted by military intelligence to find the biblical Ark of the Covenant before German troops harness its supernatural power."
    },
    {
        "id": "H51",
        "category": "comedy",
        "genre": "comedy",
        "expected_id": 10259,
        "expected_title": "Ghostbusters",
        "expected_year": "1984",
        "query": "Three eccentric Columbia parapsychology professors establish a paranormal pest control business in New York City and battle an ancient Sumerian god summoned atop a high-rise."
    },
    {
        "id": "H52",
        "category": "comedy",
        "genre": "drama",
        "expected_id": 10375,
        "expected_title": "The Breakfast Club",
        "expected_year": "1985",
        "query": "Five high school students representing different adolescent cliques spend a Saturday in detention together under a strict assistant principal and open up about their personal struggles."
    },
    {
        "id": "H53",
        "category": "adventure",
        "genre": "drama",
        "expected_id": 10662,
        "expected_title": "Stand by Me",
        "expected_year": "1986",
        "query": "Four young Oregon boys hike along a railroad track over Labor Day weekend in 1959 to find the body of a missing boy hit by a train, confronting a gang of older bullies."
    },
    {
        "id": "H54",
        "category": "adventure",
        "genre": "comedy",
        "expected_id": 49,
        "expected_title": "The Princess Bride",
        "expected_year": "1987",
        "query": "A grandfather reads a fairytale adventure to his sick grandson about a farmhand turned masked pirate who battles sword masters, giants, and an arrogant prince to rescue his true love."
    },
    {
        "id": "H55",
        "category": "comedy",
        "genre": "comedy",
        "expected_id": 41,
        "expected_title": "The Big Lebowski",
        "expected_year": "1998",
        "query": "A laid-back bowling enthusiast named the Dude is mistaken for a wheelchair-bound millionaire of the same name and gets drawn into a bizarre kidnapping ransom scheme when his rug is ruined."
    }
]

# Negative Out-of-Domain Test Queries
NEGATIVE_HELDOUT_QUERIES = [
    {
        "id": "N_H01",
        "query": "Quantum mechanical density functional theory calculations of high temperature superconducting cuprates",
        "expected_behavior": "Low similarity / negative cross-encoder logit"
    },
    {
        "id": "N_H02",
        "query": "Step by step instructions for replacing an engine oil filter and spark plugs on a 2012 Honda Civic sedan",
        "expected_behavior": "Low similarity / negative cross-encoder logit"
    }
]


def run_heldout_benchmark(
    top_k_eval: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    print("=" * 80)
    print("CINETRACE PHASE 5.8 — INDEPENDENT HELD-OUT SEARCH VALIDATION BENCHMARK")
    print("=" * 80)
    print(f"Total Held-Out Queries: {len(HELDOUT_QUERIES)}")
    print(f"Catalogue DB: {database.DEFAULT_DB_PATH}")
    print(f"Embedding Model: {config.DEFAULT_MODEL_NAME}")
    print(f"Cross-Encoder Model: {config.CROSS_ENCODER_MODEL_NAME}")
    print("-" * 80)

    # 1. Initialize SceneMatcher
    t0_init = time.perf_counter()
    matcher = SceneMatcher()
    init_sec = time.perf_counter() - t0_init
    print(f"[Init] SceneMatcher loaded {len(matcher.records)} records in {init_sec:.2f}s")

    # 2. Initialize and load bi-encoder model
    sem_svc = matcher.semantic_service or semantic_service.get_semantic_service()
    t0_bi = time.perf_counter()
    bi_loaded, bi_msg = sem_svc.load_model_sync(allow_download=False)
    bi_sec = time.perf_counter() - t0_bi
    print(f"[Init] Semantic Bi-Encoder Loaded: {bi_loaded} ({bi_msg}) in {bi_sec:.3f}s")

    # 3. Initialize and load cross-encoder model
    ce_svc = semantic_service.get_cross_encoder_service()
    t0_ce = time.perf_counter()
    ce_loaded, ce_msg = ce_svc.load_model_sync()
    ce_sec = time.perf_counter() - t0_ce
    print(f"[Init] Cross-Encoder Loaded: {ce_loaded} ({ce_msg}) in {ce_sec:.3f}s")

    # Benchmarking configurations
    # We will run 4 pipelines:
    # 1. Pure Semantic (dense similarity only)
    # 2. Hybrid (dense semantic + calibrated lexical)
    # 3. Hybrid + Cross-Encoder Top 25
    # 4. Hybrid + Cross-Encoder Top 50

    pipelines = ["pure_semantic", "hybrid", "rerank_25", "rerank_50"]
    labels = {
        "pure_semantic": "Pure Semantic (Dense all-MiniLM-L6-v2)",
        "hybrid": "Hybrid Retrieval (Semantic + Calibrated Lexical)",
        "rerank_25": "Hybrid + Cross-Encoder Rerank (Top 25)",
        "rerank_50": "Hybrid + Cross-Encoder Rerank (Top 50)"
    }

    metrics = {
        pipe: {
            "label": labels[pipe],
            "total_queries": len(HELDOUT_QUERIES),
            "hit_count_at_1": 0,
            "recall_count_at_5": 0,
            "recall_count_at_10": 0,
            "mrr_sum": 0.0,
            "zero_results_count": 0,
            "latencies_ms": [],
            "candidate_recall_count": 0,
            "retrieval_failure_count": 0,
            "rerank_failure_count": 0
        }
        for pipe in pipelines
    }

    per_query_results = []

    print("\nExecuting evaluation queries across all 4 pipelines...")

    for idx, item in enumerate(HELDOUT_QUERIES, 1):
        qid = item["id"]
        qtext = item["query"]
        expected_id = int(item["expected_id"])
        expected_title = item["expected_title"]

        query_record = {
            "id": qid,
            "expected_id": expected_id,
            "expected_title": expected_title,
            "category": item.get("category"),
            "genre": item.get("genre"),
            "pipelines": {}
        }

        # -------------------------------------------------------------
        # Pipeline 1: Pure Semantic
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        # Retrieve candidates with pure dense cosine similarity
        sem_map = sem_svc.search_by_query(qtext, top_k=50)
        pure_cands = []
        for cid, s in sorted(sem_map.items(), key=lambda x: x[1], reverse=True)[:top_k_eval]:
            pure_cands.append({"id": cid, "similarity": s})
        pure_lat_ms = (time.perf_counter() - t0) * 1000.0

        pure_rank = None
        for r_idx, c in enumerate(pure_cands, 1):
            if c["id"] == expected_id:
                pure_rank = r_idx
                break

        metrics["pure_semantic"]["latencies_ms"].append(pure_lat_ms)
        if len(pure_cands) == 0:
            metrics["pure_semantic"]["zero_results_count"] += 1
        if pure_rank == 1:
            metrics["pure_semantic"]["hit_count_at_1"] += 1
        if pure_rank and pure_rank <= 5:
            metrics["pure_semantic"]["recall_count_at_5"] += 1
        if pure_rank and pure_rank <= 10:
            metrics["pure_semantic"]["recall_count_at_10"] += 1
        if pure_rank:
            metrics["pure_semantic"]["mrr_sum"] += (1.0 / pure_rank)

        query_record["pipelines"]["pure_semantic"] = {
            "rank": pure_rank,
            "latency_ms": round(pure_lat_ms, 2)
        }

        # -------------------------------------------------------------
        # First Stage Hybrid Candidates (Top 50)
        # -------------------------------------------------------------
        t0_hyb = time.perf_counter()
        raw_hybrid_cands = matcher.search(
            query=qtext,
            mode="hybrid",
            top_k=50,
            rerank=False,
            min_threshold=0.0
        )
        hyb_lat_ms = (time.perf_counter() - t0_hyb) * 1000.0

        hybrid_rank = None
        for r_idx, c in enumerate(raw_hybrid_cands[:top_k_eval], 1):
            if c["id"] == expected_id:
                hybrid_rank = r_idx
                break

        metrics["hybrid"]["latencies_ms"].append(hyb_lat_ms)
        if len(raw_hybrid_cands) == 0:
            metrics["hybrid"]["zero_results_count"] += 1
        if hybrid_rank == 1:
            metrics["hybrid"]["hit_count_at_1"] += 1
        if hybrid_rank and hybrid_rank <= 5:
            metrics["hybrid"]["recall_count_at_5"] += 1
        if hybrid_rank and hybrid_rank <= 10:
            metrics["hybrid"]["recall_count_at_10"] += 1
        if hybrid_rank:
            metrics["hybrid"]["mrr_sum"] += (1.0 / hybrid_rank)

        query_record["pipelines"]["hybrid"] = {
            "rank": hybrid_rank,
            "latency_ms": round(hyb_lat_ms, 2)
        }

        # Candidate recall check for Top 25 and Top 50
        in_top_25_pool = any(c["id"] == expected_id for c in raw_hybrid_cands[:25])
        in_top_50_pool = any(c["id"] == expected_id for c in raw_hybrid_cands[:50])

        # -------------------------------------------------------------
        # Pipeline 3: Hybrid + Cross-Encoder Rerank (Top 25)
        # -------------------------------------------------------------
        t0_r25 = time.perf_counter()
        r25_cands = ce_svc.rerank(qtext, [dict(c) for c in raw_hybrid_cands], top_k=25)
        r25_lat_ms = hyb_lat_ms + ((time.perf_counter() - t0_r25) * 1000.0)

        r25_rank = None
        for r_idx, c in enumerate(r25_cands[:top_k_eval], 1):
            if c["id"] == expected_id:
                r25_rank = r_idx
                break

        metrics["rerank_25"]["latencies_ms"].append(r25_lat_ms)
        if len(r25_cands) == 0:
            metrics["rerank_25"]["zero_results_count"] += 1
        if in_top_25_pool:
            metrics["rerank_25"]["candidate_recall_count"] += 1
        else:
            metrics["rerank_25"]["retrieval_failure_count"] += 1

        if r25_rank == 1:
            metrics["rerank_25"]["hit_count_at_1"] += 1
        else:
            if in_top_25_pool:
                metrics["rerank_25"]["rerank_failure_count"] += 1

        if r25_rank and r25_rank <= 5:
            metrics["rerank_25"]["recall_count_at_5"] += 1
        if r25_rank and r25_rank <= 10:
            metrics["rerank_25"]["recall_count_at_10"] += 1
        if r25_rank:
            metrics["rerank_25"]["mrr_sum"] += (1.0 / r25_rank)

        query_record["pipelines"]["rerank_25"] = {
            "rank": r25_rank,
            "in_candidate_pool": in_top_25_pool,
            "latency_ms": round(r25_lat_ms, 2)
        }

        # -------------------------------------------------------------
        # Pipeline 4: Hybrid + Cross-Encoder Rerank (Top 50)
        # -------------------------------------------------------------
        t0_r50 = time.perf_counter()
        r50_cands = ce_svc.rerank(qtext, [dict(c) for c in raw_hybrid_cands], top_k=50)
        r50_lat_ms = hyb_lat_ms + ((time.perf_counter() - t0_r50) * 1000.0)

        r50_rank = None
        for r_idx, c in enumerate(r50_cands[:top_k_eval], 1):
            if c["id"] == expected_id:
                r50_rank = r_idx
                break

        metrics["rerank_50"]["latencies_ms"].append(r50_lat_ms)
        if len(r50_cands) == 0:
            metrics["rerank_50"]["zero_results_count"] += 1
        if in_top_50_pool:
            metrics["rerank_50"]["candidate_recall_count"] += 1
        else:
            metrics["rerank_50"]["retrieval_failure_count"] += 1

        if r50_rank == 1:
            metrics["rerank_50"]["hit_count_at_1"] += 1
        else:
            if in_top_50_pool:
                metrics["rerank_50"]["rerank_failure_count"] += 1

        if r50_rank and r50_rank <= 5:
            metrics["rerank_50"]["recall_count_at_5"] += 1
        if r50_rank and r50_rank <= 10:
            metrics["rerank_50"]["recall_count_at_10"] += 1
        if r50_rank:
            metrics["rerank_50"]["mrr_sum"] += (1.0 / r50_rank)

        query_record["pipelines"]["rerank_50"] = {
            "rank": r50_rank,
            "in_candidate_pool": in_top_50_pool,
            "latency_ms": round(r50_lat_ms, 2)
        }

        per_query_results.append(query_record)

        if verbose:
            p_str = f"#{pure_rank}" if pure_rank else "Miss"
            h_str = f"#{hybrid_rank}" if hybrid_rank else "Miss"
            r25_str = f"#{r25_rank}" if r25_rank else "Miss"
            r50_str = f"#{r50_rank}" if r50_rank else "Miss"
            print(f"[{idx:02d}/55] {expected_title:<28} | Pure: {p_str:<5} | Hyb: {h_str:<5} | R25: {r25_str:<5} | R50: {r50_str:<5} | ({r50_lat_ms:.0f}ms)")

    # 4. Evaluate Negative Out-of-Domain Queries
    print("\nEvaluating Negative Out-of-Domain Queries...")
    neg_results = []
    for neg in NEGATIVE_HELDOUT_QUERIES:
        n_cands = matcher.search(query=neg["query"], mode="hybrid", top_k=5, rerank=True, rerank_top_k=5)
        top_score = n_cands[0]["similarity"] if n_cands else 0.0
        top_label = n_cands[0].get("score_label", "N/A") if n_cands else "None"
        neg_results.append({
            "id": neg["id"],
            "query": neg["query"],
            "top_candidate": n_cands[0]["title"] if n_cands else "None",
            "top_score": top_score,
            "top_label": top_label
        })
        print(f"  -> [{neg['id']}] Top: {neg_results[-1]['top_candidate']} | Score: {top_score} ({top_label})")

    # 5. Aggregate final statistics
    n_queries = len(HELDOUT_QUERIES)
    final_report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": "Independent 55-Query Held-Out Validation Suite",
            "tuning_query_overlap": 0,
            "total_queries": n_queries,
            "catalogue_size": len(matcher.records),
            "bi_encoder_model": config.DEFAULT_MODEL_NAME,
            "cross_encoder_model": config.CROSS_ENCODER_MODEL_NAME,
            "cold_load_bi_sec": round(bi_sec, 3),
            "cold_load_ce_sec": round(ce_sec, 3)
        },
        "aggregate_metrics": {},
        "per_query_results": per_query_results,
        "negative_test_results": neg_results
    }

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT SEARCH VALIDATION SUMMARY (55 UNSEEN QUERIES)")
    print("=" * 80)
    print(f"{'Pipeline':<40} | {'Hit@1':<8} | {'Rec@5':<8} | {'Rec@10':<8} | {'MRR':<8} | {'Avg Lat':<10}")
    print("-" * 80)

    for pipe in pipelines:
        m = metrics[pipe]
        hit1 = m["hit_count_at_1"] / n_queries
        rec5 = m["recall_count_at_5"] / n_queries
        rec10 = m["recall_count_at_10"] / n_queries
        mrr = m["mrr_sum"] / n_queries
        zero_rate = m["zero_results_count"] / n_queries
        lats = m["latencies_ms"]
        avg_lat = sum(lats) / max(1, len(lats))
        cand_recall = m["candidate_recall_count"] / n_queries if "rerank" in pipe else None

        final_report["aggregate_metrics"][pipe] = {
            "label": m["label"],
            "total_queries": n_queries,
            "hit_rate_at_1": round(hit1, 4),
            "hit_count_at_1": m["hit_count_at_1"],
            "recall_at_5": round(rec5, 4),
            "recall_count_at_5": m["recall_count_at_5"],
            "recall_at_10": round(rec10, 4),
            "recall_count_at_10": m["recall_count_at_10"],
            "mrr": round(mrr, 4),
            "zero_result_rate": round(zero_rate, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "candidate_recall_before_rerank": round(cand_recall, 4) if cand_recall is not None else None,
            "retrieval_failure_count": m["retrieval_failure_count"],
            "rerank_failure_count": m["rerank_failure_count"]
        }

        print(f"{m['label']:<40} | {hit1*100:>6.1f}% | {rec5*100:>6.1f}% | {rec10*100:>6.1f}% | {mrr:>8.4f} | {avg_lat:>7.1f}ms")

    print("-" * 80)

    out_file = BASE_DIR / "benchmark_heldout_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nDetailed held-out results saved to: {out_file}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CineTrace Held-Out Search Benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K evaluation cutoff")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    args = parser.parse_args()

    run_heldout_benchmark(top_k_eval=args.top_k, verbose=not args.quiet)
