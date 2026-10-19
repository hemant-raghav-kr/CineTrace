"""
CineTrace Phase 5.6.2 - Independent Semantic Search Validation Benchmark

Evaluates unseen natural-language plot queries covering paraphrases, indirect descriptions,
diverse genres, short and long queries, and movies with incomplete plot descriptions against
the 34,570-record Wikipedia Movie Plots catalogue.

Compares Pure Semantic Ranking against Hybrid Score Fusion.
Measures Recall@1, Recall@5, Recall@10, MRR, zero-result rate, and latencies.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure CineTrace root is on path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scene_matcher import SceneMatcher
import config

# Benchmark Evaluation Dataset (55 Labelled Test Cases)
BENCHMARK_QUERIES = [
    # -------------------------------------------------------------
    # 1. Canonical 5 Queries (Verbatim from Phase 5.6 / 5.6.1)
    # -------------------------------------------------------------
    {
        "id": 1,
        "category": "canonical",
        "genre": "drama",
        "expected_id": 57,
        "expected_title": "Gifted",
        "expected_year": "2017",
        "query": "A young girl discovers she has extraordinary mathematical abilities and is raised by her uncle, who wants her to have a normal childhood."
    },
    {
        "id": 2,
        "category": "canonical",
        "genre": "comedy",
        "expected_id": 3,
        "expected_title": "Palm Springs",
        "expected_year": "2020",
        "query": "Two wedding guests get stuck in an infinite time loop in the desert and relive the same wedding day repeatedly."
    },
    {
        "id": 3,
        "category": "canonical",
        "genre": "sci-fi",
        "expected_id": 17023,
        "expected_title": "The Martian",
        "expected_year": "2015",
        "query": "An astronaut is stranded on Mars after a dust storm and struggles to survive using science until rescue arrives."
    },
    {
        "id": 4,
        "category": "canonical",
        "genre": "animation",
        "expected_id": 21,
        "expected_title": "Finding Nemo",
        "expected_year": "2003",
        "query": "A clownfish embarks on a dangerous journey across the ocean to find his captured son who was taken by a diver."
    },
    {
        "id": 5,
        "category": "canonical",
        "genre": "sci-fi",
        "expected_id": 4,
        "expected_title": "Inception",
        "expected_year": "2010",
        "query": "A thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea into the mind of a CEO"
    },

    # -------------------------------------------------------------
    # 2. Paraphrases (Synonyms, Restructured Phrasing, Semantic Shifts)
    # -------------------------------------------------------------
    {
        "id": 6,
        "category": "paraphrase",
        "genre": "sci-fi",
        "expected_id": 5,
        "expected_title": "The Matrix",
        "expected_year": "1999",
        "query": "A computer hacker discovers that perceived reality is an elaborate cybernetic simulation created by sentient machines to harvest human bio-energy."
    },
    {
        "id": 7,
        "category": "paraphrase",
        "genre": "romance",
        "expected_id": 17,
        "expected_title": "Eternal Sunshine of the Spotless Mind",
        "expected_year": "2004",
        "query": "A heartbroken man undergoes an experimental medical procedure to erase all memories of his tumultuous relationship with an ex-girlfriend."
    },
    {
        "id": 8,
        "category": "paraphrase",
        "genre": "comedy",
        "expected_id": 1,
        "expected_title": "Groundhog Day",
        "expected_year": "1993",
        "query": "An arrogant TV meteorologist finds himself reliving February second indefinitely in a small Pennsylvania town."
    },
    {
        "id": 9,
        "category": "paraphrase",
        "genre": "sci-fi",
        "expected_id": 10,
        "expected_title": "Jurassic Park",
        "expected_year": "1993",
        "query": "Genetically cloned prehistoric creatures break containment on an isolated island theme park, terrorizing scientists and visiting children."
    },
    {
        "id": 10,
        "category": "paraphrase",
        "genre": "drama",
        "expected_id": 8,
        "expected_title": "The Truman Show",
        "expected_year": "1998",
        "query": "An unsuspecting insurance salesman gradually realizes his entire life is an unscripted reality television broadcast staged inside a gigantic dome."
    },
    {
        "id": 11,
        "category": "paraphrase",
        "genre": "animation",
        "expected_id": 15333,
        "expected_title": "Ratatouille",
        "expected_year": "2007",
        "query": "A Parisian rodent with refined gastronomic culinary talent secretly guides an awkward kitchen apprentice by hiding under his toque hat."
    },
    {
        "id": 12,
        "category": "paraphrase",
        "genre": "animation",
        "expected_id": 34,
        "expected_title": "Monsters, Inc.",
        "expected_year": "2001",
        "query": "Creatures in a parallel dimension generate municipal energy from children's terrified screams until a toddler accidentally enters their factory."
    },
    {
        "id": 13,
        "category": "paraphrase",
        "genre": "animation",
        "expected_id": 33,
        "expected_title": "Toy Story",
        "expected_year": "1995",
        "query": "A pull-string cowboy doll experiences intense jealousy when his child owner receives a high-tech spaceman action figure for his birthday."
    },
    {
        "id": 14,
        "category": "paraphrase",
        "genre": "sci-fi",
        "expected_id": 7,
        "expected_title": "Back to the Future",
        "expected_year": "1985",
        "query": "A teenager accidentally travels thirty years into the past via a plutonium-powered vehicle and must ensure his teenage parents fall in love."
    },
    {
        "id": 15,
        "category": "paraphrase",
        "genre": "adventure",
        "expected_id": 38,
        "expected_title": "Cast Away",
        "expected_year": "2000",
        "query": "A cargo systems engineer survives a catastrophic plane crash in the Pacific and learns to endure years of solitary existence on a deserted atoll."
    },

    # -------------------------------------------------------------
    # 3. Indirect Descriptions (Thematic, Situational, No Character Names)
    # -------------------------------------------------------------
    {
        "id": 16,
        "category": "indirect",
        "genre": "thriller",
        "expected_id": 18,
        "expected_title": "Memento",
        "expected_year": "2000",
        "query": "An investigator suffering from anterograde amnesia uses tattoos and Polaroid photos to track down the perpetrator who murdered his spouse."
    },
    {
        "id": 17,
        "category": "indirect",
        "genre": "thriller",
        "expected_id": 12,
        "expected_title": "Fight Club",
        "expected_year": "1999",
        "query": "A depressed corporate insomniac and a charismatic soap maker establish a subterranean violent recreation league that morphs into an anti-consumerist anarchist movement."
    },
    {
        "id": 18,
        "category": "indirect",
        "genre": "horror",
        "expected_id": 31,
        "expected_title": "The Sixth Sense",
        "expected_year": "1999",
        "query": "A child psychologist treats an anxious young boy who claims he can perceive departed souls walking among the living."
    },
    {
        "id": 19,
        "category": "indirect",
        "genre": "mystery",
        "expected_id": 43,
        "expected_title": "Shutter Island",
        "expected_year": "2010",
        "query": "A federal marshal travels to an isolated fortress psychiatric asylum on an island to investigate the disappearance of an inmate."
    },
    {
        "id": 20,
        "category": "indirect",
        "genre": "drama",
        "expected_id": 11,
        "expected_title": "The Prestige",
        "expected_year": "2006",
        "query": "Two rival nineteenth-century stage magicians engage in a ruthless, lethal obsession to outperform each other with the ultimate teleportation illusion."
    },
    {
        "id": 21,
        "category": "indirect",
        "genre": "horror",
        "expected_id": 23,
        "expected_title": "Psycho",
        "expected_year": "1960",
        "query": "A secretary absconds with embezzled cash and checks into a secluded roadside motel operated by a nervous young taxidermist dominated by his mother."
    },
    {
        "id": 22,
        "category": "indirect",
        "genre": "horror",
        "expected_id": 54,
        "expected_title": "The Shining",
        "expected_year": "1980",
        "query": "An aspiring author takes a winter caretaking post at an isolated mountain hotel, where supernatural presences drive him into murderous madness toward his family."
    },
    {
        "id": 23,
        "category": "indirect",
        "genre": "horror",
        "expected_id": 36,
        "expected_title": "Get Out",
        "expected_year": "2017",
        "query": "An African-American photographer visits his white partner's affluent suburban family estate for the weekend, discovering a sinister neurosurgical conspiracy."
    },
    {
        "id": 24,
        "category": "indirect",
        "genre": "horror",
        "expected_id": 39,
        "expected_title": "A Quiet Place",
        "expected_year": "2018",
        "query": "A family must navigate daily life in total silence using sign language to avoid blind extraterrestrial predators that hunt purely by sound."
    },
    {
        "id": 25,
        "category": "indirect",
        "genre": "thriller",
        "expected_id": 25,
        "expected_title": "The Silence of the Lambs",
        "expected_year": "1991",
        "query": "An FBI trainee seeks psychological insights from an incarcerated cannibalistic psychiatrist to apprehend an active serial killer targeting women."
    },

    # -------------------------------------------------------------
    # 4. Short Queries (High Semantic Density, 4 to 7 Words)
    # -------------------------------------------------------------
    {
        "id": 26,
        "category": "short_query",
        "genre": "sci-fi",
        "expected_id": 15,
        "expected_title": "Alien",
        "expected_year": "1979",
        "query": "Extraterrestrial creature stalks spaceship cargo crew."
    },
    {
        "id": 27,
        "category": "short_query",
        "genre": "thriller",
        "expected_id": 22,
        "expected_title": "Jaws",
        "expected_year": "1975",
        "query": "Great white shark terrorizes coastal resort."
    },
    {
        "id": 28,
        "category": "short_query",
        "genre": "action",
        "expected_id": 12353,
        "expected_title": "Speed",
        "expected_year": "1994",
        "query": "Bomb on commuter bus explodes under fifty."
    },
    {
        "id": 29,
        "category": "short_query",
        "genre": "comedy",
        "expected_id": 11468,
        "expected_title": "Home Alone",
        "expected_year": "1990",
        "query": "Abandoned boy defends house against burglars."
    },
    {
        "id": 30,
        "category": "short_query",
        "genre": "sci-fi",
        "expected_id": 16544,
        "expected_title": "Gravity",
        "expected_year": "2013",
        "query": "Astronaut adrift in space after shuttle destroyed."
    },
    {
        "id": 31,
        "category": "short_query",
        "genre": "drama",
        "expected_id": 28,
        "expected_title": "Whiplash",
        "expected_year": "2014",
        "query": "Abusive music conservatory instructor pushes drummer."
    },
    {
        "id": 32,
        "category": "short_query",
        "genre": "drama",
        "expected_id": 51,
        "expected_title": "12 Angry Men",
        "expected_year": "1957",
        "query": "Jurors deliberate murder verdict in sweltering room."
    },
    {
        "id": 33,
        "category": "short_query",
        "genre": "drama",
        "expected_id": 12398,
        "expected_title": "Apollo 13",
        "expected_year": "1995",
        "query": "Damaged lunar spacecraft struggles returning safely."
    },
    {
        "id": 34,
        "category": "short_query",
        "genre": "action",
        "expected_id": 48,
        "expected_title": "Gladiator",
        "expected_year": "2000",
        "query": "Betrayed general becomes arena slave seeking vengeance."
    },
    {
        "id": 35,
        "category": "short_query",
        "genre": "western",
        "expected_id": 5696,
        "expected_title": "High Noon",
        "expected_year": "1952",
        "query": "Marshal stands alone against vengeful outlaws."
    },

    # -------------------------------------------------------------
    # 5. Long Queries (Elaborate Multi-Clause Prose, 25-45 Words)
    # -------------------------------------------------------------
    {
        "id": 36,
        "category": "long_query",
        "genre": "drama",
        "expected_id": 13,
        "expected_title": "The Shawshank Redemption",
        "expected_year": "1994",
        "query": "A successful banker wrongfully convicted of murdering his wife and her lover is sentenced to life imprisonment in a brutal state penitentiary, where he uses his accounting skills and quiet patience over two decades to plan a daring escape."
    },
    {
        "id": 37,
        "category": "long_query",
        "genre": "drama",
        "expected_id": 12230,
        "expected_title": "Forrest Gump",
        "expected_year": "1994",
        "query": "A kind-hearted man from Alabama with a low intelligence quotient inadvertently witnesses and influences several defining historical events of the twentieth century while continually longing for his childhood sweetheart."
    },
    {
        "id": 38,
        "category": "long_query",
        "genre": "drama",
        "expected_id": 11200,
        "expected_title": "Dead Poets Society",
        "expected_year": "1989",
        "query": "An eccentric and unorthodox English literature teacher at an elite conservative boys boarding school inspires his reluctant pupils through poetry to challenge rigid conformity and seize the day."
    },
    {
        "id": 39,
        "category": "long_query",
        "genre": "drama",
        "expected_id": 12110,
        "expected_title": "Schindler's List",
        "expected_year": "1993",
        "query": "A self-interested German industrialist and war profiteer in occupied Poland gradually transforms his enamelware factory into a safe refuge to protect over a thousand Jewish laborers from extermination in concentration camps."
    },
    {
        "id": 40,
        "category": "long_query",
        "genre": "romance",
        "expected_id": 20,
        "expected_title": "Titanic",
        "expected_year": "1997",
        "query": "A penniless itinerant artist wins passage on a luxurious British passenger liner and falls in love with an aristocratic young woman betrothed to an arrogant heir before the vessel collides with an iceberg."
    },
    {
        "id": 41,
        "category": "long_query",
        "genre": "romance",
        "expected_id": 29,
        "expected_title": "La La Land",
        "expected_year": "2016",
        "query": "An aspiring jazz pianist and an ambitious auditioning actress meet and fall in love in contemporary Los Angeles while striving to achieve their creative dreams, only for their blossoming careers to tear them in different directions."
    },
    {
        "id": 42,
        "category": "long_query",
        "genre": "romance",
        "expected_id": 12410,
        "expected_title": "Before Sunrise",
        "expected_year": "1995",
        "query": "An American tourist and a French university student meet by chance on a European passenger train and decide to disembark together in Vienna, spending one unforgettable evening walking the streets and conversing about life and love."
    },
    {
        "id": 43,
        "category": "long_query",
        "genre": "action",
        "expected_id": 37,
        "expected_title": "Mad Max: Fury Road",
        "expected_year": "2015",
        "query": "In a scorched post-apocalyptic wasteland, a taciturn drifter reluctantly joins forces with an imperious female rebel who has hijacked an armored war rig carrying the enslaved wives of a tyrannical water warlord."
    },
    {
        "id": 44,
        "category": "long_query",
        "genre": "action",
        "expected_id": 30,
        "expected_title": "The Dark Knight",
        "expected_year": "2008",
        "query": "A costumed vigilante, an honest police lieutenant, and a crusading district attorney attempt to dismantle organized crime in a decaying metropolis, but their efforts are plunged into violent chaos by an anarchic sociopath wearing clown makeup."
    },
    {
        "id": 45,
        "category": "long_query",
        "genre": "action",
        "expected_id": 10984,
        "expected_title": "Die Hard",
        "expected_year": "1988",
        "query": "An off-duty New York City police detective visits his estranged wife at her corporate office building Christmas Eve party, only to find the high-rise skyscraper seized by a ruthless band of heavily armed German terrorists."
    },

    # -------------------------------------------------------------
    # 6. Diverse Genres (Westerns, Crime, Noir, Animation, Fantasy)
    # -------------------------------------------------------------
    {
        "id": 46,
        "category": "genre_diversity",
        "genre": "western",
        "expected_id": 16266,
        "expected_title": "Django Unchained",
        "expected_year": "2012",
        "query": "A liberated slave partners with an eccentric German dentist bounty hunter to rescue his enslaved wife from a brutal Mississippi cotton plantation."
    },
    {
        "id": 47,
        "category": "genre_diversity",
        "genre": "western",
        "expected_id": 11964,
        "expected_title": "Unforgiven",
        "expected_year": "1992",
        "query": "An aging widower and retired outlaw reluctantly picks up his firearms one last time to collect a reward on two cowboys who slashed a prostitute's face."
    },
    {
        "id": 48,
        "category": "genre_diversity",
        "genre": "crime",
        "expected_id": 8847,
        "expected_title": "The Godfather",
        "expected_year": "1972",
        "query": "The reluctant youngest son of an Italian-American organized crime family is drawn into a vicious gangland conflict after an assassination attempt on his patriarch father."
    },
    {
        "id": 49,
        "category": "genre_diversity",
        "genre": "crime",
        "expected_id": 11454,
        "expected_title": "Goodfellas",
        "expected_year": "1990",
        "query": "A Brooklyn youngster rises through the ranks of the local mob hierarchy over three decades, reveling in luxury and violent privileges before cocaine addiction and paranoia lead to his downfall."
    },
    {
        "id": 50,
        "category": "genre_diversity",
        "genre": "crime",
        "expected_id": 26,
        "expected_title": "Pulp Fiction",
        "expected_year": "1994",
        "query": "The intersecting chronicles of two philosophical hitmen, a prizefighter who refuses to take a fall, a gangster's heroin-overdosing wife, and a pair of diner bandits in Los Angeles."
    },
    {
        "id": 51,
        "category": "genre_diversity",
        "genre": "animation",
        "expected_id": 14,
        "expected_title": "Spirited Away",
        "expected_year": "2001",
        "query": "A sullen ten-year-old girl wanders into an abandoned amusement park that transforms at dusk into a bathhouse for spirits, where she must labor to free her parents after they are turned into pigs."
    },

    # -------------------------------------------------------------
    # 7. Short / Incomplete Plot Descriptions (Plots < 200 Characters)
    # -------------------------------------------------------------
    {
        "id": 52,
        "category": "short_plot",
        "genre": "horror",
        "expected_id": 86,
        "expected_title": "Frankenstein",
        "expected_year": "1910",
        "query": "A medical student creates a living creature through chemical alchemy, which becomes jealous and haunts his wedding."
    },
    {
        "id": 53,
        "category": "short_plot",
        "genre": "fantasy",
        "expected_id": 150,
        "expected_title": "Neptune's Daughter",
        "expected_year": "1914",
        "query": "A mythological sea princess assumes human form on land to avenge the destruction of her aquatic kingdom by mortal fishermen."
    },
    {
        "id": 54,
        "category": "short_plot",
        "genre": "war",
        "expected_id": 175,
        "expected_title": "Four Feathers",
        "expected_year": "1915",
        "query": "A British officer branded a coward by his companions and sweetheart redeems his honor during a perilous African military campaign."
    },
    {
        "id": 55,
        "category": "short_plot",
        "genre": "comedy",
        "expected_id": 65,
        "expected_title": "The Suburbanite",
        "expected_year": "1904",
        "query": "A city family relocates to the countryside seeking peace but encounters endless domestic chaos and misadventures with neighborhood tradesmen."
    }
]


def run_benchmark(top_k: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """Execute the independent semantic search validation benchmark."""
    print("=" * 80)
    print("CINETRACE PHASE 5.6.2 - INDEPENDENT SEMANTIC SEARCH VALIDATION")
    print("=" * 80)
    print(f"Total labelled test queries: {len(BENCHMARK_QUERIES)}")
    print(f"Top-K evaluation window:    {top_k}")
    print("Initializing SceneMatcher and pre-loading catalogue...")

    matcher = SceneMatcher()
    sem_svc = matcher.semantic_service

    if sem_svc and not sem_svc.is_model_loaded():
        print(f"Loading local model '{sem_svc.model_name}' offline from cache...")
        sem_svc.load_model_sync(allow_download=False)

    if not sem_svc or not sem_svc.is_ready():
        print("[ERROR] Semantic service is not available or model is not loaded!")
        sys.exit(1)

    print(f"Catalogue size: {len(matcher.records):,} movies.")
    print("Starting evaluation across all test cases...\n")

    query_results = []
    
    # Metrics accumulators
    pure_ranks: List[Optional[int]] = []
    hybrid_ranks: List[Optional[int]] = []
    pure_latencies: List[float] = []
    hybrid_latencies: List[float] = []
    
    improved_count = 0
    preserved_count = 0
    degraded_count = 0

    for idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q_id = item["id"]
        q_text = item["query"]
        exp_id = item["expected_id"]
        exp_title = item["expected_title"]
        exp_year = item["expected_year"]
        cat = item["category"]

        # 1. Pure Semantic Ranking (Dense Cosine Similarity over top 500)
        t0 = time.perf_counter()
        sem_scores_map = sem_svc.search_by_query(q_text, top_k=500)
        pure_lat = (time.perf_counter() - t0) * 1000.0
        pure_latencies.append(pure_lat)

        # Find expected movie's pure semantic rank and score
        pure_rank = None
        raw_sem_score = 0.0
        for rank_pos, (m_id, sim) in enumerate(sem_scores_map.items(), 1):
            if int(m_id) == exp_id:
                pure_rank = rank_pos
                raw_sem_score = float(sim)
                break
        pure_ranks.append(pure_rank)

        # 2. Hybrid Search (Full Pipeline: Dense Semantic + Calibrated Lexical BM25/TF-IDF)
        t1 = time.perf_counter()
        hybrid_results = matcher.search(
            query=q_text,
            mode="hybrid",
            top_k=500,  # Search full candidate pool
            min_threshold=0.01,
            semantic_top_k=500
        )
        hybrid_lat = (time.perf_counter() - t1) * 1000.0
        hybrid_latencies.append(hybrid_lat)

        # Find expected movie's hybrid rank and score
        hybrid_rank = None
        raw_lex_score = 0.0
        cal_lex_score = 0.0
        hybrid_score = 0.0
        match_type = "none"
        score_label = "N/A"

        for rank_pos, cand in enumerate(hybrid_results, 1):
            cand_id = cand.get("id")
            cand_title = cand.get("title", "")
            cand_year = str(cand.get("year", ""))
            
            # Match by ID or exact title & year
            if (cand_id is not None and int(cand_id) == exp_id) or \
               (cand_title.lower() == exp_title.lower() and cand_year == exp_year):
                hybrid_rank = rank_pos
                raw_lex_score = float(cand.get("raw_lexical_score", 0.0))
                cal_lex_score = float(cand.get("calibrated_lexical_score", 0.0))
                hybrid_score = float(cand.get("similarity", 0.0))
                if not raw_sem_score:
                    raw_sem_score = float(cand.get("raw_semantic_score", 0.0))
                match_type = cand.get("match_type", "unknown")
                score_label = cand.get("score_label", "")
                break

        hybrid_ranks.append(hybrid_rank)

        # Comparison analysis
        if hybrid_rank is not None and pure_rank is not None:
            if hybrid_rank < pure_rank:
                improved_count += 1
                comparison = "IMPROVED"
            elif hybrid_rank == pure_rank:
                preserved_count += 1
                comparison = "PRESERVED"
            else:
                degraded_count += 1
                comparison = "DEGRADED"
        elif hybrid_rank is not None and pure_rank is None:
            improved_count += 1
            comparison = "IMPROVED (Lexical surfaced out-of-pool)"
        elif hybrid_rank is None and pure_rank is not None:
            degraded_count += 1
            comparison = "DEGRADED (Lost in fusion)"
        else:
            preserved_count += 1
            comparison = "NOT FOUND"

        # Record top-10 candidates for this query
        top_10 = []
        for cand in hybrid_results[:10]:
            top_10.append({
                "id": cand.get("id"),
                "title": cand.get("title"),
                "year": cand.get("year"),
                "score": cand.get("similarity"),
                "label": cand.get("score_label"),
                "type": cand.get("match_type")
            })

        query_record = {
            "query_id": q_id,
            "category": cat,
            "genre": item["genre"],
            "expected_id": exp_id,
            "expected_title": exp_title,
            "expected_year": exp_year,
            "query_text": q_text,
            "pure_semantic_rank": pure_rank,
            "raw_semantic_score": round(raw_sem_score, 4),
            "raw_lexical_score": round(raw_lex_score, 4),
            "calibrated_lexical_score": round(cal_lex_score, 4),
            "hybrid_rank": hybrid_rank,
            "hybrid_score": round(hybrid_score, 4),
            "match_type": match_type,
            "score_label": score_label,
            "pure_latency_ms": round(pure_lat, 2),
            "hybrid_latency_ms": round(hybrid_lat, 2),
            "comparison": comparison,
            "top_10_results": top_10
        }
        query_results.append(query_record)

        if verbose:
            p_str = f"#{pure_rank}" if pure_rank else ">500"
            h_str = f"#{hybrid_rank}" if hybrid_rank else ">50"
            print(f"[{q_id:02d}/55] [{cat:15s}] '{exp_title} ({exp_year})'")
            print(f"     Pure Sem: {p_str} ({raw_sem_score:.4f}) | Hybrid: {h_str} ({hybrid_score:.4f}) [{comparison}] | Lat: {hybrid_lat:.1f}ms")

    # Aggregate Evaluation Metrics Calculation
    total_q = len(BENCHMARK_QUERIES)

    def calc_metrics(ranks: List[Optional[int]]) -> Dict[str, float]:
        hits_1 = sum(1 for r in ranks if r is not None and r <= 1)
        hits_5 = sum(1 for r in ranks if r is not None and r <= 5)
        hits_10 = sum(1 for r in ranks if r is not None and r <= 10)
        zero_res = sum(1 for r in ranks if r is None)
        
        mrr_sum = sum(1.0 / r for r in ranks if r is not None and r > 0)
        mrr = mrr_sum / total_q if total_q > 0 else 0.0

        return {
            "hit_rate_at_1": round(hits_1 / total_q, 4),
            "recall_at_5": round(hits_5 / total_q, 4),
            "recall_at_10": round(hits_10 / total_q, 4),
            "mrr": round(mrr, 4),
            "zero_result_rate": round(zero_res / total_q, 4)
        }

    pure_metrics = calc_metrics(pure_ranks)
    hybrid_metrics = calc_metrics(hybrid_ranks)

    # Category breakdown
    categories = sorted(list(set(q["category"] for q in BENCHMARK_QUERIES)))
    category_breakdown = {}
    for c in categories:
        c_hybrid_ranks = [r["hybrid_rank"] for r in query_results if r["category"] == c]
        c_pure_ranks = [r["pure_semantic_rank"] for r in query_results if r["category"] == c]
        c_count = len(c_hybrid_ranks)
        c_hits_1 = sum(1 for r in c_hybrid_ranks if r is not None and r <= 1)
        c_hits_5 = sum(1 for r in c_hybrid_ranks if r is not None and r <= 5)
        c_hits_10 = sum(1 for r in c_hybrid_ranks if r is not None and r <= 10)
        c_mrr = sum(1.0 / r for r in c_hybrid_ranks if r is not None and r > 0) / c_count if c_count else 0.0
        category_breakdown[c] = {
            "count": c_count,
            "hit_rate_at_1": round(c_hits_1 / c_count, 4),
            "recall_at_5": round(c_hits_5 / c_count, 4),
            "recall_at_10": round(c_hits_10 / c_count, 4),
            "mrr": round(c_mrr, 4)
        }

    summary = {
        "benchmark_name": "CineTrace Phase 5.6.2 Independent Semantic Search Validation",
        "catalogue_size": len(matcher.records),
        "total_queries": total_q,
        "pure_semantic_metrics": pure_metrics,
        "hybrid_metrics": hybrid_metrics,
        "hybrid_vs_pure_comparison": {
            "improved": improved_count,
            "preserved": preserved_count,
            "degraded": degraded_count,
            "improved_or_preserved_pct": round((improved_count + preserved_count) / total_q * 100.0, 2)
        },
        "latencies_ms": {
            "pure_semantic_mean": round(sum(pure_latencies) / len(pure_latencies), 2),
            "pure_semantic_p95": round(sorted(pure_latencies)[int(len(pure_latencies) * 0.95)], 2),
            "hybrid_mean": round(sum(hybrid_latencies) / len(hybrid_latencies), 2),
            "hybrid_p95": round(sorted(hybrid_latencies)[int(len(hybrid_latencies) * 0.95)], 2),
        },
        "category_breakdown": category_breakdown,
        "query_results": query_results
    }

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"Metric                | Pure Semantic | Hybrid Fusion | Delta")
    print("-" * 65)
    print(f"Hit Rate@1 (Top-1)    | {pure_metrics['hit_rate_at_1']*100:6.1f}%       | {hybrid_metrics['hit_rate_at_1']*100:6.1f}%       | {(hybrid_metrics['hit_rate_at_1']-pure_metrics['hit_rate_at_1'])*100:+5.1f}%")
    print(f"Recall@5              | {pure_metrics['recall_at_5']*100:6.1f}%       | {hybrid_metrics['recall_at_5']*100:6.1f}%       | {(hybrid_metrics['recall_at_5']-pure_metrics['recall_at_5'])*100:+5.1f}%")
    print(f"Recall@10             | {pure_metrics['recall_at_10']*100:6.1f}%       | {hybrid_metrics['recall_at_10']*100:6.1f}%       | {(hybrid_metrics['recall_at_10']-pure_metrics['recall_at_10'])*100:+5.1f}%")
    print(f"MRR                   | {pure_metrics['mrr']:6.3f}        | {hybrid_metrics['mrr']:6.3f}        | {hybrid_metrics['mrr']-pure_metrics['mrr']:+6.3f}")
    print(f"Zero-Result Rate      | {pure_metrics['zero_result_rate']*100:6.1f}%       | {hybrid_metrics['zero_result_rate']*100:6.1f}%       | {(hybrid_metrics['zero_result_rate']-pure_metrics['zero_result_rate'])*100:+5.1f}%")
    print(f"Mean Latency (ms)     | {summary['latencies_ms']['pure_semantic_mean']:6.1f} ms    | {summary['latencies_ms']['hybrid_mean']:6.1f} ms    | {summary['latencies_ms']['hybrid_mean']-summary['latencies_ms']['pure_semantic_mean']:+6.1f} ms")
    print("-" * 65)
    print(f"Hybrid vs Pure Semantic: Improved: {improved_count} | Preserved: {preserved_count} | Degraded: {degraded_count}")
    print(f"Hybrid Improvement / Preservation Rate: {summary['hybrid_vs_pure_comparison']['improved_or_preserved_pct']}%")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run CineTrace Phase 5.6.2 Semantic Search Benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K evaluation cutoff")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, default="benchmark_semantic_validation_results.json", help="Path to save JSON output")
    args = parser.parse_args()

    results = run_benchmark(top_k=args.top_k, verbose=not args.json)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved benchmark results to {out_path.resolve()}]")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
