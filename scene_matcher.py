"""
CineTrace — Scene-to-Movie Finder Hybrid Matching Engine (Phase 5.2)
Implements local hybrid search across movie plot descriptions, titles, and cast/actor names.
Features:
- Multi-signal matching: TF-IDF cosine similarity for plots, exact/partial title matching,
  cast/actor name matching, and SQLite FTS5 integration.
- Supports search modes: 'hybrid', 'plot', 'title', 'actor'.
- Multiple filters: Genre, Year Min, Year Max, and Configurable Min Match Threshold.
- Precomputed index built once on catalogue load and updated on import.
- Optional semantic embedding interface with zero-dependency fallback.
- OMDb enrichment with local SQLite caching and offline resilience.
"""

import csv
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import config
import database
import catalogue_service
import semantic_service

# Standard English stopwords for the built-in fallback tokenizer
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Optional scikit-learn dependency
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DatasetError(Exception):
    """Raised when there is an issue loading or parsing the plot dataset."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when the specified dataset file does not exist."""
    pass


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and replacing non-alphanumeric chars with spaces."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return cleaned.lower().strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into alphanumeric words of length >= 2, optionally removing stopwords."""
    norm = normalize_text(text)
    tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", norm)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return tokens


def extract_year_int(year_str: Any) -> Optional[int]:
    """Safely extract 4-digit integer year from string or int."""
    if not year_str:
        return None
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(year_str))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def load_plot_dataset(file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Load and validate a local movie-plot dataset from JSON or CSV.
    Returns a list of validated movie dictionaries with non-empty titles and plots.
    Safely ignores malformed records.
    """
    path = Path(file_path) if file_path else config.PLOT_DATASET_FILE
    if not path.exists():
        raise DatasetNotFoundError(
            f"Plot dataset file not found at '{path}'. Please ensure a dataset file is present."
        )

    records: List[Dict[str, Any]] = []

    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raise DatasetError(f"Expected JSON root to be a list of movie objects, got {type(raw_data).__name__}.")
            items = raw_data
        elif suffix in (".csv", ".tsv", ".txt"):
            items = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(2048)
                f.seek(0)
                delim = "\t" if "\t" in sample and sample.count("\t") > sample.count(",") else ","
                reader = csv.DictReader(f, delimiter=delim)
                for row in reader:
                    items.append(row)
        else:
            raise DatasetError(f"Unsupported dataset format '{suffix}'. Supported formats are .json and .csv.")
    except json.JSONDecodeError as e:
        raise DatasetError(f"Failed to parse JSON dataset '{path.name}': {e}")
    except Exception as e:
        if isinstance(e, DatasetError):
            raise
        raise DatasetError(f"Failed to read dataset from '{path}': {e}")

    # Validate and clean records
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or item.get("Title") or "").strip()
        plot = (item.get("plot") or item.get("Plot") or item.get("plot_summary") or "").strip()

        # Both title and plot must be non-empty strings
        if not title or not plot:
            continue

        imdb_id = (item.get("imdb_id") or item.get("imdbID") or "").strip()
        year = str(item.get("year") or item.get("Year") or "N/A").strip()
        genre = str(item.get("genre") or item.get("Genre") or "N/A").strip()
        actors = str(item.get("actors") or item.get("Actors") or item.get("cast") or item.get("Cast") or "N/A").strip()
        director = str(item.get("director") or item.get("Director") or "N/A").strip()
        alternate_titles = str(item.get("alternate_titles") or item.get("originalTitle") or "").strip()
        source_prov = str(item.get("source_provenance") or "").strip()
        poster_url = str(item.get("poster_url") or item.get("poster") or item.get("Poster") or "N/A").strip()

        records.append({
            "imdb_id": imdb_id,
            "title": title,
            "alternate_titles": alternate_titles,
            "year": year,
            "genre": genre,
            "plot": plot,
            "actors": actors,
            "director": director,
            "source_provenance": source_prov,
            "poster_url": poster_url if poster_url and poster_url != "N/A" else "N/A",
        })

    return records


class PurePythonTfidf:
    """
    Standard-library TF-IDF vectorizer and cosine similarity engine.
    Ensures robust, zero-dependency text similarity matching when scikit-learn is unavailable.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def fit_transform(self, documents: List[str]):
        """Fit on documents and precompute doc vectors, norms, and inverted index."""
        n_docs = len(documents)
        df: Counter = Counter()
        tokenized_docs: List[List[str]] = []

        for doc in documents:
            tokens = tokenize(doc, remove_stopwords=True)
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        # Build vocabulary & calculate smoothed IDF: log((1 + n) / (1 + df)) + 1
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self.idf = {
            term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for term, count in df.items()
        }

        self.doc_vectors = []
        self.doc_norms = []
        self.inverted_index = defaultdict(list)

        for doc_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                self.doc_vectors.append({})
                self.doc_norms.append(0.0)
                continue

            tf = Counter(tokens)
            vec: Dict[str, float] = {}
            sum_sq = 0.0
            for term, count in tf.items():
                if term in self.idf:
                    weight = (count / len(tokens)) * self.idf[term]
                    vec[term] = weight
                    sum_sq += weight * weight
                    self.inverted_index[term].append((doc_idx, weight))
            norm = math.sqrt(sum_sq)
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    def transform_query(self, query: str) -> Tuple[Dict[str, float], float]:
        """Compute TF-IDF vector and norm for a query string."""
        tokens = tokenize(query, remove_stopwords=True)
        if not tokens:
            return {}, 0.0

        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        sum_sq = 0.0
        for term, count in tf.items():
            if term in self.idf:
                weight = (count / len(tokens)) * self.idf[term]
                vec[term] = weight
                sum_sq += weight * weight
        return vec, math.sqrt(sum_sq)

    def score_all(self, query: str) -> List[float]:
        """Calculate cosine similarity using the inverted index for sub-millisecond ranking."""
        q_vec, q_norm = self.transform_query(query)
        n_docs = len(self.doc_norms)
        if q_norm == 0.0 or n_docs == 0:
            return [0.0] * n_docs

        # Accumulate dot products across only documents containing query terms
        dot_products = [0.0] * n_docs
        for term, q_val in q_vec.items():
            postings = self.inverted_index.get(term)
            if postings:
                for doc_idx, d_val in postings:
                    dot_products[doc_idx] += q_val * d_val

        scores: List[float] = [0.0] * n_docs
        for i in range(n_docs):
            d_norm = self.doc_norms[i]
            if d_norm > 0.0 and dot_products[i] > 0.0:
                sim = dot_products[i] / (q_norm * d_norm)
                scores[i] = max(0.0, min(1.0, sim))
        return scores


class OptionalSemanticEngine:
    """
    Optional semantic embedding engine for dense natural-language matching.
    Only activates if an embedding library (e.g., sentence-transformers) is explicitly installed and enabled.
    Falls back gracefully to TF-IDF with zero dependencies.
    Never downloads models silently.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.enabled = False
        self.model = None

    @property
    def is_available(self) -> bool:
        """Return True only if sentence_transformers package is locally importable."""
        try:
            import sentence_transformers # noqa: F401
            return True
        except ImportError:
            return False

    def enable(self) -> bool:
        """Enable semantic embedding engine if package is available."""
        if not self.is_available:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.enabled = True
            return True
        except Exception:
            self.enabled = False
            return False

    def encode(self, texts: List[str]) -> Optional[Any]:
        if not self.enabled or self.model is None:
            return None
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception:
            return None


class SceneMatcher:
    """
    Hybrid movie discovery matching engine.
    Matches user queries across:
    1. Plot descriptions (TF-IDF cosine similarity and optional dense semantic embeddings)
    2. Exact and partial movie titles
    3. Actor and cast names
    4. SQLite FTS5 full-text keyword indexing
    Applies genre/year filters and transparent scoring.
    """

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        file_path: Optional[Union[str, Path]] = None,
        db_path: Optional[Path] = None,
        force_pure_python: bool = False,
        use_sqlite_catalogue: bool = True,
        auto_seed: bool = True,
        semantic_service_instance: Optional[Any] = None
    ):
        self.force_pure_python = force_pure_python
        self.file_path = file_path
        self.db_path = db_path
        self.use_sqlite_catalogue = use_sqlite_catalogue
        self.auto_seed = auto_seed
        self.records: List[Dict[str, Any]] = []
        self._title_year_to_idx: Dict[Tuple[str, str], int] = {}
        self._sklearn_vectorizer: Optional[Any] = None
        self._sklearn_matrix: Optional[Any] = None
        self._pure_engine: Optional[PurePythonTfidf] = None
        self.semantic_engine = OptionalSemanticEngine()
        self.semantic_service = semantic_service_instance or semantic_service.get_semantic_service(db_path=self.db_path)

        if records is not None:
            self.set_records(records)
        elif file_path is not None:
            self.load(file_path)
        elif self.use_sqlite_catalogue:
            self.load_from_catalogue(self.db_path, auto_seed=auto_seed)

    def load_from_catalogue(self, db_path: Optional[Path] = None, auto_seed: bool = True):
        """
        Load records from SQLite catalogue table.
        Automatically seeds catalogue from data/movies_plots.json on first run if empty and auto_seed is True.
        """
        if auto_seed:
            catalogue_service.ensure_catalogue_initialized(db_path=db_path)
        cat_movies = database.get_catalogue_movies(db_path=db_path)
        if cat_movies:
            self.set_records(cat_movies)
        elif auto_seed and config.PLOT_DATASET_FILE.exists():
            self.load(config.PLOT_DATASET_FILE)

    def load(self, file_path: Optional[Union[str, Path]] = None):
        """Load dataset from file path and initialize matching index."""
        target_path = file_path or self.file_path or config.PLOT_DATASET_FILE
        self.file_path = target_path
        records = load_plot_dataset(target_path)
        self.set_records(records)

    def reload(self):
        """Reload catalogue records from SQLite and rebuild search index."""
        if self.use_sqlite_catalogue:
            cat_movies = database.get_catalogue_movies(db_path=self.db_path)
            if cat_movies:
                self.set_records(cat_movies)
                return
        if self.file_path:
            self.load(self.file_path)

    def set_records(self, records: List[Dict[str, Any]]):
        """Build search index over a list of movie records with fast pre-parsed attributes."""
        cleaned: List[Dict[str, Any]] = []
        title_year_map: Dict[Tuple[str, str], int] = {}
        _person_cache: Dict[str, Tuple[str, str, str]] = {}

        for r in records:
            title = (r.get("title") or "").strip()
            plot = (r.get("plot") or "").strip()
            if title or plot:
                # Pre-normalize title and alternate titles in-place
                r["_norm_title"] = normalize_text(title)
                r["_norm_alt"] = normalize_text(r.get("alternate_titles", ""))
                r["_year_int"] = extract_year_int(r.get("year"))

                # Pre-parse and normalize cast and director names with shared person cache
                actors_str = str(r.get("actors") or "")
                director_str = str(r.get("director") or "")
                raw_people = []
                if actors_str and actors_str != "N/A":
                    raw_people.extend([p.strip() for p in actors_str.split(",") if p.strip()])
                if director_str and director_str != "N/A":
                    raw_people.extend([d.strip() for d in director_str.split(",") if d.strip()])

                parsed_people = []
                for p in raw_people:
                    cached_entry = _person_cache.get(p)
                    if cached_entry is not None:
                        parsed_people.append(cached_entry)
                        continue
                    p_norm = normalize_text(p)
                    if p_norm and len(p_norm) >= 4:
                        toks = p_norm.split()
                        surname = toks[-1] if len(toks) >= 2 and len(toks[-1]) >= 4 else ""
                        entry = (p, p_norm, surname)
                        _person_cache[p] = entry
                        parsed_people.append(entry)
                r["_people"] = parsed_people

                idx = len(cleaned)
                cleaned.append(r)
                k = (title.lower(), str(r.get("year") or "").strip())
                title_year_map[k] = idx

        _person_cache.clear()
        del _person_cache

        self.records = cleaned
        self._title_year_to_idx = title_year_map

        if not self.records:
            self._sklearn_vectorizer = None
            self._sklearn_matrix = None
            self._pure_engine = None
            return

        if SKLEARN_AVAILABLE and not self.force_pure_python:
            loaded_from_cache = False
            # Check if pre-fitted TF-IDF cache file exists and matches current catalogue size
            if config.TFIDF_CACHE_FILE.exists():
                try:
                    with open(config.TFIDF_CACHE_FILE, "rb") as f:
                        cache_data = pickle.load(f)
                    if (
                        isinstance(cache_data, dict)
                        and cache_data.get("count") == len(self.records)
                        and cache_data.get("matrix") is not None
                        and cache_data.get("vectorizer") is not None
                    ):
                        self._sklearn_vectorizer = cache_data["vectorizer"]
                        self._sklearn_matrix = cache_data["matrix"]
                        loaded_from_cache = True
                    del cache_data
                except Exception:
                    loaded_from_cache = False

            if not loaded_from_cache:
                documents = [r.get("plot", "") or r.get("title", "") for r in self.records]
                try:
                    self._sklearn_vectorizer = TfidfVectorizer(
                        stop_words="english",
                        preprocessor=normalize_text,
                        token_pattern=r"(?u)\b\w+\b",
                        ngram_range=(1, 2),
                        norm="l2",
                        sublinear_tf=True
                    )
                    self._sklearn_matrix = self._sklearn_vectorizer.fit_transform(documents)

                    # Persist cache for large catalogues to ensure instantaneous warm startups
                    if len(self.records) > 1000:
                        try:
                            config.TFIDF_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                            tmp_cache = config.TFIDF_CACHE_FILE.with_suffix(".tmp.pkl")
                            with open(tmp_cache, "wb") as f:
                                pickle.dump({
                                    "vectorizer": self._sklearn_vectorizer,
                                    "matrix": self._sklearn_matrix,
                                    "count": len(self.records)
                                }, f, protocol=pickle.HIGHEST_PROTOCOL)
                            tmp_cache.replace(config.TFIDF_CACHE_FILE)
                        except Exception:
                            pass
                except Exception:
                    self._sklearn_vectorizer = None
                    self._pure_engine = PurePythonTfidf()
                    self._pure_engine.fit_transform(documents)
                finally:
                    del documents
        else:
            documents = [r.get("plot", "") or r.get("title", "") for r in self.records]
            self._pure_engine = PurePythonTfidf()
            self._pure_engine.fit_transform(documents)
            del documents

    def is_ready(self) -> bool:
        """Return True if dataset has records and index is built."""
        return len(self.records) > 0

    def get_indexed_count(self) -> int:
        """Return total count of indexed movies."""
        return len(self.records)

    def search(
        self,
        query: str,
        genre: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_threshold: float = 0.10,
        top_k: int = 12,
        mode: str = "hybrid",
        semantic_top_k: int = 500,
        force_lexical: bool = False,
        rerank: bool = False,
        rerank_top_k: int = 25,
        cross_encoder_service: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search across movie plots, titles, and cast members.
        Supported modes:
        - 'hybrid' / 'all': Intelligently combines title, cast, and plot signals.
        - 'plot' / 'scene': TF-IDF cosine similarity for scene/plot descriptions.
        - 'title': Exact and token matching for movie titles.
        - 'actor' / 'cast': Matching against actor and director names.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query or not self.records:
            return []

        norm_q = normalize_text(cleaned_query)
        q_tokens = tokenize(cleaned_query, remove_stopwords=False)
        q_keywords = tokenize(cleaned_query, remove_stopwords=True)
        norm_mode = (mode or "hybrid").strip().lower()

        # 1. Compute raw plot similarity scores
        plot_scores = [0.0] * len(self.records)
        if norm_mode in ("hybrid", "all", "plot", "scene"):
            if self._sklearn_vectorizer is not None and self._sklearn_matrix is not None:
                try:
                    q_vec = self._sklearn_vectorizer.transform([cleaned_query])
                    sims = cosine_similarity(q_vec, self._sklearn_matrix)[0]
                    plot_scores = [float(s) for s in sims]
                except Exception:
                    engine = PurePythonTfidf()
                    engine.fit_transform([r.get("plot", "") or r.get("title", "") for r in self.records])
                    plot_scores = engine.score_all(cleaned_query)
            elif self._pure_engine is not None:
                plot_scores = self._pure_engine.score_all(cleaned_query)

            # SQLite FTS5 BM25 keyword signal fusion (O(1) mapping via precomputed index)
            if len(self.records) > 0:
                try:
                    fts_matches = database.search_catalogue_fts(
                        cleaned_query,
                        mode="plot" if norm_mode in ("plot", "scene") else "all",
                        limit=30,
                        db_path=self.db_path
                    )
                    if fts_matches:
                        n_matches = len(fts_matches)
                        for rank_idx, fm in enumerate(fts_matches):
                            k = (fm.get("title", "").strip().lower(), str(fm.get("year", "")).strip())
                            boost = 0.08 * (1.0 - (rank_idx / max(1, n_matches)))
                            idx = self._title_year_to_idx.get(k)
                            if idx is not None and idx < len(plot_scores):
                                cur_score = plot_scores[idx]
                                if cur_score >= 0.08:
                                    plot_scores[idx] = round(cur_score + boost, 4)
                                elif cur_score > 0.0:
                                    plot_scores[idx] = round(cur_score * (1.0 + boost), 4)
                except Exception:
                    pass

        # 1b. Retrieve dense semantic similarities if semantic engine is active and not forced lexical
        sem_scores_map: Dict[int, float] = {}
        if not force_lexical and norm_mode in ("hybrid", "all", "plot", "scene"):
            if self.semantic_service and self.semantic_service.is_ready():
                try:
                    sem_scores_map = self.semantic_service.search_by_query(cleaned_query, top_k=semantic_top_k)
                except Exception:
                    sem_scores_map = {}

        # 2. Filter and rank candidates
        results: List[Dict[str, Any]] = []
        cleaned_genre = (genre or "").strip().lower()
        has_genre_filter = cleaned_genre not in ("", "all")
        is_short_query = len(q_tokens) <= 4
        max_lex_score = max(plot_scores) if plot_scores else 0.0

        for idx, (movie, raw_plot_score) in enumerate(zip(self.records, plot_scores)):
            # Genre filter
            if has_genre_filter:
                m_genre = (movie.get("genre") or "").lower()
                if cleaned_genre not in m_genre:
                    continue

            # Year filter
            if year_min is not None or year_max is not None:
                m_year = movie.get("_year_int")
                if m_year is None:
                    m_year = extract_year_int(movie.get("year"))
                if m_year is not None:
                    if year_min is not None and m_year < year_min:
                        continue
                    if year_max is not None and m_year > year_max:
                        continue

            title = movie.get("title", "Untitled")
            t_norm = movie.get("_norm_title") or normalize_text(title)
            alt_norm = movie.get("_norm_alt") or normalize_text(movie.get("alternate_titles", ""))

            # A. Title Signal (only computed if in title or hybrid mode)
            title_score = 0.0

            if norm_mode not in ("plot", "scene", "actor", "cast"):
                if norm_q and t_norm:
                    if norm_q == t_norm or (alt_norm and norm_q == alt_norm):
                        title_score = 1.0
                    elif norm_mode == "title" or is_short_query:
                        if t_norm.startswith(norm_q) or (norm_q.startswith(t_norm) and len(t_norm) >= 4):
                            title_score = 0.92
                        elif norm_q in t_norm:
                            title_score = 0.85
                        elif q_tokens and len(q_tokens) > 0:
                            t_toks = set(tokenize(t_norm, remove_stopwords=False))
                            overlap = len(set(q_tokens) & t_toks)
                            if overlap == len(q_tokens):
                                title_score = 0.88
                            elif overlap > 0:
                                jacc = overlap / max(len(q_tokens), len(t_toks))
                                if jacc >= 0.5:
                                    title_score = 0.50 + 0.35 * jacc
                    elif len(t_norm.split()) >= 4 and len(t_norm.split()) >= len(q_tokens) * 0.7:
                        # Long title that forms the dominant portion of the query
                        if t_norm in norm_q and re.search(rf"\b{re.escape(t_norm)}\b", norm_q):
                            title_score = 0.88

            # Early exit for pure title mode if no title match
            if norm_mode == "title" and title_score < min_threshold:
                continue

            # B. Actor / Cast Signal (only computed if in actor, cast, or hybrid mode)
            cast_score = 0.0
            matched_person = ""

            if norm_mode not in ("plot", "scene", "title"):
                people = movie.get("_people")
                if people is not None:
                    for person, p_norm, surname in people:
                        if not p_norm or len(p_norm) < 4:
                            continue
                        p_words = p_norm.split()
                        is_multi_word = len(p_words) >= 2

                        if norm_q == p_norm:
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif is_multi_word and p_norm in norm_q and re.search(rf"\b{re.escape(p_norm)}\b", norm_q):
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif (norm_mode in ("actor", "cast") or is_short_query) and p_norm in norm_q and re.search(rf"\b{re.escape(p_norm)}\b", norm_q):
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif norm_mode in ("actor", "cast") or is_short_query:
                            if norm_q in p_norm and len(norm_q) >= 4:
                                cast_score = max(cast_score, 0.88)
                                matched_person = person
                            elif surname and surname in q_keywords and len(surname) >= 4:
                                cast_score = max(cast_score, 0.80)
                                if not matched_person:
                                    matched_person = person
                else:
                    actors_str = str(movie.get("actors") or "")
                    director_str = str(movie.get("director") or "")
                    raw_people = []
                    if actors_str and actors_str != "N/A":
                        raw_people.extend([p.strip() for p in actors_str.split(",") if p.strip()])
                    if director_str and director_str != "N/A":
                        raw_people.extend([d.strip() for d in director_str.split(",") if d.strip()])
                    for person in raw_people:
                        p_norm = normalize_text(person)
                        if not p_norm or len(p_norm) < 4:
                            continue
                        p_words = p_norm.split()
                        is_multi_word = len(p_words) >= 2

                        if norm_q == p_norm:
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif is_multi_word and p_norm in norm_q and re.search(rf"\b{re.escape(p_norm)}\b", norm_q):
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif (norm_mode in ("actor", "cast") or is_short_query) and p_norm in norm_q and re.search(rf"\b{re.escape(p_norm)}\b", norm_q):
                            cast_score = max(cast_score, 0.95)
                            matched_person = person
                            break
                        elif norm_mode in ("actor", "cast") or is_short_query:
                            if norm_q in p_norm and len(norm_q) >= 4:
                                cast_score = max(cast_score, 0.88)
                                matched_person = person
                            else:
                                p_toks = p_norm.split()
                                if p_toks:
                                    surname = p_toks[-1]
                                    if surname in q_keywords and len(surname) >= 4:
                                        cast_score = max(cast_score, 0.80)
                                        if not matched_person:
                                            matched_person = person

            # Early exit for pure actor mode if no actor match
            if norm_mode in ("actor", "cast") and cast_score < min_threshold:
                continue

            # C. Determine plot score with semantic and calibrated lexical fusion
            cat_id = movie.get("id")
            s_sem = 0.0
            if cat_id is not None:
                s_sem = sem_scores_map.get(int(cat_id), 0.0)
            elif (idx + 1) in sem_scores_map:
                s_sem = sem_scores_map.get(idx + 1, 0.0)

            s_lex = round(raw_plot_score, 4)
            # Calibrate lexical score so it is on the same scale as dense cosine similarity
            s_lex_cal = min(1.0, round(s_lex * 2.0, 4)) if s_lex > 0 else 0.0

            if s_sem > 0.0:
                if s_sem >= 0.30 or (s_sem >= 0.20 and s_lex >= 0.08):
                    # Meaningful semantic similarity or corroborated by solid lexical evidence
                    p_score = round(max(s_sem, 0.70 * s_sem + 0.30 * s_lex_cal), 4)
                    p_type = "semantic"
                    p_details = "Semantic plot match"
                    p_label = f"Semantic Match: {int(round(p_score * 100))}%"
                else:
                    # Below dense vector noise floor without sufficient lexical corroboration
                    p_score = s_lex
                    p_type = "plot"
                    p_details = "Low confidence plot similarity"
                    p_label = f"Plot Match: {int(round(p_score * 100))}%"
            else:
                p_score = s_lex
                p_type = "plot"
                p_details = "Keyword/plot match"
                p_label = f"Plot Match: {int(round(p_score * 100))}%"

            if norm_mode == "title":
                final_score = title_score
                match_type = "title"
                pct = int(round(final_score * 100))
                score_label = f"Title Match: {pct}%"
                match_details = "Title match"

            elif norm_mode in ("actor", "cast"):
                final_score = cast_score
                match_type = "cast"
                pct = int(round(final_score * 100))
                score_label = f"Cast Match: {pct}% ({matched_person})" if matched_person else f"Cast Match: {pct}%"
                match_details = f"Cast match: {matched_person}" if matched_person else "Cast match"

            elif norm_mode in ("plot", "scene"):
                final_score = p_score
                match_type = p_type
                pct = int(round(final_score * 100))
                score_label = p_label
                match_details = p_details

            else:
                # 'hybrid' / 'all' mode: transparent multi-signal fusion
                if title_score >= 0.85 and title_score >= cast_score and title_score >= p_score:
                    final_score = title_score
                    match_type = "title"
                    pct = int(round(final_score * 100))
                    score_label = f"Title Match: {pct}%"
                    match_details = "Title match"
                elif cast_score >= 0.75:
                    if p_score >= 0.10:
                        final_score = round(cast_score * 0.70 + p_score * 0.30, 4)
                        match_type = "hybrid"
                        pct = int(round(final_score * 100))
                        if p_type == "semantic":
                            score_label = f"Hybrid: Cast ({matched_person}) + Semantic Plot ({int(round(p_score * 100))}%)"
                            match_details = f"Cast match ({matched_person}) with semantic plot match"
                        else:
                            score_label = f"Hybrid: Cast ({matched_person}) + Plot ({int(round(p_score * 100))}%)"
                            match_details = f"Cast match ({matched_person}) with keyword plot match"
                    else:
                        final_score = cast_score
                        match_type = "cast"
                        pct = int(round(final_score * 100))
                        score_label = f"Cast Match: {pct}% ({matched_person})" if matched_person else f"Cast Match: {pct}%"
                        match_details = f"Cast match: {matched_person}" if matched_person else "Cast match"
                else:
                    final_score = p_score
                    match_type = p_type
                    pct = int(round(final_score * 100))
                    score_label = p_label
                    match_details = p_details

            # Discard candidates below minimum score threshold
            if final_score < min_threshold:
                continue

            poster_url = movie.get("poster_url") or "N/A"
            rating = movie.get("imdb_rating") or "N/A"
            imdb_id = str(movie.get("imdb_id") or "").strip()

            results.append({
                "id": movie.get("id") if movie.get("id") is not None else (idx + 1),
                "imdb_id": imdb_id,
                "title": title,
                "year": str(movie.get("year") or "N/A"),
                "genre": str(movie.get("genre") or "N/A"),
                "actors": str(movie.get("actors") or "N/A"),
                "director": str(movie.get("director") or "N/A"),
                "plot": str(movie.get("plot") or ""),
                "poster_url": poster_url,
                "imdb_rating": rating,
                "similarity": final_score,
                "similarity_pct": f"{int(round(final_score * 100))}%",
                "score_label": score_label,
                "match_type": match_type,
                "match_details": match_details,
                "source_provenance": str(movie.get("source_provenance") or ""),
                "raw_semantic_score": s_sem,
                "raw_lexical_score": s_lex,
                "calibrated_lexical_score": s_lex_cal,
                "first_stage_score": final_score
            })

        # Rank by first-stage score descending
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Deduplicate candidates by stable catalogue ID
        seen_ids = set()
        deduped: List[Dict[str, Any]] = []
        for cand in results:
            cid = cand.get("id")
            if cid is not None:
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
            else:
                key = (cand.get("title", "").strip().lower(), str(cand.get("year", "")).strip())
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            deduped.append(cand)
        results = deduped

        # Optional Second-Stage Cross-Encoder Reranking (Phase 5.7)
        if rerank:
            ce_svc = cross_encoder_service or semantic_service.get_cross_encoder_service()
            if ce_svc and ce_svc.is_ready():
                results = ce_svc.rerank(
                    query=cleaned_query,
                    candidates=results,
                    top_k=rerank_top_k
                )

        top_results = results[:top_k]


        # Check local SQLite cache first for instant poster and rating on the top_k candidates only
        for cand in top_results:
            c_imdb = cand.get("imdb_id")
            if c_imdb and (cand.get("poster_url") in (None, "", "N/A") or cand.get("imdb_rating") in (None, "", "N/A")):
                try:
                    cached = database.get_movie_by_id(c_imdb, db_path=self.db_path)
                    if cached:
                        if cand.get("poster_url") in (None, "", "N/A"):
                            cand["poster_url"] = cached.get("poster_url") or "N/A"
                        if cand.get("imdb_rating") in (None, "", "N/A"):
                            cand["imdb_rating"] = cached.get("imdb_rating") or "N/A"
                except Exception:
                    pass

        return top_results


# Caches to avoid hammering OMDb and respect rate limits
_FAILED_ENRICHMENT_CACHE: set = set()
_RATE_LIMIT_HIT: bool = False


def clear_enrichment_cache() -> None:
    """Clear failed enrichment cache and reset rate limit flag."""
    global _FAILED_ENRICHMENT_CACHE, _RATE_LIMIT_HIT
    _FAILED_ENRICHMENT_CACHE.clear()
    _RATE_LIMIT_HIT = False


def enrich_candidate(
    candidate: Dict[str, Any],
    omdb_client: Optional[Any] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Enrich a matched candidate movie with poster and rating metadata.
    1. Checks local SQLite cache first for instant retrieval.
    2. If missing and OMDb client is available, attempts verified API enrichment.
    3. Caches failed lookups to prevent redundant API queries.
    4. Halts queries if OMDb rate limit is encountered.
    5. Updates catalogue table with verified enrichment metadata.
    6. If OMDb fails, offline, or returns error, preserves local matching data without loss.
    """
    global _RATE_LIMIT_HIT

    imdb_id = str(candidate.get("imdb_id") or "").strip()
    existing_poster = candidate.get("poster_url") or candidate.get("poster") or "N/A"
    existing_rating = candidate.get("imdb_rating") or "N/A"

    enriched = dict(candidate)
    enriched["poster_url"] = existing_poster
    enriched["imdb_rating"] = existing_rating

    # 1. Check local SQLite cache first (movies table)
    if imdb_id and imdb_id.upper() != "N/A":
        try:
            cached = database.get_movie_by_id(imdb_id, db_path=db_path)
            if cached:
                cached_poster = cached.get("poster_url")
                cached_rating = cached.get("imdb_rating")
                if cached_poster and cached_poster != "N/A":
                    enriched["poster_url"] = cached_poster
                if cached_rating and cached_rating != "N/A":
                    enriched["imdb_rating"] = cached_rating
                enriched["rated"] = cached.get("rated") or candidate.get("rated", "N/A")
                enriched["runtime"] = cached.get("runtime") or candidate.get("runtime", "N/A")
                return enriched
        except Exception:
            pass

    # Stop if rate limit reached or client unavailable
    if omdb_client is None or _RATE_LIMIT_HIT:
        return enriched

    lookup_key = (imdb_id or f"{candidate.get('title', '')}::{candidate.get('year', '')}").strip().lower()
    if lookup_key in _FAILED_ENRICHMENT_CACHE:
        return enriched

    # 2. If not in SQLite, fetch verified data from OMDb
    try:
        data = None
        if imdb_id and imdb_id.upper() != "N/A":
            data = omdb_client.get_movie_details(imdb_id)
            if not data or data.get("error"):
                err_msg = str(data.get("error") if data else "")
                if "request limit" in err_msg.lower():
                    _RATE_LIMIT_HIT = True
                _FAILED_ENRICHMENT_CACHE.add(lookup_key)
                return enriched
        elif candidate.get("title"):
            # Title search fallback with strict title & year verification
            title_clean = candidate["title"].strip()
            cand_year = extract_year_int(candidate.get("year"))
            search_res = omdb_client.search_movies(title_clean, year=cand_year)

            if not search_res or search_res.get("error"):
                err_msg = str(search_res.get("error") if search_res else "")
                if "request limit" in err_msg.lower():
                    _RATE_LIMIT_HIT = True
                _FAILED_ENRICHMENT_CACHE.add(lookup_key)
                return enriched

            matches = search_res.get("movies", [])
            cand_norm = normalize_text(title_clean)
            best_imdb_id = None

            for m in matches:
                m_title = m.get("title", "")
                m_norm = normalize_text(m_title)
                # Ensure titles match closely
                if m_norm == cand_norm or (len(cand_norm) >= 5 and (m_norm in cand_norm or cand_norm in m_norm)):
                    if cand_year is not None:
                        m_year = extract_year_int(m.get("year"))
                        if m_year is not None and abs(m_year - cand_year) > 1:
                            continue
                    best_imdb_id = m.get("imdb_id")
                    break

            if best_imdb_id:
                data = omdb_client.get_movie_details(best_imdb_id)
            else:
                _FAILED_ENRICHMENT_CACHE.add(lookup_key)
                return enriched

        if data and isinstance(data, dict) and not data.get("error"):
            # Save to local SQLite database (movies table)
            database.save_movie(data, db_path=db_path)

            new_id = data.get("imdb_id") or data.get("imdbID") or ""
            if not enriched.get("imdb_id") and new_id:
                enriched["imdb_id"] = new_id

            p_url = data.get("poster") or data.get("Poster") or data.get("poster_url")
            if p_url and p_url != "N/A":
                enriched["poster_url"] = p_url

            r_val = data.get("imdb_rating") or data.get("imdbRating")
            if r_val and r_val != "N/A":
                enriched["imdb_rating"] = r_val

            enriched["rated"] = data.get("rated") or data.get("Rated") or "N/A"
            enriched["runtime"] = data.get("runtime") or data.get("Runtime") or "N/A"

            # Update catalogue table with verified enrichment
            try:
                database.update_catalogue_enrichment(
                    title=candidate.get("title", ""),
                    year=candidate.get("year"),
                    imdb_id=enriched.get("imdb_id", ""),
                    poster_url=enriched.get("poster_url", ""),
                    db_path=db_path
                )
            except Exception:
                pass

            return enriched
        else:
            _FAILED_ENRICHMENT_CACHE.add(lookup_key)
    except Exception:
        _FAILED_ENRICHMENT_CACHE.add(lookup_key)

    return enriched

