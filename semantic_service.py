"""
CineTrace — Semantic Search & Dense Embedding Service (Phase 5.4)
Provides optional dense vector search across movie plot descriptions using a local
sentence-embedding model.

Key Features:
- Model: all-MiniLM-L6-v2 (384-dimensional dense vectors, Apache 2.0 license).
- Zero-Dependency Fallback: The application remains fully functional with TF-IDF/FTS5
  lexical search if sentence-transformers is not installed or model is not downloaded.
- Never downloads weights silently: Requires explicit user action via CLI or UI.
- Persistent Embedding Index: Binary storage via SQLite catalogue_embeddings table
  and cached .npz binary file.
- Fast Cosine Similarity: Normalized NumPy matrix multiplication (sub-millisecond latency).
- Incremental Indexing: Tracks plot_hash to only re-index new or changed plots.
- Cooperative Cancellation & Safe Recovery: Safe transactions per batch prevent corruption.
- Thread-Safe: All heavy computations run in background worker threads.
- Deterministic Mock Model: Built-in offline mock model for tests without network access.
"""

import abc
import enum
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import config
import database


class ModelStatus(enum.Enum):
    UNAVAILABLE = "unavailable"          # sentence-transformers library not installed
    NOT_DOWNLOADED = "not_downloaded"    # library installed, but model weights not present on disk
    DOWNLOADED = "downloaded"            # weights cached on disk, ready to load offline into RAM
    LOADING = "loading"                  # model currently loading into RAM
    READY = "ready"                      # model loaded and ready for inference
    ERROR = "error"                      # model failed to load


class IndexStatus(enum.Enum):
    EMPTY = "empty"                      # 0 embeddings
    INDEXING = "indexing"                # background indexing in progress
    PARTIAL = "partial"                  # some plots indexed, some pending
    READY = "ready"                      # 100% of catalogue plots indexed
    CANCELLED = "cancelled"              # indexing interrupted cooperatively
    ERROR = "error"                      # indexing failed


def get_local_model_info(
    cache_dir: Optional[Path] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inspect local disk cache for model weights without network access.
    Supports both flat directory layouts and Hugging Face Hub snapshot cache layouts.
    Returns:
        {
            "found": bool,
            "path": Optional[Path],
            "revision": Optional[str],
            "size_bytes": int
        }
    """
    cdir = Path(cache_dir or config.MODEL_CACHE_DIR)
    mname = model_name or config.DEFAULT_MODEL_NAME

    if not cdir.exists():
        return {"found": False, "path": None, "revision": None, "size_bytes": 0}

    # 1. Flat folder structure: e.g. cache_dir / all-MiniLM-L6-v2
    flat = cdir / mname
    if flat.is_dir():
        has_weights = (flat / "model.safetensors").is_file() or (flat / "pytorch_model.bin").is_file()
        if has_weights:
            size = sum(f.stat().st_size for f in flat.glob("**/*") if f.is_file())
            return {"found": True, "path": flat, "revision": "local", "size_bytes": size}

    # 2. Hugging Face Hub cache layout: e.g. models--sentence-transformers--all-MiniLM-L6-v2
    safe_name = mname.replace("/", "--")
    hub_candidates = list(cdir.glob(f"models--*{safe_name}*"))
    for hdir in hub_candidates:
        snapshots_dir = hdir / "snapshots"
        if snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    has_weights = (snap / "model.safetensors").is_file() or (snap / "pytorch_model.bin").is_file()
                    if has_weights:
                        revision = snap.name
                        ref_file = hdir / "refs" / "main"
                        if ref_file.is_file():
                            try:
                                revision = ref_file.read_text(encoding="utf-8").strip()
                            except Exception:
                                pass
                        size = sum(f.stat().st_size for f in snap.glob("**/*") if f.is_file())
                        return {"found": True, "path": snap, "revision": revision, "size_bytes": size}

    return {"found": False, "path": None, "revision": None, "size_bytes": 0}


class BaseEmbeddingModel(abc.ABC):
    """Abstract interface for sentence embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the generated embedding vectors."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        pass

    @property
    @abc.abstractmethod
    def revision(self) -> str:
        """Revision or version string of the embedding model."""
        pass

    @abc.abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode a list of text strings into an (N, D) float32 NumPy array of normalized vectors.
        """
        pass


class MockEmbeddingModel(BaseEmbeddingModel):
    """
    Offline deterministic embedding model for testing and offline development.
    Uses semantic concept synonym clusters and deterministic hash projections
    to simulate real semantic similarity without downloading weights or accessing the network.
    """

    def __init__(self, dimension: int = 384, model_name: str = "mock-mini-lm"):
        self._dim = dimension
        self._name = model_name

        # Semantic concept clusters that map paraphrased phrases into shared vector dimensions
        self._concept_clusters = [
            # Dimension 0: Time loops & reliving the same day
            (0, [
                "time loop", "temporal loop", "repeats the day", "reliving the same day",
                "relives the same day", "waking up to the same day", "escape the cycle",
                "break the cycle", "stuck in a loop", "groundhog", "same day repeatedly"
            ]),
            # Dimension 1: Mathematical genius, gifted child & uncle
            (1, [
                "mathematical abilities", "mathematics", "math genius", "child prodigy",
                "gifted", "calculus", "raised by her uncle", "normal childhood",
                "young girl discovers", "extraordinary mathematical"
            ]),
            # Dimension 2: Space exploration, black holes & relativity
            (2, [
                "space exploration", "wormhole", "black hole", "relativity", "astronaut",
                "interstellar", "space station", "quantum data", "galaxy"
            ]),
            # Dimension 3: Dreams, subconscious & inception
            (3, [
                "dreams inside dreams", "subconscious", "inception", "dream heist",
                "shared dreaming", "totem", "architect of dreams"
            ]),
            # Dimension 4: Martial arts, samurai & Kyoto
            (4, [
                "samurai", "kyoto", "sword fight", "katana", "feudal japan", "warrior"
            ])
        ]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return "mock-v1"

    def _encode_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        norm_text = text.lower().strip()
        if not norm_text:
            return vec

        # 1. Check semantic concept clusters
        concept_activated = False
        for dim_idx, keywords in self._concept_clusters:
            for kw in keywords:
                if kw in norm_text:
                    vec[dim_idx] += 3.5
                    concept_activated = True

        # 2. Deterministic word hash projection into remaining dimensions
        words = norm_text.split()
        for w in words:
            if len(w) < 3:
                continue
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            target_dim = 10 + (h % (self._dim - 10))
            vec[target_dim] += 1.0

        # 3. L2 Normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        arr = np.array([self._encode_single(t) for t in texts], dtype=np.float32)
        return arr


class SentenceTransformerModel(BaseEmbeddingModel):
    """
    Wrapper for local sentence-transformers SentenceTransformer model.
    Loads from a local cache folder and strictly never downloads weights silently.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME, cache_dir: Optional[Path] = None):
        self._name = model_name
        self._cache_dir = cache_dir or config.MODEL_CACHE_DIR
        self._model = None
        self._dim = config.SEMANTIC_DIMENSION
        self._revision = ""

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    def load(self, allow_download: bool = False) -> bool:
        """
        Load model weights into memory.
        If allow_download is False, strictly loads from local cache without network calls.
        """
        try:
            from sentence_transformers import SentenceTransformer

            info = get_local_model_info(self._cache_dir, self._name)

            if not allow_download:
                if not info["found"]:
                    return False
                self._model = SentenceTransformer(
                    self._name,
                    cache_folder=str(self._cache_dir),
                    local_files_only=True
                )
                self._revision = info.get("revision") or ""
            else:
                self._model = SentenceTransformer(
                    self._name,
                    cache_folder=str(self._cache_dir)
                )
                fresh_info = get_local_model_info(self._cache_dir, self._name)
                self._revision = fresh_info.get("revision") or ""

            # Determine actual embedding dimension
            test_emb = self._model.encode(["test"], convert_to_numpy=True)
            self._dim = test_emb.shape[1]
            return True
        except Exception as e:
            print(f"[CineTrace Semantic Error] Failed to load SentenceTransformer: {e}")
            self._model = None
            return False

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() before encoding.")
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        try:
            import torch
            with torch.inference_mode():
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        except (ImportError, AttributeError):
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return np.asarray(embeddings, dtype=np.float32)


class SemanticIndex:
    """
    In-memory normalized vector index mapped to catalogue IDs.
    Supports sub-millisecond cosine similarity search across 35,000+ records.
    """

    def __init__(self):
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        self._matrix: np.ndarray = np.empty((0, config.SEMANTIC_DIMENSION), dtype=np.float32)
        self._id_to_index: Dict[int, int] = {}
        self._plot_hashes: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._model_name = config.DEFAULT_MODEL_NAME
        self._provenance_status = "unverified"
        self._model_revision = ""
        self._embedding_type = "unknown"
        self._schema_version = 1

    @property
    def count(self) -> int:
        return len(self._ids)

    def is_empty(self) -> bool:
        return len(self._ids) == 0

    @property
    def provenance_status(self) -> str:
        return self._provenance_status

    @provenance_status.setter
    def provenance_status(self, val: str):
        self._provenance_status = str(val)

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @model_revision.setter
    def model_revision(self, val: str):
        self._model_revision = str(val)

    @property
    def embedding_type(self) -> str:
        return self._embedding_type

    @embedding_type.setter
    def embedding_type(self, val: str):
        self._embedding_type = str(val)

    @property
    def schema_version(self) -> int:
        return self._schema_version

    def set_data(
        self,
        ids: List[int],
        vectors: np.ndarray,
        plot_hashes: Optional[Dict[int, str]] = None,
        model_name: Optional[str] = None,
        provenance_status: Optional[str] = None,
        model_revision: Optional[str] = None,
        embedding_type: Optional[str] = None,
        schema_version: Optional[int] = None
    ):
        """Set in-memory index vectors and IDs with L2 normalization and provenance metadata."""
        with self._lock:
            self._ids = np.array(ids, dtype=np.int64)
            if vectors.size > 0:
                # Ensure float32 and L2-normalized rows
                mat = np.asarray(vectors, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = mat / norms
            else:
                self._matrix = np.empty((0, config.SEMANTIC_DIMENSION), dtype=np.float32)

            self._id_to_index = {int(cat_id): idx for idx, cat_id in enumerate(self._ids)}
            self._plot_hashes = dict(plot_hashes or {})
            if model_name:
                self._model_name = model_name
            if provenance_status is not None:
                self._provenance_status = str(provenance_status)
            if model_revision is not None:
                self._model_revision = str(model_revision)
            if embedding_type is not None:
                self._embedding_type = str(embedding_type)
            if schema_version is not None:
                self._schema_version = int(schema_version)

    def append_batch(
        self,
        ids: List[int],
        vectors: np.ndarray,
        plot_hashes: Optional[Dict[int, str]] = None
    ):
        """Append or update a batch of vectors in memory."""
        if not ids or vectors.size == 0:
            return
        with self._lock:
            norm_vecs = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(norm_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_vecs = norm_vecs / norms

            id_arr = np.array(ids, dtype=np.int64)
            if len(self._ids) == 0:
                self._ids = id_arr
                self._matrix = norm_vecs
            else:
                self._ids = np.concatenate([self._ids, id_arr])
                self._matrix = np.vstack([self._matrix, norm_vecs])

            self._id_to_index = {int(cat_id): idx for idx, cat_id in enumerate(self._ids)}
            if plot_hashes:
                self._plot_hashes.update(plot_hashes)

    def clear(self):
        with self._lock:
            self._ids = np.array([], dtype=np.int64)
            self._matrix = np.empty((0, config.SEMANTIC_DIMENSION), dtype=np.float32)
            self._id_to_index.clear()
            self._plot_hashes.clear()
            self._provenance_status = "unverified"
            self._model_revision = ""
            self._embedding_type = "unknown"

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 100
    ) -> Dict[int, float]:
        """
        Compute cosine similarity of query_vector against all indexed plots.
        Returns a mapping of {catalogue_id: similarity_score}.
        """
        with self._lock:
            if len(self._ids) == 0 or self._matrix.size == 0:
                return {}

            q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            else:
                return {}

            # Matrix-vector multiplication for sub-millisecond cosine similarity
            scores = self._matrix @ q

            # Clamp scores to [0.0, 1.0]
            scores = np.clip(scores, 0.0, 1.0)

            # Fast vectorized top-k selection avoiding materializing 34,570 dictionary keys
            n_items = len(self._ids)
            eff_k = min(top_k if (top_k and top_k > 0) else n_items, n_items)

            if eff_k < n_items:
                # Fast argpartition for top_k
                candidate_indices = np.argpartition(-scores, eff_k)[:eff_k]
                candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
            else:
                candidate_indices = np.argsort(-scores)

            # Map only top_k positive matches to dictionary of catalogue_id -> score
            result: Dict[int, float] = {}
            for idx in candidate_indices:
                s = float(scores[idx])
                if s <= 0.0:
                    break
                result[int(self._ids[idx])] = round(s, 4)
            return result

    def save_cache_file(self, file_path: Optional[Path] = None) -> bool:
        """Save the in-memory index to a compressed NumPy .npz archive."""
        path = file_path or config.EMBEDDING_INDEX_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp.npz")
            with self._lock:
                np.savez_compressed(
                    tmp_path,
                    ids=self._ids,
                    vectors=self._matrix,
                    model_name=self._model_name,
                    dimension=self._matrix.shape[1] if self._matrix.size > 0 else config.SEMANTIC_DIMENSION,
                    provenance_status=self._provenance_status,
                    model_revision=self._model_revision,
                    embedding_type=self._embedding_type,
                    schema_version=self._schema_version
                )
            # Atomic rename
            if tmp_path.exists():
                tmp_path.replace(path)
            return True
        except Exception as e:
            print(f"[CineTrace Semantic Warning] Failed to save .npz cache: {e}")
            return False

    def load_cache_file(self, file_path: Optional[Path] = None) -> bool:
        """Load index from a compressed NumPy .npz archive."""
        path = file_path or config.EMBEDDING_INDEX_FILE
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                ids = data["ids"]
                vectors = data["vectors"]
                model_name = str(data["model_name"]) if "model_name" in data else config.DEFAULT_MODEL_NAME
                prov = str(data["provenance_status"]) if "provenance_status" in data else "unverified"
                rev = str(data["model_revision"]) if "model_revision" in data else ""
                etype = str(data["embedding_type"]) if "embedding_type" in data else "unknown"
                sver = int(data["schema_version"]) if "schema_version" in data else 1
                self.set_data(
                    ids,
                    vectors,
                    model_name=model_name,
                    provenance_status=prov,
                    model_revision=rev,
                    embedding_type=etype,
                    schema_version=sver
                )
            return True
        except Exception as e:
            print(f"[CineTrace Semantic Warning] Failed to load .npz cache: {e}")
            return False


class SemanticService:
    """
    Central service managing the semantic embedding model, background indexing,
    persistent storage, query encoding, and vector similarity search.
    """

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL_NAME,
        db_path: Optional[Path] = None,
        custom_model: Optional[BaseEmbeddingModel] = None
    ):
        self.model_name = model_name
        self.db_path = db_path or config.DB_FILE
        self.model: Optional[BaseEmbeddingModel] = custom_model
        self.index = SemanticIndex()

        self._model_status = ModelStatus.UNAVAILABLE
        self._index_status = IndexStatus.EMPTY
        self._query_cache: Dict[str, np.ndarray] = {}
        self._query_cache_lock = threading.Lock()
        self._is_indexing = False
        self._cancel_indexing_requested = False
        self._indexing_thread: Optional[threading.Thread] = None
        self._load_error: Optional[str] = None

        # Check initial library and model availability
        self._evaluate_model_status()

    # -------------------------------------------------------------
    # Status & Diagnostic Properties
    # -------------------------------------------------------------

    @property
    def is_mock_active(self) -> bool:
        """Return True if active embedding model is a deterministic mock instead of real ML weights."""
        return isinstance(self.model, MockEmbeddingModel)

    @property
    def inference_engine(self) -> str:
        """Identify active inference engine for provenance auditing."""
        if self.model is None:
            return "none"
        if self.is_mock_active:
            return "mock_deterministic"
        return "real_sentence_transformer"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _evaluate_model_status(self):

        """Determine whether sentence-transformers is installed and model weights exist on disk."""
        if self.model is not None:
            self._model_status = ModelStatus.READY
            return

        if not self.is_available():
            self._model_status = ModelStatus.UNAVAILABLE
            return

        info = get_local_model_info(config.MODEL_CACHE_DIR, self.model_name)
        if info["found"]:
            self._model_status = ModelStatus.DOWNLOADED
        else:
            self._model_status = ModelStatus.NOT_DOWNLOADED

    @property
    def model_status(self) -> ModelStatus:
        return self._model_status

    @property
    def index_status(self) -> IndexStatus:
        if self._is_indexing:
            return IndexStatus.INDEXING
        if self.index.count == 0:
            return IndexStatus.EMPTY
        stats = database.get_catalogue_embedding_stats(self.db_path, self.model_name)
        if stats["pending_count"] > 0:
            return IndexStatus.PARTIAL
        return IndexStatus.READY

    def is_available(self) -> bool:
        """Return True if sentence_transformers library is importable."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def is_model_loaded(self) -> bool:
        """Return True if an embedding model is currently loaded in memory."""
        return self._model_status == ModelStatus.READY and self.model is not None

    def is_ready(self) -> bool:
        """
        Return True only if:
        1. Semantic search is globally enabled.
        2. Embedding model is loaded into RAM.
        3. Index contains vectors.
        4. Vector provenance has been verified against the active model.
        """
        return (
            config.SEMANTIC_SEARCH_ENABLED
            and self.is_model_loaded()
            and self.index.count > 0
            and self.index.provenance_status in ("verified", "mock_verified")
        )

    def is_indexing(self) -> bool:
        return self._is_indexing

    def is_loading_model(self) -> bool:
        """Return True if embedding model is actively loading into RAM."""
        return self._model_status == ModelStatus.LOADING

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        """
        Block current worker thread until semantic service is ready or timeout expires.
        Must never be called from the Tkinter main UI thread.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready():
                return True
            if self._model_status in (ModelStatus.ERROR, ModelStatus.UNAVAILABLE, ModelStatus.NOT_DOWNLOADED):
                return False
            time.sleep(0.05)
        return self.is_ready()

    def set_custom_model(self, custom_model: BaseEmbeddingModel):
        """Inject a custom or mock embedding model (for offline testing)."""
        self.model = custom_model
        self.model_name = custom_model.model_name
        self._model_status = ModelStatus.READY
        if self.index.count > 0:
            self.verify_provenance()

    # -------------------------------------------------------------
    # Model Loading & Explicit Download
    # -------------------------------------------------------------

    def load_model_sync(self, allow_download: bool = False) -> Tuple[bool, str]:
        """
        Synchronously load embedding model into memory.
        If allow_download is False, strictly loads from local disk cache without network access.
        """
        if self.is_model_loaded():
            return True, "Model already loaded."

        if not self.is_available():
            self._model_status = ModelStatus.UNAVAILABLE
            return False, "sentence-transformers library is not installed."

        info = get_local_model_info(config.MODEL_CACHE_DIR, self.model_name)
        if not allow_download and not info["found"]:
            self._model_status = ModelStatus.NOT_DOWNLOADED
            return False, "Model weights not downloaded locally. Explicit download required."

        self._model_status = ModelStatus.LOADING
        try:
            model_wrapper = SentenceTransformerModel(self.model_name)
            success = model_wrapper.load(allow_download=allow_download)
            if success:
                self.model = model_wrapper
                self._model_status = ModelStatus.READY
                self._load_error = None
                # If vectors are already loaded, verify provenance
                if self.index.count > 0 and self.index.provenance_status not in ("verified", "mock_verified"):
                    self.verify_provenance()
                return True, f"Model '{self.model_name}' loaded successfully."
            else:
                self._model_status = ModelStatus.ERROR
                self._load_error = "Failed to load model weights from disk (weights may be corrupted, incomplete, or unreadable)."
                return False, self._load_error
        except Exception as e:
            self._model_status = ModelStatus.ERROR
            self._load_error = str(e)
            return False, f"Model load error: {str(e)}"


    def load_model(
        self,
        allow_download: bool = False,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ) -> bool:
        """
        Load embedding model into memory asynchronously.
        If allow_download is False, never attempts network download.
        """
        if self.is_model_loaded():
            if on_complete:
                on_complete(True, "Model already loaded.")
            return True

        if not self.is_available():
            self._model_status = ModelStatus.UNAVAILABLE
            msg = "sentence-transformers is not installed. Please install via: pip install sentence-transformers"
            if on_complete:
                on_complete(False, msg)
            return False

        self._model_status = ModelStatus.LOADING

        def _worker():
            success, msg = self.load_model_sync(allow_download=allow_download)
            if on_complete:
                on_complete(success, msg)

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def download_and_load_model_sync(self) -> Tuple[bool, str]:
        """
        Explicit synchronous model download and load.
        Must only be called upon explicit user command or button click.
        """
        return self.load_model_sync(allow_download=True)

    # -------------------------------------------------------------
    # Provenance Verification & Metadata Persistence
    # -------------------------------------------------------------

    def _persist_provenance_meta(self):
        """Write verified index metadata to SQLite and .npz cache."""
        database.set_catalogue_embedding_meta("schema_version", str(self.index.schema_version), self.db_path)
        database.set_catalogue_embedding_meta("model_name", self.model_name, self.db_path)
        database.set_catalogue_embedding_meta("model_revision", self.index.model_revision, self.db_path)
        database.set_catalogue_embedding_meta("embedding_type", self.index.embedding_type, self.db_path)
        database.set_catalogue_embedding_meta(
            "dimension",
            str(self.model.dimension if self.model else config.SEMANTIC_DIMENSION),
            self.db_path
        )
        database.set_catalogue_embedding_meta("provenance_status", self.index.provenance_status, self.db_path)
        database.set_catalogue_embedding_meta("verified_at", time.strftime("%Y-%m-%d %H:%M:%S"), self.db_path)
        self.index.save_cache_file()

    def verify_provenance(
        self,
        sample_size: int = 5,
        min_similarity: float = 0.999
    ) -> Tuple[bool, str]:
        """
        Verify that persisted embeddings in memory and database were generated
        by the currently active embedding model.
        """
        if not self.is_model_loaded():
            return False, "Model is not loaded."

        if self.index.count == 0:
            return False, "No embeddings indexed to verify."

        # If deterministic mock model
        if isinstance(self.model, MockEmbeddingModel):
            self.index.provenance_status = "mock_verified"
            self.index.embedding_type = "mock"
            self.index.model_revision = "mock-v1"
            self._persist_provenance_meta()
            return True, "Mock embedding model verified."

        # Verify against real model via sample encoding
        sample_ids = [int(i) for i in self.index._ids[:sample_size]]
        if not sample_ids:
            return False, "No catalogue IDs available for provenance check."

        plots_map = database.get_catalogue_plots_by_ids(sample_ids, self.db_path)
        valid_ids = [cid for cid in sample_ids if cid in plots_map and plots_map[cid].strip()]
        if not valid_ids:
            return False, "Could not retrieve plot text for sample records."

        sample_plots = [plots_map[cid] for cid in valid_ids]
        computed_vecs = self.model.encode(sample_plots)

        similarities = []
        for idx, cid in enumerate(valid_ids):
            vec_idx = self.index._id_to_index.get(cid)
            if vec_idx is not None:
                stored_v = self.index._matrix[vec_idx]
                comp_v = computed_vecs[idx]
                cos_sim = float(np.dot(stored_v, comp_v))
                similarities.append(cos_sim)

        if not similarities:
            return False, "Could not calculate vector similarities."

        min_sim = min(similarities)
        if min_sim >= min_similarity:
            self.index.provenance_status = "verified"
            self.index.embedding_type = "sentence_transformer"
            self.index.model_revision = getattr(self.model, "revision", "") or ""
            self._persist_provenance_meta()
            return True, f"Verified {len(similarities)} samples against {self.model_name} (min similarity: {min_sim:.6f})"
        else:
            self.index.provenance_status = "incompatible"
            database.set_catalogue_embedding_meta("provenance_status", "incompatible", self.db_path)
            return False, f"Stored embeddings are incompatible with {self.model_name} (min similarity: {min_sim:.4f} < {min_similarity})"

    # -------------------------------------------------------------
    # Index Loading & Persistence
    # -------------------------------------------------------------

    def load_index_from_storage(self) -> int:
        """
        Load precomputed embeddings into memory.
        1. Checks .npz cache file first for sub-10ms loading.
        2. Falls back to SQLite catalogue_embeddings table.
        Returns total count of loaded embeddings.
        """
        loaded_count = 0

        # 1. Try .npz cache file
        if self.index.load_cache_file():
            stats = database.get_catalogue_embedding_stats(self.db_path, self.model_name)
            if self.index.count == stats["indexed_count"] and self.index.count > 0:
                loaded_count = self.index.count

        # 2. Fall back to SQLite if .npz was missing or mismatched
        if loaded_count == 0:
            db_rows = database.get_catalogue_embeddings(self.db_path, self.model_name)
            if not db_rows:
                self.index.clear()
                return 0

            ids = []
            vectors_list = []
            hashes = {}
            dim = db_rows[0].get("dimension") or config.SEMANTIC_DIMENSION

            for r in db_rows:
                cat_id = r["catalogue_id"]
                raw_blob = r["embedding"]
                if raw_blob:
                    vec = np.frombuffer(raw_blob, dtype=np.float32)
                    if len(vec) == dim:
                        ids.append(cat_id)
                        vectors_list.append(vec)
                        hashes[cat_id] = r.get("plot_hash", "")

            if vectors_list:
                mat = np.vstack(vectors_list)
                self.index.set_data(ids, mat, plot_hashes=hashes, model_name=self.model_name)
                loaded_count = len(ids)
            else:
                self.index.clear()
                return 0

        # Populate provenance metadata from SQLite if available
        meta = database.get_all_catalogue_embedding_meta(self.db_path)
        if meta:
            if "provenance_status" in meta:
                self.index.provenance_status = meta["provenance_status"]
            if "model_revision" in meta:
                self.index.model_revision = meta["model_revision"]
            if "embedding_type" in meta:
                self.index.embedding_type = meta["embedding_type"]
            if "schema_version" in meta:
                try:
                    self.index._schema_version = int(meta["schema_version"])
                except Exception:
                    pass

        # If model is already loaded and provenance is unverified, verify immediately
        if self.is_model_loaded() and self.index.count > 0 and self.index.provenance_status not in ("verified", "mock_verified"):
            self.verify_provenance()

        # Update .npz cache with loaded state
        self.index.save_cache_file()
        return loaded_count

    # -------------------------------------------------------------
    # Batch Indexing & Invalidation
    # -------------------------------------------------------------

    def build_index_async(
        self,
        batch_size: int = config.SEMANTIC_BATCH_SIZE,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_complete: Optional[Callable[[int, int, float, bool], None]] = None
    ) -> bool:
        """
        Asynchronously generate embeddings for all pending or changed catalogue plots.
        Prevents concurrent duplicate indexing jobs.
        """
        if self._is_indexing:
            return False

        if not self.is_model_loaded():
            if on_complete:
                on_complete(0, 0, 0.0, False)
            return False

        self._is_indexing = True
        self._cancel_indexing_requested = False

        def _worker():
            t0 = time.perf_counter()
            cancelled = False
            total_indexed = 0

            try:
                # 1. Clean stale embeddings
                database.delete_stale_catalogue_embeddings(self.db_path)

                # 2. Get pending plots
                pending = database.get_pending_catalogue_plots(self.db_path, self.model_name)
                total_pending = len(pending)

                if total_pending == 0:
                    self.load_index_from_storage()
                    self._is_indexing = False
                    dt = time.perf_counter() - t0
                    if on_complete:
                        on_complete(self.index.count, 0, dt, False)
                    return

                # 3. Process in batches
                n_batches = (total_pending + batch_size - 1) // batch_size
                for b_idx in range(n_batches):
                    if self._cancel_indexing_requested:
                        cancelled = True
                        break

                    start_i = b_idx * batch_size
                    end_i = min(start_i + batch_size, total_pending)
                    batch = pending[start_i:end_i]

                    plots = [b["plot"] for b in batch]
                    ids = [b["id"] for b in batch]
                    hashes = [database.compute_plot_hash(b["plot"], b["title"]) for b in batch]

                    # Generate normalized dense embeddings
                    vecs = self.model.encode(plots, batch_size=batch_size)

                    # Prepare database records
                    records_to_save = []
                    for i, cat_id in enumerate(ids):
                        raw_bytes = vecs[i].tobytes()
                        records_to_save.append({
                            "catalogue_id": cat_id,
                            "plot_hash": hashes[i],
                            "embedding": raw_bytes,
                            "dimension": self.model.dimension,
                            "model_name": self.model_name
                        })

                    # Commit batch to SQLite (atomic transaction)
                    saved = database.save_catalogue_embeddings_batch(records_to_save, self.db_path)
                    total_indexed += saved

                    # Update in-memory index incrementally
                    batch_hash_map = {ids[i]: hashes[i] for i in range(len(ids))}
                    self.index.append_batch(ids, vecs, batch_hash_map)

                    if on_progress:
                        processed = end_i
                        msg = f"Indexed {processed} / {total_pending} plots ({int((processed / total_pending) * 100)}%)"
                        on_progress(processed, total_pending, msg)

                # Save persistent .npz cache
                self.index.save_cache_file()

            except Exception as e:
                print(f"[CineTrace Semantic Error] Indexing failed: {e}")
            finally:
                if not cancelled and (total_indexed > 0 or (total_pending == 0 and self.index.count > 0)):
                    if isinstance(self.model, MockEmbeddingModel):
                        self.index.provenance_status = "mock_verified"
                        self.index.embedding_type = "mock"
                        self.index.model_revision = "mock-v1"
                    else:
                        self.index.provenance_status = "verified"
                        self.index.embedding_type = "sentence_transformer"
                        self.index.model_revision = getattr(self.model, "revision", "") or ""
                    self._persist_provenance_meta()
                self._is_indexing = False
                dt = time.perf_counter() - t0
                if on_complete:
                    on_complete(self.index.count, total_indexed, dt, cancelled)

        self._indexing_thread = threading.Thread(target=_worker, daemon=True)
        self._indexing_thread.start()
        return True

    def cancel_indexing(self):
        """Request immediate cooperative cancellation of the active indexing worker."""
        if self._is_indexing:
            self._cancel_indexing_requested = True

    def clear_index(self) -> bool:
        """Clear all embeddings from SQLite and memory."""
        success = database.clear_catalogue_embeddings(self.db_path, self.model_name)
        database.clear_catalogue_embedding_meta(self.db_path)
        self.index.clear()
        if config.EMBEDDING_INDEX_FILE.exists():
            try:
                config.EMBEDDING_INDEX_FILE.unlink(missing_ok=True)
            except Exception:
                pass
        return success

    # -------------------------------------------------------------
    # Query Encoding & Search
    # -------------------------------------------------------------

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode a query string into a normalized dense vector, using LRU cache."""
        cleaned = query.strip()
        if not cleaned or not self.is_model_loaded():
            return None

        with self._query_cache_lock:
            if cleaned in self._query_cache:
                return self._query_cache[cleaned]

        try:
            vec = self.model.encode([cleaned])[0]
            with self._query_cache_lock:
                if len(self._query_cache) > 256:
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[cleaned] = vec
            return vec
        except Exception as e:
            print(f"[CineTrace Semantic Error] Query encoding failed: {e}")
            return None

    def search_by_query(
        self,
        query: str,
        top_k: int = 500
    ) -> Dict[int, float]:
        """
        Execute sub-millisecond semantic search for a query string.
        Returns {catalogue_id: similarity_score}.
        """
        if not self.is_ready():
            return {}

        q_vec = self.encode_query(query)
        if q_vec is None:
            return {}

        return self.index.search_vector(q_vec, top_k=top_k)


# Global singleton instance
_GLOBAL_SEMANTIC_SERVICE: Optional[SemanticService] = None
_SERVICE_LOCK = threading.Lock()


def get_semantic_service(
    model_name: str = config.DEFAULT_MODEL_NAME,
    db_path: Optional[Path] = None
) -> SemanticService:
    """Retrieve or initialize the global SemanticService singleton."""
    global _GLOBAL_SEMANTIC_SERVICE
    with _SERVICE_LOCK:
        if _GLOBAL_SEMANTIC_SERVICE is None:
            _GLOBAL_SEMANTIC_SERVICE = SemanticService(model_name=model_name, db_path=db_path)
            # Automatically load existing precomputed index from disk if present
            _GLOBAL_SEMANTIC_SERVICE.load_index_from_storage()
        return _GLOBAL_SEMANTIC_SERVICE


# =====================================================================
# Cross-Encoder Reranking Architecture (Phase 5.7)
# =====================================================================

CrossEncoderStatus = ModelStatus


def select_passage_head_tail(
    plot_text: str,
    max_head_words: int = 220,
    max_tail_words: int = 130
) -> str:
    """
    Intentionally select the most informative passage from movie plot descriptions
    to respect the 512-token limit of the cross-encoder without losing critical scene details.

    Passage Selection Strategy:
    - Plots under 350 words (~450 tokens) are evaluated in full.
    - Long plots: Naive truncation from the beginning drops crucial plot endings,
      third-act twists, and story resolutions (e.g., identity reveals, climactic confrontations).
    - Therefore, for plots > 350 words, we preserve the premise and character introductions
      (first max_head_words words) + a clear transition marker + the climax and resolution
      (last max_tail_words words).
    - This captures both the initial setup and the narrative payoff within ~450 tokens,
      guaranteeing sub-512 token compliance for ms-marco-MiniLM-L-6-v2.
    """
    if not plot_text:
        return ""
    words = plot_text.strip().split()
    if len(words) <= (max_head_words + max_tail_words):
        return plot_text.strip()
    head = " ".join(words[:max_head_words])
    tail = " ".join(words[-max_tail_words:])
    return f"{head} ... [plot progression] ... {tail}"


class MockCrossEncoder:
    """Deterministic offline mock CrossEncoder for testing without downloading weights."""

    def __init__(self, model_name: str = "mock-cross-encoder"):
        self.model_name = model_name
        self.revision = "mock-local"

    def predict(
        self,
        sentences: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        scores = []
        for pair in sentences:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                q, p = pair
            else:
                q, p = str(pair), ""
            q_norm = set(re.findall(r"\w+", (q or "").lower()))
            p_norm = set(re.findall(r"\w+", (p or "").lower()))
            if not q_norm or not p_norm:
                scores.append(-4.0)
                continue
            inter = len(q_norm & p_norm)
            # Deterministic logit: base of -2.5, scaled up to +6.5 by query keyword coverage
            score = -2.5 + 9.0 * (inter / max(1, len(q_norm)))
            scores.append(round(score, 3))
        return np.array(scores, dtype=np.float32)


class CrossEncoderService:
    """
    Manages second-stage cross-encoder reranking for movie candidate retrieval.

    Key Features:
    - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache 2.0 license).
    - Offline load: checks local cache and loads without network requests.
    - Explicit download: only downloads upon user request.
    - Thread-safe reranking and batch inference.
    - Passage selection preserving head (premise) and tail (climax/resolution).
    - Candidate score preservation for auditing and diagnostic display.
    - Clean fallback to first-stage retrieval if model is unavailable.
    - Honest relevance labels (unbounded logits formatted as logits, never fake percentages).
    """

    def __init__(
        self,
        model_name: str = config.CROSS_ENCODER_MODEL_NAME,
        cache_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or config.MODEL_CACHE_DIR)
        self.model = None
        self._status = ModelStatus.UNAVAILABLE
        self._status_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._load_error: Optional[str] = None

        self._check_initial_status()

    def _check_initial_status(self):
        with self._status_lock:
            try:
                import sentence_transformers
                info = get_local_model_info(self.cache_dir, self.model_name)
                if info["found"]:
                    self._status = ModelStatus.DOWNLOADED
                else:
                    self._status = ModelStatus.NOT_DOWNLOADED
            except ImportError:
                self._status = ModelStatus.UNAVAILABLE

    @property
    def status(self) -> ModelStatus:
        with self._status_lock:
            return self._status

    @property
    def model_status(self) -> ModelStatus:
        """Alias for status property matching SemanticService."""
        with self._status_lock:
            return self._status

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    def is_model_loaded(self) -> bool:
        with self._model_lock:
            return self.model is not None

    @property
    def is_mock_active(self) -> bool:
        """Return True if active model is a deterministic mock instead of real ML weights."""
        return isinstance(self.model, MockCrossEncoder)

    @property
    def inference_engine(self) -> str:
        """Identify active inference engine for provenance auditing."""
        if self.model is None:
            return "none"
        if self.is_mock_active:
            return "mock_deterministic"
        return "real_cross_encoder"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def is_ready(self) -> bool:
        with self._status_lock:
            return self._status == ModelStatus.READY and self.model is not None

    def is_loading_model(self) -> bool:
        with self._status_lock:
            return self._status == ModelStatus.LOADING

    def set_custom_model(self, model: Any):
        """Inject custom or mock model for offline testing."""
        with self._model_lock:
            self.model = model
            with self._status_lock:
                self._status = ModelStatus.READY


    def load_model_sync(self, allow_download: bool = False) -> Tuple[bool, str]:
        """
        Load CrossEncoder model synchronously.
        If allow_download is False, strictly loads from local cache without network access.
        """
        if not self.is_available():
            with self._status_lock:
                self._status = ModelStatus.UNAVAILABLE
            return False, "sentence-transformers library is not installed."

        with self._load_lock:
            if self.is_model_loaded():
                with self._status_lock:
                    self._status = ModelStatus.READY
                return True, "Cross-encoder model is already loaded."

            with self._status_lock:
                self._status = ModelStatus.LOADING

            try:
                from sentence_transformers import CrossEncoder

                info = get_local_model_info(self.cache_dir, self.model_name)
                if not info["found"] and not allow_download:
                    with self._status_lock:
                        self._status = ModelStatus.NOT_DOWNLOADED
                    return False, f"Model weights for '{self.model_name}' not found locally."

                # If found locally, use snapshot path directly with local_files_only=True
                if info["found"] and info.get("path"):
                    load_target = str(info["path"])
                    encoder = CrossEncoder(load_target, max_length=512, local_files_only=True)
                else:
                    # Download explicitly into cache_dir
                    encoder = CrossEncoder(
                        self.model_name,
                        max_length=512,
                        cache_folder=str(self.cache_dir)
                    )

                with self._model_lock:
                    self.model = encoder
                with self._status_lock:
                    self._status = ModelStatus.READY
                self._load_error = None
                return True, f"Successfully loaded '{self.model_name}'."
            except Exception as e:
                with self._status_lock:
                    self._status = ModelStatus.ERROR
                self._load_error = str(e)
                return False, f"Failed to load cross-encoder model: {e}"

    def download_and_load_model_sync(self) -> Tuple[bool, str]:
        return self.load_model_sync(allow_download=True)

    def load_model(
        self,
        allow_download: bool = False,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ):
        """Asynchronously load model on background thread."""
        def _worker():
            ok, msg = self.load_model_sync(allow_download=allow_download)
            if on_complete:
                try:
                    on_complete(ok, msg)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_ready():
                return True
            time.sleep(0.05)
        return self.is_ready()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = config.CROSS_ENCODER_TOP_K,
        batch_size: int = config.CROSS_ENCODER_BATCH_SIZE
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder sentence-pair inference.
        Preserves original scores, candidate metadata, and adds honest relevance labels.
        """
        if not candidates or not query.strip():
            return candidates

        if not self.is_ready():
            return candidates

        # Deduplicate candidates by stable catalogue ID first
        seen_ids = set()
        deduped: List[Dict[str, Any]] = []
        for cand in candidates:
            cid = cand.get("id")
            if cid is not None:
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
            deduped.append(cand)

        # Slice top_k candidates for reranking
        to_rerank = deduped[:top_k]
        remainder = deduped[top_k:]

        # Build query-passage pairs using intentional passage selection
        pairs = []
        for cand in to_rerank:
            plot_text = cand.get("plot", "")
            passage = select_passage_head_tail(plot_text)
            pairs.append((query.strip(), passage))

        try:
            with self._model_lock:
                try:
                    import torch
                    with torch.inference_mode():
                        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                except (ImportError, AttributeError):
                    scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

            del pairs

            # Attach scores and preserve original metadata
            for i, cand in enumerate(to_rerank):
                score = float(scores[i])
                orig_sim = cand.get("similarity", 0.0)
                orig_label = cand.get("score_label", "")
                cand["raw_cross_encoder_score"] = round(score, 4)
                cand["inference_engine"] = self.inference_engine
                cand["original_similarity"] = orig_sim
                cand["original_score_label"] = orig_label
                cand["original_rank"] = i + 1
                cand["reranked"] = True
                cand["similarity"] = round(score, 4)
                cand["similarity_pct"] = f"Logit: {score:+.2f}"
                cand["score_label"] = f"Reranked Match: {score:+.2f}"
                cand["match_type"] = "reranked"
                cand["match_details"] = f"Cross-Encoder logit {score:+.2f} (1st-stage: #{i+1} {orig_label})"

            # Sort reranked portion descending by cross-encoder score
            to_rerank.sort(key=lambda x: x["raw_cross_encoder_score"], reverse=True)

            for rank_idx, cand in enumerate(to_rerank):
                cand["reranked_rank"] = rank_idx + 1

            return to_rerank + remainder
        except Exception as e:
            print(f"[CineTrace Cross-Encoder Error] Reranking failed: {e}")
            return candidates


# Global singleton instance for CrossEncoderService
_GLOBAL_CROSS_ENCODER_SERVICE: Optional[CrossEncoderService] = None
_CROSS_ENCODER_LOCK = threading.Lock()


def get_cross_encoder_service(
    model_name: str = config.CROSS_ENCODER_MODEL_NAME,
    cache_dir: Optional[Path] = None
) -> CrossEncoderService:
    """Retrieve or initialize the global CrossEncoderService singleton."""
    global _GLOBAL_CROSS_ENCODER_SERVICE
    with _CROSS_ENCODER_LOCK:
        if _GLOBAL_CROSS_ENCODER_SERVICE is None:
            _GLOBAL_CROSS_ENCODER_SERVICE = CrossEncoderService(
                model_name=model_name,
                cache_dir=cache_dir
            )
        return _GLOBAL_CROSS_ENCODER_SERVICE

