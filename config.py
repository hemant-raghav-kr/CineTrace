"""
CineTrace - Configuration Management
Handles loading environment variables, OMDb API key management,
and configuration persistence without exposing secrets in logs or errors.
"""

import os
import socket
from pathlib import Path

# Prioritize IPv4 over IPv6 across all network libraries to prevent
# socket connect timeouts on dual-stack systems where IPv6 routes are unreachable
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

# Base directory of the CineTrace project
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DB_FILE = BASE_DIR / "cinetrace.db"
DATA_DIR = BASE_DIR / "data"
PLOT_DATASET_FILE = DATA_DIR / "movies_plots.json"

# Semantic Search & Embeddings Configuration (Phase 5.4)
EMBEDDING_DIR = DATA_DIR / "embeddings"
MODEL_CACHE_DIR = EMBEDDING_DIR / "models"
EMBEDDING_INDEX_FILE = EMBEDDING_DIR / "catalogue_embeddings.npz"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_DIMENSION = 384
SEMANTIC_BATCH_SIZE = 128
SEMANTIC_SEARCH_ENABLED = os.getenv("CINETRACE_SEMANTIC_ENABLED", "true").lower() in ("true", "1", "yes")
TFIDF_CACHE_FILE = EMBEDDING_DIR / "catalogue_tfidf.pkl"

# Cross-Encoder Reranking Configuration (Phase 5.7)
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_ENABLED = os.getenv("CINETRACE_CROSS_ENCODER_ENABLED", "true").lower() in ("true", "1", "yes")
CROSS_ENCODER_TOP_K = 25
CROSS_ENCODER_BATCH_SIZE = 32
CROSS_ENCODER_MAX_PASSAGE_WORDS = 350

# OMDb Search Pagination Configuration
OMDB_SEARCH_MAX_PAGES = int(os.getenv("CINETRACE_SEARCH_MAX_PAGES", "15"))


# Ensure embedding directory exists
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file if it exists
try:
    from dotenv import load_dotenv, set_key
    load_dotenv(dotenv_path=ENV_FILE)
except ImportError:
    # python-dotenv optional fallback
    load_dotenv = None
    set_key = None


# Global in-memory cache for API key
_omdb_api_key: str = os.getenv("OMDB_API_KEY", "").strip()


def get_api_key() -> str:
    """
    Retrieve the current OMDb API key.
    Checks in-memory cache first, then environment variable.
    """
    global _omdb_api_key
    if _omdb_api_key:
        return _omdb_api_key
    env_val = os.getenv("OMDB_API_KEY", "").strip()
    if env_val:
        _omdb_api_key = env_val
    return _omdb_api_key


def set_api_key(key: str, persist_to_env: bool = True) -> bool:
    """
    Update the OMDb API key at runtime.
    Optionally persists the key to the local .env file.
    Never prints or exposes the key to logs.
    """
    global _omdb_api_key
    cleaned_key = key.strip() if key else ""
    _omdb_api_key = cleaned_key
    os.environ["OMDB_API_KEY"] = cleaned_key

    if persist_to_env and cleaned_key:
        try:
            if set_key is not None:
                if not ENV_FILE.exists():
                    ENV_FILE.touch(mode=0o600, exist_ok=True)
                set_key(str(ENV_FILE), "OMDB_API_KEY", cleaned_key)
                return True
            else:
                # Basic fallback writer
                with open(ENV_FILE, "w", encoding="utf-8") as f:
                    f.write(f"OMDB_API_KEY={cleaned_key}\n")
                return True
        except Exception:
            # Silently fail persistence if file writing is restricted; runtime key is still set
            return False
    return True


def is_api_key_configured() -> bool:
    """Check whether a non-empty API key is configured."""
    return bool(get_api_key())


def get_masked_api_key() -> str:
    """
    Return a masked representation of the API key for safe UI display.
    Example: '••••••••1a2b' or 'Not Configured'.
    """
    key = get_api_key()
    if not key:
        return "Not Configured"
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]
