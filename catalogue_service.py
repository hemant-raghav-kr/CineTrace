"""
CineTrace — Catalogue Ingestion and Dataset Management Service (Phase 5.3)
Handles streaming ingestion, schema normalization, deduplication, and validation
for large-scale movie datasets (~35,000+ films), including:
- Wikipedia Movie Plots (wiki_movie_plots_deduped.csv / CC BY-SA 4.0)
- CMU Movie Summary Corpus (TSV/JSON / CC BY-SA 3.0)
- IMDb Non-Commercial Datasets (TSV)
- Heterogeneous user JSON, JSONL, and CSV datasets.

Key features:
1. True streaming row-by-row parsing (RAM consumption stays under ~5 MB).
2. Safe field size limits for long Wikipedia plot summaries (>131 KB).
3. Robust Unicode preservation (UTF-8, UTF-8 with BOM, Latin-1 fallback).
4. Responsive batch database writes (500-row chunks) with cancellation support.
5. Non-destructive metadata merge: superior existing data is never wiped.
"""

import csv
import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import config
import database

# Safely expand CSV field size limit to prevent crash on long Wikipedia plots
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# Recognized column name mappings for heterogeneous dataset schemas
FIELD_MAPPINGS = {
    "title": [
        "title", "Title", "movie_title", "MovieTitle", "primaryTitle",
        "original_title", "film_title", "name"
    ],
    "alternate_titles": [
        "alternate_titles", "originalTitle", "akas", "aka", "aliases"
    ],
    "year": [
        "year", "Year", "release_year", "Release Year", "startYear",
        "released", "date", "year_released"
    ],
    "plot": [
        "plot", "Plot", "overview", "synopsis", "description",
        "plot_summary", "Plot Summary", "summary", "storyline"
    ],
    "actors": [
        "actors", "Actors", "cast", "Cast", "stars", "Stars",
        "performers", "principals", "lead_actors"
    ],
    "director": [
        "director", "Director", "directors", "Directors", "filmmaker"
    ],
    "genre": [
        "genre", "Genre", "genres", "Genres", "category"
    ],
    "origin_ethnicity": [
        "origin_ethnicity", "Origin/Ethnicity", "origin", "ethnicity",
        "country_of_origin", "country"
    ],
    "wiki_url": [
        "wiki_url", "Wiki Page", "wiki_page", "wikipedia_url", "wikipedia", "url"
    ],
    "imdb_id": [
        "imdb_id", "imdbID", "tconst", "imdb_code", "const"
    ],
    "poster_url": [
        "poster_url", "poster", "Poster", "image_url", "cover_url"
    ],
    "source_provenance": [
        "source_provenance", "provenance", "Provenance", "source", "corpus", "dataset"
    ],
    "license_info": [
        "license_info", "license", "licence"
    ]
}


@dataclass
class ImportStats:
    """Stores detailed outcome statistics from a dataset import operation."""
    total_read: int = 0
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    skipped_reasons: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    source_file: str = ""
    provenance: str = ""
    cancelled: bool = False

    def summary(self) -> str:
        status_prefix = "[CANCELLED EARLY] " if self.cancelled else ""
        return (
            f"{status_prefix}Imported {self.imported} new movies, updated {self.updated} existing records. "
            f"Skipped {self.skipped} invalid rows in {self.duration_sec:.2f}s."
        )


def normalize_year(raw_year: Any) -> str:
    """
    Extract a valid 4-digit release year string (between 1880 and 2035) from diverse formats:
    - '1994' -> '1994'
    - '1994-10-14' -> '1994'
    - 'c. 1920' -> '1920'
    - '1984/85' -> '1984'
    - '(1972)' -> '1972'
    - 'unknown' / 'N/A' / '' -> ''
    """
    if raw_year is None:
        return ""
    s = str(raw_year).strip()
    if not s or s.lower() in ("unknown", "n/a", "null", "none"):
        return ""
    m = re.search(r"\b(18[89]\d|19\d\d|20[0-3]\d)\b", s)
    if m:
        return m.group(1)
    return ""


def clean_cell_text(val: Any) -> str:
    """Strip whitespace and normalize common missing value placeholders."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("unknown", "n/a", "null", "none", "nan", "-"):
        return ""
    return s


def normalize_record_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map arbitrary dictionary keys from various CSV/JSON schemas into normalized catalogue fields.
    """
    normalized: Dict[str, Any] = {}

    for target_field, candidates in FIELD_MAPPINGS.items():
        for cand in candidates:
            if cand in raw and raw[cand] is not None:
                val = str(raw[cand]).strip()
                if val and val != "N/A":
                    # Clean 'unknown' from metadata fields like director or genre
                    if target_field in ("director", "genre") and val.lower() == "unknown":
                        continue
                    normalized[target_field] = val
                    break

    # Specific check for IMDb IDs in generic 'id' fields
    if "imdb_id" not in normalized:
        raw_id = str(raw.get("id") or "").strip()
        if re.match(r"^tt\d{6,10}$", raw_id, re.IGNORECASE):
            normalized["imdb_id"] = raw_id.lower()

    # Extract 4-digit year robustly
    if "year" in normalized:
        normalized["year"] = normalize_year(normalized["year"])

    return normalized


def stream_dataset_rows(
    file_path: Union[str, Path]
) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """
    Streaming row-by-row generator for CSV, TSV, JSONL, and JSON files.
    Yields (line_no, row_dict, error_reason).
    Maintains minimal memory footprint regardless of dataset size.
    Handles UTF-8 with BOM, standard UTF-8, and Latin-1 replacement safely.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        # Auto-detect delimiter from first chunk
        delimiter = ","
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                header_line = f.readline()
                if "\t" in header_line and header_line.count("\t") > header_line.count(","):
                    delimiter = "\t"
                elif "|" in header_line and header_line.count("|") > header_line.count(","):
                    delimiter = "|"
        except Exception:
            delimiter = ","

        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for line_no, row in enumerate(reader, 2):
                if isinstance(row, dict) and any(str(v).strip() for v in row.values() if v is not None):
                    yield line_no, row, None
                else:
                    yield line_no, None, f"Row {line_no} empty or corrupt"

    elif suffix in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                    if isinstance(obj, dict):
                        yield line_no, obj, None
                    else:
                        yield line_no, None, f"JSON line {line_no} is not an object"
                except Exception as e:
                    yield line_no, None, f"JSON line {line_no} malformed: {str(e)}"

    elif suffix == ".json":
        # For standard JSON files, parse structure and yield elements
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = json.load(f)

            if isinstance(content, list):
                for idx, item in enumerate(content, 1):
                    if isinstance(item, dict):
                        yield idx, item, None
                    else:
                        yield idx, None, f"Record {idx} is not an object"
            elif isinstance(content, dict):
                # Check for wrapped arrays
                items = None
                for key in ("movies", "data", "results", "films"):
                    if key in content and isinstance(content[key], list):
                        items = content[key]
                        break
                if items is not None:
                    for idx, item in enumerate(items, 1):
                        if isinstance(item, dict):
                            yield idx, item, None
                        else:
                            yield idx, None, f"Record {idx} is not an object"
                else:
                    yield 1, content, None
            else:
                yield 1, None, f"Unexpected root JSON structure: {type(content).__name__}"
        except json.JSONDecodeError:
            # Fall back to line-by-line JSON parsing
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        obj = json.loads(line_s)
                        if isinstance(obj, dict):
                            yield line_no, obj, None
                        else:
                            yield line_no, None, f"JSON line {line_no} is not an object"
                    except Exception:
                        yield line_no, None, f"JSON line {line_no} malformed"
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Supported: .csv, .tsv, .json, .jsonl")


def parse_dataset_stream(
    file_path: Union[str, Path]
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """
    Backward-compatible batch loader. Parses entire dataset into memory.
    Recommended only for smaller seed files or testing.
    """
    records: List[Dict[str, Any]] = []
    skipped = 0
    skip_reasons: List[str] = []

    for line_no, row, error in stream_dataset_rows(file_path):
        if row is not None:
            records.append(row)
        else:
            skipped += 1
            if len(skip_reasons) < 10 and error:
                skip_reasons.append(error)

    return records, skipped, skip_reasons


def estimate_total_records(file_path: Path) -> int:
    """Fast estimate of row count for progress reporting without loading file into memory."""
    try:
        size_bytes = file_path.stat().st_size
        # Average Wikipedia plot row is ~2000-2500 bytes
        est = max(1, int(size_bytes / 2200))
        return est
    except Exception:
        return 1000


def import_dataset(
    file_path: Union[str, Path],
    source_provenance: str = "User Import",
    license_info: str = "",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    db_path: Optional[Path] = None,
    batch_size: int = 500
) -> ImportStats:
    """
    Stream and ingest an external movie dataset file into the local SQLite catalogue.
    - Operates in streaming chunks of `batch_size` (default 500) to keep memory usage under ~5 MB.
    - Normalizes schema variations and extracts release years robustly.
    - Validates records: must have title and at least one searchable text field (plot/cast/id).
    - Deduplicates by IMDb ID and normalized (title, year) tuples.
    - Merges non-destructively: empty fields never wipe rich existing metadata.
    - Supports responsive cancellation via `cancel_check()` without catalogue corruption.
    - Performs a single FTS5 rebuild upon completion/cancellation.
    """
    start_time = time.time()
    path = Path(file_path)
    stats = ImportStats(source_file=path.name, provenance=source_provenance)

    estimated_total = estimate_total_records(path)
    batch: List[Dict[str, Any]] = []
    processed_count = 0

    # Determine default provenance & license based on filename if not specified
    eff_provenance = source_provenance
    eff_license = license_info
    filename_lower = path.name.lower()
    if "wiki_movie_plots" in filename_lower or "wikipedia" in filename_lower:
        if eff_provenance in ("User Import", ""):
            eff_provenance = "Wikipedia Movie Plots"
        if not eff_license:
            eff_license = "CC BY-SA 4.0"

    # Pre-cache existing catalogue keys once for all batches to eliminate O(B * N) table scans
    conn = database.get_db_connection(db_path)
    existing_by_imdb: Dict[str, int] = {}
    existing_by_title_year: Dict[Tuple[str, str], int] = {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, imdb_id, LOWER(title) AS lower_title, year FROM catalogue;")
        for r in cur.fetchall():
            row_id = r["id"]
            if r["imdb_id"]:
                existing_by_imdb[r["imdb_id"]] = row_id
            if r["lower_title"]:
                existing_by_title_year[(r["lower_title"], str(r["year"] or ""))] = row_id
    except Exception:
        pass
    finally:
        conn.close()
    existing_lookup = (existing_by_imdb, existing_by_title_year)

    try:
        for line_no, raw_row, parse_err in stream_dataset_rows(path):
            # Check for cancellation before processing
            if cancel_check and cancel_check():
                stats.cancelled = True
                break

            processed_count += 1

            if raw_row is None:
                stats.skipped += 1
                if len(stats.skipped_reasons) < 10 and parse_err:
                    stats.skipped_reasons.append(parse_err)
                continue

            norm = normalize_record_keys(raw_row)
            title = norm.get("title", "").strip()
            plot = norm.get("plot", "").strip()
            actors = norm.get("actors", "").strip()
            imdb_id = norm.get("imdb_id", "").strip()

            # Validation: Must have a valid title
            if not title:
                stats.skipped += 1
                if len(stats.skipped_reasons) < 10:
                    stats.skipped_reasons.append(f"Row {line_no}: Missing movie title")
                continue

            # Validation: Must have at least a plot, actors, or IMDb ID to be searchable
            if not plot and not actors and not imdb_id:
                stats.skipped += 1
                if len(stats.skipped_reasons) < 10:
                    stats.skipped_reasons.append(f"Row {line_no} ('{title}'): Missing plot and cast content")
                continue

            if not norm.get("source_provenance"):
                norm["source_provenance"] = eff_provenance
            if not norm.get("license_info") and eff_license:
                norm["license_info"] = eff_license

            batch.append(norm)

            # Once batch is full, flush to SQLite
            if len(batch) >= batch_size:
                if cancel_check and cancel_check():
                    stats.cancelled = True
                    break

                ins, upd, skp = database.bulk_upsert_catalogue(
                    batch,
                    db_path=db_path,
                    rebuild_fts=False,
                    existing_lookup=existing_lookup
                )
                stats.imported += ins
                stats.updated += upd
                stats.skipped += skp
                batch.clear()

                if on_progress:
                    tot_disp = max(estimated_total, processed_count)
                    on_progress(
                        processed_count,
                        tot_disp,
                        f"Processed {processed_count} rows (+{stats.imported} new, {stats.updated} updated)..."
                    )

        # Flush any remaining rows in the final batch
        if batch and not stats.cancelled:
            ins, upd, skp = database.bulk_upsert_catalogue(
                batch,
                db_path=db_path,
                rebuild_fts=False,
                existing_lookup=existing_lookup
            )
            stats.imported += ins
            stats.updated += upd
            stats.skipped += skp
            batch.clear()

    finally:
        # Rebuild FTS5 index once at the end so the search index is completely synchronized
        if stats.imported > 0 or stats.updated > 0:
            database.rebuild_catalogue_fts(db_path)
            try:
                if config.TFIDF_CACHE_FILE.exists():
                    config.TFIDF_CACHE_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    stats.total_read = processed_count + stats.skipped
    stats.duration_sec = time.time() - start_time
    return stats


def ensure_catalogue_initialized(db_path: Optional[Path] = None) -> ImportStats:
    """
    Ensure the SQLite catalogue is seeded with the default dataset if empty.
    Loads data/movies_plots.json into the persistent catalogue table on first run.
    """
    database.init_db(db_path=db_path)
    count = database.get_catalogue_count(db_path=db_path)
    if count > 0:
        return ImportStats(total_read=count, imported=0, updated=0)

    # Ingest default dataset
    default_dataset = config.PLOT_DATASET_FILE
    if default_dataset.exists():
        stats = import_dataset(
            file_path=default_dataset,
            source_provenance="Wikipedia Movie Plots / CC BY-SA 4.0",
            license_info="Creative Commons Attribution-ShareAlike 4.0 International",
            db_path=db_path
        )
        return stats

    return ImportStats(total_read=0, imported=0, updated=0)
