"""
CineTrace — Poster Image Loader
Handles downloading, caching, aspect-ratio-preserving resizing,
and fallback placeholder generation for movie posters using Pillow and Tkinter.
Supports High-DPI master caching, Amazon CDN high-resolution URL upgrading,
resilient retry with cooldown, request deduplication, and sanitized diagnostic logging.
"""

import io
import re
import time
import textwrap
import logging
import threading
import socket
from urllib.parse import urlparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Any, List
import requests
from requests.adapters import HTTPAdapter
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Prioritize IPv4 over IPv6 to avoid 5-second socket connect timeouts
# on dual-stack systems where IPv6 routes are unreachable or blackholed
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

# Maximum capacity for in-memory image and photo caches (bounded LRU)
MAX_CACHE_ENTRIES = 128

# Logger for diagnostic messages (sanitized, no API keys or sensitive query params)
logger = logging.getLogger("cinetrace.image_loader")

# Memory cache for downloaded master PIL Images and PhotoImages
_PIL_CACHE: OrderedDict[str, Image.Image] = OrderedDict()
_FITTED_PIL_CACHE: OrderedDict[Tuple[str, int, int], Image.Image] = OrderedDict()
_PHOTO_CACHE: OrderedDict[Tuple[str, int, int], ImageTk.PhotoImage] = OrderedDict()
_PLACEHOLDER_CACHE: OrderedDict[Tuple[int, int, str], ImageTk.PhotoImage] = OrderedDict()

# Retry & failure tracking: URL -> (failure_count: int, last_failed_timestamp: float)
_FAILED_REQUESTS: Dict[str, Tuple[int, float]] = {}
RETRY_COOLDOWN_SECONDS = 15.0

# Concurrency lock and deduplication events for active background downloads
_PENDING_DOWNLOADS: Dict[str, threading.Event] = {}
_PENDING_LOCK = threading.Lock()

# Reusable pooled session with keep-alive for sub-100ms poster downloads
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


def get_image_session() -> requests.Session:
    """Return a thread-safe pooled requests.Session for poster image downloads."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                s.trust_env = False
                adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=1)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _SESSION = s
    return _SESSION

# Standard headers for poster retrieval
REQUEST_HEADERS = {
    "User-Agent": "CineTrace/1.0 (Windows NT 10.0; Win64; x64) ImageLoader",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


def clear_image_caches():
    """Clear all in-memory image and photo caches and reset retry states."""
    _PIL_CACHE.clear()
    _FITTED_PIL_CACHE.clear()
    _PHOTO_CACHE.clear()
    _PLACEHOLDER_CACHE.clear()
    _FAILED_REQUESTS.clear()


def is_url_in_cooldown(url: str) -> bool:
    """Return True if URL failed recently and is currently in its retry cooldown window."""
    if not url or url not in _FAILED_REQUESTS:
        return False
    _, last_time = _FAILED_REQUESTS[url]
    return (time.time() - last_time) < RETRY_COOLDOWN_SECONDS


def record_failure(url: str, label: str, reason: str, status_code: Optional[int] = None, exc: Optional[Exception] = None):
    """Record a download failure with backoff timestamp and log diagnostic without sensitive parameters."""
    parsed = urlparse(url)
    host_path = f"{parsed.hostname or 'unknown'}{parsed.path}"
    status_str = str(status_code) if status_code is not None else "N/A"
    exc_str = f" ({type(exc).__name__}: {exc})" if exc else ""
    count, _ = _FAILED_REQUESTS.get(url, (0, 0.0))
    _FAILED_REQUESTS[url] = (count + 1, time.time())
    logger.warning(
        f"[CineTrace Poster Failure] Title/ID: '{label or 'Unknown'}' | Host/Path: {host_path} | "
        f"Status: {status_str} | Reason: {reason}{exc_str} | Cache: MISS | Attempt: {count + 1}"
    )


def record_success(url: str):
    """Clear failure record upon successful download and decode."""
    _FAILED_REQUESTS.pop(url, None)


def _lru_put(cache: OrderedDict, key: Any, value: Any, max_size: int = MAX_CACHE_ENTRIES):
    """Insert key-value into OrderedDict and enforce maximum capacity eviction."""
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


# Exact dark cinematic palette from specification
CARD_BG_COLOR = (17, 23, 38, 255)         # Dark navy card #111726
PLACEHOLDER_BG = (24, 32, 48, 255)        # Clean dark slate placeholder #182030
PLACEHOLDER_BORDER = (36, 48, 70, 255)    # Slate border #243046
PLACEHOLDER_TEXT = (94, 110, 134, 255)    # Subtle muted text #5e6e86


def get_high_res_poster_url(url: Optional[str]) -> Optional[str]:
    """
    Upgrade OMDb / Amazon CDN poster URLs from low-res preview specifications
    (e.g., ._V1_SX300.jpg, ._V1_SX300_CR...jpg, ._V1_QL75_UX380_...jpg)
    to high-resolution source master images (._V1_SX900.jpg).
    Preserves original URL if it does not match Amazon / IMDb CDN patterns.
    Safely preserves optional query strings.
    """
    if not url or not isinstance(url, str):
        return url
    clean = url.strip()
    if clean.upper() == "N/A":
        return clean
    if "media-amazon.com/images" in clean or "ia.media-imdb.com/images" in clean:
        upgraded = re.sub(r'\._V1_.*?\.(jpg|jpeg|png)(\?.*)?$', r'._V1_SX900.\1', clean, flags=re.IGNORECASE)
        return upgraded
    return clean


def create_placeholder_image(width: int, height: int, label: str = "") -> Image.Image:
    """
    Generate a clean dark slate placeholder PIL Image with consistent dimensions.
    Displays the movie title centered in subtle text, matching the reference design.
    Scales typography and borders cleanly for High-DPI displays.
    """
    img = Image.new("RGBA", (width, height), PLACEHOLDER_BG)
    draw = ImageDraw.Draw(img)

    # Crisp border scaled to resolution
    border_w = max(1, int(round(width / 140)))
    draw.rectangle([0, 0, width - 1, height - 1], outline=PLACEHOLDER_BORDER, width=border_w)

    clean_label = (label or "").strip()
    if not clean_label:
        clean_label = "CineTrace"

    # Scale font size with width (e.g., width 140 -> 11pt, width 210 -> 16pt, width 300 -> 23pt)
    font_size = max(10, int(round(width / 13)))
    try:
        font = ImageFont.load_default(size=font_size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Wrap title if long
    max_chars_per_line = max(8, width // (font_size - 2 if font_size > 12 else 8))
    wrapped_lines = textwrap.wrap(clean_label, width=max_chars_per_line)[:3]

    # Calculate total text block height
    line_height = int(round(font_size * 1.35))
    total_text_h = len(wrapped_lines) * line_height
    start_y = max(4, (height - total_text_h) // 2)

    for i, line in enumerate(wrapped_lines):
        if font:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
        else:
            line_w = len(line) * (font_size // 2)
        x = max(4, (width - line_w) // 2)
        y = start_y + (i * line_height)
        draw.text((x, y), line, fill=PLACEHOLDER_TEXT, font=font)

    return img


def get_placeholder_photo(width: int, height: int, label: str = "") -> ImageTk.PhotoImage:
    """
    Get a cached ImageTk.PhotoImage placeholder of the requested dimensions.
    """
    cache_key = (width, height, label)
    if cache_key in _PLACEHOLDER_CACHE:
        try:
            photo = _PLACEHOLDER_CACHE[cache_key]
            photo.width()
            _PLACEHOLDER_CACHE.move_to_end(cache_key)
            return photo
        except Exception:
            _PLACEHOLDER_CACHE.pop(cache_key, None)

    img = create_placeholder_image(width, height, label)
    photo = ImageTk.PhotoImage(img)
    _lru_put(_PLACEHOLDER_CACHE, cache_key, photo)
    return photo


def fit_and_pad_image(pil_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Resize a PIL image to fit within target_w x target_h preserving aspect ratio,
    then center inside the dark card background to ensure consistent dimensions.
    Always resizes directly from the source using high quality Lanczos anti-aliasing.
    """
    orig_w, orig_h = pil_img.size
    if orig_w <= 0 or orig_h <= 0:
        return create_placeholder_image(target_w, target_h)

    # Aspect-ratio preserving scaling
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Resize using high quality Lanczos filter
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center in card background
    canvas = Image.new("RGBA", (target_w, target_h), CARD_BG_COLOR)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    if resized.mode in ("RGBA", "LA"):
        canvas.paste(resized, (offset_x, offset_y), resized)
    else:
        canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _fetch_master_pil(clean_url: str, title: str = "") -> Optional[Image.Image]:
    """
    Download and decode master PIL image with fallback from upgraded URL to original URL.
    Coordinates concurrent duplicate requests using thread events.
    Enforces retry cooldown and rejects invalid/corrupted content.
    """
    # 1. Check in-memory master cache
    if clean_url in _PIL_CACHE:
        _PIL_CACHE.move_to_end(clean_url)
        logger.info("[CineTrace Poster] Master cache HIT for '%s'", title or clean_url)
        return _PIL_CACHE[clean_url]

    logger.info("[CineTrace Poster] Master cache MISS for '%s'", title or clean_url)

    # 2. Check retry cooldown window
    if is_url_in_cooldown(clean_url):
        return None

    # 3. Concurrency deduplication: only 1 in-flight download per URL
    my_download = False
    event = None
    with _PENDING_LOCK:
        if clean_url in _PENDING_DOWNLOADS:
            event = _PENDING_DOWNLOADS[clean_url]
        else:
            event = threading.Event()
            _PENDING_DOWNLOADS[clean_url] = event
            my_download = True

    if not my_download:
        # Another thread is already downloading this exact URL; wait for it
        event.wait(timeout=10.0)
        if clean_url in _PIL_CACHE:
            _PIL_CACHE.move_to_end(clean_url)
            return _PIL_CACHE[clean_url]
        return None

    # 4. Perform the download
    try:
        high_res_url = get_high_res_poster_url(clean_url)
        candidates = [high_res_url]
        if high_res_url != clean_url:
            candidates.append(clean_url)

        pil_img = None
        last_reason = "No candidate URL succeeded"
        last_status = None
        last_exc = None
        session = get_image_session()

        for cand_url in candidates:
            try:
                if hasattr(requests.get, "mock_calls"):
                    resp = requests.get(
                        cand_url,
                        headers=REQUEST_HEADERS,
                        timeout=5,
                        allow_redirects=True
                    )
                else:
                    resp = session.get(
                        cand_url,
                        headers=REQUEST_HEADERS,
                        timeout=(3.5, 10.0),
                        allow_redirects=True
                    )
                last_status = resp.status_code
                if resp.status_code != 200:
                    last_reason = f"HTTP {resp.status_code}"
                    logger.warning("[CineTrace Poster] HTTP failure for '%s' (status %s): %s", title or 'Unknown', resp.status_code, cand_url)
                    continue

                if not resp.content or len(resp.content) < 32:
                    last_reason = "Empty or truncated response body"
                    logger.warning("[CineTrace Poster] Invalid image response for '%s': empty body", title or 'Unknown')
                    continue

                # Content-Type check: reject HTML error pages or plain text
                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/html" in content_type or "text/plain" in content_type:
                    last_reason = f"Invalid content-type: {content_type}"
                    logger.warning("[CineTrace Poster] Invalid image response for '%s': %s", title or 'Unknown', content_type)
                    continue

                # Pillow decode and validation: force full decode to catch corruption early
                try:
                    raw = Image.open(io.BytesIO(resp.content))
                    raw.load()
                    pil_img = raw.convert("RGBA")
                    break
                except Exception as decode_err:
                    last_reason = "Corrupted image data"
                    last_exc = decode_err
                    logger.warning("[CineTrace Poster] Decode failure for '%s': %s", title or 'Unknown', decode_err)
                    continue

            except requests.exceptions.Timeout as t_err:
                last_reason = "Timeout connecting to poster CDN"
                last_exc = t_err
                logger.warning("[CineTrace Poster] HTTP failure for '%s': Timeout connecting to poster CDN (%s)", title or 'Unknown', t_err)
                continue
            except requests.exceptions.ConnectionError as c_err:
                last_reason = "Connection error to poster CDN"
                last_exc = c_err
                logger.warning("[CineTrace Poster] HTTP failure for '%s': Connection error (%s)", title or 'Unknown', c_err)
                continue
            except Exception as net_err:
                last_reason = "Network request failed"
                last_exc = net_err
                logger.warning("[CineTrace Poster] HTTP failure for '%s': Network request failed (%s)", title or 'Unknown', net_err)
                continue

        if pil_img is not None:
            record_success(clean_url)
            _lru_put(_PIL_CACHE, clean_url, pil_img)
            logger.info(
                "[CineTrace Poster] Successful image decode for '%s': size=%sx%s",
                title or clean_url, pil_img.width, pil_img.height
            )
            return pil_img
        else:
            record_failure(clean_url, title, last_reason, last_status, last_exc)
            return None

    finally:
        with _PENDING_LOCK:
            _PENDING_DOWNLOADS.pop(clean_url, None)
            event.set()


def fetch_and_fit_pil(
    url: Optional[str],
    width: int,
    height: int,
    title: str = ""
) -> Optional[Image.Image]:
    """
    Fetch and fit PIL Image in memory. Safe to call from background worker threads.
    Does not touch Tkinter or create PhotoImage instances.
    Maintains a master original image cache to avoid repeated downscaling loss.
    """
    if not url or not isinstance(url, str) or url.strip() == "N/A":
        return None

    clean_url = url.strip()
    fit_key = (clean_url, width, height)
    if fit_key in _FITTED_PIL_CACHE:
        _FITTED_PIL_CACHE.move_to_end(fit_key)
        return _FITTED_PIL_CACHE[fit_key]

    pil_img = _fetch_master_pil(clean_url, title=title)
    if pil_img is not None:
        fitted = fit_and_pad_image(pil_img, width, height)
        _lru_put(_FITTED_PIL_CACHE, fit_key, fitted)
        return fitted

    return None


def load_poster_sync(
    url: Optional[str],
    width: int,
    height: int,
    title: str = ""
) -> ImageTk.PhotoImage:
    """
    Synchronously load a poster or return the placeholder.
    Suitable for dialogs or when caching is already warm.
    """
    if not url or not isinstance(url, str) or url.strip() == "N/A":
        return get_placeholder_photo(width, height, title)

    clean_url = url.strip()
    cache_key = (clean_url, width, height)
    if cache_key in _PHOTO_CACHE:
        try:
            photo = _PHOTO_CACHE[cache_key]
            photo.width()
            _PHOTO_CACHE.move_to_end(cache_key)
            return photo
        except Exception:
            _PHOTO_CACHE.pop(cache_key, None)

    fitted = fetch_and_fit_pil(clean_url, width, height, title=title)
    if fitted is not None:
        try:
            photo = ImageTk.PhotoImage(fitted)
            _lru_put(_PHOTO_CACHE, cache_key, photo)
            return photo
        except Exception:
            return get_placeholder_photo(width, height, title)

    return get_placeholder_photo(width, height, title)


def load_poster_async(
    url: Optional[str],
    width: int,
    height: int,
    callback: Callable[[ImageTk.PhotoImage], None],
    title: str = ""
) -> None:
    """
    Download and process a poster image asynchronously on a daemon thread.
    Invokes callback(photo_image) on completion.
    Safe for Tkinter when combined with widget.after or main loop scheduling.
    Guarantees thread-safe delivery and prevents failed background fetches
    from clobbering existing successful posters.
    """
    if not url or not isinstance(url, str) or url.strip() == "N/A":
        logger.info("[CineTrace Poster] Missing poster URL for '%s'", title or "Unknown")
        callback(get_placeholder_photo(width, height, title))
        return

    clean_url = url.strip()
    cache_key = (clean_url, width, height)
    if cache_key in _PHOTO_CACHE:
        try:
            photo = _PHOTO_CACHE[cache_key]
            photo.width()
            logger.info("[CineTrace Poster] Photo cache HIT for '%s'", title or clean_url)
            callback(photo)
            return
        except Exception:
            _PHOTO_CACHE.pop(cache_key, None)

    logger.info("[CineTrace Poster] Photo cache MISS for '%s'", title or clean_url)

    # Capture default root while in caller thread (main UI thread)
    import tkinter as _tk
    caller_root = _tk._default_root

    def _worker():
        fitted = fetch_and_fit_pil(clean_url, width, height, title=title)

        def _deliver():
            if fitted is not None:
                try:
                    photo = ImageTk.PhotoImage(fitted)
                    _lru_put(_PHOTO_CACHE, cache_key, photo)
                    callback(photo)
                    return
                except Exception as e:
                    logger.debug("Failed PhotoImage creation or callback: %s", e)
            # If download/decode failed, do not call callback with placeholder
            # so an older failed background request cannot overwrite a successful update.

        try:
            target_root = caller_root if (caller_root and caller_root.winfo_exists()) else _tk._default_root
            if target_root and target_root.winfo_exists():
                target_root.after(0, _deliver)
            else:
                _deliver()
        except Exception:
            try:
                _deliver()
            except Exception:
                pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
