"""
CineTrace — High-DPI Helper
Manages Windows High-DPI process awareness, display metrics,
consistent pixel scaling, and Tkinter font scaling across varying resolutions
(including 1080p, 1440p, 4K, and 125%, 150%, 200% DPI display scalings).
"""

import sys
import ctypes
from typing import Optional, Any

_DPI_INITIALIZED = False
_CACHED_DPI: Optional[int] = None


def enable_high_dpi_awareness() -> bool:
    """
    Enable Windows High-DPI awareness before Tkinter initializes its first window.
    Strictly follows Windows API hierarchy:
      1. Per-Monitor V2 (DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4) [Win 10 1703+]
      2. Per-Monitor V1 (PROCESS_PER_MONITOR_DPI_AWARE = 2) [Win 8.1+]
      3. System DPI Aware (PROCESS_SYSTEM_DPI_AWARE = 1)
      4. Legacy SetProcessDPIAware() [Win Vista/7]
    Returns True if an awareness mode was successfully applied, False otherwise.
    """
    global _DPI_INITIALIZED
    if _DPI_INITIALIZED:
        return True

    if sys.platform != "win32":
        _DPI_INITIALIZED = True
        return False

    success = False

    # 1. Windows 10 Creators Update 1703+ (Per-Monitor V2)
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is ((DPI_AWARENESS_CONTEXT)-4)
        res = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if res:
            success = True
    except Exception:
        pass

    # 2. Windows 8.1+ Per-Monitor V1
    if not success:
        try:
            shcore = ctypes.windll.shcore
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            res = shcore.SetProcessDpiAwareness(2)
            if res == 0:
                success = True
        except Exception:
            pass

    # 3. System DPI Aware
    if not success:
        try:
            shcore = ctypes.windll.shcore
            # PROCESS_SYSTEM_DPI_AWARE = 1
            res = shcore.SetProcessDpiAwareness(1)
            if res == 0:
                success = True
        except Exception:
            pass

    # 4. Vista / 7 legacy
    if not success:
        try:
            user32 = ctypes.windll.user32
            res = user32.SetProcessDPIAware()
            if res:
                success = True
        except Exception:
            pass

    _DPI_INITIALIZED = True
    return success


def get_dpi() -> int:
    """
    Query the primary display DPI in dots per inch.
    Standard display = 96 DPI (100% scale).
    120 DPI = 125% scale.
    144 DPI = 150% scale.
    192 DPI = 200% scale.
    """
    global _CACHED_DPI
    if _CACHED_DPI is not None:
        return _CACHED_DPI

    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            hdc = user32.GetDC(0)
            if hdc:
                LOGPIXELSX = 88
                dpi_x = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                user32.ReleaseDC(0, hdc)
                if dpi_x and dpi_x > 0:
                    _CACHED_DPI = int(dpi_x)
                    return _CACHED_DPI
        except Exception:
            pass

    _CACHED_DPI = 96
    return _CACHED_DPI


def get_dpi_for_window(hwnd: int) -> int:
    """
    Query dynamic DPI for a specific window handle (useful in multi-monitor setups).
    Falls back to primary monitor DPI if unavailable.
    """
    if sys.platform == "win32" and hwnd:
        try:
            user32 = ctypes.windll.user32
            user32.GetDpiForWindow.restype = ctypes.c_uint
            user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]
            dpi = user32.GetDpiForWindow(ctypes.c_void_p(hwnd))
            if dpi and dpi > 0:
                return int(dpi)
        except Exception:
            pass
    return get_dpi()


def get_dpi_scale(dpi: Optional[int] = None) -> float:
    """
    Get the scaling factor relative to standard 96 DPI baseline.
    E.g., 96 -> 1.0, 120 -> 1.25, 144 -> 1.5, 192 -> 2.0.
    """
    current_dpi = dpi if dpi is not None else get_dpi()
    return max(0.5, current_dpi / 96.0)


def scale_px(base_px: int, scale: Optional[float] = None) -> int:
    """
    Scale a pixel value by the active display DPI scaling factor.
    Guarantees at least 1 pixel for non-zero inputs.
    Negative numbers or 0 are returned untouched.
    """
    if base_px <= 0:
        return base_px
    s = scale if scale is not None else get_dpi_scale()
    return max(1, int(round(base_px * s)))


def configure_tk_scaling(root: Any) -> float:
    """
    Configure Tkinter's internal font scaling to match the physical display DPI.
    Tkinter points are defined as (pt * tk_scaling), where tk_scaling = dpi / 72.0.
    Returns the configured tk scaling factor.
    """
    dpi = get_dpi()
    tk_scaling = dpi / 72.0
    try:
        root.tk.call("tk", "scaling", tk_scaling)
    except Exception:
        pass
    return tk_scaling
