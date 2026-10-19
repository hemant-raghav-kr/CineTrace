"""
CineTrace — Desktop Movie Discovery & Tracking
Built exclusively with Python's Tkinter and ttk GUI framework and Pillow for image handling.
Dynamically retrieves and searches real movie data using the OMDb API.
Faithfully recreates the dark cinematic reference design.
"""

import math
import queue
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("cinetrace.main")

import config
from omdb_client import (
    OMDbClient,
    OMDbError,
    OMDbApiKeyMissingError,
    OMDbApiKeyInvalidError,
    OMDbConnectionError,
    OMDbMovieNotFoundError,
)
from image_loader import (
    load_poster_async,
    get_placeholder_photo,
    CARD_BG_COLOR,
)
from movie_details_dialog import MovieDetailsDialog
from api_key_dialog import ApiKeyDialog
from catalogue_stats_dialog import CatalogueStatsDialog
from semantic_dialog import SemanticDialog
from preferences_dialog import LanguagePreferencesDialog
from discovery_service import DiscoveryService
from dpi_helper import (
    enable_high_dpi_awareness,
    get_dpi,
    get_dpi_scale,
    scale_px,
    configure_tk_scaling,
)
import database
import language_service
import catalogue_service
import semantic_service
import scene_matcher
from scene_matcher import SceneMatcher, DatasetNotFoundError

# Enable Windows High-DPI process awareness before any Tkinter windows are initialized
enable_high_dpi_awareness()

# Design constants
COLOR_BG = "#0a0e17"           # Deep obsidian background
COLOR_SIDEBAR = "#080b12"      # Sidebar background
COLOR_PANEL = "#111726"        # Dark navy panels/cards
COLOR_BORDER = "#1d283c"       # Subtle border
COLOR_ACCENT = "#e11d48"       # Bright rose/crimson accent
COLOR_ACCENT_HOVER = "#be123c" # Accent hover
COLOR_TEXT_MAIN = "#ffffff"    # Pure white heading
COLOR_TEXT_MUTED = "#8e9db3"   # Slate blue muted text
COLOR_TEXT_DIM = "#5a687d"     # Secondary subtle grey
COLOR_BTN_CARD = "#1b2334"     # Card button background
COLOR_BTN_CARD_HOVER = "#253148"

GENRES_LIST = ["All", "Sci-Fi", "Action", "Drama", "Animation", "Comedy", "Thriller", "Adventure", "Romance"]


class NavPillButton(tk.Canvas):
    """Rounded pill button for sidebar navigation matching the design reference."""

    def __init__(self, parent, text: str, icon: str, command, is_active: bool = False, **kwargs):
        super().__init__(
            parent,
            width=scale_px(175),
            height=scale_px(38),
            bg=COLOR_SIDEBAR,
            highlightthickness=0,
            cursor="hand2",
            **kwargs
        )
        self.text = text
        self.icon = icon
        self.command = command
        self.is_active = is_active
        self.is_hovered = False

        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

        self.redraw()

    def set_active(self, active: bool):
        self.is_active = active
        self.redraw()

    def _on_enter(self, event):
        self.is_hovered = True
        self.redraw()

    def _on_leave(self, event):
        self.is_hovered = False
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = scale_px(175), scale_px(38)
        r = scale_px(14)  # corner radius

        if self.is_active:
            fill_color = COLOR_ACCENT
            text_color = "#ffffff"
            font_weight = "bold"
        elif self.is_hovered:
            fill_color = "#141c2c"
            text_color = "#ffffff"
            font_weight = "normal"
        else:
            fill_color = COLOR_SIDEBAR
            text_color = "#7c8ba1"
            font_weight = "normal"

        # Smooth rounded pill shape
        points = (
            r, 2,  w - r, 2,
            w - 2, 2,  w - 2, r,
            w - 2, h - r,  w - 2, h - 2,
            w - r, h - 2,  r, h - 2,
            2, h - 2,  2, h - r,
            2, r,  2, 2
        )
        self.create_polygon(points, smooth=True, fill=fill_color, outline="")

        # Text with icon
        label_text = f"{self.icon}  {self.text}"
        self.create_text(
            scale_px(22), h // 2,
            text=label_text,
            anchor="w",
            fill=text_color,
            font=("Segoe UI", 10, font_weight)
        )


class CineTraceApp(tk.Tk):
    """Main CineTrace Application Window."""

    def __init__(self):
        super().__init__()

        # Configure High-DPI font scaling and layout metrics
        configure_tk_scaling(self)

        self.title("CineTrace")
        self.geometry(f"{scale_px(1260)}x{scale_px(780)}")
        self.minsize(scale_px(1060), scale_px(680))
        self.configure(bg=COLOR_BG)

        # Initialize SQLite database automatically on application start
        database.init_db()

        # Services
        self.client = OMDbClient()
        self.discovery = DiscoveryService(self.client)

        # State
        self.active_tab = "Home"
        self.active_genre = "All"
        self.current_search_query = ""
        self.search_generation = 0
        self.active_posters: Dict[str, Any] = {}

        # Curated Popular Picks & Search containers
        self.raw_home_featured: Optional[Dict[str, Any]] = None
        self.raw_curated_picks: List[Dict[str, Any]] = []
        self.preferred_languages: List[str] = database.get_preferred_languages()
        self.home_featured: Optional[Dict[str, Any]] = None
        self.home_picks: List[Dict[str, Any]] = []
        self.home_trending: List[Dict[str, Any]] = []  # alias for backwards compatibility
        self.curated_picks: List[Dict[str, Any]] = []
        self.discover_movies: List[Dict[str, Any]] = []

        # Scene-to-Movie Finder state
        self.scene_matcher: Optional[SceneMatcher] = None
        self.scene_dataset_status: str = "Loading dataset..."
        self.scene_dataset_error: Optional[str] = None
        self.scene_search_generation: int = 0
        self.scene_query_text: str = ""
        self.scene_selected_genre = tk.StringVar(value="All")
        self.scene_search_mode_var = tk.StringVar(value="Hybrid (All Signals)")
        self.scene_min_threshold_var = tk.StringVar(value="10% (Balanced)")
        self.scene_min_year_var = tk.StringVar(value="")
        self.scene_max_year_var = tk.StringVar(value="")
        self.scene_rerank_var = tk.BooleanVar(value=config.CROSS_ENCODER_ENABLED)
        self.scene_results: List[Dict[str, Any]] = []
        self.scene_card_widgets: Dict[str, Dict[str, Any]] = {}
        self.scene_has_searched: bool = False
        self.scene_is_searching: bool = False
        self.scene_is_importing: bool = False
        self.scene_cancel_import_requested: bool = False

        # Thread-safe UI dispatch queue for background worker threads
        self._poll_timer_id = None
        self._gui_queue: queue.Queue = queue.Queue()
        self._poll_gui_queue()

        self._configure_styles()
        self._build_layout()

        # Initialize dataset matching engine asynchronously
        self._init_scene_matcher()

        # Load dynamic home data asynchronously
        self._load_dynamic_home_data()

        # Show first-launch onboarding if user has not completed or skipped it
        if not database.is_onboarding_completed():
            self.after(350, self._show_first_launch_onboarding)

    def _poll_gui_queue(self):
        """Process thread-safe UI update callbacks posted by background worker threads."""
        try:
            while not self._gui_queue.empty():
                callback, args = self._gui_queue.get_nowait()
                try:
                    callback(*args)
                except Exception:
                    pass
        finally:
            try:
                if self.winfo_exists():
                    self._poll_timer_id = self.after(20, self._poll_gui_queue)
            except Exception:
                pass

    def destroy(self):
        """Cancel background poll timer and destroy window cleanly."""
        if hasattr(self, "_poll_timer_id") and self._poll_timer_id:
            try:
                self.after_cancel(self._poll_timer_id)
            except Exception:
                pass
            self._poll_timer_id = None
        super().destroy()

    def safe_ui_call(self, callback, *args):
        """Thread-safe way to dispatch UI execution from any worker thread."""
        self._gui_queue.put((callback, args))

    def _safe_apply_poster_callback(
        self,
        gen_attr_name: str,
        expected_gen: int,
        card: Optional[tk.Widget],
        poster_lbl: Optional[tk.Widget],
        imdb_id: str,
        photo: Any
    ):
        """Safely schedule poster application on the main UI thread with widget existence checks."""
        def _apply():
            if getattr(self, gen_attr_name, expected_gen) == expected_gen:
                try:
                    if card is not None and not card.winfo_exists():
                        return
                    if poster_lbl is not None and poster_lbl.winfo_exists():
                        self._apply_card_poster(poster_lbl, imdb_id, photo)
                except Exception:
                    pass
        self.safe_ui_call(_apply)

    def _configure_styles(self):
        """Configure ttk theme and styles."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(
            "Slim.Vertical.TScrollbar",
            gripcount=0,
            background="#1e293b",
            darkcolor=COLOR_BG,
            lightcolor=COLOR_BG,
            troughcolor=COLOR_BG,
            bordercolor=COLOR_BG,
            arrowcolor="#5a687d"
        )

    def _build_layout(self):
        """Construct the overall sidebar + main viewport layout."""
        # Main layout container
        self.main_container = tk.Frame(self, bg=COLOR_BG)
        self.main_container.pack(fill="both", expand=True)

        # Left Sidebar (width 210px)
        self._build_sidebar()

        # Right Viewport Area
        self.viewport = tk.Frame(self.main_container, bg=COLOR_BG)
        self.viewport.pack(side="left", fill="both", expand=True)

        # Viewport Top Bar (Screen Title & Search Entry)
        self._build_top_bar()

        # Viewport Scrollable Area
        self._build_scrollable_viewport()

    def _build_sidebar(self):
        """Build the left sidebar matching the reference design."""
        self.sidebar = tk.Frame(self.main_container, bg=COLOR_SIDEBAR, width=scale_px(210))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Top padding frame
        top_frame = tk.Frame(self.sidebar, bg=COLOR_SIDEBAR, padx=scale_px(18), pady=scale_px(24))
        top_frame.pack(fill="x", side="top")

        # CINETRACE Brand Logo
        logo_box = tk.Frame(top_frame, bg=COLOR_SIDEBAR)
        logo_box.pack(anchor="w")

        tk.Label(
            logo_box,
            text="CINE",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg=COLOR_SIDEBAR
        ).pack(side="left")

        tk.Label(
            logo_box,
            text="TRACE",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_SIDEBAR
        ).pack(side="left")

        # Tagline: YOUR PERSONAL CINEMA
        tk.Label(
            top_frame,
            text="YOUR PERSONAL CINEMA",
            font=("Segoe UI", 7, "bold"),
            fg="#475569",
            bg=COLOR_SIDEBAR
        ).pack(anchor="w", pady=(2, 28))

        # Section Header: YOUR LIBRARY
        tk.Label(
            top_frame,
            text="YOUR LIBRARY",
            font=("Segoe UI", 7, "bold"),
            fg="#475569",
            bg=COLOR_SIDEBAR
        ).pack(anchor="w", pady=(0, 10))

        # Nav Buttons list
        self.nav_buttons: Dict[str, NavPillButton] = {}

        nav_items = [
            ("Home", "⌂"),
            ("Discover", "⊙"),
            ("Scene Finder", "🎬"),
            ("Wishlist", "♡"),
            ("Backlog", "▤"),
        ]

        for name, icon in nav_items:
            btn = NavPillButton(
                top_frame,
                text=name,
                icon=icon,
                command=lambda n=name: self._switch_tab(n),
                is_active=(name == "Home")
            )
            btn.pack(pady=3, anchor="w")
            self.nav_buttons[name] = btn

        # Sidebar Footer: Quote + API Key Button
        footer = tk.Frame(self.sidebar, bg=COLOR_SIDEBAR, padx=scale_px(18), pady=scale_px(20))
        footer.pack(fill="x", side="bottom")

        tk.Label(
            footer,
            text="Discover your next\nfavourite movie.",
            font=("Segoe UI", 8),
            fg="#475569",
            bg=COLOR_SIDEBAR,
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        self.prefs_status_btn = tk.Button(
            footer,
            text=self._get_language_button_label(),
            command=self._open_preferences_dialog,
            font=("Segoe UI", 8),
            bg=COLOR_SIDEBAR,
            fg="#8e9db3",
            activebackground="#141c2c",
            activeforeground="#ffffff",
            bd=0,
            cursor="hand2"
        )
        self.prefs_status_btn.pack(anchor="w", pady=(0, 6))

        self.key_status_btn = tk.Button(
            footer,
            text="🔑 API: " + config.get_masked_api_key(),
            command=self._open_api_key_dialog,
            font=("Segoe UI", 8),
            bg=COLOR_SIDEBAR,
            fg="#5a687d",
            activebackground="#141c2c",
            activeforeground="#cbd5e1",
            bd=0,
            cursor="hand2"
        )
        self.key_status_btn.pack(anchor="w")

    def _build_top_bar(self):
        """Top bar containing dynamic Screen Title and Search Box."""
        self.top_bar = tk.Frame(self.viewport, bg=COLOR_BG, padx=32, pady=20)
        self.top_bar.pack(fill="x", side="top")

        # Dynamic screen title
        self.screen_title_label = tk.Label(
            self.top_bar,
            text="Your Next Favourite Film",
            font=("Segoe UI", 18, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        )
        self.screen_title_label.pack(side="left", anchor="w")

        # Right Search Area
        search_box = tk.Frame(self.top_bar, bg=COLOR_BG)
        search_box.pack(side="right")

        # Language Preferences Quick Button in Top Bar
        self.lang_filter_btn = tk.Button(
            search_box,
            text=self._get_top_bar_lang_label(),
            command=self._open_preferences_dialog,
            font=("Segoe UI", 9),
            bg="#161f30",
            fg="#8e9db3",
            activebackground="#222f46",
            activeforeground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=0,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        self.lang_filter_btn.pack(side="left", padx=(0, 10))

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_box,
            textvariable=self.search_var,
            font=("Segoe UI", 10),
            bg="#161f30",
            fg="#ffffff",
            insertbackground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#222f46",
            highlightcolor=COLOR_ACCENT,
            width=24
        )
        self.search_entry.pack(side="left", ipady=5, padx=(0, 6))
        self.search_entry.bind("<Return>", lambda e: self._on_search_submitted())

        self.clear_btn = tk.Button(
            search_box,
            text="✕",
            command=self._clear_search,
            font=("Segoe UI", 9, "bold"),
            bg="#161f30",
            fg=COLOR_TEXT_MUTED,
            activebackground="#222f46",
            activeforeground="#ffffff",
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2"
        )
        self.clear_btn.pack(side="left", padx=(0, 6))

        self.search_btn = tk.Button(
            search_box,
            text="Search",
            command=self._on_search_submitted,
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_ACCENT,
            fg="#ffffff",
            activebackground=COLOR_ACCENT_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2"
        )
        self.search_btn.pack(side="left")

    def _build_scrollable_viewport(self):
        """Scrollable canvas for the active tab's content."""
        self.content_container = tk.Frame(self.viewport, bg=COLOR_BG)
        self.content_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.content_container,
            bg=COLOR_BG,
            highlightthickness=0,
            bd=0
        )
        self.scrollbar = ttk.Scrollbar(
            self.content_container,
            orient="vertical",
            command=self.canvas.yview,
            style="Slim.Vertical.TScrollbar"
        )

        self.scroll_content = tk.Frame(self.canvas, bg=COLOR_BG)

        self.scroll_content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.scroll_content,
            anchor="nw"
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfig(self.canvas_window, width=event.width)
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.scroll_content)

    def _bind_mousewheel(self, widget: tk.Widget):
        def _on_mousewheel(event):
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            elif event.num == 4:
                self.canvas.yview_scroll(-1, "units")

        widget.bind("<MouseWheel>", _on_mousewheel, add="+")
        widget.bind("<Button-4>", _on_mousewheel, add="+")
        widget.bind("<Button-5>", _on_mousewheel, add="+")

    def _switch_tab(self, tab_name: str):
        """Switch active navigation tab."""
        self.active_tab = tab_name
        for name, btn in self.nav_buttons.items():
            btn.set_active(name == tab_name)

        if tab_name == "Home":
            self.screen_title_label.config(text="Your Next Favourite Film")
            self._render_home_screen()
        elif tab_name == "Discover":
            self.screen_title_label.config(text="Explore the Collection")
            self._render_discover_screen()
        elif tab_name == "Wishlist":
            self.screen_title_label.config(text="Your Wishlist")
            self._render_wishlist_screen()
        elif tab_name == "Backlog":
            self.screen_title_label.config(text="Your Backlog")
            self._render_backlog_screen()
        elif tab_name == "Scene Finder":
            self.screen_title_label.config(text="Scene-to-Movie Finder")
            self._render_scene_finder_screen()

    def _clear_scroll_content(self):
        """Clear all child widgets inside scrollable canvas."""
        for child in self.scroll_content.winfo_children():
            child.destroy()
        self.active_posters.clear()
        self.canvas.yview_moveto(0)

    # -------------------------------------------------------------
    # Dynamic Data Fetching (Strictly NO Hardcoded Movies)
    # -------------------------------------------------------------
    def _load_dynamic_home_data(self, force_refresh: bool = False):
        """Fetch real curated movie data dynamically from OMDb and personalise."""
        def _worker():
            try:
                data = self.discovery.fetch_home_data(force_refresh=force_refresh)
                raw_featured = data.get("featured")
                raw_picks = data.get("picks") or data.get("trending", [])
                self.raw_home_featured = raw_featured
                self.raw_curated_picks = raw_picks

                # Apply language preferences
                self.preferred_languages = database.get_preferred_languages()
                feat, picks = language_service.personalize_recommendations(
                    raw_featured, raw_picks, self.preferred_languages
                )
                self.home_featured = feat
                self.home_picks = picks
                self.home_trending = picks
                self.curated_picks = picks
                if not self.current_search_query:
                    self.discover_movies = picks
                self.safe_ui_call(self._on_dynamic_data_loaded)
            except OMDbError as e:
                self.safe_ui_call(self._on_dynamic_data_failed, str(e))
            except Exception as e:
                self.safe_ui_call(self._on_dynamic_data_failed, f"Unexpected error: {str(e)}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_dynamic_data_loaded(self):
        self._update_preference_ui_indicators()
        if self.active_tab == "Home":
            self._render_home_screen()
        elif self.active_tab == "Discover":
            self._render_discover_screen()

    def _on_dynamic_data_failed(self, error_message: str):
        if self.active_tab in ("Home", "Discover"):
            self._clear_scroll_content()
            self._render_notice_box(
                title="Unable to Load Movies",
                message=error_message,
                show_key_btn=not config.is_api_key_configured()
            )

    # -------------------------------------------------------------
    # Tab 1: HOME SCREEN (Hero Featured + Popular Picks Grid)
    # -------------------------------------------------------------
    def _render_home_screen(self):
        self._clear_scroll_content()

        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=10)
        container.pack(fill="both", expand=True)
        self._bind_mousewheel(container)

        # Check if dynamic data is available
        if not self.home_featured and not self.home_picks:
            if not config.is_api_key_configured():
                self._render_notice_box(
                    title="OMDb API Key Required",
                    message="Please configure your free OMDb API key to discover real movies.",
                    show_key_btn=True
                )
            else:
                self._render_loading_state(container, "Discovering curated popular picks...")
            return

        # 1. Hero Card ("Featured Pick")
        if self.home_featured:
            self._render_hero_featured_card(container, self.home_featured)

        # 2. Section Heading: Popular Picks
        heading_frame = tk.Frame(container, bg=COLOR_BG)
        heading_frame.pack(fill="x", pady=(24, 14))
        self._bind_mousewheel(heading_frame)

        tk.Label(
            heading_frame,
            text="Popular Picks",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(anchor="w")

        if self.preferred_languages:
            sub_text = f"Curated audience favourites personalised for: {', '.join(self.preferred_languages)}"
            sub_fg = "#f43f5e"
        else:
            sub_text = "Curated cinematic classics and audience favourites"
            sub_fg = COLOR_TEXT_DIM

        tk.Label(
            heading_frame,
            text=sub_text,
            font=("Segoe UI", 9),
            fg=sub_fg,
            bg=COLOR_BG
        ).pack(anchor="w", pady=(2, 0))

        # 3. Responsive Card Grid
        self._render_movie_grid(container, self.home_picks)

    def _render_hero_featured_card(self, parent: tk.Widget, movie: Dict[str, Any]):
        """Featured movie banner with high-quality poster and clean text fallback."""
        hero = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=scale_px(28),
            pady=scale_px(20),
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        hero.pack(fill="x", pady=(0, scale_px(10)))
        self._bind_mousewheel(hero)

        poster_url = movie.get("poster_url") or movie.get("poster") or movie.get("Poster")
        has_poster = bool(poster_url and str(poster_url).strip().upper() != "N/A")

        # Two-column row container for text content + featured poster
        row_frame = tk.Frame(hero, bg=COLOR_PANEL)
        row_frame.pack(fill="x", expand=True)
        self._bind_mousewheel(row_frame)

        # Left Column: Film Information
        text_col = tk.Frame(row_frame, bg=COLOR_PANEL)
        text_col.pack(side="left", fill="both", expand=True)
        self._bind_mousewheel(text_col)

        # Small red-pink badge: ✦ FEATURED PICK
        badge = tk.Label(
            text_col,
            text="✦  FEATURED PICK",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_PANEL
        )
        badge.pack(anchor="w")
        self._bind_mousewheel(badge)

        # Movie Title
        title_text = movie.get("title", "Featured Film")
        title_lbl = tk.Label(
            text_col,
            text=title_text,
            font=("Segoe UI", 22, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL
        )
        title_lbl.pack(anchor="w", pady=(scale_px(6), scale_px(4)))
        self._bind_mousewheel(title_lbl)

        # Meta string: Year • Genre • ★ Rating
        year = movie.get("year", "N/A")
        genre = (movie.get("genre") or "Cinema").split(",")[0].strip()
        rating = movie.get("imdb_rating", "N/A")

        meta_frame = tk.Frame(text_col, bg=COLOR_PANEL)
        meta_frame.pack(anchor="w", pady=(0, scale_px(8)))
        self._bind_mousewheel(meta_frame)

        meta_text = f"{year}   •   {genre}   •   ★ {rating}"
        meta_lbl = tk.Label(
            meta_frame,
            text=meta_text,
            font=("Segoe UI", 9, "bold"),
            fg="#f43f5e",
            bg=COLOR_PANEL
        )
        meta_lbl.pack(anchor="w")
        self._bind_mousewheel(meta_lbl)

        # Description / Plot with dynamic wraplength
        plot = movie.get("plot") or "No plot summary available."
        plot_wrap = scale_px(540) if has_poster else scale_px(760)
        plot_lbl = tk.Label(
            text_col,
            text=plot,
            font=("Segoe UI", 10),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            wraplength=plot_wrap,
            justify="left"
        )
        plot_lbl.pack(anchor="w", pady=(0, scale_px(16)))
        self._bind_mousewheel(plot_lbl)

        # Explore Film Action Button
        explore_btn = tk.Button(
            text_col,
            text="Explore Film →",
            command=lambda m=movie: self._open_movie_details(m.get("imdb_id", ""), m.get("title", "")),
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_ACCENT,
            fg="#ffffff",
            activebackground=COLOR_ACCENT_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=18,
            pady=7,
            cursor="hand2"
        )
        explore_btn.pack(anchor="w")
        self._bind_mousewheel(explore_btn)

        # Right Column: Featured Poster (if valid poster_url exists)
        if has_poster:
            feat_w = scale_px(150)
            feat_h = scale_px(220)
            poster_col = tk.Frame(
                row_frame,
                bg=COLOR_PANEL,
                width=feat_w,
                height=feat_h,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER
            )
            poster_col.pack(side="right", padx=(scale_px(24), 0), anchor="center")
            poster_col.pack_propagate(False)
            self._bind_mousewheel(poster_col)

            imdb_id = movie.get("imdb_id", "hero_featured")
            placeholder = get_placeholder_photo(feat_w, feat_h, title_text)
            poster_lbl = tk.Label(poster_col, image=placeholder, bg=COLOR_PANEL, bd=0)
            poster_lbl.image = placeholder
            poster_lbl._movie_title = title_text
            poster_lbl.pack(fill="both", expand=True)
            self._bind_mousewheel(poster_lbl)

            def _apply_hero_poster(photo):
                try:
                    if poster_lbl.winfo_exists():
                        poster_lbl.config(image=photo)
                        poster_lbl.image = photo
                        self.active_posters[imdb_id] = photo
                        logger.info(
                            "[CineTrace Poster] Successful Tkinter widget assignment for Featured Hero '%s' (id=%s)",
                            title_text, imdb_id
                        )
                except Exception as e:
                    logger.warning("[CineTrace Poster] Failed featured hero widget assignment: %s", e)

            load_poster_async(
                poster_url, feat_w, feat_h,
                _apply_hero_poster,
                title=title_text
            )

    # -------------------------------------------------------------
    # Tab 2: DISCOVER SCREEN (Filter Pills + Grid)
    # -------------------------------------------------------------
    def _clear_search(self):
        """Reset search query and restore curated collection in Discover."""
        self.search_var.set("")
        self.current_search_query = ""
        self.active_genre = "All"
        self.discover_movies = self.curated_picks or self.home_picks
        self.screen_title_label.config(text="Explore the Collection")
        if self.active_tab == "Discover":
            self._render_discover_screen()

    def _render_discover_screen(self):
        self._clear_scroll_content()

        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=10)
        container.pack(fill="both", expand=True)
        self._bind_mousewheel(container)

        # Case A: User Search Results Active
        if self.current_search_query:
            heading_frame = tk.Frame(container, bg=COLOR_BG)
            heading_frame.pack(fill="x", pady=(4, 14))
            self._bind_mousewheel(heading_frame)

            top_row = tk.Frame(heading_frame, bg=COLOR_BG)
            top_row.pack(fill="x")
            self._bind_mousewheel(top_row)

            tk.Label(
                top_row,
                text=f"Search Results for '{self.current_search_query}'",
                font=("Segoe UI", 15, "bold"),
                fg=COLOR_TEXT_MAIN,
                bg=COLOR_BG
            ).pack(side="left", anchor="w")

            clear_search_btn = tk.Button(
                top_row,
                text="✕ Clear Search",
                command=self._clear_search,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_BTN_CARD,
                fg="#f87171",
                activebackground=COLOR_BTN_CARD_HOVER,
                activeforeground="#ef4444",
                bd=0,
                padx=12,
                pady=4,
                cursor="hand2"
            )
            clear_search_btn.pack(side="right")

            tk.Label(
                heading_frame,
                text=f"Found {len(self.discover_movies)} matching result{'s' if len(self.discover_movies) != 1 else ''} from OMDb",
                font=("Segoe UI", 9),
                fg=COLOR_TEXT_DIM,
                bg=COLOR_BG
            ).pack(anchor="w", pady=(2, 0))

            if not self.discover_movies:
                self._render_empty_search_state(
                    container,
                    message=f"No movies found matching '{self.current_search_query}'. Try a broader title or franchise keyword.",
                    show_reset=True
                )
            else:
                self._render_movie_grid(container, self.discover_movies)
            return

        # Case B: Curated Popular Picks Collection
        # 1. Filter Pills Row
        pills_frame = tk.Frame(container, bg=COLOR_BG)
        pills_frame.pack(fill="x", pady=(0, 20))
        self._bind_mousewheel(pills_frame)

        for genre in GENRES_LIST:
            is_active = (genre == self.active_genre)
            bg_col = COLOR_ACCENT if is_active else "#161f30"
            fg_col = "#ffffff" if is_active else COLOR_TEXT_MUTED
            bd_val = 0 if is_active else 1

            p_btn = tk.Button(
                pills_frame,
                text=genre,
                command=lambda g=genre: self._on_genre_filter_clicked(g),
                font=("Segoe UI", 9, "bold" if is_active else "normal"),
                bg=bg_col,
                fg=fg_col,
                activebackground=COLOR_ACCENT_HOVER if is_active else "#1e2a40",
                activeforeground="#ffffff",
                bd=bd_val,
                relief="solid",
                padx=14,
                pady=5,
                cursor="hand2"
            )
            p_btn.pack(side="left", padx=(0, 8))

        # 2. Section Heading: Popular Picks
        heading_frame = tk.Frame(container, bg=COLOR_BG)
        heading_frame.pack(fill="x", pady=(4, 14))
        self._bind_mousewheel(heading_frame)

        tk.Label(
            heading_frame,
            text="Popular Picks",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(anchor="w")

        if self.preferred_languages:
            sub_text = f"Explore handpicked classics personalised for: {', '.join(self.preferred_languages)}"
            sub_fg = "#f43f5e"
        else:
            sub_text = "Explore handpicked classics across genres"
            sub_fg = COLOR_TEXT_DIM

        tk.Label(
            heading_frame,
            text=sub_text,
            font=("Segoe UI", 9),
            fg=sub_fg,
            bg=COLOR_BG
        ).pack(anchor="w", pady=(2, 0))

        # 3. Filter movies
        curated = self.curated_picks or self.discover_movies or self.home_picks
        if self.active_genre != "All":
            movies_to_show = [
                m for m in curated
                if self.active_genre.lower() in (m.get("genre") or "").lower()
            ]
        else:
            movies_to_show = curated

        if not movies_to_show:
            empty_box = tk.Frame(container, bg=COLOR_BG)
            empty_box.pack(fill="both", expand=True, pady=80)
            self._bind_mousewheel(empty_box)

            tk.Label(
                empty_box,
                text=f"No curated movies found in genre '{self.active_genre}'.",
                font=("Segoe UI", 11),
                fg="#7e8b9f",
                bg=COLOR_BG
            ).pack(anchor="center")

            tk.Button(
                empty_box,
                text="Show All Genres",
                command=lambda: self._on_genre_filter_clicked("All"),
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_ACCENT,
                fg="#ffffff",
                activebackground=COLOR_ACCENT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=16,
                pady=6,
                cursor="hand2"
            ).pack(anchor="center", pady=12)
        else:
            self._render_movie_grid(container, movies_to_show)

    def _on_genre_filter_clicked(self, genre: str):
        self.active_genre = genre
        self._render_discover_screen()

    # -------------------------------------------------------------
    # Tab 3: WISHLIST SCREEN
    # -------------------------------------------------------------
    def _render_wishlist_screen(self):
        self._clear_scroll_content()

        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=10)
        container.pack(fill="both", expand=True)
        self._bind_mousewheel(container)

        heading_frame = tk.Frame(container, bg=COLOR_BG)
        heading_frame.pack(fill="x", pady=(0, 20))
        self._bind_mousewheel(heading_frame)

        tk.Label(
            heading_frame,
            text="Your Collection",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(anchor="w")

        tk.Label(
            heading_frame,
            text="Your saved movies in Wishlist",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_BG
        ).pack(anchor="w", pady=(2, 0))

        movies = database.get_wishlist_movies()
        if not movies:
            self._render_empty_search_state(container, "Your Wishlist is empty. Search or browse movies to add your favourites.")
        else:
            self._render_wishlist_grid(container, movies)

    def _render_wishlist_grid(self, parent: tk.Widget, movies: List[Dict[str, Any]]):
        grid_frame = tk.Frame(parent, bg=COLOR_BG)
        grid_frame.pack(fill="x", pady=6)
        self._bind_mousewheel(grid_frame)

        gen = self.search_generation
        cols = self._get_responsive_columns()

        for c in range(cols):
            grid_frame.columnconfigure(c, weight=1)

        for idx, movie in enumerate(movies):
            row = idx // cols
            col = idx % cols
            self._render_wishlist_card(grid_frame, movie, row, col, gen)

    def _render_wishlist_card(self, parent: tk.Widget, movie: Dict[str, Any], row: int, col: int, gen: int):
        title = movie.get("title", "Untitled")
        year = movie.get("year", "N/A")
        rating = movie.get("imdb_rating") or movie.get("rating") or "N/A"
        imdb_id = movie.get("imdb_id", "")
        poster_url = movie.get("poster_url") or movie.get("poster") or movie.get("Poster")

        card = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            width=scale_px(230),
            height=scale_px(380)
        )
        card.grid(row=row, column=col, padx=8, pady=8, sticky="n")
        card.grid_propagate(False)
        self._bind_mousewheel(card)

        # Poster Container (140x195 scaled)
        p_w = scale_px(140)
        p_h = scale_px(195)
        poster_frame = tk.Frame(card, bg=COLOR_PANEL, width=p_w, height=p_h)
        poster_frame.pack(side="top", pady=(0, 8))
        poster_frame.pack_propagate(False)

        placeholder = get_placeholder_photo(p_w, p_h, title)
        poster_lbl = tk.Label(poster_frame, image=placeholder, bg=COLOR_PANEL, bd=0)
        poster_lbl.image = placeholder
        poster_lbl._movie_title = title
        poster_lbl.pack(fill="both", expand=True)
        self._bind_mousewheel(poster_lbl)

        if poster_url and poster_url != "N/A":
            load_poster_async(
                poster_url, p_w, p_h,
                lambda photo: self._safe_apply_poster_callback("search_generation", gen, card, poster_lbl, imdb_id, photo),
                title=title
            )

        # Title with length safeguard to avoid clipping buttons
        display_title = title if len(title) <= 42 else title[:39] + "..."
        title_lbl = tk.Label(
            card,
            text=display_title,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        title_lbl.pack(fill="x", anchor="w")
        self._bind_mousewheel(title_lbl)

        # Meta
        rating_str = f"★ {rating}" if rating != "N/A" else ""
        meta_text = f"{year}   •   {rating_str}".strip(" •")
        meta_lbl = tk.Label(
            card,
            text=meta_text,
            font=("Segoe UI", 8, "bold"),
            fg="#f43f5e",
            bg=COLOR_PANEL,
            anchor="w"
        )
        meta_lbl.pack(fill="x", anchor="w", pady=(2, 6))
        self._bind_mousewheel(meta_lbl)

        # Action row at bottom: Move to Backlog + Remove
        btn_row = tk.Frame(card, bg=COLOR_PANEL)
        btn_row.pack(side="bottom", fill="x", pady=(4, 0))

        def _on_move_to_backlog(mid=imdb_id, m=movie):
            database.move_wishlist_to_backlog(mid, "Medium", "Want to Watch", m)
            self._render_wishlist_screen()

        def _on_remove_wishlist(mid=imdb_id):
            database.remove_from_wishlist(mid)
            self._render_wishlist_screen()

        move_btn = tk.Button(
            btn_row,
            text="→ Backlog",
            command=_on_move_to_backlog,
            font=("Segoe UI", 8, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            bd=0,
            padx=6,
            pady=4,
            cursor="hand2"
        )
        move_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        remove_btn = tk.Button(
            btn_row,
            text="Remove",
            command=_on_remove_wishlist,
            font=("Segoe UI", 8),
            bg=COLOR_BTN_CARD,
            fg="#f87171",
            activebackground="#2a364e",
            activeforeground="#ef4444",
            bd=0,
            padx=6,
            pady=4,
            cursor="hand2"
        )
        remove_btn.pack(side="right", padx=(4, 0))

        # View Details button
        details_btn = tk.Button(
            card,
            text="View Details",
            command=lambda i=imdb_id, t=title: self._open_movie_details(i, t),
            font=("Segoe UI", 9),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        details_btn.pack(side="bottom", fill="x", pady=(0, 4))

    # -------------------------------------------------------------
    # Tab 4: BACKLOG SCREEN
    # -------------------------------------------------------------
    def _render_backlog_screen(self):
        self._clear_scroll_content()

        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=10)
        container.pack(fill="both", expand=True)
        self._bind_mousewheel(container)

        heading_frame = tk.Frame(container, bg=COLOR_BG)
        heading_frame.pack(fill="x", pady=(0, 20))
        self._bind_mousewheel(heading_frame)

        tk.Label(
            heading_frame,
            text="Your Backlog",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(anchor="w")

        tk.Label(
            heading_frame,
            text="Movies queued to watch later",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_BG
        ).pack(anchor="w", pady=(2, 0))

        movies = database.get_backlog_movies(include_watched=False)
        if not movies:
            self._render_empty_search_state(container, "Your Backlog is empty. Queue movies to watch later.")
        else:
            self._render_backlog_grid(container, movies)

    def _render_backlog_grid(self, parent: tk.Widget, movies: List[Dict[str, Any]]):
        grid_frame = tk.Frame(parent, bg=COLOR_BG)
        grid_frame.pack(fill="x", pady=6)
        self._bind_mousewheel(grid_frame)

        gen = self.search_generation
        cols = self._get_responsive_columns()

        for c in range(cols):
            grid_frame.columnconfigure(c, weight=1)

        for idx, movie in enumerate(movies):
            row = idx // cols
            col = idx % cols
            self._render_backlog_card(grid_frame, movie, row, col, gen)

    def _render_backlog_card(self, parent: tk.Widget, movie: Dict[str, Any], row: int, col: int, gen: int):
        title = movie.get("title", "Untitled")
        year = movie.get("year", "N/A")
        rating = movie.get("imdb_rating") or movie.get("rating") or "N/A"
        imdb_id = movie.get("imdb_id", "")
        poster_url = movie.get("poster_url") or movie.get("poster") or movie.get("Poster")
        priority = movie.get("priority", "Medium")

        card = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            width=scale_px(230),
            height=scale_px(410)
        )
        card.grid(row=row, column=col, padx=8, pady=8, sticky="n")
        card.grid_propagate(False)
        self._bind_mousewheel(card)

        # Poster Container (140x195 scaled)
        p_w = scale_px(140)
        p_h = scale_px(195)
        poster_frame = tk.Frame(card, bg=COLOR_PANEL, width=p_w, height=p_h)
        poster_frame.pack(side="top", pady=(0, 6))
        poster_frame.pack_propagate(False)

        placeholder = get_placeholder_photo(p_w, p_h, title)
        poster_lbl = tk.Label(poster_frame, image=placeholder, bg=COLOR_PANEL, bd=0)
        poster_lbl.image = placeholder
        poster_lbl._movie_title = title
        poster_lbl.pack(fill="both", expand=True)
        self._bind_mousewheel(poster_lbl)

        if poster_url and poster_url != "N/A":
            load_poster_async(
                poster_url, p_w, p_h,
                lambda photo: self._safe_apply_poster_callback("search_generation", gen, card, poster_lbl, imdb_id, photo),
                title=title
            )

        # Title with length safeguard to avoid clipping buttons
        display_title = title if len(title) <= 42 else title[:39] + "..."
        title_lbl = tk.Label(
            card,
            text=display_title,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        title_lbl.pack(fill="x", anchor="w")
        self._bind_mousewheel(title_lbl)

        # Meta
        rating_str = f"★ {rating}" if rating != "N/A" else ""
        meta_text = f"{year}   •   {rating_str}".strip(" •")
        meta_lbl = tk.Label(
            card,
            text=meta_text,
            font=("Segoe UI", 8, "bold"),
            fg="#f43f5e",
            bg=COLOR_PANEL,
            anchor="w"
        )
        meta_lbl.pack(fill="x", anchor="w", pady=(1, 4))
        self._bind_mousewheel(meta_lbl)

        # Priority selector row
        prio_frame = tk.Frame(card, bg=COLOR_PANEL)
        prio_frame.pack(fill="x", pady=(0, 4))
        self._bind_mousewheel(prio_frame)

        tk.Label(
            prio_frame,
            text="Priority:",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left")

        prio_var = tk.StringVar(value=priority)
        prio_combo = ttk.Combobox(
            prio_frame,
            textvariable=prio_var,
            values=["High", "Medium", "Low"],
            state="readonly",
            width=7,
            font=("Segoe UI", 8)
        )
        prio_combo.pack(side="right")

        def _on_prio_change(event, mid=imdb_id, var=prio_var):
            database.update_backlog_priority(mid, var.get())
            self._render_backlog_screen()

        prio_combo.bind("<<ComboboxSelected>>", _on_prio_change)

        # Bottom action row: Mark Watched + Remove
        action_row = tk.Frame(card, bg=COLOR_PANEL)
        action_row.pack(side="bottom", fill="x", pady=(4, 0))

        def _on_mark_watched(mid=imdb_id):
            database.mark_backlog_watched(mid)
            self._render_backlog_screen()

        def _on_remove_backlog(mid=imdb_id):
            database.remove_from_backlog(mid)
            self._render_backlog_screen()

        watched_btn = tk.Button(
            action_row,
            text="✓ Watched",
            command=_on_mark_watched,
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            bd=0,
            padx=6,
            pady=4,
            cursor="hand2"
        )
        watched_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        remove_btn = tk.Button(
            action_row,
            text="Remove",
            command=_on_remove_backlog,
            font=("Segoe UI", 8),
            bg=COLOR_BTN_CARD,
            fg="#f87171",
            activebackground="#2a364e",
            activeforeground="#ef4444",
            bd=0,
            padx=6,
            pady=4,
            cursor="hand2"
        )
        remove_btn.pack(side="right", padx=(4, 0))

        # View Details button
        details_btn = tk.Button(
            card,
            text="View Details",
            command=lambda i=imdb_id, t=title: self._open_movie_details(i, t),
            font=("Segoe UI", 9),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        details_btn.pack(side="bottom", fill="x", pady=(0, 4))

    # -------------------------------------------------------------
    # Tab 5: SCENE-TO-MOVIE FINDER SCREEN (Phase 5.2 Hybrid Search)
    # -------------------------------------------------------------
    def _init_scene_matcher(self):
        """Asynchronously index local movie catalogue with FTS5 and TF-IDF."""
        def _worker():
            try:
                matcher = SceneMatcher()
                self.scene_matcher = matcher
                count = matcher.get_indexed_count()
                db_stats = database.get_catalogue_stats()
                p_cnt = db_stats.get("movies_with_plots", count)
                c_cnt = db_stats.get("movies_with_actors", 0)
                if count > 0:
                    self.scene_dataset_status = f"{count} movies indexed ({p_cnt} plots, {c_cnt} cast)"
                    self.scene_dataset_error = None
                else:
                    self.scene_dataset_status = "No catalogue data configured"
                    self.scene_dataset_error = (
                        "No movies are currently indexed in the catalogue. "
                        "Please use the '📥 Import Dataset' button above to import a CSV or JSON dataset, "
                        "or ensure data/movies_plots.json is present."
                    )
            except DatasetNotFoundError:
                self.scene_dataset_status = "Dataset not found"
                self.scene_dataset_error = (
                    f"Catalogue dataset file not found at '{config.PLOT_DATASET_FILE}'. "
                    "Please ensure data/movies_plots.json is present or import a custom dataset."
                )
            except Exception as e:
                self.scene_dataset_status = "Dataset error"
                self.scene_dataset_error = f"Failed to load catalogue: {str(e)}"

            if self.active_tab == "Scene Finder":
                self.safe_ui_call(lambda: self._render_scene_finder_screen(preserve_results=True))

        threading.Thread(target=_worker, daemon=True).start()

    def _render_scene_finder_screen(self, preserve_results: bool = False):
        self._clear_scroll_content()

        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=10)
        container.pack(fill="both", expand=True)
        self._bind_mousewheel(container)

        # Header Frame
        header = tk.Frame(container, bg=COLOR_BG)
        header.pack(fill="x", pady=(0, 14))
        self._bind_mousewheel(header)

        title_row = tk.Frame(header, bg=COLOR_BG)
        title_row.pack(fill="x")
        self._bind_mousewheel(title_row)

        tk.Label(
            title_row,
            text="Scene-to-Movie Finder",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(side="left")

        # Right-side action buttons on title row
        semantic_btn = tk.Button(
            title_row,
            text="🧠 Semantic Search",
            command=self._open_semantic_dialog,
            font=("Segoe UI", 8, "bold"),
            bg="#0f766e",
            fg="#ffffff",
            activebackground="#0d9488",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2"
        )
        semantic_btn.pack(side="right", padx=(6, 0))

        stats_btn = tk.Button(
            title_row,
            text="📊 Stats",
            command=self._open_catalogue_stats_dialog,
            font=("Segoe UI", 8),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2"
        )
        stats_btn.pack(side="right", padx=(6, 0))

        import_btn = tk.Button(
            title_row,
            text="📥 Import Dataset",
            command=self._on_import_dataset_clicked,
            font=("Segoe UI", 8, "bold"),
            bg="#0369a1",
            fg="#ffffff",
            activebackground="#0284c7",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2"
        )
        import_btn.pack(side="right", padx=(6, 0))

        # Cancel button (only visible during active ingestion)
        self.scene_cancel_btn = tk.Button(
            title_row,
            text="🛑 Cancel Import",
            command=self._on_cancel_import_clicked,
            font=("Segoe UI", 8, "bold"),
            bg="#b91c1c",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=3,
            cursor="hand2"
        )
        if self.scene_is_importing:
            self.scene_cancel_btn.pack(side="right", padx=(6, 0))

        # Semantic status badge
        sem_svc = semantic_service.get_semantic_service(db_path=self.scene_matcher.db_path if self.scene_matcher else None)
        model_info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.DEFAULT_MODEL_NAME)

        # If weights are cached locally and model is not yet in RAM, load offline in background (0 silent downloads)
        if model_info["found"] and not sem_svc.is_model_loaded() and sem_svc.is_available():
            sem_svc.load_model(allow_download=False, on_complete=lambda ok, msg: self._on_semantic_model_bg_loaded(ok, msg))

        if sem_svc.is_ready():
            sem_text = f"🧠 Semantic: Active ({sem_svc.index.count:,} vectors)"
            sem_fg = "#34d399"
        elif sem_svc.is_model_loaded():
            sem_text = "🧠 Semantic: Model Ready"
            sem_fg = "#38bdf8"
        elif model_info["found"]:
            sem_text = "🧠 Semantic: Loading Cached..."
            sem_fg = "#38bdf8"
        elif sem_svc.is_available():
            sem_text = "🧠 Semantic: Standby"
            sem_fg = "#f59e0b"
        else:
            sem_text = "🧠 Semantic: Lexical Fallback"
            sem_fg = "#94a3b8"

        self.scene_semantic_badge = tk.Label(
            title_row,
            text=sem_text,
            font=("Segoe UI", 8),
            fg=sem_fg,
            bg="#111726",
            padx=8,
            pady=3,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        self.scene_semantic_badge.pack(side="right", padx=(0, 6))
        self._bind_mousewheel(self.scene_semantic_badge)

        # Catalogue status badge
        status_text = self.scene_dataset_status or "Loading catalogue..."
        self.scene_status_badge = tk.Label(
            title_row,
            text=f"📚 {status_text}",
            font=("Segoe UI", 8),
            fg="#34d399" if "indexed" in status_text or "ready" in status_text else "#f87171",
            bg="#111726",
            padx=8,
            pady=3,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        self.scene_status_badge.pack(side="right")
        self._bind_mousewheel(self.scene_status_badge)

        tk.Label(
            header,
            text="Search movies by scene description, remembered plot points, character moments, title, or cast. CineTrace matches your query using a local hybrid search index and enriches candidates with real OMDb metadata.",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
            wraplength=850,
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

        # Search Card Panel
        search_card = tk.Frame(
            container,
            bg=COLOR_PANEL,
            padx=16,
            pady=14,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        search_card.pack(fill="x", pady=(0, 16))
        self._bind_mousewheel(search_card)

        tk.Label(
            search_card,
            text="Enter scene description, plot details, title, or cast:",
            font=("Segoe UI", 9, "bold"),
            fg="#cbd5e1",
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 6))

        # Multiline description text box
        self.scene_text_box = tk.Text(
            search_card,
            height=3,
            wrap="word",
            font=("Segoe UI", 10),
            bg="#161f30",
            fg="#ffffff",
            insertbackground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#222f46",
            highlightcolor=COLOR_ACCENT,
            padx=10,
            pady=8
        )
        self.scene_text_box.pack(fill="x", pady=(0, 10))
        self._bind_mousewheel(self.scene_text_box)
        self.scene_text_box.bind("<Control-Return>", lambda e: (self._on_scene_search_clicked(), "break")[1])

        # Restore previous query if available
        if self.scene_query_text:
            self.scene_text_box.insert("1.0", self.scene_query_text)

        # Filters & Controls Row
        ctrl_row = tk.Frame(search_card, bg=COLOR_PANEL)
        ctrl_row.pack(fill="x")
        self._bind_mousewheel(ctrl_row)

        # Search Mode selector
        tk.Label(
            ctrl_row,
            text="Mode:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left", padx=(0, 4))

        mode_combo = ttk.Combobox(
            ctrl_row,
            textvariable=self.scene_search_mode_var,
            values=["Hybrid (All Signals)", "Scene / Plot", "Title", "Actor / Cast"],
            state="readonly",
            width=18,
            font=("Segoe UI", 9)
        )
        mode_combo.pack(side="left", padx=(0, 12))

        # Genre selector
        tk.Label(
            ctrl_row,
            text="Genre:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left", padx=(0, 4))

        genre_combo = ttk.Combobox(
            ctrl_row,
            textvariable=self.scene_selected_genre,
            values=["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Music", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"],
            state="readonly",
            width=10,
            font=("Segoe UI", 9)
        )
        genre_combo.pack(side="left", padx=(0, 12))

        # Year Range filter
        tk.Label(
            ctrl_row,
            text="Year Min:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left", padx=(0, 4))

        min_year_entry = tk.Entry(
            ctrl_row,
            textvariable=self.scene_min_year_var,
            width=6,
            font=("Segoe UI", 9),
            bg="#161f30",
            fg="#ffffff",
            insertbackground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#222f46"
        )
        min_year_entry.pack(side="left", padx=(0, 10))

        tk.Label(
            ctrl_row,
            text="Year Max:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left", padx=(0, 4))

        max_year_entry = tk.Entry(
            ctrl_row,
            textvariable=self.scene_max_year_var,
            width=6,
            font=("Segoe UI", 9),
            bg="#161f30",
            fg="#ffffff",
            insertbackground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#222f46"
        )
        max_year_entry.pack(side="left", padx=(0, 12))

        # Min Match Threshold filter
        tk.Label(
            ctrl_row,
            text="Min Match:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_DIM,
            bg=COLOR_PANEL
        ).pack(side="left", padx=(0, 4))

        threshold_combo = ttk.Combobox(
            ctrl_row,
            textvariable=self.scene_min_threshold_var,
            values=["10% (Balanced)", "15% (High Precision)", "8% (Broad)", "5% (All Potential)"],
            state="readonly",
            width=16,
            font=("Segoe UI", 9)
        )
        threshold_combo.pack(side="left", padx=(0, 10))

        # Cross-Encoder Reranker toggle (Phase 5.7)
        rerank_chk = tk.Checkbutton(
            ctrl_row,
            text="🎯 Rerank",
            variable=self.scene_rerank_var,
            bg=COLOR_PANEL,
            fg="#e2e8f0",
            selectcolor="#161f30",
            activebackground=COLOR_PANEL,
            activeforeground="#ffffff",
            font=("Segoe UI", 9)
        )
        rerank_chk.pack(side="left", padx=(0, 12))
        self._bind_mousewheel(rerank_chk)

        # Search button

        self.scene_search_btn = tk.Button(
            ctrl_row,
            text="Searching..." if self.scene_is_searching else "Find Movie",
            state="disabled" if self.scene_is_searching else "normal",
            command=self._on_scene_search_clicked,
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_ACCENT,
            fg="#ffffff",
            activebackground=COLOR_ACCENT_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=16,
            pady=5,
            cursor="hand2"
        )
        self.scene_search_btn.pack(side="left", padx=(0, 8))

        # Clear button
        clear_btn = tk.Button(
            ctrl_row,
            text="Clear",
            command=self._on_scene_clear_clicked,
            font=("Segoe UI", 9),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=14,
            pady=5,
            cursor="hand2"
        )
        clear_btn.pack(side="left")

        # Search status indicator
        self.scene_status_lbl = tk.Label(
            ctrl_row,
            text="",
            font=("Segoe UI", 9),
            fg="#f43f5e",
            bg=COLOR_PANEL
        )
        self.scene_status_lbl.pack(side="left", padx=14)

        # Results Container
        self.scene_results_frame = tk.Frame(container, bg=COLOR_BG)
        self.scene_results_frame.pack(fill="both", expand=True)
        self._bind_mousewheel(self.scene_results_frame)

        # Check if dataset has error
        if self.scene_dataset_error:
            self._render_scene_empty_state(
                self.scene_results_frame,
                "Catalogue Empty or Not Configured",
                self.scene_dataset_error
            )
            return

        if self.scene_is_searching:
            self._render_scene_empty_state(
                self.scene_results_frame,
                "Searching Movie Catalogue...",
                "Executing hybrid matching across plot descriptions, titles, and cast members."
            )
        elif self.scene_has_searched:
            if not self.scene_results:
                th_str = self.scene_min_threshold_var.get()
                mode_str = self.scene_search_mode_var.get()
                self._render_scene_empty_state(
                    self.scene_results_frame,
                    "No Confident Match Found",
                    f"No movies in our catalogue met the {th_str} match threshold under '{mode_str}' mode.\n\n"
                    "Suggestions:\n"
                    "• Lower the Min Match threshold (e.g. to '8% (Broad)' or '5% (All Potential)').\n"
                    "• Switch to 'Hybrid (All Signals)' to search across titles, cast, and plot simultaneously.\n"
                    "• Include distinctive character names, actor names, or key plot moments.\n"
                    "• Click '📥 Import Dataset' to expand your catalogue coverage."
                )
            else:
                self._render_scene_results_grid(self.scene_results_frame, self.scene_results)
        else:
            self._render_scene_initial_state(self.scene_results_frame)

    def _open_catalogue_stats_dialog(self):
        """Open modal dialog displaying catalogue metrics and licensing details."""
        CatalogueStatsDialog(self)

    def _open_semantic_dialog(self):
        """Open modal dialog to manage local sentence-embedding model and index."""
        db_path = self.scene_matcher.db_path if self.scene_matcher else None
        SemanticDialog(self, db_path=db_path)
        if self.active_tab == "Scene Finder":
            self.safe_ui_call(lambda: self._render_scene_finder_screen(preserve_results=True))

    def _on_semantic_model_bg_loaded(self, ok: bool, msg: str):
        """Callback when local cached model finishes loading into memory in background."""
        def _update():
            if ok and hasattr(self, "scene_semantic_badge") and self.scene_semantic_badge.winfo_exists():
                sem_svc = semantic_service.get_semantic_service(db_path=self.scene_matcher.db_path if self.scene_matcher else None)
                if sem_svc.is_ready():
                    self.scene_semantic_badge.config(
                        text=f"🧠 Semantic: Active ({sem_svc.index.count:,} vectors)",
                        fg="#34d399"
                    )
                elif sem_svc.is_model_loaded():
                    self.scene_semantic_badge.config(
                        text="🧠 Semantic: Model Ready",
                        fg="#38bdf8"
                    )
        self.safe_ui_call(_update)

    def _on_cancel_import_clicked(self):
        """Request immediate cooperative cancellation of the active import."""
        if self.scene_is_importing:
            self.scene_cancel_import_requested = True
            self.scene_status_lbl.config(text="Cancelling import...", fg="#f59e0b")
            if hasattr(self, "scene_cancel_btn") and self.scene_cancel_btn.winfo_exists():
                self.scene_cancel_btn.config(state="disabled", text="Stopping...")

    def _on_import_dataset_clicked(self):
        """Open file chooser to import an external CSV, TSV, or JSON dataset."""
        if self.scene_is_importing:
            messagebox.showinfo("Import in Progress", "A dataset import is already in progress. Click '🛑 Cancel Import' to stop it.", parent=self)
            return

        file_path = filedialog.askopenfilename(
            title="Select Movie Dataset (CSV, TSV, or JSON)",
            filetypes=[
                ("Movie Datasets", "*.csv *.tsv *.json *.jsonl *.txt"),
                ("CSV/TSV Files", "*.csv *.tsv *.txt"),
                ("JSON Files", "*.json *.jsonl"),
                ("All Files", "*.*")
            ],
            parent=self
        )
        if not file_path:
            return

        self.scene_is_importing = True
        self.scene_cancel_import_requested = False
        filename = Path(file_path).name
        self.scene_status_lbl.config(text=f"Importing {filename}...", fg="#38bdf8")

        if hasattr(self, "scene_cancel_btn") and self.scene_cancel_btn.winfo_exists():
            self.scene_cancel_btn.config(state="normal", text="🛑 Cancel Import")
            self.scene_cancel_btn.pack(side="right", padx=(6, 0))

        def _worker():
            try:
                def _prog(cur, tot, msg):
                    self.safe_ui_call(lambda: self.scene_status_lbl.config(text=f"Importing: {cur} records processed...", fg="#38bdf8"))

                stats = catalogue_service.import_dataset(
                    file_path=file_path,
                    source_provenance=f"User Import: {filename}",
                    on_progress=_prog,
                    cancel_check=lambda: self.scene_cancel_import_requested
                )

                if self.scene_matcher:
                    self.scene_matcher.reload()
                else:
                    self.scene_matcher = SceneMatcher()

                count = self.scene_matcher.get_indexed_count()
                db_stats = database.get_catalogue_stats()
                p_cnt = db_stats.get("movies_with_plots", count)
                c_cnt = db_stats.get("movies_with_actors", 0)

                self.safe_ui_call(self._on_import_complete, stats, count, p_cnt, c_cnt)
            except Exception as e:
                self.safe_ui_call(self._on_import_failed, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_import_complete(self, stats: catalogue_service.ImportStats, count: int, p_cnt: int, c_cnt: int):
        self.scene_is_importing = False
        self.scene_cancel_import_requested = False
        if hasattr(self, "scene_cancel_btn") and self.scene_cancel_btn.winfo_exists():
            self.scene_cancel_btn.pack_forget()

        self.scene_dataset_status = f"{count} movies indexed ({p_cnt} plots, {c_cnt} cast)"
        self.scene_dataset_error = None

        if stats.cancelled:
            self.scene_status_lbl.config(text=f"Import cancelled: +{stats.imported} new, {stats.updated} updated.", fg="#f59e0b")
            if self.active_tab == "Scene Finder":
                self._render_scene_finder_screen(preserve_results=True)
            messagebox.showinfo(
                "Import Cancelled",
                f"Import operation cancelled safely without catalogue corruption.\n\n"
                f"• Processed rows before stop: {stats.total_read}\n"
                f"• New Movies Imported: {stats.imported}\n"
                f"• Existing Movies Updated: {stats.updated}\n"
                f"• Invalid Rows Skipped: {stats.skipped}\n"
                f"• Duration: {stats.duration_sec:.2f} seconds\n\n"
                f"Current Catalogue: {count} indexed films ({p_cnt} plots, {c_cnt} cast).",
                parent=self
            )
        else:
            self.scene_status_lbl.config(text=f"Import complete: +{stats.imported} new, {stats.updated} updated.", fg="#34d399")
            if self.active_tab == "Scene Finder":
                self._render_scene_finder_screen(preserve_results=True)
            messagebox.showinfo(
                "Dataset Import Complete",
                f"Catalogue successfully updated!\n\n"
                f"• Source File: {stats.source_file}\n"
                f"• Total Rows Processed: {stats.total_read}\n"
                f"• New Movies Imported: {stats.imported}\n"
                f"• Existing Movies Updated: {stats.updated}\n"
                f"• Invalid Rows Skipped: {stats.skipped}\n"
                f"• Duration: {stats.duration_sec:.2f} seconds\n\n"
                f"Current Catalogue: {count} indexed films ({p_cnt} plots, {c_cnt} cast).",
                parent=self
            )

    def _on_import_failed(self, error_msg: str):
        self.scene_is_importing = False
        self.scene_cancel_import_requested = False
        if hasattr(self, "scene_cancel_btn") and self.scene_cancel_btn.winfo_exists():
            self.scene_cancel_btn.pack_forget()
        self.scene_status_lbl.config(text="Import failed.", fg="#f87171")
        messagebox.showerror(
            "Import Failed",
            f"Failed to import dataset:\n\n{error_msg}\n\n"
            "Please ensure the file is a valid CSV, TSV, or JSON file following the supported schema.",
            parent=self
        )


    def _on_scene_search_clicked(self):
        query = self.scene_text_box.get("1.0", "end-1c").strip()
        if not query:
            self.scene_status_lbl.config(text="Please enter a scene, title, or cast query.", fg="#f87171")
            return

        if not self.scene_matcher or not self.scene_matcher.is_ready():
            try:
                self.scene_matcher = SceneMatcher()
            except Exception as e:
                self.scene_status_lbl.config(text=f"Dataset error: {str(e)}", fg="#f87171")
                return

        # Parse year min and max safely
        y_min = None
        y_max = None
        min_str = self.scene_min_year_var.get().strip()
        max_str = self.scene_max_year_var.get().strip()
        if min_str:
            try:
                y_min = int(min_str)
            except ValueError:
                self.scene_status_lbl.config(text="Invalid Min Year. Please enter a 4-digit year.", fg="#f87171")
                return
        if max_str:
            try:
                y_max = int(max_str)
            except ValueError:
                self.scene_status_lbl.config(text="Invalid Max Year. Please enter a 4-digit year.", fg="#f87171")
                return

        if y_min is not None and y_max is not None and y_min > y_max:
            self.scene_status_lbl.config(text="Min Year cannot be greater than Max Year.", fg="#f87171")
            return

        self.scene_query_text = query
        self.scene_is_searching = True
        self.scene_search_generation += 1
        gen = self.scene_search_generation
        genre = self.scene_selected_genre.get()

        # Parse search mode
        mode_str = self.scene_search_mode_var.get()
        mode = "hybrid"
        if "Plot" in mode_str or "Scene" in mode_str:
            mode = "plot"
        elif "Title" in mode_str:
            mode = "title"
        elif "Actor" in mode_str or "Cast" in mode_str:
            mode = "actor"

        # Parse configurable similarity threshold
        th_val = self.scene_min_threshold_var.get()
        th = 0.10
        if "15%" in th_val:
            th = 0.15
        elif "10%" in th_val:
            th = 0.10
        elif "8%" in th_val:
            th = 0.08
        elif "5%" in th_val:
            th = 0.05
        else:
            try:
                th = float(th_val.replace("%", "").split()[0]) / 100.0
            except Exception:
                th = 0.10

        # Check semantic readiness if in hybrid or plot mode
        sem_svc = semantic_service.get_semantic_service(db_path=self.scene_matcher.db_path if self.scene_matcher else None)
        wants_semantic = (mode in ("hybrid", "plot")) and config.SEMANTIC_SEARCH_ENABLED
        wants_rerank = self.scene_rerank_var.get() and (mode in ("hybrid", "plot"))

        ce_svc = semantic_service.get_cross_encoder_service()
        if wants_rerank and not ce_svc.is_ready():
            ce_info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.CROSS_ENCODER_MODEL_NAME)
            if ce_info["found"] and ce_svc.is_available() and not ce_svc.is_loading_model():
                ce_svc.load_model(allow_download=False)

        if wants_semantic:
            model_info = semantic_service.get_local_model_info(config.MODEL_CACHE_DIR, config.DEFAULT_MODEL_NAME)
            if not sem_svc.is_ready():
                if model_info["found"] and sem_svc.is_available():
                    # Weights found; ensure background loading is active and indicate waiting state
                    if not sem_svc.is_loading_model() and not sem_svc.is_model_loaded():
                        sem_svc.load_model(allow_download=False, on_complete=lambda ok, msg: self._on_semantic_model_bg_loaded(ok, msg))
                    self.scene_search_btn.config(state="disabled", text="Loading Model...")
                    self.scene_status_lbl.config(
                        text="🧠 Loading semantic model... search will execute automatically once ready.",
                        fg="#38bdf8"
                    )
                elif not model_info["found"]:
                    # Model not downloaded locally
                    self.scene_is_searching = False
                    self.scene_search_btn.config(state="normal", text="Find Movie")
                    self._on_semantic_load_failed_for_search(
                        gen, query, mode, genre, y_min, y_max, th,
                        "Semantic model weights not downloaded locally. Download weights or search with lexical fallback."
                    )
                    return
                elif not sem_svc.is_available():
                    self.scene_is_searching = False
                    self.scene_search_btn.config(state="normal", text="Find Movie")
                    self._on_semantic_load_failed_for_search(
                        gen, query, mode, genre, y_min, y_max, th,
                        "sentence-transformers library is not installed."
                    )
                    return
            else:
                self.scene_search_btn.config(state="disabled", text="Searching...")
                self.scene_status_lbl.config(text="Searching catalogue (with rerank)..." if wants_rerank else "Searching catalogue...", fg="#38bdf8")
        else:
            self.scene_search_btn.config(state="disabled", text="Searching...")
            self.scene_status_lbl.config(text="Searching catalogue...", fg="#38bdf8")

        def _worker():
            # If semantic search was requested and model is not yet ready, wait on worker thread
            if wants_semantic and not sem_svc.is_ready():
                loaded_ok = sem_svc.wait_until_ready(timeout=15.0)
                if gen != self.scene_search_generation:
                    return
                if not loaded_ok or not sem_svc.is_ready():
                    err_msg = "Semantic model weights could not be loaded into memory."
                    if sem_svc.model_status == semantic_service.ModelStatus.ERROR:
                        err_msg = "Semantic model encountered an unrecoverable error during loading."
                    self.safe_ui_call(
                        self._on_semantic_load_failed_for_search,
                        gen, query, mode, genre, y_min, y_max, th,
                        err_msg
                    )
                    return

            # Wait briefly if cross-encoder was requested and is currently finishing background load
            if wants_rerank and not ce_svc.is_ready() and ce_svc.is_loading_model():
                ce_svc.wait_until_ready(timeout=8.0)

            if gen != self.scene_search_generation:
                return

            try:
                results = self.scene_matcher.search(
                    query=query,
                    genre=genre,
                    year_min=y_min,
                    year_max=y_max,
                    min_threshold=th,
                    top_k=12,
                    mode=mode,
                    semantic_top_k=500,
                    rerank=wants_rerank and ce_svc.is_ready(),
                    rerank_top_k=config.CROSS_ENCODER_TOP_K
                )
            except Exception:
                results = []

            self.safe_ui_call(self._on_scene_results_ready, results, gen)

        threading.Thread(target=_worker, daemon=True).start()


    def _on_semantic_load_failed_for_search(
        self, gen: int, query: str, mode: str, genre: str,
        y_min: Optional[int], y_max: Optional[int], th: float, reason: str
    ):
        """Called on main UI thread if semantic model loading fails during a search."""
        if gen != self.scene_search_generation:
            return
        self.scene_is_searching = False
        if hasattr(self, "scene_search_btn") and self.scene_search_btn.winfo_exists():
            self.scene_search_btn.config(state="normal", text="Find Movie")
        if hasattr(self, "scene_status_lbl") and self.scene_status_lbl.winfo_exists():
            self.scene_status_lbl.config(
                text=f"⚠️ {reason}",
                fg="#f87171"
            )

        if self.active_tab == "Scene Finder" and hasattr(self, "scene_results_frame") and self.scene_results_frame.winfo_exists():
            for child in self.scene_results_frame.winfo_children():
                child.destroy()

            box = tk.Frame(
                self.scene_results_frame,
                bg=COLOR_PANEL,
                bd=1,
                relief="solid",
                highlightthickness=1,
                highlightbackground="#334155",
                padx=24,
                pady=24
            )
            box.pack(fill="x", padx=20, pady=20)

            tk.Label(
                box,
                text="⚠️ Semantic Search Engine Unavailable",
                font=("Segoe UI", 12, "bold"),
                fg="#f87171",
                bg=COLOR_PANEL
            ).pack(anchor="w", pady=(0, 6))

            tk.Label(
                box,
                text=f"{reason}\n\nYou can search using lexical keyword matching (TF-IDF + BM25), or configure the semantic model.",
                font=("Segoe UI", 9),
                fg=COLOR_TEXT_DIM,
                bg=COLOR_PANEL,
                justify="left",
                wraplength=700
            ).pack(anchor="w", pady=(0, 16))

            btn_row = tk.Frame(box, bg=COLOR_PANEL)
            btn_row.pack(anchor="w")

            tk.Button(
                btn_row,
                text="⚡ Search with Lexical Fallback",
                font=("Segoe UI", 9, "bold"),
                bg="#2563eb",
                fg="#ffffff",
                activebackground="#1d4ed8",
                activeforeground="#ffffff",
                bd=0,
                padx=16,
                pady=6,
                cursor="hand2",
                command=lambda: self._execute_lexical_fallback(query, genre, y_min, y_max, th)
            ).pack(side="left", padx=(0, 12))

            tk.Button(
                btn_row,
                text="⚙️ Manage Semantic Search",
                font=("Segoe UI", 9),
                bg=COLOR_BTN_CARD,
                fg="#e2e8f0",
                activebackground=COLOR_BTN_CARD_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=14,
                pady=6,
                cursor="hand2",
                command=self._open_semantic_dialog
            ).pack(side="left")

    def _execute_lexical_fallback(self, query: str, genre: str, y_min: Optional[int], y_max: Optional[int], th: float):
        """Execute search explicitly in lexical mode without semantic embeddings."""
        self.scene_search_generation += 1
        gen = self.scene_search_generation
        self.scene_is_searching = True
        if hasattr(self, "scene_search_btn") and self.scene_search_btn.winfo_exists():
            self.scene_search_btn.config(state="disabled", text="Searching...")
        if hasattr(self, "scene_status_lbl") and self.scene_status_lbl.winfo_exists():
            self.scene_status_lbl.config(text="Searching with Lexical Fallback (TF-IDF + BM25)...", fg="#38bdf8")

        def _worker():
            if gen != self.scene_search_generation:
                return
            try:
                results = self.scene_matcher.search(
                    query=query,
                    genre=genre,
                    year_min=y_min,
                    year_max=y_max,
                    min_threshold=th,
                    top_k=12,
                    mode="plot",
                    force_lexical=True
                )
            except Exception:
                results = []

            self.safe_ui_call(self._on_scene_results_ready, results, gen)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_scene_clear_clicked(self):
        self.scene_search_generation += 1
        self.scene_query_text = ""
        self.scene_selected_genre.set("All")
        self.scene_search_mode_var.set("Hybrid (All Signals)")
        self.scene_min_threshold_var.set("10% (Balanced)")
        self.scene_min_year_var.set("")
        self.scene_max_year_var.set("")
        self.scene_rerank_var.set(config.CROSS_ENCODER_ENABLED)
        self.scene_results = []
        self.scene_card_widgets.clear()
        self.scene_has_searched = False
        self.scene_is_searching = False
        if hasattr(self, "scene_text_box") and self.scene_text_box.winfo_exists():
            self.scene_text_box.delete("1.0", "end")
        if hasattr(self, "scene_status_lbl") and self.scene_status_lbl.winfo_exists():
            self.scene_status_lbl.config(text="")
        if hasattr(self, "scene_search_btn") and self.scene_search_btn.winfo_exists():
            self.scene_search_btn.config(state="normal", text="Find Movie")
        if self.active_tab == "Scene Finder":
            self._render_scene_finder_screen()

    def _on_scene_results_ready(self, results: List[Dict[str, Any]], gen: int):
        if gen != self.scene_search_generation:
            return
        self.scene_is_searching = False
        self.scene_has_searched = True
        self.scene_results = results

        if self.winfo_exists() and hasattr(self, "scene_search_btn"):
            self.scene_search_btn.config(state="normal", text="Find Movie")
            count = len(results)
            is_reranked = count > 0 and results[0].get("match_type") == "reranked"
            status_text = f"Found {count} candidate{'s' if count != 1 else ''}"
            if is_reranked:
                status_text += " (Cross-Encoder Reranked)"
            self.scene_status_lbl.config(
                text=status_text if count > 0 else "No matches found",
                fg="#c084fc" if is_reranked else ("#34d399" if count > 0 else "#f87171")
            )


        if self.active_tab == "Scene Finder":
            if hasattr(self, "scene_results_frame") and self.scene_results_frame.winfo_exists():
                for child in self.scene_results_frame.winfo_children():
                    child.destroy()
                if not results:
                    th_str = self.scene_min_threshold_var.get()
                    mode_str = self.scene_search_mode_var.get()
                    self._render_scene_empty_state(
                        self.scene_results_frame,
                        "No Confident Match Found",
                        f"No movies in our catalogue met the {th_str} match threshold under '{mode_str}' mode.\n\n"
                        "Suggestions:\n"
                        "• Lower the Min Match threshold (e.g. to '8% (Broad)' or '5% (All Potential)').\n"
                        "• Switch to 'Hybrid (All Signals)' to search across titles, cast, and plot simultaneously.\n"
                        "• Include distinctive character names, actor names, or key plot moments.\n"
                        "• Click '📥 Import Dataset' to expand your catalogue coverage."
                    )
                else:
                    self._render_scene_results_grid(self.scene_results_frame, results)
            else:
                self._render_scene_finder_screen(preserve_results=True)

        if results:
            self._enrich_scene_results_async(results, gen)

    def _enrich_scene_results_async(self, results: List[Dict[str, Any]], gen: int):
        def _worker():
            for movie in results:
                if gen != self.scene_search_generation:
                    return
                enriched = scene_matcher.enrich_candidate(movie, self.client)
                movie.update(enriched)
                self.safe_ui_call(self._on_scene_movie_enriched, movie, gen)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_scene_movie_enriched(self, movie: Dict[str, Any], gen: int):
        """Thread-safe UI callback to update card metadata and poster when enrichment completes."""
        if gen != self.scene_search_generation:
            return

        imdb_id = movie.get("imdb_id", "").strip()
        title = movie.get("title", "").strip()

        # Find widget tracking record
        entry = None
        if imdb_id and imdb_id in self.scene_card_widgets:
            entry = self.scene_card_widgets[imdb_id]
        elif title and title in self.scene_card_widgets:
            entry = self.scene_card_widgets[title]

        if not entry:
            return

        card = entry.get("card")
        if not card or not card.winfo_exists():
            return

        # 1. Update rating and year label
        meta_lbl = entry.get("meta_lbl")
        if meta_lbl and meta_lbl.winfo_exists():
            year = movie.get("year", "N/A")
            rating = movie.get("imdb_rating") or "N/A"
            rating_str = f"★ {rating}" if rating != "N/A" else ""
            meta_text = f"{year}   •   {rating_str}".strip(" •")
            meta_lbl.config(text=meta_text)

        # 1b. Update cast preview if enriched
        cast_val = movie.get("actors") or movie.get("cast")
        cast_lbl = entry.get("cast_lbl")
        if cast_lbl and cast_lbl.winfo_exists() and cast_val and cast_val != "N/A":
            cast_s = str(cast_val).strip()
            if len(cast_s) > 42:
                cast_s = cast_s[:40] + "..."
            cast_lbl.config(text=f"Cast: {cast_s}")
            if not cast_lbl.winfo_ismapped() and meta_lbl and meta_lbl.winfo_exists():
                cast_lbl.pack(fill="x", anchor="w", pady=(0, 3), after=meta_lbl)

        # 2. Update poster if not loaded yet
        poster_url = movie.get("poster_url") or movie.get("poster")
        if poster_url and poster_url != "N/A" and not entry.get("poster_loaded", False):
            entry["poster_loaded"] = True
            poster_lbl = entry.get("poster_lbl")
            if poster_lbl and poster_lbl.winfo_exists():
                load_poster_async(
                    poster_url, scale_px(140), scale_px(195),
                    lambda photo: self._safe_apply_poster_callback("scene_search_generation", gen, card, poster_lbl, imdb_id, photo),
                    title=title
                )

        # 3. Update Wishlist, Backlog, and Details buttons if IMDb ID was newly resolved
        if imdb_id:
            self.scene_card_widgets[imdb_id] = entry
            in_wish = database.is_in_wishlist(imdb_id)
            in_back = database.is_in_backlog(imdb_id)
            wish_btn = entry.get("wish_btn")
            if wish_btn and wish_btn.winfo_exists():
                wish_btn.config(
                    text="♥ In Wishlist" if in_wish else "+ Wishlist",
                    font=("Segoe UI", 8, "bold" if in_wish else "normal"),
                    bg="#e11d48" if in_wish else COLOR_BTN_CARD,
                    fg="#ffffff" if in_wish else "#cbd5e1"
                )
            back_btn = entry.get("back_btn")
            if back_btn and back_btn.winfo_exists():
                back_btn.config(
                    text="▤ In Backlog" if in_back else "+ Backlog",
                    font=("Segoe UI", 8, "bold" if in_back else "normal"),
                    bg="#0284c7" if in_back else COLOR_BTN_CARD,
                    fg="#ffffff" if in_back else "#cbd5e1"
                )

    def _render_scene_initial_state(self, parent: tk.Widget):
        box = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=28,
            pady=28,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        box.pack(fill="x", pady=16)
        self._bind_mousewheel(box)

        tk.Label(
            box,
            text="🎬 Ready to Find Your Movie",
            font=("Segoe UI", 13, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 8))

        instructions = (
            "Type any scene or plot details you remember into the box above and click 'Find Movie'.\n\n"
            "Tips for better results:\n"
            "• Describe key plot twists, character occupations, or central premises (e.g. 'time loop', 'dreams inside dreams').\n"
            "• Mention distinctive settings or moments (e.g. 'desert island with a volleyball', 'auditions in Los Angeles').\n"
            "• Use the Genre and Year Range filters to narrow down results.\n\n"
            "CineTrace ranks candidates using text similarity against our local movie plot dataset, then enriches matches with real OMDb ratings and posters."
        )
        tk.Label(
            box,
            text=instructions,
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            justify="left"
        ).pack(anchor="w")

    def _render_scene_empty_state(self, parent: tk.Widget, title: str, message: str):
        box = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=24,
            pady=24,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        box.pack(fill="x", pady=16)
        self._bind_mousewheel(box)

        tk.Label(
            box,
            text=title,
            font=("Segoe UI", 12, "bold"),
            fg="#f87171" if "Error" in title or "Not Available" in title or "No Matching" in title else "#38bdf8",
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            box,
            text=message,
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            justify="left",
            wraplength=700
        ).pack(anchor="w")

    def _render_scene_results_grid(self, parent: tk.Widget, movies: List[Dict[str, Any]]):
        self.scene_card_widgets.clear()
        head = tk.Frame(parent, bg=COLOR_BG)
        head.pack(fill="x", pady=(0, 10))
        self._bind_mousewheel(head)

        count = len(movies)
        tk.Label(
            head,
            text=f"Ranked Candidates ({count} movie{'s' if count != 1 else ''})",
            font=("Segoe UI", 12, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(side="left")

        is_reranked = any(m.get("match_type") == "reranked" for m in movies)
        header_desc = "Ranked by text similarity score (Cross-Encoder reranked)" if is_reranked else "Ranked by text similarity score"
        tk.Label(
            head,
            text=header_desc,
            font=("Segoe UI", 9),
            fg="#c084fc" if is_reranked else COLOR_TEXT_DIM,
            bg=COLOR_BG
        ).pack(side="left", padx=10)



        grid_frame = tk.Frame(parent, bg=COLOR_BG)
        grid_frame.pack(fill="x", pady=6)
        self._bind_mousewheel(grid_frame)

        gen = self.scene_search_generation
        wish_set = database.get_wishlist_ids()
        back_set = database.get_backlog_ids(active_only=True)

        initial_count = min(12, count)
        for idx in range(initial_count):
            row = idx // 4
            col = idx % 4
            self._render_scene_card(grid_frame, movies[idx], row, col, gen, wish_set=wish_set, back_set=back_set)

        if count > initial_count:
            btn_frame = tk.Frame(parent, bg=COLOR_BG)
            btn_frame.pack(fill="x", pady=(10, 20))
            self._bind_mousewheel(btn_frame)

            state = {"rendered": initial_count}

            def _on_load_more():
                cur = state["rendered"]
                nxt = min(cur + 12, count)
                for idx in range(cur, nxt):
                    row = idx // 4
                    col = idx % 4
                    self._render_scene_card(grid_frame, movies[idx], row, col, gen, wish_set=wish_set, back_set=back_set)
                state["rendered"] = nxt
                if nxt >= count:
                    btn_frame.destroy()
                else:
                    load_more_btn.config(text=f"⬇ Load More Candidates (Showing {nxt} of {count})")

            load_more_btn = tk.Button(
                btn_frame,
                text=f"⬇ Load More Candidates (Showing {initial_count} of {count})",
                command=_on_load_more,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_BTN_CARD,
                fg="#38bdf8",
                activebackground=COLOR_BTN_CARD_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=16,
                pady=8,
                cursor="hand2"
            )
            load_more_btn.pack(anchor="center")
            self._bind_mousewheel(load_more_btn)

    def _render_scene_card(
        self,
        parent: tk.Widget,
        movie: Dict[str, Any],
        row: int,
        col: int,
        gen: int,
        wish_set: Optional[Set[str]] = None,
        back_set: Optional[Set[str]] = None
    ):
        title = movie.get("title", "Untitled")
        year = movie.get("year", "N/A")
        rating = movie.get("imdb_rating") or "N/A"
        imdb_id = movie.get("imdb_id", "")
        poster_url = movie.get("poster_url") or movie.get("poster") or movie.get("Poster")
        score_label = movie.get("score_label") or f"Text Match: {movie.get('similarity_pct', 'N/A')}"
        plot = movie.get("plot", "")
        genre = movie.get("genre", "N/A")

        card = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            width=scale_px(230),
            height=scale_px(475)
        )
        card.grid(row=row, column=col, padx=8, pady=8, sticky="n")
        card.grid_propagate(False)
        self._bind_mousewheel(card)

        # 1. Poster Container (140x195 scaled)
        p_w = scale_px(140)
        p_h = scale_px(195)
        poster_frame = tk.Frame(card, bg=COLOR_PANEL, width=p_w, height=p_h)
        poster_frame.pack(side="top", pady=(0, 6))
        poster_frame.pack_propagate(False)

        placeholder = get_placeholder_photo(p_w, p_h, title)
        poster_lbl = tk.Label(poster_frame, image=placeholder, bg=COLOR_PANEL, bd=0)
        poster_lbl.image = placeholder
        poster_lbl._movie_title = title
        poster_lbl.pack(fill="both", expand=True)
        self._bind_mousewheel(poster_lbl)

        poster_loaded = False
        if poster_url and poster_url != "N/A":
            poster_loaded = True
            load_poster_async(
                poster_url, p_w, p_h,
                lambda photo: self._safe_apply_poster_callback("scene_search_generation", gen, card, poster_lbl, imdb_id, photo),
                title=title
            )

        # 2. Similarity Score Badge (cyan, emerald, or violet pill)
        match_type = movie.get("match_type", "plot")
        match_details = movie.get("match_details", "")
        if match_type == "reranked":
            pill_fg = "#c084fc"
            pill_bg = "#2e1065"
            badge_text = f"🎯 {score_label}"
        elif match_type == "semantic":
            pill_fg = "#34d399"
            pill_bg = "#064e3b"
            badge_text = f"★ {score_label}"
        else:
            pill_fg = "#38bdf8"
            pill_bg = "#0f2942"
            badge_text = f"★ {score_label}"

        pill = tk.Label(
            card,
            text=badge_text,
            font=("Segoe UI", 8, "bold"),
            fg=pill_fg,
            bg=pill_bg,
            padx=6,
            pady=2
        )
        pill.pack(fill="x", pady=(0, 4))
        self._bind_mousewheel(pill)


        # 3. Title
        title_lbl = tk.Label(
            card,
            text=title,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        title_lbl.pack(fill="x", anchor="w")
        self._bind_mousewheel(title_lbl)

        # 4. Meta: Year • Rating
        rating_str = f"★ {rating}" if rating != "N/A" else ""
        meta_text = f"{year}   •   {rating_str}".strip(" •")
        meta_lbl = tk.Label(
            card,
            text=meta_text,
            font=("Segoe UI", 8, "bold"),
            fg="#f43f5e",
            bg=COLOR_PANEL,
            anchor="w"
        )
        meta_lbl.pack(fill="x", anchor="w", pady=(1, 2))
        self._bind_mousewheel(meta_lbl)

        # 4a. Match explanation tag
        if match_details:
            explain_lbl = tk.Label(
                card,
                text=f"• {match_details}",
                font=("Segoe UI", 8),
                fg="#34d399" if match_type == "semantic" else "#38bdf8",
                bg=COLOR_PANEL,
                anchor="w",
                wraplength=scale_px(200),
                justify="left"
            )
            explain_lbl.pack(fill="x", anchor="w", pady=(0, 2))
            self._bind_mousewheel(explain_lbl)

        # 4b. Cast preview line
        actors = movie.get("actors") or movie.get("cast") or ""
        cast_text = ""
        if actors and actors != "N/A":
            cast_s = str(actors).strip()
            if len(cast_s) > 42:
                cast_s = cast_s[:40] + "..."
            cast_text = f"Cast: {cast_s}"

        cast_lbl = tk.Label(
            card,
            text=cast_text,
            font=("Segoe UI", 8, "italic"),
            fg="#94a3b8",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        if cast_text:
            cast_lbl.pack(fill="x", anchor="w", pady=(0, 3))
        self._bind_mousewheel(cast_lbl)

        # 5. Plot preview snippet (truncated)
        snippet = (plot[:85] + "...") if len(plot) > 85 else plot
        plot_lbl = tk.Label(
            card,
            text=snippet,
            font=("Segoe UI", 8),
            fg="#8e9db3",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        plot_lbl.pack(fill="x", anchor="w", pady=(0, 6))
        self._bind_mousewheel(plot_lbl)

        # 6. Action buttons at bottom:
        # Row 1: + Wishlist and + Backlog
        list_row = tk.Frame(card, bg=COLOR_PANEL)
        list_row.pack(side="bottom", fill="x", pady=(4, 0))
        self._bind_mousewheel(list_row)

        if wish_set is not None:
            in_wish = (imdb_id in wish_set) if imdb_id else False
        else:
            in_wish = database.is_in_wishlist(imdb_id) if imdb_id else False

        if back_set is not None:
            in_back = (imdb_id in back_set) if imdb_id else False
        else:
            in_back = database.is_in_backlog(imdb_id) if imdb_id else False

        wish_btn = tk.Button(
            list_row,
            text="♥ In Wishlist" if in_wish else "+ Wishlist",
            font=("Segoe UI", 8, "bold" if in_wish else "normal"),
            bg="#e11d48" if in_wish else COLOR_BTN_CARD,
            fg="#ffffff" if in_wish else "#cbd5e1",
            activebackground="#be123c",
            activeforeground="#ffffff",
            bd=0,
            padx=4,
            pady=3,
            cursor="hand2"
        )
        wish_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        back_btn = tk.Button(
            list_row,
            text="▤ In Backlog" if in_back else "+ Backlog",
            font=("Segoe UI", 8, "bold" if in_back else "normal"),
            bg="#0284c7" if in_back else COLOR_BTN_CARD,
            fg="#ffffff" if in_back else "#cbd5e1",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            bd=0,
            padx=4,
            pady=3,
            cursor="hand2"
        )
        back_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))

        def _on_wish_click(m=movie, btn=wish_btn):
            cur_id = m.get("imdb_id") or imdb_id
            if not cur_id:
                return
            if database.is_in_wishlist(cur_id):
                database.remove_from_wishlist(cur_id)
                if wish_set is not None:
                    wish_set.discard(cur_id)
                btn.config(text="+ Wishlist", bg=COLOR_BTN_CARD, fg="#cbd5e1", font=("Segoe UI", 8))
            else:
                database.add_to_wishlist(cur_id, m)
                if wish_set is not None:
                    wish_set.add(cur_id)
                btn.config(text="♥ In Wishlist", bg="#e11d48", fg="#ffffff", font=("Segoe UI", 8, "bold"))

        def _on_back_click(m=movie, btn=back_btn):
            cur_id = m.get("imdb_id") or imdb_id
            if not cur_id:
                return
            if database.is_in_backlog(cur_id):
                database.remove_from_backlog(cur_id)
                if back_set is not None:
                    back_set.discard(cur_id)
                btn.config(text="+ Backlog", bg=COLOR_BTN_CARD, fg="#cbd5e1", font=("Segoe UI", 8))
            else:
                database.add_to_backlog(cur_id, "Medium", "Want to Watch", m)
                if back_set is not None:
                    back_set.add(cur_id)
                btn.config(text="▤ In Backlog", bg="#0284c7", fg="#ffffff", font=("Segoe UI", 8, "bold"))

        wish_btn.config(command=_on_wish_click)
        back_btn.config(command=_on_back_click)

        # Row 2: View Details button
        details_btn = tk.Button(
            card,
            text="View Details",
            command=lambda m=movie, i=imdb_id, t=title: self._open_movie_details(m.get("imdb_id") or i, t),
            font=("Segoe UI", 9),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        details_btn.pack(side="bottom", fill="x", pady=(0, 4))

        # Register widget references for async enrichment updates
        entry = {
            "card": card,
            "poster_lbl": poster_lbl,
            "meta_lbl": meta_lbl,
            "cast_lbl": cast_lbl,
            "wish_btn": wish_btn,
            "back_btn": back_btn,
            "details_btn": details_btn,
            "title": title,
            "imdb_id": imdb_id,
            "poster_loaded": poster_loaded,
        }
        card_key = imdb_id if imdb_id else title
        self.scene_card_widgets[card_key] = entry
        if title:
            self.scene_card_widgets[title] = entry
        if imdb_id:
            self.scene_card_widgets[imdb_id] = entry

    # -------------------------------------------------------------
    # Responsive Movie Grid & Card Renderer
    # -------------------------------------------------------------
    def _get_responsive_columns(self) -> int:
        """Calculate optimal column count based on available viewport width to prevent cutoffs."""
        try:
            w = self.canvas.winfo_width()
            if w > 100:
                avail = max(scale_px(400), w - scale_px(80))
                return max(1, min(4, avail // scale_px(246)))
        except Exception:
            pass
        return 4

    def _render_movie_grid(self, parent: tk.Widget, movies: List[Dict[str, Any]]):
        """Renders movies into a clean responsive grid matching the reference."""
        grid_frame = tk.Frame(parent, bg=COLOR_BG)
        grid_frame.pack(fill="x", pady=6)
        self._bind_mousewheel(grid_frame)

        gen = self.search_generation
        cols = self._get_responsive_columns()

        for c in range(cols):
            grid_frame.columnconfigure(c, weight=1)

        for idx, movie in enumerate(movies):
            row = idx // cols
            col = idx % cols
            self._render_movie_card(grid_frame, movie, row, col, gen)

    def _render_movie_card(self, parent: tk.Widget, movie: Dict[str, Any], row: int, col: int, gen: int):
        """Renders an individual card matching the reference design."""
        title = movie.get("title", "Untitled")
        year = movie.get("year", "N/A")
        rating = movie.get("imdb_rating") or movie.get("rating") or "N/A"
        imdb_id = movie.get("imdb_id", "")
        poster_url = movie.get("poster_url") or movie.get("poster") or movie.get("Poster")

        card = tk.Frame(
            parent,
            bg=COLOR_PANEL,
            padx=12,
            pady=12,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            width=scale_px(230),
            height=scale_px(330)
        )
        card.grid(row=row, column=col, padx=8, pady=8, sticky="n")
        card.grid_propagate(False)
        self._bind_mousewheel(card)

        # 1. Poster Container (140x195 scaled)
        p_w = scale_px(140)
        p_h = scale_px(195)
        poster_frame = tk.Frame(card, bg=COLOR_PANEL, width=p_w, height=p_h)
        poster_frame.pack(side="top", pady=(0, 8))
        poster_frame.pack_propagate(False)

        # Initial placeholder displaying movie title
        placeholder = get_placeholder_photo(p_w, p_h, title)
        poster_lbl = tk.Label(poster_frame, image=placeholder, bg=COLOR_PANEL, bd=0)
        poster_lbl.image = placeholder
        poster_lbl._movie_title = title
        poster_lbl.pack(fill="both", expand=True)
        self._bind_mousewheel(poster_lbl)

        # Load real poster asynchronously if available
        if poster_url and poster_url != "N/A":
            load_poster_async(
                poster_url, p_w, p_h,
                lambda photo: self._safe_apply_poster_callback("search_generation", gen, card, poster_lbl, imdb_id, photo),
                title=title
            )

        # 2. Movie Title with length safeguard to avoid clipping buttons
        display_title = title if len(title) <= 42 else title[:39] + "..."
        title_lbl = tk.Label(
            card,
            text=display_title,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL,
            anchor="w",
            wraplength=scale_px(200),
            justify="left"
        )
        title_lbl.pack(fill="x", anchor="w")
        self._bind_mousewheel(title_lbl)

        # 3. Meta: Year • ★ Rating (rose/pink)
        rating_str = f"★ {rating}" if rating != "N/A" else ""
        meta_text = f"{year}   •   {rating_str}".strip(" •")
        meta_lbl = tk.Label(
            card,
            text=meta_text,
            font=("Segoe UI", 8, "bold"),
            fg="#f43f5e",
            bg=COLOR_PANEL,
            anchor="w"
        )
        meta_lbl.pack(fill="x", anchor="w", pady=(2, 8))
        self._bind_mousewheel(meta_lbl)

        # 4. View Details Button (Full Width, Dark Slate)
        btn = tk.Button(
            card,
            text="View Details",
            command=lambda i=imdb_id, t=title: self._open_movie_details(i, t),
            font=("Segoe UI", 9),
            bg=COLOR_BTN_CARD,
            fg="#cbd5e1",
            activebackground=COLOR_BTN_CARD_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn.pack(side="bottom", fill="x")

    def _apply_card_poster(self, label: tk.Label, imdb_id: str, photo: Any):
        try:
            if label.winfo_exists():
                label.config(image=photo)
                label.image = photo
                self.active_posters[imdb_id] = photo
                movie_title = getattr(label, "_movie_title", imdb_id)
                logger.info("[CineTrace Poster] Successful Tkinter widget assignment for '%s' (id=%s)", movie_title, imdb_id)
        except Exception as e:
            logger.warning("[CineTrace Poster] Failed widget assignment for %s: %s", imdb_id, e)

    # -------------------------------------------------------------
    # Search Handler (OMDb 's' Parameter)
    # -------------------------------------------------------------
    def _on_search_submitted(self):
        query = self.search_var.get().strip()
        if not query:
            if self.current_search_query:
                self._clear_search()
            return

        if not config.is_api_key_configured():
            self._open_api_key_dialog()
            return

        # Visual feedback and duplicate submission guard
        self.search_btn.config(state="disabled", text="Searching...")

        self.current_search_query = query
        self.search_generation += 1
        gen = self.search_generation

        # Switch to Discover tab to show search results
        self.active_tab = "Discover"
        for name, btn in self.nav_buttons.items():
            btn.set_active(name == "Discover")
        self.screen_title_label.config(text=f"Search Results: '{query}'")

        self._clear_scroll_content()
        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=20)
        container.pack(fill="both", expand=True)

        self._render_loading_state(container, f"Searching OMDb for '{query}'...")

        def _worker():
            try:
                res = self.client.search_movies(query=query, max_pages=config.OMDB_SEARCH_MAX_PAGES)
                movies = res.get("movies", [])
                self.safe_ui_call(self._on_search_success, gen, movies)
            except OMDbMovieNotFoundError:
                self.safe_ui_call(self._on_search_empty, gen)
            except OMDbError as e:
                self.safe_ui_call(self._on_search_failed, gen, str(e))
            except Exception as e:
                self.safe_ui_call(self._on_search_failed, gen, f"Unexpected error: {str(e)}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_search_success(self, gen: int, movies: List[Dict[str, Any]]):
        self.search_btn.config(state="normal", text="Search")
        if gen != self.search_generation:
            return
        self.discover_movies = movies
        if not movies:
            self._on_search_empty(gen)
        else:
            self._render_discover_screen()

    def _on_search_empty(self, gen: int):
        self.search_btn.config(state="normal", text="Search")
        if gen != self.search_generation:
            return
        self.discover_movies = []
        self._clear_scroll_content()
        container = tk.Frame(self.scroll_content, bg=COLOR_BG, padx=32, pady=40)
        container.pack(fill="both", expand=True)
        self._render_empty_search_state(
            container,
            message=f"No movies found matching '{self.current_search_query}'. Try a broader title or franchise keyword.",
            show_reset=True
        )

    def _on_search_failed(self, gen: int, error_message: str):
        self.search_btn.config(state="normal", text="Search")
        if gen != self.search_generation:
            return
        self._clear_scroll_content()
        self._render_notice_box(
            title="Search Failed",
            message=error_message,
            show_key_btn="API key" in error_message
        )

    # -------------------------------------------------------------
    # State Renderers: Empty, Loading, Notice
    # -------------------------------------------------------------
    def _render_empty_search_state(
        self,
        parent: tk.Widget,
        message: str = "No movies found. Try another search.",
        show_reset: bool = False
    ):
        """Render clean, empty state with configurable message and optional reset action."""
        box = tk.Frame(parent, bg=COLOR_BG)
        box.pack(fill="both", expand=True, pady=100)
        self._bind_mousewheel(box)

        lbl = tk.Label(
            box,
            text=message,
            font=("Segoe UI", 11),
            fg="#7e8b9f",
            bg=COLOR_BG
        )
        lbl.pack(anchor="center")
        self._bind_mousewheel(lbl)

        if show_reset:
            reset_btn = tk.Button(
                box,
                text="Show Popular Picks",
                command=self._clear_search,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_ACCENT,
                fg="#ffffff",
                activebackground=COLOR_ACCENT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=16,
                pady=6,
                cursor="hand2"
            )
            reset_btn.pack(anchor="center", pady=14)
            self._bind_mousewheel(reset_btn)

    def _render_loading_state(self, parent: tk.Widget, message: str):
        box = tk.Frame(parent, bg=COLOR_BG)
        box.pack(fill="both", expand=True, pady=100)

        tk.Label(
            box,
            text=message,
            font=("Segoe UI", 11, "italic"),
            fg="#7c8ba1",
            bg=COLOR_BG
        ).pack(anchor="center")

    def _render_notice_box(self, title: str, message: str, show_key_btn: bool = False):
        box = tk.Frame(
            self.scroll_content,
            bg=COLOR_PANEL,
            padx=32,
            pady=36,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        box.pack(fill="x", padx=40, pady=60)

        tk.Label(
            box,
            text=title,
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg=COLOR_PANEL
        ).pack(pady=(0, 6))

        tk.Label(
            box,
            text=message,
            font=("Segoe UI", 10),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            wraplength=620,
            justify="center"
        ).pack(pady=(0, 16))

        if show_key_btn:
            tk.Button(
                box,
                text="Configure OMDb API Key",
                command=self._open_api_key_dialog,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_ACCENT,
                fg="#ffffff",
                activebackground=COLOR_ACCENT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=16,
                pady=7,
                cursor="hand2"
            ).pack()

    # -------------------------------------------------------------
    # Dialogs
    # -------------------------------------------------------------
    def _open_movie_details(self, imdb_id: str, title: str):
        if not imdb_id or imdb_id.upper() == "N/A":
            messagebox.showwarning(
                "Missing ID",
                "No valid IMDb ID available to retrieve full details.",
                parent=self
            )
            return

        def _on_details_change():
            if self.active_tab == "Wishlist":
                self._render_wishlist_screen()
            elif self.active_tab == "Backlog":
                self._render_backlog_screen()
            elif self.active_tab == "Scene Finder":
                self._render_scene_finder_screen(preserve_results=True)

        MovieDetailsDialog(
            self,
            imdb_id=imdb_id,
            initial_title=title,
            on_change_callback=_on_details_change
        )

    def _open_api_key_dialog(self):
        def _on_save():
            self.key_status_btn.config(text="🔑 API: " + config.get_masked_api_key())
            # Refresh dynamic home data with newly configured key
            self._load_dynamic_home_data(force_refresh=True)

        ApiKeyDialog(self, on_save_callback=_on_save)

    def _open_preferences_dialog(self, is_onboarding: bool = False):
        """Open language preferences or first-launch onboarding modal."""
        LanguagePreferencesDialog(
            self,
            is_onboarding=is_onboarding,
            on_save_callback=self._on_preferences_saved
        )

    def _show_first_launch_onboarding(self):
        """Show onboarding dialog on first launch if not yet completed or skipped."""
        if not database.is_onboarding_completed():
            self._open_preferences_dialog(is_onboarding=True)

    def _on_preferences_saved(self, selected_languages: List[str]):
        """Callback when preferences are updated, saved, or reset."""
        self.preferred_languages = selected_languages
        if self.raw_curated_picks or self.raw_home_featured:
            feat, picks = language_service.personalize_recommendations(
                self.raw_home_featured, self.raw_curated_picks, self.preferred_languages
            )
            self.home_featured = feat
            self.home_picks = picks
            self.home_trending = picks
            self.curated_picks = picks
            if not self.current_search_query:
                self.discover_movies = picks

        self._update_preference_ui_indicators()

        if self.active_tab == "Home":
            self._render_home_screen()
        elif self.active_tab == "Discover":
            self._render_discover_screen()

    def _get_language_button_label(self) -> str:
        """Sidebar button label reflecting active language preferences."""
        if not self.preferred_languages:
            return "🌐 Languages: All"
        if len(self.preferred_languages) == 1:
            return f"🌐 Lang: {self.preferred_languages[0]}"
        return f"🌐 Langs: {len(self.preferred_languages)} selected"

    def _get_top_bar_lang_label(self) -> str:
        """Top bar pill label reflecting active language preferences."""
        if not self.preferred_languages:
            return "🌐 All Languages"
        if len(self.preferred_languages) <= 2:
            return f"🌐 {', '.join(self.preferred_languages)}"
        return f"🌐 {len(self.preferred_languages)} Languages"

    def _update_preference_ui_indicators(self):
        """Refresh preference indicators in sidebar and top bar."""
        if hasattr(self, "prefs_status_btn") and self.prefs_status_btn.winfo_exists():
            self.prefs_status_btn.config(text=self._get_language_button_label())
        if hasattr(self, "lang_filter_btn") and self.lang_filter_btn.winfo_exists():
            self.lang_filter_btn.config(text=self._get_top_bar_lang_label())


def main():
    enable_high_dpi_awareness()
    app = CineTraceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
