"""
CineTrace - Movie Details Dialog
Displays comprehensive movie metadata retrieved using OMDb or cached locally in SQLite.
Integrates Wishlist and Backlog management with state persistence.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, Optional

from PIL import ImageTk
from omdb_client import OMDbClient, OMDbError
from image_loader import load_poster_sync, get_placeholder_photo, fetch_and_fit_pil
from dpi_helper import scale_px
import database


class MovieDetailsDialog(tk.Toplevel):
    """Modal dialog displaying detailed movie information with SQLite Wishlist & Backlog actions."""

    def __init__(
        self,
        parent: tk.Widget,
        imdb_id: str,
        initial_title: str = "Movie Details",
        on_change_callback: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent)
        self.imdb_id = (imdb_id or "").strip()
        self.client = OMDbClient()
        self.on_change_callback = on_change_callback
        self.poster_photo: Optional[Any] = None
        self.movie_data: Optional[Dict[str, Any]] = None

        dlg_w = scale_px(820)
        dlg_h = scale_px(660)
        self.title(f"CineTrace — {initial_title}")
        self.geometry(f"{dlg_w}x{dlg_h}")
        self.minsize(scale_px(700), scale_px(520))
        self.configure(bg="#0a0e17")  # Dark cinematic background

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Center on parent window
        self.update_idletasks()
        try:
            p_x = parent.winfo_rootx()
            p_y = parent.winfo_rooty()
            p_w = parent.winfo_width()
            p_h = parent.winfo_height()
            x = p_x + max(0, (p_w - dlg_w) // 2)
            y = p_y + max(0, (p_h - dlg_h) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

        self._build_ui()
        self._load_movie_workflow()

    def safe_ui_call(self, callback, *args):
        """Thread-safe UI dispatcher that routes through parent queue or falls back to after."""
        try:
            if hasattr(self.master, "safe_ui_call"):
                self.master.safe_ui_call(callback, *args)
            elif self.winfo_exists():
                self.after(0, callback, *args)
        except Exception:
            pass

    def _build_ui(self):
        # Header bar
        self.header_frame = tk.Frame(self, bg="#111726", padx=24, pady=16)
        self.header_frame.pack(fill="x", side="top")

        self.title_label = tk.Label(
            self.header_frame,
            text="Loading movie details...",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg="#111726",
            wraplength=scale_px(640),
            justify="left"
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = tk.Label(
            self.header_frame,
            text="",
            font=("Segoe UI", 10),
            fg="#8e9db3",
            bg="#111726"
        )
        self.subtitle_label.pack(anchor="w", pady=(4, 0))

        # Main content area
        self.content_container = tk.Frame(self, bg="#0a0e17")
        self.content_container.pack(fill="both", expand=True, padx=24, pady=18)

        # Left column: Poster
        self.poster_frame = tk.Frame(self.content_container, bg="#0a0e17", width=scale_px(220))
        self.poster_frame.pack(side="left", fill="y", padx=(0, 24), anchor="n")

        # Initial placeholder
        self.poster_photo = get_placeholder_photo(scale_px(200), scale_px(290), "LOADING...")
        self.poster_label = tk.Label(
            self.poster_frame,
            image=self.poster_photo,
            bg="#0a0e17",
            bd=0
        )
        self.poster_label.pack(anchor="n")

        # Right column: Details and Plot
        self.details_frame = tk.Frame(self.content_container, bg="#0a0e17")
        self.details_frame.pack(side="left", fill="both", expand=True)

        self.status_label = tk.Label(
            self.details_frame,
            text="Checking local SQLite cache...",
            font=("Segoe UI", 11, "italic"),
            fg="#7c8ba1",
            bg="#0a0e17"
        )
        self.status_label.pack(anchor="w", pady=20)

        # Bottom action bar
        self.bottom_frame = tk.Frame(self, bg="#111726", padx=24, pady=12)
        self.bottom_frame.pack(fill="x", side="bottom")

        # Wishlist button
        self.wishlist_btn = tk.Button(
            self.bottom_frame,
            text="♡ Wishlist",
            command=self._toggle_wishlist,
            font=("Segoe UI", 9, "bold"),
            bg="#1e2638",
            fg="#cbd5e1",
            activebackground="#2a364e",
            activeforeground="#ffffff",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.wishlist_btn.pack(side="left", padx=(0, 8))

        # Backlog button
        self.backlog_btn = tk.Button(
            self.bottom_frame,
            text="▤ Backlog",
            command=self._toggle_backlog,
            font=("Segoe UI", 9, "bold"),
            bg="#1e2638",
            fg="#cbd5e1",
            activebackground="#2a364e",
            activeforeground="#ffffff",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.backlog_btn.pack(side="left", padx=(0, 8))

        # Refresh from OMDb button
        self.refresh_btn = tk.Button(
            self.bottom_frame,
            text="⟳ Refresh",
            command=self._on_refresh_clicked,
            font=("Segoe UI", 9),
            bg="#1e2638",
            fg="#cbd5e1",
            activebackground="#2a364e",
            activeforeground="#ffffff",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2"
        )
        self.refresh_btn.pack(side="left")

        # SQLite status badge
        self.db_status_label = tk.Label(
            self.bottom_frame,
            text="",
            font=("Segoe UI", 9),
            fg="#5a687d",
            bg="#111726"
        )
        self.db_status_label.pack(side="left", padx=12)

        # Close button
        close_btn = tk.Button(
            self.bottom_frame,
            text="Close",
            command=self.destroy,
            font=("Segoe UI", 10, "bold"),
            bg="#e11d48",
            fg="#ffffff",
            activebackground="#be123c",
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=6,
            cursor="hand2"
        )
        close_btn.pack(side="right")

    def _load_movie_workflow(self):
        """Check local SQLite cache first for instant retrieval; fallback to OMDb."""
        cached_movie = database.get_movie_by_id(self.imdb_id)
        if cached_movie:
            self._render_details(cached_movie, source="SQLite Cache")
        else:
            self._fetch_from_omdb_async()

    def _fetch_from_omdb_async(self):
        """Fetch movie details asynchronously from OMDb API and persist to SQLite."""
        self.status_label.config(
            text="Fetching complete movie details from OMDb...",
            fg="#7c8ba1"
        )
        self.refresh_btn.config(state="disabled", text="Refreshing...")

        def _worker():
            try:
                data = self.client.get_movie_details(self.imdb_id)
                saved = database.save_movie(data)
                self.safe_ui_call(self._render_details, data, "OMDb (Saved to SQLite)" if saved else "OMDb")
            except OMDbError as e:
                cached = database.get_movie_by_id(self.imdb_id)
                if cached:
                    self.safe_ui_call(self._render_details, cached, "SQLite Cache (Offline)")
                else:
                    self.safe_ui_call(self._render_error, str(e))
            except Exception as e:
                self.safe_ui_call(self._render_error, f"Error: {str(e)}")
            finally:
                self.safe_ui_call(self._reset_refresh_button)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_refresh_clicked(self):
        self._fetch_from_omdb_async()

    def _reset_refresh_button(self):
        if self.winfo_exists():
            self.refresh_btn.config(state="normal", text="⟳ Refresh")

    def _render_error(self, message: str):
        if not self.winfo_exists():
            return
        self.status_label.config(
            text=f"Failed to load details: {message}",
            fg="#f87171",
            font=("Segoe UI", 11, "bold")
        )
        self.db_status_label.config(text="API retrieval failed", fg="#f87171")

    def _render_details(self, data: Dict[str, Any], source: str = "OMDb"):
        if not self.winfo_exists():
            return

        self.movie_data = data

        title = data.get("title") or data.get("Title") or "Untitled"
        year = data.get("year") or data.get("Year") or "N/A"
        rated = data.get("rated") or data.get("Rated") or "N/A"
        runtime = data.get("runtime") or data.get("Runtime") or "N/A"
        genre = data.get("genre") or data.get("Genre") or "N/A"
        m_type = (data.get("movie_type") or data.get("type") or data.get("Type") or "movie").capitalize()

        self.title(f"CineTrace — {title} ({year})")
        self.title_label.config(text=f"{title} ({year})")
        self.subtitle_label.config(text=f"{m_type} • Rated {rated} • {runtime} • {genre}")

        # Update poster asynchronously
        poster_url = data.get("poster_url") or data.get("poster") or data.get("Poster")
        self._load_detail_poster(poster_url, title)

        # Clear status label
        self.status_label.destroy()

        # Update SQLite status badge
        self.db_status_label.config(
            text=f"💾 {source}",
            fg="#34d399" if "SQLite" in source else "#38bdf8"
        )

        # Update Wishlist and Backlog buttons state
        self._update_membership_buttons()

        # Remove previous details content canvas if re-rendering
        for child in self.details_frame.winfo_children():
            child.destroy()

        # Canvas & scrollbar for details
        canvas = tk.Canvas(self.details_frame, bg="#0a0e17", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.details_frame, orient="vertical", command=canvas.yview)
        info_frame = tk.Frame(canvas, bg="#0a0e17")

        info_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=info_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Ratings card (crimson accent)
        ratings_box = tk.Frame(
            info_frame,
            bg="#111726",
            padx=14,
            pady=10,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground="#1d283c"
        )
        ratings_box.pack(fill="x", pady=(0, 14))

        imdb_rating = data.get("imdb_rating") or data.get("imdbRating") or "N/A"
        imdb_votes = data.get("imdb_votes") or data.get("imdbVotes") or "N/A"
        metascore = data.get("metascore") or data.get("Metascore") or "N/A"

        score_text = f"★ IMDb: {imdb_rating}/10 ({imdb_votes} votes)    |    Metascore: {metascore}"
        ratings_list = data.get("ratings")
        if isinstance(ratings_list, list):
            for r in ratings_list:
                if isinstance(r, dict) and r.get("Source") == "Rotten Tomatoes":
                    score_text += f"    |    Rotten Tomatoes: {r.get('Value')}"

        tk.Label(
            ratings_box,
            text=score_text,
            font=("Segoe UI", 10, "bold"),
            fg="#f43f5e",
            bg="#111726",
            anchor="w"
        ).pack(fill="x")

        # Plot Section
        tk.Label(
            info_frame,
            text="Plot Summary",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#0a0e17"
        ).pack(anchor="w", pady=(4, 4))

        plot_text = data.get("plot") or data.get("Plot") or "No plot summary available."
        tk.Label(
            info_frame,
            text=plot_text,
            font=("Segoe UI", 10),
            fg="#8e9db3",
            bg="#0a0e17",
            wraplength=480,
            justify="left"
        ).pack(anchor="w", pady=(0, 14))

        # Metadata grid
        meta_items = [
            ("Director", data.get("director") or data.get("Director") or "N/A"),
            ("Writers", data.get("writer") or data.get("Writer") or "N/A"),
            ("Cast", data.get("actors") or data.get("Actors") or "N/A"),
            ("Released", data.get("released") or data.get("Released") or "N/A"),
            ("Box Office", data.get("box_office") or data.get("BoxOffice") or "N/A"),
            ("Awards", data.get("awards") or data.get("Awards") or "N/A"),
            ("Country", data.get("country") or data.get("Country") or "N/A"),
            ("Language", data.get("language") or data.get("Language") or "N/A"),
            ("IMDb ID", data.get("imdb_id") or data.get("imdbID") or self.imdb_id),
        ]

        for label, val in meta_items:
            row = tk.Frame(info_frame, bg="#0a0e17")
            row.pack(fill="x", pady=2)

            tk.Label(
                row,
                text=f"{label}:",
                font=("Segoe UI", 9, "bold"),
                fg="#5a687d",
                bg="#0a0e17",
                width=11,
                anchor="w"
            ).pack(side="left", anchor="n")

            tk.Label(
                row,
                text=str(val),
                font=("Segoe UI", 9),
                fg="#cbd5e1",
                bg="#0a0e17",
                wraplength=scale_px(380),
                justify="left"
            ).pack(side="left", fill="x", expand=True, anchor="w")

    def _load_detail_poster(self, poster_url: Optional[str], title: str):
        def _fetch():
            fitted = fetch_and_fit_pil(poster_url, scale_px(200), scale_px(290))
            self.safe_ui_call(self._apply_poster_fitted, fitted, title)

        threading.Thread(target=_fetch, daemon=True).start()

    def _apply_poster_fitted(self, fitted: Any, title: str):
        if not self.winfo_exists():
            return
        if fitted is not None:
            try:
                photo = ImageTk.PhotoImage(fitted)
                self.poster_photo = photo
                self.poster_label.config(image=photo)
                return
            except Exception:
                pass
        photo = get_placeholder_photo(scale_px(200), scale_px(290), title)
        self.poster_photo = photo
        self.poster_label.config(image=photo)

    def _update_membership_buttons(self):
        """Update Wishlist and Backlog buttons to clearly indicate membership state."""
        if not self.imdb_id:
            return

        in_wish = database.is_in_wishlist(self.imdb_id)
        if in_wish:
            self.wishlist_btn.config(text="♥ In Wishlist", bg="#e11d48", fg="#ffffff")
        else:
            self.wishlist_btn.config(text="♡ Wishlist", bg="#1e2638", fg="#cbd5e1")

        in_back = database.is_in_backlog(self.imdb_id)
        if in_back:
            entry = database.get_backlog_entry(self.imdb_id)
            prio = entry.get("priority", "Medium") if entry else "Medium"
            status = entry.get("status", "Want to Watch") if entry else "Want to Watch"
            if status == "Watched":
                self.backlog_btn.config(text="✓ Watched", bg="#059669", fg="#ffffff")
            else:
                self.backlog_btn.config(text=f"▤ Backlog ({prio})", bg="#0284c7", fg="#ffffff")
        else:
            self.backlog_btn.config(text="▤ Backlog", bg="#1e2638", fg="#cbd5e1")

    def _toggle_wishlist(self):
        """Toggle Wishlist membership in SQLite."""
        if not self.imdb_id:
            return
        # Ensure movie metadata exists in SQLite
        if self.movie_data:
            database.save_movie(self.movie_data)

        if database.is_in_wishlist(self.imdb_id):
            database.remove_from_wishlist(self.imdb_id)
        else:
            database.add_to_wishlist(self.imdb_id, self.movie_data)

        self._update_membership_buttons()
        if self.on_change_callback:
            self.on_change_callback()

    def _toggle_backlog(self):
        """Toggle Backlog membership in SQLite; moves from Wishlist if already in Wishlist."""
        if not self.imdb_id:
            return
        if self.movie_data:
            database.save_movie(self.movie_data)

        if database.is_in_backlog(self.imdb_id):
            database.remove_from_backlog(self.imdb_id)
        else:
            if database.is_in_wishlist(self.imdb_id):
                database.move_wishlist_to_backlog(self.imdb_id, "Medium", "Want to Watch", self.movie_data)
            else:
                database.add_to_backlog(self.imdb_id, "Medium", "Want to Watch", self.movie_data)

        self._update_membership_buttons()
        if self.on_change_callback:
            self.on_change_callback()
