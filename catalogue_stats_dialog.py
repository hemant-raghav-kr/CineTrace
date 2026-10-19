"""
CineTrace — Movie Catalogue Statistics & Source Licensing Dialog
Displays local catalogue record metrics, plot/cast coverage, SQLite FTS5 status,
and documented dataset provenance/licensing details.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional

import database
from dpi_helper import scale_px

# Design constants matching CineTrace dark theme
COLOR_BG = "#0a0e17"
COLOR_PANEL = "#111726"
COLOR_CARD = "#161f30"
COLOR_BORDER = "#1d283c"
COLOR_ACCENT = "#e11d48"
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_MUTED = "#8e9db3"
COLOR_TEXT_DIM = "#5a687d"


class CatalogueStatsDialog(tk.Toplevel):
    """Themed modal dialog displaying catalogue metrics and licensing details."""

    def __init__(self, parent: tk.Widget, db_path: Optional[Any] = None):
        super().__init__(parent)
        self.db_path = db_path
        self.title("Catalogue Statistics & Data Sources — CineTrace")
        self.geometry(f"{scale_px(640)}x{scale_px(600)}")
        self.minsize(scale_px(580), scale_px(520))
        self.configure(bg=COLOR_BG)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._center_window(parent)

    def _center_window(self, parent: tk.Widget):
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = self.winfo_width()
            h = self.winfo_height()
            x = max(0, px + (pw - w) // 2)
            y = max(0, py + (ph - h) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _build_ui(self):
        stats = database.get_catalogue_stats(self.db_path)
        self.stats = stats

        # Header Frame
        header = tk.Frame(self, bg=COLOR_BG, padx=24, pady=18)
        header.pack(fill="x")

        tk.Label(
            header,
            text="📚 Movie Catalogue & Data Sources",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Metrics, plot/cast coverage, search engine indexing status, and dataset licensing terms.",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG
        ).pack(anchor="w", pady=(2, 0))

        # Scrollable content canvas
        container = tk.Frame(self, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        canvas = tk.Canvas(container, bg=COLOR_BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=580)
        canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Mousewheel binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Section 1: Coverage Metrics
        card1 = tk.Frame(
            scrollable_frame,
            bg=COLOR_PANEL,
            padx=18,
            pady=16,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        card1.pack(fill="x", pady=(0, 14))

        tk.Label(
            card1,
            text="Catalogue Coverage Metrics",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 10))

        grid = tk.Frame(card1, bg=COLOR_PANEL)
        grid.pack(fill="x")

        tot = stats.get("total_movies", 0)
        p_cnt = stats.get("movies_with_plots", 0)
        a_cnt = stats.get("movies_with_actors", 0)
        d_cnt = stats.get("movies_with_directors", 0)
        y_cnt = stats.get("movies_with_years", 0)
        g_cnt = stats.get("movies_with_genres", 0)
        id_cnt = stats.get("movies_with_imdb_id", 0)
        fts_on = stats.get("fts5_enabled", False)

        metrics = [
            ("Total Indexed Titles", str(tot), "#38bdf8"),
            ("Titles with Release Years", f"{y_cnt} ({y_cnt*100//max(1, tot)}%)", "#f59e0b"),
            ("Titles with Plot Summaries", f"{p_cnt} ({p_cnt*100//max(1, tot)}%)", "#34d399"),
            ("Titles with Cast / Actor Names", f"{a_cnt} ({a_cnt*100//max(1, tot)}%)", "#a78bfa"),
            ("Titles with Genre Metadata", f"{g_cnt} ({g_cnt*100//max(1, tot)}%)", "#ec4899"),
            ("Titles with Director Metadata", f"{d_cnt} ({d_cnt*100//max(1, tot)}%)", "#f472b6"),
            ("Verified IMDb IDs (OMDb-linked)", f"{id_cnt} ({id_cnt*100//max(1, tot)}%)", "#fbbf24"),
            ("SQLite FTS5 Full-Text Index", "Active & Enabled" if fts_on else "Fallback LIKE Mode", "#34d399" if fts_on else "#f87171"),
        ]

        for i, (label, val, color) in enumerate(metrics):
            r = i // 2
            c = i % 2
            box = tk.Frame(grid, bg=COLOR_CARD, padx=12, pady=8, bd=1, relief="solid", highlightthickness=1, highlightbackground=COLOR_BORDER)
            box.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            grid.grid_columnconfigure(c, weight=1)
            tk.Label(box, text=label, font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_CARD).pack(anchor="w")
            tk.Label(box, text=val, font=("Segoe UI", 11, "bold"), fg=color, bg=COLOR_CARD).pack(anchor="w", pady=(2, 0))

        # Section 2: Provenance Breakdown
        card2 = tk.Frame(
            scrollable_frame,
            bg=COLOR_PANEL,
            padx=18,
            pady=16,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        card2.pack(fill="x", pady=(0, 14))

        tk.Label(
            card2,
            text="Source Provenance Breakdown",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 8))

        provs = stats.get("provenances", {})
        if provs:
            for source, count in provs.items():
                p_row = tk.Frame(card2, bg=COLOR_PANEL)
                p_row.pack(fill="x", pady=2)
                tk.Label(p_row, text=f"• {source}", font=("Segoe UI", 9), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL).pack(side="left")
                tk.Label(p_row, text=f"{count} films", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg=COLOR_PANEL).pack(side="right")
        else:
            tk.Label(card2, text="No movies currently indexed.", font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL).pack(anchor="w")

        # Section 3: Documented Dataset Sources & Licenses
        card3 = tk.Frame(
            scrollable_frame,
            bg=COLOR_PANEL,
            padx=18,
            pady=16,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        card3.pack(fill="x", pady=(0, 14))

        tk.Label(
            card3,
            text="Legitimate Data Sources & Licensing",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 6))

        sources_info = (
            "1. Wikipedia Movie Plots Dataset (CC BY-SA 4.0)\n"
            "   • ~35,000 movie plot descriptions, release years, cast, directors, and Wikipedia pages.\n"
            "   • Origin: Kaggle (jrobischon/wikipedia-movie-plots), licensed under CC BY-SA 4.0.\n"
            "   • Free for educational, research, and non-commercial/commercial use with attribution.\n"
            "   • Preserves Share-Alike obligations; derivative works must remain open under CC BY-SA.\n\n"
            "2. CMU Movie Summary Corpus (CC BY-SA 3.0)\n"
            "   • 42,306 plot summaries and character metadata by David Bamman et al. (Carnegie Mellon).\n"
            "   • Standardized Freebase / IMDb alignment.\n\n"
            "3. IMDb Non-Commercial Datasets\n"
            "   • Official dumps (title.basics.tsv, title.principals.tsv) provide titles, years, cast, genres.\n"
            "   • Free for non-commercial personal and educational use only.\n"
            "   • Note: IMDb dumps do NOT include plot summaries; plots must come from Wikipedia.\n\n"
            "4. User Imports (CSV / TSV / JSON)\n"
            "   • CineTrace supports streaming import of arbitrary user files without proprietary lock-in."
        )
        tk.Label(
            card3,
            text=sources_info,
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            justify="left"
        ).pack(anchor="w")

        # Close Button Footer
        footer = tk.Frame(self, bg=COLOR_BG, padx=20, pady=10)
        footer.pack(fill="x", side="bottom")

        close_btn = tk.Button(
            footer,
            text="Close",
            command=self.destroy,
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_ACCENT,
            fg="#ffffff",
            activebackground="#be123c",
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=6,
            cursor="hand2"
        )
        close_btn.pack(side="right")
