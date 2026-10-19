"""
CineTrace - Language Preferences & First-Launch Onboarding Dialog
Inspired by streaming platforms such as Hotstar.
Allows users to multi-select preferred languages to personalise recommendations
on Home and Discover. Fully styled in CineTrace's dark cinematic theme with
High-DPI scaling support.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Set

import database
from dpi_helper import scale_px
import language_service
from language_service import SUPPORTED_LANGUAGES, LANGUAGE_METADATA

COLOR_BG = "#0a0e17"           # Deep obsidian
COLOR_CARD = "#111726"         # Panel background
COLOR_BORDER = "#1d283c"       # Slate border
COLOR_BORDER_SEL = "#e11d48"   # Rose accent border for selected card
COLOR_CARD_UNSEL = "#161f30"   # Unselected card background
COLOR_CARD_SEL = "#2b1424"     # Selected card background tint
COLOR_ACCENT = "#e11d48"       # Crimson/rose accent
COLOR_ACCENT_HOVER = "#be123c"
COLOR_TEXT_MAIN = "#ffffff"    # Heading white
COLOR_TEXT_MUTED = "#8e9db3"   # Slate muted
COLOR_TEXT_DIM = "#5a687d"     # Secondary subtle grey


class LanguagePreferencesDialog(tk.Toplevel):
    """
    Modal dialog for first-launch onboarding and subsequent preference management.
    Supports multi-selection of 6 languages with immediate visual state feedback.
    """

    def __init__(
        self,
        parent: tk.Widget,
        is_onboarding: bool = False,
        on_save_callback: Optional[Callable[[List[str]], None]] = None,
    ):
        super().__init__(parent)
        self.is_onboarding = is_onboarding
        self.on_save_callback = on_save_callback

        # Initial selected languages from database
        saved = database.get_preferred_languages()
        self.selected_languages: Set[str] = set(saved)

        # Window dimensions and placement
        dlg_w = scale_px(580)
        dlg_h = scale_px(490) if is_onboarding else scale_px(520)
        self.title("What do you love watching? — CineTrace" if is_onboarding else "Language Preferences — CineTrace")
        self.geometry(f"{dlg_w}x{dlg_h}")
        self.minsize(scale_px(520), scale_px(450))
        self.configure(bg=COLOR_BG)

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + max(0, (parent.winfo_width() - dlg_w) // 2)
            y = parent.winfo_rooty() + max(0, (parent.winfo_height() - dlg_h) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

        self.card_frames: dict = {}
        self._build_ui()

        # Keyboard bindings
        self.bind("<Escape>", lambda e: self._on_skip() if self.is_onboarding else self._on_cancel())
        self.bind("<Return>", lambda e: self._on_continue())

    def _build_ui(self):
        container = tk.Frame(
            self,
            bg=COLOR_CARD,
            padx=scale_px(28),
            pady=scale_px(24),
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        container.pack(fill="both", expand=True, padx=scale_px(16), pady=scale_px(16))

        # Top Badge / Tag
        badge_text = "✦ PERSONALISED FOR YOU" if self.is_onboarding else "✦ RECOMMENDATION SETTINGS"
        tk.Label(
            container,
            text=badge_text,
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_CARD,
        ).pack(anchor="w", pady=(0, 4))

        # Heading
        heading_text = "What do you love watching?" if self.is_onboarding else "Movie Language Preferences"
        tk.Label(
            container,
            text=heading_text,
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_CARD,
        ).pack(anchor="w", pady=(0, 4))

        # Explanation subtitle
        subtitle_text = (
            "Select one or more languages. CineTrace personalises your Home and\n"
            "Discover recommendations to surface the cinema you love most."
        )
        tk.Label(
            container,
            text=subtitle_text,
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_CARD,
            justify="left",
        ).pack(anchor="w", pady=(0, 16))

        # 6 Language Cards in a 2-row x 3-column Grid
        grid_frame = tk.Frame(container, bg=COLOR_CARD)
        grid_frame.pack(fill="both", expand=True, pady=(0, 16))

        for col in range(3):
            grid_frame.columnconfigure(col, weight=1, uniform="card_col")
        for row in range(2):
            grid_frame.rowconfigure(row, weight=1, uniform="card_row")

        for idx, lang in enumerate(SUPPORTED_LANGUAGES):
            row = idx // 3
            col = idx % 3
            self._create_language_card(grid_frame, lang, row, col)

        # Action Buttons Container
        btn_frame = tk.Frame(container, bg=COLOR_CARD)
        btn_frame.pack(fill="x", side="bottom", pady=(10, 0))

        if self.is_onboarding:
            # Continue Button (Primary Accent)
            continue_btn = tk.Button(
                btn_frame,
                text="Continue →",
                command=self._on_continue,
                font=("Segoe UI", 10, "bold"),
                bg=COLOR_ACCENT,
                fg="#ffffff",
                activebackground=COLOR_ACCENT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=scale_px(22),
                pady=scale_px(8),
                cursor="hand2",
            )
            continue_btn.pack(side="right")

            # Skip For Now Button (Ghost / Link)
            skip_btn = tk.Button(
                btn_frame,
                text="Skip for now",
                command=self._on_skip,
                font=("Segoe UI", 9),
                bg=COLOR_CARD,
                fg=COLOR_TEXT_MUTED,
                activebackground=COLOR_CARD,
                activeforeground=COLOR_TEXT_MAIN,
                bd=0,
                padx=scale_px(14),
                pady=scale_px(8),
                cursor="hand2",
            )
            skip_btn.pack(side="left")
        else:
            # Save Preferences Button (Primary Accent)
            save_btn = tk.Button(
                btn_frame,
                text="Save Preferences",
                command=self._on_continue,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_ACCENT,
                fg="#ffffff",
                activebackground=COLOR_ACCENT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                padx=scale_px(18),
                pady=scale_px(7),
                cursor="hand2",
            )
            save_btn.pack(side="right")

            # Reset Preferences Button
            reset_btn = tk.Button(
                btn_frame,
                text="Reset to Default",
                command=self._on_reset,
                font=("Segoe UI", 9),
                bg="#1e293b",
                fg="#fca5a5",
                activebackground="#334155",
                activeforeground="#ef4444",
                bd=0,
                padx=scale_px(14),
                pady=scale_px(7),
                cursor="hand2",
            )
            reset_btn.pack(side="left")

            # Cancel Button
            cancel_btn = tk.Button(
                btn_frame,
                text="Cancel",
                command=self._on_cancel,
                font=("Segoe UI", 9),
                bg=COLOR_CARD,
                fg=COLOR_TEXT_MUTED,
                activebackground=COLOR_CARD,
                activeforeground=COLOR_TEXT_MAIN,
                bd=0,
                padx=scale_px(12),
                pady=scale_px(7),
                cursor="hand2",
            )
            cancel_btn.pack(side="left", padx=(8, 0))

    def _create_language_card(self, parent: tk.Widget, lang: str, row: int, col: int):
        meta = LANGUAGE_METADATA.get(lang, {"native": lang, "subtitle": "", "icon": "🎬"})
        is_sel = lang in self.selected_languages

        card = tk.Frame(
            parent,
            bg=COLOR_CARD_SEL if is_sel else COLOR_CARD_UNSEL,
            bd=2 if is_sel else 1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER_SEL if is_sel else COLOR_BORDER,
            padx=scale_px(12),
            pady=scale_px(10),
            cursor="hand2",
        )
        card.grid(row=row, column=col, padx=scale_px(6), pady=scale_px(6), sticky="nsew")

        # Top row: Name + Checkmark badge
        top_row = tk.Frame(card, bg=card["bg"])
        top_row.pack(fill="x")

        lbl_name = tk.Label(
            top_row,
            text=f"{meta['icon']} {lang}",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_TEXT_MAIN if is_sel else "#cbd5e1",
            bg=card["bg"],
        )
        lbl_name.pack(side="left")

        lbl_check = tk.Label(
            top_row,
            text="✓" if is_sel else "○",
            font=("Segoe UI", 11, "bold" if is_sel else "normal"),
            fg=COLOR_ACCENT if is_sel else COLOR_TEXT_DIM,
            bg=card["bg"],
        )
        lbl_check.pack(side="right")

        # Native script
        lbl_native = tk.Label(
            card,
            text=meta["native"],
            font=("Segoe UI", 10),
            fg="#f43f5e" if is_sel else COLOR_TEXT_MUTED,
            bg=card["bg"],
        )
        lbl_native.pack(anchor="w", pady=(3, 1))

        # Subtitle
        lbl_sub = tk.Label(
            card,
            text=meta["subtitle"],
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED if is_sel else COLOR_TEXT_DIM,
            bg=card["bg"],
        )
        lbl_sub.pack(anchor="w")

        # Store widget references for dynamic updating
        self.card_frames[lang] = {
            "card": card,
            "top_row": top_row,
            "lbl_name": lbl_name,
            "lbl_check": lbl_check,
            "lbl_native": lbl_native,
            "lbl_sub": lbl_sub,
        }

        # Bind click events across all children
        for w in (card, top_row, lbl_name, lbl_check, lbl_native, lbl_sub):
            w.bind("<Button-1>", lambda e, l=lang: self._toggle_language(l))
            w.bind("<Enter>", lambda e, l=lang: self._on_card_hover(l, True))
            w.bind("<Leave>", lambda e, l=lang: self._on_card_hover(l, False))

    def _toggle_language(self, lang: str):
        """Toggle selected state for the given language card."""
        if lang in self.selected_languages:
            self.selected_languages.remove(lang)
        else:
            self.selected_languages.add(lang)
        self._update_card_visuals(lang)

    def _update_card_visuals(self, lang: str):
        widgets = self.card_frames.get(lang)
        if not widgets:
            return

        is_sel = lang in self.selected_languages
        bg_col = COLOR_CARD_SEL if is_sel else COLOR_CARD_UNSEL
        border_col = COLOR_BORDER_SEL if is_sel else COLOR_BORDER

        card = widgets["card"]
        card.config(
            bg=bg_col,
            bd=2 if is_sel else 1,
            highlightbackground=border_col,
        )
        widgets["top_row"].config(bg=bg_col)
        widgets["lbl_name"].config(bg=bg_col, fg=COLOR_TEXT_MAIN if is_sel else "#cbd5e1")
        widgets["lbl_check"].config(
            bg=bg_col,
            text="✓" if is_sel else "○",
            font=("Segoe UI", 11, "bold" if is_sel else "normal"),
            fg=COLOR_ACCENT if is_sel else COLOR_TEXT_DIM,
        )
        widgets["lbl_native"].config(
            bg=bg_col,
            fg="#f43f5e" if is_sel else COLOR_TEXT_MUTED,
        )
        widgets["lbl_sub"].config(
            bg=bg_col,
            fg=COLOR_TEXT_MUTED if is_sel else COLOR_TEXT_DIM,
        )

    def _on_card_hover(self, lang: str, is_hovering: bool):
        widgets = self.card_frames.get(lang)
        if not widgets:
            return
        is_sel = lang in self.selected_languages
        if not is_sel:
            bg_col = "#1f2a3f" if is_hovering else COLOR_CARD_UNSEL
            border_col = "#334155" if is_hovering else COLOR_BORDER
            widgets["card"].config(bg=bg_col, highlightbackground=border_col)
            widgets["top_row"].config(bg=bg_col)
            widgets["lbl_name"].config(bg=bg_col)
            widgets["lbl_check"].config(bg=bg_col)
            widgets["lbl_native"].config(bg=bg_col)
            widgets["lbl_sub"].config(bg=bg_col)

    def _on_continue(self):
        """Save selected languages to SQLite and notify caller."""
        selected_list = [lang for lang in SUPPORTED_LANGUAGES if lang in self.selected_languages]
        database.set_preferred_languages(selected_list)
        if self.on_save_callback:
            try:
                self.on_save_callback(selected_list)
            except Exception:
                pass
        self.destroy()

    def _on_skip(self):
        """Handle Skip for now: mark onboarding complete without setting preferences."""
        database.set_onboarding_completed(True)
        if self.on_save_callback:
            try:
                self.on_save_callback([])
            except Exception:
                pass
        self.destroy()

    def _on_reset(self):
        """Reset preferences to empty (default / all movies) and notify caller."""
        database.clear_preferred_languages()
        self.selected_languages.clear()
        for lang in SUPPORTED_LANGUAGES:
            self._update_card_visuals(lang)
        if self.on_save_callback:
            try:
                self.on_save_callback([])
            except Exception:
                pass
        self.destroy()

    def _on_cancel(self):
        """Close dialog without saving changes."""
        self.destroy()
