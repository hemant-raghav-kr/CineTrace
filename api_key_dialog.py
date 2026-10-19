"""
CineTrace - API Key Configuration Dialog
Provides a user-friendly UI to configure the OMDb API key without exposing
secrets or requiring manual file editing, while persisting securely to .env.
Matches the dark cinematic styling.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

import config


from dpi_helper import scale_px


class ApiKeyDialog(tk.Toplevel):
    """Dialog for configuring the OMDb API Key."""

    def __init__(self, parent: tk.Widget, on_save_callback: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.on_save_callback = on_save_callback

        dlg_w = scale_px(490)
        dlg_h = scale_px(300)
        self.title("OMDb API Key Configuration")
        self.geometry(f"{dlg_w}x{dlg_h}")
        self.resizable(False, False)
        self.configure(bg="#0a0e17")

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

        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self, bg="#111726", padx=26, pady=24, relief="solid", bd=1, highlightthickness=1, highlightbackground="#1d283c")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        tk.Label(
            container,
            text="Configure OMDb API Key",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#111726"
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            container,
            text=(
                "CineTrace uses the OMDb API for movie searches and posters.\n"
                "Get a free API key at: https://www.omdbapi.com/apikey.aspx"
            ),
            font=("Segoe UI", 9),
            fg="#8e9db3",
            bg="#111726",
            justify="left"
        ).pack(anchor="w", pady=(0, 14))

        status_text = f"Current Status: {config.get_masked_api_key()}"
        tk.Label(
            container,
            text=status_text,
            font=("Segoe UI", 9, "italic"),
            fg="#f43f5e",
            bg="#111726"
        ).pack(anchor="w", pady=(0, 8))

        entry_frame = tk.Frame(container, bg="#111726")
        entry_frame.pack(fill="x", pady=(0, 16))

        tk.Label(
            entry_frame,
            text="API Key:",
            font=("Segoe UI", 9, "bold"),
            fg="#cbd5e1",
            bg="#111726"
        ).pack(anchor="w", pady=(0, 4))

        self.key_var = tk.StringVar(value="")
        self.key_entry = tk.Entry(
            entry_frame,
            textvariable=self.key_var,
            font=("Consolas", 10),
            bg="#161f30",
            fg="#ffffff",
            insertbackground="#ffffff",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#232f44",
            highlightcolor="#e11d48",
            show="•"
        )
        self.key_entry.pack(fill="x", ipady=6)

        # Toggle show/hide key
        self.show_var = tk.BooleanVar(value=False)

        def _toggle_visibility():
            if self.show_var.get():
                self.key_entry.config(show="")
            else:
                self.key_entry.config(show="•")

        tk.Checkbutton(
            entry_frame,
            text="Show key",
            variable=self.show_var,
            command=_toggle_visibility,
            bg="#111726",
            fg="#8e9db3",
            activebackground="#111726",
            activeforeground="#ffffff",
            selectcolor="#161f30",
            font=("Segoe UI", 8)
        ).pack(anchor="w", pady=(4, 0))

        # Button row
        btn_frame = tk.Frame(container, bg="#111726")
        btn_frame.pack(fill="x", side="bottom")

        cancel_btn = tk.Button(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            font=("Segoe UI", 9),
            bg="#1e2638",
            fg="#cbd5e1",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        )
        cancel_btn.pack(side="right", padx=(8, 0))

        save_btn = tk.Button(
            btn_frame,
            text="Save Key",
            command=self._on_save,
            font=("Segoe UI", 9, "bold"),
            bg="#e11d48",
            fg="#ffffff",
            activebackground="#be123c",
            activeforeground="#ffffff",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2"
        )
        save_btn.pack(side="right")

    def _on_save(self):
        new_key = self.key_var.get().strip()
        if not new_key:
            messagebox.showwarning(
                "Empty API Key",
                "Please enter an OMDb API key.",
                parent=self
            )
            return

        config.set_api_key(new_key, persist_to_env=True)
        messagebox.showinfo(
            "Saved",
            "OMDb API key successfully saved and persisted to .env!",
            parent=self
        )
        if self.on_save_callback:
            self.on_save_callback()
        self.destroy()
