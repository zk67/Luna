"""
Luna - Graphical Interface (Tkinter, zero external dependency)
-----------------------------------------------------------------
A modern-looking, single-window desktop GUI for the Luna project,
built entirely with Tkinter (included with every Python install --
no pip install, no path-length issues, works everywhere).

Run with:
    python gui.py

Expects the same project layout as main.py:
    training/train.py        -> train, load_new_dataset, reset_model
    tokenizer/tokenizer.py   -> tokenize_function
    generation/generation.py -> run_generation
    config.py, paths.py
"""

import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import config
from generation.generation import run_generation
from training.train import train, load_new_dataset, reset_model


BG_DARK = "#f5f5f7"         
BG_PANEL = "#ffffff"         # sidebar / panels
BG_PANEL_HOVER = "#e8e8ed"
BG_BUBBLE_USER = "#1c2b4a"   # navy blue
BG_BUBBLE_LUNA = "#eef0f4"   # light gray bubble
ACCENT_CYAN = "#2e5aac"      # muted blue, used for active/hover states
ACCENT_MAGENTA = "#1c2b4a"   # navy blue, primary accent (buttons, active nav)
TEXT_PRIMARY = "#1d1d1f"     
TEXT_ON_BUBBLE = "#ffffff"
TEXT_ON_BUBBLE_LUNA = "#1d1d1f"
TEXT_MUTED = "#6e6e73"       
BORDER_LIGHT = "#d2d2d7"

FONT_FAMILY = "Segoe UI"


class LunaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Luna")
        self.geometry("1000x680")
        self.configure(bg=BG_DARK)
        self.minsize(820, 560)

        self.title_font = tkfont.Font(family=FONT_FAMILY, size=20, weight="bold")
        self.subtitle_font = tkfont.Font(family=FONT_FAMILY, size=10)
        self.page_title_font = tkfont.Font(family=FONT_FAMILY, size=18, weight="bold")
        self.page_subtitle_font = tkfont.Font(family=FONT_FAMILY, size=10)
        self.nav_font = tkfont.Font(family=FONT_FAMILY, size=11)
        self.body_font = tkfont.Font(family=FONT_FAMILY, size=11)

        self._build_layout()
        self.show_page("chat")

    # ---------------- LAYOUT ----------------
    def _build_layout(self):
        root = tk.Frame(self, bg=BG_DARK)
        root.pack(fill="both", expand=True)

        # --- Sidebar ---
        sidebar = tk.Frame(root, bg=BG_PANEL, width=220, highlightbackground=BORDER_LIGHT,
                            highlightthickness=0, bd=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        separator = tk.Frame(root, bg=BORDER_LIGHT, width=1)
        separator.pack(side="left", fill="y")

        tk.Label(sidebar, text="LUNA", bg=BG_PANEL, fg=ACCENT_MAGENTA,
                  font=self.title_font).pack(anchor="w", padx=18, pady=(20, 0))
        tk.Label(sidebar, text="Local AI Model · v0.1.0", bg=BG_PANEL, fg=TEXT_MUTED,
                  font=self.subtitle_font).pack(anchor="w", padx=18, pady=(0, 18))

        self.nav_buttons = {}
        for key, label in [("chat", "Chat"), ("training", "Training"), ("params", "Parameters")]:
            btn = tk.Label(sidebar, text=label, bg=BG_PANEL, fg=TEXT_PRIMARY,
                            font=self.nav_font, anchor="w", padx=18, pady=12, cursor="hand2")
            btn.pack(fill="x", padx=10, pady=2)
            btn.bind("<Button-1>", lambda e, k=key: self.show_page(k))
            btn.bind("<Enter>", lambda e, b=btn, k=key: self._nav_hover(b, k, True))
            btn.bind("<Leave>", lambda e, b=btn, k=key: self._nav_hover(b, k, False))
            self.nav_buttons[key] = btn

        # --- Page container ---
        self.page_container = tk.Frame(root, bg=BG_DARK)
        self.page_container.pack(side="left", fill="both", expand=True)

        self.pages = {
            "chat": ChatPage(self.page_container, self),
            "training": TrainingPage(self.page_container, self),
            "params": ParametersPage(self.page_container, self),
        }
        for page in self.pages.values():
            page.place(x=0, y=0, relwidth=1, relheight=1)

        self.active_page = None

    def _nav_hover(self, btn, key, entering):
        if key == self.active_page:
            return
        btn.configure(bg=BG_PANEL_HOVER if entering else BG_PANEL)

    def show_page(self, key):
        self.active_page = key
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(bg="#e4e9f2", fg=ACCENT_MAGENTA)
            else:
                btn.configure(bg=BG_PANEL, fg=TEXT_PRIMARY)
        self.pages[key].tkraise()


# ---------------------- CHAT PAGE ----------------------
class ChatPage(tk.Frame):
    def __init__(self, parent, app: LunaApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app

        header = tk.Frame(self, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(24, 10))
        tk.Label(header, text="Chat with Luna", bg=BG_DARK, fg=TEXT_PRIMARY,
                  font=app.page_title_font).pack(anchor="w")
        tk.Label(header, text="Ask anything. Luna is running fully offline, on your machine.",
                  bg=BG_DARK, fg=TEXT_MUTED, font=app.page_subtitle_font).pack(anchor="w")

        # Scrollable chat area
        chat_frame = tk.Frame(self, bg=BG_DARK)
        chat_frame.pack(fill="both", expand=True, padx=24, pady=6)

        self.canvas = tk.Canvas(chat_frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(chat_frame, orient="vertical", command=self.canvas.yview)
        self.messages_frame = tk.Frame(self.canvas, bg=BG_DARK)

        self.messages_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.messages_frame, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Input row
        input_row = tk.Frame(self, bg=BG_DARK)
        input_row.pack(fill="x", padx=30, pady=(6, 24))

        self.entry = tk.Entry(
            input_row, bg=BG_PANEL, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            relief="flat", font=app.body_font,
            highlightthickness=1, highlightbackground=BORDER_LIGHT, highlightcolor=ACCENT_CYAN,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_message())

        send_btn = tk.Label(
            input_row, text="Send", bg=ACCENT_MAGENTA, fg="#ffffff",
            font=tkfont.Font(family=FONT_FAMILY, size=11, weight="bold"),
            padx=22, pady=10, cursor="hand2",
        )
        send_btn.pack(side="left")
        send_btn.bind("<Button-1>", lambda e: self.send_message())
        send_btn.bind("<Enter>", lambda e: send_btn.configure(bg=ACCENT_CYAN))
        send_btn.bind("<Leave>", lambda e: send_btn.configure(bg=ACCENT_MAGENTA))

        self._add_bubble("Hi, I'm Luna. Ask me a question to get started.", is_user=False)

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _add_bubble(self, text, is_user):
        row = tk.Frame(self.messages_frame, bg=BG_DARK)
        row.pack(fill="x", pady=4)

        bubble_bg = BG_BUBBLE_USER if is_user else BG_BUBBLE_LUNA
        bubble_fg = TEXT_ON_BUBBLE if is_user else TEXT_ON_BUBBLE_LUNA
        label = tk.Label(
            row, text=text, bg=bubble_bg, fg=bubble_fg, font=self.app.body_font,
            wraplength=440, justify="left", padx=14, pady=10,
        )
        if is_user:
            label.pack(side="right", padx=(80, 0))
        else:
            label.pack(side="left", padx=(0, 80))

        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)
        return row

    def send_message(self):
        text = self.entry.get().strip()
        if not text:
            return
        self._add_bubble(text, is_user=True)
        self.entry.delete(0, tk.END)
        self.entry.configure(state="disabled")

        thinking_row = self._add_bubble("Luna is thinking…", is_user=False)

        def worker():
            try:
                response = run_generation(text)
            except Exception as e:
                response = f"[Error] {e}"
            self.after(0, lambda: self._on_response(thinking_row, response))

        threading.Thread(target=worker, daemon=True).start()

    def _on_response(self, thinking_row, response):
        thinking_row.destroy()
        self._add_bubble(response, is_user=False)
        self.entry.configure(state="normal")
        self.entry.focus_set()


# ---------------------- TRAINING PAGE ----------------------
class TrainingPage(tk.Frame):
    def __init__(self, parent, app: LunaApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app

        header = tk.Frame(self, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(24, 10))
        tk.Label(header, text="Training", bg=BG_DARK, fg=TEXT_PRIMARY,
                  font=app.page_title_font).pack(anchor="w")
        tk.Label(header, text="Manage Luna's training data and run new training loops.",
                  bg=BG_DARK, fg=TEXT_MUTED, font=app.page_subtitle_font).pack(anchor="w")

        tk.Label(self, text="Current dataset: input_text.txt", bg=BG_DARK, fg=TEXT_MUTED,
                  font=app.body_font).pack(anchor="w", padx=30, pady=(6, 16))

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(anchor="w", padx=30)

        self._make_action_button(btn_row, "Start training", self.start_training).pack(side="left", padx=(0, 10))
        self._make_action_button(btn_row, "Load new dataset", self.load_dataset).pack(side="left", padx=(0, 10))
        self._make_action_button(btn_row, "Reset model", self.reset_model_action).pack(side="left")

    def _make_action_button(self, parent, text, command):
        btn = tk.Label(
            parent, text=text, bg=BG_PANEL, fg=TEXT_PRIMARY, font=self.app.body_font,
            padx=16, pady=10, cursor="hand2", relief="flat",
            highlightthickness=1, highlightbackground=BORDER_LIGHT,
        )
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_PANEL_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_PANEL))
        return btn

    def start_training(self):
        messagebox.showinfo("Training", "Training started in the background (see console for progress).")
        threading.Thread(target=lambda: train(1), daemon=True).start()

    def load_dataset(self):
        if messagebox.askyesno("Load new dataset", "This resets the vocabulary and merge rules. Continue?"):
            load_new_dataset()
            messagebox.showinfo("Done", "Dataset, vocabulary, and merge rules reloaded.")

    def reset_model_action(self):
        if messagebox.askyesno("Reset model", "This reinitializes the embedding and output matrices. Continue?"):
            reset_model()
            messagebox.showinfo("Done", "Model weights reset.")


# ---------------------- PARAMETERS PAGE ----------------------
class ParametersPage(tk.Frame):
    PARAM_DEFS = [
        ("BATCH_SIZE", "Batch size"),
        ("HIDDEN_SIZE", "Hidden size"),
        ("CONTEXT_SIZE", "Context size"),
        ("HEADS", "Attention heads"),
        ("INTERMEDIATE_SIZE", "Intermediate size"),
        ("LAYERS", "Transformer layers"),
    ]

    def __init__(self, parent, app: LunaApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self.vars = {}

        header = tk.Frame(self, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(24, 10))
        tk.Label(header, text="Model Parameters", bg=BG_DARK, fg=TEXT_PRIMARY,
                  font=app.page_title_font).pack(anchor="w")
        tk.Label(header, text="Adjust Luna's architecture and training hyperparameters.",
                  bg=BG_DARK, fg=TEXT_MUTED, font=app.page_subtitle_font).pack(anchor="w")

        form = tk.Frame(self, bg=BG_DARK)
        form.pack(anchor="w", padx=30, pady=10)

        for i, (attr, label) in enumerate(self.PARAM_DEFS):
            tk.Label(form, text=label, bg=BG_DARK, fg=TEXT_PRIMARY, font=app.body_font,
                      width=20, anchor="w").grid(row=i, column=0, sticky="w", pady=6)
            var = tk.StringVar(value=str(getattr(config, attr)))
            entry = tk.Entry(form, textvariable=var, bg=BG_PANEL, fg=TEXT_PRIMARY,
                              insertbackground=TEXT_PRIMARY, relief="flat", font=app.body_font, width=12,
                              highlightthickness=1, highlightbackground=BORDER_LIGHT, highlightcolor=ACCENT_CYAN)
            entry.grid(row=i, column=1, sticky="w", padx=10, ipady=4)
            self.vars[attr] = var

        save_btn = tk.Label(
            self, text="Save parameters", bg=ACCENT_MAGENTA, fg="#ffffff",
            font=tkfont.Font(family=FONT_FAMILY, size=11, weight="bold"),
            padx=18, pady=10, cursor="hand2",
        )
        save_btn.pack(anchor="w", padx=30, pady=(10, 0))
        save_btn.bind("<Button-1>", lambda e: self.save_params())
        save_btn.bind("<Enter>", lambda e: save_btn.configure(bg=ACCENT_CYAN))
        save_btn.bind("<Leave>", lambda e: save_btn.configure(bg=ACCENT_MAGENTA))

    def save_params(self):
        try:
            values = {attr: int(var.get()) for attr, var in self.vars.items()}
        except ValueError:
            messagebox.showerror("Invalid value", "All parameters must be whole numbers.")
            return

        for attr, value in values.items():
            setattr(config, attr, value)

        with open("config.py", "w", encoding="utf-8") as f:
            f.write(f"VOCAB_SIZE = {config.VOCAB_SIZE}\n")
            f.write(f"HIDDEN_SIZE = {config.HIDDEN_SIZE}\n")
            f.write(f"CONTEXT_SIZE = {config.CONTEXT_SIZE}\n")
            f.write(f"HEADS = {config.HEADS}\n")
            f.write(f"INTERMEDIATE_SIZE = {config.INTERMEDIATE_SIZE}\n")
            f.write(f"LAYERS = {config.LAYERS}\n")
            f.write(f"BATCH_SIZE = {config.BATCH_SIZE}\n")
            f.write(f"TEMPERATURE = {config.TEMPERATURE}\n")
        messagebox.showinfo("Saved", "Parameters updated and saved to config.py.")


if __name__ == "__main__":
    app = LunaApp()
    app.mainloop()