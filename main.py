# main.py (Tk version) — fixes NameError on lambda using `e`
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from languagemodel import run_translation
from text_to_speech_model import run_tts
from playsound import playsound

audio_file_path: Path | None = None

def _snapshot_wavs():
    cwd = Path.cwd()
    paths = list(cwd.glob("*.wav"))
    for sub in [p for p in cwd.iterdir() if p.is_dir()]:
        paths.extend(sub.glob("*.wav"))
    return paths

def call_tts_and_locate_wav(text: str, target_lang: str) -> Path | None:
    before = _snapshot_wavs()
    t0 = time.time()
    ret = run_tts(text, target_lang)

    if isinstance(ret, (str, Path)):
        p = Path(ret)
        if p.exists():
            return p

    time.sleep(0.1)
    after = _snapshot_wavs()
    new_paths = [p for p in after if p not in before]
    if new_paths:
        newest = max(new_paths, key=lambda p: p.stat().st_mtime)
        if newest.stat().st_mtime >= t0:
            return newest

    if after:
        newest_any = max(after, key=lambda p: p.stat().st_mtime)
        if newest_any.stat().st_mtime >= t0:
            return newest_any
    return None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EdgeLingo (Tk)")
        self.geometry("700x600")

        self.lang_var = tk.StringVar(value="English")
        self.translate_var = tk.StringVar(value="No")
        self.audio_var = tk.StringVar(value="Yes")

        # Input
        ttk.Label(self, text="Input Text:").pack(anchor="w", padx=10, pady=(10, 0))
        self.input_box = scrolledtext.ScrolledText(self, height=7, wrap=tk.WORD)
        self.input_box.pack(fill="both", expand=False, padx=10, pady=5)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Target language
        ttk.Label(self, text="Target Language:").pack(anchor="w", padx=10)
        lang_frame = ttk.Frame(self); lang_frame.pack(anchor="w", padx=10)
        ttk.Radiobutton(lang_frame, text="English", variable=self.lang_var, value="English").pack(side="left")
        ttk.Radiobutton(lang_frame, text="Spanish", variable=self.lang_var, value="Spanish").pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Translate?
        ttk.Label(self, text="Do you want to translate this text?").pack(anchor="w", padx=10)
        tr_frame = ttk.Frame(self); tr_frame.pack(anchor="w", padx=10)
        ttk.Radiobutton(tr_frame, text="Yes", variable=self.translate_var, value="Yes").pack(side="left")
        ttk.Radiobutton(tr_frame, text="No", variable=self.translate_var, value="No").pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Audio?
        ttk.Label(self, text="Do you want an audio file?").pack(anchor="w", padx=10)
        au_frame = ttk.Frame(self); au_frame.pack(anchor="w", padx=10)
        ttk.Radiobutton(au_frame, text="Yes", variable=self.audio_var, value="Yes").pack(side="left")
        ttk.Radiobutton(au_frame, text="No", variable=self.audio_var, value="No").pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Buttons
        btn_frame = ttk.Frame(self); btn_frame.pack(anchor="w", padx=10)
        self.process_btn = ttk.Button(btn_frame, text="Process", command=self.run_process)
        self.process_btn.pack(side="left")
        self.play_btn = ttk.Button(btn_frame, text="Play Audio", command=self.play_audio, state="disabled")
        self.play_btn.pack(side="left", padx=8)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        ttk.Label(self, text="Output Text / Status:").pack(anchor="w", padx=10)
        self.output_box = scrolledtext.ScrolledText(self, height=12, wrap=tk.WORD, state="normal")
        self.output_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def set_output(self, text: str):
        self.output_box.config(state="normal")
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, text)
        self.output_box.config(state="normal")

    def run_process(self):
        global audio_file_path
        audio_file_path = None
        self.process_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.set_output("Processing... Please wait.")

        def worker():
            global audio_file_path
            input_text = self.input_box.get("1.0", tk.END).strip()
            if not input_text:
                self.after(0, lambda: (
                    self.set_output("No input text provided."),
                    self.process_btn.config(state="normal")
                ))
                return

            target_lang = "E" if self.lang_var.get() == "English" else "S"
            do_translate = self.translate_var.get() == "Yes"
            do_audio = self.audio_var.get() == "Yes"

            try:
                translated = run_translation(input_text, target_lang) if do_translate else input_text
            except Exception as e:
                err = str(e)  # capture now (exception var is cleared after except)
                self.after(0, lambda err=err: (
                    self.set_output(f"Translation error: {err}"),
                    self.process_btn.config(state="normal")
                ))
                return

            if do_audio:
                try:
                    wav_path = call_tts_and_locate_wav(translated, target_lang)
                    if wav_path and wav_path.exists():
                        audio_file_path = wav_path
                        self.after(0, lambda: (
                            self.set_output(translated + "\n\nAudio ready. Click 'Play Audio' to listen."),
                            self.play_btn.config(state="normal"),
                            self.process_btn.config(state="normal")
                        ))
                    else:
                        self.after(0, lambda: (
                            self.set_output(translated + "\n\nAudio file not found after TTS."),
                            self.play_btn.config(state="disabled"),
                            self.process_btn.config(state="normal")
                        ))
                except Exception as e:
                    err = str(e)  # capture now
                    self.after(0, lambda err=err: (
                        self.set_output(translated + f"\n\nTTS error: {err}"),
                        self.play_btn.config(state="disabled"),
                        self.process_btn.config(state="normal")
                    ))
            else:
                self.after(0, lambda: (
                    self.set_output(translated),
                    self.play_btn.config(state="disabled"),
                    self.process_btn.config(state="normal")
                ))

        threading.Thread(target=worker, daemon=True).start()

    def play_audio(self):
        global audio_file_path
        if audio_file_path and Path(audio_file_path).exists():
            try:
                playsound(str(audio_file_path))
            except Exception as e:
                messagebox.showerror("Playback error", str(e))
        else:
            messagebox.showinfo("No audio", "No audio file available.")

if __name__ == "__main__":
    App().mainloop()
