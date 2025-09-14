#!/usr/bin/env python3
# runners/main.py

import argparse
import os
import sys
import time
import io
import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel

# --- Try imports from your repo ---
try:
    import asr  # provides transcribe(wav_bytes, src_lang=...)
except Exception as e:
    asr = None
    ASR_IMPORT_ERR = e

try:
    import audio_io  # provides record(seconds=...) -> WAV bytes
except Exception as e:
    audio_io = None
    AUDIO_IMPORT_ERR = e


# -----------------------------
# ONNX Runtime EP helper (info)
# -----------------------------
def ort_providers():
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


# -----------------------------
# Minimal Piper TTS (offline)
# -----------------------------
class PiperTTS:
    """
    Minimal Piper wrapper.
    You must provide:
      - voice_path: path to a .onnx voice model (e.g., 'en_US-amy-medium.onnx')
      - config_path: path to its .json config (same voice pack), sometimes optional
    """
    def __init__(self, voice_path: str, config_path: str | None = None, device: str = "cpu"):
        self.voice_path = voice_path
        self.config_path = config_path
        self.device = device
        try:
            from piper.voice import PiperVoice
            self._PiperVoice = PiperVoice
            self.voice = PiperVoice(voice_path, config_path=config_path, use_cuda=False)
            self.ok = True
        except Exception as e:
            self.ok = False
            self.err = e

    def speak(self, text: str, out_wav: str | None = None, autoplay: bool = True):
        if not self.ok:
            raise RuntimeError(f"Piper not initialized: {self.err}")

        # Generate waveform (16k mono)
        audio_bytes = self.voice.synthesize(text)

        # Piper may return WAV bytes or raw float32 PCM.
        if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes[:4] == b"RIFF":
            # WAV bytes
            if out_wav:
                with open(out_wav, "wb") as f:
                    f.write(audio_bytes)
            import soundfile as sf, sounddevice as sd
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if autoplay:
                sd.play(data, sr); sd.wait()
            return data, sr
        else:
            # Raw float32 PCM
            import soundfile as sf, sounddevice as sd
            sr = 16000
            data = np.frombuffer(audio_bytes, dtype=np.float32)
            if out_wav:
                sf.write(out_wav, data, sr)
            if autoplay:
                sd.play(data, sr); sd.wait()
            return data, sr


# -----------------------------
# Translator (Argos Translate)
# -----------------------------
class Translator:
    """
    Offline translator powered by Argos Translate.
    Expects Argos .argosmodel files in cfg['models_dir'] (default: third_party/models/argos).
    Filenames used: en_es.argosmodel and es_en.argosmodel
    """
    def __init__(self, cfg: dict):
        import argostranslate.package, argostranslate.translate
        self.argos = argostranslate.translate
        self.pkg = argostranslate.package

        models_dir = cfg.get("models_dir", "third_party/models/argos")
        os.makedirs(models_dir, exist_ok=True)

        # Install local models if not yet installed
        installed = {p for p in self.pkg.get_installed_packages()}
        for fname in ["en_es.argosmodel", "es_en.argosmodel"]:
            path = os.path.join(models_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Argos model missing: {path}\n"
                    f"→ Put the .argosmodel files in {models_dir} (no internet needed at runtime)."
                )
            pkg = self.pkg.Package(path)
            # The installed set may compare by identity; this is harmless if it reinstalls.
            if pkg not in installed:
                self.pkg.install_from_path(path)

        self.loaded = True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # src_lang/tgt_lang should be ISO codes like "en" or "es"
        return self.argos.translate(text, src_lang, tgt_lang)


# -----------------------------
# Config utilities
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def must(path: str | None, label: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} missing: {path}")


# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(args):
    console = Console()
    console.print(Panel.fit("EdgeLingo: Mic/File → ASR → MT → TTS", title="Runner", border_style="cyan"))

    # 1) Load config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        console.print(f"[red]Config file not found: {cfg_path}[/red]")
        sys.exit(1)
    cfg = load_config(cfg_path)

    # 1a) Sanity check required assets on disk (asr.py expects these paths)
    try:
        must(cfg.get("encoder_path"), "ASR encoder (.onnx)")
        must(cfg.get("decoder_path"), "ASR decoder (.onnx)")
        # mel_filters.npz is required by asr.py; keep next to working dir or adjust path there.
        mel_path = os.path.join(os.getcwd(), "mel_filters.npz")
        if not os.path.exists(mel_path):
            console.print(f"[yellow]Note:[/yellow] mel_filters.npz not found at {mel_path}. "
                          "Ensure the path in asr.py matches your file location.")
        vpath = cfg.get("tts", {}).get("voice_path")
        must(vpath, "Piper TTS voice (.onnx)")

        # Argos models
        arg_dir = cfg.get("mt", {}).get("models_dir", "third_party/models/argos")
        for fname in ("en_es.argosmodel", "es_en.argosmodel"):
            must(os.path.join(arg_dir, fname), f"Argos model {fname}")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Useful config defaults
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    record_seconds = float(cfg.get("audio", {}).get("record_seconds", 5.0))
    src_lang = (args.src_lang or cfg.get("lang", {}).get("source") or "es").strip()
    tgt_lang = (args.tgt_lang or cfg.get("lang", {}).get("target") or "en").strip()

    # 2) EP info & preference (informational)
    providers = ort_providers()
    pref = args.ep
    chosen = None
    if pref == "qnn" and "QNNExecutionProvider" in providers:
        chosen = "QNNExecutionProvider"
    elif pref == "dml" and "DmlExecutionProvider" in providers:
        chosen = "DmlExecutionProvider"
    elif pref == "cpu" and "CPUExecutionProvider" in providers:
        chosen = "CPUExecutionProvider"
    else:
        # auto
        for p in ("QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if p in providers:
                chosen = p
                break
    console.print(f"[bold]Using EP (preference):[/bold] {chosen or 'N/A'}")
    console.print("[bold]ONNX Runtime providers:[/bold] ", providers or "N/A")

    # 3) Init translator + TTS
    translator = Translator(cfg.get("mt", {}))
    tts_cfg = cfg.get("tts", {})
    voice_path = tts_cfg.get("voice_path", "third_party/models/piper/en_US-amy-medium.onnx")
    voice_cfg = tts_cfg.get("config_path")  # sometimes optional
    tts = PiperTTS(voice_path, voice_cfg)
    if not tts.ok:
        console.print(f"[yellow]Piper TTS not ready:[/yellow] {tts.err}\n"
                      f"→ Update config.yaml with a valid Piper voice and (optional) config.")

    # 4) Acquire/Load audio → ensure we have WAV bytes in memory and (optionally) a saved file
    os.makedirs("tmp", exist_ok=True)
    if args.src == "mic":
        if audio_io is None:
            console.print(f"[red]audio_io import failed:[/red] {AUDIO_IMPORT_ERR}")
            sys.exit(1)
        console.print(f"[green]Recording {record_seconds:.1f}s at {sample_rate} Hz…[/green]")
        wav_bytes = audio_io.record(seconds=int(record_seconds))  # returns WAV bytes
        audio_path = os.path.join("tmp", "mic_input.wav")
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
    else:
        audio_path = args.file
        if not audio_path or not os.path.exists(audio_path):
            console.print(f"[red]Audio file not found:[/red] {audio_path}")
            sys.exit(1)
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()

    # 5) ASR
    if asr is None:
        console.print(f"[red]asr import failed:[/red] {ASR_IMPORT_ERR}")
        sys.exit(1)

    t0 = time.time()
    text = asr.transcribe(wav_bytes, src_lang=src_lang)
    t1 = time.time()
    console.print(Panel.fit(text or "(empty)", title="ASR → text", border_style="green"))
    console.print(f"[dim]ASR time:[/dim] {t1 - t0:.3f}s")

    if not text:
        console.print("[yellow]ASR returned empty text. Aborting translation/TTS.[/yellow]")
        return

    # 6) Translate
    t2 = time.time()
    out_text = translator.translate(text, src_lang, tgt_lang)
    t3 = time.time()
    console.print(Panel.fit(out_text, title=f"MT [{src_lang}->{tgt_lang}]", border_style="magenta"))
    console.print(f"[dim]MT time:[/dim] {t3 - t2:.3f}s")

    # 7) TTS
    if tts.ok:
        console.print("[green]Speaking translated text…[/green]")
        os.makedirs("out", exist_ok=True)
        tts_out = os.path.join("out", f"tts_{int(time.time())}.wav")
        try:
            t4 = time.time()
            tts.speak(out_text, out_wav=tts_out, autoplay=True)
            t5 = time.time()
            console.print(f"[bold]Saved TTS to[/bold] {tts_out}")
            console.print(f"[dim]TTS time:[/dim] {t5 - t4:.3f}s")
        except Exception as e:
            console.print(f"[yellow]TTS failed:[/yellow] {e}")
    else:
        console.print("[yellow]TTS not initialized; skipping audio playback.[/yellow]")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EdgeLingo offline runner")
    p.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--src", choices=["mic", "file"], default="mic", help="Audio source")
    p.add_argument("--file", help="Path to input .wav if --src file")
    p.add_argument("--src-lang", help="Source language code (e.g., es, en)")
    p.add_argument("--tgt-lang", help="Target language code (e.g., en, es)")
    p.add_argument("--ep", choices=["auto", "cpu", "qnn", "dml"], default="auto",
                   help="Preferred ONNX Runtime execution provider (informational)")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
