#!/usr/bin/env python3
# main.py
import argparse
import os
import sys
import time
import yaml
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

# --- Try imports from your repo ---
try:
    import asr  # Expected to provide: transcribe_file(...) or transcribe_stream(...)
except Exception as e:
    asr = None
    ASR_IMPORT_ERR = e

try:
    import audio_io  # Expected to provide: record(...), (optional) play(...)
except Exception as e:
    audio_io = None
    AUDIO_IMPORT_ERR = e

# --- ONNX Runtime EP selection helper ---
def ort_providers():
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []

# --- Simple local TTS using Piper (offline) ---
class PiperTTS:
    """
    Minimal Piper wrapper.
    You must provide:
      - voice_path: path to a .onnx voice model (e.g., 'en_US-amy-medium.onnx')
      - config_path: path to its .json config (same voice pack)
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
        # Piper returns bytes (wav) OR raw—depending on version; handle both:
        if isinstance(audio_bytes, bytes) and audio_bytes[:4] == b"RIFF":
            # Already a WAV; write to file and optionally play
            if out_wav:
                with open(out_wav, "wb") as f:
                    f.write(audio_bytes)
            import soundfile as sf, io, sounddevice as sd
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if autoplay:
                sd.play(data, sr); sd.wait()
            return data, sr
        else:
            # Raw float pcm; save/play
            import soundfile as sf, sounddevice as sd
            sr = 16000
            data = np.frombuffer(audio_bytes, dtype=np.float32)
            if out_wav:
                sf.write(out_wav, data, sr)
            if autoplay:
                sd.play(data, sr); sd.wait()
            return data, sr

# --- Translator placeholder (plug your ONNX MT here) ---
class Translator:
    """
    Replace this stub with your ONNX-based MT.
    Expected interface: translate(text, src_lang, tgt_lang) -> str
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.loaded = False
        self._load()

    def _load(self):
        # TODO: Load your ONNX MT model(s) here using onnxruntime.InferenceSession
        #       and any tokenizer you use (e.g., SentencePiece).
        self.loaded = True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # TODO: Run encoder-decoder (or decoder-only) inference.
        # For now, echo back with tag to prove the pipe is working.
        return f"[{src_lang}->{tgt_lang}] {text}"

# --- Utility: load config ---
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

# --- Wire up the pipeline ---
def run_pipeline(args):
    console = Console()
    console.print(Panel.fit("EdgeLingo: Mic/File → ASR → MT → TTS", title="Runner", border_style="cyan"))

    # 1) Load config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        console.print(f"[red]Config file not found: {cfg_path}[/red]")
        sys.exit(1)
    cfg = load_config(cfg_path)

    # Useful config defaults
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    record_seconds = float(cfg.get("audio", {}).get("record_seconds", 5.0))
    src_lang = args.src_lang or cfg.get("lang", {}).get("source", "es")
    tgt_lang = args.tgt_lang or cfg.get("lang", {}).get("target", "en")

    # 2) Show available EPs
    providers = ort_providers()
    console.print("[bold]ONNX Runtime providers:[/bold] ", providers or "N/A")

    # 3) Init translator + TTS
    translator = Translator(cfg.get("mt", {}))
    tts_cfg = cfg.get("tts", {})
    voice_path = tts_cfg.get("voice_path", "third_party/models/piper/en_US-amy-medium.onnx")
    voice_cfg = tts_cfg.get("config_path")  # sometimes optional
    tts = PiperTTS(voice_path, voice_cfg)
    if not tts.ok:
        console.print(f"[yellow]Piper TTS not ready:[/yellow] {tts.err}\n"
                      f"→ Update config.yaml with a valid Piper voice and config.")

    # 4) Acquire/Load audio
    if args.src == "mic":
        if audio_io is None:
            console.print(f"[red]audio_io import failed:[/red] {AUDIO_IMPORT_ERR}")
            sys.exit(1)
        console.print(f"[green]Recording {record_seconds}s…[/green]")
        audio, sr = audio_io.record(seconds=record_seconds, samplerate=sample_rate)
        wav_path = os.path.join("tmp", "mic_input.wav")
        os.makedirs("tmp", exist_ok=True)
        import soundfile as sf
        sf.write(wav_path, audio, sr)
        audio_path = wav_path
    else:
        audio_path = args.file
        if not audio_path or not os.path.exists(audio_path):
            console.print(f"[red]Audio file not found:[/red] {audio_path}")
            sys.exit(1)

    # 5) ASR
    if asr is None:
        console.print(f"[red]asr import failed:[/red] {ASR_IMPORT_ERR}")
        sys.exit(1)

    # Prefer file API if your ASR is non-streaming
    if hasattr(asr, "transcribe_file"):
        text = asr.transcribe_file(audio_path, cfg.get("asr", {}))
    elif hasattr(asr, "transcribe_stream"):
        # Load file → numpy → stream
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        text = asr.transcribe_stream(data, sr, cfg.get("asr", {}))
    else:
        console.print("[red]ASR API not found. Expected transcribe_file(...) or transcribe_stream(...)[/red]")
        sys.exit(1)

    console.print(Panel.fit(text or "(empty)", title="ASR → text", border_style="green"))

    if not text:
        console.print("[yellow]ASR returned empty text. Aborting translation/tts.[/yellow]")
        return

    # 6) Translate
    out_text = translator.translate(text, src_lang, tgt_lang)
    console.print(Panel.fit(out_text, title=f"MT [{src_lang}->{tgt_lang}]", border_style="magenta"))

    # 7) TTS
    if tts.ok:
        console.print("[green]Speaking translated text…[/green]")
        os.makedirs("out", exist_ok=True)
        tts_out = os.path.join("out", f"tts_{int(time.time())}.wav")
        try:
            tts.speak(out_text, out_wav=tts_out, autoplay=True)
            console.print(f"[bold]Saved TTS to[/bold] {tts_out}")
        except Exception as e:
            console.print(f"[yellow]TTS failed:[/yellow] {e}")
    else:
        console.print("[yellow]TTS not initialized; skipping audio playback.[/yellow]")

def parse_args():
    p = argparse.ArgumentParser(description="EdgeLingo offline runner")
    p.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--src", choices=["mic", "file"], default="mic", help="Audio source")
    p.add_argument("--file", help="Path to input .wav if --src file")
    p.add_argument("--src-lang", help="Source language code (e.g., es, en)")
    p.add_argument("--tgt-lang", help="Target language code (e.g., en, es)")
    return p.parse_args()

if __name__ == "__main__":
    run_pipeline(parse_args())