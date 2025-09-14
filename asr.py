# asr.py
import io, wave, os
import numpy as np
import yaml

# Use your standalone model wrapper
from third_party.aihub_whisper.standalone_model import StandaloneWhisperModel

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Load config
_CFG_PATH = os.getenv("ASR_CONFIG", "config.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

_ENCODER = CFG["encoder_path"]
_DECODER = CFG["decoder_path"]

# Build model
_model = StandaloneWhisperModel(encoder_path=_ENCODER, decoder_path=_DECODER)

# Warmup with 0.5s silence (float32)
_model.transcribe(np.zeros(RATE // 2, dtype=np.float32), sample_rate=RATE)

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == RATE and wf.getnchannels() == CHANNELS and wf.getsampwidth() == SAMPLE_WIDTH, \
            "WAV must be 16kHz/mono/PCM16"
        pcm16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0

def transcribe(wav_bytes: bytes, src_lang: str = "en") -> str:
    # NOTE: src_lang is currently unused by your Standalone model. (Argos handles translation.)
    audio = _wav_bytes_to_float32(wav_bytes)
    text = _model.transcribe(audio, sample_rate=RATE)
    return (text or "").strip()

def transcribe_file(path: str, src_lang: str = "en") -> str:
    with open(path, "rb") as f:
        wav_bytes = f.read()
    return transcribe(wav_bytes, src_lang=src_lang)