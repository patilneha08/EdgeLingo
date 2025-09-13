# asr.py
import io, wave, os
import numpy as np
import onnxruntime as ort
import yaml

# local imports from vendored repo
from third_party.aihub_whisper.standalone_model import WhisperONNX
# (if standalone_model expects mel path, pass it; otherwise it reads mel_filters.npz internally)

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Load config + sessions on import (warmup)
_CFG_PATH = os.getenv("ASR_CONFIG", "config.yaml")
with open(_CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

_ENCODER = CFG["encoder_path"]
_DECODER = CFG["decoder_path"]

# Providers: try QNN (NPU) first, then CPU
_PROVIDERS = [
    ("QNNExecutionProvider", {  # you can add provider options here later
        # "backend_path": "",   # usually not needed on Windows w/ onnxruntime-qnn
        # "profiling_level": "off",
    }),
    "CPUExecutionProvider"
]

# Build Whisper wrapper (it internally creates ORT sessions)
_model = WhisperONNX(
    encoder_path=_ENCODER,
    decoder_path=_DECODER,
    providers=_PROVIDERS,
    sample_rate=RATE,
    mel_filters_path="mel_filters.npz"  # ensure this file is present
)

# Warmup with 0.5s silence
_model.transcribe(np.zeros(RATE // 2, dtype=np.float32))

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == RATE and wf.getnchannels() == CHANNELS and wf.getsampwidth() == SAMPLE_WIDTH
        pcm16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0

def transcribe(wav_bytes: bytes, src_lang: str = "en") -> str:
    audio = _wav_bytes_to_float32(wav_bytes)
    # Their standalone runner supports chunking/streaming; we call once per utterance.
    text = _model.transcribe(audio, language=src_lang)  # returns string
    return text.strip()