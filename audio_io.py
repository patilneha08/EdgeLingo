# audio_io.py
import io
import time
import threading
from typing import Optional, List, Tuple, Generator

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes for PCM16
CHUNK = 1024  # ~64 ms @ 16kHz

# -----------------------
# Device enumeration
# -----------------------
def list_input_devices() -> List[Tuple[int, str]]:
    """Returns (device_index, device_name) for input-capable devices using sounddevice."""
    devices = sd.query_devices()
    results: List[Tuple[int, str]] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            results.append((idx, d.get("name", f"Device {idx}")))
    return results

# -----------------------
# Fixed-duration capture
# -----------------------
def record(seconds: int, device_index: Optional[int] = None) -> bytes:
    """Record mono PCM16 @16k and return a WAV file as bytes."""
    sd.default.samplerate = RATE
    sd.default.channels = CHANNELS
    data = sd.rec(int(seconds * RATE), dtype="int16", device=device_index)
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, data, RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# -----------------------
# Continuous streaming
# -----------------------
class AudioStreamer:
    """
    Continuously captures audio @16kHz mono and yields WAV-bytes windows with optional overlap.
    """
    def __init__(
        self,
        device_index: Optional[int] = None,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 0.5,
        vad_energy_threshold: Optional[float] = None,  # e.g., 0.005
        max_queue_seconds: float = 30.0,
    ):
        assert chunk_seconds > 0
        assert 0.0 <= overlap_seconds < chunk_seconds
        self.device_index = device_index
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.vad_energy_threshold = vad_energy_threshold
        self.max_queue_frames = int(RATE * max_queue_seconds)

        self.window_frames = int(RATE * self.chunk_seconds)
        self.hop_frames = int(RATE * (self.chunk_seconds - self.overlap_seconds))

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._reader_thread = None

    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.stop()

    def start(self):
        if self._running: return
        self._running = True

        def _callback(indata, frames, time_info, status):
            # indata: (frames, channels), int16
            if status:  # ignore benign over/underruns
                pass
            with self._lock:
                self._buf.extend(indata.tobytes())
                max_bytes = self.max_queue_frames * SAMPLE_WIDTH
                if len(self._buf) > max_bytes:
                    self._buf = self._buf[-max_bytes:]

        self._stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            device=self.device_index,
            callback=_callback,
            blocksize=CHUNK,
        )
        self._stream.start()
        self._reader_thread = threading.Thread(target=self._noop, daemon=True)
        self._reader_thread.start()

    def _noop(self):
        while self._running:
            time.sleep(0.05)

    def stop(self):
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None

    def _get_latest_frames(self, n_frames: int) -> Optional[bytes]:
        need = n_frames * SAMPLE_WIDTH
        with self._lock:
            if len(self._buf) < need:
                return None
            start = len(self._buf) - need
            return bytes(self._buf[start:])

    def stream_windows(self) -> Generator[bytes, None, None]:
        while self._running:
            latest = self._get_latest_frames(self.window_frames)
            if latest is None:
                time.sleep(0.01)
                continue
            audio_f32 = _pcm16_to_f32(latest)
            if self.vad_energy_threshold is not None:
                trimmed = _simple_energy_trim(audio_f32, self.vad_energy_threshold)
                if trimmed.size > 0:
                    audio_f32 = trimmed
            yield _f32_to_wav_bytes(audio_f32, RATE, CHANNELS, SAMPLE_WIDTH)
            time.sleep(max(self.hop_frames / RATE - 0.01, 0.0))

def _pcm16_to_f32(pcm_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0

def _f32_to_wav_bytes(audio_f32: np.ndarray, rate: int, channels: int, sampwidth: int) -> bytes:
    audio_f32 = np.asarray(audio_f32, dtype=np.float32)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=rate, channels=channels, subtype="PCM_16", format="WAV") as f:
        f.write(np.frombuffer(pcm16, dtype=np.int16).reshape(-1, 1))
    return buf.getvalue()

def _simple_energy_trim(audio_f32: np.ndarray, thresh: float = 0.005) -> np.ndarray:
    if audio_f32.size == 0:
        return audio_f32
    win = 400  # ~25 ms
    pad = win // 2
    x = np.pad(audio_f32, (pad, pad), mode="constant")
    rms = np.sqrt(np.convolve(x**2, np.ones(win)/win, mode="same"))[pad:-pad]
    mask = rms > thresh
    if not mask.any():
        return audio_f32
    i0, i1 = np.where(mask)[0][[0, -1]]
    return audio_f32[i0:i1+1]


