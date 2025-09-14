# audio_io.py
import io
import wave
import threading
import time
from typing import Optional, List, Tuple, Iterable, Generator

import numpy as np
import pyaudio

import sounddevice as sd
import numpy as np

def list_devices():
    return sd.query_devices()

# audio_io.py replacement for record(...) using sounddevice
import sounddevice as sd
import soundfile as sf
import io

def record(seconds: int, device_index: Optional[int] = None) -> bytes:
    sd.default.samplerate = RATE
    sd.default.channels = CHANNELS
    data = sd.rec(int(seconds * RATE), dtype='int16', device=device_index)
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, data, RATE, format='WAV', subtype='PCM_16')
    return buf.getvalue()

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes for PCM16
FORMAT = pyaudio.paInt16
CHUNK = 1024  # frames per buffer (~64 ms at 16 kHz)

# -----------------------
# Device enumeration
# -----------------------
def list_input_devices() -> List[Tuple[int, str]]:
    """
    Returns a list of (device_index, device_name) for input devices.
    """
    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                devices.append((i, info.get("name", f"Device {i}")))
    finally:
        p.terminate()
    return devices


# -----------------------
# Fixed-duration capture
# -----------------------
def record(seconds: int, device_index: Optional[int] = None) -> bytes:
    """
    Records mono PCM16 audio at 16 kHz and returns a complete WAV file as bytes.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index)

    frames = []
    num_chunks = int(RATE / CHUNK * seconds)
    for _ in range(num_chunks):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Pack into WAV bytes
    return _frames_to_wav_bytes(frames, RATE, CHANNELS, SAMPLE_WIDTH)


def _frames_to_wav_bytes(frames: List[bytes], rate: int, channels: int, sampwidth: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# -----------------------
# Continuous streaming
# -----------------------
class AudioStreamer:
    """
    Continuously reads from the default (or chosen) input device at 16 kHz mono PCM16,
    keeps a rolling buffer in memory, and yields **WAV bytes** windows with optional overlap.

    Typical use (4s windows, 0.5s overlap):
        with AudioStreamer(chunk_seconds=4.0, overlap_seconds=0.5) as s:
            for wav_bytes in s.stream_windows():
                # send to ASR
    """
    def __init__(
        self,
        device_index: Optional[int] = None,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 0.5,
        vad_energy_threshold: Optional[float] = None,  # e.g., 0.005 for simple energy VAD; None to disable
        max_queue_seconds: float = 30.0,               # safety cap for rolling buffer
    ):
        assert chunk_seconds > 0, "chunk_seconds must be positive"
        assert 0.0 <= overlap_seconds < chunk_seconds, "overlap_seconds must be in [0, chunk_seconds)"
        self.device_index = device_index
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.vad_energy_threshold = vad_energy_threshold
        self.max_queue_frames = int(RATE * max_queue_seconds)

        # Derived sizes
        self.window_frames = int(RATE * self.chunk_seconds)
        self.hop_frames = int(RATE * (self.chunk_seconds - self.overlap_seconds))

        # Internal
        self._pa = None
        self._stream = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._reader_thread = None
        self._running = False

    # Context manager helpers
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):
        if self._running:
            return
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.device_index,
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self):
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _reader_loop(self):
        # Continuously read audio into a rolling buffer
        while self._running:
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                # Small sleep to avoid tight loop if device glitches
                time.sleep(0.01)
                continue
            with self._lock:
                self._buf.extend(data)
                # Safety cap: drop old data if buffer too big
                max_bytes = self.max_queue_frames * SAMPLE_WIDTH
                if len(self._buf) > max_bytes:
                    # keep only the newest max_bytes
                    self._buf = self._buf[-max_bytes:]

    def _get_latest_frames(self, n_frames: int, offset_frames: int = 0) -> Optional[bytes]:
        """
        Returns the last n_frames worth of bytes, optionally offset by offset_frames
        from the very end (i.e., lookback).
        """
        need = (n_frames + offset_frames) * SAMPLE_WIDTH
        with self._lock:
            if len(self._buf) < need:
                return None
            start = len(self._buf) - need
            end = start + n_frames * SAMPLE_WIDTH
            return bytes(self._buf[start:end])

    def stream_windows(self) -> Generator[bytes, None, None]:
        """
        Yields WAV bytes for each rolling window of size chunk_seconds,
        advancing by (chunk_seconds - overlap_seconds) each step.

        This runs until caller breaks (Ctrl+C) or `stop()` is called.
        """
        # We advance with hop size. To keep things simple and robust against timing jitter,
        # we poll until enough frames exist, then slice out the latest window.
        frames_emitted = 0
        # We need at least one full window to begin
        while self._running:
            # Wait until we have window_frames + frames_emitted*hops (not strictly necessary; we always take latest)
            latest = self._get_latest_frames(self.window_frames)
            if latest is None:
                time.sleep(0.01)
                continue

            # Optional simple energy-based VAD trim at window level (kept tiny for speed)
            if self.vad_energy_threshold is not None:
                trimmed = _simple_energy_trim(_pcm16_to_f32(latest), thresh=self.vad_energy_threshold)
                audio_f32 = trimmed if trimmed.size > 0 else _pcm16_to_f32(latest)
            else:
                audio_f32 = _pcm16_to_f32(latest)

            # Always yield proper WAV bytes (16 kHz mono PCM16)
            wav_bytes = _f32_to_wav_bytes(audio_f32, RATE, CHANNELS, SAMPLE_WIDTH)
            yield wav_bytes

            frames_emitted += self.hop_frames
            # Sleep roughly hop duration (not exact; reader thread fills buffer continuously)
            time.sleep(max(self.hop_frames / RATE - 0.01, 0.0))


# -----------------------
# Tiny helpers (VAD + conversions)
# -----------------------
def _pcm16_to_f32(pcm_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0

def _f32_to_wav_bytes(audio_f32: np.ndarray, rate: int, channels: int, sampwidth: int) -> bytes:
    # Re-quantize to PCM16 for a WAV container (ASR frontends typically accept f32 or WAV)
    # We keep WAV to match your transcribe(wav_bytes, ...) API.
    audio_f32 = np.asarray(audio_f32, dtype=np.float32)
    # Clamp then scale
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16).tobytes()
    return _frames_to_wav_bytes([pcm16], rate, channels, sampwidth)

def _simple_energy_trim(audio_f32: np.ndarray, thresh: float = 0.005) -> np.ndarray:
    """Crude RMS gate to trim leading/trailing silence; super fast and dependency-free."""
    if audio_f32.size == 0:
        return audio_f32
    win = 400  # ~25 ms at 16 kHz
    # pad to avoid border effects
    pad = win // 2
    x = np.pad(audio_f32, (pad, pad), mode="constant")
    rms = np.sqrt(np.convolve(x**2, np.ones(win)/win, mode="same"))[pad:-pad]
    mask = rms > thresh
    if not mask.any():
        return audio_f32
    idx = np.where(mask)[0]
    return audio_f32[idx[0]: idx[-1] + 1]

