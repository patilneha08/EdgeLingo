# runners/streaming_demo.py
import time
from audio_io import AudioStreamer
from asr import transcribe  # your existing function that accepts wav_bytes

def main():
    print("Starting streaming… (Ctrl+C to stop)")
    # 4.0s windows, 0.5s overlap for near-real-time partials
    try:
        with AudioStreamer(chunk_seconds=4.0, overlap_seconds=0.5,
                           vad_energy_threshold=0.005) as s:
            for i, wav_bytes in enumerate(s.stream_windows(), start=1):
                t0 = time.time()
                text = transcribe(wav_bytes, src_lang="en")
                dt = time.time() - t0
                if text.strip():
                    print(f"[{i:03d}] {dt*1000:.0f} ms  →  {text}")
                else:
                    print(f"[{i:03d}] {dt*1000:.0f} ms  →  (silence)")
    except KeyboardInterrupt:
        print("\nStopping…")

if __name__ == "__main__":
    main()
