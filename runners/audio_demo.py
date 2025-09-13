# runners/audio_demo.py
import argparse
from pathlib import Path
import time

from audio_io import record, list_input_devices
from asr import transcribe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-seconds", type=int, default=4)
    parser.add_argument("--src", type=str, default="en", help="source language code (e.g., en, es)")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--save", type=str, default="", help="optional: path to save captured wav")
    args = parser.parse_args()

    if args.device_index is None:
        devs = list_input_devices()
        if not devs:
            print("No input devices found.")
            return
        print("Input devices:")
        for i, name in devs:
            print(f"  [{i}] {name}")

    print(f"\nRecording {args.record_seconds}s… Speak now.")
    t0 = time.time()
    wav_bytes = record(args.record_seconds, device_index=args.device_index)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved WAV -> {args.save}")

    # Latency target after warmup ≤ 800 ms for 3–4s utterance
    print("\nTranscribing…")
    t1 = time.time()
    text = transcribe(wav_bytes, src_lang=args.src)
    t2 = time.time()

    print("\n=== TRANSCRIPT ===")
    print(text if text else "[empty]")
    print("==================")
    print(f"Capture time: {t1 - t0:.3f}s | ASR time: {t2 - t1:.3f}s | Total: {t2 - t0:.3f}s")

if __name__ == "__main__":
    main()