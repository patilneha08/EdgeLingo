# text_to_speech_model.py
# Windows-only TTS using SAPI5 via pyttsx3. Saves a WAV and returns the path.

from pathlib import Path
from typing import Literal, Optional
import pyttsx3

# Simple voice preferences (tweak names if you have other voices)
VOICE_PREFS = {
    "E": ["en-US", "English", "George", "Zira", "David", "US"],
    "S": ["es-ES", "es-MX", "Spanish", "Helena", "Laura", "Pablo", "SABINA", "SABINA-ES"]
}

OUTFILES = {
    "E": "en_sapi.wav",
    "S": "es_sapi.wav",
}

def _pick_voice(engine: pyttsx3.Engine, lang_key: Literal["E","S"]) -> Optional[str]:
    prefs = [p.lower() for p in VOICE_PREFS[lang_key]]
    for v in engine.getProperty("voices"):
        name = (v.name or "").lower()
        lang = ""
        # pyttsx3 on Windows exposes languages inconsistently; check multiple fields
        if hasattr(v, "languages") and v.languages:
            lang = "".join([bytes(l).decode(errors="ignore").lower() if isinstance(l, (bytes, bytearray)) else str(l).lower() for l in v.languages])
        if hasattr(v, "id") and v.id:
            lang += " " + str(v.id).lower()

        haystack = f"{name} {lang}"
        if any(pref in haystack for pref in prefs):
            return v.id
    return None

def run_tts(text: str, targetLanguage: Literal["E","S"]) -> Path:
    out_path = Path(OUTFILES[targetLanguage]).resolve()
    engine = pyttsx3.init()  # SAPI on Windows

    vid = _pick_voice(engine, targetLanguage)
    if vid:
        engine.setProperty("voice", vid)
    # You can also tune rate/volume here if you want:
    # engine.setProperty("rate", 180)
    # engine.setProperty("volume", 1.0)

    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return out_path
