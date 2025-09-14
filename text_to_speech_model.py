from TTS.api import TTS

def run_tts(text, targetLanguage):

    # English
    if targetLanguage == "E":
        tts_en = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        tts_en.tts_to_file(
            text=text,
            file_path="en_coqui.wav"
        )
        print("Saved the english speech")

    # Spanish
    else:
        tts_es = TTS("tts_models/es/css10/vits")
        tts_es.tts_to_file(
            text=text,
            file_path="es_coqui.wav"
        )
        print("Saved the spanish speech")

# if __name__ == "__main__":
#     run_tts()