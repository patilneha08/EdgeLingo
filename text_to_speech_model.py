from TTS.api import TTS

# English
tts_en = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts_en.tts_to_file(text="This is just a random text. Much longer than the last one but a POC", file_path="en_coqui.wav")

# Spanish
tts_es = TTS("tts_models/es/css10/vits")
tts_es.tts_to_file(text="Hola, ¿cómo estás hoy?", file_path="es_coqui.wav")

print("Saved en_coqui.wav and es_coqui.wav")
