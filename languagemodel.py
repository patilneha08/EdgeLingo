from transformers import pipeline
import torch

def run_translation(text: str, targetLanguage: str) -> str:
    print("run_translation", targetLanguage)
    use_gpu = torch.cuda.is_available()
    device_map = "auto" if use_gpu else "cpu"

    if targetLanguage == "E":
        print("targetLanguage", targetLanguage)
        es2en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device_map=device_map)
        print("Pipeline loaded")
        return es2en(text, max_length=256)[0]["translation_text"]
    else:
        print("targetLanguage", targetLanguage)
        en2es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device_map=device_map)
        print("Pipeline loaded")
        return en2es(text, max_length=256)[0]["translation_text"]
