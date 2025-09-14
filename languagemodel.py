# languagemodel.py
from transformers import pipeline
import torch

def run_translation(text: str, targetLanguage: str) -> str:
    use_gpu = torch.cuda.is_available()
    device_map = "auto" if use_gpu else "cpu"

    # If the target language is English
    if targetLanguage == "E":
        es2en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device_map=device_map)
        return es2en(text, max_length=256)[0]["translation_text"]
    
    else:
        en2es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device_map=device_map)
        return en2es(text, max_length=256)[0]["translation_text"]

    # en_text = "How are you? I'm looking for a good restaurant near the museum."
    # print("EN → ES:", en2es(en_text, max_length=256)[0]["translation_text"])

    # es_text = "¿Cómo estás? Estoy buscando un buen restaurante cerca del museo."
    # print("ES → EN:", es2en(es_text, max_length=256)[0]["translation_text"])

    # en_text = "The head of the United Nations says there is no military solution in Syria"
    # print("EN → ES:", en2es(en_text, max_length=256)[0]["translation_text"])

    # es_text = "El jefe de la ONU afirmó que no existe una solución militar en Siria"
    # print("ES → EN:", es2en(es_text, max_length=256)[0]["translation_text"])

# if __name__ == "__main__":
#     lang = input("Translate to (E for English / S for Spanish): ").strip()
#     text = input("Enter your text: ").strip()
#     run_translation(text, lang)

